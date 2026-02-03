/*
 * SPECTER cuBLAS Tracer
 *
 * Intercepts cuBLAS GEMM operations via LD_PRELOAD and logs them
 * to a JSON Lines file for later analysis.
 *
 * Build: make
 * Usage: LD_PRELOAD=./libspecter_cublas.so torchrun ... train.py
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <stdint.h>
#include <unistd.h>

/* cuBLAS types */
typedef void* cublasHandle_t;
typedef void* cudaStream_t;
typedef int cudaError_t;
typedef int cublasStatus_t;

typedef enum {
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
    CUBLAS_OP_C = 2
} cublasOperation_t;

typedef enum {
    CUDA_R_16F = 2,   /* half */
    CUDA_R_32F = 0,   /* float */
    CUDA_R_64F = 1,   /* double */
    CUDA_R_16BF = 14, /* bfloat16 */
    CUDA_R_8I = 3,    /* int8 */
    CUDA_R_32I = 10,  /* int32 */
} cudaDataType_t;

typedef enum {
    CUBLAS_COMPUTE_16F = 64,
    CUBLAS_COMPUTE_16F_PEDANTIC = 65,
    CUBLAS_COMPUTE_32F = 68,
    CUBLAS_COMPUTE_32F_PEDANTIC = 69,
    CUBLAS_COMPUTE_32F_FAST_16F = 74,
    CUBLAS_COMPUTE_32F_FAST_16BF = 75,
    CUBLAS_COMPUTE_32F_FAST_TF32 = 77,
    CUBLAS_COMPUTE_64F = 70,
    CUBLAS_COMPUTE_64F_PEDANTIC = 71,
    CUBLAS_COMPUTE_32I = 72,
    CUBLAS_COMPUTE_32I_PEDANTIC = 73,
} cublasComputeType_t;

typedef int cublasGemmAlgo_t;

/* Function pointer types */
typedef cublasStatus_t (*cublasGemmEx_fn)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int, const void*,
    const void*, cudaDataType_t, int,
    const void*, cudaDataType_t, int,
    const void*, void*, cudaDataType_t, int,
    cublasComputeType_t, cublasGemmAlgo_t);

typedef cublasStatus_t (*cublasGemmStridedBatchedEx_fn)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int, const void*,
    const void*, cudaDataType_t, int, long long int,
    const void*, cudaDataType_t, int, long long int,
    const void*, void*, cudaDataType_t, int, long long int,
    int, cublasComputeType_t, cublasGemmAlgo_t);

typedef cublasStatus_t (*cublasSgemm_fn)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int, const float*,
    const float*, int, const float*, int,
    const float*, float*, int);

typedef cublasStatus_t (*cublasHgemm_fn)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int, const void*,
    const void*, int, const void*, int,
    const void*, void*, int);

typedef cudaError_t (*cudaDeviceSynchronize_fn)(void);

/* Real function pointers */
static cublasGemmEx_fn real_cublasGemmEx = NULL;
static cublasGemmStridedBatchedEx_fn real_cublasGemmStridedBatchedEx = NULL;
static cublasSgemm_fn real_cublasSgemm = NULL;
static cublasHgemm_fn real_cublasHgemm = NULL;
static cudaDeviceSynchronize_fn real_cudaDeviceSynchronize = NULL;

/* Trace state */
static FILE* trace_file = NULL;
static struct timespec start_time;
static pthread_mutex_t trace_mutex = PTHREAD_MUTEX_INITIALIZER;
static int initialized = 0;
static int rank = 0;
static uint64_t op_id = 0;
static int sync_mode = 0;

/* Get datatype size */
static size_t dtype_size(cudaDataType_t dtype) {
    switch(dtype) {
        case CUDA_R_16F:
        case CUDA_R_16BF:
            return 2;
        case CUDA_R_32F:
        case CUDA_R_32I:
            return 4;
        case CUDA_R_64F:
            return 8;
        case CUDA_R_8I:
            return 1;
        default:
            return 4;
    }
}

/* Get datatype name */
static const char* dtype_name(cudaDataType_t dtype) {
    switch(dtype) {
        case CUDA_R_16F: return "fp16";
        case CUDA_R_16BF: return "bf16";
        case CUDA_R_32F: return "fp32";
        case CUDA_R_64F: return "fp64";
        case CUDA_R_8I: return "int8";
        case CUDA_R_32I: return "int32";
        default: return "unknown";
    }
}

/* Get compute type name */
static const char* compute_name(cublasComputeType_t compute) {
    switch(compute) {
        case CUBLAS_COMPUTE_16F:
        case CUBLAS_COMPUTE_16F_PEDANTIC:
            return "fp16";
        case CUBLAS_COMPUTE_32F:
        case CUBLAS_COMPUTE_32F_PEDANTIC:
            return "fp32";
        case CUBLAS_COMPUTE_32F_FAST_16F:
            return "fp32_fast_fp16";
        case CUBLAS_COMPUTE_32F_FAST_16BF:
            return "fp32_fast_bf16";
        case CUBLAS_COMPUTE_32F_FAST_TF32:
            return "tf32";
        case CUBLAS_COMPUTE_64F:
        case CUBLAS_COMPUTE_64F_PEDANTIC:
            return "fp64";
        case CUBLAS_COMPUTE_32I:
        case CUBLAS_COMPUTE_32I_PEDANTIC:
            return "int32";
        default:
            return "unknown";
    }
}

static const char* op_to_str(cublasOperation_t op) {
    switch(op) {
        case CUBLAS_OP_N: return "N";
        case CUBLAS_OP_T: return "T";
        case CUBLAS_OP_C: return "C";
        default: return "?";
    }
}

static double get_elapsed_us(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (now.tv_sec - start_time.tv_sec) * 1e6 +
           (now.tv_nsec - start_time.tv_nsec) / 1e3;
}

static void init_tracer(void) {
    if (initialized) return;

    pthread_mutex_lock(&trace_mutex);
    if (initialized) {
        pthread_mutex_unlock(&trace_mutex);
        return;
    }

    clock_gettime(CLOCK_MONOTONIC, &start_time);

    /* Get rank */
    char* rank_str = getenv("RANK");
    if (!rank_str) rank_str = getenv("LOCAL_RANK");
    rank = rank_str ? atoi(rank_str) : 0;

    /* Check sync mode */
    char* sync_str = getenv("SPECTER_SYNC");
    sync_mode = sync_str ? atoi(sync_str) : 0;

    /* Open trace file */
    char* output_dir = getenv("SPECTER_OUTPUT");
    if (!output_dir) output_dir = ".";

    char filename[512];
    snprintf(filename, sizeof(filename), "%s/specter_cublas_rank%d.jsonl", output_dir, rank);
    trace_file = fopen(filename, "w");

    if (!trace_file) {
        fprintf(stderr, "[SPECTER] Failed to open cuBLAS trace file: %s\n", filename);
        pthread_mutex_unlock(&trace_mutex);
        return;
    }

    fprintf(trace_file, "{\"event\":\"init\",\"rank\":%d,\"sync_mode\":%d,\"pid\":%d}\n",
            rank, sync_mode, getpid());
    fflush(trace_file);

    /* Load real cuBLAS functions */
    void* cublas_handle = dlopen("libcublas.so.12", RTLD_LAZY | RTLD_GLOBAL);
    if (!cublas_handle) {
        cublas_handle = dlopen("libcublas.so.11", RTLD_LAZY | RTLD_GLOBAL);
    }
    if (!cublas_handle) {
        cublas_handle = dlopen("libcublas.so", RTLD_LAZY | RTLD_GLOBAL);
    }
    if (cublas_handle) {
        real_cublasGemmEx = (cublasGemmEx_fn)dlsym(cublas_handle, "cublasGemmEx");
        real_cublasGemmStridedBatchedEx = (cublasGemmStridedBatchedEx_fn)dlsym(cublas_handle, "cublasGemmStridedBatchedEx");
        real_cublasSgemm = (cublasSgemm_fn)dlsym(cublas_handle, "cublasSgemm_v2");
        real_cublasHgemm = (cublasHgemm_fn)dlsym(cublas_handle, "cublasHgemm");
    } else {
        fprintf(stderr, "[SPECTER] Failed to load libcublas.so: %s\n", dlerror());
    }

    /* Load CUDA runtime */
    void* cuda_handle = dlopen("libcudart.so", RTLD_LAZY | RTLD_GLOBAL);
    if (cuda_handle) {
        real_cudaDeviceSynchronize = (cudaDeviceSynchronize_fn)dlsym(cuda_handle, "cudaDeviceSynchronize");
    }

    if (rank == 0) {
        fprintf(stderr, "[SPECTER] cuBLAS tracer initialized (rank %d, sync=%d)\n", rank, sync_mode);
        fprintf(stderr, "[SPECTER] cuBLAS trace output: %s\n", filename);
    }

    initialized = 1;
    pthread_mutex_unlock(&trace_mutex);
}

static void log_gemm(const char* op, int m, int n, int k,
                     const char* dtype_a, const char* dtype_b, const char* dtype_c,
                     const char* compute, const char* trans_a, const char* trans_b,
                     int batch_count, double start_us, double duration_us) {
    if (!trace_file) return;

    /* Calculate FLOPs: 2 * M * N * K * batch_count (multiply-add) */
    double flops = 2.0 * (double)m * (double)n * (double)k * (double)(batch_count > 0 ? batch_count : 1);
    double tflops = (duration_us > 0) ? (flops / (duration_us * 1e6)) : 0;

    pthread_mutex_lock(&trace_mutex);

    fprintf(trace_file,
        "{\"id\":%lu,\"op\":\"%s\",\"start_us\":%.1f,\"duration_us\":%.1f,"
        "\"m\":%d,\"n\":%d,\"k\":%d,\"batch\":%d,"
        "\"dtype_a\":\"%s\",\"dtype_b\":\"%s\",\"dtype_c\":\"%s\","
        "\"compute\":\"%s\",\"trans_a\":\"%s\",\"trans_b\":\"%s\","
        "\"flops\":%.0f,\"tflops\":%.2f}\n",
        op_id++, op, start_us, duration_us,
        m, n, k, batch_count > 0 ? batch_count : 1,
        dtype_a, dtype_b, dtype_c, compute,
        trans_a, trans_b, flops, tflops);

    fflush(trace_file);
    pthread_mutex_unlock(&trace_mutex);
}

/* ==================== cuBLAS Function Intercepts ==================== */

cublasStatus_t cublasGemmEx(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void* alpha,
    const void* A, cudaDataType_t Atype, int lda,
    const void* B, cudaDataType_t Btype, int ldb,
    const void* beta,
    void* C, cudaDataType_t Ctype, int ldc,
    cublasComputeType_t computeType,
    cublasGemmAlgo_t algo
) {
    init_tracer();

    if (!real_cublasGemmEx) {
        fprintf(stderr, "[SPECTER] cublasGemmEx not loaded!\n");
        return 1; /* CUBLAS_STATUS_NOT_INITIALIZED */
    }

    double start_us = get_elapsed_us();

    cublasStatus_t result = real_cublasGemmEx(
        handle, transa, transb, m, n, k, alpha,
        A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc,
        computeType, algo
    );

    double end_us = get_elapsed_us();

    if (sync_mode && real_cudaDeviceSynchronize) {
        real_cudaDeviceSynchronize();
        end_us = get_elapsed_us();
    }

    log_gemm("GemmEx", m, n, k,
             dtype_name(Atype), dtype_name(Btype), dtype_name(Ctype),
             compute_name(computeType),
             op_to_str(transa), op_to_str(transb),
             0, start_us, end_us - start_us);

    return result;
}

cublasStatus_t cublasGemmStridedBatchedEx(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void* alpha,
    const void* A, cudaDataType_t Atype, int lda, long long int strideA,
    const void* B, cudaDataType_t Btype, int ldb, long long int strideB,
    const void* beta,
    void* C, cudaDataType_t Ctype, int ldc, long long int strideC,
    int batchCount,
    cublasComputeType_t computeType,
    cublasGemmAlgo_t algo
) {
    init_tracer();

    if (!real_cublasGemmStridedBatchedEx) {
        fprintf(stderr, "[SPECTER] cublasGemmStridedBatchedEx not loaded!\n");
        return 1;
    }

    double start_us = get_elapsed_us();

    cublasStatus_t result = real_cublasGemmStridedBatchedEx(
        handle, transa, transb, m, n, k, alpha,
        A, Atype, lda, strideA, B, Btype, ldb, strideB,
        beta, C, Ctype, ldc, strideC, batchCount,
        computeType, algo
    );

    double end_us = get_elapsed_us();

    if (sync_mode && real_cudaDeviceSynchronize) {
        real_cudaDeviceSynchronize();
        end_us = get_elapsed_us();
    }

    log_gemm("GemmStridedBatchedEx", m, n, k,
             dtype_name(Atype), dtype_name(Btype), dtype_name(Ctype),
             compute_name(computeType),
             op_to_str(transa), op_to_str(transb),
             batchCount, start_us, end_us - start_us);

    return result;
}

cublasStatus_t cublasSgemm_v2(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* A, int lda,
    const float* B, int ldb,
    const float* beta,
    float* C, int ldc
) {
    init_tracer();

    if (!real_cublasSgemm) {
        fprintf(stderr, "[SPECTER] cublasSgemm not loaded!\n");
        return 1;
    }

    double start_us = get_elapsed_us();

    cublasStatus_t result = real_cublasSgemm(
        handle, transa, transb, m, n, k, alpha,
        A, lda, B, ldb, beta, C, ldc
    );

    double end_us = get_elapsed_us();

    if (sync_mode && real_cudaDeviceSynchronize) {
        real_cudaDeviceSynchronize();
        end_us = get_elapsed_us();
    }

    log_gemm("Sgemm", m, n, k,
             "fp32", "fp32", "fp32", "fp32",
             op_to_str(transa), op_to_str(transb),
             0, start_us, end_us - start_us);

    return result;
}

cublasStatus_t cublasHgemm(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void* alpha,
    const void* A, int lda,
    const void* B, int ldb,
    const void* beta,
    void* C, int ldc
) {
    init_tracer();

    if (!real_cublasHgemm) {
        fprintf(stderr, "[SPECTER] cublasHgemm not loaded!\n");
        return 1;
    }

    double start_us = get_elapsed_us();

    cublasStatus_t result = real_cublasHgemm(
        handle, transa, transb, m, n, k, alpha,
        A, lda, B, ldb, beta, C, ldc
    );

    double end_us = get_elapsed_us();

    if (sync_mode && real_cudaDeviceSynchronize) {
        real_cudaDeviceSynchronize();
        end_us = get_elapsed_us();
    }

    log_gemm("Hgemm", m, n, k,
             "fp16", "fp16", "fp16", "fp16",
             op_to_str(transa), op_to_str(transb),
             0, start_us, end_us - start_us);

    return result;
}

__attribute__((destructor))
void cleanup_tracer(void) {
    pthread_mutex_lock(&trace_mutex);
    if (trace_file) {
        fprintf(trace_file, "{\"event\":\"shutdown\",\"time_us\":%.1f,\"total_ops\":%lu}\n",
                get_elapsed_us(), op_id);
        fclose(trace_file);
        trace_file = NULL;

        if (rank == 0) {
            fprintf(stderr, "[SPECTER] cuBLAS trace complete. %lu operations recorded.\n", op_id);
        }
    }
    pthread_mutex_unlock(&trace_mutex);
}
