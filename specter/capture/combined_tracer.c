/*
 * SPECTER Combined Tracer
 *
 * Single library that intercepts both NCCL and cuBLAS operations.
 * This is just a compilation unit that includes both tracers.
 *
 * We use a shared output file for coordinated timestamps.
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

/* Shared state across both tracers */
static FILE* trace_file = NULL;
static struct timespec start_time;
static pthread_mutex_t trace_mutex = PTHREAD_MUTEX_INITIALIZER;
static int initialized = 0;
static int rank = 0;
static int world_size = 1;
static uint64_t op_id = 0;
static int sync_mode = 0;

/* ======================== COMMON UTILITIES ======================== */

static double get_elapsed_us(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (now.tv_sec - start_time.tv_sec) * 1e6 +
           (now.tv_nsec - start_time.tv_nsec) / 1e3;
}

/* ======================== NCCL TYPES ======================== */

typedef void* ncclComm_t;
typedef void* cudaStream_t;
typedef int cudaError_t;

typedef enum {
    ncclSuccess = 0,
    ncclUnhandledCudaError = 1,
    ncclSystemError = 2,
    ncclInternalError = 3,
    ncclInvalidArgument = 4,
    ncclInvalidUsage = 5,
    ncclRemoteError = 6,
} ncclResult_t;

typedef enum {
    ncclInt8 = 0, ncclChar = 0,
    ncclUint8 = 1,
    ncclInt32 = 2, ncclInt = 2,
    ncclUint32 = 3,
    ncclInt64 = 4,
    ncclUint64 = 5,
    ncclFloat16 = 6, ncclHalf = 6,
    ncclFloat32 = 7, ncclFloat = 7,
    ncclFloat64 = 8, ncclDouble = 8,
    ncclBfloat16 = 9,
} ncclDataType_t;

typedef enum {
    ncclSum = 0,
    ncclProd = 1,
    ncclMax = 2,
    ncclMin = 3,
    ncclAvg = 4,
} ncclRedOp_t;

/* ======================== CUBLAS TYPES ======================== */

typedef void* cublasHandle_t;
typedef int cublasStatus_t;

typedef enum {
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
    CUBLAS_OP_C = 2
} cublasOperation_t;

typedef enum {
    CUDA_R_16F = 2,
    CUDA_R_32F = 0,
    CUDA_R_64F = 1,
    CUDA_R_16BF = 14,
    CUDA_R_8I = 3,
    CUDA_R_32I = 10,
} cudaDataType_t;

typedef enum {
    CUBLAS_COMPUTE_16F = 64,
    CUBLAS_COMPUTE_32F = 68,
    CUBLAS_COMPUTE_32F_FAST_TF32 = 77,
    CUBLAS_COMPUTE_64F = 70,
} cublasComputeType_t;

typedef int cublasGemmAlgo_t;

/* ======================== FUNCTION POINTERS ======================== */

/* NCCL functions */
typedef ncclResult_t (*ncclAllReduce_fn)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t);
typedef ncclResult_t (*ncclBroadcast_fn)(const void*, void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t);
typedef ncclResult_t (*ncclAllGather_fn)(const void*, void*, size_t, ncclDataType_t, ncclComm_t, cudaStream_t);
typedef ncclResult_t (*ncclReduceScatter_fn)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t);
typedef ncclResult_t (*ncclSend_fn)(const void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t);
typedef ncclResult_t (*ncclRecv_fn)(void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t);
typedef ncclResult_t (*ncclGroupStart_fn)(void);
typedef ncclResult_t (*ncclGroupEnd_fn)(void);
typedef ncclResult_t (*ncclCommCount_fn)(ncclComm_t, int*);
typedef ncclResult_t (*ncclCommUserRank_fn)(ncclComm_t, int*);

static ncclAllReduce_fn real_ncclAllReduce = NULL;
static ncclBroadcast_fn real_ncclBroadcast = NULL;
static ncclAllGather_fn real_ncclAllGather = NULL;
static ncclReduceScatter_fn real_ncclReduceScatter = NULL;
static ncclSend_fn real_ncclSend = NULL;
static ncclRecv_fn real_ncclRecv = NULL;
static ncclGroupStart_fn real_ncclGroupStart = NULL;
static ncclGroupEnd_fn real_ncclGroupEnd = NULL;
static ncclCommCount_fn real_ncclCommCount = NULL;
static ncclCommUserRank_fn real_ncclCommUserRank = NULL;

/* cuBLAS functions */
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

static cublasGemmEx_fn real_cublasGemmEx = NULL;
static cublasGemmStridedBatchedEx_fn real_cublasGemmStridedBatchedEx = NULL;

/* CUDA runtime */
typedef cudaError_t (*cudaStreamSynchronize_fn)(cudaStream_t);
typedef cudaError_t (*cudaDeviceSynchronize_fn)(void);

static cudaStreamSynchronize_fn real_cudaStreamSynchronize = NULL;
static cudaDeviceSynchronize_fn real_cudaDeviceSynchronize = NULL;

/* ======================== HELPER FUNCTIONS ======================== */

static size_t nccl_dtype_size(ncclDataType_t dtype) {
    switch(dtype) {
        case ncclInt8: case ncclUint8: return 1;
        case ncclFloat16: case ncclBfloat16: return 2;
        case ncclInt32: case ncclUint32: case ncclFloat32: return 4;
        case ncclInt64: case ncclUint64: case ncclFloat64: return 8;
        default: return 4;
    }
}

static const char* nccl_dtype_name(ncclDataType_t dtype) {
    switch(dtype) {
        case ncclInt8: return "int8";
        case ncclUint8: return "uint8";
        case ncclInt32: return "int32";
        case ncclUint32: return "uint32";
        case ncclInt64: return "int64";
        case ncclUint64: return "uint64";
        case ncclFloat16: return "fp16";
        case ncclFloat32: return "fp32";
        case ncclFloat64: return "fp64";
        case ncclBfloat16: return "bf16";
        default: return "unknown";
    }
}

static const char* nccl_redop_name(ncclRedOp_t op) {
    switch(op) {
        case ncclSum: return "sum";
        case ncclProd: return "prod";
        case ncclMax: return "max";
        case ncclMin: return "min";
        case ncclAvg: return "avg";
        default: return "unknown";
    }
}

static const char* cublas_dtype_name(cudaDataType_t dtype) {
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

static const char* cublas_compute_name(cublasComputeType_t compute) {
    switch(compute) {
        case CUBLAS_COMPUTE_16F: return "fp16";
        case CUBLAS_COMPUTE_32F: return "fp32";
        case CUBLAS_COMPUTE_32F_FAST_TF32: return "tf32";
        case CUBLAS_COMPUTE_64F: return "fp64";
        default: return "fp32";
    }
}

static const char* cublas_op_str(cublasOperation_t op) {
    return op == CUBLAS_OP_N ? "N" : (op == CUBLAS_OP_T ? "T" : "C");
}

/* ======================== INITIALIZATION ======================== */

static void init_tracer(void) {
    if (initialized) return;

    pthread_mutex_lock(&trace_mutex);
    if (initialized) {
        pthread_mutex_unlock(&trace_mutex);
        return;
    }

    clock_gettime(CLOCK_MONOTONIC, &start_time);

    /* Get rank/world from environment */
    char* rank_str = getenv("RANK");
    if (!rank_str) rank_str = getenv("LOCAL_RANK");
    rank = rank_str ? atoi(rank_str) : 0;

    char* world_str = getenv("WORLD_SIZE");
    world_size = world_str ? atoi(world_str) : 1;

    char* sync_str = getenv("SPECTER_SYNC");
    sync_mode = sync_str ? atoi(sync_str) : 0;

    /* Open trace file */
    char* output_dir = getenv("SPECTER_OUTPUT");
    if (!output_dir) output_dir = ".";

    char filename[512];
    snprintf(filename, sizeof(filename), "%s/specter_trace_rank%d.jsonl", output_dir, rank);
    trace_file = fopen(filename, "w");

    if (!trace_file) {
        fprintf(stderr, "[SPECTER] Failed to open trace file: %s\n", filename);
        pthread_mutex_unlock(&trace_mutex);
        return;
    }

    /* Write header */
    fprintf(trace_file,
        "{\"event\":\"init\",\"rank\":%d,\"world_size\":%d,\"sync_mode\":%d,\"pid\":%d}\n",
        rank, world_size, sync_mode, getpid());
    fflush(trace_file);

    /* Load NCCL */
    void* nccl_handle = dlopen("libnccl.so.2", RTLD_LAZY | RTLD_GLOBAL);
    if (!nccl_handle) nccl_handle = dlopen("libnccl.so", RTLD_LAZY | RTLD_GLOBAL);
    if (nccl_handle) {
        real_ncclAllReduce = (ncclAllReduce_fn)dlsym(nccl_handle, "ncclAllReduce");
        real_ncclBroadcast = (ncclBroadcast_fn)dlsym(nccl_handle, "ncclBroadcast");
        real_ncclAllGather = (ncclAllGather_fn)dlsym(nccl_handle, "ncclAllGather");
        real_ncclReduceScatter = (ncclReduceScatter_fn)dlsym(nccl_handle, "ncclReduceScatter");
        real_ncclSend = (ncclSend_fn)dlsym(nccl_handle, "ncclSend");
        real_ncclRecv = (ncclRecv_fn)dlsym(nccl_handle, "ncclRecv");
        real_ncclGroupStart = (ncclGroupStart_fn)dlsym(nccl_handle, "ncclGroupStart");
        real_ncclGroupEnd = (ncclGroupEnd_fn)dlsym(nccl_handle, "ncclGroupEnd");
        real_ncclCommCount = (ncclCommCount_fn)dlsym(nccl_handle, "ncclCommCount");
        real_ncclCommUserRank = (ncclCommUserRank_fn)dlsym(nccl_handle, "ncclCommUserRank");
    }

    /* Load cuBLAS */
    void* cublas_handle = dlopen("libcublas.so.12", RTLD_LAZY | RTLD_GLOBAL);
    if (!cublas_handle) cublas_handle = dlopen("libcublas.so.11", RTLD_LAZY | RTLD_GLOBAL);
    if (!cublas_handle) cublas_handle = dlopen("libcublas.so", RTLD_LAZY | RTLD_GLOBAL);
    if (cublas_handle) {
        real_cublasGemmEx = (cublasGemmEx_fn)dlsym(cublas_handle, "cublasGemmEx");
        real_cublasGemmStridedBatchedEx = (cublasGemmStridedBatchedEx_fn)dlsym(cublas_handle, "cublasGemmStridedBatchedEx");
    }

    /* Load CUDA runtime */
    void* cuda_handle = dlopen("libcudart.so", RTLD_LAZY | RTLD_GLOBAL);
    if (cuda_handle) {
        real_cudaStreamSynchronize = (cudaStreamSynchronize_fn)dlsym(cuda_handle, "cudaStreamSynchronize");
        real_cudaDeviceSynchronize = (cudaDeviceSynchronize_fn)dlsym(cuda_handle, "cudaDeviceSynchronize");
    }

    if (rank == 0) {
        fprintf(stderr, "[SPECTER] Combined tracer initialized (rank %d/%d, sync=%d)\n",
                rank, world_size, sync_mode);
        fprintf(stderr, "[SPECTER] Trace output: %s\n", filename);
    }

    initialized = 1;
    pthread_mutex_unlock(&trace_mutex);
}

/* ======================== NCCL INTERCEPTS ======================== */

static int in_group = 0;

#define LOG_NCCL_OP(name, count, dtype, peer, redop, comm, stream) do { \
    size_t bytes = (count) * nccl_dtype_size(dtype); \
    int comm_rank = rank, comm_size = world_size; \
    if (real_ncclCommUserRank && comm) real_ncclCommUserRank(comm, &comm_rank); \
    if (real_ncclCommCount && comm) real_ncclCommCount(comm, &comm_size); \
    pthread_mutex_lock(&trace_mutex); \
    fprintf(trace_file, \
        "{\"id\":%lu,\"type\":\"nccl\",\"op\":\"%s\",\"start_us\":%.1f,\"duration_us\":%.1f," \
        "\"count\":%zu,\"bytes\":%zu,\"dtype\":\"%s\"", \
        op_id++, name, start_us, duration_us, (size_t)(count), bytes, nccl_dtype_name(dtype)); \
    if ((peer) >= 0) fprintf(trace_file, ",\"peer\":%d", peer); \
    if ((int)(redop) >= 0) fprintf(trace_file, ",\"redop\":\"%s\"", nccl_redop_name(redop)); \
    fprintf(trace_file, ",\"comm_rank\":%d,\"comm_size\":%d,\"in_group\":%d}\n", \
            comm_rank, comm_size, in_group); \
    fflush(trace_file); \
    pthread_mutex_unlock(&trace_mutex); \
} while(0)

ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, ncclRedOp_t op,
                           ncclComm_t comm, cudaStream_t stream) {
    init_tracer();
    if (!real_ncclAllReduce) return ncclInternalError;

    double start_us = get_elapsed_us();
    ncclResult_t result = real_ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
    if (sync_mode && real_cudaStreamSynchronize) real_cudaStreamSynchronize(stream);
    double duration_us = get_elapsed_us() - start_us;

    LOG_NCCL_OP("AllReduce", count, datatype, -1, op, comm, stream);
    return result;
}

ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, int root,
                           ncclComm_t comm, cudaStream_t stream) {
    init_tracer();
    if (!real_ncclBroadcast) return ncclInternalError;

    double start_us = get_elapsed_us();
    ncclResult_t result = real_ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream);
    if (sync_mode && real_cudaStreamSynchronize) real_cudaStreamSynchronize(stream);
    double duration_us = get_elapsed_us() - start_us;

    LOG_NCCL_OP("Broadcast", count, datatype, root, -1, comm, stream);
    return result;
}

ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
                           ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
    init_tracer();
    if (!real_ncclAllGather) return ncclInternalError;

    double start_us = get_elapsed_us();
    ncclResult_t result = real_ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream);
    if (sync_mode && real_cudaStreamSynchronize) real_cudaStreamSynchronize(stream);
    double duration_us = get_elapsed_us() - start_us;

    LOG_NCCL_OP("AllGather", sendcount, datatype, -1, -1, comm, stream);
    return result;
}

ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
                               ncclDataType_t datatype, ncclRedOp_t op,
                               ncclComm_t comm, cudaStream_t stream) {
    init_tracer();
    if (!real_ncclReduceScatter) return ncclInternalError;

    double start_us = get_elapsed_us();
    ncclResult_t result = real_ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
    if (sync_mode && real_cudaStreamSynchronize) real_cudaStreamSynchronize(stream);
    double duration_us = get_elapsed_us() - start_us;

    LOG_NCCL_OP("ReduceScatter", recvcount, datatype, -1, op, comm, stream);
    return result;
}

ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, cudaStream_t stream) {
    init_tracer();
    if (!real_ncclSend) return ncclInternalError;

    double start_us = get_elapsed_us();
    ncclResult_t result = real_ncclSend(sendbuff, count, datatype, peer, comm, stream);
    if (sync_mode && real_cudaStreamSynchronize) real_cudaStreamSynchronize(stream);
    double duration_us = get_elapsed_us() - start_us;

    LOG_NCCL_OP("Send", count, datatype, peer, -1, comm, stream);
    return result;
}

ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, cudaStream_t stream) {
    init_tracer();
    if (!real_ncclRecv) return ncclInternalError;

    double start_us = get_elapsed_us();
    ncclResult_t result = real_ncclRecv(recvbuff, count, datatype, peer, comm, stream);
    if (sync_mode && real_cudaStreamSynchronize) real_cudaStreamSynchronize(stream);
    double duration_us = get_elapsed_us() - start_us;

    LOG_NCCL_OP("Recv", count, datatype, peer, -1, comm, stream);
    return result;
}

ncclResult_t ncclGroupStart(void) {
    init_tracer();
    if (!real_ncclGroupStart) return ncclSuccess;

    in_group = 1;
    pthread_mutex_lock(&trace_mutex);
    if (trace_file) {
        fprintf(trace_file, "{\"event\":\"group_start\",\"time_us\":%.1f}\n", get_elapsed_us());
        fflush(trace_file);
    }
    pthread_mutex_unlock(&trace_mutex);

    return real_ncclGroupStart();
}

ncclResult_t ncclGroupEnd(void) {
    init_tracer();
    if (!real_ncclGroupEnd) return ncclSuccess;

    double start_us = get_elapsed_us();
    ncclResult_t result = real_ncclGroupEnd();
    double end_us = get_elapsed_us();

    in_group = 0;
    pthread_mutex_lock(&trace_mutex);
    if (trace_file) {
        fprintf(trace_file, "{\"event\":\"group_end\",\"time_us\":%.1f,\"duration_us\":%.1f}\n",
                start_us, end_us - start_us);
        fflush(trace_file);
    }
    pthread_mutex_unlock(&trace_mutex);

    return result;
}

/* ======================== CUBLAS INTERCEPTS ======================== */

#define LOG_CUBLAS_OP(name, m, n, k, batch, dtype_a, dtype_b, dtype_c, compute, trans_a, trans_b) do { \
    double flops = 2.0 * (double)(m) * (double)(n) * (double)(k) * (double)((batch) > 0 ? (batch) : 1); \
    double tflops = (duration_us > 0) ? (flops / (duration_us * 1e6)) : 0; \
    pthread_mutex_lock(&trace_mutex); \
    fprintf(trace_file, \
        "{\"id\":%lu,\"type\":\"cublas\",\"op\":\"%s\",\"start_us\":%.1f,\"duration_us\":%.1f," \
        "\"m\":%d,\"n\":%d,\"k\":%d,\"batch\":%d," \
        "\"dtype_a\":\"%s\",\"dtype_b\":\"%s\",\"dtype_c\":\"%s\"," \
        "\"compute\":\"%s\",\"trans_a\":\"%s\",\"trans_b\":\"%s\"," \
        "\"flops\":%.0f,\"tflops\":%.2f}\n", \
        op_id++, name, start_us, duration_us, \
        (m), (n), (k), (batch) > 0 ? (batch) : 1, \
        dtype_a, dtype_b, dtype_c, compute, trans_a, trans_b, flops, tflops); \
    fflush(trace_file); \
    pthread_mutex_unlock(&trace_mutex); \
} while(0)

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
    if (!real_cublasGemmEx) return 1;

    double start_us = get_elapsed_us();
    cublasStatus_t result = real_cublasGemmEx(
        handle, transa, transb, m, n, k, alpha,
        A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc,
        computeType, algo
    );
    if (sync_mode && real_cudaDeviceSynchronize) real_cudaDeviceSynchronize();
    double duration_us = get_elapsed_us() - start_us;

    LOG_CUBLAS_OP("GemmEx", m, n, k, 0,
                  cublas_dtype_name(Atype), cublas_dtype_name(Btype), cublas_dtype_name(Ctype),
                  cublas_compute_name(computeType),
                  cublas_op_str(transa), cublas_op_str(transb));

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
    if (!real_cublasGemmStridedBatchedEx) return 1;

    double start_us = get_elapsed_us();
    cublasStatus_t result = real_cublasGemmStridedBatchedEx(
        handle, transa, transb, m, n, k, alpha,
        A, Atype, lda, strideA, B, Btype, ldb, strideB,
        beta, C, Ctype, ldc, strideC, batchCount,
        computeType, algo
    );
    if (sync_mode && real_cudaDeviceSynchronize) real_cudaDeviceSynchronize();
    double duration_us = get_elapsed_us() - start_us;

    LOG_CUBLAS_OP("GemmStridedBatchedEx", m, n, k, batchCount,
                  cublas_dtype_name(Atype), cublas_dtype_name(Btype), cublas_dtype_name(Ctype),
                  cublas_compute_name(computeType),
                  cublas_op_str(transa), cublas_op_str(transb));

    return result;
}

/* ======================== CLEANUP ======================== */

__attribute__((destructor))
void cleanup_tracer(void) {
    pthread_mutex_lock(&trace_mutex);
    if (trace_file) {
        fprintf(trace_file, "{\"event\":\"shutdown\",\"time_us\":%.1f,\"total_ops\":%lu}\n",
                get_elapsed_us(), op_id);
        fclose(trace_file);
        trace_file = NULL;

        if (rank == 0) {
            fprintf(stderr, "[SPECTER] Trace complete. %lu operations recorded.\n", op_id);
        }
    }
    pthread_mutex_unlock(&trace_mutex);
}
