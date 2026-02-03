/*
 * SPECTER NCCL Tracer
 *
 * Intercepts NCCL collective operations via LD_PRELOAD and logs them
 * to a JSON Lines file for later analysis and replay.
 *
 * Build: make
 * Usage: LD_PRELOAD=./libspecter_nccl.so torchrun ... train.py
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
#include <sys/syscall.h>

/* NCCL types - we define these ourselves to avoid header dependency */
typedef void* ncclComm_t;
typedef void* cudaStream_t;

typedef enum {
    ncclSuccess = 0,
    ncclUnhandledCudaError = 1,
    ncclSystemError = 2,
    ncclInternalError = 3,
    ncclInvalidArgument = 4,
    ncclInvalidUsage = 5,
    ncclRemoteError = 6,
    ncclNumResults = 7
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
    ncclNumTypes = 10
} ncclDataType_t;

typedef enum {
    ncclSum = 0,
    ncclProd = 1,
    ncclMax = 2,
    ncclMin = 3,
    ncclAvg = 4,
    ncclNumOps = 5
} ncclRedOp_t;

/* CUDA runtime types */
typedef int cudaError_t;

/* Function pointer types */
typedef ncclResult_t (*ncclAllReduce_fn)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t);
typedef ncclResult_t (*ncclBroadcast_fn)(const void*, void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t);
typedef ncclResult_t (*ncclReduce_fn)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, int, ncclComm_t, cudaStream_t);
typedef ncclResult_t (*ncclAllGather_fn)(const void*, void*, size_t, ncclDataType_t, ncclComm_t, cudaStream_t);
typedef ncclResult_t (*ncclReduceScatter_fn)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t);
typedef ncclResult_t (*ncclSend_fn)(const void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t);
typedef ncclResult_t (*ncclRecv_fn)(void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t);
typedef ncclResult_t (*ncclGroupStart_fn)(void);
typedef ncclResult_t (*ncclGroupEnd_fn)(void);
typedef ncclResult_t (*ncclCommCount_fn)(ncclComm_t, int*);
typedef ncclResult_t (*ncclCommUserRank_fn)(ncclComm_t, int*);

typedef cudaError_t (*cudaStreamSynchronize_fn)(cudaStream_t);
typedef cudaError_t (*cudaEventCreate_fn)(void**);
typedef cudaError_t (*cudaEventRecord_fn)(void*, cudaStream_t);
typedef cudaError_t (*cudaEventSynchronize_fn)(void*);
typedef cudaError_t (*cudaEventElapsedTime_fn)(float*, void*, void*);
typedef cudaError_t (*cudaEventDestroy_fn)(void*);

/* Real function pointers */
static ncclAllReduce_fn real_ncclAllReduce = NULL;
static ncclBroadcast_fn real_ncclBroadcast = NULL;
static ncclReduce_fn real_ncclReduce = NULL;
static ncclAllGather_fn real_ncclAllGather = NULL;
static ncclReduceScatter_fn real_ncclReduceScatter = NULL;
static ncclSend_fn real_ncclSend = NULL;
static ncclRecv_fn real_ncclRecv = NULL;
static ncclGroupStart_fn real_ncclGroupStart = NULL;
static ncclGroupEnd_fn real_ncclGroupEnd = NULL;
static ncclCommCount_fn real_ncclCommCount = NULL;
static ncclCommUserRank_fn real_ncclCommUserRank = NULL;

static cudaStreamSynchronize_fn real_cudaStreamSynchronize = NULL;
static cudaEventCreate_fn real_cudaEventCreate = NULL;
static cudaEventRecord_fn real_cudaEventRecord = NULL;
static cudaEventSynchronize_fn real_cudaEventSynchronize = NULL;
static cudaEventElapsedTime_fn real_cudaEventElapsedTime = NULL;
static cudaEventDestroy_fn real_cudaEventDestroy = NULL;

/* Trace state */
static FILE* trace_file = NULL;
static struct timespec start_time;
static pthread_mutex_t trace_mutex = PTHREAD_MUTEX_INITIALIZER;
static int initialized = 0;
static int rank = -1;
static int world_size = -1;
static int in_group = 0;
static uint64_t op_id = 0;
static int sync_mode = 0;  /* 0 = async (default), 1 = sync for accurate timing */

/* Get datatype size in bytes */
static size_t dtype_size(ncclDataType_t dtype) {
    switch(dtype) {
        case ncclInt8:
        case ncclUint8:
            return 1;
        case ncclFloat16:
        case ncclBfloat16:
            return 2;
        case ncclInt32:
        case ncclUint32:
        case ncclFloat32:
            return 4;
        case ncclInt64:
        case ncclUint64:
        case ncclFloat64:
            return 8;
        default:
            return 4;
    }
}

/* Get datatype name */
static const char* dtype_name(ncclDataType_t dtype) {
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

/* Get reduction op name */
static const char* op_name(ncclRedOp_t op) {
    switch(op) {
        case ncclSum: return "sum";
        case ncclProd: return "prod";
        case ncclMax: return "max";
        case ncclMin: return "min";
        case ncclAvg: return "avg";
        default: return "unknown";
    }
}

/* Get elapsed time in microseconds since start */
static double get_elapsed_us(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (now.tv_sec - start_time.tv_sec) * 1e6 +
           (now.tv_nsec - start_time.tv_nsec) / 1e3;
}

/* Initialize the tracer */
static void init_tracer(void) {
    if (initialized) return;

    pthread_mutex_lock(&trace_mutex);
    if (initialized) {
        pthread_mutex_unlock(&trace_mutex);
        return;
    }

    /* Get timing reference */
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    /* Get rank from environment */
    char* rank_str = getenv("RANK");
    if (!rank_str) rank_str = getenv("LOCAL_RANK");
    if (!rank_str) rank_str = getenv("OMPI_COMM_WORLD_RANK");
    if (!rank_str) rank_str = getenv("PMI_RANK");
    rank = rank_str ? atoi(rank_str) : 0;

    /* Get world size from environment */
    char* world_str = getenv("WORLD_SIZE");
    if (!world_str) world_str = getenv("OMPI_COMM_WORLD_SIZE");
    if (!world_str) world_str = getenv("PMI_SIZE");
    world_size = world_str ? atoi(world_str) : 1;

    /* Check sync mode */
    char* sync_str = getenv("SPECTER_SYNC");
    sync_mode = sync_str ? atoi(sync_str) : 0;

    /* Open trace file */
    char* output_dir = getenv("SPECTER_OUTPUT");
    if (!output_dir) output_dir = ".";

    char filename[512];
    snprintf(filename, sizeof(filename), "%s/specter_nccl_rank%d.jsonl", output_dir, rank);
    trace_file = fopen(filename, "w");

    if (!trace_file) {
        fprintf(stderr, "[SPECTER] Failed to open trace file: %s\n", filename);
        pthread_mutex_unlock(&trace_mutex);
        return;
    }

    /* Write header */
    fprintf(trace_file, "{\"event\":\"init\",\"rank\":%d,\"world_size\":%d,\"sync_mode\":%d,\"pid\":%d}\n",
            rank, world_size, sync_mode, getpid());
    fflush(trace_file);

    /* Load real NCCL functions */
    void* nccl_handle = dlopen("libnccl.so.2", RTLD_LAZY | RTLD_GLOBAL);
    if (!nccl_handle) {
        nccl_handle = dlopen("libnccl.so", RTLD_LAZY | RTLD_GLOBAL);
    }
    if (!nccl_handle) {
        fprintf(stderr, "[SPECTER] Failed to load libnccl.so: %s\n", dlerror());
    } else {
        real_ncclAllReduce = (ncclAllReduce_fn)dlsym(nccl_handle, "ncclAllReduce");
        real_ncclBroadcast = (ncclBroadcast_fn)dlsym(nccl_handle, "ncclBroadcast");
        real_ncclReduce = (ncclReduce_fn)dlsym(nccl_handle, "ncclReduce");
        real_ncclAllGather = (ncclAllGather_fn)dlsym(nccl_handle, "ncclAllGather");
        real_ncclReduceScatter = (ncclReduceScatter_fn)dlsym(nccl_handle, "ncclReduceScatter");
        real_ncclSend = (ncclSend_fn)dlsym(nccl_handle, "ncclSend");
        real_ncclRecv = (ncclRecv_fn)dlsym(nccl_handle, "ncclRecv");
        real_ncclGroupStart = (ncclGroupStart_fn)dlsym(nccl_handle, "ncclGroupStart");
        real_ncclGroupEnd = (ncclGroupEnd_fn)dlsym(nccl_handle, "ncclGroupEnd");
        real_ncclCommCount = (ncclCommCount_fn)dlsym(nccl_handle, "ncclCommCount");
        real_ncclCommUserRank = (ncclCommUserRank_fn)dlsym(nccl_handle, "ncclCommUserRank");
    }

    /* Load CUDA runtime functions */
    void* cuda_handle = dlopen("libcudart.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!cuda_handle) {
        cuda_handle = dlopen("libcudart.so.12", RTLD_LAZY | RTLD_GLOBAL);
    }
    if (!cuda_handle) {
        cuda_handle = dlopen("libcudart.so.11.0", RTLD_LAZY | RTLD_GLOBAL);
    }
    if (cuda_handle) {
        real_cudaStreamSynchronize = (cudaStreamSynchronize_fn)dlsym(cuda_handle, "cudaStreamSynchronize");
        real_cudaEventCreate = (cudaEventCreate_fn)dlsym(cuda_handle, "cudaEventCreate");
        real_cudaEventRecord = (cudaEventRecord_fn)dlsym(cuda_handle, "cudaEventRecord");
        real_cudaEventSynchronize = (cudaEventSynchronize_fn)dlsym(cuda_handle, "cudaEventSynchronize");
        real_cudaEventElapsedTime = (cudaEventElapsedTime_fn)dlsym(cuda_handle, "cudaEventElapsedTime");
        real_cudaEventDestroy = (cudaEventDestroy_fn)dlsym(cuda_handle, "cudaEventDestroy");
    }

    if (rank == 0) {
        fprintf(stderr, "[SPECTER] NCCL tracer initialized (rank %d/%d, sync=%d)\n",
                rank, world_size, sync_mode);
        fprintf(stderr, "[SPECTER] Trace output: %s\n", filename);
    }

    initialized = 1;
    pthread_mutex_unlock(&trace_mutex);
}

/* Log an operation */
static void log_op(const char* op, size_t count, ncclDataType_t dtype,
                   double start_us, double duration_us, ncclComm_t comm,
                   int peer, ncclRedOp_t redop) {
    if (!trace_file) return;

    size_t bytes = count * dtype_size(dtype);

    /* Get comm info if available */
    int comm_rank = rank;
    int comm_size = world_size;
    if (real_ncclCommUserRank && comm) {
        real_ncclCommUserRank(comm, &comm_rank);
    }
    if (real_ncclCommCount && comm) {
        real_ncclCommCount(comm, &comm_size);
    }

    pthread_mutex_lock(&trace_mutex);

    fprintf(trace_file,
        "{\"id\":%lu,\"op\":\"%s\",\"start_us\":%.1f,\"duration_us\":%.1f,"
        "\"count\":%zu,\"bytes\":%zu,\"dtype\":\"%s\"",
        op_id++, op, start_us, duration_us, count, bytes, dtype_name(dtype));

    if (peer >= 0) {
        fprintf(trace_file, ",\"peer\":%d", peer);
    }

    if (redop >= 0) {
        fprintf(trace_file, ",\"redop\":\"%s\"", op_name(redop));
    }

    fprintf(trace_file, ",\"comm_rank\":%d,\"comm_size\":%d,\"in_group\":%d}\n",
            comm_rank, comm_size, in_group);

    fflush(trace_file);
    pthread_mutex_unlock(&trace_mutex);
}

/* Measure operation with optional sync */
static double measure_op(cudaStream_t stream, double start_us) {
    double end_us = start_us;

    if (sync_mode && real_cudaStreamSynchronize && stream) {
        real_cudaStreamSynchronize(stream);
        end_us = get_elapsed_us();
    } else {
        end_us = get_elapsed_us();
    }

    return end_us - start_us;
}

/* ==================== NCCL Function Intercepts ==================== */

ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, ncclRedOp_t op,
                           ncclComm_t comm, cudaStream_t stream) {
    init_tracer();

    if (!real_ncclAllReduce) {
        fprintf(stderr, "[SPECTER] ncclAllReduce not loaded!\n");
        return ncclInternalError;
    }

    double start_us = get_elapsed_us();
    ncclResult_t result = real_ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
    double duration_us = measure_op(stream, start_us);

    log_op("AllReduce", count, datatype, start_us, duration_us, comm, -1, op);

    return result;
}

ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, int root,
                           ncclComm_t comm, cudaStream_t stream) {
    init_tracer();

    if (!real_ncclBroadcast) {
        fprintf(stderr, "[SPECTER] ncclBroadcast not loaded!\n");
        return ncclInternalError;
    }

    double start_us = get_elapsed_us();
    ncclResult_t result = real_ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream);
    double duration_us = measure_op(stream, start_us);

    log_op("Broadcast", count, datatype, start_us, duration_us, comm, root, -1);

    return result;
}

ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count,
                        ncclDataType_t datatype, ncclRedOp_t op, int root,
                        ncclComm_t comm, cudaStream_t stream) {
    init_tracer();

    if (!real_ncclReduce) {
        fprintf(stderr, "[SPECTER] ncclReduce not loaded!\n");
        return ncclInternalError;
    }

    double start_us = get_elapsed_us();
    ncclResult_t result = real_ncclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream);
    double duration_us = measure_op(stream, start_us);

    log_op("Reduce", count, datatype, start_us, duration_us, comm, root, op);

    return result;
}

ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
                           ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
    init_tracer();

    if (!real_ncclAllGather) {
        fprintf(stderr, "[SPECTER] ncclAllGather not loaded!\n");
        return ncclInternalError;
    }

    double start_us = get_elapsed_us();
    ncclResult_t result = real_ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream);
    double duration_us = measure_op(stream, start_us);

    log_op("AllGather", sendcount, datatype, start_us, duration_us, comm, -1, -1);

    return result;
}

ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
                               ncclDataType_t datatype, ncclRedOp_t op,
                               ncclComm_t comm, cudaStream_t stream) {
    init_tracer();

    if (!real_ncclReduceScatter) {
        fprintf(stderr, "[SPECTER] ncclReduceScatter not loaded!\n");
        return ncclInternalError;
    }

    double start_us = get_elapsed_us();
    ncclResult_t result = real_ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
    double duration_us = measure_op(stream, start_us);

    log_op("ReduceScatter", recvcount, datatype, start_us, duration_us, comm, -1, op);

    return result;
}

ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, cudaStream_t stream) {
    init_tracer();

    if (!real_ncclSend) {
        fprintf(stderr, "[SPECTER] ncclSend not loaded!\n");
        return ncclInternalError;
    }

    double start_us = get_elapsed_us();
    ncclResult_t result = real_ncclSend(sendbuff, count, datatype, peer, comm, stream);
    double duration_us = measure_op(stream, start_us);

    log_op("Send", count, datatype, start_us, duration_us, comm, peer, -1);

    return result;
}

ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, cudaStream_t stream) {
    init_tracer();

    if (!real_ncclRecv) {
        fprintf(stderr, "[SPECTER] ncclRecv not loaded!\n");
        return ncclInternalError;
    }

    double start_us = get_elapsed_us();
    ncclResult_t result = real_ncclRecv(recvbuff, count, datatype, peer, comm, stream);
    double duration_us = measure_op(stream, start_us);

    log_op("Recv", count, datatype, start_us, duration_us, comm, peer, -1);

    return result;
}

ncclResult_t ncclGroupStart(void) {
    init_tracer();

    if (!real_ncclGroupStart) {
        return ncclSuccess;
    }

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

    if (!real_ncclGroupEnd) {
        return ncclSuccess;
    }

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

/* Cleanup on unload */
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
