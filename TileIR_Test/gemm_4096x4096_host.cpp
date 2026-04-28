#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    CUresult err = call;                                                       \
    if (err != CUDA_SUCCESS) {                                                 \
      const char *errStr;                                                      \
      cuGetErrorString(err, &errStr);                                          \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
              errStr);                                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define MATRIX_SIZE 4096
#define TILE_SIZE   64
#define WARMUP_ITERS 5
#define BENCH_ITERS  100

int main() {
    CUdevice   cuDevice;
    CUcontext  cuContext;
    CUmodule   cuModule;
    CUfunction gemm_kernel;
    CUstream   stream;

    CUDA_CHECK(cuInit(0));
    CUDA_CHECK(cuDeviceGet(&cuDevice, 0));
    CUDA_CHECK(cuCtxCreate_v4(&cuContext, NULL, 0, cuDevice));
    CUDA_CHECK(cuStreamCreate(&stream, CU_STREAM_DEFAULT));

    // 加载编译好的 cubin 并获取 kernel 函数入口
    CUDA_CHECK(cuModuleLoad(&cuModule, "gemm_4096x4096_base.cubin"));
    CUDA_CHECK(cuModuleGetFunction(&gemm_kernel, cuModule,
                                   "gemm_baseline_kernel"));
    // CUDA_CHECK(cuModuleLoad(&cuModule, "gemm_4096x4096_prefetch.cubin"));
    // CUDA_CHECK(cuModuleGetFunction(&gemm_kernel, cuModule,
    //                                "gemm_square_4096_tile_32x32_fixed_kernel"));

    // ----------------------------------------------------------------
    // 分配并初始化 host 端矩阵 A、B（行主序，4096×4096 float）
    // ----------------------------------------------------------------
    const size_t N       = MATRIX_SIZE;
    const size_t n_elems = N * N;
    const size_t n_bytes = n_elems * sizeof(float);

    float *h_A = (float *)malloc(n_bytes);
    float *h_B = (float *)malloc(n_bytes);
    if (!h_A || !h_B) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    // 用简单值初始化，A[i][j] = i*0.01f，B[i][j] = j*0.01f
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++) {
            h_A[i * N + j] = (float)i * 0.01f;
            h_B[i * N + j] = (float)j * 0.01f;
        }

    // ----------------------------------------------------------------
    // 在 device 端分配 A、B、C
    // ----------------------------------------------------------------
    CUdeviceptr d_A, d_B, d_C;
    CUDA_CHECK(cuMemAlloc(&d_A, n_bytes));
    CUDA_CHECK(cuMemAlloc(&d_B, n_bytes));
    CUDA_CHECK(cuMemAlloc(&d_C, n_bytes));

    // 将 A、B 拷贝到 device，C 清零
    CUDA_CHECK(cuMemcpyHtoD(d_A, h_A, n_bytes));
    CUDA_CHECK(cuMemcpyHtoD(d_B, h_B, n_bytes));
    CUDA_CHECK(cuMemsetD32(d_C, 0, n_elems));   // 按 32-bit word 清零

    // ----------------------------------------------------------------
    // 配置 launch 参数
    //   网格：(4096/64) × (4096/64) = 64×64 tile blocks
    //   每个 tile block 的 thread block 固定为 (1,1,1)（CUDA Tile 规范）
    // ----------------------------------------------------------------
    const unsigned int grid_x = MATRIX_SIZE / TILE_SIZE;  // 64
    const unsigned int grid_y = MATRIX_SIZE / TILE_SIZE;  // 64
    const unsigned int grid_z = 1;

    void *kernel_args[] = { &d_A, &d_B, &d_C };

    // ----------------------------------------------------------------
    // CUDA Event，用于精确计时
    // ----------------------------------------------------------------
    CUevent ev_start, ev_stop;
    CUDA_CHECK(cuEventCreate(&ev_start, CU_EVENT_DEFAULT));
    CUDA_CHECK(cuEventCreate(&ev_stop,  CU_EVENT_DEFAULT));

    // ----------------------------------------------------------------
    // 预热（Warmup）：跑几次让 GPU 达到稳定频率
    // ----------------------------------------------------------------
    printf("Warming up (%d iterations)...\n", WARMUP_ITERS);
    for (int i = 0; i < WARMUP_ITERS; i++) {
        CUDA_CHECK(cuLaunchKernel(
            gemm_kernel,
            grid_x, grid_y, grid_z,   // 网格维度
            1, 1, 1,                  // block 维度（CUDA Tile 固定为 1,1,1）
            0,                        // 共享内存（CUDA Tile 不使用）
            stream,
            kernel_args,
            NULL
        ));
    }
    CUDA_CHECK(cuStreamSynchronize(stream));
    printf("Warmup done.\n\n");

    // ----------------------------------------------------------------
    // 正式 Benchmark：重复执行 BENCH_ITERS 次，整体用 Event 计时
    // ----------------------------------------------------------------
    CUDA_CHECK(cuEventRecord(ev_start, stream));

    for (int i = 0; i < BENCH_ITERS; i++) {
        CUDA_CHECK(cuLaunchKernel(
            gemm_kernel,
            grid_x, grid_y, grid_z,
            1, 1, 1,
            0,
            stream,
            kernel_args,
            NULL
        ));
    }

    CUDA_CHECK(cuEventRecord(ev_stop, stream));
    CUDA_CHECK(cuStreamSynchronize(stream));

    // cuEventElapsedTime 返回毫秒
    float total_ms = 0.0f;
    CUDA_CHECK(cuEventElapsedTime(&total_ms, ev_start, ev_stop));

    float avg_ms   = total_ms / BENCH_ITERS;
    float avg_us   = avg_ms * 1000.0f;

    // 理论 FLOPs：4096×4096×4096×2（乘加各算一次）
    double flops      = 2.0 * (double)N * (double)N * (double)N;
    double tflops_avg = flops / (avg_ms * 1e-3) / 1e12;

    printf("=== Benchmark Results (%d iterations) ===\n", BENCH_ITERS);
    printf("  Total   time : %.3f ms\n",   total_ms);
    printf("  Average time : %.3f ms  (%.1f us)\n", avg_ms, avg_us);
    printf("  Throughput   : %.4f TFLOPS\n", tflops_avg);

    // ----------------------------------------------------------------
    // （可选）把结果拷回 host 做简单校验
    // ----------------------------------------------------------------
#ifdef VERIFY
    float *h_C = (float *)malloc(n_bytes);
    CUDA_CHECK(cuMemcpyDtoH(h_C, d_C, n_bytes));
    // 只打印左上角 4×4 供目视检查
    printf("\nC[0:4][0:4]:\n");
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++)
            printf("%10.2f ", h_C[r * N + c]);
        printf("\n");
    }
    free(h_C);
#endif

    // ----------------------------------------------------------------
    // 清理
    // ----------------------------------------------------------------
    CUDA_CHECK(cuEventDestroy(ev_start));
    CUDA_CHECK(cuEventDestroy(ev_stop));
    CUDA_CHECK(cuMemFree(d_A));
    CUDA_CHECK(cuMemFree(d_B));
    CUDA_CHECK(cuMemFree(d_C));
    free(h_A);
    free(h_B);
    CUDA_CHECK(cuModuleUnload(cuModule));
    CUDA_CHECK(cuCtxDestroy(cuContext));

    return 0;
}