// First CUDA GPU oracle for the image track: one dense projection (img_in) computed on the device and
// qualified against the committed CPU oracle img_in.npy. Self-contained raw CUDA (no infernix_ops, no
// shape registration) so it isolates the GPU cycle mechanics -- build the .cu target with the server up
// (nvcc needs no VRAM), run it during a short self-restoring blackout with the GPU free, log the result.
//
// out[M,N] = x[M,K] @ W^T  where W is a torch Linear weight [N=out, K=in]. Matches qioracle::matmul_xwt.
// Skips (77) when INFERNIX_IMAGE_ORACLE is unset.

#include "models/qwen_image/qwen_image_oracle_ref.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace qo = qioracle;

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err_ = (call);                                          \
        if (err_ != cudaSuccess) {                                          \
            std::printf("CUDA_ERROR %s at %s:%d: %s\n", #call, __FILE__,   \
                        __LINE__, cudaGetErrorString(err_));                \
            std::fflush(stdout);                                            \
            return 1;                                                       \
        }                                                                   \
    } while (0)

// C[M,N] = sum_k A[M,K] * W[N,K]  (x @ W^T). One thread per (m,n) output, naive K reduction.
__global__ void matmul_xwt_kernel(const float* __restrict__ A, const float* __restrict__ W,
                                  float* __restrict__ C, int M, int N, int K) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;
    float acc = 0.0F;
    for (int k = 0; k < K; ++k) {
        acc += A[m * K + k] * W[n * K + k];
    }
    C[m * N + n] = acc;
}

int main() {
    const char* root_env = std::getenv("INFERNIX_IMAGE_ORACLE");
    if (root_env == nullptr || root_env[0] == '\0') {
        std::printf("SKIP: INFERNIX_IMAGE_ORACLE not set\n");
        return 77;
    }
    const std::string root(root_env);
    const std::string T = qo::join(root, "tensors");
    const std::string TW = qo::join(root, "tensors_toplevel_weights");

    const std::vector<float> x = qo::read_npy(qo::join(T, "in_hidden_states.npy")).floats();    // [64,64]
    const std::vector<float> w = qo::read_npy(qo::join(TW, "toplevel_img_in_weight.npy")).floats();  // [4096,64]
    const std::vector<float> ref = qo::read_npy(qo::join(T, "img_in.npy")).floats();            // [1,64,4096]

    const int M = 64, K = 64, N = 4096;  // img_in: Linear(64 -> 4096)

    // Host sanity: input sizes must match the expected oracle shapes.
    if ((int)x.size() != M * K || (int)w.size() != N * K || (int)ref.size() != M * N) {
        std::printf("SHAPE_MISMATCH x=%zu w=%zu ref=%zu\n", x.size(), w.size(), ref.size());
        return 1;
    }

    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
        std::printf("NO_CUDA_DEVICE\n");
        std::fflush(stdout);
        return 2;
    }

    float *dx = nullptr, *dw = nullptr, *dc = nullptr;
    CUDA_CHECK(cudaMalloc(&dx, (size_t)M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dw, (size_t)N * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dc, (size_t)M * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dx, x.data(), (size_t)M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw, w.data(), (size_t)N * K * sizeof(float), cudaMemcpyHostToDevice));

    const dim3 block(32, 8);
    const dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    matmul_xwt_kernel<<<grid, block>>>(dx, dw, dc, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> got(M * N, 0.0F);
    CUDA_CHECK(cudaMemcpy(got.data(), dc, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost));

    const double r = qo::rel_fro(got, ref);
    const bool ok = (r < 1e-3) && std::isfinite(r);
    std::printf("gpu_img_in  rel_fro=%.3e  tol=1.0e-03  %s\n", r, ok ? "OK" : "FAIL");
    std::printf("GPU_DEVICE=%d\n", dev_count > 0 ? 1 : 0);
    std::fflush(stdout);

    cudaFree(dx);
    cudaFree(dw);
    cudaFree(dc);

    if (!ok) {
        std::printf("IMAGE_GEMM_CUDA_TEST: FAIL\n");
        return 1;
    }
    std::printf("IMAGE_GEMM_CUDA_TEST: PASS (M=%d N=%d K=%d)\n", M, N, K);
    return 0;
}
