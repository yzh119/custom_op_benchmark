#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

/*
 * CUDA Kernel of the forward function for Masked Matrix Multiplication:
 * O = adj * (A @ B)
 * This is an unoptimized version, to better utilize shared memory, some sort of padding is required.
 * Note that we use the row and col vector to represent the sparse matrix adj.
 */

__global__ void maskedmm_forward_kernel(int* __restrict__ row, int* __restrict__ col, float* __restrict__ A, float* __restrict__ B, float* __restrict__ O, int e, int d, int n) {
    int i = ((((int)blockIdx.x) * 32) + ((int)threadIdx.x));
    if (((int)blockIdx.x) < (e / 32)) {
        O[i] = 0.000000e+00f;
        for (int k = 0; k < d; ++k) {
            O[i] = (O[i] + (A[((row[i] * d) + k)] * B[(col[i] + (k * n))]));
        }
    } else {
        if (i < e) {
            O[i] = 0.000000e+00f;
        }
        for (int k = 0; k < d; ++k) {
            if (i < e) {
                O[i] = (O[i] + (A[((row[i] * d) + k)] * B[(col[i] + (k * n))]));
            }
        }
    }
}

/*
 * CUDA Kernel of the backward function for Masked Matrix Multiplication: 
 * dA = B @ (dO * adj)
 * dB = A @ (dO * adj)
 * Mostly the same as src_mul_edge
 */
const int mb_block_dim = 512;
const int mb_thread_dim = 512;
__global__ void maskedmm_backward_kernel(int* __restrict__ row, int* __restrict__ col, float* __restrict__ A, float* __restrict__ B, float* __restrict__ dO, float* __restrict__ dA, float* __restrict__ dB, int e, int d, int n) {
    int i = (int)blockIdx.x;
    int j = (int)threadIdx.x;
    for (int ki = i; ki < n; ki += mb_block_dim)
        for (int kj = j; kj < d; kj += mb_thread_dim) {
            dA[ki * d + kj] = 0;
            dB[ki * d + kj] = 0;
        }

    for (int ki = i; ki < e; ki += mb_block_dim) {
        float dO_at = dO[ki];
        for (int kj = 0; kj < d; kj += mb_thread_dim) { 
            dA[col[ki] * d + kj] += dO_at * B[row[ki] * d + kj];
            dB[row[ki] * d + kj] += dO_at * A[col[ki] * d + kj];
        }
    }
}

/*
 * CUDA Kernel of forward function for Sparse Softmax
 * O = softmax(x), reduced by per node.
 */
__global__ void sparse_softmax_forward_kernel(int* __restrict__ head, int* __restrict__ idx, float* __restrict__ x, float* __restrict__ O, int e) {
    float max_val = *x;
    int j = (int)threadIdx.x;
    if (j < e) {
        for (int k = head[j]; k < head[j + 1]; ++k)
            max_val = max(max_val, x[idx[k]]);

        float sum = 0;
        for (int k = head[j]; k < head[j + 1]; ++k) {
            float now = exp(x[idx[k]] - max_val);
            O[idx[k]] = now;
            sum += now;
        }

        for (int k = head[j]; k < head[j + 1]; ++k)
            O[idx[k]] /= sum;
    }
}

/*
 * CUDA Kernel of backward function for Sparse Softmax.
 */
const int ss_thread_dim = 32;
__global__ void sparse_softmax_backward_kernel(int* __restrict__ head, int* __restrict__ idx, float* __restrict__ dO, float* __restrict__ O, float* __restrict__ dx, int e) {
    int i = (int)blockIdx.x;
    if (i < e) {
        for (int ki = head[i]; ki < head[i + 1]; ++ki) {
            int j = (int)threadIdx.x;
            for (int kj = head[i] + j; kj < head[i + 1]; kj += ss_thread_dim) {
                dx[idx[kj]] -= dO[idx[ki]] * O[idx[ki]] * O[idx[kj]];
                if (ki == kj) dx[idx[kj]] += dO[idx[ki]] * O[idx[ki]];
            }
        }
    }
}

at::Tensor maskedmm_cuda_forward(
    at::Tensor row,
    at::Tensor col,
    at::Tensor A,
    at::Tensor B) {
    // row, col: (e), A, B: (n, d), O: (e)
    const auto e = row.size(0);
    const auto n = A.size(0);
    const auto d = A.size(1);
    auto O = at::zeros({e}, A.options());

    const int threads = 32;
    const dim3 blocks((e + threads - 1) / threads);
    auto Bt = B.transpose(0, 1).contiguous();

    maskedmm_forward_kernel<<<blocks, threads>>>(
        row.data<int>(),
        col.data<int>(),
        A.data<float>(),
        Bt.data<float>(),
        O.data<float>(),
        e, d, n);
    return O;
}

std::vector<at::Tensor> maskedmm_cuda_backward(
    at::Tensor row,
    at::Tensor col,
    at::Tensor A,
    at::Tensor B,
    at::Tensor dO) {
    // row, col: (e), dO: (e), A, B: (n, d)
    const auto e = row.size(0);
    const auto n = A.size(0);
    const auto d = A.size(1);

    const int threads = mb_thread_dim;
    const dim3 blocks(mb_block_dim);

    auto dA = at::zeros_like(A, A.options());
    auto dB = at::zeros_like(B, B.options());
    dO = dO.contiguous();

    maskedmm_backward_kernel<<<blocks, threads>>>(
        row.data<int>(),
        col.data<int>(),
        A.data<float>(),
        B.data<float>(),
        dO.data<float>(),
        dA.data<float>(),
        dB.data<float>(),
        e, d, n);
    return {dA, dB};
}
