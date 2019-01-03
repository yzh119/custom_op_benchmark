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
__global__ void maskedmm_backward_kernel(int* __restrict__ row, int* __restrict__ col, float* __restrict__ A, float* __restrict__ B, float* __restrict__ dO, float* __restrict__ dA, float* __restrict__ dB, int e, int d, int n) {
    const int block_dim = 512;
    const int thread_dim = 512;
    __shared__ float dO_shared[block_dim];
    int i = (int)blockIdx.x;
    int j = (int)threadIdx.x;
    for (int bi = 0; bi < n; bi += block_dim)
        if (i + bi < n)
            for (int tj = 0; tj < d; tj += thread_dim) 
                if (j + tj < d) {
                    dA[(i + bi) * d + (j + tj)] = 0;
                    dB[(i + bi) * d + (j + tj)] = 0;
                }

    for (int bi = 0; bi < e; bi += block_dim)
        if (i + bi < e)
            dO_shared[i] = dO[i + bi];

    __syncthreads();

    for (int bi = 0; bi < e; bi += block_dim) {
        if (i + bi < e) {
            for (int tj = 0; tj < d; tj += thread_dim)
                if (j + tj < d) {
                    dA[col[i + bi] * d + (j + tj)] += dO_shared[i] * B[row[i + bi] * d + (j + tj)];
                    dB[row[i + bi] * d + (j + tj)] += dO_shared[i] * A[col[i + bi] * d + (j + tj)];
                }
        }
    }
}

/*
 * CUDA Kernel of Sparse Softmax 
 *
 */
/*
__global__ void sparse_softmax_forward_kernel() {

}

__global__ void sparse_softmax_backward_kernel() {

}
*/

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

    const int threads = 512; //d <= 512 ? d: 1024;
    const dim3 blocks(512);

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
