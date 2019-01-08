/* TODOs
 * segment_reduce_forward, segment_reduce_backward;
 */

#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include <vector>

/*
 * CUDA Kernel of the forward function for Masked Matrix Multiplication:
 * y = adj * (A @ B)
 * This is an unoptimized version, to better utilize shared memory, some sort of padding is required.
 * Note that we use the row and col vector to represent the sparse matrix adj. (coo format)
 */
__global__ void maskedmm_forward_kernel(int* __restrict__ row, int* __restrict__ col, float* __restrict__ A, float* __restrict__ B, float* __restrict__ y, int e, int d, int n) {
    int i = ((((int)blockIdx.x) * (int)blockDim.x) + ((int)threadIdx.x));
    if (((int)blockIdx.x) < (e / (int)blockDim.x)) {
        y[i] = 0.000000e+00f;
        for (int k = 0; k < d; ++k) {
            y[i] = (y[i] + (A[((row[i] * d) + k)] * B[(col[i] + (k * n))]));
        }
    } else {
        if (i < e) {
            y[i] = 0.000000e+00f;
        }
        for (int k = 0; k < d; ++k) {
            if (i < e) {
                y[i] = (y[i] + (A[((row[i] * d) + k)] * B[(col[i] + (k * n))]));
            }
        }
    }
}

/*
 * CUDA Kernel of the backward function for Masked Matrix Multiplication: 
 * dA = B @ (dy * adj)
 * dB = A @ (dy * adj)
 * Mostly the same as src_mul_edge
 */
__global__ void maskedmm_backward_kernel(int* __restrict__ row, int* __restrict__ col, float* __restrict__ A, float* __restrict__ B, float* __restrict__ dy, float* __restrict__ dA, float* __restrict__ dB, int e, int d, int n) {
    int j = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    for (int k = 0; k < n; ++k) {
        dA[k * d + j] = 0;
        dB[k * d + j] = 0;
    }

    for (int k = 0; k < e; ++k) {
        dA[row[k] * d + j] += dy[k] * B[col[k] * d + j];
        dB[col[k] * d + j] += dy[k] * A[row[k] * d + j];
    }
}

/*
 * CUDA Kernel of forward function for Sparse Softmax
 * y = softmax(x), grouped by node.
 * head, idx: csr format
 */
__global__ void sparse_softmax_forward_kernel(int* __restrict__ head, int* __restrict__ idx, float* __restrict__ x, float* __restrict__ y, int n) {
    float max_val = *x;
    int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (i < n) {
        for (int k = head[i]; k < head[i + 1]; ++k)
            max_val = max(max_val, x[idx[k]]);

        float sum = 0;
        for (int k = head[i]; k < head[i + 1]; ++k) {
            float now = exp(x[idx[k]] - max_val);
            y[idx[k]] = now;
            sum += now;
        }

        for (int k = head[i]; k < head[i + 1]; ++k)
            y[idx[k]] /= sum;
    }
}

/*
 * CUDA Kernel of backward function for Sparse Softmax.
 * head, idx: csr format
 */
__global__ void sparse_softmax_backward_kernel(int* __restrict__ head, int* __restrict__ idx, float* __restrict__ dy, float* __restrict__ y, float* __restrict__ dx, int n) {
    int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int j = (int)threadIdx.y;
    if (i < n) {
        for (int kj = head[i] + j; kj < head[i + 1]; kj += (int)blockDim.y) {
            for (int ki = head[i]; ki < head[i + 1]; ++ki) {
                dx[idx[kj]] -= dy[idx[ki]] * y[idx[ki]] * y[idx[kj]];
                if (ki == kj) dx[idx[kj]] += dy[idx[ki]] * y[idx[ki]];
            }
        }
    }
}

at::Tensor maskedmm_cuda_forward(
    at::Tensor row,
    at::Tensor col,
    at::Tensor A,
    at::Tensor B) {
    // row, col: (e), A, B: (n, d), y: (e)
    const auto e = row.size(0);
    const auto n = A.size(0);
    const auto d = A.size(1);
    auto y = at::zeros({e}, A.options());

    const int threads = 32;
    const dim3 blocks((e + threads - 1) / threads);
    auto Bt = B.transpose(0, 1).contiguous();

    maskedmm_forward_kernel<<<blocks, threads>>>(
        row.data<int>(),
        col.data<int>(),
        A.data<float>(),
        Bt.data<float>(),
        y.data<float>(),
        e, d, n);
    return y;
}

std::vector<at::Tensor> maskedmm_cuda_backward(
    at::Tensor row,
    at::Tensor col,
    at::Tensor A,
    at::Tensor B,
    at::Tensor dy) {
    // row, col: (e), dy: (e), A, B: (n, d);
    const auto e = row.size(0);
    const auto n = A.size(0);
    const auto d = A.size(1);

    const int threads = 1024; 
    const dim3 blocks((d + threads - 1) / 1024);

    auto dA = at::zeros_like(A, A.options());
    auto dB = at::zeros_like(B, B.options());
    dy = dy.contiguous();

    maskedmm_backward_kernel<<<blocks, threads>>>(
        row.data<int>(),
        col.data<int>(),
        A.data<float>(),
        B.data<float>(),
        dy.data<float>(),
        dA.data<float>(),
        dB.data<float>(),
        e, d, n);
    return {dA, dB};
}

at::Tensor sparse_softmax_cuda_forward(
    at::Tensor head,
    at::Tensor idx,
    at::Tensor x) {
    // head: (n + 1), idx: (e), x: (e);
    const auto n = head.size(0) - 1;
    const int threads = 1024;
    const dim3 blocks((n + threads - 1) /  1024);
    
    const auto y = at::zeros_like(x, x.options());
    
    sparse_softmax_forward_kernel<<<blocks, threads>>>(
        head.data<int>(),
        idx.data<int>(),
        x.data<float>(),
        y.data<float>(),
        n);
    return y;
}

at::Tensor sparse_softmax_cuda_backward(
    at::Tensor head,
    at::Tensor idx,
    at::Tensor y,
    at::Tensor dy) {
    // head: (n + 1), idx: (e), y: (e), dy: (e);
    const auto n = head.size(0) - 1;
    const int threads_x = 32, threads_y = 1024 / threads_x;
    const dim3 threads(threads_x, threads_y);
    const dim3 blocks((n + threads_x - 1) / threads_x);
    
    dy = dy.contiguous(); 
    
    const auto dx = at::zeros_like(dy, dy.options());

    sparse_softmax_backward_kernel<<<blocks, threads>>>(
        head.data<int>(),
        idx.data<int>(),
        dy.data<float>(),
        y.data<float>(),
        dx.data<float>(),
        n); 
    return dx;
}
