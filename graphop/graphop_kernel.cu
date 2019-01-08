/* TODOs
 * segment_reduce_forward, segment_reduce_backward;
 */

#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/Type.h>
#include <c10/util/Exception.h>

#define AT_CASE_ITYPE(enum_type, type, DTYPE, NAME, ...)                    \
  case enum_type: {                                                         \
    const at::Type& dtype = DTYPE;                                          \
    using idx_t = type;                                                     \
    switch (dtype.scalarType()) {                                           \
      case at::ScalarType::Float: {                                         \
        using data_t = float;                                               \
        return __VA_ARGS__();                                               \
      }                                                                     \
      case at::ScalarType::Double: {                                        \
        using data_t = double;                                              \
        return __VA_ARGS__();                                               \
      }                                                                     \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", dtype.toString(), "'");   \
    }                                                                       \
  }

#define AT_DISPATCH_IDX_DATA_TYPES(ITYPE, DTYPE, NAME, ...)                             \
  [&] {                                                                                 \
    const at::Type& itype = ITYPE;                                                      \
    switch (itype.scalarType()) {                                                       \
      AT_CASE_ITYPE(at::ScalarType::Int, int32_t, DTYPE, NAME, __VA_ARGS__)             \
      AT_CASE_ITYPE(at::ScalarType::Long, int64_t, DTYPE, NAME, __VA_ARGS__)            \
      default:                                                                          \
        AT_ERROR(#NAME, " not implemented for '", itype.toString(), "'");               \
    }                                                                                   \
  }()

/*
 * CUDA Kernel of the forward function for Masked Matrix Multiplication:
 * y = adj * (A @ B)
 * This is an unoptimized version, to better utilize shared memory, some sort of padding is required.
 * Note that we use the row and col vector to represent the sparse matrix adj. (coo format)
 */
template <typename idx_t, typename data_t>
__global__ void maskedmm_forward_kernel(idx_t* __restrict__ row, idx_t* __restrict__ col, data_t* __restrict__ A, data_t* __restrict__ B, data_t* __restrict__ y, int e, int d, int n) {
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
template <class idx_t, class data_t>
__global__ void maskedmm_backward_kernel(idx_t* __restrict__ row, idx_t* __restrict__ col, data_t* __restrict__ A, data_t* __restrict__ B, data_t* __restrict__ dy, data_t* __restrict__ dA, data_t* __restrict__ dB, int e, int d, int n) {
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
 * CUDA Kernel of the backward function for Masked Matrix Multiplication. (argument: csr format)
 * TODO
 */
template <class idx_t, class data_t>
__global__ void maskedmm_backward_kernel_csr(idx_t* ptr_row, idx_t* eid_row, idx_t* ptr_col, idx_t* eid_col, data_t* __restrict__ A, data_t* __restrict__ B, data_t* __restrict__ dy, data_t* __restrict__ dA, data_t* __restrict__ dB, int e, int d, int n) {}

/*
 * CUDA Kernel of forward function for Sparse Softmax
 * y = softmax(x), grouped by node.
 * ptr, eid: csr format
 */
template <class idx_t, class data_t>
__global__ void sparse_softmax_forward_kernel(idx_t* __restrict__ ptr, idx_t* __restrict__ eid, data_t* __restrict__ x, data_t* __restrict__ y, int n) {
    float max_val = *x;
    int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (i < n) {
        for (int k = ptr[i]; k < ptr[i + 1]; ++k)
            max_val = max(max_val, x[eid[k]]);

        float sum = 0;
        for (int k = ptr[i]; k < ptr[i + 1]; ++k) {
            float now = exp(x[eid[k]] - max_val);
            y[eid[k]] = now;
            sum += now;
        }

        for (int k = ptr[i]; k < ptr[i + 1]; ++k)
            y[eid[k]] /= sum;
    }
}

/*
 * CUDA Kernel of backward function for Sparse Softmax.
 * ptr, eid: csr format
 */
template <class idx_t, class data_t>
__global__ void sparse_softmax_backward_kernel(idx_t* __restrict__ ptr, idx_t* __restrict__ eid, data_t* __restrict__ dy, data_t* __restrict__ y, data_t* __restrict__ dx, int n) {
    int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int j = (int)threadIdx.y;
    if (i < n) {
        for (int kj = ptr[i] + j; kj < ptr[i + 1]; kj += (int)blockDim.y) {
            for (int ki = ptr[i]; ki < ptr[i + 1]; ++ki) {
                dx[eid[kj]] -= dy[eid[ki]] * y[eid[ki]] * y[eid[kj]];
                if (ki == kj) dx[eid[kj]] += dy[eid[ki]] * y[eid[ki]];
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

    AT_DISPATCH_IDX_DATA_TYPES(row.type(), A.type(), "maskedmm_cuda_forward", ([&] {
        maskedmm_forward_kernel<idx_t, data_t><<<blocks, threads>>>(
            row.data<idx_t>(),
            col.data<idx_t>(),
            A.data<data_t>(),
            Bt.data<data_t>(),
            y.data<data_t>(),
            e, d, n);
    }));
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

    AT_DISPATCH_IDX_DATA_TYPES(row.type(), A.type(), "maskedmm_cuda_backward", ([&] {
        maskedmm_backward_kernel<idx_t, data_t><<<blocks, threads>>>(
            row.data<idx_t>(),
            col.data<idx_t>(),
            A.data<data_t>(),
            B.data<data_t>(),
            dy.data<data_t>(),
            dA.data<data_t>(),
            dB.data<data_t>(),
            e, d, n);
    }));
    return {dA, dB};
}

at::Tensor sparse_softmax_cuda_forward(
    at::Tensor ptr,
    at::Tensor eid,
    at::Tensor x) {
    // ptr: (n + 1), eid: (e), x: (e);
    const auto n = ptr.size(0) - 1;
    const int threads = 1024;
    const dim3 blocks((n + threads - 1) /  1024);
    
    const auto y = at::zeros_like(x, x.options());
    
    AT_DISPATCH_IDX_DATA_TYPES(eid.type(), x.type(), "sparse_softmax_cuda_forward",([&] {
        sparse_softmax_forward_kernel<idx_t, data_t><<<blocks, threads>>>(
            ptr.data<idx_t>(),
            eid.data<idx_t>(),
            x.data<data_t>(),
            y.data<data_t>(),
            n);
    }));
    return y;
}

at::Tensor sparse_softmax_cuda_backward(
    at::Tensor ptr,
    at::Tensor eid,
    at::Tensor y,
    at::Tensor dy) {
    // ptr: (n + 1), eid: (e), y: (e), dy: (e);
    const auto n = ptr.size(0) - 1;
    const int threads_x = 32, threads_y = 1024 / threads_x;
    const dim3 threads(threads_x, threads_y);
    const dim3 blocks((n + threads_x - 1) / threads_x);
    
    dy = dy.contiguous(); 
    
    const auto dx = at::zeros_like(dy, dy.options());

    AT_DISPATCH_IDX_DATA_TYPES(eid.type(), y.type(), "sparse_softmax_cuda_backward", ([&] {
        sparse_softmax_backward_kernel<idx_t, data_t><<<blocks, threads>>>(
            ptr.data<idx_t>(),
            eid.data<idx_t>(),
            dy.data<data_t>(),
            y.data<data_t>(),
            dx.data<data_t>(),
            n); 
    }));
    return dx;
}
