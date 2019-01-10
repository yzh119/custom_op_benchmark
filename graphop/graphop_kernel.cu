/* TODOs
 * - segment_reduce_forward, segment_reduce_backward;
 * - switch backend from aten to dlpack
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
template <class idx_t, class data_t>
__global__ void maskedmm_forward_kernel(idx_t* __restrict__ row, idx_t* __restrict__ col, data_t* __restrict__ A, data_t* __restrict__ B, data_t* __restrict__ y, int e, int d, int n) {
    int i = (((blockIdx.x) * blockDim.x) + (threadIdx.x));
    if (i < e) {
        data_t sum = 0;
        for (int k = 0; k < d; ++k) {
            sum += A[row[i] * d + k] * B[col[i] + (k * n)];
        }
        y[i] = sum;
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
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    for (int k = 0; k < n; ++k) {
        dA[k * d + j] = 0;
        dB[k * d + j] = 0;
    }
    for (int k = 0; k < e; ++k) {
        dA[row[k] * d + j] += dy[k] * B[col[k] * d + j];
        dB[col[k] * d + j] += dy[k] * A[row[k] * d + j];
    }
}

#define COPY_TO_SHARED(_)                                                       \
if (ko + tx + (_) * blockDim.x < d) {                                           \
    A_shared[tx + (_) * blockDim.x] = A[i * d + ko + tx + (_) * blockDim.x];    \
}    


/*
 * CUDA Kernel of the forward function for Masked Matrix Multiplication. (argument: csr format)
 */
template <class idx_t, class data_t>
__global__ void maskedmm_csr_forward_kernel(idx_t* __restrict__ ptr, idx_t* __restrict__ eid, idx_t* __restrict__ nid, data_t* __restrict__ A, data_t* __restrict__ B, data_t* __restrict__ y, int d, int n) {
    int i = blockIdx.x; 
    int tx = threadIdx.x;
    __shared__ data_t A_shared[128];
    if (i < n) {
        for (int _j = ptr[i]; _j < ptr[i + 1]; _j += blockDim.x) {
            int j = _j + tx;
            data_t sum = 0;
            for (int ko = 0; ko < d; ko += 4 * blockDim.x) {
                COPY_TO_SHARED(0);
                COPY_TO_SHARED(1);
                COPY_TO_SHARED(2);
                COPY_TO_SHARED(3);
                __syncthreads();
                if (j < ptr[i + 1])
                    for (int ki = ko; ki < ko + 4 * blockDim.x && ki < d; ++ki)
                        sum += A_shared[ki - ko] * B[ki * n + nid[j]];
            }
            if (j < ptr[i + 1])
                y[eid[j]] = sum;
        }
    }
}


/*
 * CUDA Kernel of the backward function for Masked Matrix Multiplication. (argument: csr format)
 */
template <class idx_t, class data_t>
__global__ void maskedmm_csr_backward_kernel(idx_t* __restrict__ ptr_r, idx_t* __restrict__ eid_r, idx_t* __restrict__ nid_r, idx_t* __restrict__ ptr_c, idx_t* __restrict__ eid_c, idx_t* __restrict__ nid_c, data_t* __restrict__ A, data_t* __restrict__ B, data_t* __restrict__ dy, data_t* __restrict__ dA, data_t* __restrict__ dB, int d, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y;
    if (i < n && j < d) {
        data_t sum = 0;
        for (int k = ptr_r[i]; k < ptr_r[i + 1]; ++k)
            sum += dy[eid_r[k]] * B[nid_r[k] * d + j];
        dA[i * d + j] = sum;

        sum = 0;
        for (int k = ptr_c[i]; k < ptr_c[i + 1]; ++k)
            sum += dy[eid_c[k]] * A[nid_c[k] * d + j];
        dB[i * d + j] = sum;
    }
}

/*
 * CUDA Kernel of forward function for Sparse Softmax
 * y = softmax(x), grouped by node.
 * ptr, eid: csr format
 */
template <class idx_t, class data_t>
__global__ void sparse_softmax_forward_kernel(idx_t* __restrict__ ptr, idx_t* __restrict__ eid, data_t* __restrict__ x, data_t* __restrict__ y, int n) {
    data_t max_val = *x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        for (int k = ptr[i]; k < ptr[i + 1]; ++k)
            max_val = max(max_val, x[eid[k]]);

        data_t sum = 0;
        for (int k = ptr[i]; k < ptr[i + 1]; ++k) {
            data_t now = exp(x[eid[k]] - max_val);
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
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = threadIdx.y;
    if (i < n) {
        for (int kj = ptr[i] + j; kj < ptr[i + 1]; kj += blockDim.y) {
            data_t dsum = 0;
            for (int ki = ptr[i]; ki < ptr[i + 1]; ++ki) {
                dsum -= dy[eid[ki]] * y[eid[ki]] * y[eid[kj]];
                if (ki == kj) dsum += dy[eid[ki]] * y[eid[ki]];
            }
            dx[eid[kj]] = dsum;
        }
    }
}

at::Tensor maskedmm_cuda_forward(
    at::Tensor row,
    at::Tensor col,
    at::Tensor A,
    at::Tensor B) {
    // row, col: (e); A, B: (n, d); y: (e)
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

// __global__ void maskedmm_csr_forward_kernel(idx_t* __restrict__ ptr, idx_t* __restrict__ eid, idx_t* __restrict__ nid, data_t* __restrict__ A, data_t* __restrict__ B, data_t* __restrict__ y, int d, int n) {
at::Tensor maskedmm_csr_cuda_forward(
    at::Tensor ptr,
    at::Tensor eid,
    at::Tensor nid,
    at::Tensor A,
    at::Tensor B) {
    // ptr: (n + 1); eid, nid: (e); A, B: (n, d); y: (e)
    const auto e = eid.size(0);
    const auto n = A.size(0);
    const auto d = A.size(1);
    auto y = at::zeros({e}, A.options());

    const int threads = 32;
    const dim3 blocks(n);
    auto Bt = B.transpose(0, 1).contiguous();

    AT_DISPATCH_IDX_DATA_TYPES(ptr.type(), A.type(), "maskedmm_csr_cuda_forward", ([&] {
        maskedmm_csr_forward_kernel<idx_t, data_t><<<blocks, threads>>>(
            ptr.data<idx_t>(),
            eid.data<idx_t>(),
            nid.data<idx_t>(),
            A.data<data_t>(),
            Bt.data<data_t>(),
            y.data<data_t>(),
            d, n);
    }));
    return y;
}

std::vector<at::Tensor> maskedmm_cuda_backward(
    at::Tensor row,
    at::Tensor col,
    at::Tensor A,
    at::Tensor B,
    at::Tensor dy) {
    // row, col: (e); dy: (e); A, B: (n, d);
    const auto e = row.size(0);
    const auto n = A.size(0);
    const auto d = A.size(1);

    const int threads = 1024; 
    const dim3 blocks((d + threads - 1) / threads);

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

// __global__ void maskedmm_csr_backward_kernel(idx_t* __restrict__ ptr_r, idx_t* __restrict__ eid_r, idx_t* __restrict__ nid_r, idx_t* __restrict__ ptr_c, idx_t* __restrict__ eid_c, idx_t* __restrict__ nid_c, data_t* __restrict__ A, data_t* __restrict__ B, data_t* __restrict__ dy, data_t* __restrict__ dA, data_t* __restrict__ dB, int d, int n)
std::vector<at::Tensor> maskedmm_csr_cuda_backward(
    at::Tensor ptr_r,
    at::Tensor eid_r,
    at::Tensor nid_r,
    at::Tensor ptr_c,
    at::Tensor eid_c,
    at::Tensor nid_c,
    at::Tensor A,
    at::Tensor B,
    at::Tensor dy) {
    // ptr_r, ptr_c: (n + 1); eid_r, eid_c, nid_r, eid_c: (e); A, B: (n, d); dy: (e);
    const auto e = eid_r.size(0);
    const auto n = A.size(0);
    const auto d = A.size(1);
    auto y = at::zeros({e}, A.options());

    const int threads = 1024;
    const dim3 blocks((d + threads - 1) / threads, n);

    auto dA = at::zeros_like(A, A.options());
    auto dB = at::zeros_like(B, B.options());
    dy = dy.contiguous();

    AT_DISPATCH_IDX_DATA_TYPES(ptr_r.type(), A.type(), "maskedmm_csr_cuda_backward", ([&] {
        maskedmm_csr_backward_kernel<idx_t, data_t><<<blocks, threads>>>(
            ptr_r.data<idx_t>(),
            eid_r.data<idx_t>(),
            nid_r.data<idx_t>(),
            ptr_c.data<idx_t>(),
            eid_c.data<idx_t>(),
            nid_c.data<idx_t>(),
            A.data<data_t>(),
            B.data<data_t>(),
            dy.data<data_t>(),
            dA.data<data_t>(),
            dB.data<data_t>(),
            d, n);
    }));
    return {dA, dB};
}

at::Tensor sparse_softmax_cuda_forward(
    at::Tensor ptr,
    at::Tensor eid,
    at::Tensor x) {
    // ptr: (n + 1); eid: (e); x: (e);
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
    // ptr: (n + 1); eid: (e); y: (e); dy: (e);
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
