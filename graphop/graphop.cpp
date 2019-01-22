#include <torch/torch.h>
#include <vector>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor maskedmm_cuda_forward(
    const at::Tensor& row,
    const at::Tensor& col,
    const at::Tensor& A,
    const at::Tensor& B);

at::Tensor maskedmm_forward(
    const at::Tensor& row,
    const at::Tensor& col,
    const at::Tensor& A,
    const at::Tensor& B) {
    CHECK_INPUT(row);
    CHECK_INPUT(col);
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    return maskedmm_cuda_forward(row, col, A, B);
}

at::Tensor maskedmm_csr_cuda_forward(
    const at::Tensor& ptr,
    const at::Tensor& eid,
    const at::Tensor& nid,
    const at::Tensor& A,
    const at::Tensor& B);

at::Tensor maskedmm_csr_forward(
    const at::Tensor& ptr,
    const at::Tensor& eid,
    const at::Tensor& nid,
    const at::Tensor& A,
    const at::Tensor& B) {
    CHECK_INPUT(ptr);
    CHECK_INPUT(eid);
    CHECK_INPUT(nid);
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    return maskedmm_csr_cuda_forward(ptr, eid, nid, A, B);
}

at::Tensor sparse_softmax_cuda_forward(
    const at::Tensor& ptr,
    const at::Tensor& eid,
    const at::Tensor& x);

at::Tensor sparse_softmax_forward(
    const at::Tensor& ptr,
    const at::Tensor& eid,
    const at::Tensor& x) {
    CHECK_INPUT(ptr);
    CHECK_INPUT(eid);
    CHECK_INPUT(x);
    return sparse_softmax_cuda_forward(ptr, eid, x);
}

at::Tensor vector_spmm_cuda_forward(
    const at::Tensor& ptr,
    const at::Tensor& eid,
    const at::Tensor& nid,
    const at::Tensor& edata,
    const at::Tensor& x);

at::Tensor vector_spmm_forward(
    const at::Tensor& ptr,
    const at::Tensor& eid,
    const at::Tensor& nid,
    const at::Tensor& edata,
    const at::Tensor& x) {
    CHECK_INPUT(ptr);
    CHECK_INPUT(eid);
    CHECK_INPUT(nid);
    CHECK_INPUT(edata);
    CHECK_INPUT(x);
    return vector_spmm_cuda_forward(ptr, eid, nid, edata, x);
}

std::vector<at::Tensor> maskedmm_cuda_backward(
    const at::Tensor& row,
    const at::Tensor& col,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& dy);

std::vector<at::Tensor> maskedmm_backward(
    const at::Tensor& row,
    const at::Tensor& col,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& dy) {
    CHECK_INPUT(row);
    CHECK_INPUT(col);
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(dy);
    return maskedmm_cuda_backward(row, col, A, B, dy);
}

std::vector<at::Tensor> maskedmm_csr_cuda_backward(
    const at::Tensor& ptr_r,
    const at::Tensor& eid_r,
    const at::Tensor& nid_r,
    const at::Tensor& ptr_c,
    const at::Tensor& eid_c,
    const at::Tensor& nid_c,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& dy);

std::vector<at::Tensor> maskedmm_csr_backward(
    const at::Tensor& ptr_r,
    const at::Tensor& eid_r,
    const at::Tensor& nid_r,
    const at::Tensor& ptr_c,
    const at::Tensor& eid_c,
    const at::Tensor& nid_c,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& dy) {
    CHECK_INPUT(ptr_r);
    CHECK_INPUT(eid_r);
    CHECK_INPUT(nid_r);
    CHECK_INPUT(ptr_c);
    CHECK_INPUT(eid_c);
    CHECK_INPUT(nid_c);
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    return maskedmm_csr_cuda_backward(ptr_r, eid_r, nid_r, ptr_c, eid_c, nid_c, A, B, dy);
}

at::Tensor sparse_softmax_cuda_backward(
    const at::Tensor& ptr,
    const at::Tensor& eid,
    const at::Tensor& y,
    const at::Tensor& dy);

at::Tensor sparse_softmax_backward(
    const at::Tensor& ptr,
    const at::Tensor& eid,
    const at::Tensor& y,
    const at::Tensor& dy) {
    CHECK_INPUT(ptr);
    CHECK_INPUT(eid);
    CHECK_INPUT(y);
    CHECK_INPUT(dy);
    return sparse_softmax_cuda_backward(ptr, eid, y, dy);
}

std::vector<at::Tensor> vector_spmm_cuda_backward(
    const at::Tensor& ptr,
    const at::Tensor& eid,
    const at::Tensor& nid,
    const at::Tensor& ptr_t,
    const at::Tensor& eid_t,
    const at::Tensor& nid_t,
    const at::Tensor& edata,
    const at::Tensor& dy,
    const at::Tensor& x);

std::vector<at::Tensor> vector_spmm_backward(
    const at::Tensor& ptr,
    const at::Tensor& eid,
    const at::Tensor& nid, 
    const at::Tensor& ptr_t,
    const at::Tensor& eid_t,
    const at::Tensor& nid_t,
    const at::Tensor& edata,
    const at::Tensor& dy,
    const at::Tensor& x) {
    CHECK_INPUT(ptr);
    CHECK_INPUT(eid);
    CHECK_INPUT(nid);
    CHECK_INPUT(ptr_t);
    CHECK_INPUT(eid_t);
    CHECK_INPUT(nid_t);
    CHECK_INPUT(edata);
    CHECK_INPUT(dy);
    CHECK_INPUT(x);
    return vector_spmm_cuda_backward(ptr, eid, nid, ptr_t, eid_t, nid_t, edata, dy, x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("maskedmm_forward", &maskedmm_forward, "Masked Matrix Multiplication forward");
    m.def("maskedmm_csr_forward", &maskedmm_csr_forward, "Masked Matrix Multiplication forward(CSR Format)");
    m.def("maskedmm_backward", &maskedmm_backward, "Masked Matrix Multiplication backward");
    m.def("maskedmm_csr_backward", &maskedmm_csr_backward, "Masked Matrix Multiplication backward(CSR Format)");
    m.def("sparse_softmax_forward", &sparse_softmax_forward, "Sparse softmax forward");
    m.def("sparse_softmax_backward", &sparse_softmax_backward, "Sparse softmax backward");
    m.def("vector_spmm_forward", &vector_spmm_forward, "Vectorized SPMM forward");
    m.def("vector_spmm_backward", &vector_spmm_backward, "Vectorized SPMM backward");
}
