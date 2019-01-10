#include <torch/torch.h>
#include <vector>

at::Tensor maskedmm_cuda_forward(
    at::Tensor row,
    at::Tensor col,
    at::Tensor A,
    at::Tensor B);

at::Tensor maskedmm_forward(
    at::Tensor row,
    at::Tensor col,
    at::Tensor A,
    at::Tensor B) {
    // TODO: type check
    return maskedmm_cuda_forward(row, col, A, B);
}

at::Tensor maskedmm_csr_cuda_forward(
    at::Tensor ptr,
    at::Tensor eid,
    at::Tensor nid,
    at::Tensor A,
    at::Tensor B);

at::Tensor maskedmm_csr_forward(
    at::Tensor ptr,
    at::Tensor eid,
    at::Tensor nid,
    at::Tensor A,
    at::Tensor B) {
    // TODO: type check
    return maskedmm_csr_cuda_forward(ptr, eid, nid, A, B);
}

std::vector<at::Tensor> maskedmm_csr_cuda_backward(
    at::Tensor ptr_r,
    at::Tensor eid_r,
    at::Tensor nid_r,
    at::Tensor ptr_c,
    at::Tensor eid_c,
    at::Tensor nid_c,
    at::Tensor A,
    at::Tensor B,
    at::Tensor dy);

std::vector<at::Tensor> maskedmm_csr_backward(
    at::Tensor ptr_r,
    at::Tensor eid_r,
    at::Tensor nid_r,
    at::Tensor ptr_c,
    at::Tensor eid_c,
    at::Tensor nid_c,
    at::Tensor A,
    at::Tensor B,
    at::Tensor dy) {
    // TODO: type check
    return maskedmm_csr_cuda_backward(ptr_r, eid_r, nid_r, ptr_c, eid_c, nid_c, A, B, dy);
}

at::Tensor sparse_softmax_cuda_forward(
    at::Tensor ptr,
    at::Tensor eid,
    at::Tensor x);

at::Tensor sparse_softmax_forward(
    at::Tensor ptr,
    at::Tensor eid,
    at::Tensor x) {
    // TODO: type check
    return sparse_softmax_cuda_forward(ptr, eid, x);
}

at::Tensor sparse_softmax_cuda_backward(
    at::Tensor ptr,
    at::Tensor eid,
    at::Tensor y,
    at::Tensor dy);

at::Tensor sparse_softmax_backward(
    at::Tensor ptr,
    at::Tensor eid,
    at::Tensor y,
    at::Tensor dy) {
    // TODO: type check
    return sparse_softmax_cuda_backward(ptr, eid, y, dy);
}

std::vector<at::Tensor> maskedmm_cuda_backward(
    at::Tensor row,
    at::Tensor col,
    at::Tensor A,
    at::Tensor B,
    at::Tensor dy);

std::vector<at::Tensor> maskedmm_backward(
    at::Tensor row,
    at::Tensor col,
    at::Tensor A,
    at::Tensor B,
    at::Tensor dy) {
    // TODO: check type
    return maskedmm_cuda_backward(row, col, A, B, dy);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("maskedmm_forward", &maskedmm_forward, "Masked Matrix Multiplication forward");
    m.def("maskedmm_csr_forward", &maskedmm_csr_forward, "Masked Matrix Multiplication forward(CSR Format)");
    m.def("maskedmm_backward", &maskedmm_backward, "Masked Matrix Multiplication backward");
    m.def("maskedmm_csr_backward", &maskedmm_csr_backward, "Masked Matrix Multiplication backward(CSR Format)");
    m.def("sparse_softmax_forward", &sparse_softmax_forward, "Sparse softmax forward");
    m.def("sparse_softmax_backward", &sparse_softmax_backward, "Sparse softmax backward");
}
