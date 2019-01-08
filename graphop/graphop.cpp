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
    // TyDy: type check
    return maskedmm_cuda_forward(row, col, A, B);
}

at::Tensor sparse_softmax_cuda_forward(
    at::Tensor ptr,
    at::Tensor eid,
    at::Tensor x);

at::Tensor sparse_softmax_forward(
    at::Tensor ptr,
    at::Tensor eid,
    at::Tensor x) {
    // TyDy: type check
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
    // TyDy: type check
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
    // TyDy: check type
    return maskedmm_cuda_backward(row, col, A, B, dy);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("maskedmm_forward", &maskedmm_forward, "Masked Matrix Multiplication forward");
    m.def("maskedmm_backward", &maskedmm_backward, "Masked Matrix Multiplication backward");
    m.def("sparse_softmax_forward", &sparse_softmax_forward, "Sparse softmax forward");
    m.def("sparse_softmax_backward", &sparse_softmax_backward, "Sparse softmax backward");
}
