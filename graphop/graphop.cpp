#include <torch/torch.h>
#include <vector>

at::Tensor maskedmm_cuda_forward(
    at::Tensor row,
    at::Tensor col,
    at::Tensor A,
    at::Tensor B);

// To implement the function masked matrix multiplication.
at::Tensor maskedmm_forward(
    at::Tensor row,
    at::Tensor col,
    at::Tensor A,
    at::Tensor B) {
    // TODO: Check type
    return maskedmm_cuda_forward(row, col, A, B);
}

std::vector<at::Tensor> maskedmm_cuda_backward(
    at::Tensor row,
    at::Tensor col,
    at::Tensor A,
    at::Tensor B,
    at::Tensor dO);

std::vector<at::Tensor> maskedmm_backward(
    at::Tensor row,
    at::Tensor col,
    at::Tensor A,
    at::Tensor B,
    at::Tensor dO) {
    // TODO: check type
    return maskedmm_cuda_backward(row, col, A, B, dO);
}

/*
std::vector<at::Tensor> sparse_softmax_cuda_forward();
std::vector<at::Tensor> sparse_softmax_cuda_backward();
std::vector<at::Tensor> sparse_softmax_cuda_forward();
std::vector<at::Tensor> sparse_softmax_cuda_backward();
*/

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("maskedmm_forward", &maskedmm_forward, "Masked Matrix Multiplication forward");
    m.def("maskedmm_backward", &maskedmm_backward, "Masked Matrix Multiplication backward");
    // m.def("sparse_softmax_scatter_forward", &sparse_softmax_scatter_cuda_forward, "Sparse Softmax(scatter) forward");
    // m.def("sparse_softmax_scatter_backward", &sparse_softmax_scatter_cuda_backward, "Sparse Softmax(scatter) backward");
    // m.def("sparse_softmax_gather_forward", &sparse_softmax_gather_cuda_forward, "Sparse Softmax(gather) forward");
    // m.def("sparse_softmax_gather_backward", &sparse_softmax_gather_cuda_backward, "Sparse Softmax(gather) backward");
}
