# Policy-based tensor kernels

FirstAgent now has a small compile-time kernel layer in
`include/firstagent/Kernel.hpp`.

`kernel::Matrix<Scalar, Backend, Layout>` owns a runtime-sized matrix while
its scalar type, backend policy, and storage layout are selected at compile
time. `RowMajor` and `ColumnMajor` are currently supported by the CPU policy.
`MatrixView` lets kernels operate on external buffers without copying them.

The existing autograd `nn::Tensor` remains the compatibility API for the
Transformer. Its matrix multiplication is routed through the generic CPU
kernel policy and supplies the existing runtime AVX2/FMA dot-product
specialization. This keeps current model code stable while allowing new
backends to specialize `BackendOperations<Backend, Scalar>`.

CMake exposes the implementation boundaries as:

- `firstagent-kernels-cpu`: header-only policy and view layer.
- `firstagent-kernels-vulkan`: the optional Vulkan runtime backend.
- `firstagent-nn`: autograd, Transformer, tokenizer, and model code linked to
  the selected backend targets.

Dimensions and model configuration remain runtime values. Only choices that
benefit from specialization—scalar type, layout, and backend—are template
parameters. This avoids producing a separate binary for every vocabulary or
context size.
