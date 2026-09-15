#pragma once

#include "agentari/Parallel.hpp"

#include <algorithm>
#include <cstddef>
#include <concepts>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace agentari::nn::kernel {

// These policy types are intentionally empty. They select implementation and
// layout at compile time while dimensions remain runtime values.
struct RowMajor {};
struct ColumnMajor {};
struct CpuBackend {};

template <typename Scalar>
concept Numeric = std::is_arithmetic_v<Scalar>;

template <Numeric Scalar>
class MatrixView {
public:
    using value_type = Scalar;

    MatrixView(Scalar* data, std::size_t rows, std::size_t columns,
               std::size_t row_stride, std::size_t column_stride = 1U)
        : data_(data), rows_(rows), columns_(columns),
          row_stride_(row_stride), column_stride_(column_stride) {
        if (data_ == nullptr && rows_ != 0U && columns_ != 0U) {
            throw std::invalid_argument("matrix view cannot have null storage");
        }
    }

    [[nodiscard]] Scalar& operator()(std::size_t row, std::size_t column) const {
        if (row >= rows_ || column >= columns_) {
            throw std::out_of_range("matrix view index is outside its shape");
        }
        return data_[row * row_stride_ + column * column_stride_];
    }

    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t columns() const noexcept { return columns_; }
    [[nodiscard]] Scalar* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t row_stride() const noexcept { return row_stride_; }
    [[nodiscard]] std::size_t column_stride() const noexcept { return column_stride_; }

private:
    Scalar* data_;
    std::size_t rows_;
    std::size_t columns_;
    std::size_t row_stride_;
    std::size_t column_stride_;
};

template <Numeric Scalar>
struct ScalarDot {
    [[nodiscard]] Scalar operator()(const MatrixView<const Scalar>& left,
                                    const MatrixView<const Scalar>& right,
                                    std::size_t left_row,
                                    std::size_t right_column) const {
        Scalar result{};
        for (std::size_t index = 0U; index < left.columns(); ++index) {
            result += left(left_row, index) * right(index, right_column);
        }
        return result;
    }
};

template <Numeric Scalar, typename DotProduct = ScalarDot<Scalar>>
void cpu_matmul(const MatrixView<const Scalar>& left,
                const MatrixView<const Scalar>& right,
                const MatrixView<Scalar>& output,
                DotProduct dot_product = {}) {
    if (left.columns() != right.rows() || output.rows() != left.rows() ||
        output.columns() != right.columns()) {
        throw std::invalid_argument("matrix multiplication dimensions do not match");
    }
    std::size_t operations = 0U;
    if (output.rows() != 0U && output.columns() <=
                                   std::numeric_limits<std::size_t>::max() /
                                       output.rows()) {
        operations = output.rows() * output.columns();
    }
    const std::size_t workers = parallel::worker_count_for(output.rows());
    parallel::Config parallel_config;
    parallel_config.grain_size = std::max<std::size_t>(
        1U, output.rows() / std::max<std::size_t>(1U, workers * 4U));
    parallel_config.minimum_parallel_work = operations >= 4096U ? 1U : 4096U;
    parallel::parallel_for(0U, output.rows(), [&](std::size_t row) {
        for (std::size_t column = 0U; column < output.columns(); ++column) {
            output(row, column) = dot_product(left, right, row, column);
        }
    }, parallel_config);
}

template <typename Backend, Numeric Scalar, typename DotProduct = ScalarDot<Scalar>>
struct BackendOperations;

template <Numeric Scalar, typename DotProduct>
struct BackendOperations<CpuBackend, Scalar, DotProduct> {
    static void matmul(const MatrixView<const Scalar>& left,
                       const MatrixView<const Scalar>& right,
                       const MatrixView<Scalar>& output,
                       DotProduct dot_product = {}) {
        cpu_matmul(left, right, output, dot_product);
    }
};

template <Numeric Scalar, typename Backend = CpuBackend, typename Layout = RowMajor>
class Matrix {
public:
    using scalar_type = Scalar;
    using backend_type = Backend;
    using layout_type = Layout;

    Matrix() = default;

    Matrix(std::size_t rows, std::size_t columns, Scalar value = Scalar{})
        : rows_(rows), columns_(columns), values_(checked_element_count(rows, columns), value) {}

    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t columns() const noexcept { return columns_; }
    [[nodiscard]] const std::vector<Scalar>& values() const noexcept { return values_; }
    [[nodiscard]] std::vector<Scalar>& values() noexcept { return values_; }

    [[nodiscard]] Scalar& operator()(std::size_t row, std::size_t column) {
        return values_[offset(row, column)];
    }

    [[nodiscard]] const Scalar& operator()(std::size_t row, std::size_t column) const {
        return values_[offset(row, column)];
    }

    template <typename RightLayout = RowMajor>
    [[nodiscard]] Matrix<Scalar, Backend, RowMajor> matmul(
        const Matrix<Scalar, Backend, RightLayout>& right) const {
        Matrix<Scalar, Backend, RowMajor> result(rows_, right.columns());
        const MatrixView<const Scalar> left_view(
            values_.data(), rows_, columns_,
            std::same_as<Layout, RowMajor> ? columns_ : 1U,
            std::same_as<Layout, RowMajor> ? 1U : rows_);
        const MatrixView<const Scalar> right_view(right.values().data(), right.rows(),
                                                   right.columns(),
                                                   std::same_as<RightLayout, RowMajor> ? right.columns() : 1U,
                                                   std::same_as<RightLayout, RowMajor> ? 1U : right.rows());
        const MatrixView<Scalar> output_view(result.values().data(), result.rows(),
                                              result.columns(), result.columns());
        BackendOperations<Backend, Scalar>::matmul(left_view, right_view, output_view);
        return result;
    }

private:
    [[nodiscard]] static std::size_t checked_element_count(std::size_t rows,
                                                            std::size_t columns) {
        if (rows != 0U && columns > std::numeric_limits<std::size_t>::max() / rows) {
            throw std::overflow_error("matrix shape is too large");
        }
        return rows * columns;
    }

    [[nodiscard]] std::size_t offset(std::size_t row, std::size_t column) const {
        if (row >= rows_ || column >= columns_) {
            throw std::out_of_range("matrix index is outside its shape");
        }
        if constexpr (std::same_as<Layout, RowMajor>) {
            return row * columns_ + column;
        } else {
            return column * rows_ + row;
        }
    }

    std::size_t rows_{0U};
    std::size_t columns_{0U};
    std::vector<Scalar> values_;
};

using F32CpuMatrix = Matrix<float, CpuBackend, RowMajor>;

}  // namespace agentari::nn::kernel
