#pragma once

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_operands.h"

namespace ninfer::ops::detail {
// FP4 inputs and E4M3 block scales stay packed. cuBLASLt executes SM110's
// tcgen05 contraction into caller-owned FP32 storage; the original Op owns
// its final cast, split destinations, residual, or SwiGLU operation.
void nvfp4_thor_gemm(const Nvfp4A4Operands& p, cudaStream_t stream);

template <class Output, class Epilogue>
__global__ void nvfp4_thor_finish(const float* accum, int rows, int tokens, float alpha,
                                  Output output, Epilogue epilogue) {
    constexpr bool paired = requires { epilogue.apply_pair(0, 0, 0.0f, 0.0f); };
    const int output_rows = paired ? rows / 2 : rows;
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<std::int64_t>(output_rows) * tokens) return;
    const int token = index / output_rows, row = index % output_rows;
    const float value = accum[static_cast<std::int64_t>(token) * rows + row] * alpha;
    if constexpr (paired) {
        const float up = accum[static_cast<std::int64_t>(token) * rows + row + output_rows] * alpha;
        output.store(row, token, epilogue.apply_pair(row, token, value, up));
    } else {
        output.store(row, token, epilogue.apply(row, token, value));
    }
}

template <class Output, class Epilogue>
void launch_nvfp4_thor(const Nvfp4A4Operands& p, Output output, Epilogue epilogue,
                        cudaStream_t stream) {
    nvfp4_thor_gemm(p, stream);
    constexpr bool paired = requires { epilogue.apply_pair(0, 0, 0.0f, 0.0f); };
    const std::int64_t count = static_cast<std::int64_t>(paired ? p.rows / 2 : p.rows) * p.tokens;
    nvfp4_thor_finish<<<(count + 255) / 256, 256, 0, stream>>>(
        p.accumulators, p.rows, p.tokens, p.alpha, output, epilogue);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail
