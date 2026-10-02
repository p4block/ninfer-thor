#include "ops/linear/nvfp4/nvfp4_thor.cuh"
#include <cublasLt.h>
#include <map>
#include <memory>
#include <tuple>
#include <string>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
void check(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": cuBLASLt status " + std::to_string(status));
}
constexpr std::size_t kWorkspaceBytes = 32u << 20;

// NInfer's persistent weight scale layout is already cuBLASLt's M128/K4
// layout. Only transient activation scales need this byte permutation.
__global__ void swizzle_scales(Nvfp4A4Operands p, int padded_tokens) {
    const int groups = p.k / 16;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= padded_tokens * groups) return;
    const int token = index / groups, group = index % groups;
    std::uint8_t scale = 0;
    if (token < p.tokens) {
        std::int64_t source = static_cast<std::int64_t>(token) * groups + group;
        if (p.scale_layout != Nvfp4ScaleLayout::RowMajor) {
            const int tile = p.scale_layout == Nvfp4ScaleLayout::Tiled128 ? 128 : 256;
            source = (static_cast<std::int64_t>(token / tile) * (groups / 16) + group / 16) * tile * 16
                     + (token % tile) * 16 + group % 16;
        }
        scale = p.x_scales[source];
    }
    const std::int64_t dest =
        (static_cast<std::int64_t>(token / 128) * (groups / 4) + group / 4) * 512
        + (token & 31) * 16 + ((token & 127) >> 5) * 4 + (group & 3);
    p.blas_scales[dest] = scale;
}

struct Plan {
    cublasLtHandle_t handle{};
    cublasLtMatmulDesc_t desc{};
    cublasLtMatrixLayout_t a{}, b{}, d{};
    cublasLtMatmulAlgo_t algorithm{};
    Plan(int rows, int tokens, int k, const void* sa, const void* sb) {
        check(cublasLtCreate(&handle), "create Thor GEMM handle");
        check(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F), "create Thor GEMM descriptor");
        const auto trans = CUBLAS_OP_T;
        check(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)), "set transpose");
        const auto mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        check(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode)), "set weight scales");
        check(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode)), "set activation scales");
        check(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sa, sizeof(sa)), "set weight scale pointer");
        check(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sb, sizeof(sb)), "set activation scale pointer");
        check(cublasLtMatrixLayoutCreate(&a, CUDA_R_4F_E2M1, k, rows, k), "create weight layout");
        check(cublasLtMatrixLayoutCreate(&b, CUDA_R_4F_E2M1, k, tokens, k), "create activation layout");
        check(cublasLtMatrixLayoutCreate(&d, CUDA_R_32F, rows, tokens, rows), "create accumulator layout");
        cublasLtMatmulPreference_t preference{};
        check(cublasLtMatmulPreferenceCreate(&preference), "create GEMM preference");
        check(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                  &kWorkspaceBytes, sizeof(kWorkspaceBytes)), "set workspace size");
        cublasLtMatmulHeuristicResult_t result{};
        int count = 0;
        const auto status = cublasLtMatmulAlgoGetHeuristic(handle, desc, a, b, d, d, preference, 1, &result, &count);
        cublasLtMatmulPreferenceDestroy(preference);
        check(status, "select native Thor NVFP4 algorithm");
        if (count == 0) throw std::runtime_error("no native Thor NVFP4 algorithm for selected shape");
        algorithm = result.algo;
    }
    ~Plan() {
        if (d) cublasLtMatrixLayoutDestroy(d);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (desc) cublasLtMatmulDescDestroy(desc);
        if (handle) cublasLtDestroy(handle);
    }
};
} // namespace

void nvfp4_thor_gemm(const Nvfp4A4Operands& p, cudaStream_t stream) {
    if (!p.accumulators || !p.blas_scales || !p.blas_workspace || p.rows > 34816)
        throw std::invalid_argument("Thor NVFP4 requires caller-owned GEMM workspace and registered rows");
    const int padded = ((p.tokens + 31) / 32) * 32;
    const int scale_tokens = ((p.tokens + 127) / 128) * 128;
    swizzle_scales<<<(scale_tokens * (p.k / 16) + 255) / 256, 256, 0, stream>>>(p, scale_tokens);
    CUDA_CHECK(cudaGetLastError());
    if (padded != p.tokens)
        CUDA_CHECK(cudaMemsetAsync(const_cast<std::uint8_t*>(p.x) + static_cast<std::size_t>(p.tokens) * p.k / 2,
                                   0, static_cast<std::size_t>(padded - p.tokens) * p.k / 2, stream));
    // CPU descriptors are cached per execution thread and shape. No device
    // allocation occurs here; warmup creates plans before CUDA graph capture.
    thread_local std::map<std::tuple<int, int, int>, std::unique_ptr<Plan>> plans;
    auto& slot = plans[{p.rows, padded, p.k}];
    if (!slot) slot = std::make_unique<Plan>(p.rows, padded, p.k, p.scales, p.blas_scales);
    auto& plan = *slot;
    const void* sa = p.scales;
    const void* sb = p.blas_scales;
    check(cublasLtMatmulDescSetAttribute(plan.desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sa, sizeof(sa)), "bind weight scales");
    check(cublasLtMatmulDescSetAttribute(plan.desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sb, sizeof(sb)), "bind activation scales");
    const float alpha = 1.0f, beta = 0.0f;
    check(cublasLtMatmul(plan.handle, plan.desc, &alpha, p.codes, plan.a, p.x, plan.b, &beta,
                          p.accumulators, plan.d, p.accumulators, plan.d, &plan.algorithm,
                          p.blas_workspace, kWorkspaceBytes, stream), "execute native Thor NVFP4 GEMM");
}
} // namespace ninfer::ops::detail
