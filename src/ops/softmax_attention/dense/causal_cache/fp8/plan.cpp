#include "ops/softmax_attention/dense/causal_cache/fp8/plan.h"
#include "ops/softmax_attention/dense/causal_cache/fp8/operands.h"
#include "ops/softmax_attention/common/mxfp8_tiled_plan.h"
#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
constexpr int kGroupedPrefillMaxWidth = 80;
#if defined(NINFER_THOR)
constexpr int kTiledWaveCtas = 20;

int grouped_wave_ctas(int heads, int width, int batch, int visible_capacity) {
    // Short caches are dominated by partial traffic. Longer decode needs more
    // stripes as batch grows; wide batched verification benefits from the old
    // large partition budget even on Thor's 20 SMs.
    if (visible_capacity <= 512) return batch == 1 ? kCausalAttentionSmCount : 20;
    if (width == 1) {
        if (heads == 16) return batch <= 4 ? 20 : 40;
        return batch <= 2 ? 20 : batch <= 4 ? 40 : 80;
    }
    if (batch <= 2) return heads == 24 ? 40 : 20;
    return heads == 16 && batch <= 4 ? 40 : kCausalAttentionSmCount;
}
#else
constexpr int kTiledWaveCtas = kCausalAttentionSmCount;
int grouped_wave_ctas(int, int, int, int) { return kCausalAttentionSmCount; }
#endif
} // namespace

Fp8KvCausalPlan make_fp8_kv_causal_plan(int heads, int width, int batch,
                                        CausalAttentionExecutionEnvelope envelope) {
    if ((heads != 24 && heads != 16) || width < 1 || batch < 1 || batch > 8 ||
        (batch > 1 && width > 16) || envelope.min_visible_keys == 0 ||
        envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kCausalAttentionMaximumVisibleKeys)
        throw std::invalid_argument("FP8 attention: invalid plan inputs");
    constexpr int grouped_limit = Fp8KvCausalPlan::kTokenTile;
    const auto family           = width <= grouped_limit             ? Fp8KvFamily::Grouped
                                  : width <= kGroupedPrefillMaxWidth ? Fp8KvFamily::ParallelGrouped
                                                                     : Fp8KvFamily::Tiled;
    if (family == Fp8KvFamily::Tiled)
        return {family, heads,    width,
                batch,  envelope, mxfp8_tiled_partition(heads, width, envelope.max_visible_keys,
                                                         kTiledWaveCtas)};
    const int tiles =
        family == Fp8KvFamily::ParallelGrouped ? (width + grouped_limit - 1) / grouped_limit : 1;
    const int independent_tiles = batch * (heads == 24 ? 4 : 2) * tiles;
    // Decode permits two waves. Spec adds a wave when complete query tiles
    // would leave more than 10% of the selected grid budget unused.
    const int sms = grouped_wave_ctas(heads, width, batch, envelope.max_visible_keys);
    const int wave_ctas = (sms / independent_tiles) * independent_tiles;
    const int budget    = width == 1 || wave_ctas < sms * 9 / 10 ? 2 * sms : sms;
    CausalKvPartition partition{
        1, std::clamp(budget / independent_tiles, 1, CausalKvPartition::kMaxSplits)};
    // Bound partial traffic by keeping enough KV work in each split.
    partition.key_shift = (width == 1 ? 7 : 8) - (heads == 16 ? 1 : 0);
    partition.capacity  = partition.active(envelope.max_visible_keys);
    return {family, heads, width, batch, envelope, partition};
}

std::size_t fp8_kv_workspace_bytes(int heads, int batch, int min_width, int max_width,
                                   CausalAttentionExecutionEnvelope envelope) {
    std::size_t maximum = 0;
    for (int width = min_width; width <= std::min(max_width, kGroupedPrefillMaxWidth); ++width) {
        const auto plan = make_fp8_kv_causal_plan(heads, width, batch, envelope);
        if (plan.family == Fp8KvFamily::Tiled) continue;
        const int splits = plan.partition.capacity;
        WorkspaceLayoutBuilder layout;
        (void)allocate_causal_partials(layout, heads, width, splits, batch);
        maximum = std::max(maximum, layout.peak_bytes(1));
    }
    return std::max(maximum, mxfp8_tiled_workspace_bytes(
                                 heads, std::max(min_width, kGroupedPrefillMaxWidth + 1), max_width,
                                 envelope.max_visible_keys, kTiledWaveCtas));
}

} // namespace ninfer::ops::detail
