#include "lisbench/rerank.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace lisbench {

std::vector<std::int32_t> select_top_exact(
        const std::vector<std::int32_t>& ids,
        const float* query_data,
        idx_t query_row,
        idx_t query_stride0,
        idx_t query_stride1,
        const float* train_data,
        idx_t train_rows,
        idx_t dim,
        idx_t train_stride0,
        idx_t train_stride1,
        const float* train_sq_norms,
        Metric metric,
        int k) {
    std::vector<std::pair<float, std::int32_t>> scored;
    scored.reserve(ids.size());
    float q_norm = 0.0f;
    if (metric == Metric::kL2) {
        for (idx_t d = 0; d < dim; ++d) {
            const float qv = query_data[query_row * query_stride0 + d * query_stride1];
            q_norm += qv * qv;
        }
    }
    for (const std::int32_t id : ids) {
        if (id < 0 || id >= train_rows) {
            throw std::runtime_error("rerank id is out of range for train vectors");
        }
        float dot = 0.0f;
        for (idx_t d = 0; d < dim; ++d) {
            dot += query_data[query_row * query_stride0 + d * query_stride1] * train_data[id * train_stride0 + d * train_stride1];
        }
        const float score = metric == Metric::kL2 ? -(train_sq_norms[id] + q_norm - 2.0f * dot) : dot;
        scored.emplace_back(score, id);
    }
    const int top_n = std::min<int>(std::max(k, 0), static_cast<int>(scored.size()));
    auto cmp = [](const auto& left, const auto& right) {
        if (left.first != right.first) {
            return left.first > right.first;
        }
        return left.second < right.second;
    };
    if (top_n < static_cast<int>(scored.size())) {
        std::nth_element(scored.begin(), scored.begin() + top_n, scored.end(), cmp);
        scored.resize(static_cast<std::size_t>(top_n));
    }
    std::sort(scored.begin(), scored.end(), cmp);
    std::vector<std::int32_t> out;
    out.reserve(scored.size());
    for (const auto& item : scored) {
        out.push_back(item.second);
    }
    return out;
}

}  // namespace lisbench
