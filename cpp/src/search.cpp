#include "lisbench/search.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <limits>
#include <utility>
#include <vector>

#include "lisbench/candidates.hpp"
#include "lisbench/rerank.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace lisbench {

Status search_exact_batch(const ExactBatchInput& in, const ExactBatchOutput& out, ExactTimings& timings) {
    const idx_t batch_n = in.batch_n;
    const int k = in.k;
    const std::vector<LookupView>& lookup_views = *in.lookups;

    std::fill(out.retrieved, out.retrieved + batch_n * static_cast<idx_t>(k), static_cast<std::int32_t>(-1));
    std::fill(out.candidate_counts, out.candidate_counts + batch_n, static_cast<std::int32_t>(0));

    double collect_s = 0.0;
    double rerank_s = 0.0;
    std::atomic<bool> failed{false};
    char error_message[256] = {0};

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 8) reduction(+ : collect_s, rerank_s)
#endif
    for (idx_t qi = 0; qi < batch_n; ++qi) {
        if (failed.load(std::memory_order_relaxed)) {
            continue;
        }
        try {
            auto stage_start = std::chrono::steady_clock::now();
            std::vector<std::int32_t> candidates = collect_candidates_for_query(
                in.topm,
                qi,
                in.reps,
                in.m_cap,
                in.topm_stride0,
                in.topm_stride1,
                in.topm_stride2,
                lookup_views,
                std::max<int>(0, in.probe_counts[qi]),
                in.min_reps,
                in.buffer);
            collect_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start).count();
            out.candidate_counts[qi] = static_cast<std::int32_t>(candidates.size());
            if (candidates.empty()) {
                continue;
            }
            stage_start = std::chrono::steady_clock::now();
            std::vector<std::int32_t> top_ids = select_top_exact(
                candidates,
                in.queries,
                qi,
                in.query_stride0,
                in.query_stride1,
                in.train,
                in.train_rows,
                in.q_dim,
                in.train_stride0,
                in.train_stride1,
                in.train_sq_norms,
                in.metric,
                k);
            rerank_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start).count();
            for (std::size_t j = 0; j < top_ids.size(); ++j) {
                out.retrieved[qi * static_cast<idx_t>(k) + static_cast<idx_t>(j)] = top_ids[j];
            }
        } catch (const std::exception& exc) {
            if (!failed.exchange(true)) {
                std::snprintf(error_message, sizeof(error_message), "%s", exc.what());
            }
        }
    }

    if (failed.load()) {
        return Status::error(error_message[0] ? error_message : "native exact batch search failed");
    }
    timings.collect_s = collect_s;
    timings.rerank_s = rerank_s;
    return Status::success();
}

}  // namespace lisbench
