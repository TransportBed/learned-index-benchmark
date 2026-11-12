#pragma once

#include <cstdint>
#include <vector>

#include "lisbench/types.hpp"

namespace lisbench {

/**
 * @brief Inputs to the uncompressed (exact) batched search.
 *
 * Pointers are non-owning; strides are element counts.
 */
struct ExactBatchInput {
    const float* queries = nullptr;  ///< Query matrix (batch_n, q_dim).
    idx_t batch_n = 0;               ///< Number of queries.
    idx_t q_dim = 0;                 ///< Query dimensionality.
    idx_t query_stride0 = 0;         ///< Query row stride in elements.
    idx_t query_stride1 = 0;         ///< Query column stride in elements.

    const std::int32_t* topm = nullptr;  ///< Probe ids (batch_n, reps, m_cap).
    idx_t reps = 0;                      ///< Repetitions.
    idx_t m_cap = 0;                     ///< Probe columns per repetition.
    idx_t topm_stride0 = 0;              ///< Batch stride in elements.
    idx_t topm_stride1 = 0;              ///< Repetition stride in elements.
    idx_t topm_stride2 = 0;              ///< Probe stride in elements.

    const std::int32_t* probe_counts = nullptr;  ///< Per-query probe budget (batch_n).

    const std::vector<LookupView>* lookups = nullptr;  ///< CSR lists, >= reps entries.

    const float* train = nullptr;  ///< Base vectors (train_rows, q_dim).
    idx_t train_rows = 0;          ///< Number of base vectors.
    idx_t train_stride0 = 0;       ///< Row stride in elements.
    idx_t train_stride1 = 0;       ///< Column stride in elements.
    const float* train_sq_norms = nullptr;  ///< Squared norms; required for @ref Metric::kL2.

    Metric metric = Metric::kInner;  ///< Rerank metric.
    int k = 10;                      ///< Results per query.
    int min_reps = 1;                ///< Candidate min-repetition threshold.
    int buffer = 0;                  ///< Candidate truncation cap.
};

/// @brief Caller-owned output buffers for @ref search_exact_batch.
struct ExactBatchOutput {
    std::int32_t* retrieved = nullptr;         ///< Top-k ids per query (batch_n, k); -1 padded.
    std::int32_t* candidate_counts = nullptr;  ///< Probed-union size per query.
};

/**
 * @brief Collect the probed union per query and exact-rank it.
 *
 * @param in       Non-owning inputs.
 * @param out      Caller-owned, pre-allocated output buffers (filled here).
 * @param[out] timings Summed per-stage wall-clock diagnostics.
 * @return @ref Status::success on success, otherwise a failed status whose
 *         message describes the first per-query error encountered.
 */
Status search_exact_batch(const ExactBatchInput& in, const ExactBatchOutput& out, ExactTimings& timings);

}  // namespace lisbench
