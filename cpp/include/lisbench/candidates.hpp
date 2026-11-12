/**
 * @file candidates.hpp
 * @brief Candidate collection and deduplication for IVF-style probing.
 */
#pragma once

#include <cstdint>
#include <vector>

#include "lisbench/types.hpp"

namespace lisbench {

/**
 * @brief Deduplicate a gathered candidate id stream under the shared candidate
 *        contract.
 *
 * Ids surviving @p min_reps repetitions are ordered by (frequency desc, id asc)
 * and truncated to @p buffer. The output ordering is part of the serving
 * contract (deterministic truncation), identical with and without truncation.
 *
 * @param[in,out] gathered Raw id stream; consumed (sorted in place).
 * @param min_reps Minimum occurrence count for an id to be kept.
 * @param buffer   Maximum number of survivors; <= 0 yields an empty result.
 * @return Deduplicated, ordered, truncated ids.
 */
std::vector<std::int32_t> dedup_candidates(std::vector<std::int32_t>& gathered, int min_reps, int buffer);

/**
 * @brief Per-query candidate gather for the batched search paths.
 *
 * Only the first min(@p probe_count, @p m_cap) probes per repetition are
 * followed.
 *
 * @param topm_data     Rank-3 (batch, reps, m_cap) int32 buffer.
 * @param query_idx     Query row to gather for.
 * @param reps          Number of repetitions.
 * @param m_cap         Probe columns per repetition.
 * @param topm_stride0  Batch stride of @p topm_data in elements.
 * @param topm_stride1  Repetition stride of @p topm_data in elements.
 * @param topm_stride2  Probe stride of @p topm_data in elements.
 * @param lookups       CSR inverted lists, one per repetition.
 * @param probe_count   Per-query probe budget.
 * @param min_reps      Minimum occurrence count for an id to be kept.
 * @param buffer        Maximum number of survivors.
 * @return Deduplicated candidate ids for the query.
 * @throws std::runtime_error if a lookup's offsets are inconsistent with its ids.
 */
std::vector<std::int32_t> collect_candidates_for_query(
        const std::int32_t* topm_data,
        idx_t query_idx,
        idx_t reps,
        idx_t m_cap,
        idx_t topm_stride0,
        idx_t topm_stride1,
        idx_t topm_stride2,
        const std::vector<LookupView>& lookups,
        int probe_count,
        int min_reps,
        int buffer);

}  // namespace lisbench
