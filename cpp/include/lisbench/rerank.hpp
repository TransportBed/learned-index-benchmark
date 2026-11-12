#pragma once

#include <cstdint>
#include <vector>

#include "lisbench/types.hpp"

namespace lisbench {

/**
 * @brief Exact-rank @p ids against the query and return the top-@p k.
 *
 * Results are ordered by (score desc, id asc). @ref Metric::kInner ranks by dot
 * product; @ref Metric::kL2 by negative squared L2 distance reconstructed from
 * precomputed train norms.
 *
 * @param ids            Candidate ids to rank.
 * @param query_data     Query matrix base pointer.
 * @param query_row      Query row index.
 * @param query_stride0  Query row stride in elements.
 * @param query_stride1  Query column stride in elements.
 * @param train_data     Base vector matrix pointer.
 * @param train_rows     Number of base vectors (bounds check).
 * @param dim            Vector dimensionality.
 * @param train_stride0  Row stride of @p train_data in elements.
 * @param train_stride1  Column stride of @p train_data in elements.
 * @param train_sq_norms Precomputed squared norms; required for @ref Metric::kL2.
 * @param metric         Similarity metric.
 * @param k              Number of results to return.
 * @return Top-@p k ids in ranked order.
 * @throws std::runtime_error if an id is out of range for the base vectors.
 */
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
        int k);

}  // namespace lisbench
