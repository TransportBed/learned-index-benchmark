/**
 * @file types.hpp
 * @brief Plain-old-data types shared across the serving core.
 *
 * All indices and strides are 64-bit element counts (never byte counts, never
 * NumPy types), so nothing here depends on Python or NumPy.
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace lisbench {

/// Signed 64-bit index/stride type used throughout the core (element counts).
using idx_t = std::int64_t;

/**
 * @brief Outcome of a core operation.
 *
 * The batch orchestrators never call into Python, so failures are reported as a
 * status the binding layer turns into an exception, rather than by setting a
 * Python error directly.
 */
struct Status {
    bool ok = true;        ///< Whether the operation succeeded.
    std::string message;   ///< Human-readable error detail when @ref ok is false.

    /// @return A successful status.
    static Status success() { return Status{true, {}}; }
    /// @param msg Error detail. @return A failed status carrying @p msg.
    static Status error(std::string msg) { return Status{false, std::move(msg)}; }
};

/**
 * @brief CSR-style inverted list, a non-owning view over caller-owned buffers.
 *
 * @ref offsets has @ref n_buckets + 1 entries; @ref ids holds the concatenated
 * member ids of every bucket.
 */
struct LookupView {
    const std::int64_t* offsets = nullptr;  ///< Bucket boundaries, length n_buckets+1.
    const std::int32_t* ids = nullptr;      ///< Concatenated member ids, length ids_len.
    idx_t n_buckets = 0;                     ///< Number of buckets.
    idx_t ids_len = 0;                       ///< Total number of member ids.
};

/// @brief Stage timings for the uncompressed (exact) batched search.
struct ExactTimings {
    double collect_s = 0.0;  ///< Candidate collection seconds.
    double rerank_s = 0.0;   ///< Exact rerank seconds.
};

/// @brief Similarity metric used for exact reranking.
enum class Metric : int {
    kInner = 0,  ///< Rank by inner product.
    kL2 = 1      ///< Rank by negative squared L2 distance (from precomputed norms).
};

}  // namespace lisbench
