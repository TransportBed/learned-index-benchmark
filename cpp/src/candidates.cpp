#include "lisbench/candidates.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace lisbench {

std::vector<std::int32_t> dedup_candidates(std::vector<std::int32_t>& gathered, int min_reps, int buffer) {
    std::vector<std::int32_t> out;
    if (gathered.empty() || buffer <= 0) {
        return out;
    }
    std::sort(gathered.begin(), gathered.end());
    std::vector<std::pair<std::int32_t, std::int32_t>> kept;
    kept.reserve(gathered.size());
    std::size_t i = 0;
    while (i < gathered.size()) {
        std::size_t j = i + 1;
        while (j < gathered.size() && gathered[j] == gathered[i]) {
            ++j;
        }
        const std::int32_t freq = static_cast<std::int32_t>(j - i);
        if (freq >= min_reps) {
            kept.emplace_back(gathered[i], freq);
        }
        i = j;
    }
    if (kept.empty()) {
        return out;
    }
    auto cmp = [](const auto& left, const auto& right) {
        if (left.second != right.second) {
            return left.second > right.second;
        }
        return left.first < right.first;
    };
    const std::size_t limit = static_cast<std::size_t>(buffer);
    if (kept.size() > limit) {
        std::nth_element(kept.begin(), kept.begin() + static_cast<std::ptrdiff_t>(limit), kept.end(), cmp);
        kept.resize(limit);
    }
    std::sort(kept.begin(), kept.end(), cmp);
    out.reserve(kept.size());
    for (const auto& item : kept) {
        out.push_back(item.first);
    }
    return out;
}

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
        int buffer) {
    std::vector<std::int32_t> gathered;
    for (idx_t rep = 0; rep < reps; ++rep) {
        if (rep >= static_cast<idx_t>(lookups.size())) {
            break;
        }
        const LookupView& lookup = lookups[static_cast<std::size_t>(rep)];
        const int limit = std::min<int>(probe_count, static_cast<int>(m_cap));
        for (int j = 0; j < limit; ++j) {
            const std::int32_t bucket = topm_data[
                query_idx * topm_stride0 + rep * topm_stride1 + static_cast<idx_t>(j) * topm_stride2
            ];
            if (bucket < 0 || bucket >= lookup.n_buckets) {
                continue;
            }
            const std::int64_t start = lookup.offsets[bucket];
            const std::int64_t end = lookup.offsets[bucket + 1];
            if (start < 0 || end < start || end > lookup.ids_len) {
                throw std::runtime_error("lookup offsets are inconsistent with ids length");
            }
            gathered.insert(gathered.end(), lookup.ids + start, lookup.ids + end);
        }
    }
    return dedup_candidates(gathered, min_reps, buffer);
}

}  // namespace lisbench
