#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "lisbench/candidates.hpp"
#include "lisbench/rerank.hpp"
#include "lisbench/search.hpp"

namespace py = pybind11;
using lisbench::idx_t;
using lisbench::LookupView;

namespace {

/**
 * @brief A C-contiguous, dtype-coerced NumPy array of element type @p T.
 *
 * Equivalent to the old @c PyArray_FROM_OTF with @c NPY_ARRAY_IN_ARRAY plus a
 * target typenum: @c forcecast casts to @p T and @c c_style guarantees a
 * row-major contiguous layout.
 */
template <typename T>
using CArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

/**
 * @brief Stride of @p arr along @p dim expressed in elements (not bytes).
 * @tparam T Element type.
 * @param arr Coerced array.
 * @param dim Dimension index.
 * @return The element stride, as the core's pointer+stride views expect.
 */
template <typename T>
idx_t elem_stride(const CArray<T>& arr, int dim) {
    return static_cast<idx_t>(arr.strides(dim)) / static_cast<idx_t>(arr.itemsize());
}

/// @brief Map the Python @c metric_kind flag (1 = L2) to the core enum.
lisbench::Metric to_metric(int metric_kind) {
    return metric_kind == 1 ? lisbench::Metric::kL2 : lisbench::Metric::kInner;
}

/**
 * @brief Coerce the Python lookups list into non-owning core views.
 *
 * Each list entry must be an @c (offsets int64, ids int32) tuple. The coerced
 * arrays are stored in @p owned_offsets / @p owned_ids so they outlive the
 * views (which only hold pointers into them) for the duration of the call.
 *
 * @param lookups            The Python list of @c (offsets, ids) tuples.
 * @param[out] owned_offsets Keeps the coerced offset arrays alive.
 * @param[out] owned_ids     Keeps the coerced id arrays alive.
 * @param[out] views         The resulting @ref lisbench::LookupView list.
 * @throws pybind11::type_error  if an entry is not a 2-tuple.
 * @throws pybind11::value_error if offsets or ids are not rank-1.
 */
void parse_lookups(
        const py::list& lookups,
        std::vector<CArray<std::int64_t>>& owned_offsets,
        std::vector<CArray<std::int32_t>>& owned_ids,
        std::vector<LookupView>& views) {
    const std::size_t reps = lookups.size();
    owned_offsets.reserve(reps);
    owned_ids.reserve(reps);
    views.reserve(reps);
    for (std::size_t rep = 0; rep < reps; ++rep) {
        py::object item = lookups[rep];
        if (!py::isinstance<py::tuple>(item) || py::len(item) != 2) {
            throw py::type_error("each lookup must be an (offsets, ids) tuple");
        }
        py::tuple pair = item.cast<py::tuple>();
        owned_offsets.emplace_back(pair[0].cast<CArray<std::int64_t>>());
        owned_ids.emplace_back(pair[1].cast<CArray<std::int32_t>>());
        CArray<std::int64_t>& offsets = owned_offsets.back();
        CArray<std::int32_t>& ids = owned_ids.back();
        if (offsets.ndim() != 1 || ids.ndim() != 1) {
            throw py::value_error("lookup offsets and ids must be rank-1 arrays");
        }
        LookupView view;
        view.offsets = offsets.data();
        view.ids = ids.data();
        view.n_buckets = static_cast<idx_t>(offsets.shape(0)) - 1;
        view.ids_len = static_cast<idx_t>(ids.shape(0));
        views.push_back(view);
    }
}

/**
 * @brief Binding for @c search_exact_batch_cpp: the uncompressed serving path
 *        (collect probed union, exact-rank) over a query batch.
 *
 * Coerces inputs, packs @ref lisbench::ExactBatchInput, allocates the two
 * output arrays, then runs @ref lisbench::search_exact_batch with the GIL
 * released. The 10 positional arguments mirror the old extension exactly.
 *
 * @return A 3-tuple @c (retrieved, candidate_counts, timing_dict) matching the
 *         legacy return.
 * @throws pybind11::value_error / pybind11::type_error on malformed inputs.
 * @throws std::runtime_error (Python @c RuntimeError) if the core search fails.
 */
py::object search_exact_batch_cpp(
        py::object queries_obj,
        py::object topm_obj,
        py::object probe_counts_obj,
        py::object lookups_obj,
        py::object train_obj,
        py::object train_sq_norms_obj,
        int metric_kind,
        int k,
        int min_reps,
        int buffer) {
    CArray<float> queries = queries_obj.cast<CArray<float>>();
    CArray<std::int32_t> topm = topm_obj.cast<CArray<std::int32_t>>();
    CArray<std::int32_t> probe_counts = probe_counts_obj.cast<CArray<std::int32_t>>();
    CArray<float> train = train_obj.cast<CArray<float>>();

    const bool l2 = metric_kind == 1;
    CArray<float> train_sq_norms;
    if (l2) {
        train_sq_norms = train_sq_norms_obj.cast<CArray<float>>();
    }

    if (queries.ndim() != 2 || topm.ndim() != 3 || probe_counts.ndim() != 1 || train.ndim() != 2) {
        throw py::value_error("invalid native exact batch search array ranks");
    }
    const idx_t batch_n = static_cast<idx_t>(queries.shape(0));
    const idx_t q_dim = static_cast<idx_t>(queries.shape(1));
    const idx_t reps = static_cast<idx_t>(topm.shape(1));
    const idx_t m_cap = static_cast<idx_t>(topm.shape(2));
    if (static_cast<idx_t>(topm.shape(0)) != batch_n || static_cast<idx_t>(probe_counts.shape(0)) != batch_n) {
        throw py::value_error("queries, topm, and probe_counts must share batch length");
    }
    if (static_cast<idx_t>(train.shape(1)) != q_dim) {
        throw py::value_error("train and query dimensionality must match");
    }
    if (l2 && (train_sq_norms.ndim() != 1 || train_sq_norms.shape(0) != train.shape(0))) {
        throw py::value_error("train_sq_norms must match train rows for L2 search");
    }

    if (!py::isinstance<py::list>(lookups_obj)) {
        throw py::type_error("lookups must be a list of (offsets, ids) tuples");
    }
    std::vector<CArray<std::int64_t>> owned_offsets;
    std::vector<CArray<std::int32_t>> owned_ids;
    std::vector<LookupView> views;
    parse_lookups(lookups_obj.cast<py::list>(), owned_offsets, owned_ids, views);
    if (static_cast<idx_t>(views.size()) < reps) {
        throw py::value_error("lookups length must cover every repetition");
    }

    py::array_t<std::int32_t> retrieved({batch_n, static_cast<idx_t>(k)});
    py::array_t<std::int32_t> candidate_counts(batch_n);

    lisbench::ExactBatchInput in;
    in.queries = queries.data();
    in.batch_n = batch_n;
    in.q_dim = q_dim;
    in.query_stride0 = elem_stride(queries, 0);
    in.query_stride1 = elem_stride(queries, 1);
    in.topm = topm.data();
    in.reps = reps;
    in.m_cap = m_cap;
    in.topm_stride0 = elem_stride(topm, 0);
    in.topm_stride1 = elem_stride(topm, 1);
    in.topm_stride2 = elem_stride(topm, 2);
    in.probe_counts = probe_counts.data();
    in.lookups = &views;
    in.train = train.data();
    in.train_rows = static_cast<idx_t>(train.shape(0));
    in.train_stride0 = elem_stride(train, 0);
    in.train_stride1 = elem_stride(train, 1);
    in.train_sq_norms = l2 ? train_sq_norms.data() : nullptr;
    in.metric = to_metric(metric_kind);
    in.k = k;
    in.min_reps = min_reps;
    in.buffer = buffer;

    lisbench::ExactBatchOutput out;
    out.retrieved = retrieved.mutable_data();
    out.candidate_counts = candidate_counts.mutable_data();

    lisbench::ExactTimings timings;
    lisbench::Status status;
    {
        py::gil_scoped_release release;
        status = lisbench::search_exact_batch(in, out, timings);
    }
    if (!status.ok) {
        throw std::runtime_error(status.message.empty() ? "native exact batch search failed" : status.message);
    }

    py::dict timing;
    timing["collect_s"] = timings.collect_s;
    timing["rerank_s"] = timings.rerank_s;
    return py::make_tuple(retrieved, candidate_counts, timing);
}

}  // namespace

PYBIND11_MODULE(lisbench_native, mod) {
    mod.def("search_exact_batch_cpp", &search_exact_batch_cpp,
            py::arg("queries"), py::arg("topm"), py::arg("probe_counts"), py::arg("lookups"),
            py::arg("train"), py::arg("train_sq_norms"), py::arg("metric_kind"), py::arg("k"),
            py::arg("min_reps"), py::arg("buffer"),
            "Collect probed unions and exact-rank them for a query batch.");
}
