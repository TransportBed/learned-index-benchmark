from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json


@dataclass(frozen=True)
class MethodSpec:
    """Canonical experiment identity and paper-facing metadata for one method.

    ``method_id`` is the stable final-campaign identity. Obsolete identities
    are rejected rather than translated into a different method.
    ``display_name`` is the only name figures and tables should render.
    """

    method_id: str
    display_name: str
    family: str
    serving_mode: str
    order: int
    color: str
    marker: str
    roles: tuple[str, ...]
    learned: bool = False
    compressed: bool = False
    exact_rerank: bool = False
    default_seed_count: int = 1
    requires_full_vectors: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_SPECS = (
    MethodSpec("HNSW", "HNSW", "graph", "exact-graph", 0, "#1b9e77", "o", ("competitiveness", "main")),
    MethodSpec("IVF", "IVF", "classical-ivf", "exact-scan", 1, "#d95f02", "s", ("competitiveness", "main")),
    MethodSpec(
        "IVFPQ+R",
        "IVFPQ+R",
        "classical-ivf",
        "pq-rerank",
        3,
        "#e6ab02",
        "X",
        ("competitiveness", "main"),
        compressed=True,
        exact_rerank=True,
    ),
    MethodSpec(
        "BLISS",
        "BLISS",
        "learned-partition",
        "exact-scan",
        4,
        "#e7298a",
        "D",
        ("initialization", "competitiveness", "main"),
        learned=True,
    ),
    MethodSpec(
        "BLISS-KMeans",
        "BLISS-KMeans",
        "learned-partition",
        "exact-scan",
        5,
        "#c51b7d",
        "d",
        ("initialization",),
        learned=True,
    ),
    MethodSpec(
        "LIRA",
        "LIRA",
        "learned-local-index",
        "local-index",
        7,
        "#66a61e",
        "P",
        ("competitiveness", "main"),
        learned=True,
    ),
)


METHODS: dict[str, MethodSpec] = {spec.method_id: spec for spec in _SPECS}

_ALIASES = {
    "BLISS_KMEANS": "BLISS-KMeans",
    "IVFPQ_R": "IVFPQ+R",
}


def canonical_method_id(value: str) -> str:
    token = value.strip()
    if token in METHODS:
        return token
    folded = token.upper()
    for method_id in METHODS:
        if method_id.upper() == folded:
            return method_id
    if folded in _ALIASES:
        return _ALIASES[folded]
    raise ValueError(f"Unknown method: {value}")


def method_ids(role: str) -> tuple[str, ...]:
    return tuple(
        spec.method_id
        for spec in sorted(_SPECS, key=lambda item: item.order)
        if role in spec.roles
    )


def method_sort_key(value: str) -> tuple[int, str]:
    try:
        method_id = canonical_method_id(value)
    except ValueError:
        return 10_000, value
    return METHODS[method_id].order, method_id


def decorate_method_row(row: dict[str, object], requested_method: str | None = None) -> dict[str, object]:
    """Attach stable identity fields to an emitted result row."""

    method_id = canonical_method_id(str(row.get("method") or requested_method or ""))
    spec = METHODS[method_id]
    out = dict(row)
    out["method"] = method_id  # backward-compatible canonical column
    out["method_id"] = method_id
    out["method_display"] = spec.display_name
    out["method_family"] = spec.family
    out["serving_mode"] = spec.serving_mode
    emitted_requires_vectors = out.get("requires_full_vectors")
    if emitted_requires_vectors is not None and bool(emitted_requires_vectors) != spec.requires_full_vectors:
        raise ValueError(
            f"{method_id} emitted requires_full_vectors={emitted_requires_vectors!r}; "
            f"registry requires {spec.requires_full_vectors!r}"
        )
    out["requires_full_vectors"] = spec.requires_full_vectors
    if requested_method is not None:
        out["requested_method"] = requested_method
    return out


INITIALIZATION_METHODS = method_ids("initialization")
COMPETITIVENESS_METHODS = method_ids("competitiveness")
MAIN_METHODS = method_ids("main")
CLASSICAL_METHODS = tuple(
    method_id for method_id in COMPETITIVENESS_METHODS if not METHODS[method_id].learned
)
LEARNED_BASELINES = tuple(
    method_id
    for method_id in COMPETITIVENESS_METHODS
    if METHODS[method_id].learned
)
MULTISEED_LEARNED_CONTROLS = tuple(
    method_id for method_id in LEARNED_BASELINES if method_id == "BLISS"
)
EXPENSIVE_LEARNED_CONTROLS = tuple(
    method_id for method_id in LEARNED_BASELINES if method_id == "LIRA"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the canonical LisBench method registry.")
    parser.add_argument(
        "--group",
        choices=(
            "initialization",
            "competitiveness",
            "main",
            "classical",
            "learned",
            "learned-multiseed",
            "learned-expensive",
        ),
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    groups = {
        "initialization": INITIALIZATION_METHODS,
        "competitiveness": COMPETITIVENESS_METHODS,
        "main": MAIN_METHODS,
        "classical": CLASSICAL_METHODS,
        "learned": LEARNED_BASELINES,
        "learned-multiseed": MULTISEED_LEARNED_CONTROLS,
        "learned-expensive": EXPENSIVE_LEARNED_CONTROLS,
    }
    if args.group:
        payload: object = list(groups[args.group])
    else:
        payload = [spec.to_dict() for spec in sorted(_SPECS, key=lambda item: item.order)]
    if args.json:
        print(json.dumps(payload, indent=2))
    elif isinstance(payload, list) and payload and isinstance(payload[0], str):
        print(",".join(payload))
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
