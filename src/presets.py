from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    ann_bench_name: str
    metric: str
    dim: int
    indexed_size: int
    query_count: int
    eval_queries: int
    partitions: int
    normalize: bool

    @property
    def hdf5_filename(self) -> str:
        return f"{self.ann_bench_name}.hdf5"

    @property
    def public_url(self) -> str:
        return f"https://ann-benchmarks.com/{self.hdf5_filename}"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


DATASETS: dict[str, DatasetSpec] = {
    "glove": DatasetSpec(
        key="glove",
        ann_bench_name="glove-100-angular",
        metric="cosine",
        dim=100,
        indexed_size=1_183_514,
        query_count=10_000,
        eval_queries=1_000,
        partitions=1_024,
        normalize=False,
    ),
    "sift": DatasetSpec(
        key="sift",
        ann_bench_name="sift-128-euclidean",
        metric="l2",
        dim=128,
        indexed_size=1_000_000,
        query_count=10_000,
        eval_queries=10_000,
        partitions=1_024,
        normalize=False,
    ),
    "gist": DatasetSpec(
        key="gist",
        ann_bench_name="gist-960-euclidean",
        metric="l2",
        dim=960,
        indexed_size=1_000_000,
        query_count=1_000,
        eval_queries=1_000,
        partitions=1_024,
        normalize=False,
    ),
    "deep1m": DatasetSpec(
        key="deep1m",
        ann_bench_name="deep-image-96-angular",
        metric="cosine",
        dim=96,
        indexed_size=9_990_000,
        query_count=10_000,
        eval_queries=10_000,
        partitions=4_096,
        normalize=True,
    ),
    "glove10k": DatasetSpec(
        key="glove10k",
        ann_bench_name="glove-100-angular",
        metric="cosine",
        dim=100,
        indexed_size=10_000,
        query_count=500,
        eval_queries=500,
        partitions=64,
        normalize=False,
    ),
}

PAPER_DATASETS = ("glove", "sift", "gist", "deep1m")
