from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

from artifacts import write_json
from console import banner, info, print_table
from index_manifest import write_index_manifest
from measurement_contract import (
    AVG_COMPUTATIONS_MODE_LIRA,
    INDEX_OVERHEAD_MODE,
    QUERY_MEM_MODE_ISOLATED,
    serving_payload_metrics,
)
from presets import DATASETS

from methods.lira_runtime import LiraConfig, run_lira_selected_online_eval, run_lira_smallscale


class Lira:
    def __init__(self, dataset: str, *, seed: int = 0) -> None:
        self.dataset = dataset
        self.name = "LIRA"
        self.cfg = LiraConfig(
            dataset=dataset,
            n_bkt=DATASETS[dataset].partitions,
            seed=seed,
            prefer_idle_gpu=False,
            num_threads=int(os.environ.get("LISBENCH_TRAINING_THREADS", "0")),
        )

    def _train(
        self,
        *,
        data_root: Path,
        exp_dir: Path,
        force_prepare: bool,
    ):
        banner("method", f"{self.name} | dataset={self.dataset}")
        info(
            f"probe_depth={self.cfg.probe_depth}, n_bkt={self.cfg.n_bkt}, metric={self.cfg.metric}, "
            f"index_full_dataset={self.cfg.index_full_dataset}, selected_part_policy={self.cfg.selected_part_policy}"
        )
        info(
            f"epochs={self.cfg.n_epoch}, batch_size={self.cfg.batch_size}, n_mul={self.cfg.n_mul}, "
            f"repa_step={self.cfg.repa_step}, inner_index={self.cfg.inner_index_type}"
        )
        result = run_lira_smallscale(self.cfg, exp_dir, data_root=data_root, force_prepare=force_prepare)
        if not result.selected_rows:
            raise RuntimeError("LIRA produced no selected threshold rows.")
        return result

    def _threshold_points(self, selected_rows: list[dict[str, float]]) -> list[tuple[str, dict[str, float]]]:
        """Select one protocol point plus two non-degenerate frontier points.

        LIRA's headline threshold is the point nearest its configured target
        probe budget. The economy point must remain within five percentage
        points of that point's Recall@10; otherwise a zero-work/zero-recall
        threshold can masquerade as useful frontier context.
        """

        target_nprobe = float(self.cfg.probe_depth)
        selected = min(
            selected_rows,
            key=lambda row: (
                abs(float(row.get("nprobe", float("nan"))) - target_nprobe),
                -float(row.get("recall10_at_10", float("nan"))),
                float(row.get("avg_computations", float("inf"))),
            ),
        )
        selected_recall = float(selected.get("recall10_at_10", float("nan")))
        useful_economy_rows = [
            row
            for row in selected_rows
            if float(row.get("recall10_at_10", float("-inf"))) >= selected_recall - 0.05
        ]
        economy = min(
            useful_economy_rows or selected_rows,
            key=lambda row: (
                float(row.get("avg_computations", float("inf"))),
                -float(row.get("recall10_at_10", float("nan"))),
            ),
        )
        recall = max(
            selected_rows,
            key=lambda row: (
                float(row.get("recall10_at_10", float("-inf"))),
                -float(row.get("avg_computations", float("inf"))),
            ),
        )
        return [
            ("threshold-economy", economy),
            ("threshold-selected", selected),
            ("threshold-recall", recall),
        ]

    def _evaluate_point(
        self,
        *,
        result,
        summary_row: dict[str, float],
        operating_point_id: str,
        data_root: Path,
        exp_dir: Path,
        manifest_payload: dict[str, object],
    ) -> dict[str, object]:
        online = run_lira_selected_online_eval(
            self.cfg,
            exp_dir,
            data_root=data_root,
            threshold=float(summary_row["threshold"]),
        )
        canonical_point = operating_point_id == "default"
        if not canonical_point:
            point_dir = exp_dir / "operating_points" / operating_point_id
            point_dir.mkdir(parents=True, exist_ok=True)
            source = exp_dir / "per_query_debug.csv"
            if source.exists():
                shutil.copy2(source, point_dir / source.name)
        external_vector_payload_mb = (
            float(DATASETS[self.dataset].indexed_size * DATASETS[self.dataset].dim * 4)
            / (1024.0 * 1024.0)
        )
        serving_payload = serving_payload_metrics(
            serialized_artifacts_mb=float(manifest_payload["serving_footprint_mb"]),
            external_vector_payload_mb=external_vector_payload_mb,
            canonical_vector_payload_mb=external_vector_payload_mb,
            requires_full_vectors=True,
        )
        timing_fields = {
            key: value
            for key, value in online.items()
            if key.startswith("batch1_")
            or key.startswith("batch256_")
            or key
            in {
                "cpu_time_ms_per_query_batch_256",
                "cpu_time_mode",
                "timing_trials",
                "timing_statistic",
                "warmup_queries",
                "omp_wait_policy",
                "gomp_spincount",
            }
        }
        qps_batch256 = float(online.get("qps_batch_256", online.get("qps", float("nan"))))
        qps_batch1 = float(online.get("qps_batch_1", online.get("qps", float("nan"))))
        qps_native_batched = float(online.get("qps_native_batched", qps_batch256))
        latency_batch1 = float(
            online.get(
                "latency_ms_batch_1_mean",
                1000.0 / qps_batch1 if qps_batch1 > 0.0 else float("nan"),
            )
        )
        row: dict[str, object] = {
            "dataset": self.dataset,
            "method": self.name,
            "seed": int(self.cfg.seed),
            "operating_point_id": operating_point_id,
            "probe_depth": int(self.cfg.probe_depth),
            "recall_at_1": float(online["recall_at_1"]),
            "recall_at_10": float(online["recall_at_10"]),
            "recall_at_100": float(online["recall_at_100"]),
            "candidate_recall_at_1": float(online["candidate_recall_at_1"]),
            "candidate_recall_at_10": float(online["candidate_recall_at_10"]),
            "candidate_recall_at_100": float(online["candidate_recall_at_100"]),
            "avg_computations": float(online["avg_computations"]),
            "computation_min": float(online.get("computation_min", float("nan"))),
            "computation_max": float(online.get("computation_max", float("nan"))),
            "avg_computations_mode": str(online.get("avg_computations_mode", AVG_COMPUTATIONS_MODE_LIRA)),
            "avg_candidates": float(online["avg_computations"]),
            "avg_candidates_mode": str(online.get("avg_computations_mode", AVG_COMPUTATIONS_MODE_LIRA)),
            "qps": qps_batch256,
            "qps_batch_1": qps_batch1,
            "qps_batch_256": qps_batch256,
            "qps_batch_1_contract": "lira_end_to_end_single_query",
            "qps_batch_256_contract": "lira_batched_routing_serial_local_search",
            "latency_ms_batch_1_mean": latency_batch1,
            "qps_native_batched": qps_native_batched,
            "qps_native_batched_contract": str(
                online.get("qps_native_batched_contract", "legacy_unlabeled_batch")
            ),
            "query_batch_size": 256,
            "benchmark_batch_sizes": "1,256",
            **timing_fields,
            "index_overhead_mb": float(manifest_payload["index_overhead_mb"]),
            "index_overhead_mode": INDEX_OVERHEAD_MODE,
            **serving_payload,
            "index_size_mb": float(manifest_payload["index_overhead_mb"]),
            "index_size_mode": INDEX_OVERHEAD_MODE,
            "query_mem_delta_mb": float(online["query_mem_delta_mb"]),
            "query_mem_delta_mode": QUERY_MEM_MODE_ISOLATED,
            "query_mem_measurement_operating_point": "threshold_specific_in_process_delta",
            "train_s": float(result.metadata.get("timings_s", {}).get("total", 0.0)),
            "build_total_s": float(result.metadata.get("timings_s", {}).get("total", 0.0)),
            "exact_rerank_count": float(online["exact_rerank_count"]),
            "distance_scoring_equivalent_work": float(online["distance_scoring_equivalent_work"]),
            "distance_scoring_equivalent_mode": "measured_full_distance_count",
            "selection_metric": "candidate_coverage_at_10",
            "selection_contract": "target_probe_budget_nearest_threshold",
            "experiment_dir": str(exp_dir),
            "threshold": float(summary_row["threshold"]),
            "nprobe": float(online["nprobe"]),
            "selected_part": int(result.selected_part),
        }
        point_dir = exp_dir if canonical_point else exp_dir / "operating_points" / operating_point_id
        write_json(
            point_dir / "timing_breakdown.json",
            {
                "operating_point_id": operating_point_id,
                "qps_batch_1": qps_batch1,
                "qps_batch_256": qps_batch256,
                "qps_native_batched": qps_native_batched,
                "qps_batch_1_contract": row["qps_batch_1_contract"],
                "qps_batch_256_contract": row["qps_batch_256_contract"],
                "qps_native_batched_contract": row["qps_native_batched_contract"],
                **timing_fields,
            },
        )
        write_json(
            point_dir / "candidate_budget_breakdown.json",
            {
                "operating_point_id": operating_point_id,
                "avg_computations": row["avg_computations"],
                "avg_computations_mode": row["avg_computations_mode"],
                "local_distance_computations": online.get("dist_computations"),
                "exact_rerank_count": row["exact_rerank_count"],
                "distance_scoring_equivalent_work": row["distance_scoring_equivalent_work"],
            },
        )
        print_table(
            f"{self.name} | competitiveness",
            ["metric", "value"],
            [
                ["Recall@10", f"{float(row['recall_at_10']):.4f}"],
                ["Avg computations", f"{float(row['avg_computations']):.1f}"],
                ["QPS", f"{float(row['qps']):.1f}"],
                ["Index overhead (MB)", f"{float(row['index_overhead_mb']):.3f}"],
                ["Serving footprint (MB)", f"{float(row['serving_footprint_mb']):.3f}"],
                ["Selected part", str(int(row["selected_part"]))],
                ["Threshold", f"{float(row['threshold']):.4f}"],
                ["Online nprobe", f"{float(row['nprobe']):.2f}"],
            ],
        )
        return row

    def run_competitiveness(
        self,
        *,
        data_root: Path,
        exp_dir: Path,
        force_prepare: bool = False,
    ) -> dict[str, object]:
        result = self._train(data_root=data_root, exp_dir=exp_dir, force_prepare=force_prepare)
        selected = self._threshold_points(result.selected_rows)[1][1]
        manifest_payload = json.loads(write_index_manifest(exp_dir).read_text(encoding="utf-8"))
        row = self._evaluate_point(
            result=result,
            summary_row=selected,
            operating_point_id="default",
            data_root=data_root,
            exp_dir=exp_dir,
            manifest_payload=manifest_payload,
        )
        write_json(exp_dir / "competitiveness_summary.json", row)
        return row

    def run_competitiveness_sweep(
        self,
        *,
        data_root: Path,
        exp_dir: Path,
        force_prepare: bool = False,
    ) -> list[dict[str, object]]:
        result = self._train(data_root=data_root, exp_dir=exp_dir, force_prepare=force_prepare)
        manifest_payload = json.loads(write_index_manifest(exp_dir).read_text(encoding="utf-8"))
        rows = [
            self._evaluate_point(
                result=result,
                summary_row=summary_row,
                operating_point_id=point_id,
                data_root=data_root,
                exp_dir=exp_dir,
                manifest_payload=manifest_payload,
            )
            for point_id, summary_row in self._threshold_points(result.selected_rows)
        ]
        selected_debug = exp_dir / "operating_points" / "threshold-selected" / "per_query_debug.csv"
        if selected_debug.exists():
            shutil.copy2(selected_debug, exp_dir / "per_query_debug.csv")
        for name in ("timing_breakdown.json", "candidate_budget_breakdown.json"):
            selected_artifact = exp_dir / "operating_points" / "threshold-selected" / name
            if selected_artifact.exists():
                shutil.copy2(selected_artifact, exp_dir / name)
        write_json(
            exp_dir / "serving_operating_points.json",
            {
                "shared_built_index": True,
                "selection_contract": (
                    "target-probe-budget threshold is headline; economy-within-5pp and "
                    "maximum-recall thresholds are frontier context"
                ),
                "operating_points": [
                    {"operating_point_id": row["operating_point_id"], "threshold": row["threshold"]}
                    for row in rows
                ],
            },
        )
        write_json(exp_dir / "competitiveness_summary.json", rows)
        return rows
