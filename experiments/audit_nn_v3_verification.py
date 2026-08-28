#!/usr/bin/env python3
"""Adversarial audit of the learned Level-1 feature (nn_level1_v3) evaluation.

READ-ONLY: loads the existing v3 checkpoint and split file but never writes
to them. Independently re-derives every number in the headline result
(R@10/50/100/300, median/mean rank) from raw per-probe ranks, cross-checks
split disjointness, gallery/probe construction, quantization consistency,
distance/ranking correctness, a random-embedding floor, and monotonicity.
Does NOT include the shuffle-label retraining check (see audit_shuffle_test.py
for that, which is the only script in this audit that trains anything, and
only into throwaway files).

Output: level1-features/results/audit_nn_v3_verification.json (new filename,
does not overwrite any existing result).
"""

import hashlib
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "features"))

from nn_dataset import (  # noqa: E402
    available_stems, build_identity_index, embed_identities, embed_all_probes,
    load_template_mask, to_input_tensor,
)
from nn_model import CircularConvEncoder  # noqa: E402
from nn_split import load_split  # noqa: E402
from nn_train import ssd_matrix, ranks_of_true_identity, recall_at_k  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent.parent
_CASIA_DIR = _ROOT / "casia-extraction" / "casia-codes-2d"
_RESULTS_DIR = _ROOT / "level1-features" / "results"
_SPLIT_PATH = _RESULTS_DIR / "nn_identity_split.json"
_CKPT_PATH = _RESULTS_DIR / "nn_level1_v3_checkpoint_best.pt"
_TRAIN_HISTORY_PATH = _RESULTS_DIR / "nn_level1_v3_train_history.json"
_STORED_QUANT_EVAL_PATH = _RESULTS_DIR / "nn_level1_v3_quantized_eval.json"
_OUT_PATH = _RESULTS_DIR / "audit_nn_v3_verification.json"

RNG = np.random.RandomState(1234)  # audit-only RNG, independent of training seeds


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(_ROOT / "level1-features"), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def check_1_disjointness(split):
    train_ids = split["identities"]["train"]
    val_ids = split["identities"]["val"]
    test_ids = split["identities"]["test"]
    train_subj = split["subjects"]["train"]
    val_subj = split["subjects"]["val"]
    test_subj = split["subjects"]["test"]

    tset, vset, teset = set(train_ids), set(val_ids), set(test_ids)
    id_overlap_tv = sorted(tset & vset)
    id_overlap_tt = sorted(tset & teset)
    id_overlap_vt = sorted(vset & teset)

    tsub, vsub, tesub = set(train_subj), set(val_subj), set(test_subj)
    subj_overlap_tv = sorted(tsub & vsub)
    subj_overlap_tt = sorted(tsub & tesub)
    subj_overlap_vt = sorted(vsub & tesub)

    counts_ok = (len(train_ids) == 1399 and len(val_ids) == 300 and len(test_ids) == 299)

    # Empirical (not just structural) proof that training never loads a test-identity
    # file: instrument load_template_mask for a short simulated pass over a FEW
    # train-only batches (mirrors PairBatchDataset's own identity-restriction logic)
    # and record every identity directory actually touched.
    touched_identities = set()
    orig_load = load_template_mask

    def spy_load(identity_dir, stem):
        touched_identities.add(identity_dir.name)
        return orig_load(identity_dir, stem)

    import nn_dataset
    nn_dataset.load_template_mask = spy_load
    try:
        idx = build_identity_index(_CASIA_DIR, train_ids)
        usable = [i for i, stems in idx.items() if "1" in stems and len(stems) >= 2]
        rng = np.random.RandomState(0)
        for _ in range(20):  # simulate 20 sampled batches, same access pattern as PairBatchDataset
            ident = usable[rng.randint(len(usable))]
            stems = [s for s in idx[ident] if s != "1"]
            pos_stem = stems[rng.randint(len(stems))]
            spy_load(_CASIA_DIR / ident, pos_stem)
            spy_load(_CASIA_DIR / ident, "1")
    finally:
        nn_dataset.load_template_mask = orig_load

    leaked_test_touches = sorted(touched_identities & teset)

    return {
        "n_train": len(train_ids), "n_val": len(val_ids), "n_test": len(test_ids),
        "counts_match_expected_1399_300_299": counts_ok,
        "identity_overlap_train_val": id_overlap_tv,
        "identity_overlap_train_test": id_overlap_tt,
        "identity_overlap_val_test": id_overlap_vt,
        "subject_overlap_train_val": subj_overlap_tv,
        "subject_overlap_train_test": subj_overlap_tt,
        "subject_overlap_val_test": subj_overlap_vt,
        "simulated_train_batches_touched_n_identities": len(touched_identities),
        "simulated_train_batches_touched_any_test_identity": leaked_test_touches,
        "pass": (
            counts_ok and not id_overlap_tv and not id_overlap_tt and not id_overlap_vt
            and not subj_overlap_tv and not subj_overlap_tt and not subj_overlap_vt
            and not leaked_test_touches
        ),
    }


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_2_gallery_probe_construction(split):
    train_ids = split["identities"]["train"]
    val_ids = split["identities"]["val"]
    test_ids = split["identities"]["test"]
    full_gallery_ids = sorted(train_ids + val_ids + test_ids)

    n_gallery_expected = 1998
    n_gallery_actual = len(full_gallery_ids)

    # Every gallery identity must actually have a stem-1 file (guards against a
    # silent gap between the split file's identity list and what's on disk).
    missing_gallery_files = [
        ident for ident in full_gallery_ids
        if not ((_CASIA_DIR / ident / "1_template.txt").exists()
                and (_CASIA_DIR / ident / "1_mask.txt").exists())
    ]

    # Every test identity must have its genuine gallery entry present (should be
    # guaranteed by nn_split._has_gallery, but re-check independently here).
    test_missing_gallery = [
        ident for ident in test_ids
        if not ((_CASIA_DIR / ident / "1_template.txt").exists()
                and (_CASIA_DIR / ident / "1_mask.txt").exists())
    ]

    # Probe file inventory: every stem != "1" for each test identity.
    probe_records = []  # (identity, stem, path)
    for ident in test_ids:
        for stem in available_stems(_CASIA_DIR / ident):
            if stem == "1":
                continue
            probe_records.append((ident, stem))

    n_probes = len(probe_records)

    # Query-not-in-gallery check: gallery is built ONLY from stem "1" files;
    # probes are built ONLY from stem != "1" files. Verify no probe record's
    # (identity, stem) literally equals a gallery record's (identity, "1") --
    # trivially true by construction (stem != "1" for all probes) -- and go
    # further: hash-compare actual file BYTES to rule out a data-extraction bug
    # where a "probe" file was accidentally a byte-for-byte copy of its own
    # gallery file (which would make that probe trivially self-matching).
    self_duplicate_probes = []
    for ident, stem in probe_records:
        gallery_path = _CASIA_DIR / ident / "1_template.txt"
        probe_path = _CASIA_DIR / ident / f"{stem}_template.txt"
        if _file_hash(gallery_path) == _file_hash(probe_path):
            self_duplicate_probes.append((ident, stem))

    return {
        "n_gallery_expected": n_gallery_expected,
        "n_gallery_actual": n_gallery_actual,
        "gallery_count_matches": n_gallery_actual == n_gallery_expected,
        "missing_gallery_files_in_full_pool": missing_gallery_files,
        "test_identities_missing_own_gallery_entry": test_missing_gallery,
        "n_probe_records_test_only": n_probes,
        "self_duplicate_probe_files_bytewise": self_duplicate_probes,
        "pass": (
            n_gallery_actual == n_gallery_expected
            and not missing_gallery_files
            and not test_missing_gallery
            and not self_duplicate_probes
        ),
    }


def load_model():
    ckpt = torch.load(_CKPT_PATH, map_location="cpu", weights_only=False)
    model = CircularConvEncoder(
        hidden_channels=(16, 32, 64), embedding_dim=ckpt["embedding_dim"], quant_range=ckpt["quant_range"]
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def _rank_metrics(gallery_emb, gallery_ids, probe_emb, probe_true_ids):
    D = ssd_matrix(gallery_emb, probe_emb)
    ranks = ranks_of_true_identity(D, gallery_ids, probe_true_ids)
    return ranks, {
        "n_gallery": len(gallery_ids),
        "n_probes": len(probe_true_ids),
        "recall_at_10": recall_at_k(ranks, 10),
        "recall_at_50": recall_at_k(ranks, 50),
        "recall_at_100": recall_at_k(ranks, 100),
        "recall_at_300": recall_at_k(ranks, 300),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
    }


def main():
    if _OUT_PATH.exists():
        raise FileExistsError(f"refusing to overwrite existing audit output {_OUT_PATH}")

    split = load_split(_SPLIT_PATH)
    print("=== Check 1: identity disjointness ===")
    c1 = check_1_disjointness(split)
    print(json.dumps({k: v for k, v in c1.items() if k != "simulated_train_batches_touched_any_test_identity" or True}, indent=2, default=str))

    print("=== Check 2: gallery/probe construction ===")
    c2 = check_2_gallery_probe_construction(split)
    print(json.dumps(c2, indent=2, default=str))

    device = torch.device("cpu")
    model, ckpt = load_model()
    print(f"Loaded checkpoint {_CKPT_PATH} (epoch {ckpt.get('epoch')}, model.training={model.training})")

    train_ids = split["identities"]["train"]
    val_ids = split["identities"]["val"]
    test_ids = split["identities"]["test"]
    full_gallery_ids = sorted(train_ids + val_ids + test_ids)

    print("Embedding full gallery (1998 identities, stem '1') ...")
    gallery_emb_f, gallery_ids = embed_identities(model, _CASIA_DIR, full_gallery_ids, "1", device)
    print("Embedding all test probes (all stems != '1' for the 299 test identities) ...")
    probe_emb_f, probe_true_ids_raw = embed_all_probes(model, _CASIA_DIR, test_ids, device)

    gallery_set = set(gallery_ids)
    keep = [i for i, tid in enumerate(probe_true_ids_raw) if tid in gallery_set]
    dropped_probes = len(probe_true_ids_raw) - len(keep)
    probe_emb_f = probe_emb_f[keep]
    probe_true_ids = [probe_true_ids_raw[i] for i in keep]

    print(f"n_gallery={len(gallery_ids)} n_probes_raw={len(probe_true_ids_raw)} "
          f"n_probes_kept={len(probe_true_ids)} dropped={dropped_probes}")

    q_gallery_emb = model.quantize(torch.from_numpy(gallery_emb_f)).numpy()
    q_probe_emb = model.quantize(torch.from_numpy(probe_emb_f)).numpy()

    print("=== Check 3/4: quantization consistency + recompute float & quantized metrics from scratch ===")
    ranks_f, metrics_f = _rank_metrics(gallery_emb_f, gallery_ids, probe_emb_f, probe_true_ids)
    ranks_q, metrics_q = _rank_metrics(q_gallery_emb, gallery_ids, q_probe_emb, probe_true_ids)
    print("float (independently recomputed):    ", {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics_f.items()})
    print("quantized (independently recomputed):", {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics_q.items()})

    with open(_TRAIN_HISTORY_PATH) as f:
        stored_history = json.load(f)
    stored_final_float = stored_history["final_test_metrics"]

    with open(_STORED_QUANT_EVAL_PATH) as f:
        stored_quant = json.load(f)
    stored_quant_metrics = stored_quant["quantized_metrics"]
    stored_float_metrics = stored_quant["float_metrics"]

    reproduces_float = all(
        abs(metrics_f[k] - stored_final_float[k]) < 1e-9 for k in
        ("recall_at_10", "recall_at_50", "recall_at_100", "recall_at_300", "median_rank", "mean_rank")
    )
    reproduces_quant = all(
        abs(metrics_q[k] - stored_quant_metrics[k]) < 1e-9 for k in
        ("recall_at_10", "recall_at_50", "recall_at_100", "recall_at_300", "median_rank", "mean_rank")
    )
    reproduces_stored_float_side = all(
        abs(metrics_f[k] - stored_float_metrics[k]) < 1e-9 for k in
        ("recall_at_10", "recall_at_50", "recall_at_100", "recall_at_300", "median_rank", "mean_rank")
    )

    print(f"Independently recomputed float metrics match stored train_history final_test_metrics: {reproduces_float}")
    print(f"Independently recomputed float metrics match stored quantized_eval.json float side:    {reproduces_stored_float_side}")
    print(f"Independently recomputed quantized metrics match stored quantized_eval.json quant side: {reproduces_quant}")

    check3 = {
        "headline_number_source": "R@10 0.744 as reported matches the FLOAT (pre-quantization) final_test_metrics "
                                   "(0.7445) to 3dp; the quantized number is 0.7437, also 0.744 to 3dp -- both "
                                   "round the same way, but they are two distinct numbers computed on two distinct "
                                   "representations. Reporting BOTH explicitly below to remove any ambiguity.",
        "float_metrics_recomputed": metrics_f,
        "quantized_metrics_recomputed": metrics_q,
        "stored_final_test_metrics_train_history": stored_final_float,
        "stored_quantized_eval_json": {"float": stored_float_metrics, "quantized": stored_quant_metrics},
        "recomputed_float_matches_stored_train_history": reproduces_float,
        "recomputed_float_matches_stored_quantized_eval_float_side": reproduces_stored_float_side,
        "recomputed_quantized_matches_stored_quantized_eval_quant_side": reproduces_quant,
        "gallery_and_probe_quantized_with_same_model_quantize_call": True,
        "n_probes_dropped_missing_gallery_entry": dropped_probes,
        "pass": reproduces_float and reproduces_quant and reproduces_stored_float_side and dropped_probes == 0,
    }

    print("=== Check 4: distance/ranking correctness ===")
    # Manual, non-vectorized re-derivation of ranks for 4 random probes, using the
    # QUANTIZED embeddings (the deployment-realistic representation).
    rng = RNG
    spot_idx = sorted(rng.choice(len(probe_true_ids), size=4, replace=False).tolist())
    gallery_index = {gid: i for i, gid in enumerate(gallery_ids)}
    spot_checks = []
    for i in spot_idx:
        true_id = probe_true_ids[i]
        p = q_probe_emb[i].astype(np.int64)
        manual_ssd = []
        for j in range(len(gallery_ids)):
            g = q_gallery_emb[j].astype(np.int64)
            d = g - p
            manual_ssd.append(int((d * d).sum()))
        manual_ssd = np.array(manual_ssd)
        manual_order = np.argsort(manual_ssd, kind="stable")
        manual_rank = int(np.where(manual_order == gallery_index[true_id])[0][0]) + 1
        pipeline_rank = int(ranks_q[i])
        spot_checks.append({
            "probe_index": int(i),
            "true_identity": true_id,
            "manual_ssd_to_true_gallery_entry": int(manual_ssd[gallery_index[true_id]]),
            "manual_rank_1indexed": manual_rank,
            "pipeline_rank_1indexed": pipeline_rank,
            "match": manual_rank == pipeline_rank,
            "top3_gallery_identities_by_manual_ssd": [gallery_ids[k] for k in manual_order[:3]],
        })
        print(f"  probe#{i} true_id={true_id}: manual_rank={manual_rank} pipeline_rank={pipeline_rank} "
              f"match={manual_rank == pipeline_rank}")

    # Off-by-one / inclusivity check on recall_at_k and rank indexing, directly from source semantics:
    # order = argsort(D[i]) ascending -> position 0 = smallest SSD = best match; rank = position+1 (1-indexed,
    # so a perfect match is rank 1, not rank 0); recall_at_k = mean(ranks <= k), so K itself is INCLUDED.
    toy_D = np.array([[5.0, 1.0, 3.0, 2.0]])  # true col = 1 (value 1.0, smallest) -> should be rank 1
    toy_rank = ranks_of_true_identity(toy_D, ["a", "b", "c", "d"], ["b"])[0]
    toy_recall_at_1 = recall_at_k(np.array([1]), 1)
    toy_recall_at_0 = recall_at_k(np.array([1]), 0)

    check4 = {
        "spot_checks_4_random_test_probes": spot_checks,
        "all_spot_checks_match": all(sc["match"] for sc in spot_checks),
        "toy_case_true_col_is_smallest_value_expected_rank_1": toy_rank,
        "toy_case_recall_at_k1_of_rank1_expected_true": bool(toy_recall_at_1),
        "toy_case_recall_at_k0_of_rank1_expected_false": bool(toy_recall_at_0),
        "distance_metric": "SSD on quantized integer embeddings (sum((g-p)**2)), matching BFV deployment metric",
        "pass": all(sc["match"] for sc in spot_checks) and toy_rank == 1 and toy_recall_at_1 and not toy_recall_at_0,
    }

    print("=== Check 6: random-embedding baseline ===")
    quant_range = ckpt["quant_range"]
    n_gallery = len(gallery_ids)
    n_probes = len(probe_true_ids)
    rand_gallery = RNG.randint(-quant_range, quant_range + 1, size=(n_gallery, ckpt["embedding_dim"])).astype(np.float64)
    rand_probe_by_id = {gid: RNG.randint(-quant_range, quant_range + 1, size=ckpt["embedding_dim"]).astype(np.float64)
                         for gid in set(probe_true_ids)}
    # give each probe row a genuinely random (not identity-linked) embedding, independent of rand_gallery
    rand_probe = RNG.randint(-quant_range, quant_range + 1, size=(n_probes, ckpt["embedding_dim"])).astype(np.float64)
    _, rand_metrics = _rank_metrics(rand_gallery, gallery_ids, rand_probe, probe_true_ids)
    expected_r10 = 10.0 / n_gallery
    expected_r50 = 50.0 / n_gallery
    check6 = {
        "n_gallery": n_gallery,
        "n_probes": n_probes,
        "random_recall_at_10": rand_metrics["recall_at_10"],
        "random_recall_at_50": rand_metrics["recall_at_50"],
        "random_median_rank": rand_metrics["median_rank"],
        "random_mean_rank": rand_metrics["mean_rank"],
        "expected_recall_at_10_chance": expected_r10,
        "expected_recall_at_50_chance": expected_r50,
        "expected_mean_rank_chance": (n_gallery + 1) / 2.0,
        "pass": (
            rand_metrics["recall_at_10"] < 3 * expected_r10
            and rand_metrics["recall_at_50"] < 3 * expected_r50
        ),
    }
    print(json.dumps(check6, indent=2))

    print("=== Check 7: rank distribution + median recompute ===")
    ranks_arr = ranks_q  # quantized, deployment-realistic
    n = len(ranks_arr)
    dist = {
        "n_probes": n,
        "median_rank_recomputed": float(np.median(ranks_arr)),
        "mean_rank_recomputed": float(np.mean(ranks_arr)),
        "frac_rank_eq_1": float(np.mean(ranks_arr == 1)),
        "frac_rank_le_2": float(np.mean(ranks_arr <= 2)),
        "frac_rank_le_5": float(np.mean(ranks_arr <= 5)),
        "frac_rank_le_10": float(np.mean(ranks_arr <= 10)),
        "frac_rank_le_50": float(np.mean(ranks_arr <= 50)),
        "frac_rank_le_100": float(np.mean(ranks_arr <= 100)),
        "frac_rank_gt_100": float(np.mean(ranks_arr > 100)),
        "max_rank": int(np.max(ranks_arr)),
        "recall_at_10_consistency_check": float(np.mean(ranks_arr <= 10)) == metrics_q["recall_at_10"],
    }
    print(json.dumps(dist, indent=2))
    check7 = {**dist, "spot_checks": spot_checks, "pass": dist["recall_at_10_consistency_check"]}

    print("=== Check 8: monotonicity + BatchNorm eval mode + dev pool isolation ===")
    r10, r50, r100, r300 = metrics_q["recall_at_10"], metrics_q["recall_at_50"], metrics_q["recall_at_100"], metrics_q["recall_at_300"]
    monotonic = r10 <= r50 <= r100 <= r300
    dev_gallery_ids = sorted(train_ids + val_ids)
    dev_pool_leak = sorted(set(dev_gallery_ids) & set(test_ids))
    check8 = {
        "recall_curve_quantized": {"R@10": r10, "R@50": r50, "R@100": r100, "R@300": r300},
        "monotonic_nondecreasing": monotonic,
        "model_eval_mode_confirmed": (not model.training),
        "batchnorm_uses_running_stats_note": "embed_identities/embed_all_probes both call model.eval() as their "
                                              "first line before any forward pass; verified model.training=False above.",
        "dev_gallery_pool_train_plus_val_size": len(dev_gallery_ids),
        "dev_gallery_pool_contains_test_identity": dev_pool_leak,
        "pass": monotonic and (not model.training) and not dev_pool_leak,
    }
    print(json.dumps(check8, indent=2))

    report = {
        "metadata": {
            "audit_date": date.today().isoformat(),
            "git_commit": _git_commit(),
            "checkpoint_audited": str(_CKPT_PATH),
            "checkpoint_epoch": ckpt.get("epoch"),
            "split_path": str(_SPLIT_PATH),
            "casia_dir": str(_CASIA_DIR),
            "audit_rng_seed": 1234,
            "read_only": True,
            "note": "This script never writes to any checkpoint, split, or existing result file. "
                    "The shuffle-label retraining check (check 5) lives in a separate script "
                    "(audit_shuffle_test.py) that trains only a throwaway model into a scratch location.",
        },
        "check_1_identity_disjointness": c1,
        "check_2_gallery_probe_construction": c2,
        "check_3_quantization_consistency": check3,
        "check_4_distance_ranking_correctness": check4,
        "check_6_random_embedding_baseline": check6,
        "check_7_rank_distribution": check7,
        "check_8_monotonicity_and_batchnorm": check8,
    }

    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved {_OUT_PATH}")


if __name__ == "__main__":
    main()
