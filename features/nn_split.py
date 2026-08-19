#!/usr/bin/env python3
"""Subject-level train/val/test split for the learned Level-1 feature network.

Identities in casia-codes-2d/ are named "<subject>_<eye>" (e.g. "763_L"), so
each of the 1000 CASIA-Iris-Thousand subjects contributes two identities (L
and R eye). Left/right iris texture is believed uncorrelated, but the split
is made at the SUBJECT level (both eyes of a subject always land in the same
split) as the more conservative choice -- it rules out any chance of
subject-level leakage between splits, at zero cost.

Split roles (see nn_train.py for how each is actually used):
  train -- gallery+probe images of these identities are the only images ever
           used to form training triplets (gradient updates touch nothing
           else).
  val   -- used for early stopping / hyperparameter choices during
           development. Evaluated against a gallery pool of train+val
           identities only, so test never influences a development decision.
  test  -- held out untouched until a single final evaluation, at which
           point it is evaluated against the FULL 2000-identity gallery pool
           (train+val+test), so Recall@K / rank statistics are computed
           against the same N=2000 candidate pool as the paper's own
           reported numbers -- only the *query* identities are restricted to
           the held-out test set.

Default ratios: 70/15/15 by subject (700/150/150 subjects -> up to
1400/300/300 identities; some subjects are missing an eye's directory, see
`discover_subjects`). Seed fixed for reproducibility.

Two identities, 524_L and 668_L, have no "1_template.txt"/"1_mask.txt" (their
gallery image failed Open Iris segmentation during extraction -- see
level1-features/README.md's full2000 evaluation notes, which exclude the
same two identities). An identity with no gallery image cannot serve as a
genuine-match target or a distractor, so `_identities_for` drops any
identity directory missing a usable stem-1 pair, leaving 1998 usable
identities out of 2000.
"""

import argparse
import json
import random
import subprocess
from datetime import date
from pathlib import Path
from typing import Dict, List

EYES = ("L", "R")

__all__ = [
    "discover_subjects",
    "build_split",
    "save_split",
    "load_split",
]


def discover_subjects(casia_dir: Path) -> List[str]:
    """Sorted list of subject ids that have at least one eye directory present."""
    casia_dir = Path(casia_dir)
    subjects = set()
    for d in casia_dir.iterdir():
        if d.is_dir() and len(d.name) > 2 and d.name[-2] == "_" and d.name[-1] in EYES:
            subjects.add(d.name[:-2])
    return sorted(subjects)


def _has_gallery(identity_dir: Path) -> bool:
    """An identity is usable only if its gallery image (stem '1') exists.

    524_L and 668_L fail this check: their gallery image failed Open Iris
    segmentation during extraction, so no "1_template.txt"/"1_mask.txt" was
    ever written for them.
    """
    return (identity_dir / "1_template.txt").exists() and (identity_dir / "1_mask.txt").exists()


def _identities_for(casia_dir: Path, subjects: List[str]) -> List[str]:
    identities = []
    for s in subjects:
        for eye in EYES:
            d = casia_dir / f"{s}_{eye}"
            if d.is_dir() and _has_gallery(d):
                identities.append(f"{s}_{eye}")
    return identities


def build_split(casia_dir: Path, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 0) -> Dict:
    """Shuffle subjects with a fixed seed and cut into train/val/test.

    Returns a dict with "subjects" and "identities" sub-dicts, each mapping
    split name -> sorted list.
    """
    casia_dir = Path(casia_dir)
    subjects = discover_subjects(casia_dir)
    rng = random.Random(seed)
    shuffled = subjects[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_test = round(n * test_frac)
    n_val = round(n * val_frac)

    test_subj = sorted(shuffled[:n_test])
    val_subj = sorted(shuffled[n_test : n_test + n_val])
    train_subj = sorted(shuffled[n_test + n_val :])

    subject_split = {"train": train_subj, "val": val_subj, "test": test_subj}
    identity_split = {
        role: _identities_for(casia_dir, subj_list) for role, subj_list in subject_split.items()
    }

    return {"subjects": subject_split, "identities": identity_split}


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent.parent), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def save_split(split: Dict, path: Path, casia_dir: Path, val_frac: float, test_frac: float, seed: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "gallery_path": str(casia_dir),
            "subject_count": sum(len(v) for v in split["subjects"].values()),
            "identity_count": sum(len(v) for v in split["identities"].values()),
            "val_frac": val_frac,
            "test_frac": test_frac,
            "seed": seed,
            "git_commit": _git_commit(),
            "date": date.today().isoformat(),
            "split_axis": "subject (both eyes of a subject share a split)",
        },
        "counts": {
            "subjects": {k: len(v) for k, v in split["subjects"].items()},
            "identities": {k: len(v) for k, v in split["identities"].items()},
        },
        "subjects": split["subjects"],
        "identities": split["identities"],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_split(path: Path) -> Dict:
    with open(path) as f:
        payload = json.load(f)
    return {"subjects": payload["subjects"], "identities": payload["identities"], "metadata": payload["metadata"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and save the subject-level train/val/test split.")
    parser.add_argument("-d", "--casia-dir", type=Path,
                         default=Path(__file__).resolve().parent.parent.parent / "casia-extraction" / "casia-codes-2d")
    parser.add_argument("-o", "--out", type=Path,
                         default=Path(__file__).resolve().parent.parent / "results" / "nn_identity_split.json")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    split = build_split(args.casia_dir, args.val_frac, args.test_frac, args.seed)
    save_split(split, args.out, args.casia_dir, args.val_frac, args.test_frac, args.seed)
    for role in ("train", "val", "test"):
        print(f"{role}: {len(split['subjects'][role])} subjects, {len(split['identities'][role])} identities")
    print(f"Saved split to {args.out}")
