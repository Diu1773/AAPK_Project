from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .io_utils import read_csv_int64_source_id, coerce_int64_source_id
from .step_paths import (
    step5_photometry_dir,
    step8_idmatch_dir,
    step9_selection_dir,
)


_DATE_RE = re.compile(r"(20\d{6})")


def _extract_date_key(filename: str) -> str:
    match = _DATE_RE.search(str(filename))
    return match.group(1) if match else ""


def _read_table(path: Path) -> pd.DataFrame | None:
    suffix = path.suffix.lower()
    preferred_seps = ["\t", ","] if suffix == ".tsv" else [",", "\t"]
    seen: set[str] = set()

    for sep in preferred_seps:
        if sep in seen:
            continue
        seen.add(sep)
        try:
            df = read_csv_int64_source_id(path, sep=sep)
        except Exception:
            continue
        if df is None:
            continue

        # Retry with the alternate separator when a CSV/TSV was parsed into a
        # single unsplit header column (common after the Step 5/8 refactor).
        if len(df.columns) == 1:
            only_col = str(df.columns[0])
            alt_sep = "," if sep == "\t" else "\t"
            if alt_sep in only_col:
                continue
        return df

    return None


def _resolve_photometry_path(result_dir: Path, fname: str) -> Path | None:
    phot_path = step5_photometry_dir(result_dir) / f"{fname}_photometry.tsv"
    return phot_path if phot_path.exists() else None


def _resolve_idmatch_path(result_dir: Path, fname: str) -> Path | None:
    step8_out = step8_idmatch_dir(result_dir)
    date_key = _extract_date_key(fname)

    candidates: list[Path] = []
    if date_key:
        candidates.append(step8_out / date_key / f"idmatch_{fname}.csv")
    candidates.append(step8_out / f"idmatch_{fname}.csv")

    for path in candidates:
        if path.exists():
            return path
    return None


def _load_source_to_id_map(result_dir: Path, filt_hint: str | None = None) -> dict[int, int]:
    step9_out = step9_selection_dir(result_dir)
    if not step9_out.exists():
        return {}

    candidates: list[tuple[Path, str]] = []
    filt_key = str(filt_hint or "").strip().lower()
    if filt_key:
        candidates.extend(
            [
                (step9_out / f"master_catalog_{filt_key}.tsv", "\t"),
                (step9_out / f"id_mapping_{filt_key}.csv", ","),
            ]
        )
    candidates.extend((p, "\t") for p in sorted(step9_out.glob("master_catalog_*.tsv")))
    candidates.extend((p, ",") for p in sorted(step9_out.glob("id_mapping_*.csv")))

    mapping: dict[int, int] = {}
    for path, sep in candidates:
        if not path.exists():
            continue
        try:
            df = read_csv_int64_source_id(path, sep=sep)
        except Exception:
            continue
        if not {"source_id", "ID"} <= set(df.columns):
            continue
        sid_vals = coerce_int64_source_id(df["source_id"])
        id_vals = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")
        for sid_val, id_val in zip(sid_vals, id_vals):
            if pd.isna(sid_val) or pd.isna(id_val):
                continue
            sid_int = int(sid_val)
            if sid_int not in mapping:
                mapping[sid_int] = int(id_val)
    return mapping


def load_frame_photometry(result_dir: Path, fname: str, filt_hint: str | None = None) -> pd.DataFrame | None:
    """Load Step 5 photometry and enrich it with source identity from Step 8/9.

    The refactored Step 5 writes all-source photometry keyed by per-frame `det_uid`.
    Downstream steps still need `source_id` and final stable `ID`, so this loader
    joins Step 8 idmatch output and Step 9 selection catalogs when available.
    """

    phot_path = _resolve_photometry_path(result_dir, fname)
    if phot_path is None:
        return None

    df = _read_table(phot_path)
    if df is None or df.empty:
        return df

    df = df.copy()
    if "id" in df.columns and "ID" not in df.columns:
        df = df.rename(columns={"id": "ID"})
    if "det_idx" in df.columns and "det_uid" not in df.columns:
        df = df.rename(columns={"det_idx": "det_uid"})
    if "file" not in df.columns:
        df["file"] = fname

    if "det_uid" in df.columns:
        df["det_uid"] = pd.to_numeric(df["det_uid"], errors="coerce").astype("Int64")

    need_source_id = "source_id" not in df.columns or coerce_int64_source_id(df["source_id"]).notna().sum() == 0
    if need_source_id and "det_uid" in df.columns:
        idmatch_path = _resolve_idmatch_path(result_dir, fname)
        if idmatch_path is not None:
            idm = _read_table(idmatch_path)
            if idm is not None and not idm.empty:
                if "det_idx" in idm.columns and "det_uid" not in idm.columns:
                    idm = idm.rename(columns={"det_idx": "det_uid"})
                if "det_uid" in idm.columns and "source_id" in idm.columns:
                    idm = idm.copy()
                    idm["det_uid"] = pd.to_numeric(idm["det_uid"], errors="coerce").astype("Int64")
                    idm["source_id"] = coerce_int64_source_id(idm["source_id"]).astype("Int64")
                    merge_cols = ["det_uid", "source_id"]
                    for extra in ("ra_deg", "dec_deg", "sep_arcsec", "match_confidence"):
                        if extra in idm.columns and extra not in df.columns:
                            merge_cols.append(extra)
                    df = df.merge(idm[merge_cols], on="det_uid", how="left")

    if "source_id" in df.columns:
        df["source_id"] = coerce_int64_source_id(df["source_id"]).astype("Int64")

    if "source_id" in df.columns:
        filt_key = filt_hint
        if not filt_key:
            for col in ("FILTER", "filter"):
                if col in df.columns and not df.empty:
                    filt_key = str(df[col].iloc[0]).strip().lower()
                    break
        sid_map = _load_source_to_id_map(result_dir, filt_key)
        if sid_map:
            mapped_ids = df["source_id"].map(sid_map).astype("Int64")
            if "ID" in df.columns:
                # Step 5 may still carry a stale per-frame/local ID from pre-refactor runs.
                existing_ids = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")
                df["ID"] = mapped_ids.where(mapped_ids.notna(), existing_ids).astype("Int64")
            else:
                df["ID"] = mapped_ids

    return df
