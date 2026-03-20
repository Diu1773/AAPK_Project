"""Pure helpers for merged ID reconciliation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

from ...utils.io_utils import coerce_int64_source_id


def extract_row_float(row: pd.Series, *cols: str) -> float:
    for col in cols:
        if col in row.index:
            val = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
            if np.isfinite(val):
                return float(val)
    return float("nan")


def row_radec(row: pd.Series) -> tuple[float, float]:
    return (
        extract_row_float(row, "ra_deg", "ra", "RA"),
        extract_row_float(row, "dec_deg", "dec", "DEC"),
    )


def append_folder_tag(existing: str, tag: str) -> str:
    parts = [p for p in str(existing or "").split(",") if p]
    if tag not in parts:
        parts.append(tag)
    return ",".join(parts)


def best_positional_match(row: pd.Series, canonical_df: pd.DataFrame, tol_arcsec: float) -> tuple[int | None, float]:
    ra, dec = row_radec(row)
    if not (np.isfinite(ra) and np.isfinite(dec)):
        return None, float("nan")
    if canonical_df is None or canonical_df.empty or "ra_deg" not in canonical_df.columns or "dec_deg" not in canonical_df.columns:
        return None, float("nan")

    cand = canonical_df.copy()
    cand_ra = pd.to_numeric(cand["ra_deg"], errors="coerce")
    cand_dec = pd.to_numeric(cand["dec_deg"], errors="coerce")
    mask = cand_ra.notna() & cand_dec.notna()
    if not mask.any():
        return None, float("nan")

    sc = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    csc = SkyCoord(cand_ra[mask].to_numpy(float) * u.deg, cand_dec[mask].to_numpy(float) * u.deg, frame="icrs")
    sep = sc.separation(csc).arcsec
    if len(sep) == 0:
        return None, float("nan")
    best_i = int(np.argmin(sep))
    best_sep = float(sep[best_i])
    if not np.isfinite(best_sep) or best_sep > tol_arcsec:
        return None, best_sep
    best_rows = cand.loc[mask].reset_index(drop=True)
    return int(pd.to_numeric(best_rows.loc[best_i, "source_id"], errors="coerce")), best_sep


def canonicalize_catalog_row(
    row: pd.Series,
    merged_id: int,
    merged_source_id: int,
    folder_tag: str,
) -> dict:
    data = row.to_dict()
    data["ID"] = int(merged_id)
    data["source_id"] = int(merged_source_id)
    data["gaia_id"] = int(merged_source_id) if int(merged_source_id) > 0 else np.nan
    data["match_status"] = "matched" if int(merged_source_id) > 0 else "no_gaia_match"
    data["folder_count"] = 1
    data["folder_tags"] = folder_tag
    return data


def next_generated_negative_source_id(current_catalogs: dict[str, pd.DataFrame]) -> int:
    min_sid = 0
    for df in current_catalogs.values():
        if df is None or df.empty or "source_id" not in df.columns:
            continue
        sid_vals = coerce_int64_source_id(df["source_id"]).dropna().astype("int64")
        if not sid_vals.empty:
            min_sid = min(min_sid, int(sid_vals.min()))
    return min_sid - 1 if min_sid <= 0 else -1


def reconcile_workspace_catalogs(
    folders: list[Path],
    catalogs_by_folder: dict[str, dict[str, pd.DataFrame]],
    folder_tags: dict[str, str],
    pos_tol_arcsec: float,
    logger=None,
) -> dict:
    if not folders:
        return {
            "canonical_by_filter": {},
            "local_id_maps": {},
            "match_summary_rows": [],
            "match_records": [],
        }

    all_filters = sorted({
        flt for folder in folders
        for flt in catalogs_by_folder.get(str(folder), {}).keys()
    })
    if not all_filters:
        return {
            "canonical_by_filter": {},
            "local_id_maps": {},
            "match_summary_rows": [],
            "match_records": [],
        }

    base_folder = folders[0]
    next_negative_sid = next_generated_negative_source_id({})
    canonical_by_filter: dict[str, pd.DataFrame] = {}
    next_id_by_filter: dict[str, int] = {}
    local_id_maps: dict[str, dict[str, dict[int, dict[str, int]]]] = {}
    match_summary_rows: list[dict] = []
    match_records: list[dict] = []

    for folder in folders:
        folder_key = str(folder)
        folder_tag = folder_tags[folder_key]
        local_id_maps.setdefault(folder_key, {})
        filter_catalogs = catalogs_by_folder.get(folder_key, {})

        for flt in all_filters:
            df = filter_catalogs.get(flt)
            if df is None or df.empty:
                continue

            if flt not in canonical_by_filter:
                canonical_by_filter[flt] = pd.DataFrame()
                next_id_by_filter[flt] = 1

            canon = canonical_by_filter[flt].copy()
            local_map: dict[int, dict[str, int]] = {}
            n_exact = 0
            n_pos = 0
            n_new = 0

            if folder == base_folder and canon.empty:
                seeded_rows = []
                max_id = 0
                for _, row in df.iterrows():
                    local_id = pd.to_numeric(pd.Series([row.get("ID")]), errors="coerce").iloc[0]
                    if not np.isfinite(local_id):
                        continue
                    sid_val = coerce_int64_source_id(pd.Series([row.get("source_id")])).iloc[0]
                    if pd.isna(sid_val):
                        sid = next_negative_sid
                        next_negative_sid -= 1
                    else:
                        sid = int(sid_val)
                    merged_id = int(local_id)
                    max_id = max(max_id, merged_id)
                    seeded_rows.append(canonicalize_catalog_row(row, merged_id, sid, folder_tag))
                    local_map[int(local_id)] = {
                        "merged_id": merged_id,
                        "merged_source_id": sid,
                    }
                    match_records.append({
                        "folder": folder.name,
                        "folder_tag": folder_tag,
                        "filter": flt,
                        "local_id": int(local_id),
                        "local_source_id": None if pd.isna(sid_val) else int(sid_val),
                        "merged_id": merged_id,
                        "merged_source_id": sid,
                        "method": "base",
                        "sep_arcsec": np.nan,
                        "status": "base",
                    })
                canon = pd.DataFrame(seeded_rows)
                next_id_by_filter[flt] = max_id + 1 if max_id > 0 else 1
                canonical_by_filter[flt] = canon
                local_id_maps[folder_key][flt] = local_map
                match_summary_rows.append({
                    "folder": folder.name,
                    "filter": flt,
                    "exact": len(seeded_rows),
                    "pos": 0,
                    "new": 0,
                    "total": len(seeded_rows),
                    "status": "base",
                })
                continue

            canon_sid_map = {}
            if not canon.empty and "source_id" in canon.columns:
                sid_vals = coerce_int64_source_id(canon["source_id"]).astype("Int64")
                for idx_row, sid_val in enumerate(sid_vals):
                    if pd.notna(sid_val) and int(sid_val) not in canon_sid_map:
                        canon_sid_map[int(sid_val)] = idx_row

            used_canonical_sids: set[int] = set()
            for _, row in df.iterrows():
                local_id = pd.to_numeric(pd.Series([row.get("ID")]), errors="coerce").iloc[0]
                if not np.isfinite(local_id):
                    continue
                local_id = int(local_id)
                sid_val = coerce_int64_source_id(pd.Series([row.get("source_id")])).iloc[0]
                sid_int = None if pd.isna(sid_val) else int(sid_val)

                matched_sid = None
                match_method = ""
                sep_arcsec = float("nan")

                if sid_int is not None and sid_int in canon_sid_map:
                    matched_sid = sid_int
                    match_method = "source_id"
                else:
                    matched_sid, sep_arcsec = best_positional_match(row, canon, pos_tol_arcsec)
                    if matched_sid is not None and matched_sid not in used_canonical_sids:
                        match_method = "position"
                    else:
                        matched_sid = None

                if matched_sid is not None and matched_sid in canon_sid_map:
                    canon_idx = canon_sid_map[matched_sid]
                    merged_id = int(pd.to_numeric(pd.Series([canon.iloc[canon_idx]["ID"]]), errors="coerce").iloc[0])
                    local_map[local_id] = {
                        "merged_id": merged_id,
                        "merged_source_id": int(matched_sid),
                    }
                    used_canonical_sids.add(int(matched_sid))
                    if match_method == "source_id":
                        n_exact += 1
                    else:
                        n_pos += 1
                    canon.at[canon_idx, "folder_count"] = int(pd.to_numeric(pd.Series([canon.iloc[canon_idx].get("folder_count", 1)]), errors="coerce").iloc[0] or 1) + 1
                    canon.at[canon_idx, "folder_tags"] = append_folder_tag(canon.iloc[canon_idx].get("folder_tags", ""), folder_tag)
                    match_records.append({
                        "folder": folder.name,
                        "folder_tag": folder_tag,
                        "filter": flt,
                        "local_id": local_id,
                        "local_source_id": sid_int,
                        "merged_id": merged_id,
                        "merged_source_id": int(matched_sid),
                        "method": match_method,
                        "sep_arcsec": sep_arcsec,
                        "status": "matched",
                    })
                    continue

                merged_id = next_id_by_filter.get(flt, 1)
                next_id_by_filter[flt] = merged_id + 1

                if sid_int is not None and sid_int not in canon_sid_map:
                    merged_source_id = sid_int
                else:
                    merged_source_id = next_negative_sid
                    next_negative_sid -= 1

                new_row = canonicalize_catalog_row(row, merged_id, merged_source_id, folder_tag)
                canon = pd.concat([canon, pd.DataFrame([new_row])], ignore_index=True, sort=False)
                canon_sid_map[int(merged_source_id)] = len(canon) - 1
                local_map[local_id] = {
                    "merged_id": merged_id,
                    "merged_source_id": int(merged_source_id),
                }
                n_new += 1
                match_records.append({
                    "folder": folder.name,
                    "folder_tag": folder_tag,
                    "filter": flt,
                    "local_id": local_id,
                    "local_source_id": sid_int,
                    "merged_id": merged_id,
                    "merged_source_id": int(merged_source_id),
                    "method": "new",
                    "sep_arcsec": np.nan,
                    "status": "new",
                })

            canon = canon.sort_values("ID").reset_index(drop=True)
            canonical_by_filter[flt] = canon
            local_id_maps[folder_key][flt] = local_map
            match_summary_rows.append({
                "folder": folder.name,
                "filter": flt,
                "exact": n_exact,
                "pos": n_pos,
                "new": n_new,
                "total": len(local_map),
                "status": "OK" if local_map else "empty",
            })
            if logger is not None:
                logger(
                    f"[MATCH] {folder.name} / {flt}: exact={n_exact} positional={n_pos} new={n_new} total={len(local_map)}"
                )

    return {
        "canonical_by_filter": canonical_by_filter,
        "local_id_maps": local_id_maps,
        "match_summary_rows": match_summary_rows,
        "match_records": match_records,
    }
