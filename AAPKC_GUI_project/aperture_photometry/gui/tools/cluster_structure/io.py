from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

from aperture_photometry.utils.io_utils import parse_int64_series
from aperture_photometry.utils.step_paths import (
    crop_is_active,
    crop_rect_path,
    legacy_step5_refbuild_dir,
    legacy_step7_refbuild_dir,
    step2_cropped_dir,
    step5_dir,
    step6_dir,
)


@dataclass
class ResolvedInputs:
    result_dir: Path
    output_dir: Path
    diagnostics_dir: Path
    cache_dir: Path
    star_table_path: Path
    ref_meta_path: Path
    ref_fits_path: Path
    ref_frame: str
    crop_rect: Optional[dict]


def ensure_output_dirs(result_dir: Path) -> tuple[Path, Path, Path]:
    out = Path(result_dir) / "Tools_Analyze_Cluster_Structure"
    diag = out / "diagnostics"
    cache = out / "cache"
    out.mkdir(parents=True, exist_ok=True)
    diag.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    return out, diag, cache


def resolve_star_table_path(result_dir: Path) -> Optional[Path]:
    root = Path(result_dir)
    candidates = [
        root / "step10" / "cmd_with_gaia_membership.csv",
        root / "cmd_with_gaia_membership.csv",
        root / "step11" / "cmd_with_gaia_membership.csv",
        root / "step10" / "median_by_ID_filter_wide_cmd.csv",
        root / "median_by_ID_filter_wide_cmd.csv",
        root / "step11" / "median_by_ID_filter_wide_cmd.csv",
        root / "step10" / "median_by_ID_filter_wide.csv",
        root / "median_by_ID_filter_wide.csv",
        root / "step11" / "median_by_ID_filter_wide.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def resolve_ref_meta_path(result_dir: Path) -> Optional[Path]:
    root = Path(result_dir)
    candidates = [
        step6_dir(root) / "ref_build_meta.json",
        legacy_step5_refbuild_dir(root) / "ref_build_meta.json",
        legacy_step7_refbuild_dir(root) / "ref_build_meta.json",
        root / "ref_build_meta.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _astap_wcs_candidates(fits_path: Path) -> list[Path]:
    return [
        fits_path.with_suffix(".wcs"),
        Path(str(fits_path) + ".wcs"),
        fits_path.parent / (fits_path.stem + ".wcs"),
        fits_path.parent / (fits_path.name + ".wcs"),
    ]


def _parse_astap_wcs_file(wcs_path: Path) -> dict:
    out: dict[str, object] = {}
    try:
        lines = wcs_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return out
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        if "/" in s:
            s = s.split("/", 1)[0].strip()
        if "=" not in s:
            continue
        key, val = [t.strip() for t in s.split("=", 1)]
        if not key:
            continue
        if val.startswith("'") and val.endswith("'"):
            out[key] = val.strip("'")
            continue
        try:
            if "." in val or "E" in val.upper():
                out[key] = float(val)
            else:
                out[key] = int(val)
        except Exception:
            out[key] = val
    return out


def _resolve_ref_fits_path(params, result_dir: Path, ref_frame: str) -> Optional[Path]:
    result_dir = Path(result_dir)
    if crop_is_active(result_dir):
        cand = step2_cropped_dir(result_dir) / ref_frame
        if cand.exists():
            return cand
        legacy = result_dir / "cropped" / ref_frame
        if legacy.exists():
            return legacy

    candidates: list[Path] = []
    try:
        candidates.append(Path(getattr(params.P, "data_dir", result_dir)) / ref_frame)
    except Exception:
        pass
    try:
        if hasattr(params, "get_file_path"):
            candidates.append(Path(params.get_file_path(ref_frame)))
    except Exception:
        pass
    candidates.extend(
        [
            step5_dir(result_dir) / ref_frame,
            result_dir / ref_frame,
        ]
    )
    for c in candidates:
        try:
            if c.exists():
                return c
        except Exception:
            continue
    return None


def load_ref_fits_with_wcs(ref_fits_path: Path) -> tuple[np.ndarray, fits.Header, WCS]:
    with fits.open(ref_fits_path) as hdul:
        data = np.asarray(hdul[0].data, dtype=float)
        header = hdul[0].header.copy()

    w = WCS(header, relax=True)
    if getattr(w, "has_celestial", False):
        return data, header, w

    for wcs_path in _astap_wcs_candidates(ref_fits_path):
        if not wcs_path.exists():
            continue
        wd = _parse_astap_wcs_file(wcs_path)
        if not wd:
            continue
        hdr2 = fits.Header()
        for k, v in wd.items():
            try:
                hdr2[k] = v
            except Exception:
                pass
        w2 = WCS(hdr2, relax=True)
        if getattr(w2, "has_celestial", False):
            return data, header, w2

    raise RuntimeError(f"Reference FITS has no celestial WCS: {ref_fits_path}")


def resolve_inputs(params, result_dir: Path) -> ResolvedInputs:
    result_dir = Path(result_dir)
    output_dir, diagnostics_dir, cache_dir = ensure_output_dirs(result_dir)

    star_table_path = resolve_star_table_path(result_dir)
    if star_table_path is None:
        raise FileNotFoundError(
            "Star table not found. Expected cmd_with_gaia_membership.csv or median_by_ID_filter_wide_cmd.csv"
        )

    ref_meta_path = resolve_ref_meta_path(result_dir)
    if ref_meta_path is None:
        raise FileNotFoundError("ref_build_meta.json not found. Run Step6 first.")
    meta = _read_json(ref_meta_path)
    ref_frame = str(meta.get("ref_frame", "")).strip()
    if not ref_frame:
        raise RuntimeError("ref_build_meta.json missing ref_frame")

    ref_fits_path = _resolve_ref_fits_path(params, result_dir, ref_frame)
    if ref_fits_path is None:
        raise FileNotFoundError(f"Cannot resolve reference FITS path for frame: {ref_frame}")

    crop = None
    crect = crop_rect_path(result_dir)
    if crect.exists():
        crop = _read_json(crect)

    return ResolvedInputs(
        result_dir=result_dir,
        output_dir=output_dir,
        diagnostics_dir=diagnostics_dir,
        cache_dir=cache_dir,
        star_table_path=star_table_path,
        ref_meta_path=ref_meta_path,
        ref_fits_path=ref_fits_path,
        ref_frame=ref_frame,
        crop_rect=crop,
    )


def _pick_first_existing(columns: list[str], candidates: tuple[str, ...]) -> Optional[str]:
    cset = set(columns)
    for c in candidates:
        if c in cset:
            return c
    return None


def _load_master_catalog(result_dir: Path) -> Optional[pd.DataFrame]:
    candidates = [
        step6_dir(result_dir) / "master_catalog.tsv",
        result_dir / "master_catalog.tsv",
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            return pd.read_csv(p, sep="\t")
        except Exception:
            continue
    return None


def _normalize_key_series(series: pd.Series, key: str) -> pd.Series:
    if key in ("source_id", "gaia_source_id", "ID", "star_id"):
        return parse_int64_series(series).astype("Int64")
    return pd.to_numeric(series, errors="coerce")


def _ensure_join_keys_from_master(result_dir: Path, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    master = _load_master_catalog(result_dir)
    if master is None or master.empty:
        return out
    if "ID" not in out.columns or "ID" not in master.columns:
        return out

    merge_cols = ["ID"]
    for c in ("source_id", "gaia_source_id", "star_id"):
        if c in master.columns and c not in merge_cols:
            merge_cols.append(c)
    if len(merge_cols) <= 1:
        return out

    use = master[merge_cols].drop_duplicates(subset=["ID"], keep="first").copy()
    for c in ("ID", "source_id", "gaia_source_id", "star_id"):
        if c in use.columns:
            use[c] = _normalize_key_series(use[c], c)
    try:
        out["ID"] = _normalize_key_series(out["ID"], "ID")
        out = out.merge(use, on="ID", how="left", suffixes=("", "__m"))
    except Exception:
        return out

    for c in ("source_id", "gaia_source_id", "star_id"):
        aux = f"{c}__m"
        if aux not in out.columns:
            continue
        if c not in out.columns:
            out[c] = out[aux]
        else:
            m = out[c].isna()
            out.loc[m, c] = out.loc[m, aux]
        out = out.drop(columns=[aux])
    return out


def _external_membership_tables(result_dir: Path) -> list[tuple[Path, str]]:
    root = Path(result_dir)
    return [
        (root / "step10" / "cmd_with_gaia_membership.csv", ","),
        (root / "cmd_with_gaia_membership.csv", ","),
        (root / "step11" / "cmd_with_gaia_membership.csv", ","),
        (step5_dir(root) / "gaia_derived.csv", ","),
        (root / "gaia_derived.csv", ","),
        (step6_dir(root) / "master_catalog.tsv", "\t"),
        (root / "master_catalog.tsv", "\t"),
    ]


def _pick_membership_col(df: pd.DataFrame) -> Optional[str]:
    return _pick_first_existing(
        list(df.columns),
        ("pmem", "gaia_pmem", "pmem_gaia", "membership_prob_gaia", "membership_prob"),
    )


def _load_external_table(path: Path, sep: str) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, sep=sep)
    except Exception:
        return None


def _fill_membership_from_external(result_dir: Path, df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    out = _ensure_join_keys_from_master(result_dir, df.copy())
    pmem_col0 = _pick_membership_col(out)
    if pmem_col0 is not None:
        pmem = np.clip(pd.to_numeric(out[pmem_col0], errors="coerce").to_numpy(float), 0.0, 1.0)
        source_note = pmem_col0
    else:
        pmem = np.full(len(out), np.nan, dtype=float)
        source_note = "none"

    def finite_count(arr: np.ndarray) -> int:
        return int(np.isfinite(arr).sum())

    key_order = ("source_id", "gaia_source_id", "ID", "star_id")
    for ext_path, sep in _external_membership_tables(result_dir):
        if finite_count(pmem) >= 10:
            break
        ext = _load_external_table(ext_path, sep)
        if ext is None or ext.empty:
            continue
        ext_pmem_col = _pick_membership_col(ext)
        if ext_pmem_col is None:
            continue

        ext_p = np.clip(pd.to_numeric(ext[ext_pmem_col], errors="coerce").to_numpy(float), 0.0, 1.0)
        ext_name = ext_path.name

        for key in key_order:
            if key not in out.columns or key not in ext.columns:
                continue
            left = _normalize_key_series(out[key], key)
            right = _normalize_key_series(ext[key], key)
            use = pd.DataFrame({key: right, "_pmem_ext": ext_p})
            use = use[use[key].notna() & np.isfinite(use["_pmem_ext"])].drop_duplicates(subset=[key], keep="first")
            if use.empty:
                continue
            left_df = pd.DataFrame({"_idx": np.arange(len(out), dtype=int), key: left})
            try:
                merged = left_df.merge(use, on=key, how="left", sort=False)
            except Exception:
                continue
            fill = pd.to_numeric(merged["_pmem_ext"], errors="coerce").to_numpy(float)
            mfill = (~np.isfinite(pmem)) & np.isfinite(fill)
            if np.any(mfill):
                pmem[mfill] = np.clip(fill[mfill], 0.0, 1.0)
                source_note = f"{source_note}+{ext_name}:{key}"
            if finite_count(pmem) >= 10:
                break

    n_finite_before_fill = finite_count(pmem)
    if n_finite_before_fill == 0:
        source_note = f"{source_note}+all_missing"
    elif n_finite_before_fill < len(pmem):
        source_note = f"{source_note}+partial_missing"

    out["pmem"] = np.where(np.isfinite(pmem), np.clip(pmem, 0.0, 1.0), np.nan)
    return out, source_note


def _ensure_ra_dec_from_master(result_dir: Path, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ra_col = _pick_first_existing(list(out.columns), ("ra", "ra_deg", "gaia_ra_deg"))
    dec_col = _pick_first_existing(list(out.columns), ("dec", "dec_deg", "gaia_dec_deg"))
    if (ra_col is not None) and (dec_col is not None):
        return out

    master = _load_master_catalog(result_dir)
    if master is None or master.empty:
        return out

    join_key = None
    if "ID" in out.columns and "ID" in master.columns:
        join_key = "ID"
    elif "source_id" in out.columns and "source_id" in master.columns:
        out["source_id"] = parse_int64_series(out["source_id"]).astype("Int64")
        master["source_id"] = parse_int64_series(master["source_id"]).astype("Int64")
        join_key = "source_id"
    elif "gaia_source_id" in out.columns and "gaia_source_id" in master.columns:
        out["gaia_source_id"] = parse_int64_series(out["gaia_source_id"]).astype("Int64")
        master["gaia_source_id"] = parse_int64_series(master["gaia_source_id"]).astype("Int64")
        join_key = "gaia_source_id"
    if join_key is None:
        return out

    ra_m = _pick_first_existing(list(master.columns), ("ra_deg", "ra", "gaia_ra_deg"))
    dec_m = _pick_first_existing(list(master.columns), ("dec_deg", "dec", "gaia_dec_deg"))
    if ra_m is None or dec_m is None:
        return out

    keep = [join_key, ra_m, dec_m]
    use = master[keep].drop_duplicates(subset=[join_key], keep="first").copy()
    use = use.rename(columns={ra_m: "ra__master", dec_m: "dec__master"})
    try:
        out = out.merge(use, on=join_key, how="left")
    except Exception:
        return df

    if "ra" not in out.columns:
        out["ra"] = out["ra__master"]
    else:
        m = out["ra"].isna()
        out.loc[m, "ra"] = out.loc[m, "ra__master"]
    if "dec" not in out.columns:
        out["dec"] = out["dec__master"]
    else:
        m = out["dec"].isna()
        out.loc[m, "dec"] = out.loc[m, "dec__master"]

    out = out.drop(columns=[c for c in ("ra__master", "dec__master") if c in out.columns])
    return out


def load_star_table(result_dir: Path, star_table_path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(star_table_path)
    if df.empty:
        raise RuntimeError(f"Star table is empty: {star_table_path}")

    df = _ensure_ra_dec_from_master(result_dir, df)
    df, pmem_source = _fill_membership_from_external(result_dir, df)

    cols = list(df.columns)
    id_col = _pick_first_existing(cols, ("star_id", "ID", "source_id", "gaia_source_id"))
    pmem_col = _pick_membership_col(df)

    ra_col = _pick_first_existing(cols, ("ra", "ra_deg", "gaia_ra_deg"))
    dec_col = _pick_first_existing(cols, ("dec", "dec_deg", "gaia_dec_deg"))

    ruwe_col = _pick_first_existing(cols, ("ruwe",))
    mag_col = _pick_first_existing(
        cols,
        (
            "mag_r",
            "mag_g",
            "mag_std_r",
            "mag_std_g",
            "gaia_G",
            "phot_g_mean_mag",
        ),
    )
    mag_err_col = _pick_first_existing(
        cols,
        (
            "mag_err_r",
            "mag_err_g",
            "mag_err",
            "err_r",
            "err_g",
            "mag_error",
            "magerr",
            "phot_g_mean_mag_error",
        ),
    )

    if id_col is None:
        raise RuntimeError("Star table missing ID-like column (star_id/ID/source_id/gaia_source_id)")
    if ra_col is None or dec_col is None:
        raise RuntimeError("Star table missing RA/Dec columns")

    out = pd.DataFrame(index=df.index)
    sid = parse_int64_series(df[id_col])
    if sid.notna().any():
        out["star_id"] = sid.astype("Int64")
    else:
        out["star_id"] = pd.to_numeric(df[id_col], errors="coerce")
    out["ra"] = pd.to_numeric(df[ra_col], errors="coerce")
    out["dec"] = pd.to_numeric(df[dec_col], errors="coerce")
    if "pmem" in df.columns:
        out["pmem"] = np.clip(pd.to_numeric(df["pmem"], errors="coerce"), 0.0, 1.0)
    elif pmem_col is not None:
        out["pmem"] = np.clip(pd.to_numeric(df[pmem_col], errors="coerce"), 0.0, 1.0)
    else:
        out["pmem"] = np.nan
    out["ruwe"] = pd.to_numeric(df[ruwe_col], errors="coerce") if ruwe_col else np.nan
    out["mag"] = pd.to_numeric(df[mag_col], errors="coerce") if mag_col else np.nan
    out["mag_err"] = pd.to_numeric(df[mag_err_col], errors="coerce") if mag_err_col else np.nan

    # Keep potential join keys for later outputs/inspection.
    for col in ("ID", "source_id", "gaia_source_id"):
        if col in df.columns:
            out[col] = df[col]

    valid = np.isfinite(out["ra"].to_numpy(float)) & np.isfinite(out["dec"].to_numpy(float))
    out = out.loc[valid].copy()
    if out.empty:
        raise RuntimeError("No rows with finite RA/Dec after loading star table")

    pmem_vals = pd.to_numeric(out["pmem"], errors="coerce").to_numpy(float)
    pmem_finite = np.isfinite(pmem_vals)
    pmem_source_s = str(pmem_source or "")
    if "all_missing" in pmem_source_s:
        pmem_fallback_mode = "all_missing"
    elif "partial_missing" in pmem_source_s:
        pmem_fallback_mode = "partial_missing"
    else:
        pmem_fallback_mode = "none"

    meta = {
        "id_col": id_col,
        "pmem_col": "pmem" if "pmem" in df.columns else pmem_col,
        "pmem_source": pmem_source,
        "pmem_fallback_mode": pmem_fallback_mode,
        "pmem_fallback_used": pmem_fallback_mode != "none",
        "pmem_finite_fraction": float(np.mean(pmem_finite)) if len(pmem_finite) else 0.0,
        "pmem_all_unity": bool(np.allclose(pmem_vals[pmem_finite], 1.0)) if np.any(pmem_finite) else False,
        "ra_col": ra_col,
        "dec_col": dec_col,
        "ruwe_col": ruwe_col,
        "mag_col": mag_col,
        "mag_err_col": mag_err_col,
        "n_rows": int(len(out)),
    }
    return out.reset_index(drop=True), meta
