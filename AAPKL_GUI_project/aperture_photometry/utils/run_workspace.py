from __future__ import annotations

import json
import os
import re
import csv
from pathlib import Path

from .step_paths import step1_dir


_DATE_TOKEN_RE = re.compile(r"(20\d{6})")


def _date_tokens_from_name(name: str) -> list[str]:
    return _DATE_TOKEN_RE.findall(str(name or ""))


def _strip_date_tokens(name: str) -> str:
    parts = [p for p in re.split(r"[_\-\s]+", str(name or "")) if p]
    kept = [p for p in parts if not _DATE_TOKEN_RE.fullmatch(p)]
    return "_".join(kept) if kept else str(name or "").strip()


def infer_run_label(root_dir: Path, input_dirs: list[Path] | None = None) -> str:
    dirs = [Path(p) for p in (input_dirs or []) if p]
    if dirs:
        labels: list[str] = []
        seen: set[str] = set()
        for path in dirs:
            label = _strip_date_tokens(path.name)
            key = label.lower()
            if label and key not in seen:
                labels.append(label)
                seen.add(key)
        if len(labels) == 1:
            return labels[0]
    root_label = _strip_date_tokens(root_dir.name)
    return root_label or root_dir.name or "workspace"


def load_run_manifest(result_dir: Path) -> dict:
    path = Path(result_dir) / "run_manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _headers_dates(result_dir: Path) -> list[str]:
    headers_path = step1_dir(result_dir) / "headers.csv"
    if not headers_path.exists():
        return []
    dates: list[str] = []
    try:
        with headers_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = str(row.get("Filename") or "")
                dates.extend(_date_tokens_from_name(filename))
                date_obs = str(row.get("DATE-OBS") or "").strip()
                if len(date_obs) >= 10:
                    token = date_obs[:10].replace("-", "")
                    if _DATE_TOKEN_RE.fullmatch(token):
                        dates.append(token)
    except Exception:
        return []
    return sorted(set(dates))


def infer_workspace_label(result_dir: Path) -> str:
    meta = load_run_manifest(result_dir)
    label = str(meta.get("label") or "").strip()
    if label:
        return label

    name = result_dir.name
    if name.startswith("RESULT_"):
        name = name[len("RESULT_"):]
    elif name.startswith("MERGED_"):
        name = name[len("MERGED_"):]
    label = _strip_date_tokens(name)
    if label and label.lower() != "result" and label.lower() != "merged":
        return label

    parent_label = _strip_date_tokens(result_dir.parent.name)
    if parent_label and parent_label.lower() not in {"result", "merged_result", "merged_results"}:
        return parent_label

    grand_label = _strip_date_tokens(result_dir.parent.parent.name) if result_dir.parent.parent != result_dir.parent else ""
    return grand_label or result_dir.parent.name or result_dir.name


def infer_workspace_date_range(result_dir: Path) -> tuple[str | None, str | None]:
    meta = load_run_manifest(result_dir)
    start_date = str(meta.get("date_start") or "").replace("-", "")
    end_date = str(meta.get("date_end") or "").replace("-", "")
    if _DATE_TOKEN_RE.fullmatch(start_date) and _DATE_TOKEN_RE.fullmatch(end_date):
        return start_date, end_date

    dates = _headers_dates(result_dir)
    if not dates:
        dates.extend(_date_tokens_from_name(result_dir.name))
    if not dates:
        dates.extend(_date_tokens_from_name(result_dir.parent.name))
    if not dates:
        return None, None
    dates = sorted(set(dates))
    return dates[0], dates[-1]


def infer_run_date_range(root_dir: Path, input_dirs: list[Path] | None = None) -> tuple[str | None, str | None]:
    dirs = [Path(p) for p in (input_dirs or []) if p]
    dates: list[str] = []
    if dirs:
        for path in dirs:
            dates.extend(_date_tokens_from_name(path.name))
    else:
        dates.extend(_date_tokens_from_name(root_dir.name))

    if not dates:
        return None, None
    dates = sorted(set(dates))
    return dates[0], dates[-1]


def infer_run_parent_dir(root_dir: Path, input_dirs: list[Path] | None = None) -> Path:
    dirs = [Path(p) for p in (input_dirs or []) if p]
    if dirs:
        if len(dirs) == 1:
            return dirs[0].parent
        try:
            return Path(os.path.commonpath([str(p) for p in dirs]))
        except ValueError:
            # Mixed-drive Windows inputs have no commonpath; fall back to the first input parent.
            return dirs[0].parent

    if _date_tokens_from_name(root_dir.name):
        return root_dir.parent
    return root_dir


def build_result_workspace_dir(root_dir: Path, input_dirs: list[Path] | None = None) -> Path:
    parent = infer_run_parent_dir(root_dir, input_dirs)
    label = infer_run_label(root_dir, input_dirs)
    start_date, end_date = infer_run_date_range(root_dir, input_dirs)

    if start_date and end_date:
        date_part = start_date if start_date == end_date else f"{start_date}_{end_date}"
        name = f"RESULT_{label}_{date_part}"
    else:
        name = f"RESULT_{label}"
    return parent / name


def infer_result_workspace_label(result_dirs: list[Path]) -> str:
    labels: list[str] = []
    seen: set[str] = set()
    for path in [Path(p) for p in result_dirs if p]:
        label = infer_workspace_label(path)
        key = label.lower()
        if label and key not in seen:
            labels.append(label)
            seen.add(key)
    if len(labels) == 1:
        return labels[0]
    return labels[0] if labels else "workspace"


def infer_result_workspace_date_range(result_dirs: list[Path]) -> tuple[str | None, str | None]:
    dates: list[str] = []
    for path in [Path(p) for p in result_dirs if p]:
        start_date, end_date = infer_workspace_date_range(path)
        if start_date:
            dates.append(start_date)
        if end_date:
            dates.append(end_date)
    if not dates:
        return None, None
    dates = sorted(set(dates))
    return dates[0], dates[-1]


def build_merged_workspace_dir(result_dirs: list[Path]) -> Path:
    dirs = [Path(p) for p in result_dirs if p]
    if not dirs:
        return Path("MERGED_workspace")
    try:
        parent = Path(os.path.commonpath([str(p.parent) for p in dirs]))
    except ValueError:
        parent = dirs[0].parent
    label = infer_result_workspace_label(dirs)
    start_date, end_date = infer_result_workspace_date_range(dirs)
    if start_date and end_date:
        date_part = start_date if start_date == end_date else f"{start_date}_{end_date}"
        return parent / f"MERGED_{label}_{date_part}"
    return parent / f"MERGED_{label}"


def write_run_manifest(
    result_dir: Path,
    *,
    run_type: str,
    root_dir: Path,
    input_dirs: list[Path] | None = None,
    input_result_dirs: list[Path] | None = None,
    target_name: str | None = None,
    storage_mode: str | None = None,
) -> None:
    dirs = [Path(p) for p in (input_dirs or []) if p]
    result_dirs = [Path(p) for p in (input_result_dirs or []) if p]
    start_date, end_date = infer_run_date_range(root_dir, dirs)
    if (not start_date or not end_date) and result_dirs:
        start_date, end_date = infer_result_workspace_date_range(result_dirs)
    manifest = {
        "run_type": str(run_type),
        "label": target_name or infer_run_label(root_dir, dirs),
        "root_dir": str(root_dir),
        "input_data_dirs": [str(p) for p in dirs] if dirs else [str(root_dir)],
        "input_result_dirs": [str(p) for p in result_dirs],
        "date_start": start_date,
        "date_end": end_date,
        "result_dir": str(result_dir),
        "storage_mode": str(storage_mode or "full"),
    }
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    (Path(result_dir) / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
