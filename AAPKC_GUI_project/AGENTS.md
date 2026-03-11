# AGENTS.md (AAPKC Runtime Rules)

## 1) Core Principle
- Prioritize correct baseline behavior over extra features.
- Do not add speculative UI/options unless explicitly requested.
- When in doubt: simplify logic, keep deterministic behavior, and log clearly.

## 2) Editing Rules (Always)
- Keep responses and patches concise.
- Show only changed paths/blocks; do not reprint full files unless requested.
- Edit only files required for the task; avoid unrelated refactors.
- Do not modify generated outputs or large data artifacts.
- Run minimal checks for touched code (`py_compile` at minimum when feasible).
- If checks are skipped/failed, report reason and impact.

## 3) Pipeline Context (Current)
- Detection-first pipeline:
  Step4 Detection -> Step5 Aperture -> Step6 PSF(optional) -> Step7 RefBuild ->
  Step8 IDMatch -> Step9 Master ID Editor -> Step10 WCS+Gaia ->
  Step11 Zeropoint -> Step12 CMD -> Step13 Isochrone.
- Key IDs:
  - `det_uid`: per-frame local detection ID (Step4-8)
  - `master_id`: global star ID (Step7+)
  - `source_id`: Gaia int64 ID (Step10+)

## 4) Implementation Guardrails
- Keep fallback behavior explicit in logs; never silently inject critical science values.
- Prefer robust two-phase matching/fit logic over aggressive one-shot heuristics.
- Cache for speed, but avoid cache-dependent correctness.
- For heavy steps, include progress/wait logs so long runs are observable.
- Avoid broad parameter proliferation; expose only controls with clear operational value.

## 5) Retrieval/Scan Defaults
- Use `rg --files` and `rg` first.
- Read only files needed for current step.
- Do not recursively scan by default:
  - `data/`, `raw/`, `stack/`, `exports/`, `<data_dir>/result/`
  - `*.fits`, `*.fit`, `*.fts`
  - large `*.log`, large `*.csv`, large notebooks

## 6) Reporting Limits
- Log quoting: max 80 lines, then summarize.
- CSV/table preview: top 20 rows + mean/std/quantiles.

## 7) Task Notes
- Keep short working notes while editing:
  `goal`, `target_modules`, `temporary_debug_flags`, `done_criteria`.
- Remove temporary flags/rules after task completion.
