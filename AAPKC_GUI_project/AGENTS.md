# AGENTS.md (Lean Runtime Rules)

## Immutable Rules (Always Apply)
- Keep responses concise by default.
- For code changes, prefer unified diff/patch style output.
- Never reprint full files unless explicitly requested.
- Show only changed functions/blocks and affected paths.
- Edit only files required for the task; avoid unrelated refactors.
- Do not modify generated outputs or large data artifacts.
- Run minimal checks for touched code when feasible.
- If checks are skipped or fail, report reason and impact.
- Log quoting limit: max 80 lines, then summarize.
- CSV/table output: top 20 rows + mean/std/quantiles.

## Situational Rules (Task-Specific, Mutable)
- Maintain a short task context in working notes: `goal`, `target_modules`, `temporary_debug_flags`, `done_criteria`.
- Remove temporary flags/rules after task completion.

## Do-Not-Read/Scan by Default (Unless Explicitly Mentioned)
- `data/`, `raw/`, `stack/`, `exports/`, `<data_dir>/result/`
- `*.fits`, `*.fit`, `*.fts`
- large `*.log`, large `*.csv`, large notebooks
- Do not recursively scan excluded paths.

## Retrieval and Editing Defaults
- Use `rg --files` and `rg` for discovery/search first.
- Read only files needed for the current step.
- Keep this file short; move detailed policies to `docs/dev_rules.md` when needed.
