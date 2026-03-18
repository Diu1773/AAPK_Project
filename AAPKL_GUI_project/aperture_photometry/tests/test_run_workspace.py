from pathlib import Path

from aperture_photometry.utils.run_workspace import (
    build_merged_workspace_dir,
    build_result_workspace_dir,
    infer_workspace_date_range,
    infer_workspace_label,
    write_run_manifest,
)
from aperture_photometry.utils.io_utils import load_file_path_map


def test_build_result_workspace_dir_from_multi_night_input_dirs(tmp_path: Path) -> None:
    root = tmp_path / "YZBOOTIS"
    inputs = [
        root / "YZBOOTIS_20250429",
        root / "YZBOOTIS_20250430",
    ]

    out_dir = build_result_workspace_dir(root, inputs)

    assert out_dir == root / "RESULT_YZBOOTIS_20250429_20250430"


def test_build_merged_workspace_dir_from_result_workspaces(tmp_path: Path) -> None:
    root = tmp_path / "YZBOOTIS"
    inputs = [
        root / "RESULT_YZBOOTIS_20250429_20250430",
        root / "RESULT_YZBOOTIS_20250503",
    ]

    for path in inputs:
        write_run_manifest(
            path,
            run_type="result",
            root_dir=root,
            input_dirs=[],
            target_name="YZBOOTIS",
        )
    # overwrite date range to mimic real runs
    (inputs[0] / "run_manifest.json").write_text(
        f"""{{
  "run_type": "result",
  "label": "YZBOOTIS",
  "root_dir": "{root}",
  "input_data_dirs": [],
  "input_result_dirs": [],
  "date_start": "20250429",
  "date_end": "20250430",
  "result_dir": "{inputs[0]}",
  "storage_mode": "full"
}}""",
        encoding="utf-8",
    )
    (inputs[1] / "run_manifest.json").write_text(
        f"""{{
  "run_type": "result",
  "label": "YZBOOTIS",
  "root_dir": "{root}",
  "input_data_dirs": [],
  "input_result_dirs": [],
  "date_start": "20250503",
  "date_end": "20250503",
  "result_dir": "{inputs[1]}",
  "storage_mode": "full"
}}""",
        encoding="utf-8",
    )

    out_dir = build_merged_workspace_dir(inputs)

    assert out_dir == root / "MERGED_YZBOOTIS_20250429_20250503"


def test_infer_workspace_label_and_dates_from_legacy_result_headers(tmp_path: Path) -> None:
    result_dir = tmp_path / "YZBOOTIS" / "result"
    step1_dir = result_dir / "step1_file_selection"
    step1_dir.mkdir(parents=True)

    (step1_dir / "headers.csv").write_text(
        "\n".join(
            [
                "Filename,DATE-OBS,FILTER",
                "YZBOOTIS_20250429__a.fit,2025-04-29T10:11:12,r",
                "YZBOOTIS_20250507__b.fit,2025-05-07T09:08:07,g",
            ]
        ),
        encoding="utf-8",
    )

    assert infer_workspace_label(result_dir) == "YZBOOTIS"
    assert infer_workspace_date_range(result_dir) == ("20250429", "20250507")


def test_load_file_path_map_from_step1_dir(tmp_path: Path) -> None:
    result_dir = tmp_path / "RESULT_YZBOOTIS_20250429"
    step1_dir = result_dir / "step1_file_selection"
    step1_dir.mkdir(parents=True)
    (step1_dir / "file_path_map.json").write_text(
        '{"A.fit": "E:/obs/YZBOOTIS_20250429/A.fit", "B.fit": "E:/obs/YZBOOTIS_20250429/B.fit"}',
        encoding="utf-8",
    )

    assert load_file_path_map(result_dir) == {
        "A.fit": "E:/obs/YZBOOTIS_20250429/A.fit",
        "B.fit": "E:/obs/YZBOOTIS_20250429/B.fit",
    }


def test_load_file_path_map_falls_back_to_project_state(tmp_path: Path) -> None:
    result_dir = tmp_path / "RESULT_YZBOOTIS_20250429"
    result_dir.mkdir(parents=True)
    (result_dir / "project_state.json").write_text(
        """
        {
          "step_data": {
            "file_selection": {
              "file_path_map": {
                "C.fit": "E:/obs/YZBOOTIS_20250429/C.fit"
              }
            }
          }
        }
        """,
        encoding="utf-8",
    )

    assert load_file_path_map(result_dir) == {
        "C.fit": "E:/obs/YZBOOTIS_20250429/C.fit",
    }
