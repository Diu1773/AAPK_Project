from pathlib import Path

from aperture_photometry.utils.photometry_loader import load_frame_photometry


def test_load_frame_photometry_enriches_ids_from_csv_idmatch(tmp_path: Path) -> None:
    result_dir = tmp_path / "result"
    step5_dir = result_dir / "step5_photometry"
    step8_dir = result_dir / "step8_idmatch" / "20250429"
    step9_dir = result_dir / "step9_selection"

    step5_dir.mkdir(parents=True)
    step8_dir.mkdir(parents=True)
    step9_dir.mkdir(parents=True)

    fname = "frame_20250429.fit"

    (step5_dir / f"{fname}_photometry.tsv").write_text(
        "\n".join(
            [
                "det_uid\tFILTER\tmag\tmag_err",
                "0\tr\t12.3\t0.01",
                "1\tr\t13.4\t0.02",
            ]
        ),
        encoding="utf-8",
    )

    (step8_dir / f"idmatch_{fname}.csv").write_text(
        "\n".join(
            [
                "det_idx,x,y,source_id,file,filter",
                "0,100.0,200.0,123456789012345678,frame_20250429.fit,r",
                "1,101.0,201.0,223456789012345678,frame_20250429.fit,r",
            ]
        ),
        encoding="utf-8",
    )

    (step9_dir / "id_mapping_r.csv").write_text(
        "\n".join(
            [
                "ID,source_id,role,x_ref,y_ref",
                "11,123456789012345678,T,100.0,200.0",
                "22,223456789012345678,C,101.0,201.0",
            ]
        ),
        encoding="utf-8",
    )

    df = load_frame_photometry(result_dir, fname, "r")

    assert df is not None
    assert df["source_id"].tolist() == [123456789012345678, 223456789012345678]
    assert df["ID"].tolist() == [11, 22]
