from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from phipml.io.data_handler import (
    Config,
    FeatureManager,
    MetadataHandler,
    OligosHandler,
)


def _write_sample_file(
    path: Path,
    peptide_ids: list[str],
    *,
    peptide_column: str = "ID",
) -> None:
    pd.DataFrame(
        {
            peptide_column: peptide_ids,
            "fold_change": np.arange(1, len(peptide_ids) + 1, dtype=float),
            "neglogp": 2.0,
            "padj": 0.01,
            "input": 10,
            "count": 5,
        }
    ).to_csv(path, index=False)


def _write_metadata(tmp_path: Path, sample_names: list[str]) -> None:
    groups = [
        "Control" if index % 2 == 0 else "Case" for index in range(len(sample_names))
    ]
    pd.DataFrame(
        {
            "SampleName": sample_names,
            "group_test": groups,
        }
    ).to_csv(tmp_path / "metadata.csv", index=False)


def _write_config(tmp_path: Path, **updates: object) -> Path:
    values: dict[str, object] = {
        "metadata_input": "metadata.csv",
        "data_input": "ABS",
        "group_tests": ["Control", "Case"],
    }
    values.update(updates)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(values), encoding="utf-8")
    return config_path


def test_directory_of_sample_files_builds_presence_absence_matrix(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "ABS"
    data_dir.mkdir()
    sample_1 = "R52P01_30_EC86_J25_F12_P3_A_T_C2"
    sample_2 = "R52P01_31_EC86_J25_G12_P3_A_T_C2"
    _write_sample_file(
        data_dir / f"{sample_1}.csv",
        ["agilent_100286", "twist_200001", "agilent_100286"],
    )
    _write_sample_file(
        data_dir / f"{sample_2}.csv",
        ["agilent_100286", "corona2_300001"],
    )
    _write_sample_file(data_dir / "ignore_me.csv", ["agilent_ignore"])

    pd.DataFrame(
        {
            "SampleName": [sample_1, sample_2],
            "group_test": ["Control", "Case"],
        }
    ).to_csv(tmp_path / "metadata.csv", index=False)

    config = Config(
        _write_config(
            tmp_path,
            sample_file_patterns=["R52P01_*.csv"],
            transposed=True,
        )
    )
    oligos = OligosHandler(config).get_oligos_df()

    assert config.get_data_input_mode() == "sample-files"
    assert oligos.index.tolist() == [sample_1, sample_2]
    assert oligos.columns.tolist() == [
        "agilent_100286",
        "corona2_300001",
        "twist_200001",
    ]
    assert oligos.loc[sample_1].tolist() == [1, 0, 1]
    assert oligos.loc[sample_2].tolist() == [1, 1, 0]
    assert all(dtype == np.dtype("uint8") for dtype in oligos.dtypes)

    manager = FeatureManager(
        config,
        MetadataHandler(config),
        OligosHandler(config),
        with_oligos=True,
    )
    features, target = manager.get_features_target()
    pd.testing.assert_frame_equal(features, oligos)
    assert target.tolist() == [0, 1]


def test_manifest_accepts_explicit_sample_names_and_relative_paths(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "ABS"
    data_dir.mkdir()
    _write_sample_file(data_dir / "raw_a.csv", ["agilent_a"])
    _write_sample_file(data_dir / "raw_b.csv", ["agilent_b"])
    (tmp_path / "samplefiles.txt").write_text(
        "sample_name\tpath\n" "Sample-A\tABS/raw_a.csv\n" "Sample-B\tABS/raw_b.csv\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "SampleName": ["Sample-A", "Sample-B"],
            "group_test": ["Control", "Case"],
        }
    ).to_csv(tmp_path / "metadata.csv", index=False)

    config = Config(
        _write_config(
            tmp_path,
            data_input="samplefiles.txt",
            data_input_mode="sample-files",
        )
    )
    oligos = OligosHandler(config).get_oligos_df()

    assert oligos.index.tolist() == ["Sample-A", "Sample-B"]
    assert oligos.columns.tolist() == ["agilent_a", "agilent_b"]
    assert oligos.to_numpy().tolist() == [[1, 0], [0, 1]]


def test_manifest_paths_default_to_filename_stems(tmp_path: Path) -> None:
    data_dir = tmp_path / "ABS"
    data_dir.mkdir()
    _write_sample_file(data_dir / "prefix_S01_enriched.csv", ["agilent_a"])
    _write_sample_file(data_dir / "prefix_S02_enriched.csv", ["agilent_b"])
    (tmp_path / "samplefiles.txt").write_text(
        "ABS/prefix_S01_enriched.csv\nABS/prefix_S02_enriched.csv\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "SampleName": ["S01", "S02"],
            "group_test": ["Control", "Case"],
        }
    ).to_csv(tmp_path / "metadata.csv", index=False)

    config = Config(
        _write_config(
            tmp_path,
            data_input="samplefiles.txt",
            data_input_mode="sample-files",
            sample_name_regex=r"prefix_(?P<sample>.+)_enriched",
        )
    )

    assert OligosHandler(config).get_oligos_df().index.tolist() == ["S01", "S02"]


def test_sample_file_requires_configured_peptide_column(tmp_path: Path) -> None:
    data_dir = tmp_path / "ABS"
    data_dir.mkdir()
    pd.DataFrame({"peptide": ["agilent_a"]}).to_csv(data_dir / "S01.csv", index=False)
    pd.DataFrame(
        {"SampleName": ["S01", "S02"], "group_test": ["Control", "Case"]}
    ).to_csv(tmp_path / "metadata.csv", index=False)
    config = Config(_write_config(tmp_path))

    with pytest.raises(KeyError, match="Peptide column 'ID' not found"):
        OligosHandler(config).get_oligos_df()


def test_multiple_recursive_patterns_and_custom_peptide_column(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "ABS"
    batch_a = data_dir / "batch_a"
    batch_b = data_dir / "batch_b" / "nested"
    batch_a.mkdir(parents=True)
    batch_b.mkdir(parents=True)
    _write_sample_file(
        batch_a / "S01_hits.csv",
        ["agilent_a"],
        peptide_column="peptide_id",
    )
    _write_sample_file(
        batch_b / "S02_hits.csv",
        ["agilent_b"],
        peptide_column="peptide_id",
    )
    _write_sample_file(
        data_dir / "ignored.csv",
        ["agilent_ignored"],
        peptide_column="peptide_id",
    )
    _write_metadata(tmp_path, ["S01_hits", "S02_hits"])

    config = Config(
        _write_config(
            tmp_path,
            sample_file_patterns=["batch_a/*.csv", "batch_b/**/*.csv"],
            sample_file_peptide_column="peptide_id",
        )
    )
    oligos = OligosHandler(config).get_oligos_df()

    assert oligos.index.tolist() == ["S01_hits", "S02_hits"]
    assert oligos.columns.tolist() == ["agilent_a", "agilent_b"]
    assert oligos.to_numpy().tolist() == [[1, 0], [0, 1]]


def test_empty_sample_file_produces_zero_row_when_other_peptides_exist(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "ABS"
    data_dir.mkdir()
    _write_sample_file(data_dir / "S01.csv", [])
    _write_sample_file(data_dir / "S02.csv", ["agilent_a"])
    _write_metadata(tmp_path, ["S01", "S02"])

    oligos = OligosHandler(Config(_write_config(tmp_path))).get_oligos_df()

    assert oligos.columns.tolist() == ["agilent_a"]
    assert oligos.to_numpy().tolist() == [[0], [1]]


def test_all_empty_sample_files_raise_clear_error(tmp_path: Path) -> None:
    data_dir = tmp_path / "ABS"
    data_dir.mkdir()
    _write_sample_file(data_dir / "S01.csv", [])
    _write_sample_file(data_dir / "S02.csv", [])
    _write_metadata(tmp_path, ["S01", "S02"])

    with pytest.raises(ValueError, match="No peptide IDs were found"):
        OligosHandler(Config(_write_config(tmp_path))).get_oligos_df()


def test_duplicate_explicit_sample_names_are_rejected(tmp_path: Path) -> None:
    data_dir = tmp_path / "ABS"
    data_dir.mkdir()
    _write_sample_file(data_dir / "raw_a.csv", ["agilent_a"])
    _write_sample_file(data_dir / "raw_b.csv", ["agilent_b"])
    (tmp_path / "samplefiles.txt").write_text(
        "Same-Sample\tABS/raw_a.csv\nSame-Sample\tABS/raw_b.csv\n",
        encoding="utf-8",
    )
    _write_metadata(tmp_path, ["Same-Sample", "Other-Sample"])
    config = Config(
        _write_config(
            tmp_path,
            data_input="samplefiles.txt",
            data_input_mode="sample-files",
        )
    )

    with pytest.raises(ValueError, match="Duplicate sample name 'Same-Sample'"):
        OligosHandler(config).get_oligos_df()


def test_missing_manifest_sample_file_is_reported(tmp_path: Path) -> None:
    (tmp_path / "samplefiles.txt").write_text(
        "ABS/does_not_exist.csv\n",
        encoding="utf-8",
    )
    _write_metadata(tmp_path, ["does_not_exist", "S02"])
    config = Config(
        _write_config(
            tmp_path,
            data_input="samplefiles.txt",
            data_input_mode="sample-files",
        )
    )

    with pytest.raises(FileNotFoundError, match="Input file does not exist"):
        OligosHandler(config).get_oligos_df()


def test_directory_with_no_matching_files_is_reported(tmp_path: Path) -> None:
    data_dir = tmp_path / "ABS"
    data_dir.mkdir()
    _write_sample_file(data_dir / "S01.csv", ["agilent_a"])
    _write_metadata(tmp_path, ["S01", "S02"])
    config = Config(
        _write_config(
            tmp_path,
            sample_file_patterns=["R52P01_*.csv"],
        )
    )

    with pytest.raises(FileNotFoundError, match="No sample files"):
        OligosHandler(config).get_oligos_df()


@pytest.mark.parametrize("invalid_mode", ["samples", "directory", "unknown"])
def test_invalid_data_input_mode_is_rejected(
    tmp_path: Path,
    invalid_mode: str,
) -> None:
    data_dir = tmp_path / "ABS"
    data_dir.mkdir()
    _write_metadata(tmp_path, ["S01", "S02"])

    with pytest.raises(ValueError, match="data_input_mode must be"):
        Config(_write_config(tmp_path, data_input_mode=invalid_mode))


def test_invalid_or_nonmatching_sample_name_regex_is_reported(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "ABS"
    data_dir.mkdir()
    _write_sample_file(data_dir / "S01.csv", ["agilent_a"])
    _write_metadata(tmp_path, ["S01", "S02"])

    with pytest.raises(ValueError, match="Invalid sample_name_regex"):
        Config(_write_config(tmp_path, sample_name_regex="("))

    config = Config(
        _write_config(
            tmp_path,
            sample_name_regex=r"prefix_(?P<sample>.+)_enriched",
        )
    )
    with pytest.raises(ValueError, match="does not match sample_name_regex"):
        OligosHandler(config).get_oligos_df()


def test_sample_files_support_library_filters_and_prevalence(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "ABS"
    data_dir.mkdir()
    _write_sample_file(
        data_dir / "S01.csv",
        ["agilent_human_common", "agilent_human_rare", "agilent_bacterial"],
    )
    _write_sample_file(
        data_dir / "S02.csv",
        ["agilent_human_common", "agilent_bacterial"],
    )
    _write_sample_file(
        data_dir / "S03.csv",
        ["agilent_human_common"],
    )
    _write_metadata(tmp_path, ["S01", "S02", "S03"])
    pd.DataFrame(
        {
            "peptide_id": [
                "agilent_human_common",
                "agilent_human_rare",
                "agilent_bacterial",
            ],
            "Species": ["Homo sapiens", "Homo sapiens", "Escherichia coli"],
        }
    ).to_csv(tmp_path / "library.csv", index=False)
    config = Config(
        _write_config(
            tmp_path,
            lib_metadata_input="library.csv",
            lib_col_peptide_name="peptide_id",
        )
    )
    manager = FeatureManager(
        config,
        MetadataHandler(config),
        OligosHandler(config),
        oligo_filters={"Species": "Homo sapiens"},
        with_oligos=True,
        prevalence_threshold_min=50,
        prevalence_threshold_max=100,
    )

    features, target = manager.get_features_target()

    assert features.columns.tolist() == ["agilent_human_common"]
    assert features.iloc[:, 0].tolist() == [1, 1, 1]
    assert target.tolist() == [0, 1, 0]


def test_auto_mode_preserves_combined_matrix_input(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "peptide": ["agilent_a", "agilent_b"],
            "S01": [1, 0],
            "S02": [0, 1],
        }
    ).to_csv(tmp_path / "combined.csv", index=False)
    _write_metadata(tmp_path, ["S01", "S02"])
    config = Config(
        _write_config(
            tmp_path,
            data_input="combined.csv",
            data_input_mode="auto",
            transposed=True,
        )
    )

    oligos = OligosHandler(config).get_oligos_df()

    assert config.get_data_input_mode() == "matrix"
    assert oligos.index.tolist() == ["S01", "S02"]
    assert oligos.columns.tolist() == ["agilent_a", "agilent_b"]
    assert oligos.to_numpy().tolist() == [[1, 0], [0, 1]]
