from __future__ import annotations

import os
from pathlib import Path
from typing import Any

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


def _write_config(tmp_path: Path, **updates: object) -> Path:
    values: dict[str, object] = {
        "metadata_input": "metadata.csv",
        "data_input": "matrix.csv",
        "group_tests": ["Control", "Case"],
        "transposed": True,
    }
    values.update(updates)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    return path


def _write_metadata(
    tmp_path: Path,
    sample_names: list[str] | None = None,
    targets: list[object] | None = None,
    **extra_columns: list[object],
) -> None:
    names = sample_names or ["S1", "S2", "S3"]
    target_values = targets or ["Control", "Case", "Case"]
    values: dict[str, list[object]] = {
        "SampleName": names,
        "group_test": target_values,
    }
    values.update(extra_columns)
    pd.DataFrame(values).to_csv(tmp_path / "metadata.csv", index=False)


def _write_matrix(
    path: Path,
    *,
    sep: str = ",",
    sample_names: list[str] | None = None,
) -> None:
    names = sample_names or ["S1", "S2", "S3"]
    matrix = pd.DataFrame(
        {
            "peptide": ["agilent_p1", "agilent_p2", "twist_p3"],
            names[0]: [1, 0, 0],
            names[1]: [0, 1, 0],
            names[2]: [1, 1, 1],
        }
    )
    matrix.to_csv(path, index=False, sep=sep)


def _write_sample_table(
    path: Path,
    peptide_ids: list[object],
    *,
    peptide_column: str = "ID",
    sep: str = ",",
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
    ).to_csv(path, index=False, sep=sep)


def _base_project(tmp_path: Path) -> Path:
    _write_metadata(tmp_path)
    _write_matrix(tmp_path / "matrix.csv")
    return tmp_path


# ---------------------------------------------------------------------------
# File formats and invalid file contents
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("suffix", [".tsv", ".txt"])
def test_combined_tab_delimited_matrix_formats(
    tmp_path: Path,
    suffix: str,
) -> None:
    _write_metadata(tmp_path)
    path = tmp_path / f"matrix{suffix}"
    _write_matrix(path, sep="\t")
    config = Config(_write_config(tmp_path, data_input=path.name))

    oligos = OligosHandler(config).get_oligos_df()

    assert oligos.index.tolist() == ["S1", "S2", "S3"]
    assert oligos.columns.tolist() == ["agilent_p1", "agilent_p2", "twist_p3"]
    assert oligos.to_numpy().tolist() == [[1, 0, 0], [0, 1, 0], [1, 1, 1]]


def test_excel_metadata_input(tmp_path: Path) -> None:
    pytest.importorskip("openpyxl")
    _write_matrix(tmp_path / "matrix.csv")
    metadata = pd.DataFrame(
        {
            "SampleName": ["S1", "S2", "S3"],
            "group_test": ["Control", "Case", "Case"],
            "Age": [40, 50, 60],
        }
    )
    metadata.to_excel(tmp_path / "metadata.xlsx", index=False)
    config = Config(_write_config(tmp_path, metadata_input="metadata.xlsx"))

    loaded = MetadataHandler(config).get_individuals_metadata_df()

    assert loaded.index.tolist() == ["S1", "S2", "S3"]
    assert loaded["Age"].tolist() == [40, 50, 60]


def test_excel_library_metadata_input(tmp_path: Path) -> None:
    pytest.importorskip("openpyxl")
    _base_project(tmp_path)
    library = pd.DataFrame(
        {
            "peptide_id": ["agilent_p1", "agilent_p2", "twist_p3"],
            "Species": ["Human", "Bacterial", "Human"],
        }
    )
    library.to_excel(tmp_path / "library.xlsx", index=False)
    config = Config(
        _write_config(
            tmp_path,
            lib_metadata_input="library.xlsx",
            lib_col_peptide_name="peptide_id",
        )
    )

    loaded = OligosHandler(config).get_oligos_metadata_df()

    assert loaded.index.tolist() == ["agilent_p1", "agilent_p2", "twist_p3"]
    assert loaded["Species"].tolist() == ["Human", "Bacterial", "Human"]


def test_per_sample_tsv_input(tmp_path: Path) -> None:
    data_dir = tmp_path / "ABS"
    data_dir.mkdir()
    _write_sample_table(data_dir / "S1.tsv", ["agilent_p1"], sep="\t")
    _write_sample_table(data_dir / "S2.tsv", ["agilent_p2"], sep="\t")
    _write_metadata(tmp_path, ["S1", "S2"], ["Control", "Case"])
    config = Config(
        _write_config(
            tmp_path,
            data_input="ABS",
            data_input_mode="sample-files",
            sample_file_patterns=["*.tsv"],
        )
    )

    oligos = OligosHandler(config).get_oligos_df()

    assert oligos.index.tolist() == ["S1", "S2"]
    assert oligos.to_numpy().tolist() == [[1, 0], [0, 1]]


def test_library_pickle_must_contain_dataframe(tmp_path: Path) -> None:
    _base_project(tmp_path)
    pd.to_pickle({"not": "a dataframe"}, tmp_path / "library.pkl")
    config = Config(_write_config(tmp_path, lib_metadata_input="library.pkl"))

    with pytest.raises(TypeError, match="pickle must contain a pandas DataFrame"):
        OligosHandler(config).get_oligos_metadata_df()


@pytest.mark.parametrize("invalid_id", [None, "", "   "])
def test_sample_file_rejects_missing_or_empty_peptide_ids(
    tmp_path: Path,
    invalid_id: object,
) -> None:
    data_dir = tmp_path / "ABS"
    data_dir.mkdir()
    _write_sample_table(data_dir / "S1.csv", ["agilent_p1", invalid_id])
    _write_metadata(tmp_path, ["S1", "S2"], ["Control", "Case"])
    config = Config(_write_config(tmp_path, data_input="ABS"))

    with pytest.raises(ValueError, match="Missing peptide IDs|Empty peptide IDs"):
        OligosHandler(config).get_oligos_df()


def test_sample_file_rejects_unsupported_extension(tmp_path: Path) -> None:
    data_dir = tmp_path / "ABS"
    data_dir.mkdir()
    (data_dir / "S1.json").write_text('{"ID": "agilent_p1"}', encoding="utf-8")
    _write_metadata(tmp_path, ["S1", "S2"], ["Control", "Case"])
    config = Config(
        _write_config(
            tmp_path,
            data_input="ABS",
            data_input_mode="sample-files",
            sample_file_patterns=["*.json"],
        )
    )

    with pytest.raises(ValueError, match="Unsupported per-sample table format"):
        OligosHandler(config).get_oligos_df()


# ---------------------------------------------------------------------------
# Manifest formats and errors
# ---------------------------------------------------------------------------


def test_empty_manifest_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "samples.txt").write_text("\n# no files\n", encoding="utf-8")
    _write_metadata(tmp_path, ["S1", "S2"], ["Control", "Case"])
    config = Config(
        _write_config(
            tmp_path,
            data_input="samples.txt",
            data_input_mode="sample-files",
        )
    )

    with pytest.raises(ValueError, match="manifest is empty"):
        OligosHandler(config).get_oligos_df()


@pytest.mark.parametrize("header", ["path", "file", "file_path"])
def test_manifest_one_column_headers_comments_and_blank_lines(
    tmp_path: Path,
    header: str,
) -> None:
    data_dir = tmp_path / "ABS"
    data_dir.mkdir()
    _write_sample_table(data_dir / "S1.csv", ["agilent_p1"])
    _write_sample_table(data_dir / "S2.csv", ["agilent_p2"])
    (tmp_path / "samples.txt").write_text(
        f"# generated manifest\n\n{header}\nABS/S1.csv\nABS/S2.csv\n",
        encoding="utf-8",
    )
    _write_metadata(tmp_path, ["S1", "S2"], ["Control", "Case"])
    config = Config(
        _write_config(
            tmp_path,
            data_input="samples.txt",
            data_input_mode="sample-files",
        )
    )

    oligos = OligosHandler(config).get_oligos_df()

    assert oligos.index.tolist() == ["S1", "S2"]
    assert oligos.to_numpy().tolist() == [[1, 0], [0, 1]]


def test_manifest_accepts_absolute_paths(tmp_path: Path) -> None:
    data_dir = tmp_path / "ABS"
    data_dir.mkdir()
    sample_1 = data_dir / "S1.csv"
    sample_2 = data_dir / "S2.csv"
    _write_sample_table(sample_1, ["agilent_p1"])
    _write_sample_table(sample_2, ["agilent_p2"])
    (tmp_path / "samples.txt").write_text(
        f"{sample_1.resolve()}\n{sample_2.resolve()}\n",
        encoding="utf-8",
    )
    _write_metadata(tmp_path, ["S1", "S2"], ["Control", "Case"])
    config = Config(
        _write_config(
            tmp_path,
            data_input="samples.txt",
            data_input_mode="sample-files",
        )
    )

    assert OligosHandler(config).get_oligos_df().index.tolist() == ["S1", "S2"]


def test_manifest_rejects_malformed_rows(tmp_path: Path) -> None:
    (tmp_path / "samples.txt").write_text(
        "S1\tABS/S1.csv\textra-field\n",
        encoding="utf-8",
    )
    _write_metadata(tmp_path, ["S1", "S2"], ["Control", "Case"])
    config = Config(
        _write_config(
            tmp_path,
            data_input="samples.txt",
            data_input_mode="sample-files",
        )
    )

    with pytest.raises(ValueError, match="PATH or SAMPLE_NAME"):
        OligosHandler(config).get_oligos_df()


# ---------------------------------------------------------------------------
# Library filters and clinical variables
# ---------------------------------------------------------------------------


def _write_filter_project(tmp_path: Path) -> Path:
    _write_metadata(tmp_path)
    _write_matrix(tmp_path / "matrix.csv")
    # Deliberately use a different row order and omit twist_p3.
    pd.DataFrame(
        {
            "peptide_id": ["agilent_p2", "agilent_extra", "agilent_p1"],
            "Species": ["Bacterial", "Human", "Human"],
            "flag": ["false", "yes", "true"],
        }
    ).to_csv(tmp_path / "library.csv", index=False)
    return tmp_path


def test_library_filter_preserves_matrix_column_order_and_drops_unannotated(
    tmp_path: Path,
) -> None:
    _write_filter_project(tmp_path)
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
        oligo_filters={"Species": ["Human", "Bacterial"]},
    )

    features, _ = manager.get_features_target()

    # Matrix order is p1, p2, p3; p3 is absent from library metadata.
    assert features.columns.tolist() == ["agilent_p1", "agilent_p2"]


def test_config_level_oligo_filters_are_used_by_default(tmp_path: Path) -> None:
    _write_filter_project(tmp_path)
    config = Config(
        _write_config(
            tmp_path,
            lib_metadata_input="library.csv",
            lib_col_peptide_name="peptide_id",
            oligo_filters={"Species": "Human"},
        )
    )
    manager = FeatureManager(
        config,
        MetadataHandler(config),
        OligosHandler(config),
    )

    features, _ = manager.get_features_target()

    assert features.columns.tolist() == ["agilent_p1"]


def test_boolean_library_filter_understands_common_representations(
    tmp_path: Path,
) -> None:
    _write_metadata(
        tmp_path,
        ["S1", "S2", "S3", "S4", "S5", "S6"],
        ["Control", "Case", "Control", "Case", "Control", "Case"],
    )
    peptide_ids = [f"agilent_p{i}" for i in range(1, 7)]
    matrix: dict[str, Any] = {"peptide": peptide_ids}
    for index in range(1, 7):
        matrix[f"S{index}"] = [1 if row == index else 0 for row in range(1, 7)]
    pd.DataFrame(matrix).to_csv(tmp_path / "matrix.csv", index=False)
    pd.DataFrame(
        {
            "peptide_id": peptide_ids,
            "flag": ["true", "1", "yes", "false", "0", "n"],
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
        oligo_filters={"flag": True},
    )

    features, _ = manager.get_features_target()

    assert features.columns.tolist() == ["agilent_p1", "agilent_p2", "agilent_p3"]


def test_categorical_clinical_features_remain_unchanged(tmp_path: Path) -> None:
    _write_metadata(
        tmp_path,
        Stage=["I", "II", "III"],
        Biomarker=["positive", "negative", None],
    )
    _write_matrix(tmp_path / "matrix.csv")
    config = Config(
        _write_config(
            tmp_path,
            extra_features_to_include=["Stage", "Biomarker"],
            fillna_value=None,
        )
    )
    manager = FeatureManager(
        config,
        MetadataHandler(config),
        OligosHandler(config),
        with_oligos=False,
        with_additional_features=True,
    )

    features, _ = manager.get_features_target()

    assert features["Stage"].tolist() == ["I", "II", "III"]
    assert features["Biomarker"].iloc[:2].tolist() == ["positive", "negative"]
    assert pd.isna(features.loc["S3", "Biomarker"])


def test_existing_sex_and_gender_columns_are_encoded_independently(
    tmp_path: Path,
) -> None:
    _write_metadata(
        tmp_path,
        ["S1", "S2"],
        ["Control", "Case"],
        Sex=["F", "M"],
        Gender=["male", "female"],
    )
    _write_matrix(tmp_path / "matrix.csv", sample_names=["S1", "S2", "S3"])
    metadata = MetadataHandler(
        Config(_write_config(tmp_path))
    ).get_individuals_metadata_df()

    assert metadata["Sex"].tolist() == [0, 1]
    assert metadata["Gender"].tolist() == [1, 0]


# ---------------------------------------------------------------------------
# Cache refresh and target behavior
# ---------------------------------------------------------------------------


def test_oligos_refresh_reloads_modified_input_file(tmp_path: Path) -> None:
    _base_project(tmp_path)
    handler = OligosHandler(Config(_write_config(tmp_path)))
    initial = handler.get_oligos_df()
    assert initial.loc["S1", "agilent_p1"] == 1

    changed = pd.read_csv(tmp_path / "matrix.csv")
    changed.loc[changed["peptide"] == "agilent_p1", "S1"] = 0
    changed.to_csv(tmp_path / "matrix.csv", index=False)

    assert handler.get_oligos_df().loc["S1", "agilent_p1"] == 1
    assert handler.get_oligos_df(refresh=True).loc["S1", "agilent_p1"] == 0


def test_entirely_numeric_targets_are_encoded(tmp_path: Path) -> None:
    _write_metadata(tmp_path, targets=[0, 1, 1])
    _write_matrix(tmp_path / "matrix.csv")
    config = Config(_write_config(tmp_path))
    manager = FeatureManager(
        config,
        MetadataHandler(config),
        OligosHandler(config),
    )

    _, target = manager.get_features_target()

    assert target.tolist() == [0, 1, 1]


def test_multiclass_targets_follow_configured_order(tmp_path: Path) -> None:
    _write_metadata(
        tmp_path,
        targets=["Healthy", "Disease", "Other"],
    )
    _write_matrix(tmp_path / "matrix.csv")
    config = Config(
        _write_config(
            tmp_path,
            group_tests=["Healthy", "Disease", "Other"],
        )
    )
    manager = FeatureManager(
        config,
        MetadataHandler(config),
        OligosHandler(config),
    )

    _, target = manager.get_features_target()

    assert target.tolist() == [0, 1, 2]


def test_all_invalid_numeric_target_codes_are_rejected(tmp_path: Path) -> None:
    _write_metadata(tmp_path, targets=[2, 2, 3])
    _write_matrix(tmp_path / "matrix.csv")

    with pytest.raises(ValueError, match="No metadata rows remain"):
        MetadataHandler(Config(_write_config(tmp_path))).get_individuals_metadata_df()


def test_mixed_valid_and_invalid_targets_drop_invalid_rows(tmp_path: Path) -> None:
    _write_metadata(tmp_path, targets=["Control", 2, "Case"])
    _write_matrix(tmp_path / "matrix.csv")
    config = Config(_write_config(tmp_path))
    manager = FeatureManager(
        config,
        MetadataHandler(config),
        OligosHandler(config),
    )

    features, target = manager.get_features_target()

    # This documents the current non-strict behavior of MetadataHandler.
    assert features.index.tolist() == ["S1", "S3"]
    assert target.tolist() == [0, 1]


# ---------------------------------------------------------------------------
# Optional smoke test using a real phipml YAML configuration
# ---------------------------------------------------------------------------


def test_real_data_from_environment() -> None:
    """Run with PHIPML_REAL_CONFIG=/absolute/path/config.yaml.

    The test is skipped during ordinary CI. When enabled, it checks that real
    sample IDs align, feature names are unique, and per-sample input is binary.
    """
    config_value = os.environ.get("PHIPML_REAL_CONFIG")
    if not config_value:
        pytest.skip("Set PHIPML_REAL_CONFIG to run the real-data smoke test")

    config_path = Path(config_value).expanduser().resolve()
    config = Config(config_path)
    metadata_handler = MetadataHandler(config)
    oligos_handler = OligosHandler(config)
    manager = FeatureManager(
        config,
        metadata_handler,
        oligos_handler,
        with_oligos=True,
        with_additional_features=False,
    )

    features, target = manager.get_features_target()

    assert not features.empty
    assert not target.empty
    assert features.index.is_unique
    assert features.columns.is_unique
    assert target.index.equals(features.index)
    assert not features.index.to_series().str.strip().eq("").any()
    assert not features.columns.to_series().str.strip().eq("").any()

    if config.get_data_input_mode() == "sample-files":
        observed = set(np.unique(features.to_numpy()))
        assert observed <= {0, 1}, f"Non-binary sample-file values found: {observed}"
