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


def _write_config(tmp_path: Path, **updates: object) -> Path:
    values: dict[str, object] = {
        "metadata_input": "metadata.csv",
        "data_input": "peptides.csv",
        "lib_metadata_input": "library.csv",
        "group_tests": ["Control", "Case"],
        "peptide_prefixes": ["agilent", "twist", "agilent_"],
        "extra_features_to_include": ["Age", "Score"],
    }
    values.update(updates)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(values), encoding="utf-8")
    return config_path


@pytest.fixture
def project(tmp_path: Path) -> Path:
    pd.DataFrame(
        {
            "SampleName": ["S1", "S2", "S3"],
            "group_test": ["Control", "Case", "Case"],
            "cohort": ["train", "train", "external"],
            "site": ["A", "B", "B"],
            "Sex": ["Female", "m", 0],
            "Age": [40.0, np.nan, 60.0],
            "Score": [0.5, 0.6, np.nan],
        }
    ).to_csv(tmp_path / "metadata.csv", index=False)

    # Combined input is peptide x sample; transposed=True makes sample x peptide.
    pd.DataFrame(
        {
            "peptide": ["agilent_p1", "agilent_p2", "twist_p3"],
            "S1": [1, 0, 0],
            "S2": [0, 1, 0],
            "S3": [1, 1, 1],
        }
    ).to_csv(tmp_path / "peptides.csv", index=False)

    pd.DataFrame(
        {
            "peptide_id": ["agilent_p1", "agilent_p2", "twist_p3"],
            "is_PNP": [True, True, False],
            "Species": ["Homo sapiens", "Escherichia coli", "Homo sapiens"],
        }
    ).to_csv(tmp_path / "library.csv", index=False)
    return tmp_path


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_resolves_paths_prefixes_and_derived_groups(project: Path) -> None:
    config = Config(_write_config(project))

    assert config.data_input == (project / "peptides.csv").resolve()
    assert config.metadata_input == (project / "metadata.csv").resolve()
    assert config.lib_metadata_input == (project / "library.csv").resolve()
    assert config.peptide_prefixes == ["agilent_", "twist_"]
    assert config.group_label_encoding == {"Control": 0, "Case": 1}
    assert config.group_code_to_label == {0: "Control", 1: "Case"}
    assert config.label_group_tests == "Control-Case"
    assert config.get_data_input_mode() == "matrix"


def test_config_supports_legacy_aliases_and_explicit_overrides(project: Path) -> None:
    config_path = _write_config(project)
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    values.pop("lib_metadata_input")
    values.pop("peptide_prefixes")
    values["lib_meta_data"] = "library.csv"
    values["libraries_prefixes"] = ["agilent", "twist"]
    config_path.write_text(yaml.safe_dump(values), encoding="utf-8")

    config = Config(config_path, random_state=7)

    assert config.lib_metadata_input == (project / "library.csv").resolve()
    assert config.peptide_prefixes == ["agilent_", "twist_"]
    assert config.random_state == 7


def test_config_set_group_tests_refreshes_derived_values(project: Path) -> None:
    config = Config(_write_config(project))

    config.set_group_tests(["Healthy", "Disease", "Other"])

    assert config.group_label_encoding == {"Healthy": 0, "Disease": 1, "Other": 2}
    assert config.group_code_to_label == {0: "Healthy", 1: "Disease", 2: "Other"}
    assert config.label_group_tests == "Healthy-Disease-Other"


@pytest.mark.parametrize(
    ("updates", "error", "message"),
    [
        ({"group_tests": ["Control"]}, ValueError, "at least two"),
        ({"group_tests": ["Control", "Control"]}, ValueError, "duplicate"),
        ({"group_tests": ["Control", None]}, ValueError, "missing"),
        ({"peptide_prefixes": "agilent"}, TypeError, "sequence"),
        ({"extra_features_to_include": "Age"}, TypeError, "sequence"),
        ({"filters_metadata": ["cohort"]}, TypeError, "mapping"),
        ({"oligo_filter_mode": "xor"}, ValueError, "all.*any"),
        ({"classification": []}, TypeError, "mapping"),
    ],
)
def test_config_rejects_invalid_values(
    project: Path,
    updates: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        Config(_write_config(project, **updates))


def test_config_rejects_unknown_and_missing_mandatory_keys(project: Path) -> None:
    config_path = _write_config(project, unexpected_key=True)
    with pytest.raises(ValueError, match="Unknown configuration keys"):
        Config(config_path)

    values = {
        "metadata_input": "metadata.csv",
        "group_tests": ["Control", "Case"],
    }
    config_path.write_text(yaml.safe_dump(values), encoding="utf-8")
    with pytest.raises(ValueError, match="data_input"):
        Config(config_path)


def test_config_rejects_unsupported_input_extensions(project: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported data_input extension"):
        Config(_write_config(project, data_input="peptides.json"))

    with pytest.raises(ValueError, match="Unsupported metadata_input extension"):
        Config(_write_config(project, metadata_input="metadata.json"))

    with pytest.raises(ValueError, match="Unsupported lib_metadata_input extension"):
        Config(_write_config(project, lib_metadata_input="library.json"))


def test_config_converts_bayesian_grid_without_mutating_original(project: Path) -> None:
    pytest.importorskip("skopt")
    config = Config(
        _write_config(
            project,
            param_grid={
                "xgboost": {
                    "estimator__max_depth": {
                        "type": "integer",
                        "low": 3,
                        "high": 8,
                    },
                    "estimator__learning_rate": {
                        "type": "real",
                        "low": 0.01,
                        "high": 0.3,
                        "prior": "log-uniform",
                    },
                    "estimator__max_features": {
                        "type": "categorical",
                        "categories": ["sqrt", "log2"],
                    },
                }
            },
        )
    )
    original = config.param_grid.copy()

    converted = config.get_bayesian_param_grid("xgboost")

    assert set(converted) == {
        "estimator__max_depth",
        "estimator__learning_rate",
        "estimator__max_features",
    }
    assert config.param_grid == original


# ---------------------------------------------------------------------------
# MetadataHandler
# ---------------------------------------------------------------------------


def test_metadata_encodes_sex_and_gender_variants_and_removes_unnamed(
    project: Path,
) -> None:
    values: list[object] = ["F", "female", "M", "male", "0", "1", "Unknown", np.nan]
    pd.DataFrame(
        {
            "SampleName": [f"S{i}" for i in range(len(values))],
            "group_test": ["Control", "Case"] * 4,
            "Gender": values,
            "Unnamed: 0": range(len(values)),
        }
    ).to_csv(project / "metadata.csv", index=False)
    handler = MetadataHandler(Config(_write_config(project)))

    metadata = handler.get_individuals_metadata_df()

    assert "Unnamed: 0" not in metadata.columns
    assert metadata["Gender"].iloc[:6].tolist() == [0, 0, 1, 1, 0, 1]
    assert metadata["Gender"].iloc[6] == "Unknown"
    assert pd.isna(metadata["Gender"].iloc[7])
    pd.testing.assert_series_equal(
        metadata["Sex"],
        metadata["Gender"],
        check_names=False,
    )


def test_metadata_applies_and_filter(project: Path) -> None:
    config = Config(
        _write_config(
            project,
            filters_metadata={"cohort": "train", "site": ["B"]},
        )
    )

    metadata = MetadataHandler(config).get_individuals_metadata_df()

    assert metadata.index.tolist() == ["S2"]


def test_metadata_combined_filters_use_or_between_conditions(project: Path) -> None:
    config = Config(
        _write_config(
            project,
            combined_filters_metadata=[
                {"cohort": "external"},
                {"site": "A"},
            ],
        )
    )

    metadata = MetadataHandler(config).get_individuals_metadata_df()

    assert metadata.index.tolist() == ["S1", "S3"]


def test_metadata_accepts_mixed_labels_and_positional_codes(project: Path) -> None:
    metadata = pd.read_csv(project / "metadata.csv")
    metadata["group_test"] = ["Control", 1, "Case"]
    metadata.to_csv(project / "metadata.csv", index=False)
    config = Config(_write_config(project))
    handler = MetadataHandler(config)

    selected = handler.get_individuals_metadata_df()
    manager = FeatureManager(config, handler, OligosHandler(config))
    _, target = manager.get_features_target()

    assert selected.attrs["target_representation"] == "labels and numeric codes"
    assert target.tolist() == [0, 1, 1]


def test_metadata_reports_no_matching_target_values(project: Path) -> None:
    metadata = pd.read_csv(project / "metadata.csv")
    metadata["group_test"] = ["A", "B", "C"]
    metadata.to_csv(project / "metadata.csv", index=False)

    with pytest.raises(ValueError, match="No metadata rows remain"):
        MetadataHandler(Config(_write_config(project))).get_individuals_metadata_df()


def test_metadata_rejects_duplicate_or_missing_sample_ids(project: Path) -> None:
    metadata = pd.read_csv(project / "metadata.csv")
    metadata.loc[1, "SampleName"] = "S1"
    metadata.to_csv(project / "metadata.csv", index=False)
    with pytest.raises(ValueError, match="duplicate sample IDs"):
        MetadataHandler(Config(_write_config(project))).get_individuals_metadata_df()

    metadata = metadata.drop(columns="SampleName")
    metadata.to_csv(project / "metadata.csv", index=False)
    with pytest.raises(KeyError, match="Sample column 'SampleName' not found"):
        MetadataHandler(Config(_write_config(project))).get_individuals_metadata_df()


def test_metadata_rejects_missing_target_or_filter_column(project: Path) -> None:
    metadata = pd.read_csv(project / "metadata.csv").drop(columns="group_test")
    metadata.to_csv(project / "metadata.csv", index=False)
    with pytest.raises(KeyError, match="Target column 'group_test' not found"):
        MetadataHandler(Config(_write_config(project))).get_individuals_metadata_df()

    # Restore metadata before testing the filter-column failure.
    project = _restore_base_metadata(project)
    config = Config(_write_config(project, filters_metadata={"missing": "value"}))
    with pytest.raises(KeyError, match="Metadata filter column 'missing' not found"):
        MetadataHandler(config).get_individuals_metadata_df()


def _restore_base_metadata(project: Path) -> Path:
    pd.DataFrame(
        {
            "SampleName": ["S1", "S2", "S3"],
            "group_test": ["Control", "Case", "Case"],
            "cohort": ["train", "train", "external"],
            "site": ["A", "B", "B"],
            "Sex": ["Female", "m", 0],
            "Age": [40.0, np.nan, 60.0],
            "Score": [0.5, 0.6, np.nan],
        }
    ).to_csv(project / "metadata.csv", index=False)
    return project


def test_additional_features_success_and_missing_column(project: Path) -> None:
    config = Config(_write_config(project))
    additional = MetadataHandler(config).get_additional_features_df()

    assert additional.columns.tolist() == ["Age", "Score"]
    assert additional.index.tolist() == ["S1", "S2", "S3"]

    missing_config = Config(
        _write_config(project, extra_features_to_include=["Age", "Missing"])
    )
    with pytest.raises(KeyError, match="Additional metadata features not found"):
        MetadataHandler(missing_config).get_additional_features_df()


def test_metadata_cache_returns_copies_and_refreshes(project: Path) -> None:
    handler = MetadataHandler(Config(_write_config(project)))
    first = handler.get_individuals_metadata_df()
    first.loc["S1", "Age"] = -1

    cached = handler.get_individuals_metadata_df()
    assert cached.loc["S1", "Age"] == 40

    metadata = pd.read_csv(project / "metadata.csv")
    metadata.loc[metadata["SampleName"] == "S1", "Age"] = 99
    metadata.to_csv(project / "metadata.csv", index=False)
    assert handler.get_individuals_metadata_df().loc["S1", "Age"] == 40
    assert handler.get_individuals_metadata_df(refresh=True).loc["S1", "Age"] == 99


# ---------------------------------------------------------------------------
# OligosHandler
# ---------------------------------------------------------------------------


def test_oligos_reads_and_transposes_combined_matrix(project: Path) -> None:
    oligos = OligosHandler(Config(_write_config(project))).get_oligos_df()

    assert oligos.index.tolist() == ["S1", "S2", "S3"]
    assert oligos.columns.tolist() == ["agilent_p1", "agilent_p2", "twist_p3"]
    assert oligos.to_numpy().tolist() == [[1, 0, 0], [0, 1, 0], [1, 1, 1]]


def test_oligos_supports_sample_by_peptide_matrix(project: Path) -> None:
    pd.DataFrame(
        {
            "SampleName": ["S1", "S2"],
            "agilent_p1": [1, 0],
            "agilent_p2": [0, 1],
        }
    ).to_csv(project / "sample_by_peptide.csv", index=False)
    config = Config(
        _write_config(
            project,
            data_input="sample_by_peptide.csv",
            transposed=False,
        )
    )

    oligos = OligosHandler(config).get_oligos_df()

    assert oligos.index.tolist() == ["S1", "S2"]
    assert oligos.to_numpy().tolist() == [[1, 0], [0, 1]]


def test_oligos_rejects_duplicate_samples_and_peptides(project: Path) -> None:
    pd.DataFrame(
        {
            "SampleName": ["S1", "S1"],
            "agilent_p1": [1, 0],
        }
    ).to_csv(project / "duplicates.csv", index=False)
    config = Config(
        _write_config(project, data_input="duplicates.csv", transposed=False)
    )
    with pytest.raises(ValueError, match="duplicate sample IDs"):
        OligosHandler(config).get_oligos_df()

    pd.DataFrame(
        {
            "peptide": ["agilent_p1", "agilent_p1"],
            "S1": [1, 0],
            "S2": [0, 1],
        }
    ).to_csv(project / "duplicates.csv", index=False)
    config = Config(_write_config(project, data_input="duplicates.csv"))
    with pytest.raises(ValueError, match="duplicate peptide IDs"):
        OligosHandler(config).get_oligos_df()


def test_library_metadata_uses_named_peptide_column_for_csv(project: Path) -> None:
    config = Config(_write_config(project, lib_col_peptide_name="peptide_id"))

    library = OligosHandler(config).get_oligos_metadata_df()

    assert library.index.name == "peptide_id"
    assert library.index.tolist() == ["agilent_p1", "agilent_p2", "twist_p3"]
    assert library.columns.tolist() == ["is_PNP", "Species"]


def test_library_metadata_supports_pickle_with_named_index(project: Path) -> None:
    library = pd.read_csv(project / "library.csv").set_index("peptide_id")
    library.to_pickle(project / "library.pkl")
    config = Config(
        _write_config(
            project,
            lib_metadata_input="library.pkl",
            lib_col_peptide_name="peptide_id",
        )
    )

    loaded = OligosHandler(config).get_oligos_metadata_df()

    pd.testing.assert_frame_equal(loaded, library)


def test_library_metadata_rejects_duplicate_peptide_ids(project: Path) -> None:
    library = pd.read_csv(project / "library.csv")
    library.loc[1, "peptide_id"] = "agilent_p1"
    library.to_csv(project / "library.csv", index=False)
    config = Config(_write_config(project, lib_col_peptide_name="peptide_id"))

    with pytest.raises(ValueError, match="duplicate peptide IDs"):
        OligosHandler(config).get_oligos_metadata_df()


def test_oligos_cache_returns_independent_copies(project: Path) -> None:
    handler = OligosHandler(Config(_write_config(project)))
    first = handler.get_oligos_df()
    first.loc["S1", "agilent_p1"] = 99

    second = handler.get_oligos_df()

    assert second.loc["S1", "agilent_p1"] == 1


# ---------------------------------------------------------------------------
# FeatureManager
# ---------------------------------------------------------------------------


def _manager(
    project: Path,
    **manager_kwargs: object,
) -> FeatureManager:
    config = Config(
        _write_config(
            project,
            lib_col_peptide_name="peptide_id",
        )
    )
    return FeatureManager(
        config,
        MetadataHandler(config),
        OligosHandler(config),
        **manager_kwargs,
    )


def test_feature_manager_requires_at_least_one_feature_source(project: Path) -> None:
    with pytest.raises(ValueError, match="Select oligos"):
        _manager(
            project,
            with_oligos=False,
            with_additional_features=False,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prevalence_threshold_min": -1},
        {"prevalence_threshold_max": 101},
        {"prevalence_threshold_min": 80, "prevalence_threshold_max": 20},
    ],
)
def test_feature_manager_validates_prevalence_thresholds(
    project: Path,
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="prevalence|Prevalence"):
        _manager(project, **kwargs)


def test_feature_manager_supports_legacy_boolean_subgroup(project: Path) -> None:
    features, target = _manager(project, subgroup="is_PNP").get_features_target()

    assert features.columns.tolist() == ["agilent_p1", "agilent_p2"]
    assert target.tolist() == [0, 1, 1]


def test_feature_manager_combines_oligo_filters_with_all_or_any(project: Path) -> None:
    filters = {"is_PNP": True, "Species": "Homo sapiens"}
    all_features, _ = _manager(
        project,
        oligo_filters=filters,
        oligo_filter_mode="all",
    ).get_features_target()
    any_features, _ = _manager(
        project,
        oligo_filters=filters,
        oligo_filter_mode="any",
    ).get_features_target()

    assert all_features.columns.tolist() == ["agilent_p1"]
    assert any_features.columns.tolist() == [
        "agilent_p1",
        "agilent_p2",
        "twist_p3",
    ]


def test_feature_manager_rejects_conflicting_or_invalid_filters(project: Path) -> None:
    with pytest.raises(ValueError, match="either subgroup or oligo_filters"):
        _manager(project, subgroup="is_PNP", oligo_filters={"Species": "Homo sapiens"})

    with pytest.raises(KeyError, match="Oligo metadata column 'Missing'"):
        _manager(project, oligo_filters={"Missing": True}).get_features_target()

    with pytest.raises(ValueError, match="No peptide features matched"):
        _manager(
            project,
            oligo_filters={"Species": "Not present"},
        ).get_features_target()


def test_feature_manager_supports_additional_features_without_oligos(
    project: Path,
) -> None:
    features, target = _manager(
        project,
        with_oligos=False,
        with_additional_features=True,
    ).get_features_target()

    assert features.columns.tolist() == ["Age", "Score"]
    assert features.index.tolist() == ["S1", "S2", "S3"]
    assert target.tolist() == [0, 1, 1]


def test_feature_manager_fills_missing_values_when_configured(project: Path) -> None:
    config = Config(
        _write_config(
            project,
            fillna_value=0,
            lib_col_peptide_name="peptide_id",
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

    assert features.isna().sum().sum() == 0
    assert features.loc["S2", "Age"] == 0
    assert features.loc["S3", "Score"] == 0


def test_feature_manager_applies_presence_prevalence_boundaries(project: Path) -> None:
    features, _ = _manager(
        project,
        prevalence_threshold_min=50,
        prevalence_threshold_max=70,
    ).get_features_target()

    # p1 and p2 occur in 2/3 samples; p3 occurs in only 1/3.
    assert features.columns.tolist() == ["agilent_p1", "agilent_p2"]


def test_public_prevalence_filter_preserves_clinical_columns(project: Path) -> None:
    manager = _manager(
        project,
        prevalence_threshold_min=50,
        prevalence_threshold_max=100,
    )
    values = pd.DataFrame(
        {
            "agilent_rare": [1, 0, 0],
            "agilent_common": [1, 1, 0],
            "Age": [40, 50, 60],
        }
    )

    filtered = manager.filter_prevalence(values)

    assert filtered.columns.tolist() == ["agilent_common", "Age"]


def test_feature_manager_reports_no_common_samples(project: Path) -> None:
    pd.DataFrame(
        {
            "peptide": ["agilent_p1"],
            "X1": [1],
            "X2": [0],
        }
    ).to_csv(project / "peptides.csv", index=False)

    with pytest.raises(ValueError, match="no sample IDs in common"):
        _manager(project).get_features_target()


def test_feature_manager_reports_duplicate_combined_feature_names(
    project: Path,
) -> None:
    metadata = pd.read_csv(project / "metadata.csv")
    metadata["agilent_p1"] = [0.1, 0.2, 0.3]
    metadata.to_csv(project / "metadata.csv", index=False)
    config = Config(
        _write_config(
            project,
            extra_features_to_include=["agilent_p1"],
            lib_col_peptide_name="peptide_id",
        )
    )
    manager = FeatureManager(
        config,
        MetadataHandler(config),
        OligosHandler(config),
        with_oligos=True,
        with_additional_features=True,
    )

    with pytest.raises(ValueError, match="Duplicate feature names"):
        manager.get_features_target()


def test_feature_manager_reports_when_prevalence_removes_everything(
    project: Path,
) -> None:
    manager = _manager(
        project,
        prevalence_threshold_min=90,
        prevalence_threshold_max=95,
    )

    with pytest.raises(ValueError, match="No features remain"):
        manager.get_features_target()
