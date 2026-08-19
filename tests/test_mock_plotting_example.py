"""End-to-end plotting checks using the repository's 50-peptide mock data."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
import pandas as pd
import yaml

matplotlib.use("Agg")

from phipml.cli.plot_results import main as plot_main  # noqa: E402
from phipml.cli.train_test import main as train_main  # noqa: E402
from phipml.io.data_handler import Config  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MOCK_ROOT = REPOSITORY_ROOT / "mock_examples"
EXTERNAL_ROOT = MOCK_ROOT / "external_validation_noisy"


def _temporary_external_config(tmp_path: Path) -> Path:
    source = EXTERNAL_ROOT / "config_noisy_external_no_tuning.yaml"
    with source.open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle)

    values["metadata_input"] = str(
        (EXTERNAL_ROOT / "metadata_noisy_train_external.csv").resolve()
    )
    values["data_input"] = str(
        (EXTERNAL_ROOT / "data_noisy_train_external.csv").resolve()
    )
    values["lib_metadata_input"] = str(
        (MOCK_ROOT / "peptide_library_metadata.csv").resolve()
    )
    values["classification"]["output_dir"] = str(tmp_path / "results")
    values["classification"]["bootstrap_validation"] = True
    values["classification"]["bootstrap_n_resamples"] = 30
    values["classification"]["bootstrap_confidence_level"] = 0.95

    destination = tmp_path / "config_plotting_integration.yaml"
    destination.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return destination


def test_mock_config_snapshot_preserves_resolved_plotting_inputs(
    tmp_path: Path,
) -> None:
    # Build a self-contained config that explicitly includes the optional
    # peptide-library metadata path. This keeps the snapshot test independent
    # of whether a user's copy of the demonstration YAML includes that optional
    # annotation input.
    source = _temporary_external_config(tmp_path)
    original = Config(source)
    restored = Config.from_mapping(
        original.to_mapping(),
        config_file=original.config_file,
    )

    assert restored.data_input == original.data_input
    assert restored.metadata_input == original.metadata_input
    assert restored.lib_metadata_input == original.lib_metadata_input
    assert restored.data_input.is_absolute()
    assert restored.metadata_input.is_absolute()
    assert restored.lib_metadata_input is not None
    assert restored.lib_metadata_input.is_absolute()
    assert restored.group_tests == ["Control", "Case"]


def test_real_mock_validation_artifact_reconstructs_and_plots(
    tmp_path: Path,
) -> None:
    config_path = _temporary_external_config(tmp_path)

    assert train_main(["--config", str(config_path)]) == 0

    artifact = (
        tmp_path
        / "results"
        / "validation_random-forest_noisy_external_untuned_420.joblib"
    )
    assert artifact.is_file()

    saved = joblib.load(artifact)
    assert {
        "metrics_test",
        "scores_test",
        "test_shap_values",
        "selected_features_test",
        "target_test",
        "data_context",
    }.issubset(saved)
    assert len(saved["target_test"]) == 40
    assert saved["data_context"]["artifact_type"] == "validation"
    assert saved["data_context"]["validation_name"] == "noisy_external_untuned"
    resolved = saved["data_context"]["resolved_config"]
    assert Path(resolved["data_input"]).is_absolute()
    assert Path(resolved["metadata_input"]).is_absolute()
    assert Path(resolved["lib_metadata_input"]).is_absolute()

    output_dir = tmp_path / "plots"
    assert (
        plot_main(
            [
                str(artifact),
                "--split",
                "test",
                "--output-dir",
                str(output_dir),
                "--output-prefix",
                "mock_external",
                "--table-annotation-columns",
                "Description",
                "Species",
                "Protein",
            ]
        )
        == 0
    )

    expected = {
        "mock_external_performance.pdf",
        "mock_external_roc.pdf",
        "mock_external_pr.pdf",
        "mock_external_confusion.pdf",
        "mock_external_classification.pdf",
        "mock_external_shap_importance.pdf",
        "mock_external_shap_heatmap.pdf",
        "mock_external_shap_beeswarm.pdf",
        "mock_external_feature_importance.csv",
        "mock_external_feature_table.pdf",
    }
    generated = {path.name for path in output_dir.iterdir()}
    assert expected.issubset(generated)
    assert "mock_external_performance.svg" in generated
    assert "mock_external_performance.png" in generated
    for filename in expected:
        assert (output_dir / filename).stat().st_size > 0

    feature_table = pd.read_csv(output_dir / "mock_external_feature_importance.csv")
    assert {
        "Feature",
        "Feature type",
        "Mean |SHAP|",
        "Description",
        "Species",
        "Protein",
        "Top-k SHAP frequency (%)",
        "Mean rank when in top K",
        "Feature ranking method",
        "Displayed",
    }.issubset(feature_table.columns)
    assert set(feature_table["Feature ranking method"]) == {"mean-abs-shap"}
    assert feature_table["Displayed"].sum() == 20
    peptide_rows = feature_table[feature_table["Feature type"] == "peptide"]
    assert not peptide_rows.empty
    assert peptide_rows["Description"].notna().all()
    assert peptide_rows["Species"].notna().all()
