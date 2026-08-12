"""Tests for command-line overrides used by the classification entry point."""

from __future__ import annotations

import pytest

from phipml import __version__
from phipml.cli.train_test import parse_args_classification


BOOLEAN_DESTINATIONS = (
    "run_nested_cv",
    "use_pretrained",
    "only_train_model",
    "with_oligos",
    "with_additional_features",
    "impute_extra_numeric",
    "fill_missing_peptides_with_zero",
    "split_only",
)


def test_unspecified_boolean_overrides_remain_none() -> None:
    args = parse_args_classification(["--config", "config.yaml"])

    for destination in BOOLEAN_DESTINATIONS:
        assert getattr(args, destination) is None


@pytest.mark.parametrize(
    ("option", "destination"),
    (
        ("--run-nested-cv", "run_nested_cv"),
        ("--use-pretrained", "use_pretrained"),
        ("--only-train-model", "only_train_model"),
        ("--with-oligos", "with_oligos"),
        ("--with-additional-features", "with_additional_features"),
        ("--impute-extra-numeric", "impute_extra_numeric"),
        (
            "--fill-missing-peptides-with-zero",
            "fill_missing_peptides_with_zero",
        ),
        ("--split-only", "split_only"),
    ),
)
def test_boolean_options_accept_positive_and_negative_forms(
    option: str,
    destination: str,
) -> None:
    positive = parse_args_classification(["--config", "config.yaml", option])
    negative = parse_args_classification(
        ["--config", "config.yaml", option.replace("--", "--no-", 1)]
    )

    assert getattr(positive, destination) is True
    assert getattr(negative, destination) is False


def test_boolean_options_do_not_accept_string_values() -> None:
    with pytest.raises(SystemExit):
        parse_args_classification(
            ["--config", "config.yaml", "--with-oligos", "false"]
        )


def test_version_option_does_not_require_a_config(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        parse_args_classification(["--version"])

    assert error.value.code == 0
    assert capsys.readouterr().out.strip() == f"phipml {__version__}"


def test_help_header_contains_installed_version(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        parse_args_classification(["--help"])

    assert error.value.code == 0
    assert f"phipml {__version__}" in capsys.readouterr().out
