# ======================
# Standard library
# ======================
import argparse
import json
import shlex

# ======================
# Local / project imports
# ======================
from phipml.classification.train_test_utils import (
    apply_prevalence_filter_train_only,
    build_validation_set,
    get_best_estimator,
    make_dataset,
    run_and_save_nested_cv,
    safe_concat_Xy,
    setup_feature_manager,
    validate_and_save,
)
from phipml.io.data_handler import Config


class _ArgParser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, arg_line: str):
        line = arg_line.strip()
        if not line or line.startswith("#"):
            return []
        # remove inline comments
        line = line.split("#", 1)[0].strip()
        if not line:
            return []
        return shlex.split(line)


def str2bool(x):
    xl = x.lower()
    if xl in ("yes", "true", "t", "y", "1"):
        return True
    if xl in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {x!r}")


def parse_args_ML(argv=None) -> argparse.Namespace:
    # Parse the command-line argument for the random seed
    parser = _ArgParser(
        description="Run nested CV and validation with custom random seed and metadata filters for classification models.",
        fromfile_prefix_chars="@",  # <--- enables @argsfile
    )

    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        nargs="?",
        default=420,
        help="Random seed (default: 420)",
    )

    parser.add_argument(
        "--config",
        "-cf",
        type=str,
        default="config.yaml",
        help="Full path of the config file to use (default: config.yaml in script directory)",
    )

    parser.add_argument(
        "--run_nested_cv",
        "-ncv",
        type=str2bool,
        default=True,
        help="Run nested cv for training set (default: True)",
    )

    parser.add_argument(
        "--train_size",
        "-ts",
        type=float,
        default=0.7,
        help="Train split size for a given group. Default = 0.7",
    )

    parser.add_argument(
        "--no_additional_train_test_data",
        "-nat",
        type=str2bool,
        default=False,
        help="Whether to concatenate extra train/test data (True|False).",
    )

    parser.add_argument(
        "--train_test_split_data",
        "-sp",
        type=json.loads,
        default={},
        help=(
            "JSON dict of metadata for splitting, e.g. "
            '\'{"group_test":"Controls","other_key":"value"}\'. '
            "Default = {}"
        ),
    )

    parser.add_argument(
        "--train",
        "-t",
        type=json.loads,
        default={},
        help=(
            "JSON dict of metadata for train, e.g. "
            '\'{"group_test":"Controls","other_key":"value"}\'. '
            "Default = {}"
        ),
    )

    parser.add_argument(
        "--model_type",
        "-mt",
        type=str,
        choices=["xgboost", "random-forest"],
        default="xgboost",
        help="Choose the model estimator to use (default: xgboost)",
    )

    parser.add_argument(
        "--use_pretrained",
        "-upr",
        type=str2bool,
        default=False,
        help="Load pre trained model from joblib file (default: False)",
    )

    parser.add_argument(
        "--only_train_model",
        "-otm",
        type=str2bool,
        default=False,
        help="Whether to only train model or return predictions as well(True|False).",
    )

    parser.add_argument(
        "--subgroup",
        "-sub",
        type=str,
        default="all",
        help="What subgroup of peptides to include in the analysis. Default = all",
    )

    parser.add_argument(
        "--with_oligos",
        "-wo",
        type=str2bool,
        default=True,
        help="Include or not peptides in the analysis. Default = True",
    )

    parser.add_argument(
        "--with_additional_features",
        "-wa",
        type=str2bool,
        default=False,
        help="Include or not additional features in the analysis (e.g. Sex and Age). Default = False",
    )

    parser.add_argument(
        "--prevalence_threshold_min",
        "-min",
        type=float,
        default=2.0,
        help="Minimum prevalence threshold filter for training set. Default = 2.0",
    )

    parser.add_argument(
        "--prevalence_threshold_max",
        "-max",
        type=float,
        default=98.0,
        help="Maximum prevalence threshold filter for training set. Default = 98.0",
    )

    parser.add_argument(
        "--outer_cv_split",
        "-ocv",
        type=int,
        default=10,
        help="Number of k folds for outer cross-validation. Default = 10",
    )

    parser.add_argument(
        "--inner_cv_split",
        "-icv",
        type=int,
        default=5,
        help="Number of k folds for inner cross-validation. Default = 5",
    )

    # Instead of two separate filter_val flags, we do:
    #   --validate '{"treatment":"ICI"}' Cirrhosis-ICI-H
    #   --validate '{"treatment":"TKI"}' Cirrhosis-ICI-TKI
    parser.add_argument(
        "-v",
        "--validate",
        nargs=2,  # two arguments per occurrence
        action="append",
        default=[],
        metavar=("FILTER_JSON", "OUT_BASENAME"),
        help='One validation set: JSON filter and output‐base, e.g. \'{"treatment":"ICI"} Cirrhosis-ICI-H\'.',
    )

    parser.add_argument(
        "--input_dir",
        "-id",
        type=str,
        default=".",
        help="Base name for directory to input joblib files (default: .)",
    )

    parser.add_argument(
        "--input_val",
        "-iv",
        type=str,
        default="out_name",
        help="Base name for validation file containing best estimator (default: out_name)",
    )

    parser.add_argument(
        "--out_dir",
        "-d",
        type=str,
        default=".",
        help="Base name for directory to save files (default: .)",
    )

    parser.add_argument(
        "--out_name",
        "-o",
        type=str,
        default="out_name",
        help="Base name for nested‐CV and train_test split predictions (default: out_name)",
    )

    return parser.parse_args(argv)


def main(argv=None):

    args = parse_args_ML(argv)
    val_specs = [(json.loads(filt), outname) for filt, outname in (args.validate or [])]

    config = Config(args.config)
    config.get_bayesian_param_grid_from_dict_items(
        args.model_type
    )  # format bayesian param grid from config file

    # Optional split dataset (gives X_split_train/test)
    split_data = None
    if args.train_test_split_data:
        split_fm = setup_feature_manager(config, args.train_test_split_data, args)
        split_data = make_dataset(
            split_fm,
            args,
            do_split=True,
            apply_prevalence_filter=False,  # IMPORTANT: keep raw until you decide how it’s used
        )

        if args.no_additional_train_test_data:
            # if split_data is None or not split_data.has_test:
            #    raise ValueError("--no_additional_train_test_data requires --train_test_split_data")
            args.train = {}
            # Apply prevalence filter only to the training part
            X_train = apply_prevalence_filter_train_only(
                split_fm, split_data.X_train, args
            )
            y_train = split_data.y_train
            X_test, y_test = split_data.X_test, split_data.y_test

            # Optional nested CV
            if args.run_nested_cv:

                run_and_save_nested_cv(X_train, y_train, args, config)

            validate_and_save(
                X_train, y_train, X_test, y_test, args, config, args.out_name
            )

    # Otherwise, standard training mode
    if args.train:
        train_fm = setup_feature_manager(config, args.train, args)
        train_data = make_dataset(
            train_fm,
            args,
            do_split=False,
            apply_prevalence_filter=False,  # training set gets prevalence filter
        )
        X_train, y_train = train_data.X_train, train_data.y_train

        # Optionally concatenate split_train
        if split_data is not None:
            X_train, y_train = safe_concat_Xy(
                train_data.X_train,
                train_data.y_train,
                split_data.X_train,
                split_data.y_train,
            )

        # Apply prevalence filter only to the training part
        X_train = apply_prevalence_filter_train_only(train_fm, X_train, args)

        # Nested CV on final training set
        if args.run_nested_cv:

            run_and_save_nested_cv(X_train, y_train, args, config)

        # Train best model if validations exist
        if val_specs:
            best_estimator = get_best_estimator(X_train, y_train, args, config)
            if not args.only_train_model:
                for filter_val, out_val in val_specs:
                    X_test, y_test = build_validation_set(
                        train_fm,
                        config,
                        filter_val,
                        split_data=split_data,
                    )

                    validate_and_save(
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        args,
                        config,
                        out_val,
                        best_estimator=best_estimator,
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
