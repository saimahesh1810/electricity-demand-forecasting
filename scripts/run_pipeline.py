import argparse

from electricity_demand.pipeline import (
    build_model_comparison,
    run_full_pipeline,
    save_model_comparison_outputs,
)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line options.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run the German electricity-demand "
            "forecasting pipeline."
        )
    )

    parser.add_argument(
        "--comparison-only",
        action="store_true",
        help=(
            "Build the final comparison using "
            "existing metric files without "
            "rerunning models."
        ),
    )

    parser.add_argument(
        "--skip-neural",
        action="store_true",
        help=(
            "Skip LSTM training during the "
            "full pipeline run."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    Run either the full workflow or comparison only.
    """

    arguments = parse_arguments()

    if arguments.comparison_only:
        print(
            "\nBuilding comparison from "
            "existing metric files..."
        )

        comparison = (
            build_model_comparison()
        )

        save_model_comparison_outputs(
            comparison
        )

        return

    run_full_pipeline(
        include_neural=not arguments.skip_neural
    )


if __name__ == "__main__":
    main()