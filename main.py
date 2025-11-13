import logging

import click

from serve import global_vars
from pyvisdiff.pipeline import load_config, load_data_from_csv, run_pipeline


@click.command()
@click.option("--config", help="config file")
@click.option("--cache-dir", type=click.Path(path_type=str), default=None, help="Custom cache directory")
def main(config, cache_dir):
    if cache_dir:
        global_vars.set_cache_dir(cache_dir)
    logging.info("Loading config...")
    args = load_config(config)

    logging.info("Loading data...")
    dataset1, dataset2, group_names = load_data_from_csv(args)

    logging.info("Running VisDiff pipeline...")
    ranked_hypotheses, metrics = run_pipeline(args, dataset1, dataset2, group_names)
    logging.info("Top hypothesis: %s", ranked_hypotheses[0] if ranked_hypotheses else "None")
    if metrics:
        logging.info("Metrics: %s", metrics)


if __name__ == "__main__":
    main()
