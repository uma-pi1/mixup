import logging
import math
import os
from argparse import ArgumentParser
from dataclasses import dataclass
from enum import Enum
from typing import cast

import numpy as np
from matplotlib import pyplot as plt

from graph_mixup.config.typing import DatasetName
from graph_mixup.ged_database.handlers.ged_plots_database_handler import (
    GEDPlotsDatabaseHandler,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

colors = plt.get_cmap("tab20")
target_color = colors(1)
target_text_color = colors(0)
danger_color = colors(7)
danger_text_color = colors(6)


@dataclass
class Args:
    dataset_name: DatasetName
    method_name: str
    lam: float
    batch_size: int
    out_dir: str
    max_batch_count: int
    log: bool
    grid: bool
    tol: float
    text: bool


def parse_args() -> Args:
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=DatasetName,
        choices=[cast(Enum, dataset).value for dataset in DatasetName],
        required=True,
    )
    parser.add_argument(
        "--method_name",
        type=str,
        help="Choose (pre-batch) mixup method.",
        required=False,
    )
    parser.add_argument("--lam", type=float, required=True, help="Mixup lambda")
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="limit/offset in SQL queries",
    )
    parser.add_argument(
        "--out_dir", type=str, required=True, help="output directory"
    )
    parser.add_argument(
        "--max_batch_count",
        type=int,
        default=math.inf,
        help="max number of processed batches (optional)",
    )
    parser.add_argument(
        "--log",
        action="store_true",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--text",
        action="store_true",
    )
    return Args(**vars(parser.parse_args()))


def mpd(
    parents_ged: int,
    p0_ged: int,
    p1_ged: int,
    lam: float,
    not_inverse: bool,
) -> float:
    if parents_ged == 0:
        return np.nan

    if not_inverse:
        logger.info("Used normal (not inverse) MPD.")
        return (
            abs(p0_ged - lam * parents_ged) / parents_ged
            + abs(p1_ged - (1 - lam) * parents_ged) / parents_ged
        )
    else:
        return (
            abs(p1_ged - lam * parents_ged) / parents_ged
            + abs(p0_ged - (1 - lam) * parents_ged) / parents_ged
        )


class GEDPlotter:
    def __init__(self, args: Args):
        self.dataset_name = args.dataset_name
        self.method_name = args.method_name
        self.mixup_lambda = args.lam
        self.batch_size = args.batch_size
        self.out_dir = args.out_dir
        self.max_batch_count = args.max_batch_count
        self.log = args.log
        self.grid = args.grid
        self.tol = args.tol
        self.text = args.text

        self.db_manager = GEDPlotsDatabaseHandler()

    def create_plots(self) -> None:
        plt.rcParams.update({"font.size": 28})
        linewidth = 2
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlabel(r"Output-to-input distances $d(G_i, G_M)$")
        ax.set_ylabel(r"Input-to-input distance $d(G_1, G_2)$")

        max_ged, batch_count, count = 0, 0, 0
        mpds: list[float] = []
        for results in self.db_manager.get_mixup_ged_triples(
            self.dataset_name,
            self.method_name,
            self.mixup_lambda,
            self.batch_size,
            self.tol,
        ):
            for triple in results:
                count += 1
                max_ged = max(
                    max_ged, triple.parents_ged, *triple.parent_child_geds
                )

                ged_p0 = triple.parent_child_geds[0]
                ged_p1 = triple.parent_child_geds[1]

                mpds.append(
                    mpd(
                        triple.parents_ged,
                        ged_p0,
                        ged_p1,
                        self.mixup_lambda,
                        self.method_name == "geomix"
                        or self.method_name == "geomix_se"
                        or self.method_name == "fgw_mixup"
                        or self.method_name == "ged_mixup",
                    )
                )

                # Make those GEDs visible (as a dot).
                if ged_p0 == ged_p1:
                    ged_p0 -= 0.1
                    ged_p1 += 0.1

                ax.plot(
                    [ged_p0, ged_p1],
                    [triple.parents_ged, triple.parents_ged],
                    alpha=0.2,
                    color="black",
                    linewidth=2 * linewidth,
                )
                logger.info(f"Triple: {triple} with MPD: {mpds[-1]}")

            batch_count += 1
            logger.info(f"Batch {batch_count}: {len(results)} results")

            if batch_count >= self.max_batch_count:
                logging.info("max batch count reached")
                break

        x_vals = np.linspace(0, 1.2 * max_ged, 500)

        ax.fill_between(x_vals, 0, x_vals, color=danger_color, alpha=0.3)

        if self.text:
            ax.text(
                0.3 * max_ged,
                0.15 * max_ged,
                "Non-interpolation area\n($d(G_i, G_M) > d(G_1, G_2)$)",
                color=danger_text_color,
                fontsize=34,
            )

        # Lambda-based line or band with tolerance
        if self.tol > 1e-3:
            lower_lam = self.mixup_lambda - self.tol / 2
            upper_lam = self.mixup_lambda + self.tol / 2

            ax.fill_between(
                x_vals,
                (1 / upper_lam) * x_vals,
                (1 / lower_lam) * x_vals,
                color=target_color,
                # alpha=0.4,
                label=f"λ ∈ [{lower_lam:.2f}, {upper_lam:.2f}]",
            )

            ax.fill_between(
                x_vals,
                (1 / (1 - lower_lam)) * x_vals,
                (1 / (1 - upper_lam)) * x_vals,
                color=target_color,
                # alpha=0.4,
            )
        else:
            ax.plot(
                x_vals,
                (1 / self.mixup_lambda) * x_vals,
                color=target_color,
                linewidth=linewidth,
                label=r"$y = (1/\lambda) \cdot x$",
            )
            if self.mixup_lambda != 1:
                ax.plot(
                    x_vals,
                    (1 / (1 - self.mixup_lambda)) * x_vals,
                    color=target_color,
                    linewidth=linewidth,
                    label=r"$y = (1/(1−\lambda)) \cdot x$",
                )

        if self.text:
            ax.text(
                0.55 * max_ged,
                1.1 * max_ged,
                "TARGET"
                if self.mixup_lambda == 1 / 2
                else r"$\leftarrow$ TARGETS $\rightarrow$",
                fontsize=34,
                color=target_text_color,
                ha="center",
                va="center",
                fontweight="bold",
            )

        if self.log:
            ax.set_xscale("log")
            ax.set_yscale("log")
        else:
            ax.set_xlim(0, 1.2 * max_ged)
            ax.set_ylim(0, 1.2 * max_ged)

        if self.grid:
            ax.grid(
                True,
                which="both",
                linestyle="-",
                linewidth=0.5,
                color="lightgray",
            )
            ax.set_xticks(range(int(1.2 * max_ged) + 1))
            ax.set_yticks(range(int(1.2 * max_ged) + 1))

        os.makedirs(self.out_dir, exist_ok=True)
        file = os.path.join(
            self.out_dir,
            f"{self.method_name}_{self.dataset_name}_lambda-{self.mixup_lambda}"
            f"_count-{count}_log-{self.log}_tol-{self.tol}"
            f"_text-{self.text}_mpd-{round(np.nanmean(mpds), 4)}.pdf",
        )

        if not os.path.isfile(file):
            plt.savefig(file, bbox_inches="tight")
        else:
            logging.info(f"File already exists: {file}")


if __name__ == "__main__":
    plotter = GEDPlotter(parse_args())
    plotter.create_plots()
