import os
import pickle

from graph_exporter.typing import ImportData


def load(root: str) -> list[ImportData]:
    data: list[ImportData] = list()

    for root, dirs, files in os.walk(root):
        for filename in files:
            timestamp, _ = filename.split("_")

            if filename.endswith(".pkl"):
                print(f"Importing {filename}")
                with open(os.path.join(root, filename), "rb") as f:
                    config, mixup_items = pickle.load(f)

                data.append(
                    ImportData(
                        mixup_items=mixup_items,
                        config=config,
                    )
                )

    return data
