import logging
import re
import subprocess

from graph_mixup.compute_ged.computers.abstract import AbstractGEDComputer
from graph_mixup.compute_ged.typing import (
    GEDResult,
    MissingGEDException,
    MissingMappingException,
)
from graph_mixup.compute_ged.computers.utils import (
    _make_temp_file,
    _remove_temp_file,
)
from graph_mixup.ged_database.models import Graph


logger = logging.getLogger(__name__)


class ExactGEDComputer(AbstractGEDComputer):
    def _compute(self, g0: Graph, g1: Graph, lb: int) -> GEDResult:
        assert g0.num_nodes() <= g1.num_nodes()

        file0 = _make_temp_file(g0)
        file1 = _make_temp_file(g1)

        try:
            command = ["./ged", "-q", file0, "-d", file1, "-g"]
            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout,
            )
            output: str = process.stdout.decode()
            err_output: str = process.stderr.decode()

            # Extract GED
            ged_match = re.search(r"GED: (\d+)", output)
            if ged_match:
                ged = int(ged_match.group(1))
            else:
                raise MissingGEDException(
                    "GED value not found in output:"
                    + "\nSTDOUT:\n "
                    + output
                    + "\nSTDERR:\n "
                    + err_output
                )

            # Extract mapping
            mapping_match = re.search(r"Mapping: (.+)", output)
            if mapping_match:
                mapping: dict[int, int] = {}
                pairs = mapping_match.group(1).split(", ")
                for pair in pairs:
                    if "->" in pair:
                        q, g = map(int, pair.split(" -> "))
                        mapping[q] = g
            else:
                raise MissingMappingException(
                    "Mapping not found in output:"
                    + "\nSTDOUT:\n "
                    + output
                    + "\nSTDERR:\n "
                    + err_output
                )

            # ===
            # Extract total time. For some unknown reason, time is not always
            # present in the binary's output, hence None is also accepted here.
            # ===
            total_time_match = re.search(
                r"Total time: ([\d,]+) \(microseconds\)", output
            )
            time = (
                int(total_time_match.group(1).replace(",", ""))
                if total_time_match
                else None
            )
            res = GEDResult(g0.graph_id, g1.graph_id, ged, time, mapping, lb)
            logger.info(f"Successfully computed: {res}")
            return res

        except subprocess.TimeoutExpired:
            logger.info("Not successful due to timeout.")
            return GEDResult(
                g0.graph_id,
                g1.graph_id,
                -1,
                self.timeout * 1_000_000,
                None,
                lb,
            )  # Time in microseconds, hence multiplied by 1e6.

        finally:
            _remove_temp_file(file0)
            _remove_temp_file(file1)
