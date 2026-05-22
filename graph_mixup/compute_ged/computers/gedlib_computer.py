import json
import logging
import os
import subprocess
import tempfile

import psutil

from graph_mixup.compute_ged.computers.abstract import AbstractGEDComputer
from graph_mixup.compute_ged.typing import GEDLIBMethod, GEDResult
from graph_mixup.compute_ged.computers.utils import (
    _write_gxl,
    _write_xml_collection,
)
from graph_mixup.ged_database.models import Graph

logger = logging.getLogger(__name__)


class GEDLibComputer(AbstractGEDComputer):
    def __init__(
        self,
        timeout: int,
        lb_threshold: int,
        exec_path: str,
        ged_approx_method: GEDLIBMethod,
    ) -> None:
        super().__init__(timeout, lb_threshold)
        self.exec_path = exec_path
        self.ged_approx_method = ged_approx_method

    def _compute(self, g0: Graph, g1: Graph, lb: int) -> GEDResult:
        data0 = g0.get_pyg_data()
        data1 = g1.get_pyg_data()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write GXL files.
            _write_gxl(
                data0, os.path.join(tmpdir, "graph1.gxl"), graph_id="graph1"
            )
            _write_gxl(
                data1, os.path.join(tmpdir, "graph2.gxl"), graph_id="graph2"
            )

            # Build XML collection.
            xml_coll_path = os.path.join(tmpdir, "collection.xml")
            _write_xml_collection(["graph1.gxl", "graph2.gxl"], xml_coll_path)

            # Prepare and run the GED executable.
            cmd = [
                self.exec_path,
                tmpdir,
                xml_coll_path,
                self.ged_approx_method.value,
            ]
            proc = psutil.Process(os.getpid())
            start = proc.cpu_times()

            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=self.timeout,
            )

            end = proc.cpu_times()
            runtime = int(
                ((end.user + end.system) - (start.user + start.system)) * 1e6
            )

            # Parse JSON output.
            out = json.loads(res.stdout)

        # Extract GED and mapping and return.
        if "graph_edit_distance" not in out:
            logger.info("Unsuccessful.")
            return GEDResult(
                g0.graph_id,
                g1.graph_id,
                -1,
                None,
                None,
                lb,
            )

        ged = out["graph_edit_distance"]
        mapping = {int(k): int(v) for k, v in out.get("node_map", {}).items()}
        res =  GEDResult(g0.graph_id, g1.graph_id, ged, runtime, mapping, lb)

        logger.info(f"Successfully computed:  {res}")
        return res
