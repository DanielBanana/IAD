"""
shift.py

A `ShiftSession` models one shift worker's inspection session: the on-disk
conventions for where captured images land (tmp/<name>/<dataset>/<product>/<timestamp>/)
and the running inspection stats (images inspected, anomalies found), kept
independent of the console/threading/queue machinery in ad_worker.py -- the
same way Product/DatasetSession model their own domains in setup.py.

Captured images are staged one at a time in `stagingDir` (mirroring how
dataset_build/annotate stage camera captures in a `new/` folder, since the
camera's shutter is triggered by external hardware, not this code) and then
claimed into their own timestamped folder via `claim_next_image_slot`, ready
to be run through AnomalyDetectionManager.inference() as a one-image batch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING

from run_registry import generate_run_id

if TYPE_CHECKING:
    from setup import Product

logger = logging.getLogger(__name__)


@dataclass
class ShiftSession:
    """
    One shift's worth of one-image-at-a-time inspection.

    `runId` is generated once and reused across every `manager.inference()`
    call for the whole shift, so they all write into a single shared results
    directory instead of a fresh one per image (see AD_Worker's shift_inspect).
    """
    name: str
    runId: str
    sessionDir: Path            # tmp/<name>/<datasetName>/<productName>
    stagingDir: Path            # sessionDir/_staging -- camera.imagePath points here
    datasetName: str
    productName: str
    startTime: datetime
    imageCounter: int = 0       # global, keeps incrementing across the whole shift
    imagesInspected: int = 0
    anomaliesDetected: int = 0

    @property
    def anomalyPercentage(self) -> float:
        if self.imagesInspected == 0:
            return 0.0
        return self.anomaliesDetected / self.imagesInspected * 100

    def claim_next_image_slot(self, source: Path) -> Path:
        """
        Reserve the destination path for `source`: a fresh timestamped folder
        under `sessionDir`, containing a single file named after the shift's
        global image counter (not reset per folder, so the filename alone
        reflects the image's position in the whole shift). Increments the
        counter. Does not touch `source` -- the caller moves it.
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S%f")
        imageDir = self.sessionDir / timestamp
        imageDir.mkdir(parents=True, exist_ok=True)
        destPath = imageDir / f"{self.imageCounter:03d}{source.suffix}"
        self.imageCounter += 1
        return destPath

    def record_result(self, anomalous: bool) -> None:
        """Update running stats after one image has been inferenced."""
        self.imagesInspected += 1
        if anomalous:
            self.anomaliesDetected += 1

    def to_summary_dict(self, endTime: datetime) -> Dict[str, Any]:
        """Serializable shift summary, written as shift_summary.yaml by shift_end."""
        return {
            "sessionName": self.name,
            "runId": self.runId,
            "datasetName": self.datasetName,
            "productName": self.productName,
            "startTime": self.startTime.isoformat(),
            "endTime": endTime.isoformat(),
            "imagesInspected": self.imagesInspected,
            "anomaliesDetected": self.anomaliesDetected,
            "anomalyPercentage": self.anomalyPercentage,
        }


def start_shift_session(product: "Product", sessionName: str, tmpRoot: Path = Path("tmp")) -> ShiftSession:
    """
    Create the on-disk session/staging folders for a new shift and return the
    ShiftSession tracking it. Pure creation -- no camera/manager side effects;
    callers (AD_Worker) wire the camera and manager up separately.
    """
    datasetName = product.datasetConfig.name
    productName = product.name

    sessionDir = tmpRoot / sessionName / datasetName / productName
    stagingDir = sessionDir / "_staging"
    stagingDir.mkdir(parents=True, exist_ok=True)

    session = ShiftSession(
        name=sessionName,
        runId=generate_run_id(label=sessionName),
        sessionDir=sessionDir,
        stagingDir=stagingDir,
        datasetName=datasetName,
        productName=productName,
        startTime=datetime.now(),
    )
    logger.info(f"Started shift session {sessionName!r} for {productName} ({datasetName}); runId={session.runId}")
    return session
