from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Union
from uuid import UUID

import torch

from detection.types import Detections
from .cost import iou_2d

class AssociateMethod(str, Enum):
    GREEDY = "greedy"
    HUNGARIAN = "hungarian"


ActorID = Union[int, str, UUID]


class SingleTracklet:
    def __init__(
        self, frame_ids: List[int], bboxes_traj: List[torch.Tensor], scores: List[float]
    ):
        """Class for a single 2D BEV tracklet.

        Args:
            frame_ids: [N] list of frame ids associated to the detection
            bboxes_traj: [N] list of [5,] tensor (x, y, l, w, yaw_rad) for the bbox trajectory
            scores: [N] list of association scores at each frame
        """
        assert len(frame_ids) == len(bboxes_traj) == len(scores)
        self.frame_ids = frame_ids
        self.bboxes_traj = bboxes_traj
        self.scores = scores

    @property
    def num_steps(self):
        return len(self.bboxes_traj)

    def reset(self):
        self.frame_ids: List[int] = []
        self.bboxes_traj: List[torch.Tensor] = []
        self.scores: List[float] = []

    def insert_new_observation(
        self, new_frame_id: int, new_bbox: torch.Tensor, new_score: float
    ):
        self.frame_ids.append(new_frame_id)
        self.bboxes_traj.append(new_bbox)
        self.scores.append(new_score)
    
    def is_connected(self, other: "SingleTracklet", iou_th=0.1) -> bool:  # change iou_th
        """Check if two tracklets are connected i.e. the bbox in last frame of self tracklet
        overlaps with the bbox in first frame of other tracklet.

        Args:
            other: another tracklet

        Returns:
            True if the two tracklets are connected, False otherwise
        """
        if len(self.frame_ids) == 0 or len(other.frame_ids) == 0:
            return False
        if self.frame_ids[-1] > other.frame_ids[0]:
            return False
        # there exists gap frames between the two tracklets
        last_bbox_self = self.bboxes_traj[-1].unsqueeze(0).numpy()
        first_bbox_other = other.bboxes_traj[0].unsqueeze(0).numpy()
        iou_mat = torch.tensor(iou_2d(last_bbox_self, first_bbox_other))
        iou = iou_mat.squeeze().item()  # for single bbox1 and bbox2 iou is a scalar
        print(f"iou: {iou}")
        if iou > iou_th:  # Assumption: stationary actor via IoU
            return True
        
        return False


class Tracklets:
    """Class to store mappings between object IDs and the associated tracklet."""

    def __init__(self, tracks: Dict[ActorID, SingleTracklet]):
        self.tracks = tracks

    def __len__(self) -> int:
        """Return the number of tacklet."""
        return len(self.tracks)

    @property
    def uids(self):
        return [*self.tracks]

    @classmethod
    def from_seq_labels(cls, frame_ids, seq_labels, voxelizer):
        tracks = {}
        # sort sequence based on frame ids
        for frame_id, actor_labels in sorted(zip(frame_ids, seq_labels)):
            for _, labels in actor_labels:
                dets = Detections(
                    labels.centroids[:, :2],
                    labels.yaws,
                    labels.boxes[:, :2],
                )
                bev_dets = voxelizer.project_detections(dets)
                for uid, centroid, box, yaw in zip(
                    labels.uids, bev_dets.centroids, bev_dets.boxes, bev_dets.yaws
                ):
                    bbox = torch.cat([centroid[:2], box[:2], yaw[None]])
                    if uid in tracks:
                        tracks[uid].insert_new_observation(frame_id, bbox, None)
                    else:
                        tracks[uid] = SingleTracklet([frame_id], [bbox], [None])

        return cls(tracks)


@dataclass
class TrackingInputs:
    """Dataclass for tracking inputs (a sequence of detections results).

    Args:
        bboxes: List of [N x 5] boxes tensor. Each row is (x, y, x_size, y_size, yaw).
        scores: List of [N] detection scores tensor.
    """

    frame_ids: List[int]
    seq_bboxes: List[torch.Tensor]
    seq_scores: List[torch.Tensor]

    @property
    def bboxes(self) -> torch.Tensor:
        """Return the 2D bounding boxes."""
        return self.seq_bboxes

    @property
    def scores(self) -> torch.Tensor:
        """Return the detection scores."""
        return self.seq_scores

    def to(self, device: torch.device) -> "TrackingInputs":
        """Return a copy of the detections moved to another device."""
        return TrackingInputs(
            self.bboxes.to(device),
            self.scores.to(device) if self.scores is not None else None,
        )

    def __len__(self) -> int:
        """Return the number of detections."""
        return len(self.bboxes)
