import json
import os
import pickle

import matplotlib.pyplot as plt
from tqdm import tqdm

from detection.modules.voxelizer import VoxelizerConfig
from detection.pandaset.dataset import PandasetConfig
from detection.pandaset.util import LabelClass
from tracking.dataset import OfflineTrackingDataset
from tracking.metrics.evaluator import Evaluator
from tracking.tracker import Tracker
from tracking.types import AssociateMethod, Tracklets
from tracking.visualization import plot_tracklets


def track(
    dataset_path,
    detection_path="tracking/detection_results/csc490_detector",
    result_path="tracking/tracking_results/results.pkl",
    tracker_associate_method=AssociateMethod.HUNGARIAN,
):
    print(f"Loading Pandaset from {dataset_path}")
    print(f"Loading dumped detection results from {detection_path}")
    voxelizer_config = VoxelizerConfig(
        x_range=(-76.0, 76.0),
        y_range=(-50.0, 50.0),
        z_range=(0.0, 10.0),
        step=0.25,
    )
    tracking_dataset = OfflineTrackingDataset(
        PandasetConfig(
            basepath=dataset_path,
            classes_to_keep=[LabelClass.CAR],
        ),
        detection_path,
        voxelizer_config,
    )
    print(
        f"Tracking with association method {AssociateMethod(tracker_associate_method)}"
    )
    tracking_results = {}
    for tracking_data in tqdm(tracking_dataset):
        seq_id = tracking_data.sequence_id
        tracking_inputs = tracking_data.tracking_inputs
        tracking_label = tracking_data.tracking_labels
        tracker = Tracker(
            track_steps=80, associate_method=AssociateMethod(tracker_associate_method)
        )
        # tracker.tracks is a dictionary of Dict[ActorID, SingleTracklet]

        # for debugging: design small tracklets where scores = 1 for all
        # tracking_inputs = ?
        tracker.track(tracking_inputs.bboxes, tracking_inputs.scores)

        # post-process the tracklets to connect tracklets i.e. reduce fragmentation
        # look at all tracklets (after calling track), and connect similar actors
        # predict re-appearance location, or somewhere in between
            # actors close to each other problem: higher match in Iou greedy/hungarian, and threshold
            # bonus when exploring other ways of predicting for occlusion handling
        
        actors_to_remove = []
        actors_looked_at = []
        for actor_id1 in tracker.tracks:
            for actor_id2 in tracker.tracks:
                if actor_id1 == actor_id2 or actor_id2 in actors_looked_at or actor_id2 in actors_to_remove:  
                    continue  # skip all actors started at the same frame or previous frame, or to be removed (i.e. matched)
                if tracker.tracks[actor_id1].is_connected(tracker.tracks[actor_id2]):
                    # append actor_id2 tracklet to actor_id1 tracklet  [TODO: add middle preds]
                    tracker.tracks[actor_id1].frame_ids.extend(tracker.tracks[actor_id2].frame_ids)
                    tracker.tracks[actor_id1].bboxes_traj.extend(tracker.tracks[actor_id2].bboxes_traj)
                    tracker.tracks[actor_id1].scores.extend(tracker.tracks[actor_id2].scores)
                    actors_to_remove.append(actor_id2)
            actors_looked_at.append(actor_id1)

        for actor_id in actors_to_remove:
            removed = tracker.tracks.pop(actor_id)
            print("removed actor_id, frames_ids, traj, scores:", actor_id, removed.frame_ids, removed.bboxes_traj, removed.scores)

        tracking_pred = Tracklets(tracker.tracks)
        save_dict = {
            "sequence_id": seq_id,
            "tracking_label": tracking_label,
            "tracking_pred": tracking_pred,
        }
        tracking_results[seq_id] = save_dict

    print(f"Saving tracking results to {result_path}")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "wb") as f:
        pickle.dump(tracking_results, f)


def visualize(result_path="tracking/tracking_results/results.pkl"):
    viz_path = os.path.join(os.path.dirname(result_path), "viz")
    os.makedirs(viz_path, exist_ok=True)

    with open(result_path, "rb") as f:
        results_dict = pickle.load(f)

    for seq_id, result_dict in tqdm(results_dict.items()):
        tracking_label = result_dict["tracking_label"]
        tracking_pred = result_dict["tracking_pred"]
        fig, _ = plot_tracklets(
            tracking_pred,
            title=f"Estimated Tracklets for Pandaset Log{seq_id:03d} in World Frame",
        )
        fig.savefig(
            os.path.join(viz_path, f"log{seq_id:03d}_track_est.png"),
        )
        fig, _ = plot_tracklets(
            tracking_label,
            title=f"Ground-Truth Tracklets for Pandaset Log{seq_id:03d} in World Frame",
        )
        fig.savefig(os.path.join(viz_path, f"log{seq_id:03d}_track_gt.png"))
        plt.close("all")


def evaluate(result_path="tracking/tracking_results/results.pkl"):
    with open(result_path, "rb") as f:
        results_dict = pickle.load(f)

    evaluator = Evaluator()
    for seq_id, result_dict in tqdm(results_dict.items()):
        tracking_label = result_dict["tracking_label"]
        tracking_pred = result_dict["tracking_pred"]
        eval_results = evaluator.evaluate(tracking_label, tracking_pred)
        print(f"[Sequence: {seq_id:03d}]", eval_results)

    final_results_mean = evaluator.aggregate("mean")
    final_results_median = evaluator.aggregate("median")
    print(f"[Results (mean)", final_results_mean)
    print(f"[Results (median)", final_results_median)


if __name__ == "__main__":
    import fire

    fire.Fire()
