import numpy as np
from shapely.geometry import Polygon

from tracking.types import ActorID


def get_params(info):
    # get list of tuples (x,y) of the corner points of parameter info
    
    centroid_x, centroid_y = info[0], info[1]
    box_x, box_y = info[2], info[3]
    heading = info[-1]

    # before_rotation matrix should be 2 x 4 array
    before_rotation = np.array([[-box_x/2, -box_y/2], [-box_x/2, box_y/2], [box_x/2, box_y/2], [box_x/2, -box_y/2]]).T
    rotation_matrix = np.array([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]])
    
    # rotation about the point as the origin, then translate
    after_rotation = (np.matmul(rotation_matrix, before_rotation) + np.array([[centroid_x], [centroid_y]])).T
    return after_rotation.tolist()

def iou_2d(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """Computes 2D intersection over union of two sets of bounding boxes

    Args:
        bboxes1: bounding box set of shape [M, 5], each row corresponding to x, y, l, w, yaw of the bounding box
        bboxes2: bounding box set of shape [N, 5], each row corresponding to x, y, l, w, yaw of the bounding box
    Returns:
        iou_mat: matrix of shape [M, N], where iou_mat[i, j] is the 2D IoU value between bboxes[i] and bboxes[j].
        You should use the Polygon class from the shapely package to compute the area of intersection/union.
    """
    M, N = bboxes1.shape[0], bboxes2.shape[0]
    # TODO: Replace this stub code.
    iou_mat = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            polygon1params = get_params(bboxes1[i])
            polygon2params = get_params(bboxes2[j])
            polygon1 = Polygon(polygon1params)
            polygon2 = Polygon(polygon2params)
            intersect = polygon1.intersection(polygon2).area
            union = polygon1.union(polygon2).area
            iou = intersect / union
            iou_mat[i][j] = iou
    return iou_mat


def geom_2d(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    M, N = bboxes1.shape[0], bboxes2.shape[0]
    geom_mat = np.zeros((M, N))
    lam_t = 5
    lam_size = 3
    lam_yaw = 5

    for i in range(M):
        for j in range(N):

            bbox1_kps = [get_params(bboxes1[i])[1], get_params(bboxes1[i])[3]]  # top-left, bottom-right corners
            bbox2_kps = [get_params(bboxes2[j])[1], get_params(bboxes2[j])[3]]

            # normalized keypoint
            bbox1_t = [(bboxes1[i][0] - bbox1_kps[0][0]) / (bbox1_kps[1][0] - bbox1_kps[0][0]),   # x-coorrd normalized
                        (bboxes1[i][1] - bbox1_kps[0][1]) / (bbox1_kps[1][1] - bbox1_kps[0][1])]  # y-coord normalized
            bbox2_t = [(bboxes2[j][0] - bbox2_kps[0][0]) / (bbox2_kps[1][0] - bbox2_kps[0][0]),   # x-coorrd normalized
                        (bboxes2[j][1] - bbox2_kps[0][1]) / (bbox2_kps[1][1] - bbox2_kps[0][1])]

            t_diff = abs(bbox1_t[0] - bbox2_t[0]) + abs(bbox1_t[1] - bbox2_t[1])

            size_diff = abs(bboxes1[i][2] * bboxes1[i][3] - bboxes2[j][2] * bboxes2[j][3])
            yaw_diff = abs(bboxes1[i][-1] - bboxes2[j][-1])

            geom_mat[i][j] = lam_t * t_diff + lam_size * size_diff + lam_yaw * yaw_diff


    return geom_mat


def motion_2d(bboxes1: np.ndarray, bboxes2: np.ndarray, track_ids, all_tracklets_in_seq) -> np.ndarray:  
    # arg: track_ids for bboxes1 actors, 
    # arg: all_tracklets_in_seq 

    # find the difference in velocities between at frame bboxes1 and at frame bboxes2
    # 1. calculate the velocity of bboxes1 and their previous frame in corresponding tracklets
    # 2. calculate the velocity i.e. displacement between bboxes1 centroid and bboxes2 centroid

    M, N = bboxes1.shape[0], bboxes2.shape[0]
    motion_mat = np.zeros((M, N))
    for i in range(M):
        # calculate the velocity of bboxes1 and their previous frame in corresponding tracklets
        track_id = track_ids[i]
        tracklet = all_tracklets_in_seq[track_id]
        # get velocities from the last 2 tracklet.bboxes_traj bboxes
        velocity1 = np.array([tracklet.bboxes_traj[-2][0], tracklet.bboxes_traj[-2][1]]) - np.array([bboxes1[i][0], bboxes1[i][1]])

        for j in range(N):
            # calculate the velocity i.e. displacement between bboxes1 centroid and bboxes2 centroid
            velocity2 = np.array([bboxes1[i][0], bboxes1[i][1]]) - np.array([bboxes2[j][0], bboxes2[j][1]])
            motion_mat[i][j] = np.linalg.norm(velocity1 - velocity2)

    return motion_mat
