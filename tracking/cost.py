import numpy as np
from shapely.geometry import Polygon


def get_params(info):
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
