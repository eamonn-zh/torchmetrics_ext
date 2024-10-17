import torch


def get_batch_aabb_pair_ious(batch_boxes_1_bound: torch.Tensor, batch_boxes_2_bound: torch.Tensor) -> torch.Tensor:
    """
    :param batch_boxes_1_bound: a batch of axis-aligned bounding box min max bounds (B, 2, 3)
    :param batch_boxes_2_bound: a batch of axis-aligned bounding box min max bounds (B, 2, 3)
    :return: IoU values for each pair of axis-aligned bounding boxes (B, )
    """
    eps = torch.finfo(batch_boxes_1_bound.dtype).eps

    # calculate min and max points of the intersection box
    min_inter = torch.max(batch_boxes_1_bound[:, 0], batch_boxes_2_bound[:, 0])
    max_inter = torch.min(batch_boxes_1_bound[:, 1], batch_boxes_2_bound[:, 1])

    # compute intersection volume
    inter_dims = torch.clamp(max_inter - min_inter, min=0)
    intersection_volume = torch.prod(inter_dims, dim=1)

    # compute volumes of the individual boxes
    dims_1 = batch_boxes_1_bound[:, 1] - batch_boxes_1_bound[:, 0]
    dims_2 = batch_boxes_2_bound[:, 1] - batch_boxes_2_bound[:, 0]
    box_1_volume = torch.prod(dims_1, dim=1)
    box_2_volume = torch.prod(dims_2, dim=1)

    # compute IoU
    union_volume = box_1_volume + box_2_volume - intersection_volume
    ious = intersection_volume / (union_volume + eps)
    return ious.flatten()



def get_batch_aabb_ious(boxes_1: torch.Tensor, boxes_2: torch.Tensor) -> torch.Tensor:

    eps = torch.finfo(boxes_1.dtype).eps

    # calculate intersections
    max_min = torch.max(boxes_1[:, 0][..., None, :], boxes_2[:, 0][None, ...])  # NxMx3
    min_max = torch.min(boxes_1[:, 1][..., None, :], boxes_2[:, 1][None, ...])  # NxMx3
    inter_dims = torch.clamp(min_max - max_min, min=0)  # NxMx3
    inter_vol = inter_dims[..., 0] * inter_dims[..., 1] * inter_dims[..., 2]  # NxM

    # calculate volumes of original boxes
    vol1 = torch.prod(boxes_1[:, 1] - boxes_1[:, 0], dim=1)  # N
    vol2 = torch.prod(boxes_2[:, 1] - boxes_2[:, 0], dim=1)  # M

    # calculate union
    union = vol1[..., None] + vol2 - inter_vol  # NxM

    # calculate IoU
    iou = inter_vol / (union + eps)  # NxM

    return iou
