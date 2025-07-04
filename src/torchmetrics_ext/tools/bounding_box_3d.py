import torch


def get_batch_aabb_ious(batch_boxes_1_bound: torch.Tensor, batch_boxes_2_bound: torch.Tensor) -> torch.Tensor:
    """
    Computes the Intersection over Union (IoU) for a batch of 3D axis-aligned bounding boxes (AABB).

    This function calculates the IoU for two batches of axis-aligned bounding boxes (AABBs), where IoU is
    defined as the volume of the intersection of two bounding boxes divided by the volume of their union.
    The bounding boxes are provided as tensors representing the minimum and maximum coordinates of each
    box in a batch-wise fashion. Both input tensors must have the same shape and correspond to pairs
    of bounding boxes.

    Args:
        batch_boxes_1_bound (torch.Tensor): A tensor of shape (B, 2, D), where B is the batch size and
            D is the dimensionality of the bounding boxes. Represents the minimum and maximum coordinates
            of the first set of bounding boxes.
        batch_boxes_2_bound (torch.Tensor): A tensor of shape (B, 2, D), where B is the batch size and
            D is the dimensionality of the bounding boxes. Represents the minimum and maximum coordinates
            of the second set of bounding boxes.

    Returns:
        torch.Tensor: A 1D tensor of shape (B,), where each element represents the IoU value for the
        corresponding pair of bounding boxes in the input batches.
    """
    eps = torch.finfo(batch_boxes_1_bound.dtype).eps

    # calculate min and max points of the intersection box
    min_inter = torch.max(batch_boxes_1_bound[:, 0], batch_boxes_2_bound[:, 0])
    max_inter = torch.min(batch_boxes_1_bound[:, 1], batch_boxes_2_bound[:, 1])

    # compute intersection volume
    inter_dims = torch.clamp(max_inter - min_inter, min=0)
    intersection_volume = torch.prod(inter_dims, dim=1)

    # compute volumes of the individual boxes
    box_1_volume = torch.prod(batch_boxes_1_bound[:, 1] - batch_boxes_1_bound[:, 0], dim=1)
    box_2_volume = torch.prod(batch_boxes_2_bound[:, 1] - batch_boxes_2_bound[:, 0], dim=1)

    # compute IoUs
    union_volume = box_1_volume + box_2_volume - intersection_volume
    ious = intersection_volume / (union_volume + eps)
    return ious.flatten()



def get_aabb_per_pair_ious(boxes_1_bound: torch.Tensor, boxes_2_bound: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Intersection over Union (IoU) for alll combinations of boxes in two sets of 3D axis-aligned bounding boxes (AABB).

    This function computes the IoU between each pair of AABBs in two sets， where IoU is defined as the volume of the
    intersection of two bounding boxes divided by the volume of their union. The bounding boxes are provided as
    tensors representing the minimum and maximum coordinates of each box.

    Args:
        boxes_1_bound (torch.Tensor): A tensor of shape (N, 2, 3), where N is the batch size and
            D is the dimensionality of the bounding boxes. Represents the minimum and maximum coordinates
            of the first set of bounding boxes.
        boxes_2_bound (torch.Tensor): A tensor of shape (N, 2, 3), where M is the batch size and
            D is the dimensionality of the bounding boxes. Represents the minimum and maximum coordinates
            of the second set of bounding boxes.

    Returns:
        torch.Tensor: A tensor of shape (N, M) containing the pairwise IoU values
            for all box combinations. The element at position (i, j) represents the
            IoU between the i-th box from the first set and the j-th box from the
            second set.
    """
    eps = torch.finfo(boxes_1_bound.dtype).eps

    # calculate min and max points of the intersection box
    max_min = torch.max(boxes_1_bound[:, 0][..., None, :], boxes_2_bound[:, 0][None, ...])  # NxMx3
    min_max = torch.min(boxes_1_bound[:, 1][..., None, :], boxes_2_bound[:, 1][None, ...])  # NxMx3

    # compute intersection volume
    inter_dims = torch.clamp(min_max - max_min, min=0)  # NxMx3
    intersection_volume = inter_dims[..., 0] * inter_dims[..., 1] * inter_dims[..., 2]  # NxM

    # compute volumes of the individual boxes
    volume_1 = torch.prod(boxes_1_bound[:, 1] - boxes_1_bound[:, 0], dim=1)  # N
    volume_2 = torch.prod(boxes_2_bound[:, 1] - boxes_2_bound[:, 0], dim=1)  # M

    # compute IoUs
    union_volume = volume_1[..., None] + volume_2 - intersection_volume  # NxM
    ious = intersection_volume / (union_volume + eps)  # NxM
    return ious
