"""
Tile stitching for inference on large images.

Handles merging predictions from overlapping tiles using Non-Maximum Suppression (NMS)
to remove duplicate detections in overlap regions.
"""

import torch
from typing import List, Tuple

try:
    from torchvision.ops import box_convert, nms
except ImportError:
    from torchvision import ops
    box_convert = ops.box_convert
    nms = ops.nms


def stitch_tile_predictions(
    tile_predictions: List[Tuple[torch.Tensor, Tuple[int, int]]],
    image_size: Tuple[int, int],
    tile_size: int,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5,
) -> torch.Tensor:
    """
    Merge predictions from overlapping tiles using NMS.

    Args:
        tile_predictions: List of (predictions, (y_offset, x_offset)) tuples
            Each predictions tensor has shape (N, 5+num_classes) where:
            - predictions[:, 0]: x center (normalized to tile, 0-1)
            - predictions[:, 1]: y center (normalized to tile, 0-1)
            - predictions[:, 2]: width (normalized to tile)
            - predictions[:, 3]: height (normalized to tile)
            - predictions[:, 4]: objectness score
            - predictions[:, 5:]: class probabilities
        image_size: Original image (H, W) in pixels
        tile_size: Size of tiles in pixels
        iou_threshold: NMS threshold for removing duplicates (default: 0.5)
        score_threshold: Minimum objectness score to keep (default: 0.5)

    Returns:
        Merged predictions in original image coordinates (N, 5+num_classes)
        Coordinates are normalized to the full image (0-1)
    """
    H, W = image_size
    all_predictions = []

    for preds, (y_off, x_off) in tile_predictions:
        if preds.shape[0] == 0:
            continue

        # Filter by objectness score
        keep_mask = preds[:, 4] >= score_threshold
        if not keep_mask.any():
            continue

        preds = preds[keep_mask]

        # Convert from tile coordinates (0-1 normalized to tile)
        # to full image coordinates (0-1 normalized to image)
        preds_image = preds.clone()

        # Transform center coordinates
        preds_image[:, 0] = (preds[:, 0] * tile_size + x_off) / W  # x center
        preds_image[:, 1] = (preds[:, 1] * tile_size + y_off) / H  # y center

        # Transform width/height
        preds_image[:, 2] = preds[:, 2] * tile_size / W  # width
        preds_image[:, 3] = preds[:, 3] * tile_size / H  # height

        all_predictions.append(preds_image)

    if not all_predictions:
        # No predictions passed threshold
        return torch.zeros((0, tile_predictions[0][0].shape[1]))

    # Concatenate all predictions
    merged = torch.cat(all_predictions, dim=0)

    # Apply NMS to remove duplicates from overlapping regions
    merged = nms_predictions(merged, iou_threshold)

    return merged


def nms_predictions(
    predictions: torch.Tensor,
    iou_threshold: float
) -> torch.Tensor:
    """
    Apply Non-Maximum Suppression to remove duplicate detections.

    Args:
        predictions: (N, 5+num_classes) tensor with:
            - predictions[:, :4]: bounding boxes [cx, cy, w, h] (normalized)
            - predictions[:, 4]: objectness scores
            - predictions[:, 5:]: class probabilities
        iou_threshold: IoU threshold for NMS

    Returns:
        Filtered predictions after NMS
    """
    if predictions.shape[0] == 0:
        return predictions

    boxes = predictions[:, :4]  # cx, cy, w, h (normalized)
    scores = predictions[:, 4]  # objectness

    # Convert from cxcywh to xyxy format for NMS
    boxes_xyxy = box_convert(boxes, "cxcywh", "xyxy")

    # Apply NMS
    keep_indices = nms(boxes_xyxy, scores, iou_threshold)

    return predictions[keep_indices]


def tile_image_for_inference(
    image: torch.Tensor,
    tile_size: int,
    overlap: int
) -> List[Tuple[torch.Tensor, Tuple[int, int]]]:
    """
    Tile an image into overlapping patches for inference.

    Args:
        image: (C, H, W) tensor
        tile_size: Size of tiles in pixels
        overlap: Overlap between tiles in pixels

    Returns:
        List of (tile_tensor, (y_offset, x_offset)) tuples
    """
    stride = tile_size - overlap
    C, H, W = image.shape
    tiles = []

    # Compute tile positions
    y_positions = list(range(0, H - tile_size + 1, stride))
    x_positions = list(range(0, W - tile_size + 1, stride))

    # Ensure we cover the entire image
    if not y_positions or y_positions[-1] + tile_size < H:
        y_positions.append(H - tile_size)
    if not x_positions or x_positions[-1] + tile_size < W:
        x_positions.append(W - tile_size)

    for y in y_positions:
        for x in x_positions:
            tile = image[:, y:y+tile_size, x:x+tile_size]
            tiles.append((tile, (y, x)))

    return tiles


def predict_full_image(
    model,
    image: torch.Tensor,
    tile_size: int,
    overlap: int,
    device: torch.device = torch.device('cpu'),
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5,
) -> torch.Tensor:
    """
    Run inference on a full image using tiling and stitching.

    Args:
        model: YOGO model
        image: (C, H, W) tensor (full image)
        tile_size: Size of tiles in pixels
        overlap: Overlap between tiles in pixels
        device: Device to run inference on
        iou_threshold: NMS threshold for stitching
        score_threshold: Minimum objectness score

    Returns:
        Predictions for full image (N, 5+num_classes)
    """
    model.eval()

    # Tile the image
    tiles = tile_image_for_inference(image, tile_size, overlap)

    # Run inference on each tile
    tile_predictions = []
    with torch.no_grad():
        for tile, offset in tiles:
            tile = tile.unsqueeze(0).to(device)  # Add batch dimension
            preds = model(tile)  # Get predictions

            # Extract predictions from output tensor
            # YOGO output shape: (1, 5+num_classes, Sy, Sx)
            # Need to convert to (N, 5+num_classes) format

            # This is a simplified version - actual implementation depends on
            # how YOGO formats its output. You may need to adapt this.
            # Typically you'd threshold by objectness and extract boxes.

            tile_predictions.append((preds.squeeze(0), offset))

    # Stitch predictions
    H, W = image.shape[1:]
    stitched = stitch_tile_predictions(
        tile_predictions,
        (H, W),
        tile_size,
        iou_threshold,
        score_threshold
    )

    return stitched
