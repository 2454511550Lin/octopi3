"""
Compute object detection metrics for YOGO models.

This module evaluates trained YOGO models on test datasets, computing comprehensive
metrics following ML journal best practices:
- Precision, Recall, F1 at various confidence thresholds
- Average Precision (AP) / mAP
- Matthews Correlation Coefficient (MCC)
- Per-class metrics and confusion matrix
- Sample-level metrics (per-FOV detection rate, count correlation)
- Precision-Recall curves for threshold analysis

Outputs are organized under results/{experiment_id}/metrics/:
- predictions/  - cached model outputs (avoids re-running inference)
- thresholds/   - metrics at each confidence threshold
- curves/       - P-R curves for plotting
- visualizations/ - GT vs Predicted images

Usage:
    python -m yogo.compute_metrics checkpoint.pth dataset.yaml
    python -m yogo.compute_metrics checkpoint.pth dataset.yaml --conf-thresholds 0.3 0.5 0.7 0.9
    python -m yogo.compute_metrics checkpoint.pth dataset.yaml --split test
    python -m yogo.compute_metrics checkpoint.pth dataset.yaml --save-images --vis-conf 0.5
    python -m yogo.compute_metrics checkpoint.pth dataset.yaml --load-predictions  # skip inference
"""

import torch
import json
import argparse
import numpy as np
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import torchvision.ops as ops

from yogo.model import YOGO
from yogo.data.yogo_dataloader import get_datasets
from yogo.data.dataset_definition_file import DatasetDefinition
from yogo.utils import choose_device


# Colors for visualization (BGR format for OpenCV)
# positive = green, unsure = yellow
CLASS_COLORS_GT = [(0, 255, 0), (0, 255, 255)]  # GT boxes: green, yellow
CLASS_COLORS_PRED = [(255, 0, 0), (255, 165, 0)]  # Pred boxes: blue, orange


def save_predictions_cache(
    cache_dir: Path,
    all_outputs: List[torch.Tensor],
    all_labels: List[torch.Tensor],
    tile_metadata: List[Dict],
    model_config: Dict,
) -> None:
    """
    Save model outputs and labels to disk for reuse without re-running inference.

    Args:
        cache_dir: Directory to save the cache
        all_outputs: List of model output tensors
        all_labels: List of label tensors
        tile_metadata: List of tile metadata dicts
        model_config: Model configuration dict
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Save as a single file with all data
    cache_data = {
        'outputs': torch.stack(all_outputs),
        'labels': torch.stack(all_labels),
    }
    torch.save(cache_data, cache_dir / "predictions.pt")

    # Save tile metadata and model config as JSON
    metadata = {
        'model_config': model_config,
        'num_samples': len(all_outputs),
        'tile_metadata': [
            {
                'fov_path': str(t['fov_path']) if t and 'fov_path' in t else None,
                'offset': list(t['offset']) if t and 'offset' in t else None,
                'image_size': list(t['image_size']) if t and 'image_size' in t else None,
            } if t else None
            for t in tile_metadata
        ]
    }
    with open(cache_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved predictions cache to: {cache_dir}")


def load_predictions_cache(cache_dir: Path) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Dict], Dict]:
    """
    Load cached model outputs and labels.

    Args:
        cache_dir: Directory containing the cache

    Returns:
        Tuple of (all_outputs, all_labels, tile_metadata, model_config)
    """
    cache_file = cache_dir / "predictions.pt"
    metadata_file = cache_dir / "metadata.json"

    if not cache_file.exists() or not metadata_file.exists():
        raise FileNotFoundError(f"Cache not found at {cache_dir}")

    # Load tensors
    cache_data = torch.load(cache_file, map_location='cpu')
    all_outputs = [cache_data['outputs'][i] for i in range(cache_data['outputs'].shape[0])]
    all_labels = [cache_data['labels'][i] for i in range(cache_data['labels'].shape[0])]

    # Load metadata
    with open(metadata_file) as f:
        metadata = json.load(f)

    # Reconstruct tile metadata with Path objects
    tile_metadata = []
    for t in metadata['tile_metadata']:
        if t and t.get('fov_path'):
            tile_metadata.append({
                'fov_path': Path(t['fov_path']),
                'offset': tuple(t['offset']) if t.get('offset') else None,
                'image_size': tuple(t['image_size']) if t.get('image_size') else None,
            })
        else:
            tile_metadata.append(None)

    print(f"Loaded {len(all_outputs)} cached predictions from: {cache_dir}")
    return all_outputs, all_labels, tile_metadata, metadata['model_config']


def compute_mcc(tp: int, fp: int, tn: int, fn: int) -> float:
    """
    Compute Matthews Correlation Coefficient.

    MCC is considered superior to F1 for imbalanced datasets (Chicco & Jurman 2020).
    Range: [-1, 1] where 1 is perfect, 0 is random, -1 is inverse prediction.

    Args:
        tp: True positives
        fp: False positives
        tn: True negatives
        fn: False negatives

    Returns:
        MCC value
    """
    numerator = tp * tn - fp * fn
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_pr_curve(
    all_outputs: List[torch.Tensor],
    all_labels: List[torch.Tensor],
    Sx: int,
    Sy: int,
    iou_threshold: float = 0.5,
    nms_iou_threshold: float = 0.5,
    num_thresholds: int = 21,
) -> Dict:
    """
    Compute precision-recall curve by sweeping confidence thresholds.

    Optimized: extracts all predictions ONCE at threshold=0, then filters by
    threshold during sweep. This is much faster than re-extracting for each threshold.

    Args:
        all_outputs: List of model output tensors
        all_labels: List of label tensors
        Sx, Sy: Grid dimensions
        iou_threshold: IoU threshold for matching
        nms_iou_threshold: IoU threshold for NMS
        num_thresholds: Number of threshold points (default 21 for 0.00-1.00 at 0.05 increments)

    Returns:
        Dict with 'thresholds', 'precision', 'recall' arrays
    """
    # Pre-extract ALL predictions at threshold=0 (keep objectness score for filtering)
    print("  Pre-extracting all predictions...")
    all_predictions_raw = []  # List of predictions with scores
    all_ground_truths = []

    for outputs, labels in zip(all_outputs, all_labels):
        # Extract at threshold 0 (get all predictions with their scores)
        preds = extract_predictions_from_grid(
            outputs, Sx, Sy, conf_threshold=0.0,
            apply_nms_filter=False  # Don't apply NMS yet - do it per threshold
        )
        gts = extract_ground_truth_from_grid(labels, Sx, Sy)
        all_predictions_raw.append(preds)
        all_ground_truths.append(gts)

    # Sweep thresholds
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    precisions = []
    recalls = []

    print(f"  Sweeping {num_thresholds} thresholds...")
    for conf_thresh in thresholds:
        # Filter predictions by threshold and apply NMS
        all_predictions_filtered = []
        for preds_raw in all_predictions_raw:
            # Filter by threshold
            preds_filtered = [p for p in preds_raw if p[4] >= conf_thresh]
            # Apply NMS
            if len(preds_filtered) > 0:
                preds_filtered = apply_nms(preds_filtered, nms_iou_threshold)
            all_predictions_filtered.append(preds_filtered)

        metrics = compute_metrics_at_threshold(
            all_predictions_filtered, all_ground_truths, iou_threshold, num_classes=2
        )

        tp = metrics["overall"]["tp"]
        fp = metrics["overall"]["fp"]
        fn = metrics["overall"]["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0  # 1.0 when no predictions
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)

    return {
        'thresholds': thresholds.tolist(),
        'precision': precisions,
        'recall': recalls,
    }


def compute_ap(precisions: List[float], recalls: List[float]) -> float:
    """
    Compute Average Precision using the 11-point interpolation method.

    This is the standard PASCAL VOC AP calculation.

    Args:
        precisions: List of precision values (from P-R curve)
        recalls: List of recall values (from P-R curve)

    Returns:
        Average Precision value
    """
    # Sort by recall (descending to get 0->1 order for typical P-R curve)
    sorted_pairs = sorted(zip(recalls, precisions), reverse=True)
    recalls_sorted = [r for r, _ in sorted_pairs]
    precisions_sorted = [p for _, p in sorted_pairs]

    # 11-point interpolation
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 11):
        # Find max precision at recall >= t
        p_at_r = 0.0
        for r, p in zip(recalls_sorted, precisions_sorted):
            if r >= t:
                p_at_r = max(p_at_r, p)
        ap += p_at_r / 11.0

    return ap


def compute_sample_level_metrics(
    all_predictions: List[List[List[float]]],
    all_ground_truths: List[List[List[float]]],
    tile_metadata: List[Dict],
) -> Dict:
    """
    Compute sample-level (FOV-level) metrics.

    Useful for understanding model performance at the sample level rather than
    individual detection level. Key metrics:
    - Detection rate: % of FOVs where model detected at least one object
    - Count correlation: Pearson correlation between GT count and pred count per FOV
    - Per-FOV statistics

    Args:
        all_predictions: Predictions per tile
        all_ground_truths: Ground truths per tile
        tile_metadata: Tile metadata with FOV paths

    Returns:
        Dict with sample-level metrics
    """
    # Group by FOV
    fov_stats = {}  # fov_path -> {'gt_count': int, 'pred_count': int}

    for preds, gts, tile_info in zip(all_predictions, all_ground_truths, tile_metadata):
        if tile_info is None:
            continue

        fov_path = str(tile_info.get('fov_path', 'unknown'))
        if fov_path not in fov_stats:
            fov_stats[fov_path] = {'gt_count': 0, 'pred_count': 0}

        fov_stats[fov_path]['gt_count'] += len(gts)
        fov_stats[fov_path]['pred_count'] += len(preds)

    if len(fov_stats) == 0:
        return {'error': 'No FOV metadata available'}

    # Compute metrics
    gt_counts = [s['gt_count'] for s in fov_stats.values()]
    pred_counts = [s['pred_count'] for s in fov_stats.values()]

    # Detection rate: % of positive FOVs where model predicted at least one
    positive_fovs = sum(1 for g in gt_counts if g > 0)
    detected_fovs = sum(1 for g, p in zip(gt_counts, pred_counts) if g > 0 and p > 0)
    detection_rate = detected_fovs / positive_fovs if positive_fovs > 0 else 0.0

    # False positive rate at FOV level: % of negative FOVs with predictions
    negative_fovs = sum(1 for g in gt_counts if g == 0)
    false_positive_fovs = sum(1 for g, p in zip(gt_counts, pred_counts) if g == 0 and p > 0)
    fov_fp_rate = false_positive_fovs / negative_fovs if negative_fovs > 0 else 0.0

    # Count correlation (Pearson)
    if len(gt_counts) > 1:
        gt_mean = np.mean(gt_counts)
        pred_mean = np.mean(pred_counts)
        gt_std = np.std(gt_counts)
        pred_std = np.std(pred_counts)

        if gt_std > 0 and pred_std > 0:
            covariance = np.mean([(g - gt_mean) * (p - pred_mean) for g, p in zip(gt_counts, pred_counts)])
            count_correlation = covariance / (gt_std * pred_std)
        else:
            count_correlation = 0.0
    else:
        count_correlation = 0.0

    # Mean absolute error in counts
    count_mae = np.mean([abs(g - p) for g, p in zip(gt_counts, pred_counts)])

    return {
        'num_fovs': len(fov_stats),
        'positive_fovs': positive_fovs,
        'negative_fovs': negative_fovs,
        'detection_rate': detection_rate,  # Sensitivity at FOV level
        'fov_false_positive_rate': fov_fp_rate,  # FP rate at FOV level
        'count_correlation': count_correlation,  # Pearson r for counts
        'count_mae': count_mae,  # Mean absolute error in counts
        'per_fov': {
            fov_path: stats
            for fov_path, stats in fov_stats.items()
        }
    }


def apply_nms(
    predictions: List[List[float]],
    iou_threshold: float = 0.5,
) -> List[List[float]]:
    """
    Apply Non-Maximum Suppression to remove duplicate predictions.

    Uses class-agnostic NMS: suppresses overlapping boxes regardless of class,
    keeping only the highest-scoring prediction per location. This ensures
    one detection per object (e.g., a parasite is either "positive" or "unsure",
    not both).

    Args:
        predictions: List of predictions, each [x1, y1, x2, y2, objectness, class_id]
        iou_threshold: IoU threshold for suppression (boxes with IoU >= threshold
                       are considered duplicates)

    Returns:
        Filtered list of predictions after NMS
    """
    if len(predictions) == 0:
        return predictions

    # Convert to tensors
    boxes = torch.tensor([p[:4] for p in predictions])
    scores = torch.tensor([p[4] for p in predictions])

    # Apply class-agnostic NMS (one detection per location)
    keep_indices = ops.nms(boxes, scores, iou_threshold)

    # Return filtered predictions
    return [predictions[i] for i in keep_indices.tolist()]


def extract_predictions_from_grid(
    outputs: torch.Tensor,
    Sx: int,
    Sy: int,
    conf_threshold: float = 0.5,
    apply_nms_filter: bool = True,
    nms_iou_threshold: float = 0.5,
) -> List[List[float]]:
    """
    Extract bounding box predictions from YOGO grid output.

    Args:
        outputs: Model output tensor (C, Sy, Sx) where C includes:
                 [x_offset, y_offset, w, h, objectness, class_probs...]
        Sx: Grid width
        Sy: Grid height
        conf_threshold: Minimum objectness score to include prediction
        apply_nms_filter: Whether to apply NMS to remove duplicates
        nms_iou_threshold: IoU threshold for NMS

    Returns:
        List of predictions, each [x1, y1, x2, y2, objectness, class_id]
    """
    pred_objectness = outputs[4, :, :]
    pred_mask = pred_objectness > conf_threshold

    predictions = []
    for i in range(Sy):
        for j in range(Sx):
            if pred_mask[i, j]:
                # Extract grid cell prediction
                x_off = outputs[0, i, j].item()
                y_off = outputs[1, i, j].item()
                w = outputs[2, i, j].item()
                h = outputs[3, i, j].item()
                obj = outputs[4, i, j].item()

                # Get class prediction (if multi-class)
                if outputs.shape[0] > 5:
                    class_probs = outputs[5:, i, j]
                    class_id = class_probs.argmax().item()
                else:
                    class_id = 0

                # Convert to normalized image coordinates
                cx = (j + x_off) / Sx
                cy = (i + y_off) / Sy

                # Convert to xyxy format
                x1 = cx - w/2
                y1 = cy - h/2
                x2 = cx + w/2
                y2 = cy + h/2

                predictions.append([x1, y1, x2, y2, obj, class_id])

    # Apply NMS to remove duplicate predictions from adjacent grid cells
    if apply_nms_filter and len(predictions) > 0:
        predictions = apply_nms(predictions, nms_iou_threshold)

    return predictions


def extract_ground_truth_from_grid(
    labels: torch.Tensor,
    Sx: int,
    Sy: int,
) -> List[List[float]]:
    """
    Extract ground truth boxes from YOGO label tensor.

    Args:
        labels: Label tensor (C, Sy, Sx) where C includes:
                [objectness, x1, y1, x2, y2, class_id]
        Sx: Grid width
        Sy: Grid height

    Returns:
        List of ground truth boxes, each [x1, y1, x2, y2, class_id]
    """
    gt_mask = labels[0, :, :] > 0.5

    ground_truths = []
    for i in range(Sy):
        for j in range(Sx):
            if gt_mask[i, j]:
                x1 = labels[1, i, j].item()
                y1 = labels[2, i, j].item()
                x2 = labels[3, i, j].item()
                y2 = labels[4, i, j].item()

                # Get class if available
                if labels.shape[0] > 5:
                    class_id = labels[5, i, j].item()
                else:
                    class_id = 0

                ground_truths.append([x1, y1, x2, y2, class_id])

    return ground_truths


def draw_boxes_on_image(
    image: np.ndarray,
    boxes: List[List[float]],
    colors: List[Tuple[int, int, int]],
    thickness: int = 2,
    label_prefix: str = "",
) -> np.ndarray:
    """
    Draw bounding boxes on an image.

    Args:
        image: RGB image as numpy array (H, W, 3)
        boxes: List of boxes, each [x1, y1, x2, y2, score, class_id] (normalized coords)
        colors: List of colors per class (RGB tuples)
        thickness: Line thickness
        label_prefix: Prefix for labels (e.g., "GT" or "Pred")

    Returns:
        Image with boxes drawn
    """
    import cv2

    img = image.copy()
    H, W = img.shape[:2]

    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        class_id = int(box[-1])

        # Convert normalized to pixel coordinates
        px1 = int(x1 * W)
        py1 = int(y1 * H)
        px2 = int(x2 * W)
        py2 = int(y2 * H)

        # Clamp to image bounds
        px1 = max(0, min(W - 1, px1))
        py1 = max(0, min(H - 1, py1))
        px2 = max(0, min(W - 1, px2))
        py2 = max(0, min(H - 1, py2))

        color = colors[class_id] if class_id < len(colors) else (255, 255, 255)
        cv2.rectangle(img, (px1, py1), (px2, py2), color, thickness)

    return img


def create_fov_visualization(
    fov_path: Path,
    gt_boxes: List[List[float]],
    pred_boxes: List[List[float]],
    class_names: List[str],
    output_path: Path,
    dpc_weight: float = 0.33,
) -> None:
    """
    Create side-by-side GT vs Predicted visualization for a FOV.

    Args:
        fov_path: Path to FOV directory containing dpc.png and fluorescent.png
        gt_boxes: Ground truth boxes in normalized coordinates
        pred_boxes: Predicted boxes in normalized coordinates
        class_names: List of class names
        output_path: Where to save the visualization
        dpc_weight: Weight for DPC channel in blend (default 0.33)
    """
    import cv2

    # Load images
    dpc_path = fov_path / "dpc.png"
    fluor_path = fov_path / "fluorescent.png"

    if not dpc_path.exists() or not fluor_path.exists():
        print(f"Warning: Could not find images in {fov_path}")
        return

    dpc = cv2.imread(str(dpc_path), cv2.IMREAD_GRAYSCALE)
    fluor = cv2.imread(str(fluor_path), cv2.IMREAD_COLOR)

    # Convert DPC to 3-channel
    dpc_rgb = cv2.cvtColor(dpc, cv2.COLOR_GRAY2BGR)

    # Blend: 33% DPC + 67% fluorescent
    blended = cv2.addWeighted(dpc_rgb, dpc_weight, fluor, 1.0 - dpc_weight, 0)

    # Draw GT boxes on left image
    gt_image = draw_boxes_on_image(
        blended.copy(), gt_boxes, CLASS_COLORS_GT, thickness=2
    )

    # Draw predicted boxes on right image
    pred_image = draw_boxes_on_image(
        blended.copy(), pred_boxes, CLASS_COLORS_PRED, thickness=2
    )

    # Add labels - scale based on image size
    H, W = blended.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Use a more reasonable font scale (0.8-1.2 for typical images)
    font_scale = max(0.6, min(1.2, min(W, H) / 2500.0))
    thickness = max(1, int(font_scale * 2))

    # Scale spacing based on image size
    margin = int(W * 0.01)  # 1% margin
    title_y = int(H * 0.02)  # 2% from top
    legend_y = int(H * 0.04)  # 4% from top
    legend_spacing = int(H * 0.025)  # 2.5% spacing between legend items
    box_size = int(H * 0.015)  # 1.5% legend box size

    # GT label
    cv2.putText(gt_image, f"Ground Truth ({len(gt_boxes)} objects)",
                (margin, title_y + int(font_scale * 20)), font, font_scale, (255, 255, 255), thickness)

    # Pred label
    cv2.putText(pred_image, f"Predictions ({len(pred_boxes)} objects)",
                (margin, title_y + int(font_scale * 20)), font, font_scale, (255, 255, 255), thickness)

    # Add legend
    for i, class_name in enumerate(class_names):
        y_pos = legend_y + i * legend_spacing + int(font_scale * 25)

        # GT legend
        cv2.rectangle(gt_image, (margin, y_pos), (margin + box_size, y_pos + box_size),
                      CLASS_COLORS_GT[i], -1)
        cv2.putText(gt_image, class_name, (margin + box_size + 5, y_pos + box_size - 2),
                    font, font_scale * 0.6, (255, 255, 255), max(1, thickness - 1))

        # Pred legend
        cv2.rectangle(pred_image, (margin, y_pos), (margin + box_size, y_pos + box_size),
                      CLASS_COLORS_PRED[i], -1)
        cv2.putText(pred_image, class_name, (margin + box_size + 5, y_pos + box_size - 2),
                    font, font_scale * 0.6, (255, 255, 255), max(1, thickness - 1))

    # Concatenate side by side
    combined = np.concatenate([gt_image, pred_image], axis=1)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), combined)


def aggregate_tile_to_fov(
    tile_boxes: List[List[float]],
    tile_offset: Tuple[int, int],
    tile_size: int,
    fov_size: Tuple[int, int],
) -> List[List[float]]:
    """
    Convert tile-normalized coordinates to FOV-normalized coordinates.

    Args:
        tile_boxes: Boxes in tile-normalized coordinates [x1, y1, x2, y2, ...]
        tile_offset: (y, x) offset of tile in FOV pixels
        tile_size: Tile size in pixels
        fov_size: (H, W) FOV size in pixels

    Returns:
        Boxes in FOV-normalized coordinates
    """
    y_off, x_off = tile_offset
    H, W = fov_size

    fov_boxes = []
    for box in tile_boxes:
        # Convert tile-normalized to tile-pixel
        x1_pix = box[0] * tile_size + x_off
        y1_pix = box[1] * tile_size + y_off
        x2_pix = box[2] * tile_size + x_off
        y2_pix = box[3] * tile_size + y_off

        # Convert to FOV-normalized
        x1_norm = x1_pix / W
        y1_norm = y1_pix / H
        x2_norm = x2_pix / W
        y2_norm = y2_pix / H

        # Keep other fields (score, class_id)
        fov_boxes.append([x1_norm, y1_norm, x2_norm, y2_norm] + box[4:])

    return fov_boxes


def compute_metrics_at_threshold(
    all_predictions: List[List[List[float]]],
    all_ground_truths: List[List[List[float]]],
    iou_threshold: float = 0.5,
    num_classes: int = 2,
) -> Dict:
    """
    Compute TP, FP, FN for a set of predictions and ground truths.

    Now includes per-class metrics and confusion matrix!

    Args:
        all_predictions: List of predictions per image, each prediction is
                        [x1, y1, x2, y2, objectness, class_id]
        all_ground_truths: List of ground truths per image, each is
                          [x1, y1, x2, y2, class_id]
        iou_threshold: IoU threshold for matching predictions to ground truth
        num_classes: Number of object classes

    Returns:
        Dictionary with overall and per-class metrics
    """
    # Overall metrics
    tp, fp, fn = 0, 0, 0

    # Per-class metrics
    per_class_tp = [0] * num_classes
    per_class_fp = [0] * num_classes
    per_class_fn = [0] * num_classes
    per_class_gt_count = [0] * num_classes

    # Confusion matrix: confusion[pred_class][gt_class]
    # Also track FP per predicted class
    confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    fp_per_pred_class = [0] * num_classes

    for pred_boxes, gt_boxes in zip(all_predictions, all_ground_truths):
        # Count ground truth per class
        for gt in gt_boxes:
            gt_class = int(gt[4])
            per_class_gt_count[gt_class] += 1

        # Handle empty cases
        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            continue

        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            # Convert to tensors (use only bbox coordinates for IoU)
            pred_tensor = torch.tensor([p[:4] for p in pred_boxes])
            gt_tensor = torch.tensor([g[:4] for g in gt_boxes])

            # Compute IoU matrix
            ious = ops.box_iou(pred_tensor, gt_tensor)

            # Match predictions to ground truth (greedy)
            matched_gt = set()
            for p_idx in range(len(pred_boxes)):
                best_iou, best_gt_idx = ious[p_idx].max(0)
                best_gt_idx = best_gt_idx.item()

                pred_class = int(pred_boxes[p_idx][5])

                if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
                    # Matched! Check if class matches
                    gt_class = int(gt_boxes[best_gt_idx][4])

                    # Update confusion matrix
                    confusion[pred_class][gt_class] += 1

                    # Count as overall TP regardless of class match
                    tp += 1

                    # For per-class metrics, require class match
                    if pred_class == gt_class:
                        per_class_tp[pred_class] += 1
                    else:
                        # Misclassified: FP for predicted class, FN for true class
                        per_class_fp[pred_class] += 1
                        per_class_fn[gt_class] += 1

                    matched_gt.add(best_gt_idx)
                else:
                    # False positive
                    fp += 1
                    per_class_fp[pred_class] += 1
                    fp_per_pred_class[pred_class] += 1

            # Unmatched ground truths are false negatives
            for gt_idx, gt in enumerate(gt_boxes):
                if gt_idx not in matched_gt:
                    fn += 1
                    gt_class = int(gt[4])
                    per_class_fn[gt_class] += 1

        elif len(pred_boxes) > 0:
            # All predictions are false positives
            fp += len(pred_boxes)
            for pred in pred_boxes:
                pred_class = int(pred[5])
                per_class_fp[pred_class] += 1
                fp_per_pred_class[pred_class] += 1
        else:
            # All ground truths are false negatives
            fn += len(gt_boxes)
            for gt in gt_boxes:
                gt_class = int(gt[4])
                per_class_fn[gt_class] += 1

    # Compute per-class metrics
    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []

    for c in range(num_classes):
        p = per_class_tp[c] / (per_class_tp[c] + per_class_fp[c]) if (per_class_tp[c] + per_class_fp[c]) > 0 else 0.0
        r = per_class_tp[c] / (per_class_tp[c] + per_class_fn[c]) if (per_class_tp[c] + per_class_fn[c]) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        per_class_precision.append(p)
        per_class_recall.append(r)
        per_class_f1.append(f)

    return {
        "overall": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
        },
        "per_class": {
            "tp": per_class_tp,
            "fp": per_class_fp,
            "fn": per_class_fn,
            "gt_count": per_class_gt_count,
            "precision": per_class_precision,
            "recall": per_class_recall,
            "f1": per_class_f1,
        },
        "confusion_matrix": confusion,
        "fp_by_predicted_class": fp_per_pred_class,
    }


def evaluate_model(
    checkpoint_path: Path,
    dataset_path: Path,
    split: str = "test",
    conf_thresholds: List[float] = [0.3, 0.5, 0.7, 0.9],
    iou_threshold: float = 0.5,
    nms_iou_threshold: float = 0.5,
    max_samples: Optional[int] = None,
    device: Optional[torch.device] = None,
    output_dir: Optional[Path] = None,
    save_visualizations: bool = False,
    visualization_conf_threshold: float = 0.5,
    load_predictions: bool = False,
    save_predictions: bool = True,
    compute_curves: bool = True,
) -> Dict:
    """
    Evaluate a YOGO model on a dataset split with comprehensive metrics.

    Args:
        checkpoint_path: Path to model checkpoint
        dataset_path: Path to dataset definition YAML
        split: Dataset split to evaluate ('train', 'val', or 'test')
        conf_thresholds: List of confidence thresholds to test
        iou_threshold: IoU threshold for matching predictions to GT
        nms_iou_threshold: IoU threshold for NMS (default: 0.5)
        max_samples: Maximum number of samples to evaluate (None for all)
        device: Device to use (auto-detected if None)
        output_dir: Directory for all outputs (predictions, metrics, curves, visualizations)
        save_visualizations: Whether to save GT vs Predicted images
        visualization_conf_threshold: Confidence threshold for visualization (default: 0.5)
        load_predictions: Load cached predictions instead of running inference
        save_predictions: Save predictions cache for future use
        compute_curves: Compute P-R curves and AP (can be slow)

    Returns:
        Dictionary with comprehensive metrics at each confidence threshold
    """
    # Setup output directories
    if output_dir is None:
        # Default: results/{experiment_id}/metrics/{split}/
        model_dir = checkpoint_path.parent.parent
        output_dir = model_dir / "metrics" / split
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_dir = output_dir / "predictions"
    curves_dir = output_dir / "curves"
    vis_dir = output_dir / "visualizations" / f"conf{visualization_conf_threshold}"

    # Load dataset definition first (needed even when loading cached predictions)
    print(f"Loading dataset from {dataset_path}")
    dataset_def = DatasetDefinition.from_yaml(dataset_path)

    # Check for cached predictions
    if load_predictions and predictions_dir.exists():
        print("Loading cached predictions...")
        all_outputs, all_labels, tile_metadata, model_config = load_predictions_cache(predictions_dir)
        Sx = model_config['Sx']
        Sy = model_config['Sy']
        image_hw = tuple(model_config['image_hw'])
        input_channels = model_config['input_channels']
        print(f"Model config from cache: Grid {Sx}×{Sy}, Image {image_hw}, Channels {input_channels}")
    else:
        if device is None:
            device = choose_device()

        print(f"Loading model from {checkpoint_path}")
        model, config = YOGO.from_pth(checkpoint_path, inference=True)
        model = model.to(device).eval()

        # Get grid dimensions
        Sx = int(model.Sx) if hasattr(model.Sx, 'item') else int(model.Sx)
        Sy = int(model.Sy) if hasattr(model.Sy, 'item') else int(model.Sy)

        # Get image size and channels
        img_size = model.img_size
        image_hw = (
            int(img_size[0]) if hasattr(img_size[0], 'item') else int(img_size[0]),
            int(img_size[1]) if hasattr(img_size[1], 'item') else int(img_size[1])
        )
        input_channels = int(model.input_channels) if hasattr(model, 'input_channels') else 3

        print(f"Model: Grid {Sx}×{Sy}, Image {image_hw}, Channels {input_channels}")

    print(f"NMS IoU threshold: {nms_iou_threshold}")
    print(f"Output directory: {output_dir}")

    tile_size = 1024
    tile_overlap = 256

    # Load or run inference
    if load_predictions and predictions_dir.exists():
        # Already loaded above, just need tile_info alias
        all_tile_info = tile_metadata
    else:
        # Load dataset and run inference
        datasets = get_datasets(
            dataset_definition=dataset_def,
            Sx=Sx,
            Sy=Sy,
            image_hw=image_hw,
            rgb=(input_channels == 3),
            normalize_images=True,
            input_channels=input_channels if input_channels not in [1, 3] else None,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )

        if split not in datasets:
            raise ValueError(f"Split '{split}' not found in dataset. Available: {list(datasets.keys())}")

        dataset = datasets[split]
        print(f"{split.capitalize()} dataset: {len(dataset)} samples")

        # Try to get tile metadata for FOV visualization
        tile_metadata = []
        try:
            from torch.utils.data import ConcatDataset
            if isinstance(dataset, ConcatDataset):
                tile_idx = 0
                for sub_ds in dataset.datasets:
                    if hasattr(sub_ds, 'tiles'):
                        for tile_info in sub_ds.tiles:
                            tile_metadata.append({
                                'global_idx': tile_idx,
                                'fov_path': tile_info['fov_path'],
                                'offset': tile_info['offset'],
                                'image_size': tile_info['image_size'],
                            })
                            tile_idx += 1
        except Exception as e:
            print(f"Note: Could not extract tile metadata for FOV visualization: {e}")

        # Run inference and collect predictions
        print("Running inference...")
        all_outputs = []
        all_labels = []
        all_tile_info = []

        num_samples = min(len(dataset), max_samples) if max_samples else len(dataset)

        with torch.no_grad():
            for idx in tqdm(range(num_samples), desc="Inference"):
                image, label = dataset[idx]
                image = image.unsqueeze(0).to(device)

                outputs = model(image)

                all_outputs.append(outputs[0].cpu())
                all_labels.append(label)

                # Track tile info if available
                if idx < len(tile_metadata):
                    all_tile_info.append(tile_metadata[idx])
                else:
                    all_tile_info.append(None)

        print(f"Processed {len(all_outputs)} samples")

        # Save predictions cache for future use
        if save_predictions:
            model_config = {
                'Sx': Sx,
                'Sy': Sy,
                'image_hw': list(image_hw),
                'input_channels': input_channels,
                'checkpoint': str(checkpoint_path),
            }
            save_predictions_cache(predictions_dir, all_outputs, all_labels, all_tile_info, model_config)

    # Generate FOV visualizations if requested
    # Do this BEFORE computing metrics so visualizations match exactly what goes into metrics
    if save_visualizations and len(all_tile_info) > 0 and any(t is not None for t in all_tile_info):
        print(f"\nGenerating FOV visualizations at conf={visualization_conf_threshold}...")
        vis_dir.mkdir(parents=True, exist_ok=True)

        # Group tiles by FOV
        fov_data = {}  # fov_path -> {'gt': [], 'pred': [], 'image_size': ()}
        for idx, (outputs, labels, tile_info) in enumerate(zip(all_outputs, all_labels, all_tile_info)):
            if tile_info is None:
                continue

            fov_path = tile_info['fov_path']
            if fov_path not in fov_data:
                fov_data[fov_path] = {
                    'gt': [],
                    'pred': [],
                    'image_size': tile_info['image_size'],
                }

            # Extract predictions and GT for this tile (using the SAME extraction that will be used for metrics)
            preds = extract_predictions_from_grid(
                outputs, Sx, Sy, visualization_conf_threshold,
                apply_nms_filter=True, nms_iou_threshold=nms_iou_threshold
            )
            gts = extract_ground_truth_from_grid(labels, Sx, Sy)

            # Convert to FOV coordinates
            fov_preds = aggregate_tile_to_fov(
                preds, tile_info['offset'], tile_size, tile_info['image_size']
            )
            fov_gts = aggregate_tile_to_fov(
                gts, tile_info['offset'], tile_size, tile_info['image_size']
            )

            fov_data[fov_path]['gt'].extend(fov_gts)
            fov_data[fov_path]['pred'].extend(fov_preds)

        # Apply NMS at FOV level to remove duplicates from overlapping tiles
        for fov_path, data in fov_data.items():
            if len(data['pred']) > 0:
                data['pred'] = apply_nms(data['pred'], nms_iou_threshold)

        # Generate visualizations
        for fov_path, data in tqdm(fov_data.items(), desc="Saving FOV images"):
            fov_name = fov_path.name
            output_path = vis_dir / f"{fov_name}_gt_vs_pred.png"
            create_fov_visualization(
                fov_path=fov_path,
                gt_boxes=data['gt'],
                pred_boxes=data['pred'],
                class_names=dataset_def.classes,
                output_path=output_path,
            )

        print(f"Saved {len(fov_data)} FOV visualizations to {vis_dir}")

    # Evaluate at different confidence thresholds
    print("\nComputing metrics at different confidence thresholds...")
    print("="*80)

    results = {
        "metadata": {
            "checkpoint": str(checkpoint_path),
            "dataset": str(dataset_path),
            "split": split,
            "num_samples": len(all_outputs),
            "iou_threshold": iou_threshold,
            "grid_size": f"{Sx}×{Sy}",
            "image_size": f"{image_hw[0]}×{image_hw[1]}",
            "input_channels": input_channels,
            "class_names": dataset_def.classes,
        },
        "thresholds": {}
    }

    for conf_thresh in conf_thresholds:
        # Extract predictions at this confidence threshold (with NMS)
        all_predictions = []
        all_ground_truths = []

        for outputs, labels in zip(all_outputs, all_labels):
            preds = extract_predictions_from_grid(
                outputs, Sx, Sy, conf_thresh,
                apply_nms_filter=True, nms_iou_threshold=nms_iou_threshold
            )
            gts = extract_ground_truth_from_grid(labels, Sx, Sy)

            all_predictions.append(preds)
            all_ground_truths.append(gts)

        # Compute metrics (now returns dict with per-class info)
        metrics = compute_metrics_at_threshold(
            all_predictions,
            all_ground_truths,
            iou_threshold,
            num_classes=len(dataset_def.classes)
        )

        # Overall metrics
        tp = metrics["overall"]["tp"]
        fp = metrics["overall"]["fp"]
        fn = metrics["overall"]["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Store detailed results
        results["thresholds"][conf_thresh] = {
            "overall": {
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "false_positive_rate": fp / (fp + tp) if (fp + tp) > 0 else 0.0,
                "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0.0,
            },
            "per_class": metrics["per_class"],
            "confusion_matrix": metrics["confusion_matrix"],
            "fp_by_predicted_class": metrics["fp_by_predicted_class"],
        }

        # Print overall metrics
        fp_rate = fp / (fp + tp) if (fp + tp) > 0 else 0.0
        fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        print(f"\nConf={conf_thresh:.1f} OVERALL:")
        print(f"  TP={tp:4d}, FP={fp:4d}, FN={fn:4d} | "
              f"P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
        print(f"  FP rate: {fp_rate:.3f}, FN rate: {fn_rate:.3f}")

        # Print per-class metrics
        print(f"\n  Per-class breakdown:")
        for c, class_name in enumerate(dataset_def.classes):
            c_tp = metrics["per_class"]["tp"][c]
            c_fp = metrics["per_class"]["fp"][c]
            c_fn = metrics["per_class"]["fn"][c]
            c_p = metrics["per_class"]["precision"][c]
            c_r = metrics["per_class"]["recall"][c]
            c_f1 = metrics["per_class"]["f1"][c]
            c_gt = metrics["per_class"]["gt_count"][c]
            c_fp_pred = metrics["fp_by_predicted_class"][c]

            print(f"    {class_name:10s}: TP={c_tp:4d}, FP={c_fp:4d}, FN={c_fn:4d}, GT={c_gt:4d} | "
                  f"P={c_p:.3f}, R={c_r:.3f}, F1={c_f1:.3f}")
            print(f"                  (FP predicted as '{class_name}': {c_fp_pred})")

        # Print confusion matrix
        if len(dataset_def.classes) > 1:
            print(f"\n  Confusion matrix (rows=predicted, cols=ground_truth):")
            print(f"              ", end="")
            for c, class_name in enumerate(dataset_def.classes):
                print(f"{class_name:>10s}", end=" ")
            print()
            for pred_c, pred_class_name in enumerate(dataset_def.classes):
                print(f"    {pred_class_name:>10s}:", end=" ")
                for gt_c in range(len(dataset_def.classes)):
                    print(f"{metrics['confusion_matrix'][pred_c][gt_c]:10d}", end=" ")
                print()

    print("="*80)

    # Compute P-R curve and AP if requested
    if compute_curves:
        num_pr_thresholds = 101  # Fine granularity for smooth P-R curve
        print(f"\nComputing Precision-Recall curve ({num_pr_thresholds} thresholds)...")
        pr_curve = compute_pr_curve(
            all_outputs, all_labels, Sx, Sy,
            iou_threshold, nms_iou_threshold,
            num_thresholds=num_pr_thresholds
        )
        results["pr_curve"] = pr_curve

        # Compute AP (Average Precision)
        ap = compute_ap(pr_curve['precision'], pr_curve['recall'])
        results["average_precision"] = ap
        print(f"Average Precision (AP): {ap:.4f}")

        # Save P-R curve data for plotting
        curves_dir.mkdir(parents=True, exist_ok=True)
        with open(curves_dir / "pr_curve.json", 'w') as f:
            json.dump(pr_curve, f, indent=2)
        print(f"Saved P-R curve to: {curves_dir / 'pr_curve.json'}")

    # Compute MCC for best threshold
    # For MCC, we need TN which requires treating background as negatives
    # Approximation: use the total number of grid cells minus TP/FP/FN
    best_thresh_result = max(results["thresholds"].items(),
                             key=lambda x: x[1]["overall"]["f1_score"])
    best_conf = best_thresh_result[0]
    best_metrics = best_thresh_result[1]

    tp = best_metrics["overall"]["true_positives"]
    fp = best_metrics["overall"]["false_positives"]
    fn = best_metrics["overall"]["false_negatives"]
    # Estimate TN: For object detection, this is tricky. We'll use total GT negatives
    # (grid cells without objects that weren't predicted)
    total_cells = len(all_outputs) * Sx * Sy
    total_gt_positives = sum(sum(1 for g in extract_ground_truth_from_grid(l, Sx, Sy)) for l in all_labels)
    total_gt_negatives = total_cells - total_gt_positives
    total_pred_positives = tp + fp
    # TN = negatives that weren't predicted as positive
    tn = total_gt_negatives - fp  # This is an approximation

    mcc = compute_mcc(tp, fp, max(0, tn), fn)
    results["mcc"] = mcc
    print(f"Matthews Correlation Coefficient (MCC) at conf={best_conf}: {mcc:.4f}")

    # Compute sample-level metrics at best threshold
    print("\nComputing sample-level (FOV) metrics...")
    all_predictions_best = []
    all_ground_truths_best = []
    for outputs, labels in zip(all_outputs, all_labels):
        preds = extract_predictions_from_grid(
            outputs, Sx, Sy, best_conf,
            apply_nms_filter=True, nms_iou_threshold=nms_iou_threshold
        )
        gts = extract_ground_truth_from_grid(labels, Sx, Sy)
        all_predictions_best.append(preds)
        all_ground_truths_best.append(gts)

    sample_metrics = compute_sample_level_metrics(
        all_predictions_best, all_ground_truths_best, all_tile_info
    )
    results["sample_level"] = sample_metrics

    if 'error' not in sample_metrics:
        print(f"  FOVs: {sample_metrics['num_fovs']} ({sample_metrics['positive_fovs']} positive, {sample_metrics['negative_fovs']} negative)")
        print(f"  Detection rate (sensitivity at FOV level): {sample_metrics['detection_rate']:.3f}")
        print(f"  FOV false positive rate: {sample_metrics['fov_false_positive_rate']:.3f}")
        print(f"  Count correlation (Pearson r): {sample_metrics['count_correlation']:.3f}")
        print(f"  Count MAE: {sample_metrics['count_mae']:.1f}")

    # Find best F1 threshold (based on overall F1)
    best_thresh = max(results["thresholds"].items(),
                      key=lambda x: x[1]["overall"]["f1_score"])
    print(f"\n{'='*80}")
    print(f"SUMMARY at best threshold (conf={best_thresh[0]:.1f}):")
    print(f"  F1: {best_thresh[1]['overall']['f1_score']:.4f}")
    print(f"  Precision: {best_thresh[1]['overall']['precision']:.4f}")
    print(f"  Recall: {best_thresh[1]['overall']['recall']:.4f}")
    if compute_curves:
        print(f"  AP: {ap:.4f}")
    print(f"  MCC: {mcc:.4f}")

    # Also show best F1 per class at this threshold
    print(f"\nPer-class F1 at best threshold:")
    for c, class_name in enumerate(dataset_def.classes):
        c_f1 = best_thresh[1]["per_class"]["f1"][c]
        c_r = best_thresh[1]["per_class"]["recall"][c]
        print(f"  {class_name}: F1={c_f1:.3f}, Recall={c_r:.3f}")

    results["best_threshold"] = {
        "confidence": best_thresh[0],
        "metrics": best_thresh[1]
    }

    # Save full results
    results_path = output_dir / "metrics.json"
    with open(results_path, 'w') as f:
        # Convert any non-serializable items
        def make_serializable(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, Path):
                return str(obj)
            return obj

        # Deep copy with serialization
        import copy
        results_copy = json.loads(json.dumps(results, default=make_serializable))
        json.dump(results_copy, f, indent=2)
    print(f"\nSaved full metrics to: {results_path}")

    return results


def save_evaluation_metadata(
    output_dir: Path,
    checkpoint_path: Path,
    dataset_path: Path,
    split: str,
    conf_thresholds: List[float],
    iou_threshold: float,
    nms_iou_threshold: float,
    vis_conf: float,
    save_visualizations: bool,
    compute_curves: bool,
) -> None:
    """Save metadata JSON file for reproducibility."""
    import datetime

    cmd_parts = [
        f"python -m yogo.compute_metrics {checkpoint_path} {dataset_path}",
        f"--split {split}",
        f"--conf-thresholds {' '.join(map(str, conf_thresholds))}",
        f"--iou-threshold {iou_threshold}",
        f"--nms-iou-threshold {nms_iou_threshold}",
    ]
    if save_visualizations:
        cmd_parts.append(f"--save-images --vis-conf {vis_conf}")
    if compute_curves:
        cmd_parts.append("--compute-curves")

    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "checkpoint": str(checkpoint_path.absolute()),
        "dataset": str(dataset_path.absolute()),
        "split": split,
        "parameters": {
            "conf_thresholds": conf_thresholds,
            "iou_threshold": iou_threshold,
            "nms_iou_threshold": nms_iou_threshold,
            "visualization_conf_threshold": vis_conf,
            "save_visualizations": save_visualizations,
            "compute_curves": compute_curves,
        },
        "command": " ".join(cmd_parts),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "evaluation_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved evaluation metadata to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute comprehensive object detection metrics for YOGO models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation on test set
  python -m yogo.compute_metrics results/fold1/checkpoints/best.pth dataset/malaria_fold1.yaml

  # Full evaluation with visualizations and P-R curves
  python -m yogo.compute_metrics model.pth dataset.yaml --save-images --compute-curves

  # Reuse cached predictions (skip inference)
  python -m yogo.compute_metrics model.pth dataset.yaml --load-predictions

  # Custom confidence thresholds
  python -m yogo.compute_metrics model.pth dataset.yaml --conf-thresholds 0.3 0.5 0.7

  # Evaluate on validation set
  python -m yogo.compute_metrics model.pth dataset.yaml --split val

Output structure:
  results/{experiment_id}/metrics/{split}/
    ├── predictions/         # Cached model outputs
    │   ├── predictions.pt   # Tensor data
    │   └── metadata.json    # Model config
    ├── curves/              # P-R curves for plotting
    │   └── pr_curve.json
    ├── visualizations/      # GT vs Predicted images
    │   └── conf0.5/
    ├── metrics.json         # Full results
    └── evaluation_metadata.json
        """
    )

    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to model checkpoint (.pth file)"
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Path to dataset definition YAML file"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate (default: test)"
    )
    parser.add_argument(
        "--conf-thresholds",
        type=float,
        nargs="+",
        default=[0.3, 0.5, 0.7, 0.9],
        help="Confidence thresholds to evaluate (default: 0.3 0.5 0.7 0.9)"
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for matching predictions to GT (default: 0.5)"
    )
    parser.add_argument(
        "--nms-iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for NMS to remove duplicate predictions (default: 0.5)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, auto-detected if not specified)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results/{experiment_id}/metrics/{split}/)"
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save GT vs Predicted visualization images per FOV"
    )
    parser.add_argument(
        "--vis-conf",
        type=float,
        default=0.5,
        help="Confidence threshold for visualization (default: 0.5)"
    )
    parser.add_argument(
        "--load-predictions",
        action="store_true",
        help="Load cached predictions instead of running inference"
    )
    parser.add_argument(
        "--no-save-predictions",
        action="store_true",
        help="Don't save predictions cache"
    )
    parser.add_argument(
        "--compute-curves",
        action="store_true",
        help="Compute P-R curves and Average Precision (slower but more comprehensive)"
    )
    # Legacy argument for backwards compatibility
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=argparse.SUPPRESS  # Hidden, for backwards compatibility
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    dataset_path = Path(args.dataset)

    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return 1

    if not dataset_path.exists():
        print(f"ERROR: Dataset definition not found: {dataset_path}")
        return 1

    # Determine output directory
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
    # Default will be set in evaluate_model based on checkpoint path

    # Set device
    device = None
    if args.device:
        device = torch.device(args.device)

    # Run evaluation
    try:
        results = evaluate_model(
            checkpoint_path=checkpoint_path,
            dataset_path=dataset_path,
            split=args.split,
            conf_thresholds=args.conf_thresholds,
            iou_threshold=args.iou_threshold,
            nms_iou_threshold=args.nms_iou_threshold,
            max_samples=args.max_samples,
            device=device,
            output_dir=output_dir,
            save_visualizations=args.save_images,
            visualization_conf_threshold=args.vis_conf,
            load_predictions=args.load_predictions,
            save_predictions=not args.no_save_predictions,
            compute_curves=args.compute_curves,
        )

        # Get actual output dir from results (in case default was used)
        actual_output_dir = output_dir
        if actual_output_dir is None:
            model_dir = checkpoint_path.parent.parent
            actual_output_dir = model_dir / "metrics" / args.split

        # Save evaluation metadata
        save_evaluation_metadata(
            output_dir=actual_output_dir,
            checkpoint_path=checkpoint_path,
            dataset_path=dataset_path,
            split=args.split,
            conf_thresholds=args.conf_thresholds,
            iou_threshold=args.iou_threshold,
            nms_iou_threshold=args.nms_iou_threshold,
            vis_conf=args.vis_conf,
            save_visualizations=args.save_images,
            compute_curves=args.compute_curves,
        )

        # Legacy output argument support
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                def make_serializable(obj):
                    if isinstance(obj, (np.floating, np.integer)):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, Path):
                        return str(obj)
                    return obj
                json.dump(results, f, indent=2, default=make_serializable)
            print(f"\nLegacy output saved to: {output_path}")

        return 0

    except Exception as e:
        print(f"\nERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
