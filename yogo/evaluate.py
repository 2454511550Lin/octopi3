"""
Standard evaluation script for YOGO models.

Computes standard object detection metrics on test data, handling corner cases
like empty predictions or negative-only test sets.
"""

import torch
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from tqdm import tqdm
import numpy as np

from yogo.model import YOGO
from yogo.data.yogo_dataloader import get_dataloader
from yogo.data.dataset_definition_file import DatasetDefinition
from yogo.utils import choose_device


def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """
    Compute IoU between two boxes in xyxy format.

    Args:
        box1: (4,) tensor [x1, y1, x2, y2]
        box2: (4,) tensor [x1, y1, x2, y2]

    Returns:
        IoU score
    """
    # Intersection area
    x1 = max(box1[0].item(), box2[0].item())
    y1 = max(box1[1].item(), box2[1].item())
    x2 = min(box1[2].item(), box2[2].item())
    y2 = min(box1[3].item(), box2[3].item())

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)

    # Union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def evaluate_detections(
    predictions: List[torch.Tensor],
    ground_truths: List[torch.Tensor],
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.5,
    num_classes: int = 2
) -> Dict[str, Any]:
    """
    Evaluate object detection predictions against ground truth.

    Handles corner cases:
    - Empty predictions
    - Empty ground truth (negative samples)
    - Mixed scenarios

    Args:
        predictions: List of prediction tensors per image (N, 5+num_classes)
                    [x1, y1, x2, y2, objectness, class_probs...]
        ground_truths: List of ground truth tensors per image (M, 5)
                      [class_id, x1, y1, x2, y2]
        iou_threshold: IoU threshold for matching predictions to ground truth
        conf_threshold: Confidence threshold for considering predictions
        num_classes: Number of object classes

    Returns:
        Dictionary with evaluation metrics
    """
    total_gt = 0
    total_pred = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Per-class metrics
    per_class_tp = [0] * num_classes
    per_class_fp = [0] * num_classes
    per_class_fn = [0] * num_classes
    per_class_gt = [0] * num_classes

    for pred_boxes, gt_boxes in zip(predictions, ground_truths):
        # Filter predictions by confidence
        if len(pred_boxes) > 0:
            conf_mask = pred_boxes[:, 4] >= conf_threshold
            pred_boxes = pred_boxes[conf_mask]

        total_pred += len(pred_boxes)
        total_gt += len(gt_boxes)

        # Handle empty cases
        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            # True negative - no objects, no predictions
            continue

        if len(gt_boxes) == 0 and len(pred_boxes) > 0:
            # False positives - predictions on negative sample
            false_positives += len(pred_boxes)
            for pred in pred_boxes:
                pred_class = pred[5:].argmax().item()
                per_class_fp[pred_class] += 1
            continue

        if len(gt_boxes) > 0 and len(pred_boxes) == 0:
            # False negatives - missed all ground truth
            false_negatives += len(gt_boxes)
            for gt in gt_boxes:
                gt_class = int(gt[0].item())
                per_class_fn[gt_class] += 1
                per_class_gt[gt_class] += 1
            continue

        # Match predictions to ground truth
        matched_gt = set()

        for pred in pred_boxes:
            pred_box = pred[:4]  # x1, y1, x2, y2
            pred_class = pred[5:].argmax().item()

            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue

                gt_class = int(gt[0].item())
                gt_box = gt[1:5]  # x1, y1, x2, y2

                # Only match same class
                if pred_class != gt_class:
                    continue

                iou = compute_iou(pred_box, gt_box)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            # Check if match is good enough
            if best_iou >= iou_threshold:
                true_positives += 1
                matched_gt.add(best_gt_idx)
                per_class_tp[pred_class] += 1
            else:
                false_positives += 1
                per_class_fp[pred_class] += 1

        # Count unmatched ground truth as false negatives
        for gt_idx, gt in enumerate(gt_boxes):
            gt_class = int(gt[0].item())
            per_class_gt[gt_class] += 1

            if gt_idx not in matched_gt:
                false_negatives += 1
                per_class_fn[gt_class] += 1

    # Compute overall metrics
    precision = true_positives / total_pred if total_pred > 0 else 0.0
    recall = true_positives / total_gt if total_gt > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # False positive rate (for negative samples)
    # FPR = FP / (FP + TN), but we don't explicitly count true negatives
    # For negative-only test sets, FPR = total_predictions / total_images would be meaningful

    # Compute per-class metrics
    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []

    for i in range(num_classes):
        tp = per_class_tp[i]
        fp = per_class_fp[i]
        fn = per_class_fn[i]
        gt = per_class_gt[i]

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / gt if gt > 0 else 0.0
        f1_score = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class_precision.append(prec)
        per_class_recall.append(rec)
        per_class_f1.append(f1_score)

    return {
        "overall": {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "total_predictions": total_pred,
            "total_ground_truth": total_gt,
        },
        "per_class": {
            "precision": per_class_precision,
            "recall": per_class_recall,
            "f1_score": per_class_f1,
            "true_positives": per_class_tp,
            "false_positives": per_class_fp,
            "false_negatives": per_class_fn,
            "ground_truth_count": per_class_gt,
        }
    }


def run_inference(
    model: YOGO,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Run inference on a dataset.

    Returns:
        predictions: List of prediction tensors per image
        ground_truths: List of ground truth tensors per image
    """
    model.eval()
    predictions = []
    ground_truths = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Running inference"):
            images = images.to(device)

            # Forward pass
            outputs = model(images)

            # Process each image in batch
            for i in range(images.shape[0]):
                pred = outputs[i]  # (C, H, W)
                gt = labels[i]  # (5+num_classes, H, W)

                # Extract predictions from output grid
                # This is simplified - real YOGO post-processing is more complex
                # You may need to adapt this based on actual YOGO output format

                predictions.append(pred.cpu())
                ground_truths.append(gt.cpu())

    return predictions, ground_truths


def evaluate_model(
    checkpoint_path: Path,
    dataset_path: Path,
    device: Optional[torch.device] = None,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    batch_size: int = 8,
) -> Dict[str, Any]:
    """
    Evaluate a trained YOGO model on test data.

    Args:
        checkpoint_path: Path to model checkpoint (.pth file)
        dataset_path: Path to dataset definition YAML
        device: Device to run on (auto-detected if None)
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for matching
        batch_size: Batch size for inference

    Returns:
        Dictionary with evaluation metrics
    """
    if device is None:
        device = choose_device()

    print(f"Loading model from {checkpoint_path}")
    model, config = YOGO.from_pth(checkpoint_path, inference=True)
    model = model.to(device)
    model.eval()

    print(f"Loading dataset from {dataset_path}")
    dataset_def = DatasetDefinition.from_yaml(dataset_path)

    # Get test dataloader
    _, _, test_dataloader = get_dataloader(
        dataset_def=dataset_def,
        batch_size=batch_size,
        rank=0,
        world_size=1,
        normalize_images=True,  # Should match training
    )

    if test_dataloader is None or len(test_dataloader) == 0:
        print("WARNING: Test dataloader is empty!")
        return {
            "error": "No test data available",
            "overall": {
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
            }
        }

    print(f"Test dataset size: {len(test_dataloader.dataset)} samples")

    # Run inference
    print("Running inference...")
    predictions, ground_truths = run_inference(model, test_dataloader, device)

    # Compute metrics
    print("Computing metrics...")
    metrics = evaluate_detections(
        predictions=predictions,
        ground_truths=ground_truths,
        iou_threshold=iou_threshold,
        conf_threshold=conf_threshold,
        num_classes=len(dataset_def.class_names),
    )

    # Add metadata
    metrics["metadata"] = {
        "checkpoint": str(checkpoint_path),
        "dataset": str(dataset_path),
        "num_test_samples": len(predictions),
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "class_names": dataset_def.class_names,
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate YOGO model on test data"
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to model checkpoint (.pth)"
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Path to dataset definition YAML"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for metrics (default: same dir as checkpoint)"
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for detections"
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for matching"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, auto-detected if not specified)"
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    dataset_path = Path(args.dataset)

    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return

    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}")
        return

    # Set device
    device = None
    if args.device:
        device = torch.device(args.device)

    # Run evaluation
    metrics = evaluate_model(
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
        device=device,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        batch_size=args.batch_size,
    )

    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(json.dumps(metrics, indent=2))

    # Save to file
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = checkpoint_path.parent.parent / "metrics" / "test_metrics.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to: {output_path}")


if __name__ == "__main__":
    main()
