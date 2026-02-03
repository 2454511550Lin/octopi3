"""
Compute object detection metrics for YOGO models.

This module evaluates trained YOGO models on test datasets, computing precision,
recall, and F1 scores at various confidence thresholds. It properly handles the
YOGO grid-based output format and supports both standard YOLO datasets and
malaria detection datasets.

Usage:
    python -m yogo.compute_metrics checkpoint.pth dataset.yaml
    python -m yogo.compute_metrics checkpoint.pth dataset.yaml --conf-thresholds 0.3 0.5 0.7 0.9
    python -m yogo.compute_metrics checkpoint.pth dataset.yaml --split test --output results.json
"""

import torch
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import torchvision.ops as ops

from yogo.model import YOGO
from yogo.data.yogo_dataloader import get_datasets
from yogo.data.dataset_definition_file import DatasetDefinition
from yogo.utils import choose_device


def extract_predictions_from_grid(
    outputs: torch.Tensor,
    Sx: int,
    Sy: int,
    conf_threshold: float = 0.5,
) -> List[List[float]]:
    """
    Extract bounding box predictions from YOGO grid output.

    Args:
        outputs: Model output tensor (C, Sy, Sx) where C includes:
                 [x_offset, y_offset, w, h, objectness, class_probs...]
        Sx: Grid width
        Sy: Grid height
        conf_threshold: Minimum objectness score to include prediction

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


def compute_metrics_at_threshold(
    all_predictions: List[List[List[float]]],
    all_ground_truths: List[List[List[float]]],
    iou_threshold: float = 0.5,
) -> Tuple[int, int, int]:
    """
    Compute TP, FP, FN for a set of predictions and ground truths.

    Args:
        all_predictions: List of predictions per image, each prediction is
                        [x1, y1, x2, y2, objectness, class_id]
        all_ground_truths: List of ground truths per image, each is
                          [x1, y1, x2, y2, class_id]
        iou_threshold: IoU threshold for matching predictions to ground truth

    Returns:
        (true_positives, false_positives, false_negatives)
    """
    tp, fp, fn = 0, 0, 0

    for pred_boxes, gt_boxes in zip(all_predictions, all_ground_truths):
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

                if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1

            fn += len(gt_boxes) - len(matched_gt)

        elif len(pred_boxes) > 0:
            # All predictions are false positives
            fp += len(pred_boxes)
        else:
            # All ground truths are false negatives
            fn += len(gt_boxes)

    return tp, fp, fn


def evaluate_model(
    checkpoint_path: Path,
    dataset_path: Path,
    split: str = "test",
    conf_thresholds: List[float] = [0.3, 0.5, 0.7, 0.9],
    iou_threshold: float = 0.5,
    max_samples: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Evaluate a YOGO model on a dataset split.

    Args:
        checkpoint_path: Path to model checkpoint
        dataset_path: Path to dataset definition YAML
        split: Dataset split to evaluate ('train', 'val', or 'test')
        conf_thresholds: List of confidence thresholds to test
        iou_threshold: IoU threshold for matching predictions to GT
        max_samples: Maximum number of samples to evaluate (None for all)
        device: Device to use (auto-detected if None)

    Returns:
        Dictionary with metrics at each confidence threshold
    """
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

    # Load dataset
    print(f"Loading dataset from {dataset_path}")
    dataset_def = DatasetDefinition.from_yaml(dataset_path)

    datasets = get_datasets(
        dataset_definition=dataset_def,
        Sx=Sx,
        Sy=Sy,
        image_hw=image_hw,
        rgb=(input_channels == 3),
        normalize_images=True,
        input_channels=input_channels if input_channels not in [1, 3] else None,
        tile_size=1024,
        tile_overlap=256,
    )

    if split not in datasets:
        raise ValueError(f"Split '{split}' not found in dataset. Available: {list(datasets.keys())}")

    dataset = datasets[split]
    print(f"{split.capitalize()} dataset: {len(dataset)} samples")

    # Run inference and collect predictions
    print("Running inference...")
    all_outputs = []
    all_labels = []

    num_samples = min(len(dataset), max_samples) if max_samples else len(dataset)

    with torch.no_grad():
        for idx in tqdm(range(num_samples), desc="Inference"):
            image, label = dataset[idx]
            image = image.unsqueeze(0).to(device)

            outputs = model(image)

            all_outputs.append(outputs[0].cpu())
            all_labels.append(label)

    print(f"Processed {len(all_outputs)} samples")

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
        # Extract predictions at this confidence threshold
        all_predictions = []
        all_ground_truths = []

        for outputs, labels in zip(all_outputs, all_labels):
            preds = extract_predictions_from_grid(outputs, Sx, Sy, conf_thresh)
            gts = extract_ground_truth_from_grid(labels, Sx, Sy)

            all_predictions.append(preds)
            all_ground_truths.append(gts)

        # Compute metrics
        tp, fp, fn = compute_metrics_at_threshold(
            all_predictions,
            all_ground_truths,
            iou_threshold
        )

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        results["thresholds"][conf_thresh] = {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

        print(f"Conf={conf_thresh:.1f}: TP={tp:4d}, FP={fp:4d}, FN={fn:4d} | "
              f"P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

    print("="*80)

    # Find best F1 threshold
    best_thresh = max(results["thresholds"].items(),
                      key=lambda x: x[1]["f1_score"])
    print(f"\nBest F1 score: {best_thresh[1]['f1_score']:.3f} at conf={best_thresh[0]:.1f}")
    results["best_threshold"] = {
        "confidence": best_thresh[0],
        "metrics": best_thresh[1]
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compute object detection metrics for YOGO models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on test set with default thresholds
  python -m yogo.compute_metrics results/fold1/checkpoints/best.pth dataset/malaria_fold1.yaml

  # Evaluate on validation set with custom thresholds
  python -m yogo.compute_metrics model.pth dataset.yaml --split val --conf-thresholds 0.5 0.7

  # Evaluate first 50 samples and save results
  python -m yogo.compute_metrics model.pth dataset.yaml --max-samples 50 --output metrics.json
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
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for detailed metrics (default: print only)"
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
        return 1

    if not dataset_path.exists():
        print(f"ERROR: Dataset definition not found: {dataset_path}")
        return 1

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
            max_samples=args.max_samples,
            device=device,
        )

        # Save results if output specified
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nDetailed metrics saved to: {output_path}")

        return 0

    except Exception as e:
        print(f"\nERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
