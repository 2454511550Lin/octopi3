"""
Malaria detection dataset with tiling support for large 4-channel microscopy images.

Loads paired DPC (grayscale) + fluorescent (RGB) images, tiles them into overlapping
patches, and maps annotations to tile coordinates.
"""

import csv
import torch
from pathlib import Path
from typing import List, Tuple, Dict, Set
from torch.utils.data import Dataset
from PIL import Image

from yogo.data.utils import read_image_robust
from yogo.data.yogo_dataset import format_labels_tensor, LABEL_TENSOR_PRED_DIM_SIZE

try:
    from torchvision.ops import box_convert
except ImportError:
    from torchvision import ops
    box_convert = ops.box_convert


class MalariaDetectionDataset(Dataset):
    """
    Dataset for malaria parasite detection from paired microscopy images.

    Each FOV (Field of View) contains:
    - dpc.png: Grayscale differential phase contrast image (1 channel)
    - fluorescent.png: RGB fluorescent image (3 channels)
    - spots.csv: Annotations with labels (positive, unsure, negative)

    The dataset tiles large images (2800x2800 or 3000x3000) into smaller patches
    (default 1024x1024) with overlap (default 256px) to preserve full resolution
    for detection of ~31px parasites.

    Args:
        image_folder_path: Root path containing sample folders
        label_folder_path: Root path for labels (same as image_folder_path)
        Sx: Grid width for YOGO model
        Sy: Grid height for YOGO model
        classes: List of class names (e.g., ["positive", "unsure"])
        tile_size: Size of tiles in pixels (default: 1024)
        overlap: Overlap between tiles in pixels (default: 256)
        normalize_images: If True, normalize images to [0, 1] (default: False)
        filter_labels: Labels to include (default: ["positive", "unsure"])
    """

    def __init__(
        self,
        image_folder_path: Path,
        label_folder_path: Path,
        Sx: int,
        Sy: int,
        classes: List[str],
        tile_size: int = 1024,
        overlap: int = 256,
        normalize_images: bool = False,
        filter_labels: List[str] = None,
        **kwargs
    ):
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap

        if filter_labels is None:
            filter_labels = ["positive", "unsure"]
        self.filter_labels = set(filter_labels)

        self.Sx = Sx
        self.Sy = Sy
        self.classes = classes
        self.normalize_images = normalize_images

        # Find all FOVs
        self.fov_paths = self._find_fovs(image_folder_path)

        if len(self.fov_paths) == 0:
            raise ValueError(f"No FOVs found in {image_folder_path}")

        # Pre-compute tile metadata
        self.tiles = self._compute_tile_metadata()

        print(f"MalariaDetectionDataset initialized:")
        print(f"  FOVs found: {len(self.fov_paths)}")
        print(f"  Total tiles: {len(self.tiles)}")
        print(f"  Tile size: {tile_size}x{tile_size}")
        print(f"  Overlap: {overlap}px (stride: {self.stride}px)")
        print(f"  Filter labels: {filter_labels}")

    def _find_fovs(self, root: Path) -> List[Path]:
        """
        Find all FOV directories containing dpc.png and fluorescent.png.

        Expected structure:
        root/
            sample_1/
                FOV_1/
                    dpc.png
                    fluorescent.png
                    spots.csv
                FOV_2/
                    ...
            sample_2/
                ...
        """
        fovs = []
        root = Path(root)

        # Check if root itself is a sample directory
        if (root / "dpc.png").exists() and (root / "fluorescent.png").exists():
            fovs.append(root)
            return fovs

        # Otherwise, iterate through sample directories
        for sample_dir in sorted(root.iterdir()):
            if not sample_dir.is_dir():
                continue

            # Check if this is a FOV directory
            if (sample_dir / "dpc.png").exists() and (sample_dir / "fluorescent.png").exists():
                fovs.append(sample_dir)
                continue

            # Otherwise, check subdirectories (FOVs within sample)
            for fov_dir in sorted(sample_dir.iterdir()):
                if (fov_dir.is_dir() and
                    (fov_dir / "dpc.png").exists() and
                    (fov_dir / "fluorescent.png").exists()):
                    fovs.append(fov_dir)

        return fovs

    def _compute_tile_metadata(self) -> List[Dict]:
        """
        Pre-compute all tiles from all FOVs.

        Returns:
            List of tile metadata dicts with keys:
                - fov_path: Path to FOV directory
                - offset: (y, x) offset of tile in original image
                - image_size: (H, W) size of original image
        """
        tiles = []

        for fov_path in self.fov_paths:
            # Get image dimensions
            dpc_path = fov_path / "dpc.png"
            with Image.open(dpc_path) as dpc:
                H, W = dpc.height, dpc.width

            # Compute tile positions with overlap
            y_positions = list(range(0, H - self.tile_size + 1, self.stride))
            x_positions = list(range(0, W - self.tile_size + 1, self.stride))

            # Ensure we cover the entire image
            if not y_positions or y_positions[-1] + self.tile_size < H:
                y_positions.append(H - self.tile_size)
            if not x_positions or x_positions[-1] + self.tile_size < W:
                x_positions.append(W - self.tile_size)

            for y in y_positions:
                for x in x_positions:
                    tiles.append({
                        'fov_path': fov_path,
                        'offset': (y, x),
                        'image_size': (H, W)
                    })

        return tiles

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single tile and its labels.

        Returns:
            tile: (4, tile_size, tile_size) tensor with channels [DPC, R, G, B]
            labels: (LABEL_TENSOR_PRED_DIM_SIZE, Sy, Sx) formatted label tensor
        """
        tile_info = self.tiles[index]
        fov_path = tile_info['fov_path']
        y_off, x_off = tile_info['offset']
        H, W = tile_info['image_size']

        # Load full images
        dpc = read_image_robust(fov_path / "dpc.png", rgb=False)  # (1, H, W)
        fluorescent = read_image_robust(fov_path / "fluorescent.png", rgb=True)  # (3, H, W)

        # Concatenate to 4 channels: [DPC, R, G, B]
        full_image = torch.cat([dpc, fluorescent], dim=0)  # (4, H, W)

        # Extract tile
        tile = full_image[:, y_off:y_off+self.tile_size, x_off:x_off+self.tile_size]

        # Load and filter labels
        csv_path = fov_path / "spots.csv"
        labels = self._load_and_map_labels(csv_path, (y_off, x_off), (H, W))

        # Normalize if requested
        if self.normalize_images:
            tile = tile.float() / 255.0

        return tile, labels

    def _load_and_map_labels(
        self,
        csv_path: Path,
        tile_offset: Tuple[int, int],
        image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Load CSV annotations, filter by class, and map to tile coordinates.

        Args:
            csv_path: Path to spots.csv file
            tile_offset: (y, x) offset of tile in original image
            image_size: (H, W) original image size

        Returns:
            Formatted label tensor for YOGO
        """
        y_off, x_off = tile_offset
        H, W = image_size

        tile_labels = []

        # Check if CSV exists
        if not csv_path.exists():
            # No labels for this FOV (likely a negative control)
            return torch.zeros(LABEL_TENSOR_PRED_DIM_SIZE, self.Sy, self.Sx)

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Filter by label type (skip "negative" annotations)
                if row['label'] not in self.filter_labels:
                    continue

                # Get normalized coordinates
                x_norm = float(row['x_norm'])
                y_norm = float(row['y_norm'])
                w_norm = float(row['w_norm'])
                h_norm = float(row['h_norm'])

                # Convert to pixel coordinates
                x_center = x_norm * W
                y_center = y_norm * H

                # Check if center is in tile
                if not (x_off <= x_center < x_off + self.tile_size and
                        y_off <= y_center < y_off + self.tile_size):
                    continue

                # Map label to class index
                try:
                    class_idx = self.classes.index(row['label'])
                except ValueError:
                    # Label not in classes list, skip
                    continue

                # Adjust to tile coordinate system (normalized to tile)
                new_x_norm = (x_center - x_off) / self.tile_size
                new_y_norm = (y_center - y_off) / self.tile_size
                new_w_norm = (w_norm * W) / self.tile_size
                new_h_norm = (h_norm * H) / self.tile_size

                tile_labels.append([
                    float(class_idx),
                    new_x_norm,
                    new_y_norm,
                    new_w_norm,
                    new_h_norm
                ])

        # If no labels in tile, return empty tensor
        if not tile_labels:
            return torch.zeros(LABEL_TENSOR_PRED_DIM_SIZE, self.Sy, self.Sx)

        # Convert to tensor and format
        labels_tensor = torch.Tensor(tile_labels)

        # Convert from cxcywh to xyxy format for format_labels_tensor
        labels_tensor[:, 1:] = box_convert(labels_tensor[:, 1:], "cxcywh", "xyxy")

        return format_labels_tensor(labels_tensor, self.Sx, self.Sy)
