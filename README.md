# Malaria Detection with YOGO

Automated malaria parasite (Plasmodium falciparum) detection from microscopy images using the YOGO (You Only Glance Once) object detection framework.

## Project Overview

This project uses YOGO, a simplified YOLO architecture optimized for real-time object detection on uniform-sized objects, to detect malaria parasites in microscopy images. The dataset includes annotated DPC (Differential Phase Contrast) and fluorescent microscopy images from samples collected across Africa.

**Key Features:**
- Real-time malaria parasite detection
- Support for 4-channel input (DPC + RGB fluorescent) or grayscale/RGB
- Tiling support for large microscopy images with overlap
- Optimized for low-latency inference on limited hardware
- Dataset includes 15 patients (60 FOVs) with 3,576 verified annotations

## Dataset

The dataset contains microscopy images from multiple sources:
- **Uganda**: 4 samples (3,278 positives) - high parasitemia
- **Rwanda**: 4 samples (111 positives) - low parasitemia
- **Nigeria**: 3 samples (187 positives, 2 negatives)
- **SBC (USA)**: 4 samples (quality control negatives)

**Image specifications:**
- Resolution: 2800×2800 (73%) or 3000×3000 (27%) pixels
- DPC format: PNG, 8-bit grayscale (1 channel)
- Fluorescent format: PNG, RGB (3 channels)
- Combined: 4-channel input (DPC + RGB fluorescent)
- Bounding box size: 31×31 pixels (fixed)
- Coordinate format: YOLO-style normalized centers
- Tiling: 1024×1024 with 256px overlap (9 tiles per FOV)

For detailed dataset information, see [dataset/CLAUDE.md](dataset/CLAUDE.md) and [dataset/README.md](dataset/README.md).

## Installation

Requires Python >= 3.9 and < 3.11.

```bash
# Install in development mode
python3 -m pip install -e ".[dev]"
```

## Basic Usage

```bash
# Train a model with 4-channel input (DPC + RGB fluorescent)
python -m yogo.train dataset/malaria_stratified_cv_fold1.yaml \
    --image-hw 1024 1024 \
    --input-channels 4 \
    --tile-size 1024 \
    --tile-overlap 256 \
    --batch-size 8 \
    --epochs 50 \
    --model base_model \
    --normalize-images \
    --half \
    --no-obj-weight 0.05

# Compute metrics on test data
python -m yogo.compute_metrics \
    results/fold1/checkpoints/best.pth \
    dataset/malaria_stratified_cv_fold1.yaml

# Compute metrics at specific confidence thresholds
python -m yogo.compute_metrics model.pth dataset.yaml \
    --conf-thresholds 0.3 0.5 0.7 \
    --output metrics.json

# Run inference
yogo infer path/to/model.pth

# Export model for deployment
yogo export path/to/model.pth

# Get help
yogo --help
```

**Note:** GPU training is currently required (uses PyTorch Distributed Data Parallel).

**Key Parameters:**
- `--input-channels 4`: Use 4-channel input (DPC + RGB fluorescent)
- `--tile-size 1024`: Tile size for large images
- `--tile-overlap 256`: Overlap between tiles (reduces edge artifacts)
- `--no-obj-weight 0.05`: Background loss weight (critical for malaria detection)

## Project Structure

```
.
├── yogo/                      # YOGO source code (model, training, inference)
├── dataset/                   # Malaria microscopy dataset
│   ├── CLAUDE.md             # Dataset metadata and statistics
│   ├── README.md             # Annotation format specification
│   └── [sample_dirs]/        # Individual samples with FOVs
├── docs/                      # YOGO documentation
│   ├── dataset-definition.md # How to format datasets
│   ├── yogo-high-level.md    # Architecture overview
│   ├── cli.md                # Command line guide
│   └── recipes.md            # Code examples
├── trained_models/           # Saved model checkpoints
├── examples/                 # Test scripts and examples
├── scripts/                  # Utility scripts
├── tests/                    # Test suite
├── yogo_paper.pdf           # YOGO research paper
└── pyproject.toml           # Package configuration
```

## Documentation

- **[docs/yogo-high-level.md](docs/yogo-high-level.md)** - YOGO architecture explanation
- **[docs/dataset-definition.md](docs/dataset-definition.md)** - Dataset format specification
- **[docs/cli.md](docs/cli.md)** - Command line interface guide
- **[docs/recipes.md](docs/recipes.md)** - Code usage examples
- **[yogo_paper.pdf](yogo_paper.pdf)** - Original YOGO research paper

## Development Guidelines

This project uses Claude Code with project-specific rules in `.claude/rules/`:
- **Documentation Rule** - All code changes must include documentation updates in the same commit
- See `.claude/rules/documentation.md` for the full documentation workflow

## Training Considerations

Based on the dataset characteristics:

**Image preprocessing:**
- Original images are 2800×2800 or 3000×3000 pixels
- **Tiling approach (implemented):** 1024×1024 tiles with 256px overlap
  - Preserves full resolution for ~31px parasites
  - 9 tiles per FOV, 540 total training samples from 60 FOVs
  - NMS during inference removes duplicate detections from overlaps
- 4-channel input: DPC (1 channel) + RGB fluorescent (3 channels)

**Data split:**
- **Patient-level splitting (critical):** All FOVs from same patient stay together
- Site-stratified split recommended for diversity
- Current: 67% train / 20% val / 13% test (10 / 3 / 2 patients)

**Loss weighting:**
- **Critical:** Use `--no-obj-weight 0.05` to balance background vs object loss
- Default 0.5 causes objectness learning failure due to 1743:1 background ratio
- Lower weight (0.05) enables model to learn confident predictions

**Class imbalance:**
- High parasitemia samples (Uganda) dominate positive detections
- Empty tiles from negative samples help reduce false positives
- Both positive and negative tiles included naturally via tiling

**Data augmentation:**
- Rotation and flips (parasites have no preferred orientation)
- Brightness/contrast adjustments (simulate microscopy variations)
- Minimal elastic deformations (RBC deformation may confuse model)

## Weights & Biases Integration

YOGO uses [Weights and Biases](http://wandb.ai) for run tracking. You don't need a W&B account - runs without an account are logged to an anonymous page that can be [claimed later](https://docs.wandb.ai/guides/app/features/anon).

## Reference

YOGO is based on the YOLO architecture (versions 1-3) and was developed by the bioengineering team at Chan-Zuckerberg Biohub SF for the [Remoscope project](https://www.czbiohub.org/life-science/seeing-malaria-in-a-new-light/).

Original YOGO documentation: [examples/YOGO_original_README.md](examples/YOGO_original_README.md)
