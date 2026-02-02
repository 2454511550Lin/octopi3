# Malaria Detection with YOGO

Automated malaria parasite (Plasmodium falciparum) detection from microscopy images using the YOGO (You Only Glance Once) object detection framework.

## Project Overview

This project uses YOGO, a simplified YOLO architecture optimized for real-time object detection on uniform-sized objects, to detect malaria parasites in microscopy images. The dataset includes annotated DPC (Differential Phase Contrast) and fluorescent microscopy images from samples collected across Africa.

**Key Features:**
- Real-time malaria parasite detection
- Support for grayscale DPC and RGB fluorescent microscopy
- Optimized for low-latency inference on limited hardware
- Dataset includes 15 samples (9 positive, 6 negative) with 3,576 verified annotations

## Dataset

The dataset contains microscopy images from multiple sources:
- **Uganda**: 4 samples (3,278 positives) - high parasitemia
- **Rwanda**: 4 samples (111 positives) - low parasitemia
- **Nigeria**: 3 samples (187 positives, 2 negatives)
- **SBC (USA)**: 4 samples (quality control negatives)

**Image specifications:**
- Resolution: 3000×3000 pixels
- DPC format: PNG, 8-bit grayscale
- Fluorescent format: PNG, RGB or 8-bit grayscale
- Bounding box size: 31×31 pixels (fixed)
- Coordinate format: YOLO-style normalized centers

For detailed dataset information, see [dataset/CLAUDE.md](dataset/CLAUDE.md) and [dataset/README.md](dataset/README.md).

## Installation

Requires Python >= 3.9 and < 3.11.

```bash
# Install in development mode
python3 -m pip install -e ".[dev]"
```

## Basic Usage

```bash
# Train a model
yogo train path/to/dataset-definition.yml

# Test a model
yogo test path/to/model.pth path/to/dataset-definition.yml

# Run inference
yogo infer path/to/model.pth

# Export model for deployment
yogo export path/to/model.pth

# Get help
yogo --help
```

**Note:** GPU training is currently required (uses PyTorch Distributed Data Parallel).

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
- Original images are 3000×3000, may need resizing/tiling
- Recommended: 1024×1024 tiles with overlap OR resize to 1536×1536
- DPC (grayscale) matches YOGO default input format

**Data split:**
- Stratified split by geography recommended for diversity
- Suggested: 70% train / 15% val / 15% test

**Class imbalance:**
- High parasitemia samples (Uganda) will dominate training
- Consider weighted sampling or augmentation for low-positive samples
- Ensure negative samples are well-represented

**Data augmentation:**
- Rotation and flips (parasites have no preferred orientation)
- Brightness/contrast adjustments (simulate microscopy variations)
- Minimal elastic deformations (RBC deformation may confuse model)

## Weights & Biases Integration

YOGO uses [Weights and Biases](http://wandb.ai) for run tracking. You don't need a W&B account - runs without an account are logged to an anonymous page that can be [claimed later](https://docs.wandb.ai/guides/app/features/anon).

## Reference

YOGO is based on the YOLO architecture (versions 1-3) and was developed by the bioengineering team at Chan-Zuckerberg Biohub SF for the [Remoscope project](https://www.czbiohub.org/life-science/seeing-malaria-in-a-new-light/).

Original YOGO documentation: [examples/YOGO_original_README.md](examples/YOGO_original_README.md)
