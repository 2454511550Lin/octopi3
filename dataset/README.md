# Annotation Output Format

This folder contains human-reviewed annotations and images for machine learning training.

**Created by:** Annotation GUI (`./run_gui.sh`)
**Format Version:** 2.0 (January 2026)

---

## Directory Structure

```
annotated/
├── {sample_name}/
│   └── {fov_id}/
│       ├── dpc.png              # DPC image (grayscale)
│       ├── fluorescent.png      # Fluorescent image
│       ├── spots.csv            # Spot locations and labels
│       └── metadata.json        # Image info and statistics
└── README.md
```

**Note:** Previous versions used `annotated/{split}/{sample}/{fov_id}/` structure with additional files. See [Legacy Format](#legacy-format) section.

---

## File Descriptions

### dpc.png
- **Format:** PNG (lossless compression)
- **Content:** Differential Phase Contrast image (grayscale)
- **Use:** ML input channel 1

### fluorescent.png
- **Format:** PNG (lossless compression)
- **Content:** Fluorescence image (405nm excitation for DAPI)
- **Use:** ML input channels 2-4 (can be RGB or grayscale depending on source)

### spots.csv

Contains all annotated spots with both pixel and normalized coordinates.

| Column | Type | Description |
|--------|------|-------------|
| `spot_id` | int | Unique identifier for the spot |
| `x` | int | X coordinate (pixels, top-left of bounding box) |
| `y` | int | Y coordinate (pixels, top-left of bounding box) |
| `width` | int | Bounding box width (pixels) |
| `height` | int | Bounding box height (pixels) |
| `x_norm` | float | Normalized X center (0-1, relative to image width) |
| `y_norm` | float | Normalized Y center (0-1, relative to image height) |
| `w_norm` | float | Normalized width (0-1) |
| `h_norm` | float | Normalized height (0-1) |
| `label` | string | `positive`, `negative`, `unsure`, or `ignored` |
| `confidence` | float | Model confidence score (NaN for human-added spots) |
| `source` | string | `model`, `human_added`, or `human_modified` |

**Example:** (for 3000×3000 image)
```csv
spot_id,x,y,width,height,x_norm,y_norm,w_norm,h_norm,label,confidence,source
1,1234,567,31,31,0.4163,0.1942,0.010333,0.010333,positive,0.95,model
2,890,123,31,31,0.3017,0.0462,0.010333,0.010333,negative,0.87,model
3,456,789,31,31,0.1570,0.2680,0.010333,0.010333,unsure,0.52,human_modified
4,600,400,31,31,0.2052,0.1385,0.010333,0.010333,positive,,human_added
```

**Bounding box size:** 31×31 pixels (radius=15, diameter = 2×15+1 = 31)

**Notes:**
- Normalized coordinates use YOLO convention: `x_norm`, `y_norm` are **center** coordinates
- Pixel coordinates `x`, `y` are **top-left** of bounding box
- "Excluded" spots (outside RBC mask) are **not included** in output
- Empty confidence means human-added spot (no model prediction)

### metadata.json

Contains image information, RBC statistics, and annotation summary.

```json
{
  "fov_id": "HP242868_30",
  "sample": "HP242868 R_20251210_042253",

  "image": {
    "width": 3000,
    "height": 3000,
    "cropped": false,
    "crop_offset": [0, 0]
  },

  "rbc_stats": {
    "total_rbcs": 1847,
    "mask_coverage_percent": 45.2
  },

  "annotation_stats": {
    "total_spots": 156,
    "positive": 23,
    "negative": 120,
    "unsure": 8,
    "ignored": 5,
    "human_added": 3,
    "human_modified": 15
  },

  "timestamps": {
    "detection_run": "2026-01-20T14:30:00",
    "last_annotated": "2026-01-28T10:15:00"
  }
}
```

**Fields:**
- `image.cropped`: Whether image was cropped from original
- `image.crop_offset`: `[x, y]` offset if cropped (for coordinate mapping)
- `rbc_stats.total_rbcs`: Number of RBCs detected by Cellpose
- `rbc_stats.mask_coverage_percent`: Percentage of image covered by RBC mask

---

## Generating Training Data

### YOLO Format

Generate YOLO labels from spots.csv:

```python
import pandas as pd

df = pd.read_csv('spots.csv')

# Filter to positive spots only (or include negative as class 1)
positives = df[df['label'] == 'positive']

# Write YOLO format: class_id x_center y_center width height
with open('labels.txt', 'w') as f:
    for _, row in positives.iterrows():
        f.write(f"0 {row['x_norm']:.6f} {row['y_norm']:.6f} {row['w_norm']:.6f} {row['h_norm']:.6f}\n")
```

### ResNet Patches

Extract 31x31 patches for classification:

```python
import numpy as np
import pandas as pd
from PIL import Image

# Load images
dpc = np.array(Image.open('dpc.png'))
fluor = np.array(Image.open('fluorescent.png'))

# Load spots
df = pd.read_csv('spots.csv')
df = df[df['label'].isin(['positive', 'negative'])]  # Exclude unsure/ignored

patches = []
labels = []
half_size = 15  # 31x31 patch

for _, row in df.iterrows():
    cx = row['x'] + row['width'] // 2
    cy = row['y'] + row['height'] // 2

    # Extract patch (handle boundaries as needed)
    dpc_patch = dpc[cy-half_size:cy+half_size+1, cx-half_size:cx+half_size+1]
    fluor_patch = fluor[cy-half_size:cy+half_size+1, cx-half_size:cx+half_size+1]

    # Stack to 4 channels: DPC (1) + Fluorescent (3 or replicated)
    if len(fluor_patch.shape) == 2:
        fluor_patch = np.stack([fluor_patch]*3, axis=-1)

    patch = np.concatenate([dpc_patch[..., np.newaxis], fluor_patch], axis=-1)
    patches.append(patch.transpose(2, 0, 1))  # (4, 31, 31)
    labels.append(1 if row['label'] == 'positive' else 0)

np.save('patches.npy', np.array(patches))
np.save('patch_labels.npy', np.array(labels))
```

### 4-Channel Input for Object Detection

```python
import numpy as np
from PIL import Image

# Load both images
dpc = np.array(Image.open('dpc.png'))      # (H, W) or (H, W, 1)
fluor = np.array(Image.open('fluorescent.png'))  # (H, W) or (H, W, 3)

# Ensure correct shapes
if len(dpc.shape) == 2:
    dpc = dpc[..., np.newaxis]
if len(fluor.shape) == 2:
    fluor = np.stack([fluor]*3, axis=-1)

# Stack to 4 channels: (H, W, 4)
combined = np.concatenate([dpc, fluor], axis=-1)
```

---

## Legacy Format

Previous versions (before January 2026) used a different structure:

```
annotated/{split}/{sample}/{fov_id}/
├── annotations.csv       # Human annotations (different schema)
├── overlay.png           # DPC+fluorescence composite
├── labels.txt            # YOLO format (pre-generated)
├── patches.npy           # ResNet patches (pre-generated)
├── patch_labels.npy      # Patch labels
└── metadata.csv          # Patch metadata
```

The legacy format is preserved in the `legacy-output-format` git branch.

To migrate legacy annotations to the new format:
```bash
python scripts/migrate_annotations.py
```

---

## Workflow

1. Run the annotation GUI: `./run_gui.sh`
2. Review and annotate spots:
   - **P** = Positive (parasite)
   - **N** = Negative (not parasite)
   - **U** = Unsure
   - **I** = Ignored (artifact)
3. Press **Ctrl+S** or **Enter** to save
4. Files are automatically exported to `annotated/{sample}/{fov_id}/`

---

## Statistics Across Dataset

To get aggregate statistics across all annotations:

```python
import json
from pathlib import Path

annotated_dir = Path('annotated')
total_stats = {'positive': 0, 'negative': 0, 'unsure': 0, 'ignored': 0}

for meta_file in annotated_dir.glob('*/*/metadata.json'):
    with open(meta_file) as f:
        meta = json.load(f)
    for key in total_stats:
        total_stats[key] += meta['annotation_stats'].get(key, 0)

print(f"Total annotations: {sum(total_stats.values())}")
print(f"Positive: {total_stats['positive']}")
print(f"Negative: {total_stats['negative']}")
```
