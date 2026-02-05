# Malaria Detection Dataset - Metadata and Statistics

**Document Created:** February 2026
**Total Samples:** 15 (9 positive, 6 negative)
**Total FOVs:** 60 (4 FOVs per sample)
**Total Annotations:** 3,576 positive spots verified

---

## Dataset Overview

This dataset contains annotated microscopy images for malaria parasite (Plasmodium falciparum) detection from multiple sources across Africa. Each sample contains 4 Fields of View (FOVs) with both DPC (Differential Phase Contrast) and fluorescent microscopy images at 3000×3000 pixels.

**Annotation Format:** See [README.md](./README.md) for detailed file format specifications.

---

## Sample Inventory

### Uganda Samples (High Positive)
*Source: gs://octopi-malaria-uganda-2022-data*

| Sample ID | Status | Total Positives | FOVs | Verified |
|-----------|--------|----------------|------|----------|
| `BUS-114-1_2023-01-21_19-25-3.663354` | POSITIVE | 1,601 | 4 | ✓ |
| `PAT-097-2_2023-03-31_23-36-59.835903` | POSITIVE | 787 | 4 | ✓ |
| `PAT-104-AR-1_2023-04-02_19-50-32.175255` | POSITIVE | 492 | 4 | ✓ |
| `PAT-101-2_2023-09-02_17-04-12.388423` | POSITIVE | 398 | 4 | ✓ |
| **Subtotal** | | **3,278** | **16** | |

### Brown University / Rwanda Samples
*Source: gs://brownu*

| Sample ID | Status | Total Positives | FOVs | Verified |
|-----------|--------|----------------|------|----------|
| `3457_20X_20240827_081250` | POSITIVE | 60 | 4 | ✓ |
| `3461_20X_20240827_083141` | POSITIVE | 37 | 4 | ✓ |
| `3456_20X_20240827_080751` | Low positive | 8 | 4 | ✓ |
| `3459_20X_20240827_082100` | Low positive | 6 | 4 | ✓ |
| **Subtotal** | | **111** | **16** | |

### Nigeria Samples
*Source: Local Hard Drive (HP/Octopi samples)*

| Sample ID | Status | Total Positives | FOVs | Verified |
|-----------|--------|----------------|------|----------|
| `HP242868 R_20251210_042253` | POSITIVE | 187 | 4 | ✓ |
| `Octopi 254 HP243945_20250912_230224` | NEGATIVE | 0 | 4 | ✓ |
| `Octopi 258 HP230692_20250912_231709` | NEGATIVE | 0 | 4 | ✓ |
| **Subtotal** | | **187** | **12** | |

### SBC Negative Controls
*Source: gs://sbc07252024-reprocess*

| Sample ID | Status | Total Positives | FOVs | Verified |
|-----------|--------|----------------|------|----------|
| `SBC_20240725_06A` | NEGATIVE | 0 | 4 | ✓ |
| `SBC_20240725_09A` | NEGATIVE | 0 | 4 | ✓ |
| `SBC_20240725_12A` | NEGATIVE | 0 | 4 | ✓ |
| `SBC_20240725_15A` | NEGATIVE | 0 | 4 | ✓ |
| **Subtotal** | | **0** | **16** | |

---

## Dataset Statistics

### By Status
- **Positive samples:** 9 (60%)
- **Negative samples:** 6 (40%)
- **Low positive samples:** 2 (included in positive count)

### By Geography
- **Uganda:** 4 samples (3,278 positives)
- **Rwanda:** 4 samples (111 positives)
- **Nigeria:** 3 samples (187 positives, 2 negatives)
- **SBC (USA):** 4 samples (negatives only, quality controls)

### Annotation Summary
- **Total annotated spots:** 3,576 positives (verified)
- **Additional negative annotations:** Present in all FOVs
- **Annotation types:** `positive`, `negative`, `unsure`, `ignored`
- **Sources:** `model`, `human_added`, `human_modified`

### Parasitemia Distribution
- **Very high (>1000 parasites):** 1 sample (BUS-114-1)
- **High (400-1000):** 2 samples (PAT-097-2, PAT-104-AR-1)
- **Medium (100-400):** 2 samples (PAT-101-2, HP242868)
- **Low (10-100):** 2 samples (3457, 3461)
- **Very low (<10):** 2 samples (3456, 3459)
- **Negative (0):** 6 samples

---

## Image Specifications (Verified February 2026)

### Dimension Statistics
- **DPC images:** 60 total
  - 2800×2800: 44 images (73%)
  - 3000×3000: 16 images (27%)
  - Format: 8-bit grayscale PNG
  - Mean size: ~4.4 MB

- **Fluorescent images:** 60 total
  - 2800×2800: 44 images (73%)
  - 3000×3000: 16 images (27%)
  - Format: RGB PNG (8-bit per channel)
  - Mean size: ~7.7 MB

### Dimension Distribution by Sample
- **2800×2800 samples (11):** Rwanda, Nigeria, SBC samples
- **3000×3000 samples (4):** Uganda high-parasitemia samples

**Note:** All image pairs within FOVs have matching dimensions.

### Detection Target
- **Bounding box size:** 31 × 31 pixels (fixed)
- **Parasite size:** ~31px diameter (requires full resolution)
- **Coordinate format:** YOLO-style (normalized centers)

### Tiling Strategy for Training
To preserve full resolution for small (~31px) parasites while fitting memory constraints:

- **Tile size:** 1024×1024 pixels
- **Overlap:** 256 pixels (stride=768)
- **Tiles per FOV:** 9 (3×3 grid) for both 2800×2800 and 3000×3000 images
- **Total training samples:** 540 tiles (60 FOVs × 9 tiles)
- **Grid capacity per tile:** 128×128 cells = 16,384 cells (36× margin for max ~450 detections/tile)
- **Memory per tile:** 4.2MB (1024×1024×4 channels)

**Rationale:**
- Preserves parasites at full ~31px resolution (no downsampling)
- Overlap ensures parasites near tile edges appear in multiple tiles
- NMS during inference removes duplicate detections from overlaps
- 540 samples provide good training data from 60 FOVs

---

## Training Configuration (February 2026)

### YOGO Training Command
```bash
python -m yogo.train dataset/malaria_dataset_defn.yaml \
    --image-hw 1024 1024 \
    --input-channels 4 \
    --tile-size 1024 \
    --tile-overlap 256 \
    --batch-size 16 \
    --epochs 100 \
    --learning-rate 1e-4 \
    --model base_model \
    --normalize-images \
    --half
```

### Key Training Parameters
- **Input:** 4 channels (DPC + RGB fluorescent)
- **Image size:** 1024×1024 pixels (tile size)
- **Classes:** 2 (positive, unsure)
- **Batch size:** 16 (adjust based on GPU memory)
- **Training samples:** 540 tiles (360 train / 108 val / 72 test)
- **Patient-level splits:** Critical to avoid data leakage

### Dataset Split Strategy
- **Train:** 10 patients (40 FOVs, 360 tiles, 67%)
- **Validation:** 3 patients (12 FOVs, 108 tiles, 20%)
- **Test:** 2 patients (8 FOVs, 72 tiles, 13%)

**Important:** All FOVs from the same patient must stay in the same split to prevent data leakage.

---

## Excluded from Current Training Set

The following sample was annotated but excluded due to high number of "unsure" annotations:

| Sample ID | Status | Total Positives | Notes |
|-----------|--------|----------------|-------|
| `HP242880 R_20251210_035653` | POSITIVE | 34 | High uncertainty - reserved for testing |

This sample can be used for:
- Model validation
- Uncertainty quantification testing
- Edge case analysis

---

## Data Quality Notes

### Verified Consistency ✓
- All 15 samples present in dataset directory
- All samples have exactly 4 FOVs
- Positive counts match expected values
- Negative samples confirmed with 0 positives
- All FOVs contain: `dpc.png`, `fluorescent.png`, `spots.csv`, `metadata.json`

### Annotation Quality
- **High confidence samples:** Uganda samples (high parasitemia, clear annotations)
- **Low confidence samples:** Rwanda samples (low parasitemia, fewer parasites)
- **Quality controls:** SBC samples provide true negatives for model training

### Class Balance
- **Positive samples:** 9 (60% of samples, but 3,576 positive spots)
- **Negative samples:** 6 (40% of samples)
- **Positive FOVs:** ~36 FOVs (estimated, based on distribution)
- **Negative FOVs:** ~24 FOVs (includes negative FOVs from positive samples)

---

## Related Files

- **[README.md](./README.md)** - Annotation format and file structure details
- **`spots.csv`** - Annotation data (see README for schema)
- **`metadata.json`** - Image statistics and annotation summary

---

## Verification Command

To verify dataset integrity, run:

```python
import os
import csv

base_dir = 'dataset'
expected_samples = 15
expected_fovs_per_sample = 4

sample_count = 0
for sample in os.listdir(base_dir):
    sample_path = os.path.join(base_dir, sample)
    if os.path.isdir(sample_path) and sample != '.git':
        sample_count += 1
        fov_count = len([f for f in os.listdir(sample_path)
                        if os.path.isdir(os.path.join(sample_path, f))])
        assert fov_count == expected_fovs_per_sample, f"{sample}: Expected {expected_fovs_per_sample} FOVs, found {fov_count}"

assert sample_count == expected_samples, f"Expected {expected_samples} samples, found {sample_count}"
print(f"✓ Dataset verified: {sample_count} samples, {sample_count * expected_fovs_per_sample} FOVs")
```

**Last verified:** February 1, 2026 - All checks passed ✓

---

## Contact & Attribution

- **Dataset compiled by:** Octopi Malaria Detection Project
- **Sources:** Uganda, Rwanda, Nigeria, Stanford Blood Center
- **Annotation tool:** Custom GUI (see main repository)
- **ML framework:** YOGO (You Only Glance Once)

For questions about specific samples or annotations, refer to the source metadata in each sample's `metadata.json` file.
