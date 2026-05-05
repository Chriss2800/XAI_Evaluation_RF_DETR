# XAI_Evaluation_RF_DETR

# dataprep.py

This script prepares the Lung-PET-CT-Dx dataset for object detection training in YOLO format. It reads DICOM images and their corresponding XML annotations, converts CT slices into PNG images, and exports the bounding boxes as YOLO label files.

## What the Script Does

The script performs the following steps:

### 1. Determine Project Paths

The script automatically determines the main project directory based on the location of the script file.

Expected project structure:

```text
project/
├── data/
│   ├── raw/
│   │   └── Lung-PET-CT-Dx/
│   ├── Annotation/
│   └── processed/
└── scripts/
    └── <this_script>.py
```

### 2. Locate DICOM Images and XML Annotations
### 3. Create Patient-Level Train/Validation/Test Splits
### 4. Load DICOM Images and Apply CT Windowing
### 5. Match DICOM Images with XML Annotations
### 6. Convert Bounding Boxes to YOLO Format
### 7. Export YOLO-Compatible Dataset Structure

The processed dataset is saved under:

```text
data/processed/
```

with the following structure:

```text
data/processed/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## Output

For every DICOM image with a matching XML annotation, the script creates one PNG image and one YOLO label file.

Example:

```text
data/processed/train/images/R01-001_1.2.840....png
data/processed/train/labels/R01-001_1.2.840....txt
```

Each label file contains one line per bounding box:

```text
0 x_center y_center width height
```

Example:

```text
0 0.512345 0.438921 0.104321 0.087654
```

## Important Notes

- The train/validation/test split is performed at patient level.
- Only DICOM files with a matching XML annotation are exported.
- DICOM images without matching XML annotations are skipped.
- The script assumes a single detection class and writes `0` as the class ID for every bounding box.
- CT windowing is applied using a lung window with width `1400` and level `-700`.
- Errors during individual DICOM or XML processing are skipped using `try/except`.
- The output is compatible with YOLO-style object detection training pipelines.