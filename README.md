# XAI_Evaluation_RF_DETR

This repository contains the code for a study on explainable AI for transformer-based medical object detection. The goal is to train and evaluate an RF-DETR Medium model for lung nodule detection on the Lung-PET-CT-Dx dataset and to compare different XAI methods with respect to how well their explanations spatially correspond to annotated tumor regions.

The workflow covers the full experimental pipeline: DICOM images and XML annotations are first converted into a YOLO-compatible dataset, then transformed into COCO format for RF-DETR training. After training, the model is evaluated using standard object detection metrics at different IoU thresholds. Finally, Grad-CAM, Attention Rollout, and D-RISE are applied to the trained model and quantitatively compared using Pointing Game, Energy-Based Pointing Game, and IoU-based explanation metrics.

The analysis focuses on both detection performance and explanation quality. In addition to the overall XAI comparison, the results are grouped by imaging modality and cancer subtype to examine whether explanation behavior differs between CT, PET/CT, and tumor classes.

# dataprep.py

This script prepares the Lung-PET-CT-Dx dataset for object detection training in YOLO format. It reads DICOM images and their corresponding XML annotations, converts CT slices into PNG images, and exports the bounding boxes as YOLO label files.

## Important Notes

- The train/validation/test split is performed at patient level.
- Only DICOM files with a matching XML annotation are exported.
- DICOM images without matching XML annotations are skipped.
- The script assumes a single detection class and writes `0` as the class ID for every bounding box.
- CT windowing is applied using a lung window with width `1400` and level `-700`.
- Errors during individual DICOM or XML processing are skipped using `try/except`.
- The output is compatible with YOLO-style object detection training pipelines.

# yolo_to_coco.ipynb

This notebook converts the YOLO-formatted Lung-PET-CT-Dx dataset into COCO annotation format. It loads the train, validation, and test splits from `data/processed`, exports COCO JSON annotation files, and performs basic checks and visualizations of the converted bounding boxes.

## Important Notes

- The dataset is expected in YOLO format under `data/processed`.
- The notebook expects `train`, `val`, and `test` folders with separate `images` and `labels` directories.
- A `data.yaml` file is required for loading the YOLO dataset with `supervision`.
- The class name is set to `nodule`.
- COCO annotations are exported to `data/processed/coco`.
- The notebook creates separate COCO files for train, validation, and test splits.
- It prints dataset sizes and checks the generated COCO structure.
- It visualizes random training images with converted bounding boxes.
- It calculates basic bounding-box statistics such as width, height, and area.
- The exported COCO files can be loaded with `torchvision.datasets.CocoDetection`.
- The notebook also shows how to wrap the COCO dataset for torchvision v2 transforms.

# model_training.py

This script trains an RF-DETR Medium object detection model on the processed Lung-PET-CT-Dx dataset. It loads the RF-DETR Medium architecture, checks the NumPy/PyTorch setup and CUDA availability, and starts training using the dataset stored in `data/processed`.

## Important Notes

- The script uses the `RFDETRMedium` model.
- The processed dataset is expected under `data/processed`.
- Training outputs are saved to `outputs/rfdetr_medium_exp2`.
- CUDA is required because training is started with `device="cuda"`.
- Distributed/SLURM environment variables are removed before training to avoid unwanted multi-process behavior.
- Training is configured for a maximum of 200 epochs.
- Early stopping is enabled with a patience of 15 epochs.
- The learning rate is set to `2e-4`.
- The batch size is `32` with `grad_accum_steps=2`.


# model_testing.py

This script evaluates a trained RF-DETR Medium model on the test split of the processed Lung-PET-CT-Dx dataset. It loads the best checkpoint, runs inference on all test images, compares predictions with COCO annotations, and saves detection metrics as a JSON file.

## Important Notes

- The test dataset is expected under `data/processed/test`.
- The script expects COCO annotations at `data/processed/test/_annotations.coco.json`.
- The trained checkpoint is loaded from `outputs/rfdetr_medium_exp2/checkpoint_best_total.pth`.
- The script assumes a single detection class.
- Precision, recall, and F1-score are calculated at IoU thresholds `0.25` and `0.50`.
- COCO-style AP/mAP is calculated for IoU `0.25`, `0.50`, `0.50:0.75`, and `0.50:0.95`.
- Predictions for precision/recall/F1 use a confidence threshold of `0.5`.
- Predictions for AP calculation use a lower threshold of `0.001`.
- True negatives are not defined for standard object detection and are therefore stored as `None`.
- Results are saved to `outputs/rfdetr_medium_exp2/test_metrics_2.json`.

# XAI.py

This script runs the XAI evaluation for a trained RF-DETR Medium model. It generates Grad-CAM, Attention Rollout, and D-RISE heatmaps for all test images and compares them with the ground-truth bounding boxes.

## Important Notes

- The script loads a trained RF-DETR Medium checkpoint.
- The test images are expected under `data/processed/test`.
- COCO annotations are expected at `data/processed/test/_annotations.coco.json`.
- Three XAI methods are evaluated: Grad-CAM, Attention Rollout, and D-RISE.
- The XAI maps are evaluated using Pointing Game, Energy-Based Pointing Game, and IoU after thresholding.
- Pointing Game checks whether the maximum heatmap point lies inside a ground-truth bounding box.
- Energy-Based Pointing Game measures how much heatmap mass lies inside the ground-truth boxes.
- IoU compares a thresholded heatmap mask with the union of the ground-truth boxes.
- D-RISE uses random masking with `1028` masks and a `16x16` grid.
- Results are saved as a JSON file, for example `outputs/results_eval.json`.
- Per-sample metrics are additionally exported as a CSV file.
- CUDA is used automatically if available.


# XAI_performance_testing.ipynb

This notebook analyzes the XAI evaluation results from the JSON output file. It groups the results by XAI method, cancer type, and image modality, and calculates performance summaries for Pointing Game, Energy-Based Pointing Game, and IoU.

## Important Notes

- The notebook expects an XAI evaluation JSON file, for example `outputs/results_eval.json`.
- It analyzes the three XAI methods: Grad-CAM, Attention Rollout, and D-RISE.
- It reads the result blocks `pointing_game`, `mass_in_bounding_box`, and `iou`.
- It also supports the older key `mass_in_box` as a fallback.
- File names are used to infer cancer subgroup labels `A`, `B`, `E`, and `G`.
- File names are also used to infer modality information from `8bit` and `16bit`.
- The notebook calculates grouped mean values for Pointing Game, EBPG, and IoU.
- Results are displayed as tables and exported to `outputs/grouped_metrics.csv`.


# XAI_statistical_testing.ipynb

This notebook performs statistical testing on the per-sample XAI metrics exported by the XAI evaluation script. It compares Grad-CAM, Attention Rollout, and D-RISE using paired statistical tests for Pointing Game, Energy-Based Pointing Game, and IoU.

## Important Notes

- The notebook expects a per-sample metrics CSV file, for example `outputs/results_eval_per_sample_metrics.csv`.
- This input CSV is created by the XAI evaluation script after computing Grad-CAM, Attention Rollout, and D-RISE.
- Required columns are `sample_id`, `image_id`, `file_name`, `method`, `pg`, `ebpg`, and `iou`.
- The notebook adds modality labels based on `8bit` and `16bit` file names.
- The notebook adds cancer type labels based on filename prefixes `A`, `B`, `E`, and `G`.
- Pointing Game is tested with the exact McNemar test because it is binary.
- EBPG and IoU are tested with the Wilcoxon signed-rank test because they are paired numerical metrics.
- Holm-Bonferroni correction is applied to adjust for multiple comparisons.
- The notebook saves the following CSV files to `outputs/`:
  - `xai_per_sample_metrics_with_groups.csv`: per-sample metrics with added modality and cancer-type group labels.
  - `xai_summary_table.csv`: grouped summary table for PG, EBPG, and IoU across methods and groups.
  - `xai_statistical_tests_total.csv`: statistical test results comparing the XAI methods on the full dataset.