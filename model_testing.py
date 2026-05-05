import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from rfdetr import RFDETRMedium
from PIL import Image


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def coco_bbox_to_xyxy(bbox):
    x, y, w, h = bbox
    return [float(x), float(y), float(x + w), float(y + h)]


def box_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def greedy_match(preds, gts, iou_thr):
    preds = sorted(preds, key=lambda x: x["score"], reverse=True)
    matched_gt = set()

    tp = 0
    fp = 0

    for pred in preds:
        best_iou = -1.0
        best_gt_idx = -1

        for i, gt in enumerate(gts):
            if i in matched_gt:
                continue

            iou = box_iou_xyxy(pred["bbox_xyxy"], gt["bbox_xyxy"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= iou_thr and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(gts) - len(matched_gt)
    return tp, fp, fn


def precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def run_coco_ap(coco_gt, coco_results, iou_thrs):
    if len(coco_results) == 0:
        return 0.0

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.iouThrs = np.array(iou_thrs, dtype=np.float32)
    coco_eval.params.useCats = 1
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return float(coco_eval.stats[0])


def evaluate_prf1(img_ids, preds_for_prf1, gts_for_prf1, iou_thr):
    total_tp, total_fp, total_fn = 0, 0, 0

    for image_id in img_ids:
        tp, fp, fn = greedy_match(
            preds=preds_for_prf1[image_id],
            gts=gts_for_prf1[image_id],
            iou_thr=iou_thr,
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision, recall, f1 = precision_recall_f1(total_tp, total_fp, total_fn)

    return {
        "iou_threshold": iou_thr,
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
        "tn": None,  # not defined in standard object detection
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def main():
    test_dir = Path("./data/processed/test")
    ann_path = test_dir / "_annotations.coco.json"
    checkpoint = "./outputs/rfdetr_medium_exp2/checkpoint_best_total.pth"
    output_json = "./outputs/rfdetr_medium_exp2/test_metrics_2.json"

    conf_thresh_for_prf1 = 0.5
    predict_threshold_for_ap = 0.001

    coco_gt = COCO(str(ann_path))
    img_ids = coco_gt.getImgIds()
    imgs = coco_gt.loadImgs(img_ids)

    categories = coco_gt.loadCats(coco_gt.getCatIds())
    if len(categories) != 1:
        raise ValueError(f"Expected exactly 1 class, but found {len(categories)} classes.")

    category_id = categories[0]["id"]
    category_name = categories[0]["name"]

    model = RFDETRMedium(pretrain_weights=checkpoint)

    coco_results_all = []
    preds_for_prf1 = defaultdict(list)
    gts_for_prf1 = defaultdict(list)

    for idx, img in enumerate(imgs, start=1):
        image_id = img["id"]
        image_path = test_dir / img["file_name"]

        print(f"[{idx}/{len(imgs)}] Processing {img['file_name']}")

        ann_ids = coco_gt.getAnnIds(imgIds=[image_id], catIds=[category_id])
        anns = coco_gt.loadAnns(ann_ids)

        for ann in anns:
            gts_for_prf1[image_id].append({
                "bbox_xyxy": coco_bbox_to_xyxy(ann["bbox"])
            })
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)
        detections = model.predict(img_np, threshold=predict_threshold_for_ap)

        xyxy = np.asarray(detections.xyxy)
        confidence = np.asarray(detections.confidence)

        for box, score in zip(xyxy, confidence):
            score = float(score)
            box = [float(v) for v in box]

            coco_results_all.append({
                "image_id": int(image_id),
                "category_id": int(category_id),
                "bbox": xyxy_to_xywh(box),
                "score": score,
            })

            if score >= conf_thresh_for_prf1:
                preds_for_prf1[image_id].append({
                    "bbox_xyxy": box,
                    "score": score,
                })

    prf1_025 = evaluate_prf1(img_ids, preds_for_prf1, gts_for_prf1, iou_thr=0.25)
    prf1_050 = evaluate_prf1(img_ids, preds_for_prf1, gts_for_prf1, iou_thr=0.50)

    ap_025 = run_coco_ap(coco_gt, coco_results_all, [0.25])
    ap_050 = run_coco_ap(coco_gt, coco_results_all, [0.50])
    ap_5075 = run_coco_ap(coco_gt, coco_results_all, np.arange(0.50, 0.76, 0.05))
    ap_5095 = run_coco_ap(coco_gt, coco_results_all, np.arange(0.50, 0.96, 0.05))

    payload = {
        "dataset": {
            "test_dir": str(test_dir),
            "annotation_file": str(ann_path),
            "num_images": int(len(imgs)),
            "num_classes": 1,
            "class_id": int(category_id),
            "class_name": category_name,
        },
        "model": {
            "checkpoint": checkpoint,
            "single_class_detection": True,
            "ap_equals_map": True,
        },
        "settings": {
            "confidence_threshold_for_precision_recall_f1": conf_thresh_for_prf1,
            "prediction_threshold_for_ap": predict_threshold_for_ap,
        },
        "metrics": {
            "precision_recall_f1": {
                "iou_0.25": prf1_025,
                "iou_0.50": prf1_050,
            },
            "average_precision": {
                "ap_0.25": float(ap_025),
                "ap_0.50": float(ap_050),
                "ap_0.50_to_0.75": float(ap_5075),
                "ap_0.50_to_0.95": float(ap_5095),
                "map_0.25": float(ap_025),
                "map_0.50": float(ap_050),
                "map_0.50_to_0.75": float(ap_5075),
                "map_0.50_to_0.95": float(ap_5095),
            },
        },
    }

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n===== RESULTS =====")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nSaved JSON to: {output_path}")


if __name__ == "__main__":
    main()