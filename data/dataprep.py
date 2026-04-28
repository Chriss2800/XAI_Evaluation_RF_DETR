import os
import glob
import pydicom
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import random
import os


SCRIPT_PATH = os.path.abspath(__file__)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)


BASE_DIR = os.path.dirname(SCRIPT_DIR)


DICOM_ROOT = os.path.join(BASE_DIR, "data", "raw", "Lung-PET-CT-Dx")
XML_ROOT = os.path.join(BASE_DIR, "data", "Annotation")
EXPORT_BASE = os.path.join(BASE_DIR, "data", "processed")


print(f"Projekt-Hauptpfad: {BASE_DIR}")
print(f"Suche DICOMs in:  {DICOM_ROOT}")
print(f"Suche XMLs in:    {XML_ROOT}")

if not os.path.exists(XML_ROOT):
    print(f"❌ FEHLER: Der Pfad wurde nicht gefunden!")
    print(f"Bitte prüfe, ob der Ordner existiert: {XML_ROOT}")



TRAIN_RATIO = 0.70
VAL_RATIO = 0.15



for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(EXPORT_BASE, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(EXPORT_BASE, split, 'labels'), exist_ok=True)

def apply_windowing(dicom_img, window_width=1400, window_level=-700):
    img = dicom_img.pixel_array.astype(np.float32)
    if 'RescaleIntercept' in dicom_img and 'RescaleSlope' in dicom_img:
        img = img * dicom_img.RescaleSlope + dicom_img.RescaleIntercept
    img_min = window_level - window_width // 2
    img_max = window_level + window_width // 2
    windowed = np.clip(img, img_min, img_max)
    return ((windowed - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)

def convert_to_yolo(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    return [x * dw, y * dh, w * dw, h * dh]



all_patients = sorted([d for d in os.listdir(XML_ROOT) if os.path.isdir(os.path.join(XML_ROOT, d))])
random.seed(42)
random.shuffle(all_patients)


num_patients = len(all_patients)
train_end = int(num_patients * TRAIN_RATIO)
val_end = train_end + int(num_patients * VAL_RATIO)

train_patients = all_patients[:train_end]
val_patients = all_patients[train_end:val_end]
test_patients = all_patients[val_end:]

print(f"Gesamt Patienten: {num_patients}")
print(f"Training: {len(train_patients)} | Validierung: {len(val_patients)} | Test: {len(test_patients)}")

def run_full_export(patient_list, split_name):
    img_count = 0
    for p_id in patient_list:
        p_xml_dir = os.path.join(XML_ROOT, p_id)
        p_dicom_search = glob.glob(os.path.join(DICOM_ROOT, f"*{p_id}*"))

        if not p_dicom_search:
            continue

        p_dicom_dir = p_dicom_search[0]
        all_dicoms = glob.glob(os.path.join(p_dicom_dir, "**", "*.dcm"), recursive=True)
        xml_list = [f.replace(".xml", "") for f in os.listdir(p_xml_dir) if f.endswith(".xml")]

        for d_path in all_dicoms:
            try:
                ds_meta = pydicom.dcmread(d_path, stop_before_pixels=True)
                uid = ds_meta.SOPInstanceUID

                if uid in xml_list:
                    ds = pydicom.dcmread(d_path)
                    img = apply_windowing(ds)
                    h, w = img.shape

                    file_base = f"{p_id}_{uid}"
                    img_out = os.path.join(EXPORT_BASE, split_name, 'images', f"{file_base}.png")
                    lbl_out = os.path.join(EXPORT_BASE, split_name, 'labels', f"{file_base}.txt")

                    cv2.imwrite(img_out, img)

                    tree = ET.parse(os.path.join(p_xml_dir, f"{uid}.xml"))
                    yolo_lines = []
                    for obj in tree.getroot().findall('object'):
                        b = obj.find('bndbox')
                        box = [float(b.find('xmin').text), float(b.find('ymin').text),
                               float(b.find('xmax').text), float(b.find('ymax').text)]
                        y_box = convert_to_yolo((w, h), box)
                        yolo_lines.append(f"0 {y_box[0]:.6f} {y_box[1]:.6f} {y_box[2]:.6f} {y_box[3]:.6f}")

                    with open(lbl_out, "w") as f:
                        f.write("\n".join(yolo_lines))
                    img_count += 1
            except:
                continue
        print(f"[{split_name}] Patient {p_id} exportiert.")
    return img_count


total_train = run_full_export(train_patients, 'train')
total_val = run_full_export(val_patients, 'val')
total_test = run_full_export(test_patients, 'test')

print(f"\nEXPORT FERTIG!")
print(f"Training: {total_train} | Validierung: {total_val} | Test: {total_test}")