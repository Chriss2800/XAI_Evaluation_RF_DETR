import numpy
print(numpy.__version__)
from rfdetr import RFDETRMedium
import torch
print(torch.cuda.is_available())
print(torch.__version__)

model = RFDETRMedium() 
print(type(model))

from pathlib import Path
from rfdetr.datasets.aug_config import AUG_CONSERVATIVE, AUG_AGGRESSIVE, AUG_AERIAL, AUG_INDUSTRIAL

import os

for k in [
    "RANK", "WORLD_SIZE", "LOCAL_RANK",
    "MASTER_ADDR", "MASTER_PORT",
    "SLURM_PROCID", "SLURM_NTASKS", "SLURM_LOCALID"
]:
    os.environ.pop(k, None)

DATASET_RFDETR = Path("./data/processed")   # anpassen
OUTPUT_DIR = Path("./outputs/rfdetr_medium_exp2")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

results = model.train(
    dataset_dir=str(DATASET_RFDETR),
    output_dir=str(OUTPUT_DIR),
    epochs=200,
    batch_size=32,
    grad_accum_steps=2,
    lr=2e-4,
    device="cuda",
    early_stopping=True,
    early_stopping_patience=15,  # Wait 15 epochs before stopping
    early_stopping_min_delta=0.001,  # Require 0.1% mAP improvement
    early_stopping_use_ema=True,  # Track EMA model performance
    print_freq=100,
    progress_bar=True,
)