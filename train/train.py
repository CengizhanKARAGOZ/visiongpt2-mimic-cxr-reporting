"""
Vision-GPT2: Chest X-Ray Report Generation Training Script
==========================================================
Bu script, göğüs röntgeni görüntülerinden otomatik radyoloji raporu
üretmek için Vision-GPT2 modelini eğitir.

Kullanım (Kaggle/Colab):
    python train.py

Gereksinimler:
    pip install torch torchvision timm transformers tqdm pandas pillow
"""

import os
import re
import ast
import math
import time
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import torchvision.transforms as T
import timm
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:

    # Paths (Kaggle için varsayılan)
    ROOT_DATASET = "/kaggle/input/mimic-cxr"
    ROOT = os.path.join(ROOT_DATASET, "official_data_iccv_final")
    FILES_ROOT = os.path.join(ROOT, "files")

    TRAIN_CSV = os.path.join(ROOT_DATASET, "mimic_cxr_aug_train.csv")
    VAL_CSV = os.path.join(ROOT_DATASET, "mimic_cxr_aug_validate.csv")

    OUTPUT_DIR = "./outputs"

    # Model
    IMG_SIZE = 384
    MAX_LEN = 256
    CNN_BACKBONE = "densenet121"

    # Training
    BATCH_SIZE = 8
    GRAD_ACCUM = 4
    NUM_WORKERS = 2
    EPOCHS = 4

    # Optimizer
    LR = 8e-5
    WEIGHT_DECAY = 0.01
    LABEL_SMOOTHING = 0.1

    # Data
    USE_AUG_TEXT = True
    TEXT_COL = "text_augment" if USE_AUG_TEXT else "text"

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed
    SEED = 42


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def parse_list(x):

    try:
        v = ast.literal_eval(x)
        if isinstance(v, list):
            return v
    except:
        pass
    return []


def parse_text(x):

    try:
        v = ast.literal_eval(x)
        if isinstance(v, list) and len(v) > 0:
            return str(v[0])
    except:
        pass
    return str(x)


def clean_text(t):

    t = str(t).strip()
    t = re.sub(r"\s+", " ", t)
    return t


def pick_best_rel(row):

    for col in ["PA", "AP", "Lateral", "image"]:
        if col in row and pd.notna(row[col]):
            lst = parse_list(row[col])
            if len(lst) > 0:
                return lst[0]
    return None


def prepare_dataframes(config):

    print("Loading CSVs...")
    train_df = pd.read_csv(config.TRAIN_CSV)
    val_df = pd.read_csv(config.VAL_CSV)

    train_df["rel_path"] = train_df.apply(pick_best_rel, axis=1)
    val_df["rel_path"] = val_df.apply(pick_best_rel, axis=1)

    train_df["image_path"] = train_df["rel_path"].apply(
        lambda p: os.path.join(config.ROOT, p) if isinstance(p, str) else None
    )
    val_df["image_path"] = val_df["rel_path"].apply(
        lambda p: os.path.join(config.ROOT, p) if isinstance(p, str) else None
    )

    # Filter missing
    train_df = train_df[train_df["image_path"].notnull()].copy()
    val_df = val_df[val_df["image_path"].notnull()].copy()

    train_df = train_df[train_df["image_path"].apply(os.path.exists)].copy()
    val_df = val_df[val_df["image_path"].apply(os.path.exists)].copy()

    # Text
    train_df["report_text"] = train_df[config.TEXT_COL].apply(
        lambda x: clean_text(parse_text(x))
    )
    val_df["report_text"] = val_df[config.TEXT_COL].apply(
        lambda x: clean_text(parse_text(x))
    )

    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")

    return train_df, val_df


# ============================================================================
# DATASET
# ============================================================================

class MIMICReportDataset(Dataset):

    def __init__(self, df, transform, tokenizer, max_len):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Image
        img = Image.open(row["image_path"]).convert("RGB")
        img = self.transform(img)

        # Text
        enc = self.tok(
            row["report_text"],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attn_mask = enc["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        labels[attn_mask == 0] = -100

        return img, input_ids, attn_mask, labels


def get_transforms(img_size, is_train=True):

    if is_train:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(5),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


# ============================================================================
# MODEL
# ============================================================================

class CNNEncoder(nn.Module):

    def __init__(self, name="densenet121", d_model=768):
        super().__init__()
        self.backbone = timm.create_model(
            name,
            pretrained=True,
            num_classes=0,
            global_pool=""
        )
        c = self.backbone.feature_info[-1]["num_chs"]
        self.proj = nn.Conv2d(c, d_model, kernel_size=1)

    def forward(self, x):
        feat = self.backbone(x)  # [B, C, H, W]
        feat = self.proj(feat)  # [B, d_model, H, W]
        B, D, H, W = feat.shape
        return feat.flatten(2).transpose(1, 2).contiguous()  # [B, H*W, d_model]


class VisionGPT2(nn.Module):

    def __init__(self, cnn_name="densenet121"):
        super().__init__()

        cfg = GPT2Config.from_pretrained("gpt2")
        cfg.add_cross_attention = True

        self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2", config=cfg)
        self.enc = CNNEncoder(name=cnn_name, d_model=cfg.n_embd)

    def forward(self, images, input_ids, attention_mask, labels=None):

        enc_states = self.enc(images)  # [B, N, d_model]

        # GPT-2 forward with cross-attention
        outputs = self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=enc_states,
            labels=labels,
        )

        return outputs


# ============================================================================
# TRAINING
# ============================================================================

class LabelSmoothingLoss(nn.Module):

    def __init__(self, smoothing=0.1, ignore_index=-100):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        # logits: [B, seq_len, vocab_size]
        # labels: [B, seq_len]

        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)

        mask = labels != self.ignore_index
        labels_masked = labels[mask]
        logits_masked = logits[mask]

        if logits_masked.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        log_probs = torch.log_softmax(logits_masked, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=labels_masked.unsqueeze(-1)).squeeze(-1)
        smooth_loss = -log_probs.mean(dim=-1)

        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, scheduler, scaler, criterion, config, epoch):

    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{config.EPOCHS}")
    optimizer.zero_grad()

    for step, (images, input_ids, attn_mask, labels) in enumerate(pbar):
        images = images.to(config.DEVICE)
        input_ids = input_ids.to(config.DEVICE)
        attn_mask = attn_mask.to(config.DEVICE)
        labels = labels.to(config.DEVICE)

        with autocast():
            outputs = model(images, input_ids, attn_mask, labels)
            logits = outputs.logits

            # Shift for causal LM
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = criterion(shift_logits, shift_labels)
            loss = loss / config.GRAD_ACCUM

        scaler.scale(loss).backward()

        if (step + 1) % config.GRAD_ACCUM == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * config.GRAD_ACCUM
        pbar.set_postfix({"loss": f"{loss.item() * config.GRAD_ACCUM:.4f}"})

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, config):

    model.eval()
    total_loss = 0

    for images, input_ids, attn_mask, labels in tqdm(loader, desc="Validating"):
        images = images.to(config.DEVICE)
        input_ids = input_ids.to(config.DEVICE)
        attn_mask = attn_mask.to(config.DEVICE)
        labels = labels.to(config.DEVICE)

        with autocast():
            outputs = model(images, input_ids, attn_mask, labels)
            logits = outputs.logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = criterion(shift_logits, shift_labels)

        total_loss += loss.item()

    return total_loss / len(loader)


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, path):

    model_to_save = model.module if hasattr(model, 'module') else model

    torch.save({
        "epoch": epoch,
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "val_loss": val_loss,
    }, path)
    print(f"Checkpoint saved: {path}")


# ============================================================================
# MAIN
# ============================================================================

def main():

    config = Config()
    set_seed(config.SEED)

    print("=" * 60)
    print("Vision-GPT2 Training")
    print("=" * 60)
    print(f"Device: {config.DEVICE}")
    print(f"GPUs: {torch.cuda.device_count()}")
    print(f"Image size: {config.IMG_SIZE}")
    print(f"Batch size: {config.BATCH_SIZE} x {config.GRAD_ACCUM} (grad accum)")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Learning rate: {config.LR}")
    print("=" * 60)

    # Output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Data
    train_df, val_df = prepare_dataframes(config)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_tfm = get_transforms(config.IMG_SIZE, is_train=True)
    val_tfm = get_transforms(config.IMG_SIZE, is_train=False)

    train_ds = MIMICReportDataset(train_df, train_tfm, tokenizer, config.MAX_LEN)
    val_ds = MIMICReportDataset(val_df, val_tfm, tokenizer, config.MAX_LEN)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # Model
    print("Loading model...")
    model = VisionGPT2(cnn_name=config.CNN_BACKBONE)

    # Multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model.to(config.DEVICE)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY,
    )

    total_steps = len(train_loader) * config.EPOCHS // config.GRAD_ACCUM
    warmup_steps = total_steps // 10

    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Loss & Scaler
    criterion = LabelSmoothingLoss(smoothing=config.LABEL_SMOOTHING)
    scaler = GradScaler()

    # Training loop
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    print("\nStarting training...")
    for epoch in range(config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, criterion, config, epoch
        )
        val_loss = validate(model, val_loader, criterion, config)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                os.path.join(config.OUTPUT_DIR, "best_vgpt2.pt")
            )

        # Save latest
        save_checkpoint(
            model, optimizer, scheduler, epoch, val_loss,
            os.path.join(config.OUTPUT_DIR, f"checkpoint_epoch{epoch + 1}.pt")
        )

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("=" * 60)

    # Save history
    pd.DataFrame(history).to_csv(
        os.path.join(config.OUTPUT_DIR, "training_history.csv"),
        index=False
    )

    return history


if __name__ == "__main__":
    main()