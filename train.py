"""
Training script for Crack Segmentation
====================================================

Experimental protocol implemented:
  * Loss      : torch.nn.BCEWithLogitsLoss with
                pos_weight = (num_negative_pixels / num_positive_pixels) / 4
  * Optimizer : MuonOpti (muon_opti.py), constant learning rate 0.01
  * Precision : BF16 mixed-precision training with gradient scaling
  * Batch size: 16
  * Epochs    : 100 (no early stopping used)
  * Schedule  : constant learning rate (no decay)
  * Selection : after each epoch the F1 score is evaluated on the
                validation (eval) set; the weights yielding the highest
                validation F1 are retained as the optimal model
  * Threshold : the optimal F1 threshold is determined on the validation set
                by sweeping candidate thresholds during inference
  * Seed      : no specific random seed is set
  * Resolution: input images are uniformly resized to a multiple of 32 px

Usage:
    python train.py
    python train.py --data_root "F:\\szu\\crack_seg_models_6\\dataset"
    python train.py --epochs 100 --batch_size 16 --lr 0.01
"""
from __future__ import annotations

import argparse
import os
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from crackseg_sasd import CrackSegSASD
from muon_opti import MuonOpti
from crack_dataset import CrackDataset, compute_pos_weight

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #
def _counts(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float, float]:
    """Return (tp, fp, fn) for a binary prediction vs. binary target."""
    tgt = target > 0.5
    tp = float((pred & tgt).sum().item())
    fp = float((pred & ~tgt).sum().item())
    fn = float((~pred & tgt).sum().item())
    return tp, fp, fn


def _f1(tp: float, fp: float, fn: float) -> float:
    denom = 2.0 * tp + fp + fn
    return (2.0 * tp / denom) if denom > 0.0 else 0.0


@torch.no_grad()
def evaluate_sweep(
    model: nn.Module, loader: DataLoader, device: torch.device,
    thresholds: List[float], autocast: torch.amp.autocast,
) -> Tuple[float, float]:
    """Sweep candidate thresholds on a split and return (best_f1, best_thr).

    Confusion counts are accumulated in float (not stored as full tensors) so
    the validation/inference step stays memory-lean.
    """
    model.eval()
    tp_t = {thr: 0.0 for thr in thresholds}
    fp_t = {thr: 0.0 for thr in thresholds}
    fn_t = {thr: 0.0 for thr in thresholds}

    for img, target in loader:
        img = img.to(device)
        target = target.to(device)
        with autocast:
            logits = model(img)
        probs = torch.sigmoid(logits.float())
        bin_tgt = target > 0.5
        for thr in thresholds:
            pred = probs >= thr
            tp_t[thr] += float((pred & bin_tgt).sum().item())
            fp_t[thr] += float((pred & ~bin_tgt).sum().item())
            fn_t[thr] += float((~pred & bin_tgt).sum().item())

    best_f1 = -1.0
    best_thr = thresholds[0]
    for thr in thresholds:
        f1 = _f1(tp_t[thr], fp_t[thr], fn_t[thr])
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_f1, best_thr


@torch.no_grad()
def evaluate_at(
    model: nn.Module, loader: DataLoader, device: torch.device,
    threshold: float, autocast: torch.amp.autocast,
) -> float:
    """Return pixel-level F1 for a fixed threshold on a split."""
    model.eval()
    tp = fp = fn = 0.0
    for img, target in loader:
        img = img.to(device)
        target = target.to(device)
        with autocast:
            logits = model(img)
        probs = torch.sigmoid(logits.float())
        a, b, c = _counts(probs >= threshold, target)
        tp += a; fp += b; fn += c
    return _f1(tp, fp, fn)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Train SASD-CrackSeg for crack segmentation.")
    parser.add_argument("--data_root", default=os.path.join(_PROJECT_DIR, "dataset"),
                        help="Root of the dataset directory (train/test/eval).")
    parser.add_argument("--img_size", type=int, default=512,
                        help="Square size to resize images to (rounds up to a multiple of 32).")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Constant learning rate for MuonOpti.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="cuda / cpu (auto-detected).")
    parser.add_argument("--save_path", default=os.path.join(_PROJECT_DIR, "best_model.pth"),
                        help="Where to save the best (highest val F1) weights.")
    parser.add_argument("--thresholds", default="0.05,0.95,0.05",
                        help="Threshold sweep as start,end,step (e.g. 0.05,0.95,0.05).")
    parser.add_argument("--log", default=os.path.join(_PROJECT_DIR, "train_log.csv"),
                        help="CSV log of per-epoch training loss / val F1.")
    args = parser.parse_args()

    # --- device ----------------------------------------------------------- #
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    # --- threshold sweep -------------------------------------------------- #
    start, end, step = (float(x) for x in args.thresholds.split(","))
    thresholds: List[float] = [round(v, 4) for v in np.arange(start, end + 1e-9, step)]
    if not thresholds or thresholds[-1] < end:
        thresholds.append(round(end, 4))
    thresholds = sorted(set(thresholds))

    # --- datasets --------------------------------------------------------- #
    data_root = args.data_root
    train_ds = CrackDataset(os.path.join(data_root, "train", "image"),
                            os.path.join(data_root, "train", "target"),
                            img_size=args.img_size, train=True, augment=True)
    val_ds = CrackDataset(os.path.join(data_root, "eval", "image"),
                          os.path.join(data_root, "eval", "target"),
                          img_size=args.img_size, train=False, augment=False)
    test_ds = CrackDataset(os.path.join(data_root, "test", "image"),
                           os.path.join(data_root, "test", "target"),
                           img_size=args.img_size, train=False, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True,
                              pin_memory=use_cuda)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=use_cuda)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=use_cuda)

    # --- pos_weight (neg/pos/4 over the training target masks) ------------ #
    pos_weight_float = compute_pos_weight(os.path.join(data_root, "train", "target"))
    pos_weight = torch.tensor([pos_weight_float], dtype=torch.float32)

    # --- model / loss / optimizer ----------------------------------------- #
    model = CrackSegSASD(input_channel=3, output_channel=1).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    optimizer = MuonOpti(model.parameters(), lr=args.lr)

    # --- BF16 mixed precision + gradient scaling --------------------------- #
    amp_dtype = torch.bfloat16
    autocast = torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_cuda)
    scaler = torch.amp.GradScaler(device.type, enabled=use_cuda)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[data] train={len(train_ds)}  eval={len(val_ds)}  test={len(test_ds)}")
    print(f"[data] pos_weight (neg/pos/4) = {pos_weight_float:.4f}")
    print(f"[model] SASD-CrackSeg params = {n_params:,}")
    print(f"[config] device={device}  bf16_amp={use_cuda}  batch={args.batch_size}  "
          f"epochs={args.epochs}  lr={args.lr}  img_size={args.img_size}")
    print(f"[thresholds] {thresholds}")

    best_f1 = -1.0
    best_thr = 0.5
    with open(args.log, "w") as log_fp:
        log_fp.write("epoch,time_s,train_loss,val_f1,best_f1,threshold\n")
        log_fp.flush()

        for epoch in range(args.epochs):
            model.train()
            t0 = time.time()
            running_loss = 0.0
            n_seen = 0
            for img, target in train_loader:
                img = img.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                with autocast:
                    logits = model(img)
                    loss = criterion(logits, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item() * img.size(0)
                n_seen += img.size(0)

            avg_loss = running_loss / max(n_seen, 1)
            val_f1, val_thr = evaluate_sweep(model, val_loader, device, thresholds, autocast)

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_thr = val_thr
                torch.save(model.state_dict(), args.save_path)
                saved = True
            else:
                saved = False

            elapsed = time.time() - t0
            print(f"[epoch {epoch + 1:3d}/{args.epochs}] loss={avg_loss:.4f}  "
                  f"val_f1={val_f1:.4f}  threshold={val_thr:.3f}  "
                  f"best_f1={best_f1:.4f}  {str(saved):5s}  {elapsed:.1f}s")
            log_fp.write(f"{epoch + 1},{elapsed:.2f},{avg_loss:.6f},"
                         f"{val_f1:.6f},{best_f1:.6f},{val_thr:.4f}\n")
            log_fp.flush()

    # --- final test evaluation using the retained optimal model ------------ #
    print("\n[final] restoring the best (highest val F1) weights ...")
    model.load_state_dict(torch.load(args.save_path, map_location=device))
    test_f1 = evaluate_at(model, test_loader, device, best_thr, autocast)
    print(f"[final] best validation F1             = {best_f1:.4f}  (threshold = {best_thr:.3f})")
    print(f"[final] test F1 at validation thr      = {test_f1:.4f}  (threshold = {best_thr:.3f})")


if __name__ == "__main__":
    main()
