#!/usr/bin/env python3
"""
LEAVE-ONE-BEARING-OUT CROSS-VALIDATION
=======================================
Validates AION NEXUS v6 F1 without data leakage.

The original F1=93.43% uses StratifiedShuffleSplit which mixes samples
from the same bearing across train/test. Since consecutive vibration
recordings from the same bearing are highly autocorrelated, this inflates
reported metrics.

This script implements proper Leave-One-Bearing-Out (LOBO) CV:
- 6 bearings = 6 folds
- Each fold: 1 bearing = test, remaining 5 = train (80%) + val (20%)
- Val split: last bearing temporally from the 5 training bearings
- Full 3-stage progressive training per fold (identical to original)
- Reports per-fold and mean +/- std F1

FEMTO PHM 2012 Bearings:
  Condition 1 (1800 RPM): Bearing1_1 (2803 files), Bearing1_2 (871 files)
  Condition 2 (1650 RPM): Bearing2_1 (911 files), Bearing2_2 (797 files)
  Condition 3 (1500 RPM): Bearing3_1 (515 files), Bearing3_2 (1637 files)
  Total: 7534 samples across 6 bearings

Author: Claude Code (LOBO validation audit)
Date: 2026-02-11
"""

import sys
import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import time
import json
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from aion_nexus_v6 import create_aion_nexus_v6
from aion_advanced_loss import AdvancedAIONLoss
from robust_augmentation import RobustAugmentationPipeline


# ============================================================================
# FEMTO BEARING METADATA
# ============================================================================

FEMTO_GEOMETRY = {
    'N': 8,        # Number of rolling elements
    'd': 7.92,     # Ball diameter (mm)
    'D': 38.5,     # Pitch diameter (mm)
    'phi': 0.0,    # Contact angle (degrees)
}

BEARING_CONDITIONS = {
    'Bearing1_1': {'rpm': 1800, 'load_N': 4000, 'condition': 1},
    'Bearing1_2': {'rpm': 1800, 'load_N': 4000, 'condition': 1},
    'Bearing2_1': {'rpm': 1650, 'load_N': 4200, 'condition': 2},
    'Bearing2_2': {'rpm': 1650, 'load_N': 4200, 'condition': 2},
    'Bearing3_1': {'rpm': 1500, 'load_N': 5000, 'condition': 3},
    'Bearing3_2': {'rpm': 1500, 'load_N': 5000, 'condition': 3},
}


# ============================================================================
# DATA LOADING (self-contained, no metadata JSON dependency)
# ============================================================================

def load_bearing_data(bearing_dir: Path, bearing_name: str) -> List[dict]:
    """Load all samples from a single bearing directory.

    Returns list of dicts with keys: signal, label, bearing_id, file_index, degradation_pct
    """
    csv_files = sorted(bearing_dir.glob("acc_*.csv"))
    if not csv_files:
        logging.warning(f"No acc_*.csv files in {bearing_name}")
        return []

    total_files = len(csv_files)
    samples = []

    for file_idx, csv_path in enumerate(csv_files):
        try:
            data = pd.read_csv(csv_path, header=None)

            if len(data.columns) >= 6:
                horizontal = data.iloc[:, 4].values.astype(np.float32)
                vertical = data.iloc[:, 5].values.astype(np.float32)
            elif len(data.columns) >= 2:
                horizontal = data.iloc[:, 0].values.astype(np.float32)
                vertical = data.iloc[:, 1].values.astype(np.float32)
            else:
                continue

            expected_length = 2560
            if len(horizontal) < 500 or len(horizontal) > 10000:
                continue
            if np.isnan(horizontal).sum() > len(horizontal) * 0.1:
                continue

            # Resize to expected length
            if len(horizontal) != expected_length:
                if len(horizontal) > expected_length:
                    horizontal = horizontal[:expected_length]
                    vertical = vertical[:expected_length]
                else:
                    pad_length = expected_length - len(horizontal)
                    horizontal = np.pad(horizontal, (0, pad_length), mode='constant')
                    vertical = np.pad(vertical, (0, pad_length), mode='constant')

            # Normalize each channel
            horizontal = (horizontal - np.mean(horizontal)) / (np.std(horizontal) + 1e-8)
            vertical = (vertical - np.mean(vertical)) / (np.std(vertical) + 1e-8)

            # High-pass filter (1 Hz cutoff)
            try:
                from scipy import signal as scipy_signal
                sos = scipy_signal.butter(2, 1.0, 'highpass', fs=25600, output='sos')
                horizontal = scipy_signal.sosfilt(sos, horizontal).astype(np.float32)
                vertical = scipy_signal.sosfilt(sos, vertical).astype(np.float32)
            except ImportError:
                horizontal = (horizontal - np.mean(horizontal)).astype(np.float32)
                vertical = (vertical - np.mean(vertical)).astype(np.float32)

            signal = np.stack([horizontal, vertical], axis=0)  # [2, 2560]

            # Degradation label (proxy based on temporal position)
            degradation_pct = file_idx / (total_files - 1) if total_files > 1 else 0.0
            if degradation_pct < 0.2:
                label = 0  # Normal
            elif degradation_pct < 0.5:
                label = 1  # Early degradation
            elif degradation_pct < 0.8:
                label = 2  # Medium degradation
            else:
                label = 3  # Late/Failure

            rpm = BEARING_CONDITIONS[bearing_name]['rpm']

            samples.append({
                'signal': signal,
                'label': label,
                'bearing_id': bearing_name,
                'file_index': file_idx,
                'degradation_pct': degradation_pct,
                'rpm': rpm,
            })

        except Exception as e:
            continue

    return samples


class BearingDataset(Dataset):
    """Dataset from pre-loaded bearing samples."""

    def __init__(self, samples: List[dict]):
        self.samples = samples
        self.labels = [s['label'] for s in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        signal = torch.FloatTensor(s['signal'])
        label = torch.LongTensor([s['label']])
        metadata = {
            'bearing_id': s['bearing_id'],
            'file_index': s['file_index'],
            'degradation_percentage': s['degradation_pct'],
            'rpm': s['rpm'],
            'bearing_geometry': FEMTO_GEOMETRY,
            'temporal_index': idx,
        }
        return signal, label, metadata


def bearing_collate_fn(batch):
    signals, labels, metadata_list = zip(*batch)
    signals_batch = torch.stack(signals)
    labels_batch = torch.stack(labels).squeeze(1)
    rpm_values = [meta['rpm'] for meta in metadata_list]
    rpm_batch = torch.FloatTensor(rpm_values)
    return signals_batch, labels_batch, list(metadata_list), rpm_batch


# ============================================================================
# TRAINER (streamlined from train_nexus_ultra_v6.py)
# ============================================================================

class LOBOTrainer:
    """Train AION NEXUS v6 for one LOBO fold."""

    def __init__(self, train_loader, val_loader, device='cpu'):
        self.device = device

        # Create fresh model
        self.model = create_aion_nexus_v6(num_classes=4).to(device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        # Loss
        self.loss_fn = AdvancedAIONLoss(
            num_classes=4, focal_gamma=2.0, use_focal=True,
            use_temporal=False, use_margin=True, auto_balance_focal=True
        )
        train_labels = train_loader.dataset.labels
        train_class_counts = Counter(train_labels)
        if hasattr(self.loss_fn, 'focal_loss'):
            self.loss_fn.focal_loss.set_class_weights(train_class_counts)

        self.augmenter = None
        self.best_val_f1 = 0.0
        self.best_state = None

    def train_epoch(self, optimizer, augmentation_prob=0.0):
        self.model.train()
        total_loss = 0.0
        all_preds, all_targets = [], []

        for signals, labels, metadata, rpm_values in self.train_loader:
            signals = signals.to(self.device)
            labels = labels.to(self.device)

            if augmentation_prob > 0 and self.augmenter is not None:
                signals = self.augmenter(signals)

            optimizer.zero_grad()
            outputs = self.model(signals, N_supervision=1)
            loss_dict = self.loss_fn(outputs, labels, metadata)
            loss = loss_dict['total_loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            with torch.no_grad():
                preds = torch.argmax(outputs['logits'], dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(labels.cpu().numpy())

        avg_loss = total_loss / max(len(self.train_loader), 1)
        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        return avg_loss, f1

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        all_preds, all_targets = [], []

        for signals, labels, metadata, rpm_values in self.val_loader:
            signals = signals.to(self.device)
            outputs = self.model(signals, N_supervision=1)
            preds = torch.argmax(outputs['logits'], dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())

        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        acc = accuracy_score(all_targets, all_preds)
        return f1, acc

    def train_stage(self, num_epochs, lr, freeze_temporal=False, aug_prob=0.0):
        if aug_prob > 0:
            self.augmenter = RobustAugmentationPipeline(p_apply=aug_prob, min_augs=2, max_augs=4)
        else:
            self.augmenter = None

        if freeze_temporal:
            for p in self.model.temporal_attention.parameters():
                p.requires_grad = False
            for p in self.model.recursive_reasoner.parameters():
                p.requires_grad = False
        else:
            for p in self.model.temporal_attention.parameters():
                p.requires_grad = True
            for p in self.model.recursive_reasoner.parameters():
                p.requires_grad = True

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr, weight_decay=1e-4
        )
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=lr * 0.01)

        for epoch in range(num_epochs):
            train_loss, train_f1 = self.train_epoch(optimizer, aug_prob)
            val_f1, val_acc = self.validate()
            scheduler.step()

            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            logging.info(
                f"    Epoch {epoch+1:2d}: train_f1={train_f1:.4f}, val_f1={val_f1:.4f}, val_acc={val_acc:.4f}"
            )

    def train_progressive(self):
        """3-stage progressive training (identical to original protocol)."""
        logging.info("  Stage 1: Warm-up (5 epochs, freeze TempAttn+TRM, no aug)")
        self.train_stage(num_epochs=5, lr=1e-3, freeze_temporal=True, aug_prob=0.0)

        logging.info("  Stage 2: Fine-tune (20 epochs, heavy aug p=0.8)")
        self.train_stage(num_epochs=20, lr=5e-4, freeze_temporal=False, aug_prob=0.8)

        logging.info("  Stage 3: Polish (15 epochs, moderate aug p=0.6)")
        self.train_stage(num_epochs=15, lr=1e-4, freeze_temporal=False, aug_prob=0.6)

    @torch.no_grad()
    def evaluate(self, test_loader):
        """Evaluate on held-out bearing using best model."""
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        self.model.eval()
        self.model.to(self.device)

        all_preds, all_targets = [], []

        for signals, labels, metadata, rpm_values in test_loader:
            signals = signals.to(self.device)
            outputs = self.model(signals, N_supervision=1)
            preds = torch.argmax(outputs['logits'], dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())

        f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        f1_weighted = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        acc = accuracy_score(all_targets, all_preds)
        cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2, 3])
        report = classification_report(all_targets, all_preds, digits=4, zero_division=0)

        return {
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'accuracy': acc,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'predictions': all_preds,
            'targets': all_targets,
        }


# ============================================================================
# MAIN: LEAVE-ONE-BEARING-OUT CV
# ============================================================================

def main():
    output_dir = Path('lobo_cv_results')
    output_dir.mkdir(exist_ok=True, parents=True)

    # Setup logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / 'lobo_cv.log', encoding='utf-8')
        ]
    )

    logging.info("=" * 80)
    logging.info("LEAVE-ONE-BEARING-OUT CROSS-VALIDATION")
    logging.info("Validating AION NEXUS v6 F1 without data leakage")
    logging.info("=" * 80)

    # Data path
    data_root = Path(__file__).parent / "FEMTO+Bearing" / "10. FEMTO Bearing" / "FEMTOBearingDataSet" / "Test_set" / "Training_set" / "Learning_set"

    if not data_root.exists():
        # Fallback: try original path
        data_root = Path(__file__).parent / "data" / "FEMTOBearingDataSet" / "Test_set" / "Training_set" / "Learning_set"

    if not data_root.exists():
        logging.error(f"Data not found at {data_root}")
        logging.error("Please set the correct path in the script.")
        return

    logging.info(f"Data root: {data_root}")

    # Load all bearings
    bearing_names = sorted([d.name for d in data_root.iterdir()
                           if d.is_dir() and 'Bearing' in d.name])

    logging.info(f"Found {len(bearing_names)} bearings: {bearing_names}")

    all_bearing_data = {}
    for bname in bearing_names:
        bdir = data_root / bname
        samples = load_bearing_data(bdir, bname)
        all_bearing_data[bname] = samples
        label_dist = Counter([s['label'] for s in samples])
        logging.info(f"  {bname}: {len(samples)} samples, labels={dict(sorted(label_dist.items()))}")

    total_samples = sum(len(v) for v in all_bearing_data.values())
    logging.info(f"Total samples: {total_samples}")

    # Run LOBO CV
    device = 'cpu'
    batch_size = 16
    fold_results = []

    logging.info("\n" + "=" * 80)
    logging.info("STARTING LOBO CV (6 folds)")
    logging.info("=" * 80)

    start_time = time.time()

    for fold_idx, test_bearing in enumerate(bearing_names):
        fold_start = time.time()
        logging.info(f"\n{'='*80}")
        logging.info(f"FOLD {fold_idx+1}/6: Test bearing = {test_bearing}")
        logging.info(f"{'='*80}")

        # Test set: held-out bearing
        test_samples = all_bearing_data[test_bearing]

        # Train+Val: remaining bearings
        train_bearings = [b for b in bearing_names if b != test_bearing]
        logging.info(f"  Train bearings: {train_bearings}")

        # Use last bearing (by name) as validation, rest for training
        val_bearing = train_bearings[-1]
        pure_train_bearings = train_bearings[:-1]

        train_samples = []
        for b in pure_train_bearings:
            train_samples.extend(all_bearing_data[b])

        val_samples = all_bearing_data[val_bearing]

        logging.info(f"  Train: {len(train_samples)} samples from {pure_train_bearings}")
        logging.info(f"  Val:   {len(val_samples)} samples from [{val_bearing}]")
        logging.info(f"  Test:  {len(test_samples)} samples from [{test_bearing}]")

        # Check class distribution
        train_dist = Counter([s['label'] for s in train_samples])
        test_dist = Counter([s['label'] for s in test_samples])
        logging.info(f"  Train class dist: {dict(sorted(train_dist.items()))}")
        logging.info(f"  Test class dist:  {dict(sorted(test_dist.items()))}")

        # Create data loaders
        train_dataset = BearingDataset(train_samples)
        val_dataset = BearingDataset(val_samples)
        test_dataset = BearingDataset(test_samples)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=bearing_collate_fn, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=bearing_collate_fn, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 collate_fn=bearing_collate_fn, drop_last=False)

        # Train
        trainer = LOBOTrainer(train_loader, val_loader, device=device)
        trainer.train_progressive()

        # Evaluate on held-out bearing
        results = trainer.evaluate(test_loader)

        fold_time = time.time() - fold_start

        logging.info(f"\n  FOLD {fold_idx+1} RESULTS ({test_bearing}):")
        logging.info(f"    F1-macro:    {results['f1_macro']:.4f}")
        logging.info(f"    F1-weighted: {results['f1_weighted']:.4f}")
        logging.info(f"    Accuracy:    {results['accuracy']:.4f}")
        logging.info(f"    Time:        {fold_time/60:.1f} min")
        logging.info(f"\n{results['classification_report']}")

        fold_results.append({
            'fold': fold_idx + 1,
            'test_bearing': test_bearing,
            'train_bearings': pure_train_bearings,
            'val_bearing': val_bearing,
            'n_train': len(train_samples),
            'n_val': len(val_samples),
            'n_test': len(test_samples),
            'f1_macro': results['f1_macro'],
            'f1_weighted': results['f1_weighted'],
            'accuracy': results['accuracy'],
            'confusion_matrix': results['confusion_matrix'],
            'best_val_f1': trainer.best_val_f1,
            'fold_time_sec': fold_time,
        })

        # Save intermediate results after each fold
        with open(output_dir / 'lobo_cv_results.json', 'w') as f:
            json.dump({
                'status': f'fold_{fold_idx+1}_of_6_complete',
                'fold_results': fold_results,
            }, f, indent=2)

        # Free memory
        del trainer, train_loader, val_loader, test_loader
        del train_dataset, val_dataset, test_dataset
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    total_time = time.time() - start_time

    # ========================================================================
    # AGGREGATE RESULTS
    # ========================================================================
    f1_scores = [r['f1_macro'] for r in fold_results]
    f1w_scores = [r['f1_weighted'] for r in fold_results]
    acc_scores = [r['accuracy'] for r in fold_results]

    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    mean_f1w = np.mean(f1w_scores)
    std_f1w = np.std(f1w_scores)
    mean_acc = np.mean(acc_scores)
    std_acc = np.std(acc_scores)

    logging.info("\n" + "=" * 80)
    logging.info("LOBO CROSS-VALIDATION COMPLETE")
    logging.info("=" * 80)
    logging.info("")
    logging.info("Per-fold results:")
    logging.info(f"  {'Fold':<6} {'Test Bearing':<15} {'F1-macro':<12} {'F1-weighted':<12} {'Accuracy':<12}")
    logging.info("  " + "-" * 57)
    for r in fold_results:
        logging.info(
            f"  {r['fold']:<6} {r['test_bearing']:<15} "
            f"{r['f1_macro']:<12.4f} {r['f1_weighted']:<12.4f} {r['accuracy']:<12.4f}"
        )

    logging.info("")
    logging.info("AGGREGATE RESULTS:")
    logging.info(f"  F1-macro:    {mean_f1:.4f} +/- {std_f1:.4f}")
    logging.info(f"  F1-weighted: {mean_f1w:.4f} +/- {std_f1w:.4f}")
    logging.info(f"  Accuracy:    {mean_acc:.4f} +/- {std_acc:.4f}")
    logging.info("")
    logging.info("COMPARISON:")
    logging.info(f"  Original (stratified split): F1 = 0.9343")
    logging.info(f"  LOBO CV (no leakage):        F1 = {mean_f1:.4f} +/- {std_f1:.4f}")
    logging.info(f"  Difference:                  {mean_f1 - 0.9343:+.4f}")
    logging.info("")
    logging.info(f"Total time: {total_time/60:.1f} min ({total_time/3600:.1f} hours)")
    logging.info("=" * 80)

    # Save final results
    final_results = {
        'experiment': 'Leave-One-Bearing-Out Cross-Validation',
        'model': 'AION NEXUS v6',
        'training_protocol': '3-stage progressive (5+20+15 epochs, identical to original)',
        'dataset': 'FEMTO PHM 2012',
        'n_bearings': len(bearing_names),
        'n_folds': len(bearing_names),
        'total_samples': total_samples,
        'aggregate': {
            'f1_macro_mean': float(mean_f1),
            'f1_macro_std': float(std_f1),
            'f1_weighted_mean': float(mean_f1w),
            'f1_weighted_std': float(std_f1w),
            'accuracy_mean': float(mean_acc),
            'accuracy_std': float(std_acc),
        },
        'comparison': {
            'original_stratified_f1': 0.9343,
            'lobo_f1_mean': float(mean_f1),
            'lobo_f1_std': float(std_f1),
            'delta': float(mean_f1 - 0.9343),
        },
        'fold_results': fold_results,
        'total_time_sec': total_time,
        'date': '2026-02-11',
    }

    with open(output_dir / 'lobo_cv_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)

    logging.info(f"\nResults saved to: {output_dir / 'lobo_cv_results.json'}")

    return final_results


if __name__ == "__main__":
    main()
