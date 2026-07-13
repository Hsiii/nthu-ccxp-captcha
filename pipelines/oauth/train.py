import argparse
import csv
import json
import os
import random
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
import tqdm
from torch import nn
from torch.utils import data

try:
    from .net import DIGITS, Net
    from .paths import PIPELINE, resolve_repo_path
except ImportError:
    from net import DIGITS, Net
    from paths import PIPELINE, resolve_repo_path

from pipeline_artifacts import build_training_output_paths, prepare_training_output_dir

DEFAULT_SEED = None
MAX_TRAIN_RENDERS_PER_GROUP = 20
EARLY_STOPPING_PATIENCE = 3
LR_PLATEAU_PATIENCE = 3


def seed_everything(seed: Optional[int] = DEFAULT_SEED):
    if seed is None:
        seed = int(time.time() * 1000) % (2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class DatasetSplit:
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray


class CaptchaDataset(data.Dataset):
    def __init__(
        self,
        data_dir='.',
        transform=None,
        split_name: str = 'train',
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: Optional[int] = DEFAULT_SEED,
    ):
        images = np.load(os.path.join(data_dir, 'images.npy'))
        labels = np.load(os.path.join(data_dir, 'labels.npy'))
        groups = np.load(os.path.join(data_dir, 'groups.npy'))

        actual_seed = seed if seed is not None else int(time.time() * 1000) % (2**32 - 1)
        split = self._build_split(groups, train_ratio=train_ratio, val_ratio=val_ratio, seed=actual_seed)
        split_indices = {
            'train': split.train_indices,
            'val': split.val_indices,
            'test': split.test_indices,
        }
        if split_name not in split_indices:
            raise ValueError(f'Unknown split name: {split_name}')
        indices = split_indices[split_name]
        if split_name == 'train':
            indices = self._cap_group_samples(indices, groups, max_samples=MAX_TRAIN_RENDERS_PER_GROUP, seed=actual_seed)

        self.images = images[indices]
        self.labels = torch.as_tensor(labels[indices], dtype=torch.long)
        self.groups = groups[indices]
        self.transform = transform
        self.split_name = split_name

    @staticmethod
    def _build_split(groups: np.ndarray, train_ratio: float, val_ratio: float, seed: int) -> DatasetSplit:
        unique_groups = np.array(sorted(set(groups.tolist())))
        rng = np.random.default_rng(seed)
        rng.shuffle(unique_groups)

        total_groups = len(unique_groups)
        if total_groups < 3:
            raise RuntimeError('Need at least 3 captcha groups for train/val/test splits.')

        train_size = max(1, int(round(total_groups * train_ratio)))
        val_size = max(1, int(round(total_groups * val_ratio)))
        if train_size + val_size >= total_groups:
            overflow = train_size + val_size - (total_groups - 1)
            train_size = max(1, train_size - overflow)
        test_size = total_groups - train_size - val_size
        if test_size < 1:
            if train_size > val_size:
                train_size -= 1
            else:
                val_size -= 1
            test_size = total_groups - train_size - val_size

        train_groups = set(unique_groups[:train_size].tolist())
        val_groups = set(unique_groups[train_size:train_size + val_size].tolist())
        test_groups = set(unique_groups[train_size + val_size:].tolist())

        train_indices, val_indices, test_indices = [], [], []
        for idx, group in enumerate(groups.tolist()):
            if group in train_groups:
                train_indices.append(idx)
            elif group in val_groups:
                val_indices.append(idx)
            else:
                test_indices.append(idx)

        return DatasetSplit(
            train_indices=np.array(train_indices, dtype=np.int64),
            val_indices=np.array(val_indices, dtype=np.int64),
            test_indices=np.array(test_indices, dtype=np.int64),
        )

    @staticmethod
    def _cap_group_samples(indices: np.ndarray, groups: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
        grouped_indices = defaultdict(list)
        for idx in indices.tolist():
            grouped_indices[groups[idx]].append(idx)

        rng = np.random.default_rng(seed)
        selected = []
        for group in sorted(grouped_indices):
            group_indices = grouped_indices[group]
            if len(group_indices) > max_samples:
                chosen = rng.choice(group_indices, size=max_samples, replace=False).tolist()
                selected.extend(sorted(chosen))
            else:
                selected.extend(group_indices)

        return np.array(selected, dtype=np.int64)

    def __getitem__(self, item):
        image = self.images[item]
        label = self.labels[item]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return self.images.shape[0]


def build_transforms() -> Tuple[T.Compose, T.Compose]:
    train_transform = T.Compose([
        T.ToPILImage(),
        T.RandomAffine(degrees=12, translate=(0.05, 0.08), shear=(-8, 8, -4, 4), fill=(255, 255, 255)),
        T.ColorJitter(brightness=0.15, contrast=0.2, saturation=0.2, hue=0.03),
        T.ToTensor(),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.2),
        T.RandomErasing(p=0.2, scale=(0.01, 0.05), ratio=(0.3, 3.0), value=1.0),
    ])
    eval_transform = T.Compose([T.ToTensor()])
    return train_transform, eval_transform


def multi_head_loss(logits: torch.Tensor, labels: torch.Tensor, loss_fn: nn.Module) -> torch.Tensor:
    return sum(loss_fn(logits[:, idx], labels[:, idx]) for idx in range(DIGITS)) / DIGITS


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    predictions = logits.argmax(dim=-1)
    digit_accuracy = (predictions == labels).float().mean().item()
    sequence_accuracy = (predictions == labels).all(dim=1).float().mean().item()
    return {
        'digit_accuracy': digit_accuracy,
        'sequence_accuracy': sequence_accuracy,
    }


def build_train_sampler(dataset: CaptchaDataset) -> data.WeightedRandomSampler:
    group_counts = Counter(dataset.groups.tolist())
    weights = torch.tensor([1.0 / group_counts[group] for group in dataset.groups.tolist()], dtype=torch.double)
    return data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def train_one_epoch(model, dataset, loss_fn, optim, device: torch.device):
    losses = []
    metric_history = []
    for x, y in tqdm.tqdm(dataset, desc='Training... ', leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = multi_head_loss(logits, y, loss_fn)
        losses.append(loss.item())
        metric_history.append(compute_metrics(logits, y))

        optim.zero_grad()
        loss.backward()
        optim.step()

    return float(np.mean(losses)), average_metrics(metric_history)


def collect_prediction_rows(logits: torch.Tensor, labels: torch.Tensor, groups: np.ndarray) -> List[Dict[str, object]]:
    predictions = logits.argmax(dim=-1).detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    rows = []
    for predicted_digits, label_digits, group in zip(predictions, labels_np, groups.tolist()):
        rows.append({
            'group': str(group),
            'truth': ''.join(str(int(digit)) for digit in label_digits),
            'prediction': ''.join(str(int(digit)) for digit in predicted_digits),
            'per_position_correct': [int(pred == truth) for pred, truth in zip(predicted_digits, label_digits)],
        })
    return rows


def summarize_prediction_rows(rows: List[Dict[str, object]]) -> Dict[str, object]:
    total_images = len(rows)
    image_exact = sum(1 for row in rows if row['prediction'] == row['truth']) / total_images

    position_correct = [0] * DIGITS
    confusion = np.zeros((DIGITS, 10, 10), dtype=np.int64)
    grouped_rows = defaultdict(list)
    for row in rows:
        grouped_rows[row['group']].append(row)
        truth = row['truth']
        prediction = row['prediction']
        for idx, (truth_digit, predicted_digit) in enumerate(zip(truth, prediction)):
            if truth_digit == predicted_digit:
                position_correct[idx] += 1
            confusion[idx, int(truth_digit), int(predicted_digit)] += 1

    group_exact_matches = 0
    for group_rows in grouped_rows.values():
        truth = group_rows[0]['truth']
        majority_prediction = Counter(row['prediction'] for row in group_rows).most_common(1)[0][0]
        if majority_prediction == truth:
            group_exact_matches += 1

    return {
        'image_sequence_accuracy': image_exact,
        'group_sequence_accuracy': group_exact_matches / len(grouped_rows),
        'position_accuracy': [correct / total_images for correct in position_correct],
        'confusion_matrix': confusion,
    }


def export_failure_rows(rows: List[Dict[str, object]], output_path: Path):
    failures = [row for row in rows if row['prediction'] != row['truth']]
    with output_path.open('w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['group', 'truth', 'prediction'])
        writer.writeheader()
        for row in failures:
            writer.writerow({
                'group': row['group'],
                'truth': row['truth'],
                'prediction': row['prediction'],
            })


def test_one_epoch(model, dataset, loss_fn, device: torch.device):
    losses = []
    metric_history = []
    prediction_rows = []
    offset = 0
    with torch.no_grad():
        for x, y in tqdm.tqdm(dataset, desc='Testing... ', leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = multi_head_loss(logits, y, loss_fn)
            losses.append(loss.item())
            metric_history.append(compute_metrics(logits, y))
            batch_groups = dataset.dataset.groups[offset:offset + x.shape[0]]
            prediction_rows.extend(collect_prediction_rows(logits, y, batch_groups))
            offset += x.shape[0]

    return float(np.mean(losses)), average_metrics(metric_history), summarize_prediction_rows(prediction_rows), prediction_rows


def average_metrics(metric_history):
    return {
        name: float(np.mean([metrics[name] for metrics in metric_history]))
        for name in metric_history[0]
    }


def describe_split(name: str, dataset: CaptchaDataset):
    unique_groups = len(set(dataset.groups.tolist()))
    print(f'{name}: {len(dataset)} images across {unique_groups} captcha groups')


def fit(model, train_ld, val_ld, loss_fn, optim, device: torch.device, scheduler=None, epochs=20):
    best_state = None
    best_sequence_accuracy = -1.0
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        train_loss, train_metrics = train_one_epoch(model, train_ld, loss_fn, optim, device)
        model.eval()
        val_loss, val_metrics, val_summary, _ = test_one_epoch(model, val_ld, loss_fn, device)

        print(
            f'Epoch {epoch:>2}: '
            f'train_loss = {train_loss:.6f}, '
            f'val_loss = {val_loss:.6f}, '
            f'train_seq_acc = {train_metrics["sequence_accuracy"]:.6f}, '
            f'val_seq_acc = {val_metrics["sequence_accuracy"]:.6f}, '
            f'val_digit_acc = {val_metrics["digit_accuracy"]:.6f}, '
            f'val_group_seq_acc = {val_summary["group_sequence_accuracy"]:.6f}'
        )

        if val_metrics['sequence_accuracy'] > best_sequence_accuracy:
            best_sequence_accuracy = val_metrics['sequence_accuracy']
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            scheduler.step(val_loss)

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f'Early stopping at epoch {epoch:>2} after {epochs_without_improvement} stale validation epochs.')
            break

    return best_state, best_sequence_accuracy


def export_quantized_model(model: nn.Module, output_path: str):
    supported_engines = [engine for engine in torch.backends.quantized.supported_engines if engine != 'none']
    if not supported_engines:
        warnings.warn(
            'Skipping dynamic quantized export because no quantization backend is available on this system.',
            RuntimeWarning,
        )
        return False

    original_engine = torch.backends.quantized.engine
    selected_engine = supported_engines[0]

    try:
        if original_engine != selected_engine:
            torch.backends.quantized.engine = selected_engine

        quantized_model = torch.quantization.quantize_dynamic(model.cpu(), {nn.Linear}, dtype=torch.qint8)
        scripted_model = torch.jit.script(quantized_model)
        scripted_model.save(output_path)
        return True
    except Exception as exc:
        warnings.warn(
            f'Skipping dynamic quantized export to {output_path}: {exc}',
            RuntimeWarning,
        )
        return False
    finally:
        if original_engine in supported_engines and original_engine != selected_engine:
            torch.backends.quantized.engine = original_engine


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', default=str(PIPELINE.default_resume_path()))
    parser.add_argument('--epochs', type=int, default=PIPELINE.train_epochs)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--data', default=str(PIPELINE.default_processed_dir()))
    parser.add_argument('--out', default=str(PIPELINE.paths.out_dir))
    parser.add_argument('--overwrite', action='store_true', default=True)
    parser.add_argument('--no-overwrite', dest='overwrite', action='store_false')
    args = parser.parse_args()
    args.resume = str(resolve_repo_path(args.resume))
    args.data = str(resolve_repo_path(args.data))
    args.out = str(resolve_repo_path(args.out))
    return args


def maybe_resume(model: nn.Module, checkpoint_path: str):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f'No resume checkpoint found at {checkpoint_path}; starting from scratch.')
        return

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    print(f'Resumed weights from {checkpoint_path}.')


def load_existing_checkpoint(checkpoint_path: Path) -> Optional[Dict[str, object]]:
    if not checkpoint_path.exists():
        return None
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if not isinstance(checkpoint, dict):
        return None
    return checkpoint


def extract_checkpoint_sequence_metrics(checkpoint: Optional[Dict[str, object]]) -> Tuple[Optional[float], Optional[float]]:
    if checkpoint is None:
        return None, None

    val_sequence_accuracy = checkpoint.get('sequence_accuracy')
    test_metrics = checkpoint.get('test_metrics')
    test_sequence_accuracy = None
    if isinstance(test_metrics, dict):
        test_sequence_accuracy = test_metrics.get('sequence_accuracy')

    return val_sequence_accuracy, test_sequence_accuracy


def should_replace_best_checkpoint(
    previous_checkpoint: Optional[Dict[str, object]],
    new_val_sequence_accuracy: float,
    new_test_sequence_accuracy: float,
) -> bool:
    previous_val_sequence_accuracy, previous_test_sequence_accuracy = extract_checkpoint_sequence_metrics(previous_checkpoint)
    if previous_val_sequence_accuracy is None or previous_test_sequence_accuracy is None:
        return True

    return (
        new_val_sequence_accuracy >= previous_val_sequence_accuracy
        and new_test_sequence_accuracy >= previous_test_sequence_accuracy
        and (
            new_val_sequence_accuracy > previous_val_sequence_accuracy
            or new_test_sequence_accuracy > previous_test_sequence_accuracy
        )
    )


def run_training(
    data_dir: str,
    output_dir_str: str,
    resume_path: str,
    epochs: int,
    seed: Optional[int],
    overwrite_output: bool,
) -> Dict[str, object]:
    seed_everything(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_transform, eval_transform = build_transforms()
    train_dataset = CaptchaDataset(data_dir=data_dir, transform=train_transform, split_name='train', seed=seed)
    val_dataset = CaptchaDataset(data_dir=data_dir, transform=eval_transform, split_name='val', seed=seed)
    test_dataset = CaptchaDataset(data_dir=data_dir, transform=eval_transform, split_name='test', seed=seed)

    describe_split('train', train_dataset)
    describe_split('val', val_dataset)
    describe_split('test', test_dataset)

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=128,
        sampler=build_train_sampler(train_dataset),
        num_workers=0,
    )
    val_loader = data.DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)
    test_loader = data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    model = Net().to(device)
    maybe_resume(model, resume_path)

    output_dir = Path(output_dir_str)
    current_output_paths = build_training_output_paths(output_dir)
    existing_best_checkpoint = load_existing_checkpoint(current_output_paths['best_checkpoint'])
    output_paths = prepare_training_output_dir(
        output_dir,
        overwrite_output=overwrite_output,
        preserve_paths=[
            Path(resume_path),
            current_output_paths['best_checkpoint'],
            current_output_paths['quantized_checkpoint'],
        ],
    )

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=LR_PLATEAU_PATIENCE,
    )

    best_state, best_sequence_accuracy = fit(
        model,
        train_loader,
        val_loader,
        loss,
        optimizer,
        device=device,
        scheduler=scheduler,
        epochs=epochs,
    )
    if best_state is None:
        raise RuntimeError('Training did not produce a checkpoint.')

    last_checkpoint = {
        'state_dict': {name: value.detach().cpu() for name, value in model.state_dict().items()},
        'sequence_accuracy': best_sequence_accuracy,
        'digits': DIGITS,
    }
    torch.save(last_checkpoint, output_paths['last_checkpoint'])

    model.load_state_dict(best_state)
    val_loss, val_metrics, val_summary, val_rows = test_one_epoch(model, val_loader, loss, device)
    test_loss, test_metrics, test_summary, test_rows = test_one_epoch(model, test_loader, loss, device)
    print(
        f'Best checkpoint validation metrics: '
        f'val_loss = {val_loss:.6f}, '
        f'val_seq_acc = {val_metrics["sequence_accuracy"]:.6f}, '
        f'val_group_seq_acc = {val_summary["group_sequence_accuracy"]:.6f}, '
        f'val_digit_acc = {val_metrics["digit_accuracy"]:.6f}'
    )
    print(
        f'Best checkpoint test metrics: '
        f'test_loss = {test_loss:.6f}, '
        f'test_seq_acc = {test_metrics["sequence_accuracy"]:.6f}, '
        f'test_group_seq_acc = {test_summary["group_sequence_accuracy"]:.6f}, '
        f'test_digit_acc = {test_metrics["digit_accuracy"]:.6f}'
    )

    np.save(output_paths['val_confusion'], val_summary['confusion_matrix'])
    np.save(output_paths['test_confusion'], test_summary['confusion_matrix'])
    export_failure_rows(val_rows, output_paths['val_failures'])
    export_failure_rows(test_rows, output_paths['test_failures'])
    print(f'val_position_acc = {val_summary["position_accuracy"]}')
    print(f'test_position_acc = {test_summary["position_accuracy"]}')

    checkpoint = {
        'state_dict': best_state,
        'sequence_accuracy': best_sequence_accuracy,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'val_summary': {
            'group_sequence_accuracy': val_summary['group_sequence_accuracy'],
            'position_accuracy': val_summary['position_accuracy'],
        },
        'test_summary': {
            'group_sequence_accuracy': test_summary['group_sequence_accuracy'],
            'position_accuracy': test_summary['position_accuracy'],
        },
        'digits': DIGITS,
        'seed': seed,
        'data_dir': data_dir,
    }
    if should_replace_best_checkpoint(
        existing_best_checkpoint,
        new_val_sequence_accuracy=val_metrics['sequence_accuracy'],
        new_test_sequence_accuracy=test_metrics['sequence_accuracy'],
    ):
        torch.save(checkpoint, output_paths['best_checkpoint'])
        if export_quantized_model(model, str(output_paths['quantized_checkpoint'])):
            print(f'Saved quantized checkpoint to {output_paths["quantized_checkpoint"]}')
        print(f'Saved best checkpoint to {output_paths["best_checkpoint"]}')
    else:
        previous_val_sequence_accuracy, previous_test_sequence_accuracy = extract_checkpoint_sequence_metrics(
            existing_best_checkpoint
        )
        print(
            'Keeping existing best checkpoint because the new run did not improve both sequence metrics: '
            f'old_val_seq_acc = {previous_val_sequence_accuracy:.6f}, '
            f'old_test_seq_acc = {previous_test_sequence_accuracy:.6f}, '
            f'new_val_seq_acc = {val_metrics["sequence_accuracy"]:.6f}, '
            f'new_test_seq_acc = {test_metrics["sequence_accuracy"]:.6f}'
        )

    summary = {
        'seed': seed,
        'data_dir': data_dir,
        'resume': resume_path,
        'output_dir': str(output_dir),
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'val_position_accuracy': val_summary['position_accuracy'],
        'test_position_accuracy': test_summary['position_accuracy'],
        'best_sequence_accuracy': best_sequence_accuracy,
    }
    with output_paths['metrics_summary'].open('w') as fp:
        json.dump(summary, fp, indent=2)
    print(f'Saved metrics summary to {output_paths["metrics_summary"]}')
    return summary


if __name__ == '__main__':
    args = parse_args()
    summary = run_training(
        data_dir=args.data,
        output_dir_str=args.out,
        resume_path=args.resume,
        epochs=args.epochs,
        seed=args.seed,
        overwrite_output=args.overwrite,
    )
    print(
        f'val_seq_acc = {summary["val_metrics"]["sequence_accuracy"]:.6f}, '
        f'test_seq_acc = {summary["test_metrics"]["sequence_accuracy"]:.6f}, '
        f'out = {summary["output_dir"]}'
    )
