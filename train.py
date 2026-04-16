"""
train.py — Main training loop for replay-free continual learning.

Runs one method across all 5 tasks of Split-CIFAR-100, collecting:
    • per-task accuracy matrix
    • Average Accuracy (AA)
    • Backward Transfer (BWT)
    • RAM delta and wall-clock time per task

Usage:
    python train.py --method naive
    python train.py --method ewc
    python train.py --method lwf
    python train.py --method hybrid
    python train.py --method all        # runs all four sequentially
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.optim as optim

from data    import get_task_loaders, NUM_TASKS, get_task_classes
from model   import ContinualResNet
from metrics import evaluate, compute_aa_bwt, ResourceTracker
from methods import METHOD_REGISTRY, apply_prototype_alignment

# ── config ───────────────────────────────────────────────────────────────────
EPOCHS_PER_TASK = 5
LR              = 1e-3
RESULTS_DIR     = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── single method run ────────────────────────────────────────────────────────

def run_method(method_name: str, device: torch.device,
               align: bool = False, unfreeze: bool = False,
               scale: float = 1.0) -> dict:
    label = method_name.upper()
    if unfreeze:
        label += " + Unfreeze Layer4"
    if align:
        label += " + Prototype Alignment"
    if scale != 1.0:
        label += f" + Scale={scale}"
    print(f"\n{'='*60}")
    print(f"  Method: {label}")
    print(f"{'='*60}")

    method_cls = METHOD_REGISTRY[method_name]
    method     = method_cls()
    model      = ContinualResNet(unfreeze_last_block=unfreeze, scale=scale).to(device)

    acc_matrix  = np.zeros((NUM_TASKS, NUM_TASKS))   # [after_task, task_id]
    task_times  = []
    task_rams   = []
    proto_store = {}   # class_id -> normalised prototype; accumulated across tasks

    # Preload test loaders for all tasks (to evaluate older tasks later)
    all_test_loaders = []

    for task_id in range(NUM_TASKS):
        print(f"\n── Task {task_id + 1}/{NUM_TASKS} "
              f"(classes {get_task_classes(task_id)[0]}–{get_task_classes(task_id)[-1]}) ──")

        train_loader, test_loader = get_task_loaders(task_id)
        all_test_loaders.append(test_loader)

        # Expand head for this task
        model.expand_head(task_id)
        model.to(device)

        # Optimizer recreated each task (only head params)
        optimizer = optim.Adam(model.get_trainable_params(), lr=LR)

        # Before-task hook (LwF snapshot, etc.)
        method.before_task(model, task_id, train_loader, device)

        # ── training ─────────────────────────────────────────────────────────
        with ResourceTracker() as tracker:
            model.train()
            for epoch in range(EPOCHS_PER_TASK):
                epoch_loss = 0.0
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    loss = method.loss(model, x, y, task_id)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                avg_loss = epoch_loss / len(train_loader)
                print(f"  Epoch {epoch+1}/{EPOCHS_PER_TASK}  loss={avg_loss:.4f}")

        task_times.append(tracker.elapsed_sec)
        task_rams.append(tracker.peak_ram_mb)
        print(f"  Time: {tracker.elapsed_sec:.1f}s  |  RAM Δ: {tracker.peak_ram_mb:.1f} MB")

        # Optional prototype alignment — runs BEFORE after_task so EWC snapshots
        # the aligned weights (protects post-alignment state, not raw gradient state)
        if align:
            apply_prototype_alignment(model, train_loader, task_id, device, proto_store)

        # After-task hook (EWC fisher computed on aligned weights when align=True)
        method.after_task(model, task_id, train_loader, device)

        # ── evaluation on all seen tasks ──────────────────────────────────────
        model.eval()
        for prev_id in range(task_id + 1):
            acc = evaluate(model, all_test_loaders[prev_id], device)
            acc_matrix[task_id, prev_id] = acc
            print(f"  Acc on task {prev_id+1}: {acc*100:.1f}%")

    # ── summary ───────────────────────────────────────────────────────────────
    aa, bwt = compute_aa_bwt(acc_matrix)
    print(f"\n  ── {label} Final Results ──")
    print(f"  Average Accuracy (AA):   {aa*100:.2f}%")
    print(f"  Backward Transfer (BWT): {bwt*100:.2f}%")
    print(f"  Mean time/task:          {np.mean(task_times):.1f}s")
    print(f"  Mean RAM Δ/task:         {np.mean(task_rams):.1f} MB")

    suffix = ""
    if align:
        suffix += "_align"
    if unfreeze:
        suffix += "_ft"
    if scale != 1.0:
        suffix += f"_s{int(scale)}"
    result_key = f"{method_name}{suffix}"
    results = {
        "method":       result_key,
        "align":        align,
        "unfreeze":     unfreeze,
        "scale":        scale,
        "acc_matrix":   acc_matrix.tolist(),
        "aa":           round(aa * 100, 2),
        "bwt":          round(bwt * 100, 2),
        "task_times":   [round(t, 2) for t in task_times],
        "task_rams_mb": [round(r, 2) for r in task_rams],
    }

    out_path = os.path.join(RESULTS_DIR, f"{result_key}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved → {out_path}")

    return results


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="all",
                        choices=list(METHOD_REGISTRY.keys()) + ["all"])
    parser.add_argument("--align", action="store_true",
                        help="Apply prototype alignment after each task")
    parser.add_argument("--unfreeze", action="store_true",
                        help="Unfreeze ResNet-18 layer4 during training (fine-tune backbone)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Cosine classifier temperature scale (e.g. 10.0)")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    methods = list(METHOD_REGISTRY.keys()) if args.method == "all" else [args.method]
    all_results = {}

    for m in methods:
        all_results[m] = run_method(m, device, align=args.align,
                                    unfreeze=args.unfreeze, scale=args.scale)

    if len(methods) > 1:
        parts = []
        if args.align:
            parts.append("+align")
        if args.unfreeze:
            parts.append("+ft")
        if args.scale != 1.0:
            parts.append(f"+scale={args.scale}")
        suffix = " (" + ", ".join(parts) + ")" if parts else ""
        print(f"\n\n── Summary Table{suffix} ─────────────────────────────────────")
        print(f"{'Method':<10} {'AA (%)':>8} {'BWT (%)':>9} {'Time/task (s)':>15} {'RAM Δ (MB)':>12}")
        print("-" * 58)
        for m, r in all_results.items():
            t_mean = round(float(np.mean(r["task_times"])), 1)
            ram_mean = round(float(np.mean(r["task_rams_mb"])), 1)
            print(f"{m:<10} {r['aa']:>8.2f} {r['bwt']:>9.2f} {t_mean:>15.1f} {ram_mean:>12.1f}")


if __name__ == "__main__":
    main()
