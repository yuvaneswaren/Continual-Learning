"""
model.py — Frozen ResNet-18 backbone with an incrementally expanding linear head.
Only the head is trained; the backbone is fixed throughout.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from data import CLASSES_PER_TASK


class ContinualResNet(nn.Module):
    """
    ResNet-18 with frozen pretrained backbone.
    The classifier head expands by CLASSES_PER_TASK neurons after each task.
    """

    def __init__(self, classes_per_task: int = CLASSES_PER_TASK,
                 unfreeze_last_block: bool = False,
                 scale: float = 1.0):
        super().__init__()
        self.classes_per_task    = classes_per_task
        self.unfreeze_last_block = unfreeze_last_block
        self.scale               = scale   # temperature for cosine classifier (e.g. 10.0)

        # ── backbone ─────────────────────────────────────────────────────────
        # Try pretrained weights; fall back to random init if offline
        try:
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            print("Loaded pretrained ResNet-18 backbone.")
        except Exception:
            backbone = models.resnet18(weights=None)
            print("WARNING: pretrained weights unavailable — using random init.")
        self.feature_dim = backbone.fc.in_features   # 512

        # Remove the original classification head
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # Freeze entire backbone first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Optionally unfreeze layer4 (backbone[7]) for fine-tuning
        if unfreeze_last_block:
            for param in self.backbone[7].parameters():
                param.requires_grad = True
            print("Unfrozen backbone layer4 for fine-tuning.")

        # ── head (trained) ───────────────────────────────────────────────────
        # Starts empty; call expand_head() before each task
        self.head = nn.Linear(self.feature_dim, 0)
        self._num_classes = 0

    # ── helpers ──────────────────────────────────────────────────────────────

    def expand_head(self, task_id: int):
        """
        Add CLASSES_PER_TASK output neurons for task_id.
        Previous weights are preserved exactly.
        """
        new_total = (task_id + 1) * self.classes_per_task
        if new_total <= self._num_classes:
            return  # already expanded

        old_head = self.head
        new_head = nn.Linear(self.feature_dim, new_total)

        # Copy old weights
        if self._num_classes > 0:
            with torch.no_grad():
                new_head.weight[:self._num_classes] = old_head.weight
                new_head.bias[:self._num_classes]   = old_head.bias

        self.head = new_head
        self._num_classes = new_total

    def get_trainable_params(self):
        """Head parameters, plus layer4 if unfrozen."""
        params = list(self.head.parameters())
        if self.unfreeze_last_block:
            params += list(self.backbone[7].parameters())
        return params

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.unfreeze_last_block:
            feats = self.backbone(x)          # gradients flow through layer4
        else:
            with torch.no_grad():
                feats = self.backbone(x)
        feats = feats.flatten(1)              # (B, 512)
        if self.scale != 1.0:
            feats = F.normalize(feats, dim=1) # cosine classifier: unit-norm features
        logits = self.head(feats)             # (B, num_classes_so_far)
        return logits * self.scale

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return backbone features (no grad)."""
        with torch.no_grad():
            feats = self.backbone(x)
        return feats.flatten(1)


if __name__ == "__main__":
    model = ContinualResNet()
    for t in range(5):
        model.expand_head(t)
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        print(f"Task {t+1}: head output shape = {out.shape}")
    print("Model OK.")
