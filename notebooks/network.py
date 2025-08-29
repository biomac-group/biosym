# JAX/Flax 1D U-Net for sequence segmentation (input size 45x500)
# ----------------------------------------------------------------
# - Input shape: (batch, 45, 500, 1), where 500 is the long axis (sequence length)
# - Uses Conv1D along the long axis (500) while keeping 45 as a feature/channel dimension
# - Includes training loop with BCE-with-logits + Dice loss, metrics, and a toy dataset

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import random

from flax import linen as nn
from flax.training import train_state
import optax
from tqdm import tqdm

# ---------------------------
# Model: 1D U-Net building blocks
# ---------------------------

class ConvBlock1D(nn.Module):
    features: int
    groups: int = 8

    @nn.compact
    def __call__(self, x, *, train: bool):
        x = nn.Conv(self.features, kernel_size=(5,), padding="SAME")(x)
        x = nn.GroupNorm(num_groups=min(self.groups, self.features))(x)
        x = nn.gelu(x)
        x = nn.Conv(self.features, kernel_size=(5,), padding="SAME")(x)
        x = nn.GroupNorm(num_groups=min(self.groups, self.features))(x)
        x = nn.gelu(x)
        return x

class Down1D(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x, *, train: bool):
        x = nn.max_pool(x, window_shape=(2,), strides=(2,), padding="SAME")
        x = ConvBlock1D(self.features)(x, train=train)
        return x

class Up1D(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x, skip, *, train: bool):
        target_len = skip.shape[1]
        # Resize along the sequence length axis only, keep batch and channels intact
        x = jax.image.resize(x, (skip.shape[0], target_len, x.shape[2]), method="linear")
        x = jnp.concatenate([x, skip], axis=-1)
        x = ConvBlock1D(self.features)(x, train=train)
        return x
    
class UNet1D(nn.Module):
    base_features: int = 32
    num_classes: int = 1

    @nn.compact
    def __call__(self, x, *, train: bool = True):
        # Encoder
        c1 = ConvBlock1D(self.base_features)(x, train=train)
        d1 = Down1D(self.base_features * 2)(c1, train=train)
        d2 = Down1D(self.base_features * 4)(d1, train=train)
        d3 = Down1D(self.base_features * 8)(d2, train=train)
        d4 = Down1D(self.base_features * 16)(d3, train=train)

        # Bottleneck
        b = ConvBlock1D(self.base_features * 32)(d4, train=train)

        # Decoder
        u1 = Up1D(self.base_features * 16)(b, d3, train=train)
        u2 = Up1D(self.base_features * 8)(u1, d2, train=train)
        u3 = Up1D(self.base_features * 4)(u2, d1, train=train)
        u4 = Up1D(self.base_features * 2)(u3, c1, train=train)

        logits = nn.Conv(45, (1,), padding="SAME")(u4)
        return logits

# ---------------------------
# Losses & Metrics
# ---------------------------

def bce_with_logits(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    return optax.sigmoid_binary_cross_entropy(logits, targets).mean()

def dice_loss(logits: jnp.ndarray, targets: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    probs = jax.nn.sigmoid(logits)
    intersection = jnp.sum(probs * targets)
    union = jnp.sum(probs) + jnp.sum(targets)
    dice = (2 * intersection + eps) / (union + eps)
    return 1.0 - dice

def compute_metrics(logits: jnp.ndarray, targets: jnp.ndarray) -> dict:
    probs = jax.nn.sigmoid(logits)
    bce = bce_with_logits(logits, targets)
    dl = dice_loss(logits, targets)
    preds = (probs > 0.5).astype(jnp.float32)
    acc = jnp.mean((preds == targets).astype(jnp.float32))
    inter = jnp.sum(preds * targets)
    union = jnp.sum(jnp.clip(preds + targets, 0, 1)) + 1e-6
    iou = inter / union
    return {"bce": bce, "dice": dl, "acc": acc, "iou": iou}

# ---------------------------
# Train state
# ---------------------------
@dataclass
class TrainConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    seed: int = 0

def create_train_state(rng, model: nn.Module, input_shape: Tuple[int, int, int], cfg: TrainConfig):
    params = model.init(rng, jnp.zeros(input_shape), train=True)["params"]
    tx = optax.adamw(learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# ---------------------------
# Training & evaluation steps
# ---------------------------
@jax.jit
def train_step(state: train_state.TrainState, batch: dict):
    seqs, masks = batch["seq"], batch["mask"]

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, seqs, train=True)
        loss = bce_with_logits(logits, masks) + dice_loss(logits, masks)
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits, masks) | {"loss": loss}
    return state, metrics

@jax.jit
def eval_step(state: train_state.TrainState, batch: dict):
    seqs, masks = batch["seq"], batch["mask"]
    logits = state.apply_fn({"params": state.params}, seqs, train=False)
    return compute_metrics(logits, masks)

# ---------------------------
# Toy dataset utilities
# ---------------------------
def make_toy_batch(rng, batch_size: int, length: int = 500, channels: int = 45) -> dict:
    key_img, _ = random.split(rng)

    def sample_one(key):
        # shape (length, channels) -> (500, 45)
        seq = random.normal(key, (length, channels))
        # mask depends only on length axis, keep channel dim = 1
        mask = (seq.mean(axis=-1, keepdims=True) > 0).astype(jnp.float32)  # (length, 1)
        return seq, mask

    seqs, masks = [], []
    for i in range(batch_size):
        s, m = sample_one(random.fold_in(key_img, i))
        seqs.append(s)
        masks.append(m)

    # stack into (batch, length, channels) and (batch, length, 1)
    return {"seq": jnp.stack(seqs), "mask": jnp.stack(masks)}

# ---------------------------
# Training loop demo
# ---------------------------
def train_demo(num_steps: int = 200, batch_size: int = 4):
    cfg = TrainConfig()
    rng = random.PRNGKey(cfg.seed)
    model = UNet1D(base_features=32, num_classes=1)
    input_shape = (batch_size, 500, 45)  # (batch, length, channels)
    state = create_train_state(rng, model, input_shape, cfg)

    metrics_history = {"loss": [], "bce": [], "dice": [], "acc": [], "iou": []}

    for step in tqdm(range(1, num_steps + 1), desc="Training"):
        rng, key = random.split(rng)
        batch = make_toy_batch(key, batch_size=batch_size, length=500, channels=45)
        state, metrics = train_step(state, batch)
        for k in metrics_history:
            metrics_history[k].append(float(metrics[k]))
        if step % 20 == 0:
            eval_metrics = eval_step(state, batch)
            print(f"Step {step:04d} | loss={metrics['loss']:.4f} bce={metrics['bce']:.4f} dice={metrics['dice']:.4f} acc={metrics['acc']:.3f} iou={metrics['iou']:.3f}")

    return state, metrics_history

# Example usage (uncomment to run):
if __name__ == "__main__":
    state, hist = train_demo(num_steps=200, batch_size=4)
    print(state, hist)
