# -*- coding: utf-8 -*-
"""
Build demonstration video loops showing how each architecture's segmentation
degrades as texture disparity is perturbed.

Layout is 4 columns x 6 rows: one row per architecture, and per row
    col 1  clean (in-domain) image
    col 2  ground truth overlaid on the clean image
    col 3  perturbed image at the current texture-disparity level
    col 4  that model's prediction on the perturbed image, with the ground
           truth outlined in a second colour for comparison

The time axis walks the texture-disparity level (hardest -> easiest -> hardest,
so the clip loops seamlessly) and cycles over several test images.

Usage:
    python -m toskipornot.visualization.make_robustness_video --dataset busi
    python -m toskipornot.visualization.make_robustness_video --dataset all
"""

import argparse
import json
import os
import subprocess
import sys
from glob import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import to_rgb
from monai.networks.layers import Norm
from monai.networks.nets import AttentionUnet, BasicUNetPlusPlus, UNet, VNet
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from toskipornot.config import (  # noqa: E402
    checkpoint_path,
    data_path,
    describe,
    require,
    results_path,
)
from toskipornot.models.NoSkipUnet import NoSkipUNet  # noqa: E402
from toskipornot.models.NoSkipVnet import NoSkipVNet  # noqa: E402

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #

# Ordered from lowest texture disparity (hardest) to highest (easiest); see
# toskipornot/analyze/generate_figure_histogram_busi.py for how each is made.
VARIANTS = ["lower", "low", "in-domain", "high", "higher"]
VARIANT_ALIAS = {
    "lower": "Hardest  (speckle $\\sigma^2$=0.3)",
    "low": "Harder  (speckle $\\sigma^2$=0.1)",
    "in-domain": "Unperturbed  (in-domain)",
    "high": "Easier  (background blur)",
    "higher": "Easiest  (stronger background blur)",
}
# Ping-pong so the last frame leads back into the first.
SWEEP = ["lower", "low", "in-domain", "high", "higher", "high", "in-domain", "low"]

MODELS = ["UNet", "AttentionUNet", "UNet++", "NoSkipUNet", "VNet", "NoSkipVNet"]
MODEL_ALIAS = {
    "UNet": "U-Net\n(identity skips)",
    "AttentionUNet": "AGU-Net\n(attention-gated skips)",
    "UNet++": "U-Net++\n(dense skips)",
    "NoSkipUNet": "NoSkip U-Net\n(no skips)",
    "VNet": "V-Net\n(identity skips)",
    "NoSkipVNet": "NoSkip V-Net\n(no skips)",
}

# Locations come from toskipornot.config, so they follow TOSKIPORNOT_DATA /
# TOSKIPORNOT_CHECKPOINTS (or a .env file) rather than being baked in here.
DATASETS = {
    "busi": {
        "title": "Breast Ultrasound (BUSI)",
        "data": data_path("BUSI-experiment"),
        "models": checkpoint_path("busi"),
    },
    "glas": {
        "title": "Colon Histology (GLaS)",
        "data": data_path("GLaS-experiment"),
        "models": checkpoint_path("glas"),
    },
    "heart": {
        "title": "Left Atrium (Heart MRI)",
        "data": data_path("Task02_Heart-experiment"),
        "models": checkpoint_path("heart"),
    },
    "spleen": {
        "title": "Spleen (CT)",
        "data": data_path("Task09_Spleen-experiment"),
        "models": checkpoint_path("spleen"),
    },
}

# The medical (256px) runs use a 6-level encoder (16..512), one level deeper than
# the synthetic runs; see section 2.3 of the paper. Used when a run directory has
# no config.json of its own (the busi runs store training previews instead).
MEDICAL_PARAMS = {
    "UNet": dict(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2, 2),
        num_res_units=0,
        norm=Norm.BATCH,
        act="ReLU",
        bias=False,
    ),
    "AttentionUNet": dict(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2, 2),
    ),
    "UNet++": dict(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        features=(16, 32, 64, 128, 256, 512),
        norm=Norm.BATCH,
        act="ReLU",
        bias=False,
    ),
    "VNet": dict(
        spatial_dims=2, in_channels=1, out_channels=1, act="ReLU", dropout_prob=0.0
    ),
}
MEDICAL_PARAMS["NoSkipUNet"] = MEDICAL_PARAMS["UNet"]
MEDICAL_PARAMS["NoSkipVNet"] = MEDICAL_PARAMS["VNet"]

CONFIG_KEY = {
    "UNet": "unet_model_params",
    "AttentionUNet": "att_model_params",
    "UNet++": "unetplusplus_model_params",
    "NoSkipUNet": "noskipunet_model_params",
    "VNet": "vnet_model_params",
    "NoSkipVNet": "noskipvnet_model_params",
}

BUILDERS = {
    "UNet": UNet,
    "AttentionUNet": AttentionUnet,
    "UNet++": BasicUNetPlusPlus,
    "NoSkipUNet": NoSkipUNet,
    "VNet": VNet,
    "NoSkipVNet": NoSkipVNet,
}

GT_COLOUR = "#00e5ff"  # cyan
PRED_COLOUR = "#ff2d95"  # magenta

# --------------------------------------------------------------------------- #
# model loading
# --------------------------------------------------------------------------- #


def resolve_run_dir(models_root, model_name):
    """Find the run directory for a model, preferring seed 1."""
    for seed in (1, 2, 3):
        cand = os.path.join(models_root, f"{model_name}_256_{seed}")
        if os.path.isfile(
            os.path.join(cand, "best_metric_model_segmentation2d_dict.pth")
        ):
            return cand, seed
    raise FileNotFoundError(f"no checkpoint for {model_name} under {models_root}")


def build_model(model_name, run_dir):
    """Instantiate a model, taking params from config.json when present."""
    params = None
    cfg_path = os.path.join(run_dir, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as fh:
            cfg = json.load(fh)
        key = CONFIG_KEY[model_name]
        if key in cfg:
            raw = cfg[key]
            params = raw["value"] if isinstance(raw, dict) and "value" in raw else raw
    if params is None:
        params = MEDICAL_PARAMS[model_name]
    params = dict(params)
    if model_name == "VNet" and "dropout_prob" in params:
        # MONAI >=1.2 split this into down/up, and its deprecation shim forwards a
        # bare float into the up path where a tuple is required. Dropout is inert
        # at eval, so the translation does not change outputs.
        prob = params.pop("dropout_prob")
        params["dropout_prob_down"] = prob
        params["dropout_prob_up"] = (prob, prob)
    model = BUILDERS[model_name](**params)
    state = torch.load(
        os.path.join(run_dir, "best_metric_model_segmentation2d_dict.pth"),
        map_location="cpu",
    )
    model.load_state_dict(state)
    model.eval()
    return model


def load_models(models_root):
    loaded = {}
    for name in MODELS:
        run_dir, seed = resolve_run_dir(models_root, name)
        loaded[name] = {"model": build_model(name, run_dir), "seed": seed}
        note = "" if seed == 1 else f"  (seed {seed} — seed 1 unavailable)"
        print(f"    loaded {name:14s} from {os.path.basename(run_dir)}{note}")
    return loaded


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #


def scale_intensity(arr):
    """MONAI ScaleIntensity: min-max to [0, 1]."""
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def read_gray(path):
    return np.asarray(Image.open(path).convert("L")).astype(np.float32)


def dice(pred, gt):
    pred = pred > 0.5
    gt = gt > 0.5
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, gt).sum() / denom)


def pick_images(data_root, n_images, models, device, pool_size=28):
    """Pick the test cases that best demonstrate the effect.

    A case is only informative if the models segment it well unperturbed and
    then visibly fail as texture disparity drops. So we shortlist cases with a
    clearly visible foreground, score each by (in-domain Dice) - (hardest Dice)
    averaged over architectures, and keep the largest drops.
    """
    labels = sorted(glob(os.path.join(data_root, "in-domain", "test", "label", "*.png")))
    sized = []
    for idx, path in enumerate(labels):
        frac = float((read_gray(path) > 127).mean())
        if 0.02 <= frac <= 0.45:
            sized.append((idx, frac))
    if len(sized) < n_images:
        sized = [(i, float((read_gray(p) > 127).mean())) for i, p in enumerate(labels)]

    # Even stride over the shortlist so the probe spans the whole test set.
    stride = max(1, len(sized) // pool_size)
    candidates = [sized[i][0] for i in range(0, len(sized), stride)][:pool_size]

    print(f"    probing {len(candidates)} candidate cases for the clearest effect")
    ranked = []
    for idx in candidates:
        clean_img, gt = load_case(data_root, "in-domain", idx)
        hard_img, _ = load_case(data_root, "lower", idx)
        base, hard = [], []
        for name in MODELS:
            model = models[name]["model"]
            base.append(dice(predict(model, clean_img, device), gt))
            hard.append(dice(predict(model, hard_img, device), gt))
        base_m, hard_m = float(np.mean(base)), float(np.mean(hard))
        # Require a usable starting point, else the drop is meaningless.
        if base_m >= 0.55:
            ranked.append((base_m - hard_m, base_m, idx))

    if len(ranked) < n_images:  # fall back to biggest foregrounds
        return sorted(idx for _, idx in sorted(sized, key=lambda t: -t[1])[:n_images])

    ranked.sort(reverse=True)
    chosen = [idx for _, _, idx in ranked[:n_images]]
    for drop, base_m, idx in ranked[:n_images]:
        print(f"      case {idx:4d}  in-domain {base_m:.3f} -> hardest {base_m - drop:.3f}")
    return sorted(chosen)


def load_case(data_root, variant, idx):
    img = sorted(glob(os.path.join(data_root, variant, "test", "image", "*.png")))[idx]
    lbl = sorted(glob(os.path.join(data_root, variant, "test", "label", "*.png")))[idx]
    return scale_intensity(read_gray(img)), (read_gray(lbl) > 127).astype(np.float32)


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #


def predict(model, image, device):
    """Segment one image, in the orientation the networks were trained in.

    MONAI's PILReader swaps the first two axes relative to PIL, so the training
    and evaluation pipeline in this repo fed the networks transposed arrays. We
    transpose going in and coming back out, which keeps everything else here in
    natural image orientation while reproducing the published per-image Dice
    exactly (verified against results/BUSI-results to ~2e-8).
    """
    tensor = torch.from_numpy(np.ascontiguousarray(image.T))[None, None].to(device)
    with torch.no_grad():
        out = model(tensor)
    if isinstance(out, (list, tuple)):
        out = out[0]
    prob = torch.sigmoid(out)[0, 0].cpu().numpy()
    return (prob.T > 0.5).astype(np.float32)


def run_inference(dataset, models, image_ids, device, cache_dir):
    """Predictions and Dice for every (model, image, variant); cached to npz."""
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"{dataset}_predictions.npz")
    meta_path = os.path.join(cache_dir, f"{dataset}_dice.json")
    if os.path.isfile(cache) and os.path.isfile(meta_path):
        print(f"    reusing cached predictions: {cache}")
        blob = np.load(cache)
        with open(meta_path) as fh:
            scores = json.load(fh)
        return blob, scores

    data_root = DATASETS[dataset]["data"]
    arrays = {}
    scores = {}
    total = len(MODELS) * len(image_ids) * len(VARIANTS)
    step = 0
    for idx in image_ids:
        for variant in VARIANTS:
            image, gt = load_case(data_root, variant, idx)
            arrays[f"img_{idx}_{variant}"] = image
            arrays[f"gt_{idx}"] = gt
            for name in MODELS:
                pred = predict(models[name]["model"], image, device)
                arrays[f"pred_{idx}_{variant}_{name}"] = pred
                scores[f"{idx}|{variant}|{name}"] = dice(pred, gt)
                step += 1
                if step % 30 == 0 or step == total:
                    print(f"    inference {step}/{total}")
    np.savez_compressed(cache, **arrays)
    with open(meta_path, "w") as fh:
        json.dump(scores, fh, indent=1)
    print(f"    wrote {cache}")
    return np.load(cache), scores


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #


def tint(mask, colour, alpha):
    """Build an RGBA overlay for a binary mask."""
    rgba = np.zeros(mask.shape + (4,), dtype=np.float32)
    rgba[..., :3] = to_rgb(colour)
    rgba[..., 3] = (mask > 0.5) * alpha
    return rgba


def draw_grid(
    clean, gt, perturbed, preds, dices, seeds, title, subtitle, out_path
):
    """Render one 4-column x 6-row frame.

    clean/gt/perturbed are 2D arrays; preds and dices are keyed by model name.
    Shared by the medical and synthetic drivers.
    """
    nrows, ncols = len(MODELS), 4
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.35 * ncols + 1.15, 2.35 * nrows + 1.0),
        facecolor="#0d1117",
    )
    fig.subplots_adjust(
        left=0.135, right=0.985, top=0.918, bottom=0.038, wspace=0.02, hspace=0.04
    )

    col_titles = [
        "Clean image",
        "Ground truth",
        "Perturbed image",
        "Prediction vs. truth",
    ]

    for r, name in enumerate(MODELS):
        pred = preds[name]
        dsc = dices[name]
        panels = [
            (clean, []),
            (clean, [tint(gt, GT_COLOUR, 0.45)]),
            (perturbed, []),
            (perturbed, [tint(pred, PRED_COLOUR, 0.45)]),
        ]
        for c, (base, overlays) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(base, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            for ov in overlays:
                ax.imshow(ov, interpolation="nearest")
            if c == 3:
                # Ground truth as an outline so both regions stay readable.
                ax.contour(gt, levels=[0.5], colors=[GT_COLOUR], linewidths=1.4)
                ax.text(
                    0.035,
                    0.045,
                    f"Dice {dsc:.3f}",
                    transform=ax.transAxes,
                    fontsize=11,
                    color="#0d1117",
                    weight="bold",
                    bbox=dict(
                        facecolor="#e6edf3", edgecolor="none", pad=2.6, alpha=0.92
                    ),
                )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#30363d")
            if r == 0:
                ax.set_title(
                    col_titles[c], fontsize=12.5, color="#e6edf3", pad=8, weight="600"
                )
        seed = seeds[name]
        label = MODEL_ALIAS[name] + ("" if seed == 1 else f"\n[seed {seed}]")
        axes[r, 0].set_ylabel(
            label, fontsize=11.5, color="#e6edf3", labelpad=12, linespacing=1.5
        )

    # Title on its own line, conditions below, so long scenario names do not
    # overflow the figure width.
    fig.suptitle(
        f"{title}\n{subtitle}",
        fontsize=14.5,
        color="#e6edf3",
        y=0.986,
        linespacing=1.5,
    )

    # Legend for the two overlay colours.
    fig.text(
        0.137,
        0.006,
        "■ ground truth",
        color=GT_COLOUR,
        fontsize=11,
        weight="bold",
        ha="left",
    )
    fig.text(
        0.30, 0.006, "■ model prediction", color=PRED_COLOUR, fontsize=11,
        weight="bold", ha="left",
    )

    fig.savefig(out_path, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)


def encode(frame_dir, out_file, seconds_per_frame, crf=20, fps=25):
    """Frames -> H.264 mp4, padded to even dimensions.

    The frames are long static holds of high-entropy (noisy) content, so `crf`
    dominates file size: 20 for archive quality, ~30 with `-tune stillimage` for
    a repo-friendly copy.
    """
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", f"{1.0 / seconds_per_frame:.6f}",
        "-i", os.path.join(frame_dir, "frame_%04d.png"),
        "-vf", f"fps={fps},pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf),
        "-tune", "stillimage", "-movflags", "+faststart",
        out_file,
    ]
    subprocess.run(cmd, check=True)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def build(dataset, args):
    cfg = DATASETS[dataset]
    print(f"\n=== {dataset}: {cfg['title']}")
    require(cfg["data"])
    require(cfg["models"])

    device = torch.device(args.device)
    models = load_models(cfg["models"])

    image_ids = pick_images(cfg["data"], args.n_images, models, device)
    print(f"    test cases: {image_ids}")

    blob, scores = run_inference(dataset, models, image_ids, device, args.cache_dir)

    frame_dir = os.path.join(args.out_dir, f"frames_{dataset}")
    os.makedirs(frame_dir, exist_ok=True)
    for old in glob(os.path.join(frame_dir, "*.png")):
        os.remove(old)

    n = 0
    for idx in image_ids:
        for variant in SWEEP:
            draw_grid(
                clean=blob[f"img_{idx}_in-domain"],
                gt=blob[f"gt_{idx}"],
                perturbed=blob[f"img_{idx}_{variant}"],
                preds={m: blob[f"pred_{idx}_{variant}_{m}"] for m in MODELS},
                dices={m: scores[f"{idx}|{variant}|{m}"] for m in MODELS},
                seeds={m: models[m]["seed"] for m in MODELS},
                title=cfg["title"],
                subtitle=(
                    f"texture disparity: {VARIANT_ALIAS[variant]}"
                    f"     ·     test case {idx}"
                ),
                out_path=os.path.join(frame_dir, f"frame_{n:04d}.png"),
            )
            n += 1
    print(f"    rendered {n} frames")

    os.makedirs(args.out_dir, exist_ok=True)
    out_file = os.path.join(args.out_dir, f"robustness_{dataset}.mp4")
    encode(frame_dir, out_file, args.seconds_per_frame, crf=args.crf)
    size_mb = os.path.getsize(out_file) / 1e6
    print(f"    wrote {out_file}  ({size_mb:.1f} MB, {n * args.seconds_per_frame:.0f}s)")

    # Report the degradation each architecture shows, hardest vs unperturbed.
    print("    Dice, mean over test cases:")
    for name in MODELS:
        row = [
            np.mean([scores[f"{i}|{v}|{name}"] for i in image_ids]) for v in VARIANTS
        ]
        cells = "  ".join(f"{v:>5.3f}" for v in row)
        print(f"      {name:14s} {cells}")
    print(f"      {'':14s} " + "  ".join(f"{v:>5s}" for v in VARIANTS))
    return out_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=list(DATASETS) + ["all"])
    ap.add_argument("--n-images", type=int, default=5)
    ap.add_argument("--seconds-per-frame", type=float, default=0.9)
    ap.add_argument("--out-dir", default=str(results_path("videos")))
    ap.add_argument("--cache-dir", default=str(results_path("video-cache")))
    ap.add_argument("--crf", type=int, default=20, help="x264 quality; higher = smaller file")
    ap.add_argument(
        "--show-config",
        action="store_true",
        help="print the resolved data/checkpoint/results locations and exit",
    )
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if args.show_config:
        print(describe())
        return

    targets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    made = [build(d, args) for d in targets]
    print("\nDone:")
    for m in made:
        print("  " + m)


if __name__ == "__main__":
    main()
