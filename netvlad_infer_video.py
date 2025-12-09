import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F

from mmengine.config import Config
from oslactionspotting.models.builder_2 import build_model


_n_classes = [
    'Penalty',
    'Kick-off',
    'Goal',
    'Substitution',
    'Offside',
    'Shots on target',
    'Shots off target',
    'Clearance',
    'Ball out of play',
    'Throw-in',
    'Foul',
    'Indirect free-kick',
    'Direct free-kick',
    'Corner',
    'Yellow card',
    'Red card',
    'Yellow->red card',
]


def load_pca_npy(path):
    arr = np.load(path)
    assert len(arr.shape) == 2, "PCA input must be shape (T, 512)"
    return arr


def sliding_window(features, chunk_size):
    T = features.shape[0]
    clips = []
    idxs = []
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        clip = features[start:end]
        if clip.shape[0] < chunk_size:
            pad = np.zeros((chunk_size - clip.shape[0], clip.shape[1]))
            clip = np.concatenate([clip, pad], axis=0)
        clips.append(clip)
        idxs.append((start, end))
    return np.stack(clips), idxs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--out_json", type=str, default="preds.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--chunk_size", type=int, default=60)
    parser.add_argument("--framerate", type=float, default=2)
    parser.add_argument("--threshold", type=float, default=0.2)
    args = parser.parse_args()

    device = torch.device(args.device)

    print("[INFO] Loading PCA512 .npy…")
    raw = load_pca_npy(args.features)  # (T, 512)
    T = raw.shape[0]

    print(f"[INFO] Loaded {T} feature frames")

    print("[INFO] Creating sliding windows…")
    clips_np, clip_ranges = sliding_window(raw, args.chunk_size)
    print(f"[INFO] Created {len(clips_np)} clips")

    clips = torch.from_numpy(clips_np).float().to(device)  # (B, T, 512)

    print("[INFO] Loading config + model…")
    cfg = Config.fromfile(args.config)

    # Build full model defined in _base_/models/learnablepooling.py
    model = build_model(cfg.model)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()

    print("[INFO] Running inference…")
    all_logits = []

    with torch.no_grad():
        for i in range(clips.shape[0]):
            x = clips[i:i+1]  # (1, chunk, 512)
            out = model(x)

            # NetVLAD++ + Linear layer returns a SINGLE vector per clip:
            # shape = (B, 17)
            # We treat this as uniform logit for all frames in that clip.
            logits = out  # (1, 17)
            logits = logits.detach().cpu().numpy()[0]
            all_logits.append(logits)

    all_logits = np.stack(all_logits)  # (num_clips, 17)
    print("[INFO] Inference complete.")

    # Expand clip-level logits to frame-level
    print("[INFO] Expanding logits to frame-level")
    # frame_scores = np.zeros((T, cfg.model.head.num_classes))
    num_classes = all_logits.shape[1]
    frame_scores = np.zeros((T, num_classes))

    for (start, end), clip_logits in zip(clip_ranges, all_logits):
        #frame_scores[start:end] = clip_logits  # broadcast
        frame_scores[start:end] = clip_logits[None, :]


    # Sigmoid to probabilities
    probs = 1 / (1 + np.exp(-frame_scores))

    print("[INFO] Applying threshold + building JSON output…")
    events = []
    for t in range(T):
        ts = t / args.framerate  # seconds
        for cls in range(num_classes):
            p = probs[t, cls]
            if p >= args.threshold:
                events.append({
                    "timestamp": float(ts),
                    "label": _n_classes[int(cls)],
                    "confidence": float(p),
                })

    print(f"[INFO] Saving {len(events)} events → {args.out_json}")
    with open(args.out_json, "w") as f:
        json.dump(events, f, indent=2)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()

