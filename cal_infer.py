import os
import json
import argparse
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from mmengine import Config
from oslactionspotting.models.builder_2 import build_model

# ------------------------------
# Dataset for full fixed-length clips
# ------------------------------
class FullFeatureClipDataset(Dataset):
    """
    Load full PCA512 features .npy file and split into fixed length clips
    for CALF model (chunk_size * framerate = 240 frames).
    Pads shorter clips at the end.
    """

    def __init__(self, npy_path, chunk_size=120, framerate=2):
        self.features = np.load(npy_path)  # shape (T, 512)
        self.chunk_size = chunk_size
        self.framerate = framerate
        self.required_frames = chunk_size * framerate

        if self.features.ndim != 2 or self.features.shape[1] != 512:
            raise ValueError(f"Expected (T, 512) features, got {self.features.shape}")

        self.T = self.features.shape[0]
        self.indices = []
        step = self.required_frames  # no overlap; adjust if needed

        for start in range(0, max(1, self.T - self.required_frames + 1), step):
            end = start + self.required_frames
            self.indices.append((start, min(end, self.T)))

        logging.info(f"Loaded {self.T} frames, generated {len(self.indices)} clips.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start, end = self.indices[idx]
        clip = self.features[start:end]  # shape (L, 512)
        if clip.shape[0] < self.required_frames:
            pad_len = self.required_frames - clip.shape[0]
            pad = np.zeros((pad_len, 512), dtype=np.float32)
            clip = np.concatenate([clip, pad], axis=0)

        clip = clip[:self.required_frames]
        clip = torch.tensor(clip, dtype=torch.float32)  # (240, 512)
        clip = clip.unsqueeze(0).unsqueeze(0)  # (1, 1, 240, 512)
        return {"features": clip, "start_frame": start}

# ------------------------------
# Helper: peak picking (local maxima)
# ------------------------------
def peak_pick(scores_1d, threshold=0.2):
    peaks = []
    T = len(scores_1d)
    for t in range(1, T - 1):
        if scores_1d[t] > threshold and scores_1d[t] >= scores_1d[t-1] and scores_1d[t] >= scores_1d[t+1]:
            peaks.append((t, float(scores_1d[t])))
    return peaks

# ------------------------------
# Load checkpoint robustly
# ------------------------------
def try_load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = None
    if isinstance(ckpt, dict):
        for k in ("model_state_dict", "state_dict", "model_state", "state"):
            if k in ckpt:
                state_dict = ckpt[k]
                break
        if state_dict is None:
            candidate = {k:v for k,v in ckpt.items() if isinstance(v, torch.Tensor)}
            if candidate:
                state_dict = candidate
    else:
        state_dict = ckpt
    if state_dict is None:
        logging.warning("No state dict found in checkpoint; skipping loading weights.")
        return model
    # Clean keys if needed
    cleaned = {}
    for k,v in state_dict.items():
        newk = k.replace("module.", "")
        newk = newk.replace("_features.", "backbone.")
        cleaned[newk] = v
    model.load_state_dict(cleaned, strict=False)
    logging.info("Checkpoint loaded successfully.")
    return model

# ------------------------------
# Main inference
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to model config.py")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth/.tar)")
    parser.add_argument("--features", required=True, help="Path to PCA512 features (.npy)")
    parser.add_argument("--out_json", default="predictions.json", help="Output JSON filename")
    parser.add_argument("--chunk_size", type=int, default=120, help="Chunk size in seconds")
    parser.add_argument("--framerate", type=int, default=2, help="Feature framerate (fps)")
    parser.add_argument("--threshold", type=float, default=0.2, help="Peak detection threshold")
    parser.add_argument("--device", default="cpu", help="Device: cpu or cuda")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    # Load config and build model
    cfg = Config.fromfile(args.config)
    if hasattr(cfg, "model"):
        cfg_model = cfg.model
    else:
        raise RuntimeError("Config must define 'model'.")

    logging.info("Building model...")
    model = build_model(cfg_model)
    model = try_load_checkpoint(model, args.checkpoint)
    device = torch.device(args.device)
    model = model.to(device).eval()

    # Load dataset
    dataset = FullFeatureClipDataset(args.features, args.chunk_size, args.framerate)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Class names (try from config else fallback)
    if hasattr(cfg, "classes"):
        classes = list(cfg.classes)
    else:
        classes = ["Penalty", "Kick-off", "Goal", "Substitution", "Offside", "Shots on target", "Shots off target",
                   "Clearance", "Ball out of play", "Throw-in", "Foul", "Indirect free-kick", "Direct free-kick",
                   "Corner", "Yellow card", "Red card", "Yellow->red card"]
    n_classes = len(classes)

    all_events = []

    logging.info(f"Starting inference on {len(dataset)} clips...")

    for idx, sample in enumerate(loader):
        features = sample["features"].to(device)  # shape (1, 1, 240, 512)
        features = features.squeeze(1)

        start_frame = sample["start_frame"].item()

        # Model expects shape: (B, 1, T, 512)
        # Current shape (1, 1, 240, 512) — good to go

        # After model call
        outputs = model(features)
        print("Model output type:", type(outputs))
        if isinstance(outputs, tuple) or isinstance(outputs, list):
            for i, out in enumerate(outputs):
                print(f"Output[{i}] type: {type(out)}, shape: {getattr(out, 'shape', None)}")
        else:
            print(f"Output shape: {getattr(outputs, 'shape', None)}")
       
        if isinstance(outputs, (tuple, list)):
           for i, out in enumerate(outputs):
               print(f"Output[{i}] type: {type(out)}, shape: {out.shape}")
           # Use the first output and permute if needed
           scores = outputs[0]
           if scores.shape[1] == 120 and scores.shape[2] == n_classes:
              scores = scores.permute(0, 2, 1)  # to (B, C, T)
        else:
            scores = outputs
            if scores.shape[1] == 120 and scores.shape[2] == n_classes:
               scores = scores.permute(0, 2, 1)

        print(f"Scores tensor shape after adjustment: {scores.shape}")

        if scores.shape[1] != n_classes:
           raise RuntimeError("Scores tensor does not have the expected class dimension.")

     

        # probs = torch.sigmoid(scores).cpu().numpy()[0]  # (C, T)
        probs = torch.sigmoid(scores).detach().cpu().numpy()[0]


        # Peak pick per class
        for cls_idx in range(n_classes):
            cls_probs = probs[cls_idx]
            peaks = peak_pick(cls_probs, threshold=args.threshold)
            for t_idx, conf in peaks:
                # Calculate absolute frame index in full video timeline
                abs_frame = start_frame + t_idx
                timestamp_sec = abs_frame / args.framerate
                event = {
                    "label": classes[cls_idx],
                    "score": float(conf),
                    "timestamp": timestamp_sec,
                }
                all_events.append(event)

        logging.info(f"Processed clip {idx+1}/{len(dataset)}")

    # Sort events by timestamp
    all_events = sorted(all_events, key=lambda x: x["timestamp"])

    # Save to JSON
    with open(args.out_json, "w") as f:
        json.dump(all_events, f, indent=2)

    logging.info(f"Saved {len(all_events)} events to {args.out_json}")

if __name__ == "__main__":
    main()

