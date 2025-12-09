#!/usr/bin/env python3
import numpy as np
import pickle as pkl
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input raw ResNet features (.npy) shape (T,2048)")
    parser.add_argument("--output", required=True, help="output PCA 512D features (.npy)")
    parser.add_argument("--pca", required=True, help="pca_512_TF2.pkl")
    parser.add_argument("--mean", required=True, help="average_512_TF2.pkl")
    args = parser.parse_args()

    print("Loading features:", args.input)
    feats = np.load(args.input)
    print("  shape:", feats.shape)

    print("Loading PCA:", args.pca)
    with open(args.pca, "rb") as f:
        pca = pkl.load(f)

    print("Loading mean:", args.mean)
    with open(args.mean, "rb") as f:
        mean = pkl.load(f)

    print("Applying mean normalization...")
    feats_norm = feats - mean

    print("Applying PCA transform...")
    feats_512 = pca.transform(feats_norm)

    print("Saving:", args.output)
    np.save(args.output, feats_512)

    print("Done. Final shape:", feats_512.shape)

if __name__ == "__main__":
    main()

