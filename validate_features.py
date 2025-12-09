# check_features.py
import numpy as np
import sys, os

def main(path):
    print("File:", path)
    if not os.path.exists(path):
        print(" -> NOT FOUND")
        return
    try:
        a = np.load(path, allow_pickle=False)
    except Exception as e:
        print("np.load failed (allow_pickle=False):", e)
        try:
            a = np.load(path, allow_pickle=True)
            print("Loaded with allow_pickle=True — WARNING: file contains object(s).")
        except Exception as e2:
            print("np.load failed with allow_pickle=True:", e2)
            return

    print("type:", type(a))
    try:
        print("shape:", a.shape)
        print("dtype:", a.dtype)
        print("ndim:", a.ndim)
        if a.ndim == 1:
            print("First element type:", type(a[0]))
    except Exception as e:
        print("Error inspecting array:", e)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_features.py <file.npy>")
    else:
        main(sys.argv[1])

