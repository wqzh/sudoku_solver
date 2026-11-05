import argparse
import os
import time

import cv2
import numpy as np

import utils as sutils


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def read_image_rgb(fpath):
    """Read an image as RGB. Uses imdecode so non-ASCII Windows paths work."""
    try:
        data = np.fromfile(fpath, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_image_rgb(fpath, rgb_img):
    """Write an RGB image to disk (works with non-ASCII Windows paths)."""
    bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    ext = os.path.splitext(fpath)[1] or ".png"
    ok, buf = cv2.imencode(ext, bgr)
    if not ok:
        return False
    try:
        buf.tofile(fpath)
    except OSError:
        return False
    return True


def solve_sudoku_puzzle(loaded_model, img_fpath, time_budget=20.0):
    """Solve a single sudoku image. Returns a sutils.SolveResult."""
    if not os.path.exists(img_fpath):
        return sutils.SolveResult(message=f"file not found: '{img_fpath}'")

    img = read_image_rgb(img_fpath)
    if img is None:
        return sutils.SolveResult(message=f"cannot decode image: '{img_fpath}'")

    img = sutils.resize_and_maintain_aspect_ratio(img, new_width=1000)
    return sutils.solve_sudoku_image(loaded_model, img, time_budget=time_budget)


def main():
    ap = argparse.ArgumentParser(
        description="Batch sudoku solver: detect, solve and save results for every image in a directory.")
    ap.add_argument("--input_dir", type=str, required=True,
                    help="Directory containing sudoku images to process")
    ap.add_argument("--output_dir", type=str, required=True,
                    help="Directory where annotated result images will be saved")
    ap.add_argument("--model_fpath", default="src_onnx/model.onnx", type=str,
                    help="Path to digit-classification model in ONNX format")
    ap.add_argument("--debug_dir", default=None, type=str,
                    help="Optional: save a debug picture (warped grid + detected lines + "
                         "recognised digits) for every image that fails")
    ap.add_argument("--time_budget", default=20.0, type=float,
                    help="Max seconds spent on a single image (guards against hangs)")
    args = ap.parse_args()

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: '{args.input_dir}'")
    if not os.path.exists(args.model_fpath):
        raise FileNotFoundError(f"Model file not found: '{args.model_fpath}'")

    os.makedirs(args.output_dir, exist_ok=True)
    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)

    img_fpaths = sorted(
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    )
    if not img_fpaths:
        print(f"No images found in '{args.input_dir}'.")
        return

    print(f"Found {len(img_fpaths)} image(s) in '{args.input_dir}'.")
    print(f"Results will be saved to '{args.output_dir}'.\n")

    loaded_model = sutils.OnnxModel(args.model_fpath)

    n_solved = 0
    failed = []  # (filename, reason)

    try:
        for i, img_fpath in enumerate(img_fpaths, start=1):
            if i==5: break
            filename = os.path.basename(img_fpath)
            out_fpath = os.path.join(args.output_dir, filename)
            print(f"[{i}/{len(img_fpaths)}] Processing '{filename}'...", flush=True)

            t0 = time.time()
            try:
                result = solve_sudoku_puzzle(loaded_model, img_fpath, args.time_budget)
            except Exception as exc:  # keep the batch going no matter what
                result = sutils.SolveResult(message=f"unexpected error: {exc!r}")
            dt = time.time() - t0

            if result.ok:
                if write_image_rgb(out_fpath, result.annotated):
                    print(f"  [OK] {result.message} -> '{out_fpath}' ({dt:.1f}s)")
                    n_solved += 1
                else:
                    print(f"  [ERROR] failed to write '{out_fpath}'")
                    failed.append((filename, "write failed"))
            else:
                print(f"  [FAIL] {result.message} ({dt:.1f}s)")
                failed.append((filename, result.message))
                if args.debug_dir and result.debug_image is not None:
                    stem = os.path.splitext(filename)[0]
                    write_image_rgb(os.path.join(args.debug_dir, stem + "_debug.png"),
                                    result.debug_image)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    print(f"\nDone. Solved and saved: {n_solved}, failed/skipped: {len(failed)}.")
    if failed:
        print("Failed files:")
        for name, reason in failed:
            print(f"  - {name}: {reason}")

if __name__ == "__main__":
    main()