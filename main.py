import cv2
import os
import numpy as np
import argparse

import utils as sutils


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def solve_sudoku_puzzle(loaded_model, img_fpath):
    """Solve a single sudoku image and return the annotated result image (RGB).

    Returns None if the grid cannot be detected or the puzzle cannot be solved.
    """
    # Check for valid filepath because cv2.imread fails silently
    if not os.path.exists(img_fpath):
        print(f"  [SKIP] File not found: '{img_fpath}'")
        return None

    # Load image, change color space from BGR to RGB, and resize
    img = cv2.imread(img_fpath)
    if img is None:
        print(f"  [SKIP] cv2.imread failed to read: '{img_fpath}'")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = sutils.resize_and_maintain_aspect_ratio(input_image=img, new_width=1000)

    # Load the digit-classification model (ONNX format)
    # loaded_model = sutils.OnnxModel(model_fpath)

    # Locate grid cells in image
    detection = sutils.get_valid_cells_from_image(img)
    if detection is None:
        print("  [FAIL] Could not locate a valid sudoku grid in the image.")
        return None
    cells, M, board_image = detection

    # Get the 2D array of the puzzle grid to be passed to the solver
    grid_array = sutils.get_predicted_sudoku_grid(loaded_model, cells)

    # Create an instance of SudokuSolver and try to solve the puzzle
    solver = sutils.SudokuSolver(board=grid_array)
    solver.solve()

    # If there are no zeros left, the puzzle is solved
    if np.any(solver.board == 0):
        print("  [FAIL] Could not solve the puzzle. Check for misclassified digits.")
        return None

    # Get the image of the board annotated with the solution
    print("  [OK] Sudoku solved.")
    return sutils.generate_solution_image(img, board_image, cells, solver.board, M)


def save_result_image(annotated_img, out_fpath):
    """Save the annotated (RGB) image to disk as a file, no popup display."""
    bgr_img = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
    return cv2.imwrite(out_fpath, bgr_img)


def main():
    # Construct an argument parser and parse the arguments
    ap = argparse.ArgumentParser(description="Batch sudoku solver: detect, solve and save results for every image in an input directory.")
    ap.add_argument("--input_dir", type=str, required=True, help="Directory containing sudoku images to process")
    ap.add_argument("--output_dir", type=str, required=True, help="Directory where annotated result images will be saved")
    ap.add_argument("--model_fpath", default="src_onnx/model.onnx", type=str, help="Path to digit-classification model in ONNX format")
    args = vars(ap.parse_args())

    input_dir = args["input_dir"]
    output_dir = args["output_dir"]
    model_fpath = args["model_fpath"]

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: '{input_dir}'")
    if not os.path.exists(model_fpath):
        raise FileNotFoundError(f"Model file not found: '{model_fpath}'")

    # Create the output directory if it does not exist yet
    os.makedirs(output_dir, exist_ok=True)

    # Collect all image files (non-recursive) from the input directory
    img_fpaths = sorted(
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    )
    if not img_fpaths:
        print(f"No images found in '{input_dir}'.")
        return

    print(f"Found {len(img_fpaths)} image(s) in '{input_dir}'.")
    print(f"Results will be saved to '{output_dir}'.\n")

    n_solved = 0
    n_failed = 0
    loaded_model = sutils.OnnxModel(model_fpath)
    
    for i, img_fpath in enumerate(img_fpaths, start=1):
        filename = os.path.basename(img_fpath)
        out_fpath = os.path.join(output_dir, filename)
        print(f"[{i}/{len(img_fpaths)}] Processing '{filename}'...")

        try:
            annotated_img = solve_sudoku_puzzle(loaded_model, img_fpath)
        except Exception as exc:
            print(f"  [ERROR] Unexpected error: {exc}")
            annotated_img = None

        if annotated_img is not None:
            if save_result_image(annotated_img, out_fpath):
                print(f"  [SAVED] '{out_fpath}'")
                n_solved += 1
            else:
                print(f"  [ERROR] cv2.imwrite failed to write '{out_fpath}'.")
                n_failed += 1
        else:
            n_failed += 1

    print(f"\nDone. Solved and saved: {n_solved}, failed/skipped: {n_failed}.")


if __name__ == "__main__":
    main()
