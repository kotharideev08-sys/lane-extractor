import os
import cv2
import numpy as np

IMAGE_TYPES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
WORK_WIDTH = 960

DEFAULT_SETTINGS = {
    "horizon": 0.60,       # top of road perspective region, fraction of image height
    "bottom_margin": 0.05, # left/right source points inset from image edges
    "white_v": None,       # None = estimate white brightness threshold
    "blur": 5,             # positive odd number
    "canny_low": 50,
    "canny_high": 150,
    "sliding_windows": 9,
    "window_margin": 70,
    "min_pixels": 30,
}


def ask(prompt, default, cast=int):
    raw = input(f"  {prompt} [{default}]: ").strip()
    if raw == "":
        return default
    try:
        return cast(raw)
    except ValueError:
        print("  Not a valid number, keeping the default.")
        return default


def ask_settings(current):
    s = dict(current)
    s["horizon"] = min(max(ask("Perspective top height, 0.40-0.85", s["horizon"], float), 0.40), 0.85)
    s["bottom_margin"] = min(max(ask("Bottom side inset, 0.00-0.20", s["bottom_margin"], float), 0.0), 0.20)

    raw = input("  White brightness 0-255 (Enter = auto): ").strip()
    if raw == "" or raw.lower() == "auto":
        s["white_v"] = None
    else:
        try:
            s["white_v"] = min(max(int(raw), 0), 255)
        except ValueError:
            print("  Not a valid number, using auto.")
            s["white_v"] = None

    s["blur"] = max(1, ask("Blur kernel size (odd)", s["blur"], int))
    if s["blur"] % 2 == 0:
        s["blur"] += 1

    s["canny_low"] = max(0, ask("Canny low threshold", s["canny_low"], int))
    s["canny_high"] = max(0, ask("Canny high threshold", s["canny_high"], int))
    if s["canny_low"] > s["canny_high"]:
        s["canny_low"], s["canny_high"] = s["canny_high"], s["canny_low"]

    s["sliding_windows"] = min(max(ask("Sliding windows (5-15)", s["sliding_windows"], int), 5), 15)
    s["window_margin"] = min(max(ask("Sliding-window half-width (30-150 px)", s["window_margin"], int), 30), 150)
    s["min_pixels"] = min(max(ask("Minimum lane pixels per window", s["min_pixels"], int), 5), 200)
    return s


def list_images(path):
    if os.path.isfile(path):
        return [path] if path.lower().endswith(IMAGE_TYPES) else []
    if os.path.isdir(path):
        return [
            os.path.join(path, name)
            for name in sorted(os.listdir(path))
            if name.lower().endswith(IMAGE_TYPES)
        ]
    return []


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = WORK_WIDTH / w
    new_h = max(1, int(h * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    img = cv2.resize(img, (WORK_WIDTH, new_h), interpolation=interpolation)
    if img.shape[0] < 240:
        print("  Warning: image is short; lane fitting may be less reliable.")
    return img


def perspective_points(w, h, horizon, bottom_margin):
    """Quadrilateral in the camera image, mapped to a rectangle."""
    top_y = int(horizon * h)
    left_bottom = int(bottom_margin * w)
    right_bottom = int((1.0 - bottom_margin) * w)
    top_left = int(0.40 * w)
    top_right = int(0.60 * w)

    src = np.float32([
        [left_bottom, h - 1],
        [top_left, top_y],
        [top_right, top_y],
        [right_bottom, h - 1],
    ])
    dst = np.float32([
        [int(0.20 * w), h - 1],
        [int(0.20 * w), 0],
        [int(0.80 * w), 0],
        [int(0.80 * w), h - 1],
    ])
    return src, dst


def make_perspective(img, horizon, bottom_margin):
    h, w = img.shape[:2]
    src, dst = perspective_points(w, h, horizon, bottom_margin)
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, matrix, (w, h), flags=cv2.INTER_LINEAR)
    return warped, src, dst, matrix


def hsv_filter(img, white_v):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if white_v is None:
        # Use the brighter part of the image, with a safe threshold range.
        white_v = int(np.clip(np.percentile(hsv[:, :, 2], 90), 150, 230))

    white_mask = cv2.inRange(
        hsv, np.array([0, 0, white_v]), np.array([179, 75, 255])
    )
    yellow_mask = cv2.inRange(
        hsv, np.array([12, 65, 100]), np.array([40, 255, 255])
    )
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    filtered = cv2.bitwise_and(img, img, mask=mask)
    return mask, filtered, white_v


def find_lane_pixels(binary, windows=9, margin=70, min_pixels=30):
    """Use a histogram and sliding windows to collect left/right lane pixels."""
    h, w = binary.shape
    histogram = np.sum(binary[h // 2:, :] > 0, axis=0)
    midpoint = w // 2
    left_base = int(np.argmax(histogram[:midpoint])) if midpoint else 0
    right_base = midpoint + int(np.argmax(histogram[midpoint:])) if midpoint < w else w - 1

    nonzero_y, nonzero_x = binary.nonzero()
    left_current, right_current = left_base, right_base
    window_height = max(1, h // windows)
    left_indices, right_indices = [], []
    debug = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    for window in range(windows):
        y_low = h - (window + 1) * window_height
        y_high = h - window * window_height
        y_low = max(0, y_low)
        xleft_low, xleft_high = left_current - margin, left_current + margin
        xright_low, xright_high = right_current - margin, right_current + margin

        cv2.rectangle(debug, (max(0, xleft_low), y_low),
                      (min(w - 1, xleft_high), y_high), (0, 255, 0), 2)
        cv2.rectangle(debug, (max(0, xright_low), y_low),
                      (min(w - 1, xright_high), y_high), (0, 255, 0), 2)

        good_left = ((nonzero_y >= y_low) & (nonzero_y < y_high) &
                     (nonzero_x >= xleft_low) & (nonzero_x < xleft_high)).nonzero()[0]
        good_right = ((nonzero_y >= y_low) & (nonzero_y < y_high) &
                      (nonzero_x >= xright_low) & (nonzero_x < xright_high)).nonzero()[0]
        left_indices.append(good_left)
        right_indices.append(good_right)

        if len(good_left) > min_pixels:
            left_current = int(np.mean(nonzero_x[good_left]))
        if len(good_right) > min_pixels:
            right_current = int(np.mean(nonzero_x[good_right]))

    left_indices = np.concatenate(left_indices) if left_indices else np.array([], dtype=int)
    right_indices = np.concatenate(right_indices) if right_indices else np.array([], dtype=int)

    left_x, left_y = nonzero_x[left_indices], nonzero_y[left_indices]
    right_x, right_y = nonzero_x[right_indices], nonzero_y[right_indices]
    return left_x, left_y, right_x, right_y, debug


def fit_polynomial(x, y, min_points=100):
    if len(x) < min_points or len(y) < min_points:
        return None
    try:
        fit = np.polyfit(y, x, 2)  # x = A*y^2 + B*y + C
        if not np.all(np.isfinite(fit)):
            return None
        return fit
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return None


def draw_curved_lanes(original, warped, matrix, left_fit, right_fit):
    """Draw fitted curves in bird's-eye view, then warp them onto the original image."""
    h, w = warped.shape[:2]
    plot_y = np.linspace(0, h - 1, h)
    overlay_warped = np.zeros_like(warped)
    left_found, right_found = left_fit is not None, right_fit is not None

    left_points = right_points = None
    if left_fit is not None:
        left_x = left_fit[0] * plot_y**2 + left_fit[1] * plot_y + left_fit[2]
        left_points = np.column_stack((left_x, plot_y)).astype(np.int32)
        cv2.polylines(overlay_warped, [left_points], False, (0, 0, 255), 8)
    if right_fit is not None:
        right_x = right_fit[0] * plot_y**2 + right_fit[1] * plot_y + right_fit[2]
        right_points = np.column_stack((right_x, plot_y)).astype(np.int32)
        cv2.polylines(overlay_warped, [right_points], False, (0, 0, 255), 8)

    # Map the bird's-eye overlay back to the camera perspective.
    inverse = np.linalg.inv(matrix)
    camera_overlay = cv2.warpPerspective(overlay_warped, inverse, (w, h))
    result = cv2.addWeighted(original, 1.0, camera_overlay, 0.85, 0)
    return result, left_found, right_found, overlay_warped


def run_pipeline(img, s):
    h, w = img.shape[:2]
    warped, src_points, dst_points, matrix = make_perspective(
        img, s["horizon"], s["bottom_margin"]
    )

    # Color mask is computed on the original image, then warped with the same transform.
    color_mask_original, _, white_v = hsv_filter(img, s["white_v"])
    warped_mask = cv2.warpPerspective(color_mask_original, matrix, (w, h))

    blurred = cv2.GaussianBlur(warped_mask, (s["blur"], s["blur"]), 0)
    edges = cv2.Canny(blurred, s["canny_low"], s["canny_high"])

    # Combine color evidence with edge evidence to reduce unrelated edges.
    binary = cv2.bitwise_or(warped_mask, edges)
    left_x, left_y, right_x, right_y, windows_debug = find_lane_pixels(
        binary, s["sliding_windows"], s["window_margin"], s["min_pixels"]
    )
    left_fit = fit_polynomial(left_x, left_y)
    right_fit = fit_polynomial(right_x, right_y)

    result, found_left, found_right, lane_overlay = draw_curved_lanes(
        img, warped, matrix, left_fit, right_fit
    )

    roi_preview = img.copy()
    cv2.polylines(roi_preview, [src_points.astype(np.int32)], True, (0, 255, 0), 3)

    # Show detected pixels and fitted curves in the warped view.
    fit_debug = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    plot_y = np.linspace(0, h - 1, h)
    if left_fit is not None:
        left_fit_x = left_fit[0] * plot_y**2 + left_fit[1] * plot_y + left_fit[2]
        pts = np.column_stack((left_fit_x, plot_y)).astype(np.int32)
        cv2.polylines(fit_debug, [pts], False, (0, 0, 255), 5)
    if right_fit is not None:
        right_fit_x = right_fit[0] * plot_y**2 + right_fit[1] * plot_y + right_fit[2]
        pts = np.column_stack((right_fit_x, plot_y)).astype(np.int32)
        cv2.polylines(fit_debug, [pts], False, (0, 0, 255), 5)

    stages = {
        "1_original": img,
        "2_perspective_region": roi_preview,
        "3_birdseye_view": warped,
        "4_hsv_mask": color_mask_original,
        "5_warped_lane_mask": warped_mask,
        "6_blurred_mask": blurred,
        "7_canny_edges": edges,
        "8_sliding_windows": windows_debug,
        "9_polynomial_fit": fit_debug,
        "10_final_curved_lanes": result,
    }
    info = {
        "white_v": white_v,
        "left": found_left,
        "right": found_right,
        "left_pixels": len(left_x),
        "right_pixels": len(right_x),
    }
    return stages, info


def make_grid(stages):
    first = next(iter(stages.values()))
    tile_w = 400
    tile_h = max(1, int(tile_w * first.shape[0] / first.shape[1]))
    tiles = []
    for name, im in stages.items():
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        tile = cv2.resize(im, (tile_w, tile_h))
        cv2.putText(tile, name[2:].replace("_", " "), (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        tiles.append(tile)

    cols = 5
    rows = []
    for start in range(0, len(tiles), cols):
        row = tiles[start:start + cols]
        while len(row) < cols:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def save_results(stages, name):
    os.makedirs("output", exist_ok=True)
    for key, im in stages.items():
        cv2.imwrite(os.path.join("output", f"{name}_{key}.png"), im)
    grid = make_grid(stages)
    cv2.imwrite(os.path.join("output", f"{name}_pipeline_grid.png"), grid)
    print(f"  Saved all stages to ./output/ ({name}_pipeline_grid.png)")


def show_results(stages):
    try:
        cv2.imshow("Curved lane pipeline stages (press any key)", make_grid(stages))
        cv2.imshow("Final curved lanes", stages["10_final_curved_lanes"])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        print("  Could not open a display window. Check the output folder instead.")


def main():
    print("=== Curved Lane Line Detector (Perspective + Sliding Windows + Polynomial Fit) ===")
    settings = dict(DEFAULT_SETTINGS)

    while True:
        path = input("\nEnter image or folder path (q to quit): ").strip().strip('"').strip("'")
        if path.lower() == "q":
            break
        files = list_images(os.path.expanduser(path))
        if not files:
            print("  No supported image found. Check the path and file extension.")
            continue

        settings = dict(DEFAULT_SETTINGS)
        if input("Customize settings? (y/N): ").strip().lower() == "y":
            settings = ask_settings(settings)

        quit_all = False
        for file in files:
            img = load_image(file)
            if img is None:
                print(f"  Skipping {file} (could not open it)")
                continue

            name = os.path.splitext(os.path.basename(file))[0]
            print(f"\nProcessing {os.path.basename(file)}")

            while True:
                stages, info = run_pipeline(img, settings)
                print(f"  White threshold used: {info['white_v']}")
                print(f"  Left curved lane:  {'found' if info['left'] else 'NOT found'} ({info['left_pixels']} pixels)")
                print(f"  Right curved lane: {'found' if info['right'] else 'NOT found'} ({info['right_pixels']} pixels)")
                save_results(stages, name)
                show_results(stages)

                choice = input("Enter = next image, r = retune this image, q = quit: ").strip().lower()
                if choice == "r":
                    settings = ask_settings(settings)
                    continue
                if choice == "q":
                    quit_all = True
                break

            if quit_all:
                break
        if quit_all:
            break


if __name__ == "__main__":
    main()
 