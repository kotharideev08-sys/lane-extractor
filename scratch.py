# Lane Line Extractor - classical computer vision with OpenCV
#
# Pipeline (in the order the assignment asks for):
#   1. Load the road image
#   2. HSV colour filter (white + yellow paint)
#   3. Gaussian blur
#   4. Canny edge detection
#   5. Region of Interest (ROI) mask
#   + Hough lines -> draw the final left/right lane lines
#
# Run:  python lane_line_extractor_final.py
# You can type the path of ONE image or of a whole FOLDER of images.

import os
import cv2
import numpy as np

IMAGE_TYPES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
WORK_WIDTH = 960  # every image is resized to this width

# default settings (you can change them while the program runs)
DEFAULT_SETTINGS = {
    "horizon": 0.60,   # where the ROI top edge is (0 = top of image, 1 = bottom)
    "white_v": None,   # None = choose the white brightness threshold automatically
    "blur": 5,         # Gaussian blur kernel size (odd number)
    "canny_low": 50,
    "canny_high": 150,
}


# ---------------------------------------------------------------- helpers
def ask(prompt, default, cast=int):
    """Ask the user for a number. Press Enter to keep the default."""
    raw = input(f"  {prompt} [{default}]: ").strip()
    if raw == "":
        return default
    try:
        return cast(raw)
    except ValueError:
        print("  Not a valid number, keeping the default.")
        return default


def ask_settings(current):
    """Let the user change the settings."""
    s = dict(current)
    s["horizon"] = ask("ROI top height, 0.4 - 0.85 (bigger = lower)", s["horizon"], float)
    s["horizon"] = min(max(s["horizon"], 0.4), 0.85)

    raw = input("  White brightness 0-255 (Enter = auto): ").strip()
    if raw == "" or raw.lower() == "auto":
        s["white_v"] = None
    else:
        try:
            s["white_v"] = min(max(int(raw), 0), 255)
        except ValueError:
            print("  Not a valid number, using auto.")
            s["white_v"] = None

    s["blur"] = ask("Blur kernel size (odd)", s["blur"], int)
    if s["blur"] < 1:
        s["blur"] = 1
    if s["blur"] % 2 == 0:
        s["blur"] += 1  # kernel must be odd

    s["canny_low"] = ask("Canny low threshold", s["canny_low"], int)
    s["canny_high"] = ask("Canny high threshold", s["canny_high"], int)
    if s["canny_low"] > s["canny_high"]:  # low must be smaller than high
        s["canny_low"], s["canny_high"] = s["canny_high"], s["canny_low"]
    return s


def list_images(path):
    """Return a list of image files from a file path or a folder path."""
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        names = sorted(os.listdir(path))
        return [os.path.join(path, n) for n in names if n.lower().endswith(IMAGE_TYPES)]
    return []


def load_image(path):
    """Step 1: load the image and resize it to a standard width."""
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = WORK_WIDTH / w
    method = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    img = cv2.resize(img, (WORK_WIDTH, int(h * scale)), interpolation=method)
    if img.shape[0] > img.shape[1]:
        print("  Warning: this image is portrait. A landscape road photo works best.")
    return img


# ---------------------------------------------------------------- pipeline steps
def get_roi_points(w, h, horizon):
    """Trapezoid covering the road in front of the camera."""
    top_y = int(horizon * h)
    return np.array([[
        (int(0.05 * w), h),            # bottom left
        (int(0.42 * w), top_y),        # top left
        (int(0.58 * w), top_y),        # top right
        (int(0.95 * w), h),            # bottom right
    ]], dtype=np.int32)


def hsv_filter(img, roi_mask, white_v):
    """Step 2: keep only white and yellow pixels."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Automatic white threshold: lane paint is among the brightest pixels
    # of the road area, so look at the brightness of the road area only.
    if white_v is None:
        road_values = hsv[:, :, 2][roi_mask > 0]
        white_v = int(np.clip(np.percentile(road_values, 97), 150, 235))

    white_mask = cv2.inRange(hsv, np.array([0, 0, white_v]), np.array([179, 60, 255]))
    yellow_mask = cv2.inRange(hsv, np.array([15, 80, 120]), np.array([35, 255, 255]))
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    filtered = cv2.bitwise_and(img, img, mask=mask)
    return mask, filtered, white_v


def average_line(group):
    """Average many line pieces into one line (longer pieces count more)."""
    if len(group) == 0:
        return None
    arr = np.array(group)  # columns: slope, intercept, length
    slope = np.average(arr[:, 0], weights=arr[:, 2])
    intercept = np.average(arr[:, 1], weights=arr[:, 2])
    return slope, intercept


def draw_lane_lines(img, roi_edges, horizon_y):
    """Find line pieces with Hough, join them into a left and a right lane line."""
    h, w = img.shape[:2]
    lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, threshold=20,
                            minLineLength=20, maxLineGap=100)
    left, right = [], []

    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):  # works on every OpenCV version
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.3 or abs(slope) > 3:   # ignore flat / vertical pieces
                continue
            intercept = y1 - slope * x1
            length = np.hypot(x2 - x1, y2 - y1)
            mid_x = (x1 + x2) / 2
            # in an image the left lane line goes "up to the right" (negative slope)
            if slope < 0 and mid_x < 0.6 * w:
                left.append((slope, intercept, length))
            elif slope > 0 and mid_x > 0.4 * w:
                right.append((slope, intercept, length))

    left_fit = average_line(left)
    right_fit = average_line(right)

    # Stop the lines at the ROI top, or where they meet (vanishing point),
    # whichever is lower on the image. This prevents the "X" shape.
    y_top = horizon_y
    if left_fit is not None and right_fit is not None:
        m1, b1 = left_fit
        m2, b2 = right_fit
        x_vp = (b2 - b1) / (m1 - m2)
        y_vp = m1 * x_vp + b1
        if y_vp > y_top:
            y_top = int(y_vp + 0.03 * h)

    overlay = np.zeros_like(img)
    if y_top < 0.95 * h:
        for fit in (left_fit, right_fit):
            if fit is None:
                continue
            m, b = fit
            x_bottom = int((h - b) / m)
            x_top = int((y_top - b) / m)
            cv2.line(overlay, (x_bottom, h), (x_top, y_top), (0, 0, 255), 8)
    else:
        left_fit = right_fit = None  # lines made no sense, draw nothing

    result = cv2.addWeighted(img, 0.8, overlay, 1.0, 0)
    return result, left_fit is not None, right_fit is not None


def run_pipeline(img, s):
    h, w = img.shape[:2]
    roi_points = get_roi_points(w, h, s["horizon"])
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(roi_mask, roi_points, 255)

    # step 2: HSV filter
    color_mask, filtered, white_v = hsv_filter(img, roi_mask, s["white_v"])

    # step 3: Gaussian blur
    blurred = cv2.GaussianBlur(filtered, (s["blur"], s["blur"]), 0)

    # step 4: Canny edges
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, s["canny_low"], s["canny_high"])

    # step 5: ROI mask
    roi_edges = cv2.bitwise_and(edges, roi_mask)

    # extra: draw the lane lines
    result, found_left, found_right = draw_lane_lines(img, roi_edges, int(s["horizon"] * h))

    roi_preview = img.copy()
    cv2.polylines(roi_preview, roi_points, True, (0, 255, 0), 3)

    stages = {
        "1_original": img,
        "2_hsv_mask": color_mask,
        "3_hsv_filtered": filtered,
        "4_blurred": blurred,
        "5_canny_edges": edges,
        "6_roi_region": roi_preview,
        "7_roi_edges": roi_edges,
        "8_final_lanes": result,
    }
    info = {"white_v": white_v, "left": found_left, "right": found_right}
    return stages, info


# ---------------------------------------------------------------- output
def make_grid(stages):
    """One picture with all 8 stages (2 rows x 4 columns)."""
    first = stages["1_original"]
    tile_w = 480
    tile_h = int(tile_w * first.shape[0] / first.shape[1])
    tiles = []
    for name, im in stages.items():
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        tile = cv2.resize(im, (tile_w, tile_h))
        cv2.putText(tile, name[2:], (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)
        tiles.append(tile)
    return np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])


def save_results(stages, name):
    os.makedirs("output", exist_ok=True)
    for key, im in stages.items():
        cv2.imwrite(f"output/{name}_{key}.png", im)
    cv2.imwrite(f"output/{name}_pipeline_grid.png", make_grid(stages))
    print(f"  Saved to ./output/  (use {name}_pipeline_grid.png and {name}_8_final_lanes.png)")


def show_results(stages):
    try:
        cv2.imshow("Pipeline stages (press any key)", make_grid(stages))
        cv2.imshow("Final lanes", stages["8_final_lanes"])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        print("  Could not open a window (no display?). Check the output folder instead.")


# ---------------------------------------------------------------- main
def main():
    print("=== Lane Line Extractor ===")
    while True:
        path = input("\nEnter image or folder path (q to quit): ").strip().strip('"').strip("'")
        if path.lower() == "q":
            break
        files = list_images(os.path.expanduser(path))
        if len(files) == 0:
            print("  No image found there. Check the path and the file name.")
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
                print(f"  Left lane line:  {'found' if info['left'] else 'NOT found'}")
                print(f"  Right lane line: {'found' if info['right'] else 'NOT found'}")
                save_results(stages, name)
                show_results(stages)

                choice = input("Enter = next, r = retune this image, q = quit: ").strip().lower()
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