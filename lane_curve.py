import os
import cv2
import numpy as np

# ============================================================
# CURVED ROAD LANE DETECTION
# No perspective transformation
# No bird's-eye view
# Final overlay follows detected lane-color pixels
# ============================================================

IMAGE_TYPES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
WORK_WIDTH = 960

DEFAULT_SETTINGS = {
    "roi_top": 0.48,

    # Yellow lane marking
    "yellow_h_low": 12,
    "yellow_h_high": 42,
    "yellow_s_min": 65,
    "yellow_v_min": 65,

    # White lane marking
    "white_v_min": 155,
    "white_s_max": 95,

    # Noise reduction
    "blur": 5,
    "canny_low": 45,
    "canny_high": 140,

    # Overlay appearance
    "line_thickness": 3,
}


# ============================================================
# 1. USER SETTINGS
# ============================================================

def ask(prompt, default, cast=int):
    raw = input(f"  {prompt} [{default}]: ").strip()

    if not raw:
        return default

    try:
        return cast(raw)
    except ValueError:
        print("  Invalid input. Keeping the default.")
        return default


def ask_settings(current):
    settings = dict(current)

    settings["roi_top"] = float(np.clip(
        ask("ROI top fraction (0.25-0.70)",
            settings["roi_top"], float),
        0.25, 0.70
    ))

    settings["yellow_h_low"] = int(np.clip(
        ask("Yellow hue lower bound",
            settings["yellow_h_low"]),
        0, 179
    ))

    settings["yellow_h_high"] = int(np.clip(
        ask("Yellow hue upper bound",
            settings["yellow_h_high"]),
        0, 179
    ))

    settings["yellow_s_min"] = int(np.clip(
        ask("Yellow minimum saturation",
            settings["yellow_s_min"]),
        0, 255
    ))

    settings["yellow_v_min"] = int(np.clip(
        ask("Yellow minimum brightness",
            settings["yellow_v_min"]),
        0, 255
    ))

    settings["white_v_min"] = int(np.clip(
        ask("White minimum brightness",
            settings["white_v_min"]),
        0, 255
    ))

    settings["white_s_max"] = int(np.clip(
        ask("White maximum saturation",
            settings["white_s_max"]),
        0, 255
    ))

    settings["blur"] = max(
        1, ask("Gaussian blur kernel (odd)",
               settings["blur"])
    )

    if settings["blur"] % 2 == 0:
        settings["blur"] += 1

    settings["canny_low"] = max(
        0, ask("Canny low threshold",
               settings["canny_low"])
    )

    settings["canny_high"] = max(
        1, ask("Canny high threshold",
               settings["canny_high"])
    )

    if settings["canny_low"] > settings["canny_high"]:
        settings["canny_low"], settings["canny_high"] = (
            settings["canny_high"],
            settings["canny_low"]
        )

    settings["line_thickness"] = int(np.clip(
        ask("Overlay thickness",
            settings["line_thickness"]),
        1, 9
    ))

    return settings


# ============================================================
# 2. LOAD IMAGE
# ============================================================

def list_images(path):
    if os.path.isfile(path):
        if path.lower().endswith(IMAGE_TYPES):
            return [path]
        return []

    if os.path.isdir(path):
        return [
            os.path.join(path, filename)
            for filename in sorted(os.listdir(path))
            if filename.lower().endswith(IMAGE_TYPES)
        ]

    return []


def load_image(path):
    image = cv2.imread(path)

    if image is None:
        return None

    height, width = image.shape[:2]

    scale = min(1.0, WORK_WIDTH / width)

    if scale < 1.0:
        image = cv2.resize(
            image,
            (
                WORK_WIDTH,
                int(height * scale)
            ),
            interpolation=cv2.INTER_AREA
        )

    return image


# ============================================================
# 3. REGION OF INTEREST
# ============================================================

def create_roi(image, roi_top):
    """
    Select the lower portion of the image where the road is expected.

    This creates a mask only.
    It does not change the perspective or warp the image.
    """

    height, width = image.shape[:2]

    top_y = int(height * roi_top)

    points = np.array([[
        (0, height - 1),
        (int(width * 0.08), top_y),
        (int(width * 0.92), top_y),
        (width - 1, height - 1)
    ]], dtype=np.int32)

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    cv2.fillPoly(mask, points, 255)

    return mask, points


# ============================================================
# 4. HSV COLOR DETECTION
# ============================================================

def create_hsv_masks(image, settings):
    """
    Detect yellow and white separately.

    Yellow:
        Uses hue, saturation and brightness.

    White:
        Uses high brightness and low saturation.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # Yellow lane markings
    yellow_lower = np.array([
        settings["yellow_h_low"],
        settings["yellow_s_min"],
        settings["yellow_v_min"]
    ], dtype=np.uint8)

    yellow_upper = np.array([
        settings["yellow_h_high"],
        255,
        255
    ], dtype=np.uint8)

    yellow_mask = cv2.inRange(
        hsv,
        yellow_lower,
        yellow_upper
    )

    # White lane markings
    white_lower = np.array([
        0,
        0,
        settings["white_v_min"]
    ], dtype=np.uint8)

    white_upper = np.array([
        179,
        settings["white_s_max"],
        255
    ], dtype=np.uint8)

    white_mask = cv2.inRange(
        hsv,
        white_lower,
        white_upper
    )

    return hsv, yellow_mask, white_mask


# ============================================================
# 5. CLEAN COLOR MASKS
# ============================================================

def clean_mask(mask):
    """
    Remove small isolated noise and close tiny gaps.
    """

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    # Remove isolated pixels
    cleaned = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        open_kernel,
        iterations=1
    )

    # Connect small breaks in the lane paint
    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=1
    )

    return cleaned


# ============================================================
# 6. GAUSSIAN BLUR
# ============================================================

def apply_blur(image, kernel_size):
    return cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        0
    )


# ============================================================
# 7. CANNY EDGE DETECTION
# ============================================================

def detect_edges(image, settings):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blurred = apply_blur(
        gray,
        settings["blur"]
    )

    edges = cv2.Canny(
        blurred,
        settings["canny_low"],
        settings["canny_high"]
    )

    return blurred, edges


# ============================================================
# 8. DIRECT LANE OVERLAY
# ============================================================

def create_lane_overlay(
    image,
    yellow_mask,
    white_mask,
    roi_mask,
    thickness
):
    """
    Draw detected lane pixels directly on the original image.

    Important:
    No polynomial fitting is used here.
    No extrapolation is performed.

    Yellow lane -> yellow overlay
    White lane  -> red overlay

    The overlay follows the detected color pixels, including
    curved markings, rather than approximating them with a curve.
    """

    # Keep detections only inside the road ROI
    yellow = cv2.bitwise_and(
        yellow_mask,
        roi_mask
    )

    white = cv2.bitwise_and(
        white_mask,
        roi_mask
    )

    # Clean the masks
    yellow = clean_mask(yellow)
    white = clean_mask(white)

    # Optional slight thickening for visibility.
    # A thickness of 1 keeps the overlay closest to the detected pixels.
    if thickness > 1:
        kernel_size = 3

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size)
        )

        yellow = cv2.dilate(
            yellow,
            kernel,
            iterations=1
        )

        white = cv2.dilate(
            white,
            kernel,
            iterations=1
        )

        # Dilation can expand pixels outside the ROI.
        yellow = cv2.bitwise_and(
            yellow,
            roi_mask
        )

        white = cv2.bitwise_and(
            white,
            roi_mask
        )

    # Create a transparent color layer
    overlay = np.zeros_like(image)

    # Yellow marking: BGR yellow
    overlay[yellow > 0] = (0, 255, 255)

    # White marking: red, for visual distinction
    overlay[white > 0] = (0, 0, 255)

    # Blend only the detected pixels.
    # All other pixels remain unchanged.
    result = image.copy()

    detected = (yellow > 0) | (white > 0)

    if np.any(detected):
        result[detected] = cv2.addWeighted(
            image[detected],
            0.25,
            overlay[detected],
            0.75,
            0
        )

    return result, yellow, white


# ============================================================
# 9. COMPLETE PIPELINE
# ============================================================

def run_pipeline(image, settings):

    # Step 1: ROI
    roi_mask, roi_points = create_roi(
        image,
        settings["roi_top"]
    )

    # Step 2: HSV masks
    hsv, yellow_raw, white_raw = create_hsv_masks(
        image,
        settings
    )

    # Step 3: Clean color masks
    yellow_clean = clean_mask(yellow_raw)
    white_clean = clean_mask(white_raw)

    # Step 4: Restrict masks to the road region
    yellow_roi = cv2.bitwise_and(
        yellow_clean,
        roi_mask
    )

    white_roi = cv2.bitwise_and(
        white_clean,
        roi_mask
    )

    combined_mask = cv2.bitwise_or(
        yellow_roi,
        white_roi
    )

    # Step 5: HSV-filtered image
    hsv_filtered = cv2.bitwise_and(
        image,
        image,
        mask=combined_mask
    )

    # Step 6: Blur
    blurred = apply_blur(
        hsv_filtered,
        settings["blur"]
    )

    # Step 7: Canny edges
    gray = cv2.cvtColor(
        blurred,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.Canny(
        gray,
        settings["canny_low"],
        settings["canny_high"]
    )

    # Step 8: Restrict edges to ROI
    roi_edges = cv2.bitwise_and(
        edges,
        roi_mask
    )

    # Step 9: Display lane candidates
    lane_candidates = cv2.bitwise_or(
        yellow_roi,
        white_roi
    )

    # Step 10: Show ROI boundary
    roi_preview = image.copy()

    cv2.polylines(
        roi_preview,
        roi_points,
        True,
        (0, 255, 0),
        2
    )

    # Step 11: Direct overlay on detected lane pixels
    final, yellow_line, white_line = create_lane_overlay(
        image,
        yellow_clean,
        white_clean,
        roi_mask,
        settings["line_thickness"]
    )

    # Diagnostic image:
    # Yellow detected pixels = yellow
    # White detected pixels = white
    tracking_pixels = np.zeros_like(image)

    tracking_pixels[yellow_line > 0] = (
        0, 255, 255
    )

    tracking_pixels[white_line > 0] = (
        255, 255, 255
    )

    stages = {
        "1_original": image,
        "2_hsv_mask": combined_mask,
        "3_hsv_filtered": hsv_filtered,
        "4_blurred": blurred,
        "5_canny_edges": edges,
        "6_roi_region": roi_preview,
        "7_roi_edges": roi_edges,
        "8_lane_candidates": lane_candidates,
        "9_tracking_pixels": tracking_pixels,
        "10_detected_lanes": cv2.bitwise_or(
            yellow_line,
            white_line
        ),
        "11_final_lanes": final
    }

    info = {
        "yellow_pixels": cv2.countNonZero(
            yellow_line
        ),
        "white_pixels": cv2.countNonZero(
            white_line
        )
    }

    return stages, info


# ============================================================
# 10. CREATE PIPELINE GRID
# ============================================================

def make_grid(stages):
    """
    Display all stages in a grid.
    """

    first = next(iter(stages.values()))

    tile_width = 320

    tile_height = max(
        1,
        int(
            tile_width
            * first.shape[0]
            / first.shape[1]
        )
    )

    tiles = []

    for name, image in stages.items():

        if image.ndim == 2:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2BGR
            )

        tile = cv2.resize(
            image,
            (tile_width, tile_height)
        )

        label = name[2:].replace("_", " ")

        cv2.putText(
            tile,
            label,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )

        tiles.append(tile)

    columns = 4
    rows = []

    for start in range(0, len(tiles), columns):

        row = tiles[start:start + columns]

        while len(row) < columns:
            row.append(
                np.zeros_like(tiles[0])
            )

        rows.append(
            np.hstack(row)
        )

    return np.vstack(rows)


# ============================================================
# 11. SAVE RESULTS
# ============================================================

def save_results(stages, name):

    os.makedirs(
        "output",
        exist_ok=True
    )

    for key, image in stages.items():

        output_path = os.path.join(
            "output",
            f"{name}_{key}.png"
        )

        cv2.imwrite(
            output_path,
            image
        )

    grid = make_grid(stages)

    cv2.imwrite(
        os.path.join(
            "output",
            f"{name}_pipeline_grid.png"
        ),
        grid
    )

    print(
        f"  Results saved in ./output/"
    )


# ============================================================
# 12. DISPLAY RESULTS
# ============================================================

def show_results(stages):

    try:
        grid = make_grid(stages)

        cv2.imshow(
            "Lane Detection Pipeline",
            grid
        )

        cv2.imshow(
            "Final Lane Detection",
            stages["11_final_lanes"]
        )

        print(
            "  Press any key inside an image window to continue."
        )

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except cv2.error:
        print(
            "  Could not open image windows."
        )

        print(
            "  Check the saved images in the output folder."
        )


# ============================================================
# 13. MAIN PROGRAM
# ============================================================

def main():

    print("=" * 55)
    print(" CURVED ROAD LANE DETECTOR ")
    print(" No Perspective Transformation ")
    print(" Direct Lane-Pixel Overlay ")
    print("=" * 55)

    while True:

        path = input(
            "\nEnter image path or folder path (q to quit): "
        ).strip().strip('"').strip("'")

        if path.lower() == "q":
            break

        files = list_images(
            os.path.expanduser(path)
        )

        if not files:
            print(
                "No supported images found. Check the path."
            )
            continue

        settings = dict(
            DEFAULT_SETTINGS
        )

        customize = input(
            "Customize settings? (y/N): "
        ).strip().lower()

        if customize == "y":
            settings = ask_settings(
                settings
            )

        quit_all = False

        for file_path in files:

            image = load_image(
                file_path
            )

            if image is None:
                print(
                    f"Could not read: {file_path}"
                )
                continue

            name = os.path.splitext(
                os.path.basename(file_path)
            )[0]

            while True:

                stages, info = run_pipeline(
                    image,
                    settings
                )

                print(
                    f"\nProcessing: {os.path.basename(file_path)}"
                )

                print(
                    f"Yellow lane pixels: {info['yellow_pixels']}"
                )

                print(
                    f"White lane pixels: {info['white_pixels']}"
                )

                save_results(
                    stages,
                    name
                )

                show_results(
                    stages
                )

                choice = input(
                    "\nEnter = next image, "
                    "r = change settings, "
                    "q = quit: "
                ).strip().lower()

                if choice == "r":
                    settings = ask_settings(
                        settings
                    )
                    continue

                if choice == "q":
                    quit_all = True

                break

            if quit_all:
                break

        if quit_all:
            break

    print("Lane detection program closed.")


if __name__ == "__main__":
    main()