def draw_lane_lines(img, roi_edges, horizon_y):
    """
    Detect straight and curved lane markings using sliding windows
    and second-degree polynomial fitting.
    """
    h, w = img.shape[:2]

    # Find edge pixels
    nonzero_y, nonzero_x = roi_edges.nonzero()

    # Search for lane markings in the lower half of the image
    histogram = np.sum(roi_edges[h // 2:, :], axis=0)

    midpoint = w // 2
    left_base = np.argmax(histogram[:midpoint]) if midpoint > 0 else 0
    right_base = (
        midpoint + np.argmax(histogram[midpoint:])
        if midpoint < w else w - 1
    )

    # Sliding-window parameters
    nwindows = 9
    margin = max(40, int(w * 0.08))
    minpix = 20

    window_height = max(1, (h - horizon_y) // nwindows)

    left_current = int(left_base)
    right_current = int(right_base)

    left_indices = []
    right_indices = []

    # Search upward from the bottom of the image
    for window in range(nwindows):
        y_high = h - window * window_height
        y_low = max(horizon_y, y_high - window_height)

        # Left and right window boundaries
        left_low = max(0, left_current - margin)
        left_high = min(w, left_current + margin)

        right_low = max(0, right_current - margin)
        right_high = min(w, right_current + margin)

        # Find pixels inside each window
        good_left = (
            (nonzero_y >= y_low) &
            (nonzero_y < y_high) &
            (nonzero_x >= left_low) &
            (nonzero_x < left_high)
        )

        good_right = (
            (nonzero_y >= y_low) &
            (nonzero_y < y_high) &
            (nonzero_x >= right_low) &
            (nonzero_x < right_high)
        )

        left_found = np.where(good_left)[0]
        right_found = np.where(good_right)[0]

        left_indices.append(left_found)
        right_indices.append(right_found)

        # Recenter the windows when enough pixels are detected
        if len(left_found) >= minpix:
            left_current = int(
                np.mean(nonzero_x[left_found])
            )

        if len(right_found) >= minpix:
            right_current = int(
                np.mean(nonzero_x[right_found])
            )

    # Combine pixels found in all windows
    left_indices = np.concatenate(left_indices)
    right_indices = np.concatenate(right_indices)

    left_x = nonzero_x[left_indices]
    left_y = nonzero_y[left_indices]

    right_x = nonzero_x[right_indices]
    right_y = nonzero_y[right_indices]

    # Require enough points for a reliable fit
    min_points = max(30, int(h * 0.08))

    left_fit = None
    right_fit = None

    # Fit x = a*y^2 + b*y + c
    # A quadratic polynomial can represent curved lane markings.
    if len(left_x) >= min_points:
        left_fit = np.polyfit(left_y, left_x, 2)

    if len(right_x) >= min_points:
        right_fit = np.polyfit(right_y, right_x, 2)

    # Draw smooth lane curves
    overlay = np.zeros_like(img)

    y_values = np.linspace(
        max(0, min(horizon_y, h - 1)),
        h - 1,
        100
    ).astype(np.int32)

    def draw_curve(fit, color):
        if fit is None:
            return

        # Calculate the x position at every y position
        x_values = (
            fit[0] * y_values**2 +
            fit[1] * y_values +
            fit[2]
        )

        points = np.column_stack(
            (x_values, y_values)
        ).astype(np.int32)

        # Keep points inside the image
        valid = (
            (points[:, 0] >= 0) &
            (points[:, 0] < w) &
            (points[:, 1] >= 0) &
            (points[:, 1] < h)
        )

        points = points[valid]

        if len(points) >= 2:
            cv2.polylines(
                overlay,
                [points.reshape(-1, 1, 2)],
                False,
                color,
                8,
                cv2.LINE_AA
            )

    # Red left and right lane curves
    draw_curve(left_fit, (0, 0, 255))
    draw_curve(right_fit, (0, 0, 255))

    # Combine the detected curves with the original image
    result = cv2.addWeighted(
        img, 0.8,
        overlay, 1.0,
        0
    )

    return (
        result,
        left_fit is not None,
        right_fit is not None
    )
