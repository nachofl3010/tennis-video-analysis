import cv2
import numpy as np
import supervision as sv
from typing import Tuple, Optional, List

# --- Configuration ---
class TennisCourtConfiguration:
    def __init__(self):
        # Standard Tennis Court Dimensions (in meters)
        self.court_length = 23.77
        self.court_width = 10.97  # Doubles width
        self.singles_width = 8.23
        self.service_line_dist = 6.40  # Distance from net to service line
        self.alley_width = (self.court_width - self.singles_width) / 2
        self.net_position = self.court_length / 2

# --- Helper: Coordinate Transformation ---
def _to_pixel(
    point: Tuple[float, float],
    scale: float,
    padding: int,
) -> Tuple[int, int]:
    """
    Converts a real-world (x, y) point in meters to pixel coordinates (u, v).
    """
    return (
        int(round(point[1] * scale + padding)),
        int(round(point[0] * scale + padding)),
    )

# --- Core Drawing Function ---
def draw_court(
    config: TennisCourtConfiguration = TennisCourtConfiguration(),
    scale: float = 50,
    padding: int = 50,
    line_thickness: int = 4,
    line_color: sv.Color = sv.Color.WHITE,
    court_color: sv.Color = sv.Color.from_hex("#5072A7"),
    outside_color: sv.Color = sv.Color.from_hex("#3C9E6A"),
) -> np.ndarray:
    
    # --- UPDATED: Image Dimensions Swapped ---
    # Width of image corresponds to Court Width
    # Height of image corresponds to Court Length
    image_cols = int(round(config.court_width * scale))
    image_rows = int(round(config.court_length * scale))
    
    image = np.zeros(
        (image_rows + 2 * padding, image_cols + 2 * padding, 3),
        dtype=np.uint8,
    )
    image[:, :] = outside_color.as_bgr()

    # Define standard court rectangle (Top-Left to Bottom-Right)
    # Note: _to_pixel handles the swapping, so we just pass (Length, Width) as usual
    p_top_left = _to_pixel((0, 0), scale, padding)
    p_bottom_right = _to_pixel((config.court_length, config.court_width), scale, padding)
    
    cv2.rectangle(image, p_top_left, p_bottom_right, court_color.as_bgr(), -1)

    # --- Draw Lines (Logic remains (x,y) = (length, width)) ---
    
    # 1. Outer Boundary
    cv2.rectangle(image, p_top_left, p_bottom_right, line_color.as_bgr(), line_thickness)

    # 2. Singles Sidelines
    p1 = _to_pixel((0, config.alley_width), scale, padding)
    p2 = _to_pixel((config.court_length, config.alley_width), scale, padding)
    cv2.line(image, p1, p2, line_color.as_bgr(), line_thickness)
    
    p3 = _to_pixel((0, config.court_width - config.alley_width), scale, padding)
    p4 = _to_pixel((config.court_length, config.court_width - config.alley_width), scale, padding)
    cv2.line(image, p3, p4, line_color.as_bgr(), line_thickness)

    # 3. Service Lines
    service_line_x_left = config.net_position - config.service_line_dist
    service_line_x_right = config.net_position + config.service_line_dist
    
    sl_l_top = _to_pixel((service_line_x_left, config.alley_width), scale, padding)
    sl_l_bot = _to_pixel((service_line_x_left, config.court_width - config.alley_width), scale, padding)
    cv2.line(image, sl_l_top, sl_l_bot, line_color.as_bgr(), line_thickness)

    sl_r_top = _to_pixel((service_line_x_right, config.alley_width), scale, padding)
    sl_r_bot = _to_pixel((service_line_x_right, config.court_width - config.alley_width), scale, padding)
    cv2.line(image, sl_r_top, sl_r_bot, line_color.as_bgr(), line_thickness)

    # 4. Center Service Line
    center_y = config.court_width / 2
    c_start = _to_pixel((service_line_x_left, center_y), scale, padding)
    c_end = _to_pixel((service_line_x_right, center_y), scale, padding)
    cv2.line(image, c_start, c_end, line_color.as_bgr(), line_thickness)

    # 5. Center Marks
    mark_len = 0.15
    m_l_start = _to_pixel((0, center_y), scale, padding)
    m_l_end = _to_pixel((mark_len, center_y), scale, padding)
    cv2.line(image, m_l_start, m_l_end, line_color.as_bgr(), line_thickness)

    m_r_start = _to_pixel((config.court_length, center_y), scale, padding)
    m_r_end = _to_pixel((config.court_length - mark_len, center_y), scale, padding)
    cv2.line(image, m_r_start, m_r_end, line_color.as_bgr(), line_thickness)

    # 6. The Net
    net_top = _to_pixel((config.net_position, -0.9), scale, padding)
    net_bot = _to_pixel((config.net_position, config.court_width + 0.9), scale, padding)
    cv2.line(image, net_top, net_bot, (200, 200, 200), line_thickness + 2)

    return image

# --- Drawing Utilities (Points & Paths) ---

def draw_points_on_court(
    xy: np.ndarray,
    image: Optional[np.ndarray] = None,
    color: sv.Color = sv.Color.from_hex("#FF1493"), # Hot pink default
    radius: int = 10,
    config: TennisCourtConfiguration = TennisCourtConfiguration(),
    scale: float = 50,
    padding: int = 50,
) -> np.ndarray:
    """Draw points on the court given an array of (x, y) coordinates."""
    if image is None:
        image = draw_court(config=config, scale=scale, padding=padding)

    if xy is None or len(xy) == 0:
        return image

    # Ensure xy is a 2D array
    points = np.atleast_2d(xy)

    for point in points:
        center = _to_pixel(point, scale, padding)
        cv2.circle(image, center, radius, color.as_bgr(), -1, cv2.LINE_AA)
        
    return image

def draw_paths_on_court(
    paths: List[np.ndarray],
    image: Optional[np.ndarray] = None,
    color: sv.Color = sv.Color.YELLOW,
    thickness: int = 2,
    config: TennisCourtConfiguration = TennisCourtConfiguration(),
    scale: float = 50,
    padding: int = 50,
) -> np.ndarray:
    """Draw trajectories/paths on the court."""
    if image is None:
        image = draw_court(config=config, scale=scale, padding=padding)

    for path in paths:
        if path is None or len(path) < 2:
            continue
            
        # Convert all points in the path to pixels
        pixel_points = []
        for point in path:
            pixel_points.append(_to_pixel(point, scale, padding))
            
        # Draw the polyline
        pts = np.array(pixel_points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], False, color.as_bgr(), thickness, cv2.LINE_AA)

    return image

# --- Example Usage (If you run this script directly) ---
if __name__ == "__main__":
    # 1. Create a blank court
    court_img = draw_court()

    # 2. Define some points (meters)
    # Point A: Near left baseline service box
    # Point B: Near right net
    my_points = np.array([
        [2.0, 5.0],  
        [12.0, 9.0]
    ])
    
    # 3. Define a path (movement from A to B)
    my_path = np.array([
        [2.0, 5.0],
        [4.0, 6.0],
        [8.0, 7.5],
        [12.0, 9.0]
    ])

    # 4. Draw them
    court_img = draw_paths_on_court([my_path], image=court_img)
    court_img = draw_points_on_court(my_points, image=court_img)

    # 5. Display
    cv2.imshow("Tennis Court", court_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
