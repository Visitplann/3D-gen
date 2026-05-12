import cv2


def measure_contour_dimensions(contour):
    x, y, w, h = cv2.boundingRect(contour)
    rect = cv2.minAreaRect(contour)
    ((cx, cy), (rw, rh), angle) = rect
    return {
        "bbox_x": x,
        "bbox_y": y,
        "bbox_w": w,
        "bbox_h": h,
        "rotated_width": float(rw),
        "rotated_height": float(rh),
        "rotated_angle": float(angle),
    }


def infer_volumes(shapes, view_type):

  volumes = []

  is_top =  view_type == "top"
  is_side = view_type in ["left", "right", "front", "back"]

  # For complex shapes, limit the number of volumes per view
  max_volumes_per_view = 3 if is_top else 2

  for i, shape in enumerate(shapes[:max_volumes_per_view]):

    if cv2.contourArea(shape) < 500:  # Lower threshold for complex shapes
        continue

    x, y, w, h = cv2.boundingRect(shape)

    dims = measure_contour_dimensions(shape)

    if is_top:
      # For top view, create footprint volumes for each significant shape
      volume = {
          "type": "footprint",
          "contour": shape,
          "x": x,
          "y": y,
          "width": w,
          "depth": h,
          "bbox_w": dims["bbox_w"],
          "bbox_h": dims["bbox_h"],
          "rotated_width": dims["rotated_width"],
          "rotated_depth": dims["rotated_height"],
          "rotated_angle": dims["rotated_angle"],
          "shape_index": i  # Track which shape this is
      }
    elif is_side:
        # For side views, create profile volumes
        volume = {
            "type": "profile",
            "contour": shape,
            "height": h,
            "x": x,
            "y": y,
            "width": w,
            "bbox_w": dims["bbox_w"],
            "bbox_h": dims["bbox_h"],
            "rotated_width": dims["rotated_width"],
            "rotated_height": dims["rotated_height"],
            "rotated_angle": dims["rotated_angle"],
            "view": view_type,
            "shape_index": i
        }

    else:
        continue

    #DEBUG
    print(f"{view_type} contour {i} → x:{x}, y:{y}, w:{w}, h:{h}")
    #

    volumes.append(volume)

  return volumes