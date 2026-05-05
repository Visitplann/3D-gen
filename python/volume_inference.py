import cv2

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

    if is_top:
      # For top view, create footprint volumes for each significant shape
      volume = {
          "type": "footprint",
          "contour": shape,
          "x": x,
          "y": y,
          "width": w,
          "depth": h,
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