import cv2

def infer_volumes(shapes, view_type):

  volumes = []

  is_top =  view_type == "top"
  is_side = view_type in ["left", "right", "front", "back"]
  
  
  
  for shape in shapes:

    if cv2.contourArea(shape) < 1000:
        continue
      
    x, y, w, h = cv2.boundingRect(shape)
    x, y, w, h = cv2.boundingRect(shape)
    if is_top:
      volume = {
          "type": "footprint",
          "contour": shape,
          "x": x,
          "y": y,
          "width": w,
          "depth": h
      }
    elif is_side:
        volume = {
            "type": "profile",
            "contour": shape,
            "height": h,
            "x": x, 
            "y": y,
            "width": w,
            "view": view_type  
        }

    else:
        continue
    
    #DEBUG
    print(f"{view_type} contour → x:{x}, y:{y}, w:{w}, h:{h}")
    #
    
    volumes.append(volume)

  return volumes