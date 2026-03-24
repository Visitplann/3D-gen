import cv2

def infer_volumes(shapes, view):
  
  volumes = []
  
  for shapes in shapes:
    
    x,y,w,h = cv2.boundingRect(shapes)
    
    volume = {
      "type" : "box",
      "view" : view, 
      "width" : w,
      "height" : h,
      "depth": w*0.6,
      "x": x,
      "y": y
    }
    
    volumes.append(volume)
    
  return volumes


"""import cv2

def infer_volumes(shapes, view_type):

  volumes = []

  is_top =  view_type == "top"
  is_side = view_type in ["left", "right", "front", "back"]

  for shape in shapes:

    if cv2.contourArea(shape) < 1000:
        continue

    if is_top:
      volume = {
          "type": "footprint",
          "contour": shape
      }
    elif is_side:
      _, _, _, h = cv2.boundingRect(shape)

      volume = {
          "type": "profile",
          "height": h,
          "contour": shape
      }
    else:
      # fallback
      continue

    volumes.append(volume)

  return volumes"""