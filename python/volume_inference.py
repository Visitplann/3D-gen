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