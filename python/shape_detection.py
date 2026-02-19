import cv2

def detect_shapes(gray_img):
  
  edges = cv2.Canny(gray_img, 80, 160)
  
  contours,_ = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
  )
  
  #DEBUG
  debug_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
  cv2.drawContours(debug_img,contours,-1,(0,255,0),3)
  cv2.imshow("Debug Contours", debug_img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  #
  
  shapes = []
  for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 1000:
      continue
    
    
  
    approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
    shapes.append(approx)
    
  #Para um Unico Contour Solido em Vez de Dividido(subtitui o ciclo "for" acima, nao sei se funciona bem com o "mesh_builder" mas deixa a funcao texture_cutout mais pequena e deve funcionar bem com a funcao "height_map_to_normal_map" que inclusive talvez tenha de ser mudada de sitio)
  #largest = max(contours, key=cv2.contourArea)
  #approx = cv2.approxPolyDP(largest, 0.02 * cv2.arcLength(largest, True), True)
  #
  #return [approx]

  return shapes
  
#Para Recortar a Clean Image Baseada nos Contours de "detect_shapes" para a Texture
                  #clean image, monument shape(talvez monument edges em vez de shape mas tem de se ver a preview primeiro)
def texture_cutout(clean_img, mon_shape):
  h, w = clean_img.shape[:2]

  mask = np.zeros((h, w), dtype=np.uint8)

  #Preencher os shapes detectados
  cv2.drawContours(mask, shapes, -1, 255, thickness = cv2.FILLED)

  # Converte para RGBA
  rgba = cv2.cvtColor(clean_img, cv2.COLOR_RGB2RGBA)
  
  # Aplica alpha mask
  rgba[:, :, 3] = mask

  return rgba