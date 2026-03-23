import cv2
import numpy as np

def detect_shapes(gray_img):
  
  # Preenche pequenos buracos na imagem antes da detecção de contornos
  filled_img = spot_filler(gray_img)
  
  #DEBUG
  cv2.namedWindow("Test filled", cv2.WINDOW_NORMAL)
  cv2.resizeWindow("Test filled", 800, 600)
  cv2.imshow("Debug filled", rgba)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  #
    
  #Detecta bordas
  edges = cv2.Canny(gray_img, 71, 149)
  
  contours,_ = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,#_EXTERNAL for straight up shapes, _TREE if the details are needed for the "height_map_to_normal_map" function
    cv2.CHAIN_APPROX_SIMPLE
  )
  
  #FAILSAFE
  if not contours:  # Verifica se algum contorno foi detectado
      print("Nenhum contorno encontrado.")
      return []
  #
  
  #if len(contours) == 0:
  #    return None

  #largest = max(contours, key=cv2.contourArea)
  
  debug_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
  cv2.drawContours(debug_img,contours,-1,(0,255,0),3)
  
  #DEBUG
  cv2.namedWindow("Test Contours", cv2.WINDOW_NORMAL)
  debug_small = cv2.resize(debug_img, (800, 600))
  cv2.imshow("Test Contours", debug_small)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  #
  
  shapes = []
  for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 500:
      continue
    
    approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
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
  cv2.drawContours(mask, mon_shape, -1, 255, thickness = cv2.FILLED)

  # Converte para RGBA
  rgba = cv2.cvtColor(clean_img, cv2.COLOR_RGB2RGBA)
  
  # Aplica alpha mask
  rgba[:, :, 3] = mask
  
  #DEBUG
  cv2.namedWindow("Test Texture", cv2.WINDOW_NORMAL)
  cv2.resizeWindow("Test Texture", 800, 600)
  cv2.imshow("Debug Texture", rgba)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  #

  return rgba

def spot_filler(img):
  
   # Cria um elemento estruturante
    kernel = np.ones((5, 5), np.uint8)  # Você pode ajustar o tamanho do kernel conforme necessário

    # Aplica o fechamento morfológico
    fill_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    
    
    return fill_img