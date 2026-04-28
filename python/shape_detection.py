import os
import cv2
import numpy as np

DEBUG_VISUALS = os.environ.get("PIPELINE_VISUAL_DEBUG") == "1"


def _show_debug_image(name, img):
  if not DEBUG_VISUALS:
    return

  cv2.namedWindow(name, cv2.WINDOW_NORMAL)
  cv2.resizeWindow(name, 800, 600)
  cv2.imshow(name, img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def detect_shapes(segmask):#originally gray_img
  
    # Ensure binary mask
    #_, segmask = cv2.threshold(segmask, 127, 255, cv2.THRESH_BINARY)

    # Preenche pequenos buracos na imagem antes da detecção de contornos
    #filled_img = spot_filler(segmask)

    # --- Ensure binary mask ---
    #_, segmask = cv2.threshold(segmask, 127, 255, cv2.THRESH_BINARY)

    # --- Get largest contour from raw mask ---
    #contours, _ = cv2.findContours(segmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #if not contours:
    #    print("Nenhum contorno encontrado.")
    #    return []

    #largest = max(contours, key=cv2.contourArea)

    # --- Convex hull (fills missing gray areas) ---
    #hull = cv2.convexHull(largest)

    #fixed_mask = np.zeros_like(segmask)
    #cv2.drawContours(fixed_mask, [hull], -1, 255, thickness=cv2.FILLED)

    #segmask = fixed_mask

    # --- Expand mask (recover weak edges) ---
    #kernel = np.ones((10, 10), np.uint8)
    #segmask = cv2.dilate(segmask, kernel, iterations=1)

    # --- Fill small holes ---
    #filled_img = spot_filler(segmask)
    
    #Detecta bordas
    #edges = cv2.Canny(gray_img, 71, 149)

    #contours,_ = cv2.findContours(
    #filled_img,
    #cv2.RETR_EXTERNAL,#_EXTERNAL for straight up shapes, _TREE if the details are needed for the "height_map_to_normal_map" function
    #cv2.CHAIN_APPROX_SIMPLE
    #)
  
    #contours = [c for c in contours if cv2.contourArea(c) > 500]

    #FAILSAFE
    #if not contours:  # Verifica se algum contorno foi detectado
    #    print("Nenhum contorno encontrado.")
    #    return []
    #
  
    #if len(contours) == 0:
    #    return None

    #largest = max(contours, key=cv2.contourArea)
  
    #DEBUG
    #debug_img = cv2.cvtColor(filled_img, cv2.COLOR_GRAY2BGR)
    #cv2.drawContours(debug_img,contours,-1,(0,255,0),3)
    #debug_small = cv2.resize(debug_img, (800, 600))
    #_show_debug_image("Test Contours", debug_small)
    #

    # Get largest contour
    #largest = max(contours, key=cv2.contourArea)

    #approx = cv2.approxPolyDP(
    #    largest,
    #    0.01 * cv2.arcLength(largest, True),
    #    True
    #)
    
    #cv2.imwrite("debug_fixed_mask.png", filled_img)
    
    #return [approx]
    
    # --- Ensure binary ---
    _, segmask = cv2.threshold(segmask, 127, 255, cv2.THRESH_BINARY)

    # --- Get largest contour ---
    contours, _ = cv2.findContours(segmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("Nenhum contorno encontrado.")
        return []

    largest = max(contours, key=cv2.contourArea)

    # --- Convex hull (fills missing parts) ---
    hull = cv2.convexHull(largest)

    mask = np.zeros_like(segmask)
    cv2.drawContours(mask, [hull], -1, 255, thickness=cv2.FILLED)

    # --- Gentle dilation  ---
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # --- Fill holes ---
    mask = spot_filler(mask)

    # --- Smooth edges (NEW, important) ---
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # --- Final contour ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("Nenhum contorno encontrado.")
        return []

    largest = max(contours, key=cv2.contourArea)

    # --- Approx polygon ---
    approx = cv2.approxPolyDP(
        largest,
        0.01 * cv2.arcLength(largest, True),
        True
    )

    # --- DEBUG ---
    debug_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(debug_img, [approx], -1, (0, 255, 0), 3)
    _show_debug_image("Final Shape", cv2.resize(debug_img, (800, 600)))

    cv2.imwrite("debug_fixed_mask.png", mask)

    return [approx]

    #shapes = []
    #for cnt in contours:
    #  area = cv2.contourArea(cnt)
    #  if area < 500:
    #    continue

    #  approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
    # shapes.append(approx)

    #Para um Unico Contour Solido em Vez de Dividido(subtitui o ciclo "for" acima, nao sei se funciona bem com o "mesh_builder" mas deixa a funcao texture_cutout mais pequena e deve funcionar bem com a funcao "height_map_to_normal_map" que inclusive talvez tenha de ser mudada de sitio)
    #largest = max(contours, key=cv2.contourArea)
    #approx = cv2.approxPolyDP(largest, 0.02 * cv2.arcLength(largest, True), True)
    #
    #return [approx]

    #return shapes
  
  
#Para Recortar a Clean Image Baseada nos Contours de "detect_shapes" para a Texture
                  #clean image, monument shape(talvez monument edges em vez de shape mas tem de se ver a preview primeiro)
 
 
def spot_filler(img):

    # Cria um elemento estruturante
    kernel = np.ones((5, 5), np.uint8)  # Você pode ajustar o tamanho do kernel conforme necessário

    # Aplica o fechamento morfológico
    fill_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    return fill_img                   
                    
                    
def texture_cutout(clean_img, mon_shapes):
    """
    Applies the detected shape(s) as an alpha mask to the clean image,
    making everything outside the shapes fully transparent.
    """

    h, w = clean_img.shape[:2]

     # Create blank mask
    mask = np.zeros((h, w), dtype=np.uint8)

    # Draw ALL shapes into the SAME mask
    for shape in mon_shapes:
        cv2.drawContours(mask, [shape], -1, 255, thickness=cv2.FILLED)

    # Convertions
    if len(clean_img.shape) == 2:
        # grayscale → convert to BGRA
        rgba = cv2.cvtColor(clean_img, cv2.COLOR_GRAY2BGRA)

    elif clean_img.shape[2] == 3:
        # BGR → BGRA
        rgba = cv2.cvtColor(clean_img, cv2.COLOR_BGR2BGRA)

    else:
        # already RGBA
        rgba = clean_img.copy()


    # Apply mask as alpha channel(OLD)
    #rgba[:, :, 3] = mask


    # Apply mask to RGB too
    rgba[:, :, 0:3] = cv2.bitwise_and(rgba[:, :, 0:3], rgba[:, :, 0:3], mask=mask)

    # Alpha channel
    rgba[:, :, 3] = mask
    
    #Crop to bounding box of the shape
    x, y, w, h = cv2.boundingRect(mask)

    rgba = rgba[y:y+h, x:x+w]
    mask = mask[y:y+h, x:x+w]
    
    #Debug
    _show_debug_image("Debug Texture", rgba)

    return rgba


    

