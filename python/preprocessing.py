import cv2
import numpy as np
import os
from PIL import Image

DEBUG_VISUALS = os.environ.get("PIPELINE_VISUAL_DEBUG") == "1"


def _show_debug_image(name, img):
  if not DEBUG_VISUALS:
    return

  cv2.namedWindow(name, cv2.WINDOW_NORMAL)
  cv2.resizeWindow(name, 800, 600)
  cv2.imshow(name, img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()



def remove_background(img):#DEPRECATED
  h,w = img.shape[:2]
  
  mask = np.zeros((h,w), np.uint8)
  
  bgdModel = np.zeros((1,65), np.float64)#Background Model
  fgdModel = np.zeros((1,65), np.float64)#Foreground Model
  
  #Marca o Centro da Imagem
  rect = (
    int(w*0.1),
    int(h*0.1),
    int(w*0.8),
    int(h*0.8),
  )
  
  #Remove as Partes da Imagem Fora do Rectangulo Central
  cv2.grabCut(
    img,
    mask,
    rect,
    bgdModel,
    fgdModel,
    5,
    cv2.GC_INIT_WITH_RECT
  )
  
  #Mask para Binario
  maskbin = np.where(
    (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
    1,
    0
  ).astype("uint8")

  #Aplica Mask Binaria
  result = img * maskbin[:, :, np.newaxis]
  
  return result

#Will receive either a file path or an image array, and will return the preprocessed grayscale image and the cleaned RGB image without background. The cleaned RGB image can be used as a reference for albedo during volume inference.
def preprocess_image(img):
  
  #img = cv2.imread(path)
  
  # If a path was passed
  #if isinstance(input_data, str):
  #  img = cv2.imread(input_data)

  # If an image array was passed
  #else:
  #  img = input_data.copy()
      
  #Failsafe Line
  #assert img is not None, "file could not be read, check with os.path.exists()"
  
  _show_debug_image("Test Original", img)
  
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  
  #Remove "Background"
  #img = remove_background(img) #DEPRECATED
  
  #DEBUG
  #cv2.namedWindow("Test No Background", cv2.WINDOW_NORMAL #DEPRECATED
  #cv2.resizeWindow("Test No Background", 800, 600)
  #cv2.imshow("Test No Background", img)
  #cv2.waitKey(0)
  #cv2.destroyAllWindows()
  #
  
  #Bilateral Filter(remove ruido)
  img = cv2.bilateralFilter(img, d = 6, sigmaColor = 10, sigmaSpace = 20)
  
  clean = img.copy()
  
  _show_debug_image("Test Clean", clean)
  
  #Ajuste de Contraste e Brilho
  a = 1.2 #contraste
  b = 10 #brilho
  img = cv2.convertScaleAbs(img, alpha = a, beta = b)
  
  _show_debug_image("Test Constraste", img)
  
  #Converte para Greyscale
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  
  #Suaviza o background para facilitar a detecção de formas
  #gray = cv2.GaussianBlur(gray, (5,5), 0) #DEPRECATED
  
  _show_debug_image("Test Gray", gray)
  
  os.makedirs("output/debug", exist_ok = True)
  cv2.imwrite("output/debug/no_bg.png", img[:, :, ::-1])
  
  
  #USE ONLY FOR SMOOTH MODERN ARCHITECTURE AND MODERN SCULPTURES, NOT SUITABLE FOR ORNATE HISTORIC MONUMENTS BECAUSE OF LARGE AMOUNTS OF DETAILS AND "HEAVY" TEXTURES 
  #Aplica Threshold Otsu para separar o fundo do primeiro plano
  #_, thresh = cv2.threshold(
  #  gray, 0, 255,
  #  cv2.THRESH_BINARY + cv2.THRESH_OTSU
  #)
  
  #USE ONLY FOR SMOOTH MODERN ARCHITECTURE AND MODERN SCULPTURES, NOT SUITABLE FOR ORNATE HISTORIC MONUMENTS BECAUSE OF LARGE AMOUNTS OF DETAILS AND "HEAVY" TEXTURES 
  #Fecha Pequenos Buracos e Conecta Componentes Próximos
  #kernel = np.ones((7,7), np.uint8)
  #gray = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
  
  
  return gray, clean

#Intensidade por Material: 
#>Pedra Lisa: 1.0-1.5
#>Fachada: 2.0-3.0
#>Esculturas: 3.0-4.5
def height_map_to_normal_map(gray, strg = 2.0, invert_y = True):
  gray = gray.astype(np.float32) / 255.0
  
  #Grandiante
  dx=cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize = 3)
  dy=cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize = 3)
  
  if invert_y:
    dy = -dy
  
  #Normal Vector
  nx = -dx * strg
  ny = -dy * strg
  nz = np.ones_like(gray)
  
  norm = np.sqrt(nx*nx + ny*ny + nz*nz)
  nx /= norm
  ny /= norm
  nz /= norm
  
  #Converçao para RGB
  normal_map = np.zeros((gray.shape[0], gray.shape[1], 3), dtype = np.uint8)
  normal_map[:,:,0]=((nx+1)*0.5*255).astype(np.uint8)
  normal_map[:,:,1]=((ny+1)*0.5*255).astype(np.uint8)
  normal_map[:,:,2]=((nz+1)*0.5*255).astype(np.uint8)
  
  return normal_map
