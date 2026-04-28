import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from shapely import area
from ultralytics import SAM

DEBUG_VISUALS = os.environ.get("PIPELINE_VISUAL_DEBUG") == "1"
model = SAM("sam2_t.pt")


def touches_border(mask, threshold=5):
    h, w = mask.shape

    top = np.any(mask[0:threshold, :] > 0)
    bottom = np.any(mask[h-threshold:h, :] > 0)
    left = np.any(mask[:, 0:threshold] > 0)
    right = np.any(mask[:, w-threshold:w] > 0)
    
    #if intended object touches any border change to this return v
    #return (top + bottom + left + right) >= 3
    
    return top or bottom or left or right

def segment_object(img):
    
    #h, w = img.shape[:2]

    #input_point = [[w // 2, h // 2]]

    #try:
    #    results = model.predict(
    #        source=img,
    #        points=input_point,
    #        labels=[1]
    #    )
    #    print(results)
    #except Exception as expt:
   
    #    print("Falhou no predict {expt}")
   
   
    #FAILSAFE
    #if not results or results[0].masks is None or results[0].masks.data is None or len(results[0].masks.data) == 0:
    #    print("Nenhuma máscara foi retornada.")
    #    return None, None
    #
    
    #mask = results[0].masks.data[0].cpu().numpy()
    #mask = (mask * 255).astype(np.uint8)

    #segmented = cv2.bitwise_and(img, img, mask=mask)
    
    #Mostrar o resultado na tela
    #if DEBUG_VISUALS:
    #    res_plotted = results[0].plot()
    #    plt.figure(figsize=(10, 10))
    #    plt.imshow(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))
    #    plt.axis('off')
    #    plt.show()
    
    #return segmented, mask
    
    h, w = img.shape[:2]

    # --- Try multiple points (center + offsets) ---
    candidate_points = [
        [w // 2, h // 2],           # center
        [w // 3, h // 2],           # left-center
        [2 * w // 3, h // 2],       # right-center
        [w // 2, h // 3],           # top-center
        [w // 2, 2 * h // 3],       # bottom-center
    ]

    best_mask = None
    best_area = 0
    best_segmented = None

    for pt in candidate_points:
        try:
            results = model.predict(
                source=img,
                points=[pt],
                labels=[1]
            )
        except Exception as expt:
            print(f"Falhou no predict: {expt}")
            continue

        if not results or results[0].masks is None:
            continue

        masks = results[0].masks.data
        if masks is None or len(masks) == 0:
            continue

        mask = masks[0].cpu().numpy()
        mask = (mask * 255).astype(np.uint8)

        area = np.sum(mask > 0)
        
        # --- REJECT BACKGROUND ---
        if touches_border(mask):
            continue

        # --- reject tiny masks ---
        if area < (h * w * 0.05):
            continue
        
        # Keep largest detected object
        if area > best_area:
            best_area = area
            best_mask = mask
            best_segmented = cv2.bitwise_and(img, img, mask=mask)

    if best_mask is None:
        print("Nenhuma máscara foi retornada.")
        return None, None

    # --- DEBUG ---
    if DEBUG_VISUALS:
        debug = cv2.cvtColor(best_segmented, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(6, 6))
        plt.imshow(debug)
        plt.title("Best Segmentation")
        plt.axis('off')
        plt.show()

    return best_segmented, best_mask


#SAM TEST   
#Carregar o modelo SAM 2 
#model = SAM("sam2_t.pt")

#Carregar a sua imagem para pegar as dimensões
#img_path = 'caixab.jpg' # Ajuste o nome do arquivo aqui se necessário
#img = cv2.imread(img_path)
#h, w  = img.shape[:2]

#Definir o ponto central para busca do objeto
#Como seu estojo está centralizado, vamos clicar no meio da imagem
#input_point = [[w // 2, h // 2]] 

#Executar a inteligência artificial
#results = model.predict(source=img_path, points=input_point, labels=[1])

#Mostrar o resultado na tela
#res_plotted = results[0].plot()
#plt.figure(figsize=(10, 10))
#plt.imshow(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))
#plt.axis('off')
#plt.show()

#Salvar a máscara isolada para uso posterior
#results[0].save("objeto_detectado.jpg")
