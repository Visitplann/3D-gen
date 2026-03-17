import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import SAM

model = SAM("sam2_t.pt")

def segment_object(img):
    
    h, w = img.shape[:2]

    input_point = [[w // 2, h // 2]]

    try:
        results = model.predict(
            source=img,
            points=input_point,
            labels=[1]
        )
        print(results)
    except Exception as expt:
   
        print("falhou no prect {expt}")
   
   
    #FAILSAFE
    if not results or not results[0].masks.data:
        print("Nenhuma máscara foi retornada.")
        return None, None
    #
    
    mask = results[0].masks.data[0].cpu().numpy()
    mask = (mask * 255).astype(np.uint8)

    segmented = cv2.bitwise_and(img, img, mask=mask)
    
    #Mostrar o resultado na tela
    res_plotted = results[0].plot()
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    
    return segmented, mask


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