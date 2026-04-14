from preprocessing import remove_background,preprocess_image, height_map_to_normal_map
from shape_detection import detect_shapes
from shape_detection import texture_cutout
from volume_inference import infer_volumes
from mesh.trimesh_builder import TrimeshBuilder
from mesh.open3d_builder import Open3DBuilder
from mesh.builder_selector import get_mesh_builder
from export_glb import export_glb
from segmentation_sam import segment_object


import os
import cv2
import trimesh
import numpy as np
from PIL import Image
import traceback

def run_pipeline(monument_path, output_path):
  
  #all_shapes = []
  all_volumes = []
  
  textures={}
  
  valid_extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
  
  out_dir = os.path.dirname(output_path) or "."
  os.makedirs(out_dir, exist_ok = True)
  
  try:
    image_files = sorted([f for f in os.listdir(monument_path) if f.lower().endswith(valid_extensions)])
    
    if not image_files:
      raise FileNotFoundError(f"Nenhuma imagem encontrada em: {monument_path}")
      
    for file_name in image_files:
    
      img_path = os.path.join(monument_path, file_name)
      print(f"Processando: {file_name}...")

      try:
        
        img = cv2.imread(img_path)
        
        #FAILSAFE
        if img is None:
          print(f"Aviso: {file_name} não é uma imagem válida. A passar á frente...")
          continue
        
        #Validate file names
        name = file_name.lower()

        if "top" in name:
            view_type = "top"
        elif "left" in name:
            view_type = "left"
        elif "right" in name:
            view_type = "right"
        elif "front" in name:
            view_type = "front"
        elif "back" in name:
            view_type = "back"
        else:
           #DEBUG
            print(f"{file_name} → detected as {view_type}")
            print(f"Unknown view type for {file_name}")
            continue
          
          
        #segmentation_sam's function call
        segmented_img, segmt_mask = segment_object(img)
        
        #FAILSAFE
        if segmt_mask is None:
          print("Segment mask is None or empty.")
          continue
        #
        
        # DEBUG: dump types/shapes
        print("DEBUG:", file_name,
          "segmented_img type:", type(segmented_img),
          "segmented_img shape:", getattr(segmented_img, "shape", None),
          "segmt_mask type:", type(segmt_mask),
          "segmt_mask shape:", getattr(segmt_mask, "shape", None))
        #
        
        #preprocess's function call
        gray, clean = preprocess_image(segmented_img)
        
        #FAILSAFE
        if gray is None or clean is None:
          print("Gray or clean image processing failed.")
          continue
        #
        
        #Shape detection call
        #shapes = detect_shapes(gray)
        shapes = detect_shapes(segmt_mask)
                
        #FAILSAFE
        if not shapes:
          print("No shapes detected.")
          continue
        #
        
        #DEBUG
        print("DEBUG:", file_name, "shapes count:", len(shapes), 
          "first shape type:", type(shapes[0]) if shapes else None,
          "first shape shape:", getattr(shapes[0], "shape", None) if shapes else None)
        #
        
        #Geração de Texturas
        albedo_path = os.path.join(out_dir, f"{view_type}_albedo.png")
        normal_path = os.path.join(out_dir, f"{view_type}_normal.png")
          
        #Conversão de Espaço de Cores
        #albedo = cv2.cvtColor(albedo_ref,cv2.COLOR_BGR2RGB)
        
        albedo = texture_cutout(clean, shapes)  
        cv2.imwrite(albedo_path, albedo[:, :, ::-1])

        #normal with texture cutout
        graycut = texture_cutout(gray, shapes) 

        normal = height_map_to_normal_map(graycut, 3.0)
        cv2.imwrite(normal_path, normal[:, :, ::-1])
        
        
        textures = {
          "top": ("top_albedo.png", "top_normal.png"),
          "front": (...),
          "back":(...),
          "left": (...),
          "right":(...)
        } 
        
        #Shapes array- adding shapes
        #all_shapes.extend(shapes)

        
        textures[view_type] = (albedo_path, normal_path)
        
        #Volume inference call
        volumes = infer_volumes(shapes, view_type)
        
        #FAILSAFE
        if not volumes:
            print(f"Volume não inferido para {file_name}.")
            continue
        #
        
        all_volumes.append(volumes)
      
      except Exception as expt:
        print(f"Erro ao processar o ficheiro {file_name}: {expt}")
        continue
    
    if not all_volumes:
      print(f"Erro: Nenhum volume foi gerado. A abortar exportação")  
      return

    print("All volumes count:", len(all_volumes))

    #Mesh e UVS
    
    if not textures:
      print("Erro: Nenhuma textura foi gerada.")
      return
    
    builder = get_mesh_builder(method="trimesh")
    mesh = builder.build(all_volumes)
    
    objtexnorm = builder.apply_texture_to_mesh(mesh,textures)
  
    export_glb(objtexnorm, output_path)
    print(f"Sucesso! Ficheiro exportado para: {output_path}")
    
  except Exception as expt:
    print("Ocorreu um erro crítico no pipeline:")
    print(expt)
    traceback.print_exc()
  
  
if __name__ == "__main__":
  
  base_dir = os.path.dirname(os.path.abspath(__file__))

  input_folder = os.path.join(base_dir, "..", "input", "monument_01")
  output_file = os.path.join(base_dir, "output", "monument_01.glb")

  input_folder = os.path.abspath(input_folder)
  output_file = os.path.abspath(output_file)
  
  #DEBUG
  print("Resolved input path:", input_folder)
  
  #DEBUG
  #print("Current working directory:", os.getcwd())
  #print("Trying to access:", os.path.abspath(input_folder))
  
  if os.path.exists(input_folder):
    run_pipeline(input_folder, output_file)
  else:
    print(f"Erro: A pasta de entrada {input_folder} não existe.")
 