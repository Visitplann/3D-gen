from preprocessing import remove_background,preprocess_image, height_map_to_normal_map
from shape_detection import detect_shapes
from volume_inference import infer_volumes
from mesh_builder import build_mesh, apply_texture_to_mesh
from export_glb import export_glb


import os
import cv2
import trimesh
import numpy as np
from PIL import Image
import traceback

def run_pipeline(monument_path, output_path):
  
  all_volumes = []
  albedo_ref = None
  gray_ref = None
  
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
    
        gray, clean = preprocess_image(img_path)
    
        if albedo_ref is None:
            albedo_ref = clean
            gray_ref = gray
            
        shapes = detect_shapes(gray)
        volumes = infer_volumes(shapes, file_name)
        all_volumes.extend(volumes)
      
      except Exception as expt:
        print(f"Erro ao processar o ficheiro {file_name}: {expt}")
        continue
    
    if not all_volumes:
      print(f"Erro: Nenhum volume foi gerado. A abortar exportação")  
      return
    
    print("All volumes count:", len(all_volumes))
    print("albedo_ref is None?", albedo_ref is None)
    print("gray_ref is None?", gray_ref is None)
    
    #Geração de Texturas
    
    if albedo_ref is None or gray_ref is None:
      print(f"Erro: Não foi possível gerar texturas.")
      return
    
    albedo_path = os.path.join(out_dir,"albedo.png")
    normal_path = os.path.join(out_dir,"normal.png")
    
    #Conversão de Espaço de Cores
    #albedo = cv2.cvtColor(albedo_ref,cv2.COLOR_BGR2RGB)
    
    cv2.imwrite(albedo_path, albedo_ref[:, :, ::-1])
    normal = height_map_to_normal_map(gray_ref, 3.0)
    cv2.imwrite(normal_path, normal[:, :, ::-1])
    
    #Mesh e UVS
    mesh = build_mesh(all_volumes, debug=True)
    
    #Projeção Planar Simple se NÃO Existirem UVs
    if not hasattr(mesh.visual,'uv') or mesh.visual.uv is None:
      
      #Projeção do Plano XZ
      uv = mesh.vertices[:, [0, 2]].astype(np.float64)
      uv -= uv.min(axis=0)
      uv /= uv.max(axis=0)
      
      mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uv)
      
    #Aplicar Material PBR  
    material = trimesh.visual.material.PBRMaterial(
      baseColorTexture = Image.open(albedo_path),
      normalTexture = Image.open(normal_path),
      metallicFactor = 0.0,
      roughnessFactor = 1.0
    )
    mesh.visual.material = material
  
    export_glb(mesh, output_path)
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
 