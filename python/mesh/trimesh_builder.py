import trimesh
import os
import numpy as np
from PIL import Image
from .base_mesh_builder import BaseMeshBuilder

class TrimeshBuilder(BaseMeshBuilder):
  def __init__(self, debug=False, debug_dir="output/debug"):
    
    #DEBUG
    self.debug = debug
    self.debug_dir = debug_dir
    
    if self.debug:
      os.makedirs(self.debug_dir, exist_ok=True)
    #
    
  def build(self, volumes):
    
    meshes = []
    
    #DEBUG
    if debug:
          os.makedirs(debug_dir, exist_ok=True)
    #
          
    for i, vlm in enumerate(volumes):
      
      box = trimesh.creation.box(
        extents=[vlm["width"], vlm["depth"], vlm["height"]]
      )
      
      box.apply_translation([vlm["x"], 0, vlm["y"]])
      
      meshes.append(box)
      
      #DEBUG: export individual
      if self.debug:
          debug_path = os.path.join(debug_dir, f"volume_{i}.ply")
          box.export(debug_path)
    #
    
    combined = trimesh.util.concatenate(meshes)

    #DEBUG: combinado
    if self.debug:
        combined.export(os.path.join(debug_dir, "combined_debug.ply"))
    #
    
    return combined

  def apply_texture_to_mesh(mesh, texture_path):
    teximg = Image.open(texture_path)
    mat=trimesh.visual.texture.SimpleMaterial(
      image = teximg
    )
    
    #UV Simples
    uv=mesh.vertices[:, [0,2]]
    uv-=uv.min(axis = 0)
    uv/=uv.max(axis = 0)
    
    mesh.visual = trimesh.visual.texture.TextureVisuals(
      uv = uv,
      material = mat
    )
    
    return mesh

