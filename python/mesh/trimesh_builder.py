import trimesh
import os
import numpy as np
from PIL import Image
from mesh.base_mesh_builder import BaseMeshBuilder

class TrimeshBuilder(BaseMeshBuilder):
  def __init__(self, debug=False, debug_dir="output/debug"):
    
    #DEBUG
    self.debug = debug
    self.debug_dir = debug_dir
    
    if self.debug:
      os.makedirs(self.debug_dir, exist_ok=True)
    #
    
  def build(self, volumes, height_map=None):
    
    meshes = []
    
    #DEBUG
    if self.debug:
          os.makedirs(self.debug_dir, exist_ok=True)
    #
          
    for i, vlm in enumerate(volumes):
      
      box = trimesh.creation.box(
        extents=[vlm["width"], vlm["depth"], vlm["height"]]
      )
      
      box.apply_translation([vlm["x"], 0, vlm["y"]])
      
      meshes.append(box)
      
      #DEBUG: export individual
      if self.debug:
          debug_path = os.path.join(self.debug_dir, f"volume_{i}.ply")
          box.export(debug_path)
          
      if self.debug:
        trimesh.Scene(meshes).show()
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

"""  def __init__(self, debug=False, debug_dir="output/debug"):
    
    #DEBUG
    self.debug = debug
    self.debug_dir = debug_dir
    
    if self.debug:
      os.makedirs(self.debug_dir, exist_ok=True)
    #
  
  def build(self, volumes, height_map=None):

    footprint = None
    height = None

    # Separate data
    for vlm in volumes:
        if vlm["type"] == "footprint":
            contour = vlm["contour"]
            pts = contour.squeeze()
            
            #Use if the texture is upsidedown
            #pts = contour.squeeze().astype(np.float64)
            #pts[:, 1] *= -1
            #footprint = Polygon(pts)
            
            if len(pts) >= 3:
                footprint = Polygon(pts)

        elif vlm["type"] == "profile":
            height = vlm["height"]

    # FAILSAFE
    if footprint is None:
        raise ValueError("No footprint found for extrusion")

    if height is None:
        height = 50  # fallback height

    # Fix invalid polygons
    if not footprint.is_valid:
        footprint = footprint.buffer(0)
        
        
    # Create mesh by extrusion
    mesh = trimesh.creation.extrude_polygon(
    footprint,
    height,
    engine="earcut"
    )

    return mesh"""