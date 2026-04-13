import trimesh
import cv2
import os
import numpy as np
from PIL import Image
from mesh.base_mesh_builder import BaseMeshBuilder
from shapely.geometry import Polygon
import sys

#Import Tester
#print("PYTHON EXEC:", sys.executable)
#try:
#    from shapely.geometry import Polygon
#    print("Shapely import OK")
#except Exception as e:
#    print("Shapely import FAILED:", e)

class TrimeshBuilder(BaseMeshBuilder):
  def __init__(self, debug=False, debug_dir="output/debug"):
    
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

    return mesh
  
 
 #def apply_texture_to_mesh(mesh, texture_path):
  #  teximg = Image.open(texture_path)
  #  mat=trimesh.visual.texture.SimpleMaterial(
  #    image = teximg
  #  )
    
    #UV Simples
 #   uv=mesh.vertices[:, [0,2]]
  #  uv-=uv.min(axis = 0)
  #  uv/=uv.max(axis = 0)
    
   # mesh.visual = trimesh.visual.texture.TextureVisuals(
   #   uv = uv,
    #  material = mat
   # )
    
    #return mesh

  #def apply_texture_to_mesh(self, mesh, texture_path, normal_path=None):

      # --- UV mapping (box projection) ---
      #normals = mesh.vertex_normals
      #uv = np.zeros((len(mesh.vertices), 2))

      #for i, n in enumerate(normals):
          #x, y, z = mesh.vertices[i]
          #nx, ny, nz = np.abs(n)

          #if nx > ny and nx > nz:
              #uv[i] = [y, z]
          #elif ny > nx and ny > nz:
              #uv[i] = [x, z]
          #else:
              #uv[i] = [x, y]

      #uv -= uv.min(axis=0)
      #uv /= np.maximum(uv.max(axis=0), 1e-8)

      # --- Material ---
      #if normal_path:
          #material = trimesh.visual.material.PBRMaterial(
              #baseColorTexture=Image.open(texture_path),
              #normalTexture=Image.open(normal_path),
              #metallicFactor=0.0,
              #roughnessFactor=1.0
          #)
      #else:
          #material = trimesh.visual.material.SimpleMaterial(
              #image=Image.open(texture_path)
          #)

      # --- Apply ---
      #mesh.visual = trimesh.visual.texture.TextureVisuals(
          #uv=uv,
          #material=material
      #)

      #return mesh
  
  def apply_texture_to_mesh(self, mesh, texture_path, normal_path=None):
    textures = {
    "top": ("top_albedo.png", "top_normal.png"),
    "front": (...),
    "back":(...),
    "left": (...),
    "right":(...)
    }
    
    #textures = {
    #   "top": (texture_path, normal_path),
    #   "side": (texture_path, normal_path)
    # }

    faces_top = []
    faces_side = []

    
    for i, normal in enumerate(mesh.face_normals):
        nx, ny, nz = np.abs(normal)

        if nz > 0.7:
            faces_top.append(i)
        else:
            faces_side.append(i)

    meshes = []

    if faces_top:
        top_mesh = mesh.submesh([faces_top], append=True)
        tex, norm = textures["top"]
        meshes.append(self.apply_texture_simple(top_mesh, tex, norm))

    if faces_side:
        side_mesh = mesh.submesh([faces_side], append=True)
        tex, norm = textures["side"]
        meshes.append(self.apply_texture_simple(side_mesh, tex, norm))

    return trimesh.util.concatenate(meshes)