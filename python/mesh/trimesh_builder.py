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
        
    def build(self, volumes):

        meshes = []
        footprints = []
        profiles = []

        # --- Separate data ---
        for vlm in volumes:
            if vlm["type"] == "footprint":
                pts = vlm["contour"].squeeze()

                if len(pts) >= 3:
                    poly = Polygon(pts)

                    if not poly.is_valid:
                        poly = poly.buffer(0)

                    footprints.append(poly)

            elif vlm["type"] == "profile":
                profiles.append(vlm)

        if not footprints:
            raise ValueError("No footprint found for extrusion")

        
        matches = self.match_profiles_to_footprints(footprints, profiles)

        for footprint, profile in matches:

            height = profile["height"] if profile else 50

            mesh = trimesh.creation.extrude_polygon(
                footprint,
                height,
                engine="earcut"
            )

            meshes.append(mesh)

        return trimesh.util.concatenate(meshes)

    def match_profiles_to_footprints(self, footprints, profiles):

        matches = []

        for fp in footprints:
            minx, miny, maxx, maxy = fp.bounds
            fw = maxx - minx

            best_profile = None
            best_score = float("inf")

            for pr in profiles:
                px = pr["x"]
                pw = cv2.boundingRect(pr["contour"])[2]

                dx = abs(minx - px)
                dw = abs(fw - pw)

                score = dx + dw

                if score < best_score:
                    best_score = score
                    best_profile = pr

            matches.append((fp, best_profile))

        return matches

    def apply_texture_simple(self, mesh, texture_path, normal_path=None):

        # --- UV mapping (planar projection) ---
        normals = mesh.vertex_normals
        uv = np.zeros((len(mesh.vertices), 2))

        for i, n in enumerate(normals):
            x, y, z = mesh.vertices[i]
            nx, ny, nz = np.abs(n)

            if nx > ny and nx > nz:
                uv[i] = [y, z]
            elif ny > nx and ny > nz:
                uv[i] = [x, z]
            else:
                uv[i] = [x, y]

        # Normalize UVs
        uv -= uv.min(axis=0)
        uv /= np.maximum(uv.max(axis=0), 1e-8)

        # --- Material ---
        if normal_path:
            material = trimesh.visual.material.PBRMaterial(
                baseColorTexture=Image.open(texture_path),
                normalTexture=Image.open(normal_path),
                metallicFactor=0.0,
                roughnessFactor=1.0
            )
        else:
            material = trimesh.visual.material.SimpleMaterial(
                image=Image.open(texture_path)
            )

        # --- Apply ---
        mesh.visual = trimesh.visual.texture.TextureVisuals(
            uv=uv,
            material=material
        )

        return mesh

    def apply_texture_to_mesh(self, mesh, textures):

        faces_top = []
        faces_side = []

        for i, normal in enumerate(mesh.face_normals):
            nx, ny, nz = np.abs(normal)

            if nz > 0.7:
                faces_top.append(i)
            else:
                faces_side.append(i)

        meshes = []

        # --- TOP ---
        if faces_top and "top" in textures:
            top_mesh = mesh.submesh([faces_top], append=True)
            tex, norm = textures["top"]
            meshes.append(self.apply_texture_simple(top_mesh, tex, norm))

        # --- SIDES ---
        if faces_side:
            side_mesh = mesh.submesh([faces_side], append=True)

            side_key = next(
                (k for k in ["front", "back", "left", "right"] if k in textures),
                None
            )

            if side_key:
                tex, norm = textures[side_key]
                meshes.append(self.apply_texture_simple(side_mesh, tex, norm))
            else:
                meshes.append(side_mesh)

        return trimesh.util.concatenate(meshes)