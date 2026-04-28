import trimesh
import cv2
import os
import numpy as np
from PIL import Image
from trimesh import bounds
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

            meshes.append((mesh, footprint.bounds))

        return meshes

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

    def apply_texture_simple(self, mesh, texture_path, normal_path=None, bounds=None):

        # --- UV mapping ---
        uv = np.zeros((len(mesh.vertices), 2))

        if bounds is not None:
            # --- Bounds Map ---
            minx, miny, maxx, maxy = bounds

            for i, v in enumerate(mesh.vertices):
                x, y, z = v

                u = (x - minx) / (maxx - minx + 1e-8)
                v_coord = (y - miny) / (maxy - miny + 1e-8)

                uv[i] = [u, v_coord]
                
            #this or nothing should work for flipping the texture
            #uv[:, 1] = 1.0 - uv[:, 1]
            
        else:
            normals = mesh.face_normals

            for i, v in enumerate(mesh.vertices):
                x, y, z = v

                # finds dominant axis from face normals
                nx, ny, nz = np.abs(mesh.vertex_normals[i])

                if ny > nx and ny > nz:
                    # Left and Right(YZ)
                    uv[i] = [y, z]

                elif nx > ny and nx > nz:
                    # Front and Back(XZ)
                    uv[i] = [x, z]

                else:
                    # TOP Fallback
                    uv[i] = [x, y]
                       
            #normalize                
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
        
        #DEBUG
        print("UV min:", uv.min(axis=0))
        print("UV max:", uv.max(axis=0))
        #
        
        return mesh

    def apply_texture_to_mesh(self, mesh_data, textures):
        
        final_meshes = []
        
        #DEBUG
        print("Textures available:", textures.keys())
        #
        
        for mesh, mesh_bounds in mesh_data:

            faces_top = []
            faces_front = []
            faces_back = []
            faces_left = []
            faces_right = []
            
            meshes = []
            
            for i, normal in enumerate(mesh.face_normals):
                nx, ny, nz = normal

                if abs(nz) > 0.5:
                    faces_top.append(i)

                elif abs(ny) > abs(nx):
                    if ny > 0:
                        faces_front.append(i)
                    else:
                        faces_back.append(i)

                else:
                    if nx > 0:
                        faces_right.append(i)
                    else:
                        faces_left.append(i)
                        
            # --- TOP ---
            if faces_top and "top" in textures:
                top_mesh = mesh.submesh([faces_top], append=True)
                tex, norm = textures["top"]

                top_mesh = self.apply_texture_simple(
                    top_mesh,
                    tex,
                    norm,
                    bounds= mesh_bounds
                )
                meshes.append(top_mesh)

            # --- FRONT ---
            if faces_front and "front" in textures:
                m = mesh.submesh([faces_front], append=True)
                tex, norm = textures["front"]
                m = self.apply_texture_simple(m, tex, norm)
                meshes.append(m)

            # --- BACK ---
            if faces_back and "back" in textures:
                m = mesh.submesh([faces_back], append=True)
                tex, norm = textures["back"]
                m = self.apply_texture_simple(m, tex, norm)
                meshes.append(m)

            # --- LEFT ---
            if faces_left and "left" in textures:
                m = mesh.submesh([faces_left], append=True)
                tex, norm = textures["left"]
                m = self.apply_texture_simple(m, tex, norm)
                meshes.append(m)

            # --- RIGHT ---
            if faces_right and "right" in textures:
                m = mesh.submesh([faces_right], append=True)
                tex, norm = textures["right"]
                m = self.apply_texture_simple(m, tex, norm)
                meshes.append(m)
            
            #DEBUG
            print("Mesh bounds:", mesh.bounds)
            print("Footprint bounds:", mesh_bounds)
            #

            final_meshes.append(trimesh.util.concatenate(meshes))

        return trimesh.util.concatenate(final_meshes)