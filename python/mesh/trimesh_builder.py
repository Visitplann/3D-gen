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
        
    def build(self, volumes, overall_scale=1.0, complex_mode=False):

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

        if complex_mode and len(footprints) > 1:
            # For complex shapes with multiple footprints, try to create a more detailed mesh
            return self._build_complex_mesh(footprints, profiles, overall_scale)
        else:
            # Standard approach for simple shapes
            matches = self.match_profiles_to_footprints(footprints, profiles)

            for footprint, profile in matches:

                if profile:
                    profile_height = float(profile.get("height", 30))
                    profile_width = float(profile.get("width", 0))
                    rotated_width = float(profile.get("rotated_width", profile_width))
                    rotated_height = float(profile.get("rotated_height", profile_height))

                    # Use the rotated contour dimensions if they provide a better height/width estimate
                    if rotated_width > 0 and rotated_height > 0:
                        if rotated_height >= rotated_width:
                            profile_height = rotated_height
                            profile_width = rotated_width
                        else:
                            profile_height = rotated_width
                            profile_width = rotated_height

                    height = profile_height

                    # Apply scaling based on footprint dimensions
                    if profile["view"] in ("front", "back"):
                        footprint_width = footprint.bounds[2] - footprint.bounds[0]
                        if profile_width > 0:
                            scale = footprint_width / profile_width
                        else:
                            scale = 1.0
                    else:
                        footprint_depth = footprint.bounds[3] - footprint.bounds[1]
                        if profile_width > 0:
                            scale = footprint_depth / profile_width
                        else:
                            scale = 1.0

                    height = height * scale
                else:
                    height = 30  # Default height for complex shapes

                mesh = trimesh.creation.extrude_polygon(
                    footprint,
                    height,
                    engine="earcut"
                )

                if overall_scale != 1.0:
                    mesh.apply_scale(overall_scale)

                meshes.append((mesh, footprint.bounds))

        return meshes

        return meshes

    def _build_complex_mesh(self, footprints, profiles, overall_scale):
        """Build a more complex mesh by combining multiple footprints."""
        meshes = []

        # Try to find a reasonable height from profiles
        avg_height = 30  # Default
        if profiles:
            heights = [p["height"] for p in profiles if "height" in p]
            if heights:
                avg_height = sum(heights) / len(heights)

        # Create extruded meshes for each footprint with some variation in height
        for i, footprint in enumerate(footprints):
            # Add some height variation for more interesting shapes
            height_variation = 0.8 + (i / len(footprints)) * 0.4  # 0.8 to 1.2
            height = avg_height * height_variation

            mesh = trimesh.creation.extrude_polygon(
                footprint,
                height,
                engine="earcut"
            )

            if overall_scale != 1.0:
                mesh.apply_scale(overall_scale)

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
                pw = float(pr.get("rotated_width", cv2.boundingRect(pr["contour"])[2]))

                dx = abs(minx - px)
                dw = abs(fw - pw)

                score = dx + dw

                if score < best_score:
                    best_score = score
                    best_profile = pr

            matches.append((fp, best_profile))

        return matches

    def apply_texture_simple(self, mesh, texture_path, normal_path=None, bounds=None, coord_system="xy", preserve_aspect=False, rotate=0):

        # --- UV mapping ---
        uv = np.zeros((len(mesh.vertices), 2))

        if bounds is not None:
            # --- Bounds Map ---
            min_u, min_v, max_u, max_v = bounds

            for i, v in enumerate(mesh.vertices):
                x, y, z = v

                if coord_system == "xz":
                    # Front/Back faces: use X and Z coordinates
                    u = (x - min_u) / (max_u - min_u + 1e-8)
                    v_coord = (z - min_v) / (max_v - min_v + 1e-8)
                elif coord_system == "yz":
                    # Left/Right faces: use Y and Z coordinates
                    u = (y - min_u) / (max_u - min_u + 1e-8)
                    v_coord = (z - min_v) / (max_v - min_v + 1e-8)
                else:
                    # XY (top face): use X and Y coordinates
                    u = (x - min_u) / (max_u - min_u + 1e-8)
                    v_coord = (y - min_v) / (max_v - min_v + 1e-8)

                uv[i] = [u, v_coord]

            if preserve_aspect:
                texture_image = Image.open(texture_path)
                tex_w, tex_h = texture_image.size
                if rotate in (90, 270):
                    tex_w, tex_h = tex_h, tex_w

                mesh_w = max_u - min_u
                mesh_h = max_v - min_v
                if mesh_h > 0 and tex_h > 0:
                    mesh_ratio = mesh_w / mesh_h
                    texture_ratio = tex_w / tex_h

                    if mesh_ratio > texture_ratio:
                        scale = texture_ratio / mesh_ratio
                        uv[:, 1] = uv[:, 1] * scale + (1.0 - scale) * 0.5
                    else:
                        scale = mesh_ratio / texture_ratio
                        uv[:, 0] = uv[:, 0] * scale + (1.0 - scale) * 0.5

            #this or nothing should work for flipping the texture
            #uv[:, 1] = 1.0 - uv[:, 1]
            
        else:
            normals = mesh.face_normals

            for i, v in enumerate(mesh.vertices):
                x, y, z = v

                # finds dominant axis from face normals
                nx, ny, nz = np.abs(mesh.vertex_normals[i])

                if ny > nx and ny > nz:
                    # Front and Back(XZ) faces lie in the XZ plane
                    uv[i] = [x, z]

                elif nx > ny and nx > nz:
                    # Left and Right(YZ) faces lie in the YZ plane
                    uv[i] = [y, z]

                else:
                    # TOP fallback uses XY
                    uv[i] = [x, y]
                       
            #normalize                
            uv -= uv.min(axis=0)
            uv /= np.maximum(uv.max(axis=0), 1e-8)


        # --- Material ---
        base_image = Image.open(texture_path)
        if rotate != 0:
            base_image = base_image.rotate(rotate, expand=True)

        if normal_path:
            normal_image = Image.open(normal_path)
            if rotate != 0:
                normal_image = normal_image.rotate(rotate, expand=True)
            material = trimesh.visual.material.PBRMaterial(
                baseColorTexture=base_image,
                normalTexture=normal_image,
                metallicFactor=0.0,
                roughnessFactor=1.0,
                alphaMode='BLEND'
            )
        else:
            material = trimesh.visual.material.PBRMaterial(
                baseColorTexture=base_image,
                metallicFactor=0.0,
                roughnessFactor=1.0,
                alphaMode='BLEND'
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

    def apply_texture_to_mesh(self, mesh_data, textures, preserve_aspect=True, texture_rotations=None):
        
        texture_rotations = texture_rotations or {}
        final_meshes = []
        
        #DEBUG
        print("Textures available:", textures.keys())
        print("Texture rotations:", texture_rotations)
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
                rotate = texture_rotations.get("top", 0)

                top_mesh = self.apply_texture_simple(
                    top_mesh,
                    tex,
                    norm,
                    bounds=mesh_bounds,
                    coord_system="xy",
                    preserve_aspect=preserve_aspect,
                    rotate=rotate
                )
                meshes.append(top_mesh)

            object_bounds = mesh.bounds
            object_min_x, object_min_y, object_min_z = object_bounds[0]
            object_max_x, object_max_y, object_max_z = object_bounds[1]

            # --- FRONT ---
            if faces_front and "front" in textures:
                m = mesh.submesh([faces_front], append=True)
                tex, norm = textures["front"]
                front_bounds = (object_min_x, object_min_z, object_max_x, object_max_z)
                m = self.apply_texture_simple(
                    m,
                    tex,
                    norm,
                    bounds=front_bounds,
                    coord_system="xz",
                    preserve_aspect=False,
                    rotate=texture_rotations.get("front", 0)
                )
                meshes.append(m)

            # --- BACK ---
            if faces_back and "back" in textures:
                m = mesh.submesh([faces_back], append=True)
                tex, norm = textures["back"]
                back_bounds = (object_min_x, object_min_z, object_max_x, object_max_z)
                m = self.apply_texture_simple(
                    m,
                    tex,
                    norm,
                    bounds=back_bounds,
                    coord_system="xz",
                    preserve_aspect=False,
                    rotate=texture_rotations.get("back", 0)
                )
                meshes.append(m)

            # --- LEFT ---
            if faces_left and "left" in textures:
                m = mesh.submesh([faces_left], append=True)
                tex, norm = textures["left"]
                left_bounds = (object_min_y, object_min_z, object_max_y, object_max_z)
                m = self.apply_texture_simple(
                    m,
                    tex,
                    norm,
                    bounds=left_bounds,
                    coord_system="yz",
                    preserve_aspect=False,
                    rotate=texture_rotations.get("left", 0)
                )
                meshes.append(m)

            # --- RIGHT ---
            if faces_right and "right" in textures:
                m = mesh.submesh([faces_right], append=True)
                tex, norm = textures["right"]
                right_bounds = (object_min_y, object_min_z, object_max_y, object_max_z)
                m = self.apply_texture_simple(
                    m,
                    tex,
                    norm,
                    bounds=right_bounds,
                    coord_system="yz",
                    preserve_aspect=False,
                    rotate=texture_rotations.get("right", 0)
                )
                meshes.append(m)
            
            #DEBUG
            print("Mesh bounds:", mesh.bounds)
            print("Footprint bounds:", mesh_bounds)
            #

            final_meshes.append(trimesh.util.concatenate(meshes))

        return trimesh.util.concatenate(final_meshes)

    def _get_submesh_bounds_xz(self, submesh):
        """Get 2D bounds (x, z) for front/back faces projecting normals on XZ plane."""
        verts = submesh.vertices
        x_min, x_max = verts[:, 0].min(), verts[:, 0].max()
        z_min, z_max = verts[:, 2].min(), verts[:, 2].max()
        return (x_min, z_min, x_max, z_max)

    def _get_submesh_bounds_yz(self, submesh):
        """Get 2D bounds (y, z) for left/right faces projecting normals on YZ plane."""
        verts = submesh.vertices
        y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
        z_min, z_max = verts[:, 2].min(), verts[:, 2].max()
        return (y_min, z_min, y_max, z_max)
