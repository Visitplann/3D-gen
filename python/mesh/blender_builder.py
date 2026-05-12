import json
import os
import subprocess
import tempfile
from mesh.base_mesh_builder import BaseMeshBuilder


class BlenderBuilder(BaseMeshBuilder):
    def __init__(self, blender_executable="blender", debug=False, debug_dir="output/debug"):
        self.blender_executable = blender_executable
        self.debug = debug
        self.debug_dir = debug_dir

        if self.debug:
            os.makedirs(self.debug_dir, exist_ok=True)

    def build(self, volumes, overall_scale=1.0, complex_mode=False, output_path=None, textures=None):
        if output_path is None:
            raise ValueError("BlenderBuilder requires output_path")

        if textures is None:
            textures = {}

        serializable_volumes = []
        for vlm in volumes:
            converted = vlm.copy()
            contour = converted.get("contour")
            if contour is not None:
                try:
                    converted["contour"] = contour.tolist()
                except Exception:
                    converted["contour"] = list(contour)
            serializable_volumes.append(converted)

        payload = {
            "volumes": serializable_volumes,
            "scale": overall_scale,
            "complex_mode": complex_mode,
            "output_path": output_path,
            "textures": textures,
        }

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as json_file:
            json.dump(payload, json_file, ensure_ascii=False, indent=2)
            json_path = json_file.name

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix="_blender.py") as script_file:
            script_file.write(self._blender_script())
            script_path = script_file.name

        command = [
            self.blender_executable,
            "--background",
            "--python",
            script_path,
            "--",
            json_path,
        ]

        if self.debug:
            print("Running Blender builder command:", " ".join(command))

        result = subprocess.run(command, capture_output=True, text=True)

        if self.debug:
            log_path = os.path.join(self.debug_dir, "blender_builder.log")
            with open(log_path, "w", encoding="utf-8") as log_file:
                log_file.write(result.stdout)
                log_file.write("\n--- STDERR ---\n")
                log_file.write(result.stderr)
            print(f"Blender log written to: {log_path}")

        os.remove(json_path)
        os.remove(script_path)

        if result.returncode != 0:
            raise RuntimeError(
                f"Blender process failed with return code {result.returncode}: {result.stderr}"
            )

        return output_path

    def _blender_script(self):
        return r"""
import bpy
import bmesh
import json
import math
import mathutils
import os
import sys


def load_payload(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def create_material(name, image_path=None, normal_path=None):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    nodes.clear()

    output = nodes.new(type='ShaderNodeOutputMaterial')
    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    output.location = (400, 0)
    principled.location = (0, 0)
    links.new(principled.outputs['BSDF'], output.inputs['Surface'])

    if image_path and os.path.exists(image_path):
        tex = nodes.new(type='ShaderNodeTexImage')
        tex.image = bpy.data.images.load(image_path)
        tex.location = (-400, 200)
        tex.interpolation = 'Cubic'
        tex.extension = 'REPEAT'
        if tex.image.colorspace_settings is not None:
            tex.image.colorspace_settings.name = 'sRGB'
            tex.image.colorspace_settings.is_data = False
        tex.image.alpha_mode = 'STRAIGHT'
        links.new(tex.outputs['Color'], principled.inputs['Base Color'])
        if 'Alpha' in tex.outputs:
            links.new(tex.outputs['Alpha'], principled.inputs['Alpha'])
            material.blend_method = 'BLEND'
            material.shadow_method = 'NONE'

    if normal_path and os.path.exists(normal_path):
        normal_tex = nodes.new(type='ShaderNodeTexImage')
        normal_tex.image = bpy.data.images.load(normal_path)
        if normal_tex.image.colorspace_settings is not None:
            normal_tex.image.colorspace_settings.name = 'Non-Color'
            normal_tex.image.colorspace_settings.is_data = True
        normal_tex.location = (-400, -100)

        normal_map = nodes.new(type='ShaderNodeNormalMap')
        normal_map.location = (-200, -100)
        normal_map.inputs['Strength'].default_value = 0.8
        links.new(normal_tex.outputs['Color'], normal_map.inputs['Color'])
        links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])

    return material


def sample_profile_height(profile, t):
    pts = profile.get('contour', [])
    if not pts:
        return None

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if max_x == min_x or max_y == min_y:
        return None

    target_x = min_x + t * (max_x - min_x)
    samples = {}
    for x, y in pts:
        height = max_y - y
        samples.setdefault(x, []).append(height)

    points = [(x, max(heights)) for x, heights in samples.items()]
    points.sort(key=lambda item: item[0])

    if not points:
        return None

    if target_x <= points[0][0]:
        interp = points[0][1]
    elif target_x >= points[-1][0]:
        interp = points[-1][1]
    else:
        interp = points[-1][1]
        for i in range(len(points) - 1):
            x0, h0 = points[i]
            x1, h1 = points[i + 1]
            if x0 <= target_x <= x1:
                ratio = (target_x - x0) / (x1 - x0) if x1 != x0 else 0.0
                interp = h0 + (h1 - h0) * ratio
                break

    return interp / (max_y - min_y)


def get_vertex_profile_height(vertex, footprint_bounds, profiles, default_height):
    min_x, max_x, min_y, max_y = footprint_bounds
    if max_x == min_x or max_y == min_y:
        return default_height

    heights = []
    footprint_width = max_x - min_x
    footprint_depth = max_y - min_y

    for profile in profiles:
        view = profile.get('view')
        if view in ('front', 'back'):
            if footprint_width <= 0:
                continue
            t = (vertex.x - min_x) / footprint_width
            t = max(0.0, min(1.0, t))
            normalized = sample_profile_height(profile, t)
            if normalized is None:
                continue
            profile_width = float(profile.get('rotated_width', profile.get('width', footprint_width)))
            profile_height = float(profile.get('rotated_height', profile.get('height', 1.0)))
            if profile_width == 0:
                profile_width = footprint_width
            heights.append(normalized * profile_height * (footprint_width / profile_width))
        elif view in ('left', 'right'):
            if footprint_depth <= 0:
                continue
            t = (vertex.y - min_y) / footprint_depth
            t = max(0.0, min(1.0, t))
            normalized = sample_profile_height(profile, t)
            if normalized is None:
                continue
            profile_width = float(profile.get('rotated_width', profile.get('width', footprint_depth)))
            profile_height = float(profile.get('rotated_height', profile.get('height', 1.0)))
            if profile_width == 0:
                profile_width = footprint_depth
            heights.append(normalized * profile_height * (footprint_depth / profile_width))

    if not heights:
        return default_height

    return max(heights)


def apply_profile_deformation(obj, footprint, profiles, default_height):
    if not profiles:
        return

    min_x, max_x, min_y, max_y = get_polygon_bounds(footprint)
    footprint_bounds = (min_x, max_x, min_y, max_y)
    mesh = obj.data

    for vertex in mesh.vertices:
        if vertex.co.z <= 0.0:
            continue
        new_z = get_vertex_profile_height(vertex.co, footprint_bounds, profiles, default_height)
        vertex.co.z = new_z

    mesh.update()


def create_extruded_mesh(name, polygon, height, profiles=None, scale=1.0):
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    bm = bmesh.new()
    verts = []
    for x, y in polygon:
        verts.append(bm.verts.new((x, y, 0.0)))

    bm.verts.ensure_lookup_table()
    try:
        face = bm.faces.new(verts)
    except ValueError:
        face = bm.faces.new([v for v in verts if v.is_valid])

    bmesh.ops.recalc_face_normals(bm, faces=[face])

    extrude_result = bmesh.ops.extrude_face_region(bm, geom=[face])
    extruded_verts = [ele for ele in extrude_result['geom'] if isinstance(ele, bmesh.types.BMVert)]
    bmesh.ops.translate(bm, verts=extruded_verts, vec=mathutils.Vector((0.0, 0.0, height)))

    bm.to_mesh(mesh)
    bm.free()

    obj.location = (0.0, 0.0, 0.0)
    mesh.normals_split_custom_set_from_vertices([v.normal for v in mesh.vertices])
    mesh.use_auto_smooth = True
    mesh.auto_smooth_angle = math.radians(180.0)
    mesh.update()

    apply_profile_deformation(obj, polygon, profiles or [], height)

    # Apply Subdivision Surface and Bevel modifiers for a smoother Blender result
    subdiv_mod = obj.modifiers.new(name="Subdivision", type='SUBSURF')
    subdiv_mod.levels = 2
    subdiv_mod.render_levels = 3

    bevel_mod = obj.modifiers.new(name="Bevel", type='BEVEL')
    bevel_mod.width = max(0.01, height * 0.01)
    bevel_mod.segments = 3
    bevel_mod.profile = 0.7
    bevel_mod.limit_method = 'ANGLE'
    bevel_mod.angle_limit = math.radians(30.0)

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.modifier_apply(modifier=subdiv_mod.name)
    bpy.ops.object.modifier_apply(modifier=bevel_mod.name)
    obj.select_set(False)

    return obj


def face_side(face):
    normal = face.normal.normalized()
    if normal.z > 0.75:
        return 'top'
    if normal.z < -0.75:
        return 'bottom'
    if normal.y > 0.5:
        return 'front'
    if normal.y < -0.5:
        return 'back'
    if normal.x > 0.5:
        return 'right'
    if normal.x < -0.5:
        return 'left'
    return 'side'


def assign_uvs(obj):
    mesh = obj.data
    if not mesh.uv_layers:
        mesh.uv_layers.new(name='UVMap')
    uv_layer = mesh.uv_layers.active.data

    xs = [v.co.x for v in mesh.vertices]
    ys = [v.co.y for v in mesh.vertices]
    zs = [v.co.z for v in mesh.vertices]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)
    width = max(max_x - min_x, 1e-6)
    depth = max(max_y - min_y, 1e-6)
    height = max(max_z - min_z, 1e-6)

    for poly in mesh.polygons:
        side = face_side(poly)
        for loop_index in poly.loop_indices:
            loop = mesh.loops[loop_index]
            vert = mesh.vertices[loop.vertex_index]

            if side == 'top':
                u = (vert.co.x - min_x) / width
                v = (vert.co.y - min_y) / depth
            elif side in ('front', 'back'):
                u = (vert.co.x - min_x) / width
                v = (vert.co.z - min_z) / height
            elif side in ('left', 'right'):
                u = (vert.co.y - min_y) / depth
                v = (vert.co.z - min_z) / height
            else:
                u = (vert.co.x - min_x) / width
                v = (vert.co.y - min_y) / depth

            uv_layer[loop_index].uv = (u, v)


def get_polygon_bounds(polygon):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), max(xs), min(ys), max(ys)


def build_scene(payload):
    clear_scene()

    volumes = payload.get('volumes', [])
    textures = payload.get('textures', {})
    scale = payload.get('scale', 1.0)
    output_path = payload.get('output_path')

    footprints = []
    profiles = []

    for vlm in volumes:
        if vlm.get('type') == 'footprint':
            pts = vlm.get('contour', [])
            if pts:
                footprints.append(pts)
        elif vlm.get('type') == 'profile':
            profiles.append(vlm)

    if not footprints:
        raise RuntimeError('No footprint volumes found for BlenderBuilder')

    objects = []
    for index, footprint in enumerate(footprints):
        height = 30.0
        if profiles:
            best = profiles[0]
            for profile in profiles:
                if profile.get('height', 0) > best.get('height', 0):
                    best = profile
            height = float(best.get('height', 30.0))

        obj = create_extruded_mesh(
            f'blender_obj_{index}',
            footprint,
            height * float(scale),
            profiles=profiles,
            scale=scale
        )
        objects.append(obj)

    for obj in objects:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        assign_uvs(obj)
        obj.select_set(False)

        material_names = []
        material_map = {}
        for side in ('top', 'front', 'back', 'left', 'right'):
            entries = textures.get(side)
            if not entries:
                continue
            image_path, normal_path = entries
            if image_path and os.path.exists(image_path):
                mat = create_material(f'{side}_mat', image_path, normal_path)
                obj.data.materials.append(mat)
                material_names.append(side)
                material_map[side] = len(obj.data.materials) - 1

        if not material_map:
            continue

        for poly in obj.data.polygons:
            side = face_side(poly)
            if side in material_map:
                poly.material_index = material_map[side]
            else:
                poly.material_index = material_map.get('top', 0)

    if objects:
        bpy.ops.object.select_all(action='DESELECT')
        for obj in objects:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = objects[0]

    export_dir = os.path.dirname(output_path)
    if export_dir and not os.path.exists(export_dir):
        os.makedirs(export_dir, exist_ok=True)

    bpy.ops.export_scene.gltf(filepath=output_path, export_format='GLB', export_selected=False, export_materials='EXPORT')


if __name__ == '__main__':
    args = sys.argv
    if '--' not in args:
        raise RuntimeError('Missing BlenderBuilder payload path')

    payload_path = args[args.index('--') + 1]
    payload = load_payload(payload_path)
    build_scene(payload)
"""
