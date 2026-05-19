from .trimesh_builder import TrimeshBuilder
from .open3d_builder import Open3DBuilder
from .blender_builder import BlenderBuilder

def get_mesh_builder(method="trimesh"):

    if method == "trimesh":
        return TrimeshBuilder()

    elif method == "open3d":
        return Open3DBuilder()

    elif method == "blender":
        return BlenderBuilder()

    else:
        raise ValueError("Unknown mesh method")