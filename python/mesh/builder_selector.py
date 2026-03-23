def get_mesh_builder(method="trimesh"):

    if method == "trimesh":
        from mesh.trimesh_builder import TrimeshBuilder
        return TrimeshBuilder()

    elif method == "open3d":
        from mesh.open3d_builder import Open3DBuilder
        return Open3DBuilder()

    else:
        raise ValueError("Unknown mesh method")
