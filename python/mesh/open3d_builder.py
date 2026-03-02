import open3d as o3d
from .base_mesh_builder import BaseMeshBuilder

class Open3DBuilder(BaseMeshBuilder):
  
    def __init__(self, debug=False, debug_dir="output/debug"):
      
      #DEBUG
      self.debug = debug
      self.debug_dir = debug_dir
     
      if self.debug:
        os.makedirs(self.debug_dir, exist_ok=True)
      #
    
    def build(self, contours, height_map=None):
        meshes = []

        #DEBUG
        if self.debug:
          os.makedirs(self.debug_dir, exist_ok=True)
        #
        
        for i, vlm in enumerate(volumes):

          # Create box (width, height, depth)
          box = o3d.geometry.TriangleMesh.create_box(
             width=vlm["width"],
             height=vlm["height"],
             depth=vlm["depth"]
          )

          # Move it
          box.translate([vlm["x"], 0, vlm["y"]])

          # Compute normals (important for rendering)
          box.compute_vertex_normals()

          meshes.append(box)
          
          #DEBUG: export individual
          if self.debug:
              o3d.io.write_triangle_mesh(
                  os.path.join(self.debug_dir, f"volume_{i}.ply"),
                  box 
              )
          #
          
        # Combine meshes
        combined = meshes[0]
        for mesh in meshes[1:]:
            combined += mesh

        combined.compute_vertex_normals()

        #DEBUG: export combined
        if self.debug:
            o3d.io.write_triangle_mesh(
                os.path.join(self.debug_dir, "combined_debug.ply"),
                combined
            )
        #
        
        return combined
          