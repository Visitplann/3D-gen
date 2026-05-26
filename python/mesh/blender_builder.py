import json
import os
import subprocess
import tempfile
from .base_mesh_builder import BaseMeshBuilder


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

        # Compute dimensions and add to payload for Blender
        footprints = [vlm for vlm in serializable_volumes if vlm.get('type') == 'footprint']
        profiles = [vlm for vlm in serializable_volumes if vlm.get('type') == 'profile']
        
        object_dimensions = []
        for fp in footprints:
            footprint_width = float(fp.get('rotated_width', fp.get('width', 0)))
            footprint_depth = float(fp.get('rotated_depth', fp.get('depth', 0)))
            height = 30.0
            if profiles:
                preferred_order = ['front', 'right', 'left', 'back']
                def priority(profile):
                    view = profile.get('view')
                    return preferred_order.index(view) if view in preferred_order else len(preferred_order)
                best = min(profiles, key=priority)
                if priority(best) < len(preferred_order):
                    raw_height = float(best.get('height', 30.0))
                    profile_width = float(best.get('rotated_width', best.get('width', 0)))
                    profile_view = best.get('view')
                    if profile_view in ('front', 'back') and profile_width > 0:
                        height = raw_height * (footprint_width / profile_width)
                    elif profile_view in ('left', 'right') and profile_width > 0:
                        height = raw_height * (footprint_depth / profile_width)
                    else:
                        height = raw_height
            
            scaled_w = footprint_width * overall_scale
            scaled_d = footprint_depth * overall_scale
            scaled_h = height * overall_scale
            object_dimensions.append({"width": scaled_w, "depth": scaled_d, "height": scaled_h})

        payload = {
            "volumes": serializable_volumes,
            "scale": overall_scale,
            "complex_mode": complex_mode,
            "output_path": output_path,
            "textures": textures,
            "object_dimensions": object_dimensions,
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
        # Load the external Blender script file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, "blender_script.py")
        with open(script_path, "r", encoding="utf-8") as f:
            return f.read()
