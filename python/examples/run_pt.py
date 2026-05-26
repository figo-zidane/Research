"""Drive ReSTIR PT from Python: load a scene, set a camera, render, screenshot.

    cmake --preset vs2026-x64
    cmake --build --preset vs2026-debug --target research_renderer_py
    python python/examples/run_pt.py
"""

import _bootstrap

module_dir = _bootstrap.setup()
print(f"[run_pt] module dir: {module_dir}")

import research_renderer as rr

app = rr.Application()
app.initialize()

# Turn input off so mouse/WASD do not override the script-set pose.
app.set_input_enabled(False)

# The Cornell Box is built procedurally and always available.
app.load_cornell()
# To load a glTF/glb instead (path relative to the module dir), e.g.:
#   app.load_gltf("assets/scenes/sponza-gltf/sponza-png.glb")

app.set_pass_enabled("restir_di", True)
app.set_pass_enabled("restir_pt", True)
app.set_pass_enabled("denoise", True)
app.set_exposure(1.0)

# load_cornell()/load_gltf() are deferred and applied inside the next step(),
# and a scene (re)load resets the camera to its default. Flush the load with one
# step() first, then set the camera so the pose is not overwritten.
app.step()
app.set_camera_look_at(eye=(0.0, 1.0, 3.0), center=(0.0, 1.0, 0.0), up=(0.0, 1.0, 0.0))

for _ in range(100):
    if app.should_close():
        break
    app.step()

app.capture_screenshot("out/restir_pt_100f.png")
print(f"[run_pt] scene={app.scene_name} cam_pos={app.camera_position} size={app.window_size}")
app.shutdown()
