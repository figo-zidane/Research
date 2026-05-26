"""Minimal smoke test: open the window, render a few frames of the Cornell Box.

    python python/examples/cornell_smoke.py
"""

import _bootstrap

_bootstrap.setup()

import research_renderer as rr

app = rr.Application()
app.initialize()
app.load_cornell()

for _ in range(30):
    if app.should_close():
        break
    app.step()

app.capture_screenshot("out/cornell_smoke.png")
print(f"[smoke] scene={app.scene_name} size={app.window_size} exposure={app.exposure}")
app.shutdown()
