#!/usr/bin/env python
"""Interactive launcher for the Research Renderer — a Python drop-in for the
research-renderer.exe entry point (src/main.cpp).

Like the executable it opens the window and runs until you close it, with the
usual mouse + WASD camera controls. Command-line options let you preload a
scene, tweak passes/exposure, and grab a screenshot before exiting.

    python research_renderer.py
    python research_renderer.py assets/scenes/sponza-gltf/sponza-png.glb
    python research_renderer.py --pass restir_di --pass restir_pt --denoise
    python research_renderer.py --no-input --frames 200 --screenshot out/shot.png

Build the module first:
    cmake --preset vs2026-x64
    cmake --build --preset vs2026-release --target research_renderer_py
"""

import argparse
import importlib.util
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent


def _load_extension():
    """Import the compiled research_renderer extension and set up its runtime.

    This file is named research_renderer.py, which would shadow the
    research_renderer extension on a bare ``import``. We reuse the example
    bootstrap to locate the build output and register its DLL directories, then
    load the .pyd explicitly by path so there is no name clash.
    """
    sys.path.insert(0, str(_REPO / "python" / "examples"))
    import _bootstrap

    module_dir = _bootstrap.setup()  # adds DLL dirs + chdirs into the build output

    hits = list(module_dir.glob("research_renderer*.pyd")) + \
        list(module_dir.glob("research_renderer*.so"))
    # The extension's init symbol is PyInit_research_renderer, so it must be
    # loaded under exactly that name. This script runs as __main__ (not imported
    # as research_renderer), so there is no clash; register it in sys.modules so
    # any later `import research_renderer` resolves to the extension.
    spec = importlib.util.spec_from_file_location("research_renderer", str(hits[0]))
    module = importlib.util.module_from_spec(spec)
    sys.modules["research_renderer"] = module
    spec.loader.exec_module(module)
    return module


def _parse_args(argv):
    p = argparse.ArgumentParser(
        prog="research_renderer.py",
        description="Interactive Research Renderer launcher (Python entry point).",
    )
    p.add_argument("scene", nargs="?", default=None,
                   help="Path to a glTF/glb scene to load at startup "
                        "(relative to the build output dir). Defaults to the "
                        "built-in Cornell Box.")
    p.add_argument("--pass", dest="passes", action="append", default=[],
                   metavar="NAME",
                   choices=["restir_di", "restir_gi", "restir_pt"],
                   help="Enable a pass (repeatable): restir_di | restir_gi | "
                        "restir_pt.")
    p.add_argument("--denoise", action="store_true",
                   help="Enable the A-trous denoiser.")
    p.add_argument("--exposure", type=float, default=None, metavar="X",
                   help="Set tonemap exposure.")
    p.add_argument("--no-input", action="store_true",
                   help="Disable mouse/WASD camera control.")
    p.add_argument("--frames", type=int, default=None, metavar="N",
                   help="Render exactly N frames then exit (default: run until "
                        "the window is closed).")
    p.add_argument("--screenshot", default=None, metavar="PATH",
                   help="Save a PNG of the final frame before exiting.")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    rr = _load_extension()

    app = rr.Application()
    app.initialize()

    if args.no_input:
        app.set_input_enabled(False)

    # Scene + pass setup. load_*() is applied inside the next step(), and a
    # scene (re)load resets the camera, so flush it with one step() before
    # touching camera-dependent state.
    if args.scene:
        app.load_gltf(args.scene)
        app.step()

    for name in args.passes:
        app.set_pass_enabled(name, True)
    if args.denoise:
        app.set_pass_enabled("denoise", True)
    if args.exposure is not None:
        app.set_exposure(args.exposure)

    # Drive the frame loop.
    if args.frames is not None:
        for _ in range(args.frames):
            if app.should_close():
                break
            app.step()
    else:
        # Equivalent to main.cpp's Application::run(): block until the window
        # closes. run_until_closed() releases the GIL so Ctrl-C stays live.
        app.run_until_closed()

    if args.screenshot:
        app.capture_screenshot(args.screenshot)

    app.shutdown()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:  # mirror main.cpp: log + non-zero exit
        print(f"Unhandled exception: {exc}", file=sys.stderr)
        sys.exit(1)
