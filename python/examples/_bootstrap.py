"""Locate the built research_renderer module and prepare the working directory.

The module (research_renderer.pyd) is emitted next to a copy of assets/ and
slang.dll by the CMake POST_BUILD steps. We add its directory to sys.path and
chdir into it so the renderer's relative "assets/..." paths resolve.

Build it with, e.g.:
    cmake --preset vs2026-x64
    cmake --build --preset vs2026-debug --target research_renderer_py
"""

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def _config_rank(p):
    # Prefer Release: debug Python extensions are fragile on Windows (they pull
    # the debug CRT, which is not always present/compatible at run time).
    name = p.name.lower()
    return {"release": 0, "relwithdebinfo": 1, "minsizerel": 2, "debug": 3}.get(name, 1)


def _candidate_dirs():
    build = _REPO / "build"
    # VS multi-config (preset vs2026-x64): build/vs2026/src/<Config>/
    yield from sorted(build.glob("vs2026/src/*"), key=_config_rank)
    # Ninja single-config (preset windows-msvc-x64-*): build/<preset>/src/
    yield from sorted(build.glob("*/src"))
    # Generic fallback: anywhere a research_renderer module landed.
    yield from (p.parent for p in build.glob("**/research_renderer*.pyd"))
    yield from (p.parent for p in build.glob("**/research_renderer*.so"))


def setup():
    for d in _candidate_dirs():
        if not d.is_dir():
            continue
        hits = list(d.glob("research_renderer*.pyd")) + list(d.glob("research_renderer*.so"))
        if hits:
            module_dir = hits[0].parent
            sys.path.insert(0, str(module_dir))
            # Since Python 3.8, extension-module DLLs are not resolved from PATH
            # or the CWD — directories must be registered explicitly.
            if hasattr(os, "add_dll_directory"):
                # slang.dll, glfw3.dll, ... sit next to the .pyd.
                os.add_dll_directory(str(module_dir))
                # slang.dll lazily loads sibling toolchain DLLs that live only
                # in the Vulkan SDK Bin (they are not copied next to the .pyd).
                vk = os.environ.get("VULKAN_SDK")
                if vk:
                    vk_bin = Path(vk) / "Bin"
                    if vk_bin.is_dir():
                        os.add_dll_directory(str(vk_bin))
            os.chdir(module_dir)
            return module_dir
    raise RuntimeError(
        "research_renderer module not found under build/. "
        "Build the research_renderer_py target first."
    )
