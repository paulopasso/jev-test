"""Process-level environment fixes applied BEFORE mujoco/robosuite imports.

robomimic 0.5.0 imports egl_probe (Linux EGL device probe) unconditionally in
EnvRobosuite.__init__ even though it is only needed for old robosuite (<1.5)
on Linux. macOS has no EGL; a stub returning no devices is the minimal fix.
MuJoCo offscreen rendering on macOS uses the GLFW/CGL backend.
"""
import os
import sys
import types


def setup():
    os.environ.setdefault("MUJOCO_GL", "glfw")
    if "egl_probe" not in sys.modules:
        stub = types.ModuleType("egl_probe")
        stub.get_available_devices = lambda: []
        sys.modules["egl_probe"] = stub


setup()
