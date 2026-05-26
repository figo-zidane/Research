// Python bindings for the Research Renderer.
//
// Builds into research_renderer.pyd. The module exposes a single class,
// Application, mirroring src/app/Application.h. Vectors cross the ABI as plain
// float triples (std::array<float,3>) so glm types never leak into Python.
//
// Threading: GLFW / Vulkan / nanobind all assume the main thread. step() is
// wrapped in nb::gil_scoped_release so Ctrl-C and other Python threads stay
// responsive while a frame is in flight.

#include "app/Application.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <array>
#include <string>

namespace nb = nanobind;
using rr::app::Application;

NB_MODULE(research_renderer, m)
{
    m.doc() = "Python bindings for the Research Renderer (Vulkan path tracer).";

    nb::class_<Application>(m, "Application")
        .def(nb::init<>(),
             "Construct the application. GLFW/Vulkan stay uninitialized until "
             "initialize() is called. Only one instance may exist at a time.")

        // ── Lifecycle ──────────────────────────────────────────────────────
        .def("initialize", &Application::initialize,
             "Create the window + RHI and load the default Cornell Box scene.")
        .def("shutdown", &Application::shutdown,
             "Tear down all GPU/window resources. Idempotent; also runs on GC.")
        .def("step", &Application::step,
             nb::call_guard<nb::gil_scoped_release>(),
             "Pump window events and render exactly one frame.")
        .def("should_close", &Application::should_close,
             "True once the window has been asked to close.")
        .def("run_until_closed", &Application::run_until_closed,
             nb::call_guard<nb::gil_scoped_release>(),
             "Run the frame loop on the C++ side until the window closes.")

        // ── Scene ──────────────────────────────────────────────────────────
        .def("load_cornell", &Application::load_cornell,
             "Queue a reload of the built-in Cornell Box scene.")
        .def("load_gltf", &Application::load_gltf, nb::arg("path"),
             "Queue loading a glTF/glb scene from the given path.")
        .def_prop_ro("scene_name", &Application::scene_name,
                     "Name of the currently loaded scene.")

        // ── Camera ─────────────────────────────────────────────────────────
        .def("set_camera_look_at",
             [](Application& self,
                std::array<float, 3> eye,
                std::array<float, 3> center,
                std::array<float, 3> up)
             {
                 self.set_camera_look_at(eye[0], eye[1], eye[2],
                                         center[0], center[1], center[2],
                                         up[0], up[1], up[2]);
             },
             nb::arg("eye"), nb::arg("center"),
             nb::arg("up") = std::array<float, 3>{0.0f, 1.0f, 0.0f},
             "Point the camera, resetting progressive accumulation.")
        .def_prop_ro("camera_position", &Application::camera_position,
                     "Camera eye position as (x, y, z).")
        .def_prop_ro("camera_forward", &Application::camera_forward,
                     "Camera forward direction as (x, y, z).")

        // ── Pass enable/disable ────────────────────────────────────────────
        .def("set_pass_enabled", &Application::set_pass_enabled,
             nb::arg("name"), nb::arg("enabled"),
             "Enable/disable a pass: 'restir_di' | 'restir_gi' | 'restir_pt' | "
             "'denoise'. PT/GI exclusivity is resolved internally.")

        // ── Tonemap ────────────────────────────────────────────────────────
        .def("set_exposure", &Application::set_exposure, nb::arg("exposure"),
             "Set the tonemap exposure multiplier.")
        .def_prop_ro("exposure", &Application::exposure,
                     "Current tonemap exposure multiplier.")

        // ── Input ──────────────────────────────────────────────────────────
        .def("set_input_enabled", &Application::set_input_enabled,
             nb::arg("enabled"),
             "When False, mouse/WASD no longer override the script-set camera.")
        .def_prop_ro("input_enabled", &Application::input_enabled,
                     "Whether interactive mouse/keyboard camera control is on.")

        // ── Screenshot ─────────────────────────────────────────────────────
        .def("capture_screenshot", &Application::capture_screenshot,
             nb::arg("path"),
             "Save the currently displayed image (tonemapped LDR) to a PNG.")

        // ── Window ─────────────────────────────────────────────────────────
        .def_prop_ro("window_size", &Application::window_size,
                     "Framebuffer size as (width, height).");
}
