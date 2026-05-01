#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

struct GLFWwindow;

namespace rr::scene
{

// Camera supports two control modes:
//   FreeLook — WASD + right-mouse-drag (first-person fly).
//   Orbit    — alt + left-mouse-drag around a pivot point.
//
// Camera matrices are updated once per frame via update().
// The caller is responsible for passing GLFW window events by calling
// on_mouse_button() / on_mouse_move() from GLFW callbacks.
class Camera
{
public:
    Camera();

    // ── Configuration ─────────────────────────────────────────────────────
    void set_perspective(float fov_y_degrees, float aspect,
                         float near_plane, float far_plane);
    void look_at(glm::vec3 eye, glm::vec3 center, glm::vec3 up = {0,1,0});

    // ── Per-frame update ───────────────────────────────────────────────────
    // Call once per frame with the GLFW window for keyboard polling.
    // delta_time is in seconds.
    // Returns true if the camera moved or rotated this frame.
    bool update(GLFWwindow* window, float delta_time);

    // ── GLFW event callbacks ───────────────────────────────────────────────
    void on_mouse_button(int button, int action, int mods);
    void on_mouse_move(double xpos, double ypos);
    void on_resize(float aspect);

    // ── Accessors ─────────────────────────────────────────────────────────
    [[nodiscard]] const glm::mat4& view()           const noexcept { return view_; }
    [[nodiscard]] const glm::mat4& projection()     const noexcept { return proj_; }
    [[nodiscard]] const glm::mat4& inv_view()       const noexcept { return inv_view_; }
    [[nodiscard]] const glm::mat4& inv_projection() const noexcept { return inv_proj_; }
    [[nodiscard]] const glm::mat4& prev_view_proj() const noexcept { return prev_view_proj_; }
    [[nodiscard]] glm::vec3        position()       const noexcept { return position_; }
    [[nodiscard]] glm::vec3        forward()        const noexcept;
    [[nodiscard]] float            near_plane()     const noexcept { return near_; }
    [[nodiscard]] float            far_plane()      const noexcept { return far_; }
    [[nodiscard]] float            fov_y()          const noexcept { return fov_y_; }

private:
    void rebuild_matrices();

    // Intrinsics
    float fov_y_  = 45.0f;
    float aspect_ = 16.0f / 9.0f;
    float near_   = 0.01f;
    float far_    = 1000.0f;

    // Extrinsics (spherical / orbit representation)
    glm::vec3 position_  = {0.0f, 0.0f,  3.0f};
    glm::vec3 pivot_     = {0.0f, 0.0f,  0.0f};
    float     yaw_       = -90.0f; // degrees, looking toward -Z
    float     pitch_     = 0.0f;   // degrees

    // Mouse state
    bool  right_mouse_down_ = false;
    bool  left_mouse_down_  = false;
    bool  alt_held_         = false;
    float last_mouse_x_     = 0.0f;
    float last_mouse_y_     = 0.0f;
    bool  first_mouse_      = true;

    // Movement speed
    float move_speed_   = 2.0f;  // m/s
    float look_speed_   = 0.15f; // degrees / pixel

    // Matrices
    glm::mat4 view_           = glm::mat4(1.0f);
    glm::mat4 proj_           = glm::mat4(1.0f);
    glm::mat4 inv_view_       = glm::mat4(1.0f);
    glm::mat4 inv_proj_       = glm::mat4(1.0f);
    glm::mat4 prev_view_proj_ = glm::mat4(1.0f);
};

} // namespace rr::scene
