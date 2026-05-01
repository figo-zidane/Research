#include "scene/Camera.h"

#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>

namespace rr::scene
{

Camera::Camera()
{
    rebuild_matrices();
}

void Camera::set_perspective(float fov_y_degrees, float aspect,
                              float near_plane, float far_plane)
{
    fov_y_   = fov_y_degrees;
    aspect_  = aspect;
    near_    = near_plane;
    far_     = far_plane;
    rebuild_matrices();
}

void Camera::look_at(glm::vec3 eye, glm::vec3 center, glm::vec3 /*up*/)
{
    position_ = eye;
    pivot_    = center;

    glm::vec3 dir = glm::normalize(center - eye);
    pitch_ = glm::degrees(std::asin(dir.y));
    yaw_   = glm::degrees(std::atan2(dir.z, dir.x));
    rebuild_matrices();
}

bool Camera::update(GLFWwindow* window, float delta_time)
{
    const glm::mat4 old_vp = proj_ * view_;

    // WASD + QE movement when right mouse is held.
    bool moved = false;
    if (right_mouse_down_)
    {
        glm::vec3 fwd = forward();
        glm::vec3 right = glm::normalize(glm::cross(fwd, {0,1,0}));
        glm::vec3 up_dir = {0,1,0};

        float speed = move_speed_ * delta_time;
        if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
            speed *= 4.0f;

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) { position_ += fwd   * speed; moved = true; }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) { position_ -= fwd   * speed; moved = true; }
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) { position_ -= right * speed; moved = true; }
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) { position_ += right * speed; moved = true; }
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) { position_ += up_dir * speed; moved = true; }
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) { position_ -= up_dir * speed; moved = true; }
    }

    if (moved)
        rebuild_matrices();

    // Store previous VP for motion vectors.
    prev_view_proj_ = old_vp;

    return moved || (proj_ * view_ != old_vp);
}

void Camera::on_mouse_button(int button, int action, int /*mods*/)
{
    if (button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        right_mouse_down_ = (action == GLFW_PRESS);
        first_mouse_      = true;
    }
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        left_mouse_down_ = (action == GLFW_PRESS);
        first_mouse_     = true;
    }
}

void Camera::on_mouse_move(double xpos, double ypos)
{
    if (first_mouse_)
    {
        last_mouse_x_ = static_cast<float>(xpos);
        last_mouse_y_ = static_cast<float>(ypos);
        first_mouse_  = false;
        return;
    }

    float dx = static_cast<float>(xpos) - last_mouse_x_;
    float dy = last_mouse_y_ - static_cast<float>(ypos); // invert Y

    last_mouse_x_ = static_cast<float>(xpos);
    last_mouse_y_ = static_cast<float>(ypos);

    // Free look: right mouse dragged
    if (right_mouse_down_)
    {
        yaw_   += dx * look_speed_;
        pitch_ += dy * look_speed_;
        pitch_  = std::clamp(pitch_, -89.0f, 89.0f);
        rebuild_matrices();
    }
}

void Camera::on_scroll(double /*xoffset*/, double yoffset)
{
    glm::vec3 fwd = forward();
    position_ += fwd * (static_cast<float>(yoffset) * scroll_speed_);
    rebuild_matrices();
}

void Camera::on_resize(float aspect)
{
    aspect_ = aspect;
    rebuild_matrices();
}

glm::vec3 Camera::forward() const noexcept
{
    float yaw_r   = glm::radians(yaw_);
    float pitch_r = glm::radians(pitch_);
    return glm::normalize(glm::vec3(
        std::cos(pitch_r) * std::cos(yaw_r),
        std::sin(pitch_r),
        std::cos(pitch_r) * std::sin(yaw_r)));
}

void Camera::rebuild_matrices()
{
    glm::vec3 fwd  = forward();
    glm::vec3 right = glm::normalize(glm::cross(fwd, {0.0f, 1.0f, 0.0f}));
    // Handle edge case where forward is parallel to up
    if (glm::length(right) < 1e-5f)
        right = {1.0f, 0.0f, 0.0f};
    glm::vec3 up = glm::normalize(glm::cross(right, fwd));

    view_     = glm::lookAt(position_, position_ + fwd, up);
    proj_     = glm::perspective(glm::radians(fov_y_), aspect_, near_, far_);
    // Vulkan NDC: Y-flip the projection matrix.
    proj_[1][1] *= -1.0f;
    inv_view_ = glm::inverse(view_);
    inv_proj_ = glm::inverse(proj_);
}

} // namespace rr::scene
