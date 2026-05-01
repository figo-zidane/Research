#pragma once

#include "rhi/AccelStructure.h"
#include "rhi/Buffer.h"
#include "rhi/Image.h"
#include "rhi/Sampler.h"
#include "scene/Camera.h"
#include "scene/GpuScene.h"

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace rr::rhi
{
class Device;
class BindlessRegistry;
} // namespace rr::rhi

namespace rr::scene
{

// One CPU-side mesh description.
struct Mesh
{
    std::string name;
    uint32_t    vertex_base         = 0;
    uint32_t    vertex_count        = 0;
    uint32_t    index_byte_offset   = 0;
    uint32_t    index_count         = 0;
    uint32_t    material_index      = 0;
};

// One CPU-side material.
struct Material
{
    std::string name;
    glm::vec4   base_color                   = {1,1,1,1};
    float       metallic                     = 0.0f;
    float       roughness                    = 1.0f;
    glm::vec3   emissive                     = {0,0,0};
    float       ior                          = 1.5f;
    int         albedo_image_idx             = -1; // index into images_
    int         normal_image_idx             = -1;
    int         metallic_roughness_image_idx = -1;
    int         emissive_image_idx           = -1;
};

// One CPU-side scene instance (mesh + transform).
struct Instance
{
    glm::mat4   transform  = glm::mat4(1.0f);
    uint32_t    mesh_index = 0;
};

// One CPU-side light.
struct Light
{
    uint32_t  type      = 0;     // 0=point, 1=directional
    glm::vec3 position  = {0,1,0};
    glm::vec3 direction = {0,-1,0};
    float     radius    = 0.0f;
    glm::vec3 emission  = {1,1,1};
    float     intensity = 1.0f;
};

// Scene owns all geometry, material, and acceleration structure data.
// Call upload() once to push data to the GPU.
class Scene
{
public:
    Scene();
    Scene(const Scene&) = delete;
    Scene& operator=(const Scene&) = delete;
    ~Scene() = default; // call destroy() explicitly

    // ── CPU-side scene building ────────────────────────────────────────────

    void set_name(std::string name);
    [[nodiscard]] const std::string& name() const noexcept { return name_; }

    // Append geometry for a new mesh.  Returns mesh index.
    uint32_t add_mesh(std::string name,
                      std::vector<GpuVertex> vertices,
                      std::vector<uint32_t>  indices,
                      uint32_t               material_index);

    uint32_t add_material(Material mat);
    uint32_t add_instance(Instance inst);
    uint32_t add_light(Light light);

    // Add an RGBA8 texture.  Returns image index for Material fields.
    uint32_t add_image_rgba8(std::string name,
                              uint32_t width, uint32_t height,
                              const std::vector<uint8_t>& pixels);

    // ── Cornell Box factory ────────────────────────────────────────────────
    void build_cornell_box();

    // ── GPU upload ────────────────────────────────────────────────────────
    // Upload to GPU, register with BindlessRegistry, build BLAS/TLAS.
    // one_time_submit: Application-provided function that accepts a recording
    // callback, allocates a fresh VkCommandBuffer, calls the callback to record
    // GPU work, then submits and waits for the GPU to finish.
    using OneTimeSubmitFn = std::function<void(std::function<void(VkCommandBuffer)>)>;
    void upload(rr::rhi::Device&           device,
                rr::rhi::BindlessRegistry&  registry,
                OneTimeSubmitFn             one_time_submit);

    // Free all GPU resources.
    void destroy(rr::rhi::Device& device);

    // ── Per-frame GPU update ───────────────────────────────────────────────
    // Write current camera matrices to the camera data buffer.
    void update_camera(const Camera& camera,
                       uint32_t screen_width, uint32_t screen_height,
                       uint32_t frame_index);

    // ── Accessors ─────────────────────────────────────────────────────────
    [[nodiscard]] const SceneGpuHandles& gpu_handles()     const noexcept { return handles_; }
    [[nodiscard]] uint32_t               instance_count()  const noexcept
    { return static_cast<uint32_t>(instances_.size()); }
    [[nodiscard]] uint32_t               light_count()     const noexcept
    { return static_cast<uint32_t>(lights_.size()); }
    [[nodiscard]] uint32_t               mesh_count()      const noexcept
    { return static_cast<uint32_t>(meshes_.size()); }
    [[nodiscard]] uint32_t               blas_count()      const noexcept
    { return static_cast<uint32_t>(blases_.size()); }
    [[nodiscard]] uint64_t               tlas_scratch_bytes() const noexcept
    { return tlas_scratch_bytes_; }
    [[nodiscard]] bool                   is_uploaded()     const noexcept { return uploaded_; }

    // Read back geometry for ImGui / stats.
    [[nodiscard]] const std::vector<Mesh>&     meshes()    const noexcept { return meshes_; }
    [[nodiscard]] const std::vector<Instance>& instances() const noexcept { return instances_; }

private:
    std::string name_ = "Untitled";

    // CPU-side arrays
    std::vector<Mesh>         meshes_;
    std::vector<Material>     materials_;
    std::vector<Instance>     instances_;
    std::vector<Light>        lights_;

    // Flattened geometry
    std::vector<GpuVertex>    all_vertices_;
    std::vector<uint32_t>     all_indices_;

    // Staging image data
    struct ImageData
    {
        std::string          name;
        uint32_t             width  = 0;
        uint32_t             height = 0;
        std::vector<uint8_t> pixels; // RGBA8 linear
    };
    std::vector<ImageData> images_;

    // GPU resources
    rr::rhi::Buffer vertex_buffer_;
    rr::rhi::Buffer index_buffer_;
    rr::rhi::Buffer mesh_buffer_;
    rr::rhi::Buffer material_buffer_;
    rr::rhi::Buffer instance_buffer_;
    rr::rhi::Buffer light_buffer_;
    rr::rhi::Buffer camera_buffer_;   // GpuCameraData, mapped every frame

    std::vector<rr::rhi::Image>   textures_;
    std::vector<rr::rhi::Sampler> samplers_;

    // Acceleration structures (one BLAS per unique mesh, one TLAS)
    std::vector<rr::rhi::AccelStructure> blases_;
    rr::rhi::AccelStructure              tlas_;

    SceneGpuHandles handles_;
    uint64_t        tlas_scratch_bytes_ = 0;
    bool            uploaded_           = false;
};

} // namespace rr::scene
