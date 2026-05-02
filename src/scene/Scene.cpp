#include "scene/Scene.h"

#include "core/Log.h"
#include "rhi/AccelStructure.h"
#include "rhi/BindlessRegistry.h"
#include "rhi/Buffer.h"
#include "rhi/Device.h"
#include "rhi/Image.h"
#include "rhi/Sampler.h"
#include "scene/Camera.h"
#include "scene/GpuScene.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cstring>
#include <stdexcept>
#include <utility>

// VMA needed for VMA_ALLOCATION_CREATE_* flags in BufferDesc.alloc_flags.
// We include only the config header (no implementation).
#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vk_mem_alloc.h>

namespace rr::scene
{

Scene::Scene() = default;

void Scene::set_name(std::string name)
{
    name_ = std::move(name);
}

// ── Add helpers ──────────────────────────────────────────────────────

uint32_t Scene::add_mesh(std::string name,
                          std::vector<GpuVertex> vertices,
                          std::vector<uint32_t>  indices,
                          uint32_t               material_index)
{
    Mesh m;
    m.name             = std::move(name);
    m.vertex_base      = static_cast<uint32_t>(all_vertices_.size());
    m.vertex_count     = static_cast<uint32_t>(vertices.size());
    m.index_byte_offset = static_cast<uint32_t>(all_indices_.size() * sizeof(uint32_t));
    m.index_count      = static_cast<uint32_t>(indices.size());
    m.material_index   = material_index;

    all_vertices_.insert(all_vertices_.end(), vertices.begin(), vertices.end());
    all_indices_.insert(all_indices_.end(),   indices.begin(),  indices.end());

    uint32_t idx = static_cast<uint32_t>(meshes_.size());
    meshes_.push_back(std::move(m));
    return idx;
}

uint32_t Scene::add_material(Material mat)
{
    uint32_t idx = static_cast<uint32_t>(materials_.size());
    materials_.push_back(std::move(mat));
    return idx;
}

uint32_t Scene::add_instance(Instance inst)
{
    uint32_t idx = static_cast<uint32_t>(instances_.size());
    instances_.push_back(std::move(inst));
    return idx;
}

uint32_t Scene::add_light(Light light)
{
    uint32_t idx = static_cast<uint32_t>(lights_.size());
    lights_.push_back(std::move(light));
    return idx;
}

uint32_t Scene::add_image_rgba8(std::string name,
                                  uint32_t width, uint32_t height,
                                  const std::vector<uint8_t>& pixels)
{
    ImageData img;
    img.name   = std::move(name);
    img.width  = width;
    img.height = height;
    img.pixels = pixels;
    uint32_t idx = static_cast<uint32_t>(images_.size());
    images_.push_back(std::move(img));
    return idx;
}

// ── Cornell Box factory ──────────────────────────────────────────────────

namespace
{
// Build one quad (two triangles) with a given normal.
// Vertices: v0-v3 are four corners (counter-clockwise winding, front face outward).
void add_quad(std::vector<GpuVertex>& verts, std::vector<uint32_t>& idxs,
               glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, glm::vec3 v3,
               glm::vec3 normal,
               glm::vec2 uv0={0,0}, glm::vec2 uv1={1,0},
               glm::vec2 uv2={1,1}, glm::vec2 uv3={0,1})
{
    uint32_t base = static_cast<uint32_t>(verts.size());
    auto make_v = [&](glm::vec3 p, glm::vec2 uv) {
        GpuVertex v;
        v.pos[0]=p.x; v.pos[1]=p.y; v.pos[2]=p.z;
        v.normal[0]=normal.x; v.normal[1]=normal.y; v.normal[2]=normal.z;
        v.uv[0]=uv.x; v.uv[1]=uv.y;
        v.tangent[0]=1; v.tangent[1]=0; v.tangent[2]=0; v.tangent[3]=1;
        return v;
    };
    verts.push_back(make_v(v0, uv0));
    verts.push_back(make_v(v1, uv1));
    verts.push_back(make_v(v2, uv2));
    verts.push_back(make_v(v3, uv3));
    idxs.push_back(base+0); idxs.push_back(base+1); idxs.push_back(base+2);
    idxs.push_back(base+0); idxs.push_back(base+2); idxs.push_back(base+3);
}

// Build a box mesh from 8 corners.
void add_box(std::vector<GpuVertex>& verts, std::vector<uint32_t>& idxs,
              glm::vec3 mn, glm::vec3 mx)
{
    glm::vec3 a{mn.x,mn.y,mn.z};
    glm::vec3 b{mx.x,mn.y,mn.z};
    glm::vec3 c{mx.x,mn.y,mx.z};
    glm::vec3 d{mn.x,mn.y,mx.z};
    glm::vec3 e{mn.x,mx.y,mn.z};
    glm::vec3 f{mx.x,mx.y,mn.z};
    glm::vec3 g{mx.x,mx.y,mx.z};
    glm::vec3 h{mn.x,mx.y,mx.z};
    // Bottom
    add_quad(verts,idxs, d,c,b,a, {0,-1,0});
    // Top
    add_quad(verts,idxs, e,f,g,h, {0,1,0});
    // Front (+Z)
    add_quad(verts,idxs, d,h,g,c, {0,0,1});
    // Back (-Z)
    add_quad(verts,idxs, b,f,e,a, {0,0,-1});
    // Left (-X)
    add_quad(verts,idxs, a,e,h,d, {-1,0,0});
    // Right (+X)
    add_quad(verts,idxs, c,g,f,b, {1,0,0});
}
} // anonymous

void Scene::build_cornell_box()
{
    set_name("Cornell Box");

    // Materials
    Material white;
    white.name       = "white";
    white.base_color = {0.73f, 0.73f, 0.73f, 1.0f};
    white.metallic   = 0.0f;
    white.roughness  = 1.0f;
    uint32_t mat_white = add_material(white);

    Material red;
    red.name       = "red";
    red.base_color = {0.65f, 0.05f, 0.05f, 1.0f};
    red.metallic   = 0.0f;
    red.roughness  = 1.0f;
    uint32_t mat_red = add_material(red);

    Material green;
    green.name       = "green";
    green.base_color = {0.12f, 0.45f, 0.15f, 1.0f};
    green.metallic   = 0.0f;
    green.roughness  = 1.0f;
    uint32_t mat_green = add_material(green);

    Material light_mat;
    light_mat.name       = "light";
    light_mat.base_color = {1.0f, 1.0f, 1.0f, 1.0f};
    light_mat.emissive   = {15.0f, 15.0f, 15.0f};
    light_mat.roughness  = 1.0f;
    uint32_t mat_light = add_material(light_mat);

    // Room dimensions: -1 to +1 on all axes.
    // Back wall (z = -1)
    {
        std::vector<GpuVertex> v; std::vector<uint32_t> i;
        add_quad(v,i, {-1,-1,-1},{1,-1,-1},{1,1,-1},{-1,1,-1}, {0,0,1});
        add_mesh("back_wall", std::move(v), std::move(i), mat_white);
        add_instance({glm::mat4(1.0f), static_cast<uint32_t>(meshes_.size()-1)});
    }
    // Floor (y = -1)
    {
        std::vector<GpuVertex> v; std::vector<uint32_t> i;
        add_quad(v,i, {-1,-1,1},{1,-1,1},{1,-1,-1},{-1,-1,-1}, {0,1,0});
        add_mesh("floor", std::move(v), std::move(i), mat_white);
        add_instance({glm::mat4(1.0f), static_cast<uint32_t>(meshes_.size()-1)});
    }
    // Ceiling (y = +1)
    {
        std::vector<GpuVertex> v; std::vector<uint32_t> i;
        add_quad(v,i, {-1,1,-1},{1,1,-1},{1,1,1},{-1,1,1}, {0,-1,0});
        add_mesh("ceiling", std::move(v), std::move(i), mat_white);
        add_instance({glm::mat4(1.0f), static_cast<uint32_t>(meshes_.size()-1)});
    }
    // Left wall x=-1, red
    {
        std::vector<GpuVertex> v; std::vector<uint32_t> i;
        add_quad(v,i, {-1,-1,-1},{-1,-1,1},{-1,1,1},{-1,1,-1}, {1,0,0});
        add_mesh("left_wall", std::move(v), std::move(i), mat_red);
        add_instance({glm::mat4(1.0f), static_cast<uint32_t>(meshes_.size()-1)});
    }
    // Right wall x=+1, green
    {
        std::vector<GpuVertex> v; std::vector<uint32_t> i;
        add_quad(v,i, {1,-1,1},{1,-1,-1},{1,1,-1},{1,1,1}, {-1,0,0});
        add_mesh("right_wall", std::move(v), std::move(i), mat_green);
        add_instance({glm::mat4(1.0f), static_cast<uint32_t>(meshes_.size()-1)});
    }
    // Area light on ceiling (emissive quad)
    {
        std::vector<GpuVertex> v; std::vector<uint32_t> i;
        add_quad(v,i, {-0.25f,0.999f,-0.25f},{0.25f,0.999f,-0.25f},
                      {0.25f,0.999f,0.25f},{-0.25f,0.999f,0.25f}, {0,-1,0});
        add_mesh("ceiling_light", std::move(v), std::move(i), mat_light);
        add_instance({glm::mat4(1.0f), static_cast<uint32_t>(meshes_.size()-1)});
    }
    // Short box (0.5x0.5x0.5 at right-ish, rotated 18°)
    {
        std::vector<GpuVertex> v; std::vector<uint32_t> i;
        add_box(v, i, {-0.25f,-1.0f,-0.25f},{0.25f,-0.5f,0.25f});
        uint32_t mesh_idx = add_mesh("short_box", std::move(v), std::move(i), mat_white);
        Instance inst;
        inst.mesh_index = mesh_idx;
        inst.transform  = glm::translate(glm::mat4(1.0f), {0.35f, 0.0f, -0.35f})
                        * glm::rotate(glm::mat4(1.0f), glm::radians(18.0f), {0,1,0});
        add_instance(inst);
    }
    // Tall box (0.5x1.0x0.5 at left-ish, rotated -15°)
    {
        std::vector<GpuVertex> v; std::vector<uint32_t> i;
        add_box(v, i, {-0.25f,-1.0f,-0.25f},{0.25f,0.0f,0.25f});
        uint32_t mesh_idx = add_mesh("tall_box", std::move(v), std::move(i), mat_white);
        Instance inst;
        inst.mesh_index = mesh_idx;
        inst.transform  = glm::translate(glm::mat4(1.0f), {-0.35f, 0.0f, -0.2f})
                        * glm::rotate(glm::mat4(1.0f), glm::radians(-15.0f), {0,1,0});
        add_instance(inst);
    }

    // Add a point light for the ceiling light source
    Light l;
    l.type      = 0; // point
    l.position  = {0.0f, 0.9f, 0.0f};
    l.emission  = {15.0f, 15.0f, 12.0f};
    l.intensity = 1.0f;
    add_light(l);
}

// ── GPU upload ─────────────────────────────────────────────────────────

void Scene::upload(rr::rhi::Device&           device,
                    rr::rhi::BindlessRegistry&  registry,
                    OneTimeSubmitFn             one_time_submit)
{
    if (uploaded_)
        throw std::runtime_error("Scene::upload called twice.");
    if (all_vertices_.empty())
        throw std::runtime_error("Scene::upload: no vertices.");

    auto gpu_buf = [&](VkDeviceSize byte_size,
                        VkBufferUsageFlags usage,
                        const void* data,
                        const char* name) -> rr::rhi::Buffer
    {
        rr::rhi::BufferDesc desc{};
        desc.size        = byte_size;
        desc.usage       = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        desc.memory_usage = 3;  // VMA_MEMORY_USAGE_CPU_TO_GPU
        desc.alloc_flags  = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                            VMA_ALLOCATION_CREATE_MAPPED_BIT;
        desc.debug_name   = name;
        rr::rhi::Buffer buf;
        buf.create(device, desc);
        std::memcpy(buf.mapped(), data, byte_size);
        return buf;
    };

    // Vertex / index buffers
    vertex_buffer_ = gpu_buf(
        all_vertices_.size() * sizeof(GpuVertex),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        all_vertices_.data(), "scene_vertices");

    index_buffer_ = gpu_buf(
        all_indices_.size() * sizeof(uint32_t),
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        all_indices_.data(), "scene_indices");

    // Mesh descriptors
    std::vector<GpuMesh> gpu_meshes(meshes_.size());
    for (size_t i = 0; i < meshes_.size(); ++i)
    {
        gpu_meshes[i].vertex_base        = meshes_[i].vertex_base;
        gpu_meshes[i].vertex_count       = meshes_[i].vertex_count;
        gpu_meshes[i].index_byte_offset  = meshes_[i].index_byte_offset;
        gpu_meshes[i].index_count        = meshes_[i].index_count;
        gpu_meshes[i].material_index     = meshes_[i].material_index;
    }
    mesh_buffer_ = gpu_buf(
        gpu_meshes.size() * sizeof(GpuMesh),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        gpu_meshes.data(), "scene_meshes");

    // Material descriptors
    std::vector<GpuMaterial> gpu_mats(materials_.size());
    for (size_t i = 0; i < materials_.size(); ++i)
    {
        auto& src = materials_[i]; auto& dst = gpu_mats[i];
        dst.base_color[0] = src.base_color.r;
        dst.base_color[1] = src.base_color.g;
        dst.base_color[2] = src.base_color.b;
        dst.base_color[3] = src.base_color.a;
        dst.metallic  = src.metallic;
        dst.roughness = src.roughness;
        dst.emissive[0] = src.emissive.r;
        dst.emissive[1] = src.emissive.g;
        dst.emissive[2] = src.emissive.b;
        dst.ior       = src.ior;
        // Texture indices (registered later; 0xFFFFFFFF = none)
        dst.albedo_tex_idx              = UINT32_MAX;
        dst.normal_tex_idx              = UINT32_MAX;
        dst.metallic_roughness_tex_idx  = UINT32_MAX;
        dst.emissive_tex_idx            = UINT32_MAX;
        dst.sampler_idx                 = 0; // will be set after sampler registration
    }
    material_buffer_ = gpu_buf(
        gpu_mats.size() * sizeof(GpuMaterial),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        gpu_mats.data(), "scene_materials");

    // Instance descriptors
    std::vector<GpuInstance> gpu_insts(instances_.size());
    for (size_t i = 0; i < instances_.size(); ++i)
    {
        const glm::mat4& t = instances_[i].transform;
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                gpu_insts[i].transform[r*4+c] = t[r][c];
        gpu_insts[i].mesh_index = instances_[i].mesh_index;
    }
    instance_buffer_ = gpu_buf(
        gpu_insts.size() * sizeof(GpuInstance),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        gpu_insts.data(), "scene_instances");

    // Light descriptors
    std::vector<GpuLight> gpu_lights(lights_.size());
    for (size_t i = 0; i < lights_.size(); ++i)
    {
        auto& src = lights_[i]; auto& dst = gpu_lights[i];
        dst.position[0] = src.position.x;
        dst.position[1] = src.position.y;
        dst.position[2] = src.position.z;
        dst.type        = src.type;
        dst.direction[0] = src.direction.x;
        dst.direction[1] = src.direction.y;
        dst.direction[2] = src.direction.z;
        dst.radius     = src.radius;
        dst.emission[0] = src.emission.r;
        dst.emission[1] = src.emission.g;
        dst.emission[2] = src.emission.b;
        dst.intensity  = src.intensity;
    }
    if (!gpu_lights.empty())
    {
        light_buffer_ = gpu_buf(
            gpu_lights.size() * sizeof(GpuLight),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            gpu_lights.data(), "scene_lights");
    }
    else
    {
        // Allocate a placeholder 1-element buffer so the SSBO index is valid.
        GpuLight dummy{};
        light_buffer_ = gpu_buf(sizeof(GpuLight),
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                 &dummy, "scene_lights_placeholder");
    }

    // Camera data buffer (persistent mapped, updated per-frame)
    {
        rr::rhi::BufferDesc desc{};
        desc.size        = sizeof(GpuCameraData);
        desc.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        desc.memory_usage = 3;
        desc.alloc_flags  = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                            VMA_ALLOCATION_CREATE_MAPPED_BIT;
        desc.debug_name   = "camera_data";
        camera_buffer_.create(device, desc);
    }

    // Samplers — the BindlessRegistry creates and owns the VkSamplers.
    {
        handles_.linear_sampler_idx  = registry.register_sampler(device, rr::rhi::Sampler::linear_repeat());
        handles_.nearest_sampler_idx = registry.register_sampler(device, rr::rhi::Sampler::nearest_clamp());
    }

    // Textures
    textures_.resize(images_.size());
    for (size_t i = 0; i < images_.size(); ++i)
    {
        auto& img_data = images_[i];
        rr::rhi::ImageDesc desc{};
        desc.format  = rr::rhi::Format::R8G8B8A8_Srgb;
        desc.extent  = {img_data.width, img_data.height, 1};
        desc.usage   = rr::rhi::ImageUsage::Sampled | rr::rhi::ImageUsage::HostTransfer;
        desc.debug_name = img_data.name.c_str();
        textures_[i].create(device, desc);
        textures_[i].upload_host(
            device,
            img_data.pixels.data(),
            img_data.pixels.size(),
            rr::rhi::ImageLayout::ShaderReadOnly);
    }

    // Update material texture indices now that images are registered.
    // Re-write the material buffer after updating texture/sampler indices.
    for (size_t i = 0; i < materials_.size(); ++i)
    {
        auto& src = materials_[i]; auto& dst = gpu_mats[i];
        dst.sampler_idx = handles_.linear_sampler_idx;
        if (src.albedo_image_idx >= 0)
            dst.albedo_tex_idx = registry.register_texture(
                device,
                textures_[src.albedo_image_idx],
                rr::rhi::Format::R8G8B8A8_Srgb,
                rr::rhi::ImageLayout::ShaderReadOnly,
                rr::rhi::ImageAspect::Color);
    }
    std::memcpy(material_buffer_.mapped(), gpu_mats.data(),
                gpu_mats.size() * sizeof(GpuMaterial));

    // Register scene buffers with BindlessRegistry
    handles_.vertex_buf_idx  = registry.register_buffer(device,
        vertex_buffer_.device_address(), vertex_buffer_.size());
    handles_.index_buf_idx   = registry.register_buffer(device,
        index_buffer_.device_address(), index_buffer_.size());
    handles_.mesh_buf_idx    = registry.register_buffer(device,
        mesh_buffer_.device_address(), mesh_buffer_.size());
    handles_.material_buf_idx = registry.register_buffer(device,
        material_buffer_.device_address(), material_buffer_.size());
    handles_.instance_buf_idx = registry.register_buffer(device,
        instance_buffer_.device_address(), instance_buffer_.size());
    handles_.light_buf_idx   = registry.register_buffer(device,
        light_buffer_.device_address(), light_buffer_.size());
    handles_.camera_buf_idx  = registry.register_buffer(device,
        camera_buffer_.device_address(), camera_buffer_.size());

    // Build BLAS for each unique mesh (one-time GPU work).
    // We batch all builds in a single command buffer submission.
    std::vector<rr::rhi::Buffer> scratch_bufs(meshes_.size());
    blases_.resize(meshes_.size());

    one_time_submit([&](VkCommandBuffer cmd)
    {
        for (size_t i = 0; i < meshes_.size(); ++i)
        {
            const Mesh& m = meshes_[i];
            VkDeviceAddress vb_addr =
                vertex_buffer_.device_address() + m.vertex_base * sizeof(GpuVertex);
            VkDeviceAddress ib_addr =
                index_buffer_.device_address() + m.index_byte_offset;
            blases_[i] = rr::rhi::build_blas(
                device, cmd,
                vb_addr, sizeof(GpuVertex), m.vertex_count,
                ib_addr, m.index_count,
                false, scratch_bufs[i]);
        }
    });

    // Clean up BLAS scratch buffers
    for (auto& s : scratch_bufs)
        s.destroy(device);

    // Build TLAS
    std::vector<VkAccelerationStructureInstanceKHR> tlas_insts;
    tlas_insts.reserve(instances_.size());
    for (size_t i = 0; i < instances_.size(); ++i)
    {
        const Instance& inst = instances_[i];
        const Mesh& m = meshes_[inst.mesh_index];
        // Build row-major 4x4 from glm column-major
        float row_major[16];
        const glm::mat4& t = inst.transform;
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                row_major[r*4+c] = t[c][r];  // glm stores column-major
        (void)m;
        tlas_insts.push_back(rr::rhi::make_tlas_instance(
            row_major,
            blases_[inst.mesh_index].device_address(),
            static_cast<uint32_t>(i)));
    }

    rr::rhi::Buffer tlas_scratch;
    rr::rhi::Buffer tlas_instance_buf;
    one_time_submit([&](VkCommandBuffer cmd)
    {
        tlas_ = rr::rhi::build_tlas(device, cmd, tlas_insts, tlas_scratch, tlas_instance_buf);
    });
    tlas_scratch.destroy(device);
    tlas_instance_buf.destroy(device);
    tlas_scratch_bytes_ = 0; // scratch was temporary

    // Register TLAS with BindlessRegistry
    handles_.tlas_idx = registry.register_accel_struct(
        device, tlas_.device_address(), tlas_.buffer().size());

    uploaded_ = true;

    core::log()->info("Scene '{}' uploaded: {} meshes, {} instances, {} lights, "
                       "{} BLASes, TLAS at gTLAS[{}]",
                       name_, meshes_.size(), instances_.size(), lights_.size(),
                       blases_.size(), handles_.tlas_idx);
}

void Scene::clear_cpu_data()
{
    name_         = "Untitled";
    meshes_.clear();
    materials_.clear();
    instances_.clear();
    lights_.clear();
    all_vertices_.clear();
    all_indices_.clear();
    images_.clear();
    handles_ = SceneGpuHandles{};
}

void Scene::destroy(rr::rhi::Device& device)
{
    tlas_.destroy(device);
    for (auto& b : blases_) b.destroy(device);
    blases_.clear();

    for (auto& t : textures_) t.destroy(device);
    textures_.clear();
    for (auto& s : samplers_) s.destroy(device);
    samplers_.clear();

    camera_buffer_.destroy(device);
    light_buffer_.destroy(device);
    instance_buffer_.destroy(device);
    material_buffer_.destroy(device);
    mesh_buffer_.destroy(device);
    index_buffer_.destroy(device);
    vertex_buffer_.destroy(device);

    uploaded_ = false;
}

// ── Per-frame camera update ────────────────────────────────────────────

void Scene::update_camera(const Camera& camera,
                           uint32_t screen_width, uint32_t screen_height,
                           uint32_t frame_index)
{
    if (!camera_buffer_.is_valid()) return;
    GpuCameraData cam{};
    // Fill matrices (glm is column-major; we store row-major in the struct
    // matching how Slang reads float4x4 by default with row_major)
    auto copy_mat = [](float* dst, const glm::mat4& m) {
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                dst[r*4+c] = m[c][r]; // column-major glm → row-major
    };
    copy_mat(cam.view,           camera.view());
    copy_mat(cam.inv_view,       camera.inv_view());
    copy_mat(cam.inv_proj,       camera.inv_projection());
    copy_mat(cam.prev_view_proj, camera.prev_view_proj());

    glm::vec3 pos = camera.position();
    cam.position[0] = pos.x; cam.position[1] = pos.y; cam.position[2] = pos.z;
    cam.near_plane  = camera.near_plane();

    glm::vec3 dir = camera.forward();
    cam.direction[0] = dir.x; cam.direction[1] = dir.y; cam.direction[2] = dir.z;
    cam.far_plane   = camera.far_plane();

    cam.frame_index   = frame_index;
    cam.screen_width  = screen_width;
    cam.screen_height = screen_height;

    std::memcpy(camera_buffer_.mapped(), &cam, sizeof(GpuCameraData));
}

} // namespace rr::scene
