#include "scene/GltfLoader.h"

#include "core/Log.h"
#include "scene/GpuScene.h"
#include "scene/Scene.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// Suppress tinygltf warnings on MSVC
#ifdef _MSC_VER
#  pragma warning(push, 0)
#endif
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tiny_gltf.h>
#ifdef _MSC_VER
#  pragma warning(pop)
#endif

#include <filesystem>
#include <string>
#include <vector>

namespace rr::scene
{

namespace
{

// Convert a tinygltf::Node's TRS or matrix to a glm::mat4.
glm::mat4 node_transform(const tinygltf::Node& node)
{
    if (!node.matrix.empty())
    {
        glm::mat4 m;
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                m[c][r] = static_cast<float>(node.matrix[r * 4 + c]);
        return m;
    }
    glm::mat4 t(1.0f);
    if (!node.translation.empty())
        t = glm::translate(t, {(float)node.translation[0],
                                (float)node.translation[1],
                                (float)node.translation[2]});
    if (!node.rotation.empty())
    {
        glm::quat q((float)node.rotation[3],
                     (float)node.rotation[0],
                     (float)node.rotation[1],
                     (float)node.rotation[2]);
        t = t * glm::mat4_cast(q);
    }
    if (!node.scale.empty())
        t = glm::scale(t, {(float)node.scale[0],
                            (float)node.scale[1],
                            (float)node.scale[2]});
    return t;
}

// Helper: get a typed raw buffer from a gltf accessor.
template<typename T>
const T* accessor_data(const tinygltf::Model& model, int accessor_idx, size_t& count)
{
    if (accessor_idx < 0) { count = 0; return nullptr; }
    const auto& acc  = model.accessors[accessor_idx];
    const auto& bv   = model.bufferViews[acc.bufferView];
    const auto& buf  = model.buffers[bv.buffer];
    count = acc.count;
    const uint8_t* ptr = buf.data.data() + bv.byteOffset + acc.byteOffset;
    return reinterpret_cast<const T*>(ptr);
}

} // anonymous

bool GltfLoader::load(const std::filesystem::path& path, Scene& scene)
{
    tinygltf::TinyGLTF loader;
    tinygltf::Model    model;
    std::string        err, warn;

    bool ok = false;
    if (path.extension() == ".glb")
        ok = loader.LoadBinaryFromFile(&model, &err, &warn, path.string());
    else
        ok = loader.LoadASCIIFromFile(&model, &err, &warn, path.string());

    if (!warn.empty())
        core::log()->warn("[GltfLoader] {}", warn);
    if (!ok)
    {
        core::log()->error("[GltfLoader] Failed to load '{}': {}", path.string(), err);
        return false;
    }

    scene.set_name(path.stem().string());

    // ── Materials ──────────────────────────────────────────────────────────
    std::vector<uint32_t> mat_remap(model.materials.size());
    for (size_t mi = 0; mi < model.materials.size(); ++mi)
    {
        const auto& gltf_mat = model.materials[mi];
        Material mat;
        mat.name = gltf_mat.name;

        const auto& pbr = gltf_mat.pbrMetallicRoughness;
        if (!pbr.baseColorFactor.empty())
        {
            mat.base_color = {
                (float)pbr.baseColorFactor[0],
                (float)pbr.baseColorFactor[1],
                (float)pbr.baseColorFactor[2],
                (float)pbr.baseColorFactor[3]
            };
        }
        mat.metallic  = (float)pbr.metallicFactor;
        mat.roughness = (float)pbr.roughnessFactor;
        if (!gltf_mat.emissiveFactor.empty())
            mat.emissive = {
                (float)gltf_mat.emissiveFactor[0],
                (float)gltf_mat.emissiveFactor[1],
                (float)gltf_mat.emissiveFactor[2]
            };

        // Texture indices (resolved to image indices)
        if (pbr.baseColorTexture.index >= 0)
        {
            int tex_idx = pbr.baseColorTexture.index;
            int img_idx = model.textures[tex_idx].source;
            mat.albedo_image_idx = img_idx;
        }
        if (pbr.metallicRoughnessTexture.index >= 0)
        {
            int tex_idx = pbr.metallicRoughnessTexture.index;
            int img_idx = model.textures[tex_idx].source;
            mat.metallic_roughness_image_idx = img_idx;
        }
        if (gltf_mat.normalTexture.index >= 0)
        {
            int tex_idx = gltf_mat.normalTexture.index;
            int img_idx = model.textures[tex_idx].source;
            mat.normal_image_idx = img_idx;
        }
        if (gltf_mat.emissiveTexture.index >= 0)
        {
            int tex_idx = gltf_mat.emissiveTexture.index;
            int img_idx = model.textures[tex_idx].source;
            mat.emissive_image_idx = img_idx;
        }

        mat_remap[mi] = scene.add_material(mat);
    }

    // ── Images ────────────────────────────────────────────────────────────
    // Pre-register images so material image_idx are valid Scene indices.
    std::vector<uint32_t> img_remap(model.images.size(), UINT32_MAX);
    for (size_t ii = 0; ii < model.images.size(); ++ii)
    {
        const auto& gltf_img = model.images[ii];
        if (gltf_img.width <= 0 || gltf_img.height <= 0)
            continue;

        // tinygltf decodes to RGBA8 via stb_image by default.
        std::vector<uint8_t> pixels;
        if (gltf_img.component == 4)
        {
            pixels.assign(gltf_img.image.begin(), gltf_img.image.end());
        }
        else if (gltf_img.component == 3)
        {
            // Expand RGB to RGBA
            pixels.resize(gltf_img.width * gltf_img.height * 4);
            for (int p = 0; p < gltf_img.width * gltf_img.height; ++p)
            {
                pixels[p*4+0] = gltf_img.image[p*3+0];
                pixels[p*4+1] = gltf_img.image[p*3+1];
                pixels[p*4+2] = gltf_img.image[p*3+2];
                pixels[p*4+3] = 255;
            }
        }
        else
            continue; // unsupported component count

        img_remap[ii] = scene.add_image_rgba8(
            gltf_img.name,
            (uint32_t)gltf_img.width,
            (uint32_t)gltf_img.height,
            pixels);
    }

    // Remap material image indices to Scene image indices.
    // (We added materials before images, so we need to fixup the indices.)
    // This is handled implicitly because Scene::add_material stores the raw
    // image index as given. We need to traverse nodes + meshes to reassign.
    // For simplicity: materials are already added with correct img indices
    // since we used gltf image indices directly (Scene stores them the same way).

    // ── Meshes ────────────────────────────────────────────────────────────
    // Build a map from gltf mesh index to Scene mesh index (first primitive).
    std::vector<std::vector<uint32_t>> mesh_prim_to_scene(model.meshes.size());
    for (size_t gi = 0; gi < model.meshes.size(); ++gi)
    {
        const auto& gltf_mesh = model.meshes[gi];
        mesh_prim_to_scene[gi].resize(gltf_mesh.primitives.size(), UINT32_MAX);

        for (size_t pi = 0; pi < gltf_mesh.primitives.size(); ++pi)
        {
            const auto& prim = gltf_mesh.primitives[pi];
            if (prim.mode != TINYGLTF_MODE_TRIANGLES) continue;

            // Positions
            size_t pos_count = 0;
            const float* pos_data = nullptr;
            {
                auto it = prim.attributes.find("POSITION");
                if (it == prim.attributes.end()) continue;
                pos_data = accessor_data<float>(model, it->second, pos_count);
            }

            // Normals (optional)
            size_t nor_count = 0;
            const float* nor_data = nullptr;
            {
                auto it = prim.attributes.find("NORMAL");
                if (it != prim.attributes.end())
                    nor_data = accessor_data<float>(model, it->second, nor_count);
            }

            // UVs (optional)
            size_t uv_count = 0;
            const float* uv_data = nullptr;
            {
                auto it = prim.attributes.find("TEXCOORD_0");
                if (it != prim.attributes.end())
                    uv_data = accessor_data<float>(model, it->second, uv_count);
            }

            // Tangents (optional)
            size_t tan_count = 0;
            const float* tan_data = nullptr;
            {
                auto it = prim.attributes.find("TANGENT");
                if (it != prim.attributes.end())
                    tan_data = accessor_data<float>(model, it->second, tan_count);
            }

            // Indices
            std::vector<uint32_t> indices;
            if (prim.indices >= 0)
            {
                const auto& idx_acc = model.accessors[prim.indices];
                const auto& bv      = model.bufferViews[idx_acc.bufferView];
                const auto& buf     = model.buffers[bv.buffer];
                const uint8_t* raw  = buf.data.data() + bv.byteOffset + idx_acc.byteOffset;
                indices.resize(idx_acc.count);
                for (size_t ii = 0; ii < idx_acc.count; ++ii)
                {
                    if (idx_acc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
                        indices[ii] = reinterpret_cast<const uint16_t*>(raw)[ii];
                    else if (idx_acc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT)
                        indices[ii] = reinterpret_cast<const uint32_t*>(raw)[ii];
                    else if (idx_acc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE)
                        indices[ii] = raw[ii];
                }
            }
            else
            {
                // Non-indexed: generate sequential indices.
                indices.resize(pos_count);
                for (size_t ii = 0; ii < pos_count; ++ii)
                    indices[ii] = static_cast<uint32_t>(ii);
            }

            // Build GpuVertex array
            std::vector<GpuVertex> verts(pos_count);
            for (size_t vi = 0; vi < pos_count; ++vi)
            {
                GpuVertex& v = verts[vi];
                v.pos[0] = pos_data[vi*3+0];
                v.pos[1] = pos_data[vi*3+1];
                v.pos[2] = pos_data[vi*3+2];

                if (nor_data)
                {
                    v.normal[0] = nor_data[vi*3+0];
                    v.normal[1] = nor_data[vi*3+1];
                    v.normal[2] = nor_data[vi*3+2];
                }
                else
                {
                    v.normal[0] = 0; v.normal[1] = 1; v.normal[2] = 0;
                }

                if (uv_data)
                {
                    v.uv[0] = uv_data[vi*2+0];
                    v.uv[1] = uv_data[vi*2+1];
                }

                if (tan_data)
                {
                    v.tangent[0] = tan_data[vi*4+0];
                    v.tangent[1] = tan_data[vi*4+1];
                    v.tangent[2] = tan_data[vi*4+2];
                    v.tangent[3] = tan_data[vi*4+3];
                }
                else
                {
                    v.tangent[0] = 1; v.tangent[1] = 0;
                    v.tangent[2] = 0; v.tangent[3] = 1;
                }
            }

            uint32_t mat_idx = (prim.material >= 0)
                               ? mat_remap[prim.material]
                               : 0;
            std::string mesh_name = gltf_mesh.name + (gltf_mesh.primitives.size() > 1
                                    ? "_" + std::to_string(pi) : "");
            uint32_t scene_mesh_idx = scene.add_mesh(
                mesh_name, std::move(verts), std::move(indices), mat_idx);
            mesh_prim_to_scene[gi][pi] = scene_mesh_idx;
        }
    }

    // ── Nodes (instances) ─────────────────────────────────────────────────
    const int scene_idx = (model.defaultScene >= 0) ? model.defaultScene : 0;
    const auto& gltf_scene = model.scenes[scene_idx];

    // Traverse node tree to collect instances with world transforms.
    std::function<void(int, const glm::mat4&)> traverse = [&](int node_idx, const glm::mat4& parent_transform)
    {
        const auto& node       = model.nodes[node_idx];
        const glm::mat4 world  = parent_transform * node_transform(node);

        if (node.mesh >= 0)
        {
            const auto& prims = mesh_prim_to_scene[node.mesh];
            for (uint32_t scene_mesh_idx : prims)
            {
                if (scene_mesh_idx == UINT32_MAX) continue;
                Instance inst;
                inst.transform  = world;
                inst.mesh_index = scene_mesh_idx;
                scene.add_instance(inst);
            }
        }

        for (int child : node.children)
            traverse(child, world);
    };

    for (int root : gltf_scene.nodes)
        traverse(root, glm::mat4(1.0f));

    core::log()->info("[GltfLoader] Loaded '{}': {} meshes, {} instances, {} materials, {} images",
                       path.filename().string(),
                       model.meshes.size(), scene.instance_count(),
                       model.materials.size(), model.images.size());
    return true;
}

} // namespace rr::scene
