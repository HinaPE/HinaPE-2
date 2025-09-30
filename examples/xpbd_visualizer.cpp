#include "cloth.h"

#ifdef HINAPE_HAVE_VULKAN_VISUALIZER
#include <SDL3/SDL_events.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <imgui.h>
#include <stdexcept>
#include <set>
#include <vector>
#include <vk_engine.h>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>
#include <vv_camera.h>
#else
#include <cstdio>
#endif

using namespace HinaPE;

static void build_grid(int nx, int ny, float spacing, std::vector<float>& xyz, std::vector<u32>& tris, std::vector<u32>& fixed) {
    int n = nx * ny;
    xyz.resize(n * 3);
    tris.clear();
    fixed.clear();
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            int id          = y * nx + x;
            xyz[id * 3 + 0] = x * spacing;
            xyz[id * 3 + 1] = 0.0f;
            xyz[id * 3 + 2] = y * spacing;
        }
    }
    for (int x = 0; x < nx; ++x) fixed.push_back((u32) x); 
    for (int y = 0; y < ny - 1; ++y) {
        for (int x = 0; x < nx - 1; ++x) {
            int a = y * nx + x;
            int b = y * nx + x + 1;
            int c = (y + 1) * nx + x;
            int d = (y + 1) * nx + x + 1;
            tris.push_back(a);
            tris.push_back(b);
            tris.push_back(d);
            tris.push_back(a);
            tris.push_back(d);
            tris.push_back(c);
        }
    }
}

#ifndef HINAPE_HAVE_VULKAN_VISUALIZER
int main() {
    std::puts("Vulkan visualizer not available. Rebuild with HINAPE_WITH_VULKAN_VISUALIZER=ON and dependencies installed.");
    return 0;
}
#else

#ifndef VK_CHECK
#define VK_CHECK(x)                                                                            \
    do {                                                                                       \
        VkResult _r = (x);                                                                     \
        if (_r != VK_SUCCESS) throw std::runtime_error("Vulkan error: " + std::to_string(_r)); \
    } while (false)
#endif

struct GPUBuffer {
    VkBuffer buf{};
    VmaAllocation alloc{};
    void* mapped{};
    size_t size{};
};
static void destroy_buffer(const EngineContext& e, GPUBuffer& b) {
    if (b.mapped) {
        vmaUnmapMemory(e.allocator, b.alloc);
        b.mapped = nullptr;
    }
    if (b.buf) {
        vmaDestroyBuffer(e.allocator, b.buf, b.alloc);
    }
    b = {};
}
static void create_buffer(const EngineContext& e, VkDeviceSize sz, VkBufferUsageFlags usage, bool hostVisible, GPUBuffer& out) {
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size        = sz;
    bi.usage       = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    if (hostVisible) ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VK_CHECK(vmaCreateBuffer(e.allocator, &bi, &ai, &out.buf, &out.alloc, nullptr));
    out.size = (size_t) sz;
    if (hostVisible) vmaMapMemory(e.allocator, out.alloc, &out.mapped);
}

class XPBDRenderer : public IRenderer {
public:
    void get_capabilities(const EngineContext&, RendererCaps& c) override {
        c                         = RendererCaps{};
        c.enable_imgui            = true;
        c.presentation_mode       = PresentationMode::EngineBlit;
        c.color_attachments       = {AttachmentRequest{.name = "color", .format = VK_FORMAT_B8G8R8A8_UNORM}};
        c.presentation_attachment = "color";
        c.depth_attachment        = AttachmentRequest{.name = "depth", .format = VK_FORMAT_D32_SFLOAT, .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, .samples = VK_SAMPLE_COUNT_1_BIT, .aspect = VK_IMAGE_ASPECT_DEPTH_BIT};
        c.uses_depth              = VK_TRUE;
    }

    void initialize(const EngineContext& e, const RendererCaps&, const FrameContext&) override {
        ctx_ = e;
        dev_ = e.device;
        build_scene_();
        build_gpu_();
        build_pipeline_();
        init_camera_();
    }

    void destroy(const EngineContext& e, const RendererCaps&) override {
        destroy_gpu_();
        destroy_pipeline_();
        if (handle_) ::HinaPE::destroy(handle_); 
        handle_ = nullptr;
        dev_    = VK_NULL_HANDLE;
    }

    void update(const EngineContext&, const FrameContext& f) override {
        cam_.update(f.dt_sec, (int) f.extent.width, (int) f.extent.height);
        sim_accum_ += f.dt_sec;
        double fixed = std::clamp<double>(params_.fixed_dt, 1.0 / 600.0, 1.0 / 30.0);
        int maxSteps = 4;
        while (sim_accum_ >= fixed && maxSteps--) {
            step_sim_((float) fixed);
            sim_accum_ -= fixed;
        }
        if (pos_.mapped && handle_) {
            auto v = map_dynamic(handle_);
            struct P {
                float x, y, z;
            };
            P* dst = reinterpret_cast<P*>(pos_.mapped);
            for (size_t i = 0; i < v.count; ++i) {
                dst[i].x = v.pos_x[i];
                dst[i].y = v.pos_y[i];
                dst[i].z = v.pos_z[i];
            }
        }
    }

    void on_event(const SDL_Event& e, const EngineContext& eng, const FrameContext* f) override {
        cam_.handle_event(e, &eng, f);
    }

    void record_graphics(VkCommandBuffer cmd, const EngineContext&, const FrameContext& f) override {
        if ((!pipe_tri_.pipeline && !pipe_line_.pipeline) || f.color_attachments.empty()) return;
        const auto& color = f.color_attachments.front();
        const auto* depth = f.depth_attachment;
        auto barrier      = [&](VkImage img, VkImageLayout oldL, VkImageLayout newL, VkPipelineStageFlags2 src, VkPipelineStageFlags2 dst, VkAccessFlags2 sa, VkAccessFlags2 da, VkImageAspectFlags aspect) {
            VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            b.srcStageMask     = src;
            b.dstStageMask     = dst;
            b.srcAccessMask    = sa;
            b.dstAccessMask    = da;
            b.oldLayout        = oldL;
            b.newLayout        = newL;
            b.image            = img;
            b.subresourceRange = {aspect, 0, 1, 0, 1};
            VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            di.imageMemoryBarrierCount = 1;
            di.pImageMemoryBarriers    = &b;
            vkCmdPipelineBarrier2(cmd, &di);
        };
        barrier(color.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, color.aspect);
        if (depth) barrier(depth->image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT, 0, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, depth->aspect);
        VkClearValue cc{.color = {{0.02f, 0.03f, 0.04f, 1.0f}}};
        VkClearValue cd{.depthStencil = {1.0f, 0}};
        VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        ca.imageView   = color.view;
        ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        ca.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
        ca.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
        ca.clearValue  = cc;
        VkRenderingAttachmentInfo da{};
        if (depth) {
            da.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            da.imageView   = depth->view;
            da.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
            da.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
            da.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
            da.clearValue  = cd;
        }
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
        ri.renderArea           = {{0, 0}, f.extent};
        ri.layerCount           = 1;
        ri.colorAttachmentCount = 1;
        ri.pColorAttachments    = &ca;
        ri.pDepthAttachment     = depth ? &da : nullptr;
        vkCmdBeginRendering(cmd, &ri);
        VkViewport vp{0, 0, (float) f.extent.width, (float) f.extent.height, 0, 1};
        VkRect2D sc{{0, 0}, f.extent};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);
        vv::float4x4 V   = cam_.view_matrix();
        vv::float4x4 P   = cam_.proj_matrix();
        vv::float4x4 MVP = vv::mul(P, V);
        struct PC {
            float mvp[16];
            float color[4];
            float pointSize;
            float _pad0;
            float _pad1;
            float _pad2;
        } pc{};
        std::memcpy(pc.mvp, MVP.m.data(), sizeof(pc.mvp));
        VkDeviceSize offs = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &pos_.buf, &offs);

        if (params_.show_mesh && pipe_tri_.pipeline && idx_tri_.buf) {
            pc.color[0]  = 0.55f;
            pc.color[1]  = 0.75f;
            pc.color[2]  = 0.95f;
            pc.color[3]  = 1.0f;
            pc.pointSize = params_.point_size;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe_tri_.pipeline);
            vkCmdPushConstants(cmd, pipe_tri_.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PC), &pc);
            vkCmdBindIndexBuffer(cmd, idx_tri_.buf, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, (uint32_t) tri_count_, 1, 0, 0, 0);
        }

        if (params_.show_constraints && pipe_line_.pipeline && idx_line_.buf && line_count_ > 1) {
            pc.color[0]  = 0.9f;
            pc.color[1]  = 0.9f;
            pc.color[2]  = 0.9f;
            pc.color[3]  = 1.0f;
            pc.pointSize = params_.point_size;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe_line_.pipeline);
            vkCmdPushConstants(cmd, pipe_line_.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PC), &pc);
            vkCmdBindIndexBuffer(cmd, idx_line_.buf, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, (uint32_t) line_count_, 1, 0, 0, 0);
        }

        if (params_.show_vertices && pipe_point_.pipeline) {
            pc.color[0]  = 1.0f;
            pc.color[1]  = 1.0f;
            pc.color[2]  = 1.0f;
            pc.color[3]  = 1.0f;
            pc.pointSize = params_.point_size;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe_point_.pipeline);
            vkCmdPushConstants(cmd, pipe_point_.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PC), &pc);
            uint32_t vertCount = (uint32_t) (xyz_.size() / 3);
            vkCmdDraw(cmd, vertCount, 1, 0, 0);
        }
        vkCmdEndRendering(cmd);
        barrier(color.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, color.aspect);
    }

    void on_imgui(const EngineContext& eng, const FrameContext&) override {
        auto* host = static_cast<vv_ui::TabsHost*>(eng.services);
        if (!host) return;
        host->add_overlay([this] {
            cam_.imgui_draw_nav_overlay_space_tint();
            cam_.imgui_draw_mini_axis_gizmo();
        });
        host->add_tab("XPBD Cloth", [this] {
            ImGui::Checkbox("Simulate", &params_.simulate);
            ImGui::SliderFloat("Fixed dt", &params_.fixed_dt, 1.0f / 240.f, 1.0f / 30.f, "%.4f");
            ImGui::SliderInt("Substeps", &params_.substeps, 1, 8);
            ImGui::SliderInt("Iterations", &params_.iterations, 1, 60);
            ImGui::SliderFloat("Damping", &params_.damping, 0.0f, 1.0f);
            ImGui::SliderFloat3("Gravity", &params_.gravity.x, -30.f, 30.f);            ImGui::Separator();
            static int backend_idx = 0; const char* backends[] = {"Native","AVX2"};
            if (ImGui::Combo("Backend", &backend_idx, backends, 2)) {
                ExecPolicy ex{}; ex.backend = backend_idx==1 ? ExecPolicy::Backend::Avx2 : ExecPolicy::Backend::Native;
                SolvePolicy sv{}; sv.iterations = params_.iterations; sv.substeps = params_.substeps; sv.damping = params_.damping;
                if (handle_) ::HinaPE::destroy(handle_);
                InitDesc init{std::span<const float>(xyz_.data(), xyz_.size()), std::span<const u32>(tris_.data(), tris_.size()), std::span<const u32>(fixed_.data(), fixed_.size()), ex, sv};
                handle_ = create(init);
            }
            ImGui::SliderFloat("Compliance", &params_.compliance, 0.0f, 1e-1f, "%.6f");
            ImGui::Separator();
            ImGui::Checkbox("Mesh", &params_.show_mesh);
            ImGui::SameLine();
            ImGui::Checkbox("Wire", &params_.show_constraints);
            ImGui::SameLine();
            ImGui::Checkbox("Points", &params_.show_vertices);
            ImGui::SliderFloat("Point Size", &params_.point_size, 1.0f, 12.0f);
            if (ImGui::Button("Reset")) {
                reset_scene_();
            }
            ImGui::SameLine();
            if (ImGui::Button("Frame")) {
                cam_.frame_scene(1.1f);
            }
        });
        host->add_tab("Camera", [this] { cam_.imgui_panel_contents(); });
    }

private:
    struct Params {
        bool simulate{true};
        float fixed_dt{1.0f / 120.f};
        int substeps{2};
        int iterations{10};
        float damping{0.02f};
        float compliance{0.0f};
        vv::float3 gravity{0.0f, -9.81f, 0.0f};
        bool show_mesh{true};
        bool show_vertices{true};
        bool show_constraints{true};
        float point_size{5.0f};
    } params_{};
    EngineContext ctx_{};
    VkDevice dev_{VK_NULL_HANDLE};
    vv::CameraService cam_{};
    double sim_accum_{0.0};
    Handle handle_{};
    std::vector<float> xyz_;
    std::vector<u32> tris_;
    std::vector<u32> fixed_;
    size_t tri_count_{0};
    GPUBuffer pos_{}; GPUBuffer idx_tri_{}; GPUBuffer idx_line_{}; size_t line_count_{0};
    std::vector<uint32_t> line_indices_; 
    struct Pipe {
        VkPipeline pipeline{};
        VkPipelineLayout layout{};
    } pipe_tri_{}, pipe_line_{}, pipe_point_{};

    void build_scene_() {
        int nx = 40, ny = 40;
        build_grid(nx, ny, 0.04f, xyz_, tris_, fixed_);
        ExecPolicy exec{};
        SolvePolicy solve{};
        solve.substeps   = params_.substeps;
        solve.iterations = params_.iterations;
        solve.damping = params_.damping; solve.compliance_stretch = params_.compliance;
        InitDesc init{std::span<const float>(xyz_.data(), xyz_.size()), std::span<const u32>(tris_.data(), tris_.size()), std::span<const u32>(fixed_.data(), fixed_.size()), exec, solve};
        handle_    = create(init);
        tri_count_ = tris_.size();
        build_lines_from_tris_();
    }
    void reset_scene_() {
        if (handle_) ::HinaPE::destroy(handle_); 
        build_scene_();
        upload_indices_();
        upload_lines_();
    }
    void build_gpu_() {
        size_t vertCount = xyz_.size() / 3;
        create_buffer(ctx_, vertCount * sizeof(float) * 3, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, pos_);
        create_buffer(ctx_, tri_count_ * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, true, idx_tri_);
        create_buffer(ctx_, std::max<size_t>(2, line_count_) * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, true, idx_line_);
        upload_indices_();
        upload_lines_();
    }
    void upload_indices_() {
        if (!idx_tri_.mapped) return;
        std::memcpy(idx_tri_.mapped, tris_.data(), tri_count_ * sizeof(uint32_t));
    }
    void upload_lines_() {
        if(line_indices_.empty()) return;
        size_t needBytes = line_indices_.size()*sizeof(uint32_t);
        if(needBytes > idx_line_.size){
            destroy_buffer(ctx_, idx_line_);
            create_buffer(ctx_, needBytes, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, true, idx_line_);
        }
        if(!idx_line_.mapped) return;
        std::memcpy(idx_line_.mapped, line_indices_.data(), needBytes);
    }
    void destroy_gpu_() {
        destroy_buffer(ctx_, pos_);
        destroy_buffer(ctx_, idx_tri_);
        destroy_buffer(ctx_, idx_line_);
    }

    void build_lines_from_tris_() {
        line_indices_.clear();
        if (tri_count_ % 3 != 0) return;
        std::vector<std::pair<uint32_t, uint32_t>> edges;
        edges.reserve(tri_count_);
        for (size_t t = 0; t < tri_count_; t += 3) {
            uint32_t a = tris_[t];
            uint32_t b = tris_[t + 1];
            uint32_t c = tris_[t + 2];
            auto add_edge = [&](uint32_t i, uint32_t j) {
                if (i > j) std::swap(i, j);
                edges.emplace_back(i, j);
            };
            add_edge(a, b);
            add_edge(b, c);
            add_edge(c, a);
        }
        std::sort(edges.begin(), edges.end());
        edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
        line_indices_.reserve(edges.size() * 2);
        for (auto &e : edges) {
            line_indices_.push_back(e.first);
            line_indices_.push_back(e.second);
        }
        line_count_ = line_indices_.size();
    }

    void step_sim_(float dt) {
        if (!handle_) return;
        StepParams sp{};
        sp.dt        = dt;
        sp.gravity_x = params_.gravity.x;
        sp.gravity_y = params_.gravity.y;
        sp.gravity_z = params_.gravity.z;
        step(handle_, sp);
    }

    void init_camera_() {
        cam_.set_mode(vv::CameraMode::Orbit);
        auto st      = cam_.state();
        st.target    = {0, 0, 0};
        st.distance  = 2.2f;
        st.pitch_deg = 22.f;
        st.yaw_deg   = -120.f;
        cam_.set_state(st);
        vv::BoundingBox bb{};
        bb.min   = {-1.0f, -1.0f, -1.0f};
        bb.max   = {1.0f, 1.0f, 1.0f};
        bb.valid = true;
        cam_.set_scene_bounds(bb);
        cam_.frame_scene(1.05f);
    }

    void build_pipeline_() {
        std::string dir(SHADER_OUTPUT_DIR);
        auto vsData       = load_spv_(dir + "/cloth.vert.spv");
        auto fsData       = load_spv_(dir + "/cloth.frag.spv");
        VkShaderModule vs = make_shader_(vsData);
        VkShaderModule fs = make_shader_(fsData);
        VkPipelineShaderStageCreateInfo st[2]{};
        for (int i = 0; i < 2; ++i) st[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        st[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
        st[0].module = vs;
        st[0].pName  = "main";
        st[1]        = st[0];
        st[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
        st[1].module = fs;
        VkVertexInputBindingDescription bind{0, sizeof(float) * 3, VK_VERTEX_INPUT_RATE_VERTEX};
        VkVertexInputAttributeDescription attr{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0};
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vi.vertexBindingDescriptionCount   = 1;
        vi.pVertexBindingDescriptions      = &bind;
        vi.vertexAttributeDescriptionCount = 1;
        vi.pVertexAttributeDescriptions    = &attr;
        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp.viewportCount = 1;
        vp.scissorCount  = 1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode    = VK_CULL_MODE_NONE;
        rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth   = 1.0f;
        VkPipelineRasterizationStateCreateInfo rsLine = rs;
        rsLine.polygonMode = VK_POLYGON_MODE_FILL;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds.depthTestEnable  = VK_TRUE;
        ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp   = VK_COMPARE_OP_LESS;
        VkPipelineColorBlendAttachmentState ba{};
        ba.colorWriteMask = 0xF;
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount           = 1;
        cb.pAttachments              = &ba;
        const VkDynamicState dyns[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dsi{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dsi.dynamicStateCount = 2;
        dsi.pDynamicStates    = dyns;
        VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT, 0, 96};
        VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        lci.pushConstantRangeCount = 1;
        lci.pPushConstantRanges    = &pcr;
        VK_CHECK(vkCreatePipelineLayout(dev_, &lci, nullptr, &pipe_tri_.layout));
        pipe_line_.layout = pipe_tri_.layout;
        pipe_point_.layout = pipe_tri_.layout;
        VkPipelineRenderingCreateInfo rinfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        VkFormat colorFmt             = VK_FORMAT_B8G8R8A8_UNORM;
        VkFormat depthFmt             = VK_FORMAT_D32_SFLOAT;
        rinfo.colorAttachmentCount    = 1;
        rinfo.pColorAttachmentFormats = &colorFmt;
        rinfo.depthAttachmentFormat   = depthFmt;
        auto make_pipe = [&](VkPrimitiveTopology topo, Pipe& out) {
            VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            ia.topology = topo;
            VkGraphicsPipelineCreateInfo pci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            pci.pNext               = &rinfo;
            pci.stageCount          = 2;
            pci.pStages             = st;
            pci.pVertexInputState   = &vi;
            pci.pInputAssemblyState = &ia;
            pci.pViewportState      = &vp;
            pci.pRasterizationState = &rs;
            if (topo == VK_PRIMITIVE_TOPOLOGY_LINE_LIST) pci.pRasterizationState = &rsLine;
            pci.pMultisampleState   = &ms;
            pci.pDepthStencilState  = &ds;
            pci.pColorBlendState    = &cb;
            pci.pDynamicState       = &dsi;
            pci.layout              = pipe_tri_.layout;
            VK_CHECK(vkCreateGraphicsPipelines(dev_, VK_NULL_HANDLE, 1, &pci, nullptr, &out.pipeline));
        };
        make_pipe(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, pipe_tri_);
        make_pipe(VK_PRIMITIVE_TOPOLOGY_LINE_LIST,     pipe_line_);
        make_pipe(VK_PRIMITIVE_TOPOLOGY_POINT_LIST,    pipe_point_);
        vkDestroyShaderModule(dev_, vs, nullptr);
        vkDestroyShaderModule(dev_, fs, nullptr);
    }

    void destroy_pipeline_() {
        if (pipe_tri_.pipeline) vkDestroyPipeline(dev_, pipe_tri_.pipeline, nullptr);
        if (pipe_line_.pipeline) vkDestroyPipeline(dev_, pipe_line_.pipeline, nullptr);
        if (pipe_point_.pipeline) vkDestroyPipeline(dev_, pipe_point_.pipeline, nullptr);
        if (pipe_tri_.layout) vkDestroyPipelineLayout(dev_, pipe_tri_.layout, nullptr);
        pipe_tri_ = {};
        pipe_line_ = {};
        pipe_point_ = {};
    }

    static std::vector<char> load_spv_(const std::string& p) {
        FILE* f = nullptr;
#if defined(_MSC_VER) && !defined(__clang__)
        fopen_s(&f, p.c_str(), "rb");
#else
        f = fopen(p.c_str(), "rb");
#endif
        if (!f) throw std::runtime_error("open " + p);
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        std::vector<char> data((size_t) sz);
        fread(data.data(), 1, (size_t) sz, f);
        fclose(f);
        return data;
    }
    VkShaderModule make_shader_(const std::vector<char>& bytes) {
        VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        ci.codeSize = (uint32_t) bytes.size();
        ci.pCode    = (const uint32_t*) bytes.data();
        VkShaderModule m{};
        VK_CHECK(vkCreateShaderModule(dev_, &ci, nullptr, &m));
        return m;
    }
};

int main() {
    try {
        VulkanEngine e;
        e.configure_window(1280, 720, "xpbd_visualizer");
        e.set_renderer(std::make_unique<XPBDRenderer>());
        e.init();
        e.run();
        e.cleanup();
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "Fatal: %s\n", ex.what());
        return 1;
    }
    return 0;
}
#endif




