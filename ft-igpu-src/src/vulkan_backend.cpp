#include "ft/vulkan_backend.h"
#include "ft/quant.h"
#include "ft/cpu_backend.h"

#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#ifdef FT_HAVE_VULKAN
#include <vulkan/vulkan.h>
#include <fstream>
#endif

namespace ft {

namespace {

#ifdef FT_HAVE_VULKAN

struct PC {
    int32_t M, K, IC, dtype;
    int32_t phase, guStride, downStride, pad;
};

int dt_id(DType t) {
    switch (t) {
        case DType::F32: return 0;
        case DType::F16: return 1;
        case DType::Q8_0: return 2;
        case DType::Q4_0: return 3;
        default: return -1;
    }
}

std::vector<char> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    const size_t n = (size_t)f.tellg();
    std::vector<char> b(n);
    f.seekg(0);
    f.read(b.data(), (std::streamsize)n);
    return b;
}

#endif

} // namespace

struct VulkanBackend::Impl {
#ifdef FT_HAVE_VULKAN
    VkInstance inst = nullptr;
    VkPhysicalDevice pd = nullptr;
    VkDevice dev = nullptr;
    uint32_t qfam = 0;
    VkQueue queue = nullptr;
    VkCommandPool cpool = nullptr;
    VkCommandBuffer cmd = nullptr;
    VkDescriptorSetLayout dsl = nullptr;
    VkDescriptorPool dpool = nullptr;
    VkDescriptorSet dset = nullptr;
    VkPipelineLayout playout = nullptr;
    VkPipeline pipe = nullptr;

    struct HostBuf {
        VkBuffer b = nullptr;
        VkDeviceMemory m = nullptr;
        void* ptr = nullptr;
        size_t cap = 0;
        bool ok() const { return b != nullptr; }
    };
    HostBuf xbuf, obuf, hbuf;
    size_t xcap = 0, ocap = 0, hcap = 0;
    std::unordered_map<void*, HostBuf> staging;

    DeviceInfo info_;
    bool ready_ = false;

    uint32_t mem_type(VkMemoryRequirements req, VkMemoryPropertyFlags props) {
        VkPhysicalDeviceMemoryProperties mp{};
        vkGetPhysicalDeviceMemoryProperties(pd, &mp);
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
            if ((req.memoryTypeBits & (1u << i)) &&
                (mp.memoryTypes[i].propertyFlags & props) == props)
                return i;
        }
        throw std::runtime_error("vulkan: no suitable memory type");
    }

    void make_host_buffer(size_t bytes, HostBuf& out) {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = bytes;
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        if (vkCreateBuffer(dev, &bi, nullptr, &out.b) != VK_SUCCESS)
            throw std::runtime_error("vulkan: buffer create failed");
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(dev, out.b, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex =
            mem_type(req, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkResult ar = vkAllocateMemory(dev, &ai, nullptr, &out.m);
        if (ar != VK_SUCCESS) {
            vkDestroyBuffer(dev, out.b, nullptr);
            out.b = nullptr;
            throw std::runtime_error("vulkan: alloc failed");
        }
        vkBindBufferMemory(dev, out.b, out.m, 0);
        vkMapMemory(dev, out.m, 0, bytes, 0, &out.ptr);
        out.cap = bytes;
    }

    void destroy_buf(HostBuf& b) {
        if (b.ptr && b.m) vkUnmapMemory(dev, b.m);
        if (b.b) vkDestroyBuffer(dev, b.b, nullptr);
        if (b.m) vkFreeMemory(dev, b.m, nullptr);
        b = {};
    }

    void ensure_buf(HostBuf& b, size_t& cap, size_t need) {
        if (b.ok() && cap >= need) return;
        if (b.ok()) destroy_buf(b);
        cap = need * 2;
        make_host_buffer(cap, b);
    }

    void init(bool prefer_integrated) {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo ii{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ii.pApplicationInfo = &app;
        if (vkCreateInstance(&ii, nullptr, &inst) != VK_SUCCESS)
            throw std::runtime_error("vulkan: instance create failed");

        uint32_t ndev = 0;
        vkEnumeratePhysicalDevices(inst, &ndev, nullptr);
        if (!ndev) throw std::runtime_error("vulkan: no devices");
        std::vector<VkPhysicalDevice> devs(ndev);
        vkEnumeratePhysicalDevices(inst, &ndev, devs.data());

        for (VkPhysicalDevice d : devs) {
            VkPhysicalDeviceProperties p{};
            vkGetPhysicalDeviceProperties(d, &p);
            const bool integrated =
                p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
            if (!pd) { pd = d; info_.name = p.deviceName; }
            if (prefer_integrated ? integrated : !integrated) {
                pd = d;
                info_.name = p.deviceName;
                break;
            }
        }
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(pd, &props);
        info_.name = "Vulkan:" + std::string(props.deviceName);
        info_.integrated =
            props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
        info_.uma = info_.integrated;

        uint32_t nfam = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &nfam, nullptr);
        std::vector<VkQueueFamilyProperties> fams(nfam);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &nfam, fams.data());
        bool found = false;
        for (uint32_t i = 0; i < nfam; ++i) {
            if (fams[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                qfam = i;
                found = true;
                break;
            }
        }
        if (!found) throw std::runtime_error("vulkan: no compute queue");

        float prio = 1.f;
        VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qi.queueFamilyIndex = qfam;
        qi.queueCount = 1;
        qi.pQueuePriorities = &prio;
        VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        di.queueCreateInfoCount = 1;
        di.pQueueCreateInfos = &qi;
        if (vkCreateDevice(pd, &di, nullptr, &dev) != VK_SUCCESS)
            throw std::runtime_error("vulkan: device create failed");
        vkGetDeviceQueue(dev, qfam, 0, &queue);

        VkPhysicalDeviceMemoryProperties mp{};
        vkGetPhysicalDeviceMemoryProperties(pd, &mp);
        for (uint32_t i = 0; i < mp.memoryHeapCount; ++i) {
            if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                info_.mem_bytes =
                    std::max(info_.mem_bytes,
                             (uint64_t)mp.memoryHeaps[i].size);
        }

        VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pi.queueFamilyIndex = qfam;
        if (vkCreateCommandPool(dev, &pi, nullptr, &cpool) != VK_SUCCESS)
            throw std::runtime_error("vulkan: cmd pool failed");
        VkCommandBufferAllocateInfo ca{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ca.commandPool = cpool;
        ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ca.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(dev, &ca, &cmd) != VK_SUCCESS)
            throw std::runtime_error("vulkan: cmdbuf failed");

        VkDescriptorSetLayoutBinding binds[4]{};
        for (uint32_t i = 0; i < 4; ++i) {
            binds[i].binding = i;
            binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binds[i].descriptorCount = 1;
            binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dli{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dli.bindingCount = 4;
        dli.pBindings = binds;
        if (vkCreateDescriptorSetLayout(dev, &dli, nullptr, &dsl) !=
            VK_SUCCESS)
            throw std::runtime_error("vulkan: dsl failed");

        VkPushConstantRange pr{};
        pr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pr.offset = 0;
        pr.size = sizeof(PC);
        VkPipelineLayoutCreateInfo pli{
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &dsl;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pr;
        if (vkCreatePipelineLayout(dev, &pli, nullptr, &playout) != VK_SUCCESS)
            throw std::runtime_error("vulkan: pipeline layout failed");

        auto code = read_file("shaders/moe_ffn_chunk.spv");
        VkShaderModuleCreateInfo smi{
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smi.codeSize = code.size();
        smi.pCode = reinterpret_cast<const uint32_t*>(code.data());
        VkShaderModule sm = nullptr;
        if (vkCreateShaderModule(dev, &smi, nullptr, &sm) != VK_SUCCESS)
            throw std::runtime_error("vulkan: shader module failed");
        VkPipelineShaderStageCreateInfo ss{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        ss.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ss.module = sm;
        ss.pName = "main";
        VkComputePipelineCreateInfo cpi{
            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpi.stage = ss;
        cpi.layout = playout;
        if (vkCreateComputePipelines(dev, nullptr, 1, &cpi, nullptr,
                                     &pipe) != VK_SUCCESS) {
            vkDestroyShaderModule(dev, sm, nullptr);
            throw std::runtime_error("vulkan: pipeline failed");
        }
        vkDestroyShaderModule(dev, sm, nullptr);

        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4};
        VkDescriptorPoolCreateInfo dpi{
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpi.maxSets = 1;
        dpi.poolSizeCount = 1;
        dpi.pPoolSizes = &ps;
        if (vkCreateDescriptorPool(dev, &dpi, nullptr, &dpool) != VK_SUCCESS)
            throw std::runtime_error("vulkan: desc pool failed");
        VkDescriptorSetAllocateInfo dai{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = dpool;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts = &dsl;
        if (vkAllocateDescriptorSets(dev, &dai, &dset) != VK_SUCCESS)
            throw std::runtime_error("vulkan: desc set failed");

        ready_ = true;
    }

    void update_desc(VkBuffer x, VkBuffer w, VkBuffer o, VkBuffer h) {
        VkDescriptorBufferInfo infos[4]{{x, 0, VK_WHOLE_SIZE},
                                        {w, 0, VK_WHOLE_SIZE},
                                        {o, 0, VK_WHOLE_SIZE},
                                        {h, 0, VK_WHOLE_SIZE}};
        VkWriteDescriptorSet wr[4]{};
        for (uint32_t i = 0; i < 4; ++i) {
            wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            wr[i].dstSet = dset;
            wr[i].dstBinding = i;
            wr[i].descriptorCount = 1;
            wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wr[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(dev, 4, wr, 0, nullptr);
    }

    void dispatch(uint32_t gx, const PC& pc) {
        if (vkResetCommandBuffer(cmd, 0) != VK_SUCCESS)
            throw std::runtime_error("vulkan: reset cmdbuf");
        VkCommandBufferBeginInfo bi{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS)
            throw std::runtime_error("vulkan: begin cmdbuf");
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdPushConstants(cmd, playout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(PC), &pc);
        vkCmdDispatch(cmd, gx, 1, 1);
        if (pc.phase == 0) {
            VkMemoryBarrier bar{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                                 &bar, 0, nullptr, 0, nullptr);
        }
        if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
            throw std::runtime_error("vulkan: end cmdbuf");
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        if (vkQueueSubmit(queue, 1, &si, nullptr) != VK_SUCCESS)
            throw std::runtime_error("vulkan: submit");
        if (pc.phase == 1) vkQueueWaitIdle(queue);
    }
#else
    bool ready_ = false;
#endif
};

VulkanBackend::VulkanBackend(bool prefer_integrated, ThrottleConfig throttle)
#ifdef FT_HAVE_VULKAN
    : impl_(new Impl) {
    try {
        impl_->init(prefer_integrated);
    } catch (...) {
        delete impl_;
        impl_ = nullptr;
        throw;
    }
}
#else
    : impl_(nullptr) {
    (void)prefer_integrated;
    (void)throttle;
    throw std::runtime_error(
        "built without Vulkan support (cmake -DFT_WITH_VULKAN=ON)");
}
#endif

VulkanBackend::~VulkanBackend() {
#ifdef FT_HAVE_VULKAN
    if (!impl_) return;
    impl_->destroy_buf(impl_->xbuf);
    impl_->destroy_buf(impl_->obuf);
    impl_->destroy_buf(impl_->hbuf);
    for (auto& kv : impl_->staging) impl_->destroy_buf(kv.second);
    if (impl_->pipe) vkDestroyPipeline(impl_->dev, impl_->pipe, nullptr);
    if (impl_->dpool)
        vkDestroyDescriptorPool(impl_->dev, impl_->dpool, nullptr);
    if (impl_->dsl)
        vkDestroyDescriptorSetLayout(impl_->dev, impl_->dsl, nullptr);
    if (impl_->playout)
        vkDestroyPipelineLayout(impl_->dev, impl_->playout, nullptr);
    if (impl_->cpool) vkDestroyCommandPool(impl_->dev, impl_->cpool, nullptr);
    if (impl_->dev) vkDestroyDevice(impl_->dev, nullptr);
    if (impl_->inst) vkDestroyInstance(impl_->inst, nullptr);
#endif
}

bool VulkanBackend::ready() const { return impl_ && impl_->ready_; }

std::string VulkanBackend::name() const {
#ifdef FT_HAVE_VULKAN
    return impl_ ? impl_->info_.name : "Vulkan:none";
#else
    return "Vulkan:none";
#endif
}

DeviceInfo VulkanBackend::info() const {
    DeviceInfo di;
    di.name = name();
#ifdef FT_HAVE_VULKAN
    di.integrated = impl_ ? impl_->info_.integrated : false;
    di.uma = impl_ ? impl_->info_.uma : false;
    di.mem_bytes = impl_ ? impl_->info_.mem_bytes : 0;
#endif
    return di;
}

void* VulkanBackend::alloc_staging(size_t bytes) {
#ifdef FT_HAVE_VULKAN
    Impl::HostBuf hb;
    impl_->make_host_buffer(bytes, hb);
    void* key = hb.ptr;
    impl_->staging.emplace(key, hb);
    return key;
#else
    (void)bytes;
    return nullptr;
#endif
}

void VulkanBackend::free_staging(void* p) {
#ifdef FT_HAVE_VULKAN
    auto it = impl_->staging.find(p);
    if (it == impl_->staging.end()) return;
    impl_->destroy_buf(it->second);
    impl_->staging.erase(it);
#else
    (void)p;
#endif
}

void VulkanBackend::stage(void* dst, const void* src, size_t bytes) {
    std::memcpy(dst, src, bytes);
}

void VulkanBackend::sync() {
#ifdef FT_HAVE_VULKAN
    if (impl_) vkQueueWaitIdle(impl_->queue);
#endif
}

static void cpu_matmul(const float* x, int64_t M, int64_t K,
                       const HostTensor& W, float* y) {
    const int64_t N = W.rows();
    const DType dt = W.dtype;
    const size_t stride = W.row_stride_bytes();
    parallel_for(N, 0, [&](int64_t b, int64_t e) {
        for (int64_t r = b; r < e; ++r) {
            float acc = dot_row_f32(x, W.data + stride * (size_t)r, dt, K);
            for (int64_t m = 1; m < M; ++m)
                acc += dot_row_f32(x + m * K, W.data + stride * (size_t)r, dt, K);
            y[r] = acc;
        }
    });
}

void VulkanBackend::matmul(const float* x, int64_t M, int64_t K,
                           const HostTensor& W, float* y) {
    cpu_matmul(x, M, K, W, y);
}

void VulkanBackend::run_expert_chunk(const ExpertChunkJob& job) {
#ifdef FT_HAVE_VULKAN
    int did = dt_id(job.dt);
    if (did < 0 || !ready()) {
        static thread_local CpuBackend fallback(-1);
        fallback.run_expert_chunk(job);
        return;
    }

    void* base = nullptr;
    for (auto& kv : impl_->staging) {
        const uint8_t* lo = (const uint8_t*)kv.first;
        const uint8_t* hi = lo + kv.second.cap;
        if ((const uint8_t*)job.gate >= lo && (const uint8_t*)job.gate < hi) {
            base = kv.first;
            break;
        }
    }
    if (!base) throw std::runtime_error("vulkan: window not staged here");

    const size_t xsz = (size_t)(job.M * job.K) * sizeof(float);
    const size_t hsz = (size_t)(job.M * job.IC) * sizeof(float);
    impl_->ensure_buf(impl_->xbuf, impl_->xcap, xsz);
    impl_->ensure_buf(impl_->obuf, impl_->ocap, xsz);
    impl_->ensure_buf(impl_->hbuf, impl_->hcap, hsz);

    std::memcpy(impl_->xbuf.ptr, job.x, xsz);
    std::memset(impl_->obuf.ptr, 0, xsz);

    impl_->update_desc(impl_->xbuf.b, impl_->staging[base].b, impl_->obuf.b,
                       impl_->hbuf.b);

    PC pc{};
    pc.M = (int32_t)job.M;
    pc.K = (int32_t)job.K;
    pc.IC = (int32_t)job.IC;
    pc.dtype = did;
    pc.guStride = (int32_t)job.gate_up_row_stride;
    pc.downStride = (int32_t)job.down_row_stride;

    pc.phase = 0;
    impl_->dispatch((uint32_t)((job.M * job.IC + 63) / 64), pc);
    pc.phase = 1;
    impl_->dispatch((uint32_t)((job.M * job.K + 63) / 64), pc);

    std::memcpy(job.out, impl_->obuf.ptr, xsz);
#else
    static thread_local CpuBackend fallback(-1);
    fallback.run_expert_chunk(job);
#endif
}

} // namespace ft
