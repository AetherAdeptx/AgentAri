#include "agentari/VulkanBackend.hpp"

#include <cstring>
#include <cstdint>
#include <fstream>
#include <limits>
#include <utility>
#include <vector>

#if defined(AGENTARI_HAS_VULKAN) && AGENTARI_HAS_VULKAN
#include <vulkan/vulkan.h>
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
#endif

namespace agentari::gpu {

struct VulkanComputeBackend::Impl {
    bool available{false};
    std::string device_name;
    std::string error;

#if defined(AGENTARI_HAS_VULKAN) && AGENTARI_HAS_VULKAN
    VkInstance instance{VK_NULL_HANDLE};
    VkPhysicalDevice physical_device{VK_NULL_HANDLE};
    VkDevice device{VK_NULL_HANDLE};
    VkQueue queue{VK_NULL_HANDLE};
    VkCommandPool command_pool{VK_NULL_HANDLE};
    VkDescriptorSetLayout descriptor_set_layout{VK_NULL_HANDLE};
    VkPipelineLayout pipeline_layout{VK_NULL_HANDLE};
    VkPipeline pipeline{VK_NULL_HANDLE};
    VkPipeline elementwise_pipeline{VK_NULL_HANDLE};
    VkDescriptorPool descriptor_pool{VK_NULL_HANDLE};
    std::uint32_t queue_family{0U};

    ~Impl() {
        if (device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device);
            if (descriptor_pool != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
            }
            if (pipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(device, pipeline, nullptr);
            }
            if (elementwise_pipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(device, elementwise_pipeline, nullptr);
            }
            if (pipeline_layout != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
            }
            if (descriptor_set_layout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
            }
            if (command_pool != VK_NULL_HANDLE) {
                vkDestroyCommandPool(device, command_pool, nullptr);
            }
            vkDestroyDevice(device, nullptr);
        }
        if (instance != VK_NULL_HANDLE) {
            vkDestroyInstance(instance, nullptr);
        }
    }
#else
    ~Impl() = default;
#endif
};

#if defined(AGENTARI_HAS_VULKAN) && AGENTARI_HAS_VULKAN
namespace {

bool check(VkResult result, const char* operation, std::string& error) {
    if (result == VK_SUCCESS) {
        return true;
    }
    error = std::string(operation) + " failed with Vulkan error " + std::to_string(result);
    return false;
}

std::vector<char> read_binary(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        return {};
    }
    const std::streamsize size = input.tellg();
    if (size <= 0 || size % static_cast<std::streamsize>(sizeof(std::uint32_t)) != 0) {
        return {};
    }
    std::vector<char> data(static_cast<std::size_t>(size));
    input.seekg(0);
    input.read(data.data(), size);
    return input ? data : std::vector<char>{};
}

std::uint32_t find_memory_type(VkPhysicalDevice physical_device,
                               std::uint32_t type_bits,
                               VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memory_properties{};
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);
    for (std::uint32_t index = 0U; index < memory_properties.memoryTypeCount; ++index) {
        if ((type_bits & (1U << index)) != 0U &&
            (memory_properties.memoryTypes[index].propertyFlags & properties) == properties) {
            return index;
        }
    }
    return std::numeric_limits<std::uint32_t>::max();
}

struct Buffer {
    VkDevice device{VK_NULL_HANDLE};
    VkBuffer buffer{VK_NULL_HANDLE};
    VkDeviceMemory memory{VK_NULL_HANDLE};

    ~Buffer() {
        if (device != VK_NULL_HANDLE) {
            if (buffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(device, buffer, nullptr);
            }
            if (memory != VK_NULL_HANDLE) {
                vkFreeMemory(device, memory, nullptr);
            }
        }
    }
};

bool make_buffer(VkPhysicalDevice physical_device,
                 VkDevice device,
                 VkDeviceSize size,
                 VkBufferUsageFlags usage,
                 Buffer& result) {
    result.device = device;
    VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &buffer_info, nullptr, &result.buffer) != VK_SUCCESS) {
        return false;
    }

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, result.buffer, &requirements);
    const std::uint32_t memory_type = find_memory_type(
        physical_device, requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (memory_type == std::numeric_limits<std::uint32_t>::max()) {
        return false;
    }
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = memory_type;
    if (vkAllocateMemory(device, &allocation, nullptr, &result.memory) != VK_SUCCESS) {
        return false;
    }
    return vkBindBufferMemory(device, result.buffer, result.memory, 0U) == VK_SUCCESS;
}

}  // namespace
#endif

VulkanComputeBackend::VulkanComputeBackend(std::string shader_path)
    : implementation_(std::make_unique<Impl>()) {
#if !defined(AGENTARI_HAS_VULKAN) || !AGENTARI_HAS_VULKAN
    (void)shader_path;
    implementation_->error = "Vulkan support was not found at configure time";
#else
    if (shader_path.empty()) {
#ifdef AGENTARI_VULKAN_SHADER_PATH
        shader_path = AGENTARI_VULKAN_SHADER_PATH;
#endif
    }
    if (shader_path.empty()) {
        implementation_->error = "Vulkan shader path is not configured";
        return;
    }
    const std::vector<char> shader = read_binary(shader_path);
    if (shader.empty()) {
        implementation_->error = "could not read Vulkan shader: " + shader_path;
        return;
    }
    std::string elementwise_shader_path;
#ifdef AGENTARI_VULKAN_ELEMENTWISE_SHADER_PATH
    elementwise_shader_path = AGENTARI_VULKAN_ELEMENTWISE_SHADER_PATH;
#endif
    const std::vector<char> elementwise_shader = read_binary(elementwise_shader_path);
    if (elementwise_shader.empty()) {
        implementation_->error = "could not read Vulkan elementwise shader: " +
                                 elementwise_shader_path;
        return;
    }

    VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application.pApplicationName = "AgentAri GPU backend";
    application.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    application.pEngineName = "AgentAri";
    application.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    application.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &application;
    if (!check(vkCreateInstance(&instance_info, nullptr, &implementation_->instance),
               "vkCreateInstance", implementation_->error)) {
        return;
    }

    std::uint32_t device_count = 0U;
    if (!check(vkEnumeratePhysicalDevices(implementation_->instance, &device_count, nullptr),
               "vkEnumeratePhysicalDevices", implementation_->error) || device_count == 0U) {
        if (implementation_->error.empty()) {
            implementation_->error = "no Vulkan physical devices found";
        }
        return;
    }
    std::vector<VkPhysicalDevice> devices(device_count);
    if (!check(vkEnumeratePhysicalDevices(implementation_->instance, &device_count, devices.data()),
               "vkEnumeratePhysicalDevices", implementation_->error)) {
        return;
    }
    for (const VkPhysicalDevice candidate : devices) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(candidate, &properties);
        std::uint32_t family_count = 0U;
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count, nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count, families.data());
        for (std::uint32_t family = 0U; family < family_count; ++family) {
            if ((families[family].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0U) {
                continue;
            }
            const bool better = implementation_->physical_device == VK_NULL_HANDLE ||
                                (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);
            if (better) {
                implementation_->physical_device = candidate;
                implementation_->queue_family = family;
                implementation_->device_name = properties.deviceName;
            }
            break;
        }
    }
    if (implementation_->physical_device == VK_NULL_HANDLE) {
        implementation_->error = "no Vulkan compute queue found";
        return;
    }

    constexpr float queue_priority = 1.0F;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = implementation_->queue_family;
    queue_info.queueCount = 1U;
    queue_info.pQueuePriorities = &queue_priority;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.queueCreateInfoCount = 1U;
    device_info.pQueueCreateInfos = &queue_info;
    if (!check(vkCreateDevice(implementation_->physical_device, &device_info, nullptr,
                              &implementation_->device),
               "vkCreateDevice", implementation_->error)) {
        return;
    }
    vkGetDeviceQueue(implementation_->device, implementation_->queue_family, 0U,
                     &implementation_->queue);

    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = implementation_->queue_family;
    if (!check(vkCreateCommandPool(implementation_->device, &pool_info, nullptr,
                                   &implementation_->command_pool),
               "vkCreateCommandPool", implementation_->error)) {
        return;
    }

    VkDescriptorSetLayoutBinding bindings[3]{};
    for (std::uint32_t index = 0U; index < 3U; ++index) {
        bindings[index].binding = index;
        bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[index].descriptorCount = 1U;
        bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout_info.bindingCount = 3U;
    layout_info.pBindings = bindings;
    if (!check(vkCreateDescriptorSetLayout(implementation_->device, &layout_info, nullptr,
                                           &implementation_->descriptor_set_layout),
               "vkCreateDescriptorSetLayout", implementation_->error)) {
        return;
    }
    VkPushConstantRange push_constants{};
    push_constants.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_constants.size = sizeof(std::uint32_t) * 3U;
    VkPipelineLayoutCreateInfo pipeline_layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeline_layout_info.setLayoutCount = 1U;
    pipeline_layout_info.pSetLayouts = &implementation_->descriptor_set_layout;
    pipeline_layout_info.pushConstantRangeCount = 1U;
    pipeline_layout_info.pPushConstantRanges = &push_constants;
    if (!check(vkCreatePipelineLayout(implementation_->device, &pipeline_layout_info, nullptr,
                                      &implementation_->pipeline_layout),
               "vkCreatePipelineLayout", implementation_->error)) {
        return;
    }

    VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shader_info.codeSize = shader.size();
    shader_info.pCode = reinterpret_cast<const std::uint32_t*>(shader.data());
    VkShaderModule shader_module = VK_NULL_HANDLE;
    if (!check(vkCreateShaderModule(implementation_->device, &shader_info, nullptr, &shader_module),
               "vkCreateShaderModule", implementation_->error)) {
        return;
    }
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = shader_module;
    stage.pName = "main";
    VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline_info.stage = stage;
    pipeline_info.layout = implementation_->pipeline_layout;
    const VkResult pipeline_result = vkCreateComputePipelines(
        implementation_->device, VK_NULL_HANDLE, 1U, &pipeline_info, nullptr,
        &implementation_->pipeline);
    vkDestroyShaderModule(implementation_->device, shader_module, nullptr);
    if (!check(pipeline_result, "vkCreateComputePipelines", implementation_->error)) {
        return;
    }

    VkShaderModule elementwise_shader_module = VK_NULL_HANDLE;
    VkShaderModuleCreateInfo elementwise_shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    elementwise_shader_info.codeSize = elementwise_shader.size();
    elementwise_shader_info.pCode = reinterpret_cast<const std::uint32_t*>(elementwise_shader.data());
    if (!check(vkCreateShaderModule(implementation_->device, &elementwise_shader_info, nullptr,
                                    &elementwise_shader_module),
               "vkCreateShaderModule(elementwise)", implementation_->error)) {
        return;
    }
    VkPipelineShaderStageCreateInfo elementwise_stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    elementwise_stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    elementwise_stage.module = elementwise_shader_module;
    elementwise_stage.pName = "main";
    VkComputePipelineCreateInfo elementwise_pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    elementwise_pipeline_info.stage = elementwise_stage;
    elementwise_pipeline_info.layout = implementation_->pipeline_layout;
    const VkResult elementwise_pipeline_result = vkCreateComputePipelines(
        implementation_->device, VK_NULL_HANDLE, 1U, &elementwise_pipeline_info, nullptr,
        &implementation_->elementwise_pipeline);
    vkDestroyShaderModule(implementation_->device, elementwise_shader_module, nullptr);
    if (!check(elementwise_pipeline_result, "vkCreateComputePipelines(elementwise)",
               implementation_->error)) {
        return;
    }

    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 3U;
    VkDescriptorPoolCreateInfo descriptor_pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    descriptor_pool_info.maxSets = 1U;
    descriptor_pool_info.poolSizeCount = 1U;
    descriptor_pool_info.pPoolSizes = &pool_size;
    if (!check(vkCreateDescriptorPool(implementation_->device, &descriptor_pool_info, nullptr,
                                      &implementation_->descriptor_pool),
               "vkCreateDescriptorPool", implementation_->error)) {
        return;
    }
    implementation_->available = true;
#endif
}

VulkanComputeBackend::~VulkanComputeBackend() = default;

bool VulkanComputeBackend::available() const noexcept {
    return implementation_->available;
}

const std::string& VulkanComputeBackend::device_name() const noexcept {
    return implementation_->device_name;
}

const std::string& VulkanComputeBackend::error() const noexcept {
    return implementation_->error;
}

bool VulkanComputeBackend::matmul(const float* left,
                                  std::size_t rows,
                                  std::size_t inner,
                                  const float* right,
                                  std::size_t columns,
                                  float* output) const {
#if !defined(AGENTARI_HAS_VULKAN) || !AGENTARI_HAS_VULKAN
    (void)left;
    (void)rows;
    (void)inner;
    (void)right;
    (void)columns;
    (void)output;
    return false;
#else
    if (!available() || left == nullptr || right == nullptr || output == nullptr || rows == 0U ||
        inner == 0U || columns == 0U || rows > std::numeric_limits<std::uint32_t>::max() ||
        inner > std::numeric_limits<std::uint32_t>::max() ||
        columns > std::numeric_limits<std::uint32_t>::max()) {
        return false;
    }
    // This first backend is synchronous, so completed command resources can
    // be recycled safely for each toy dispatch.
    if (vkResetDescriptorPool(implementation_->device, implementation_->descriptor_pool, 0U) != VK_SUCCESS ||
        vkResetCommandPool(implementation_->device, implementation_->command_pool, 0U) != VK_SUCCESS) {
        return false;
    }
    const auto buffer_size = [](std::size_t count) {
        return static_cast<VkDeviceSize>(count * sizeof(float));
    };
    const auto checked_product = [](std::size_t left_count, std::size_t right_count,
                                    std::size_t& result) {
        if (left_count == 0U || right_count > std::numeric_limits<std::size_t>::max() / left_count) {
            return false;
        }
        result = left_count * right_count;
        return result <= std::numeric_limits<std::size_t>::max() / sizeof(float);
    };
    std::size_t left_count = 0U;
    std::size_t right_count = 0U;
    std::size_t output_count = 0U;
    if (!checked_product(rows, inner, left_count) ||
        !checked_product(inner, columns, right_count) ||
        !checked_product(rows, columns, output_count)) {
        return false;
    }
    Buffer left_buffer;
    Buffer right_buffer;
    Buffer output_buffer;
    if (!make_buffer(implementation_->physical_device, implementation_->device,
                     buffer_size(left_count), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     left_buffer) ||
        !make_buffer(implementation_->physical_device, implementation_->device,
                     buffer_size(right_count), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     right_buffer) ||
        !make_buffer(implementation_->physical_device, implementation_->device,
                     buffer_size(output_count), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     output_buffer)) {
        return false;
    }
    void* mapped = nullptr;
    if (vkMapMemory(implementation_->device, left_buffer.memory, 0U, VK_WHOLE_SIZE, 0U, &mapped) != VK_SUCCESS) {
        return false;
    }
    std::memcpy(mapped, left, left_count * sizeof(float));
    vkUnmapMemory(implementation_->device, left_buffer.memory);
    if (vkMapMemory(implementation_->device, right_buffer.memory, 0U, VK_WHOLE_SIZE, 0U, &mapped) != VK_SUCCESS) {
        return false;
    }
    std::memcpy(mapped, right, right_count * sizeof(float));
    vkUnmapMemory(implementation_->device, right_buffer.memory);

    VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocation.descriptorPool = implementation_->descriptor_pool;
    allocation.descriptorSetCount = 1U;
    allocation.pSetLayouts = &implementation_->descriptor_set_layout;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(implementation_->device, &allocation, &descriptor_set) != VK_SUCCESS) {
        return false;
    }
    VkDescriptorBufferInfo descriptor_buffers[3]{};
    descriptor_buffers[0].buffer = left_buffer.buffer;
    descriptor_buffers[0].range = VK_WHOLE_SIZE;
    descriptor_buffers[1].buffer = right_buffer.buffer;
    descriptor_buffers[1].range = VK_WHOLE_SIZE;
    descriptor_buffers[2].buffer = output_buffer.buffer;
    descriptor_buffers[2].range = VK_WHOLE_SIZE;
    VkWriteDescriptorSet writes[3]{};
    for (std::uint32_t index = 0U; index < 3U; ++index) {
        writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[index].dstSet = descriptor_set;
        writes[index].dstBinding = index;
        writes[index].descriptorCount = 1U;
        writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[index].pBufferInfo = &descriptor_buffers[index];
    }
    vkUpdateDescriptorSets(implementation_->device, 3U, writes, 0U, nullptr);

    VkCommandBufferAllocateInfo command_allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_allocation.commandPool = implementation_->command_pool;
    command_allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_allocation.commandBufferCount = 1U;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(implementation_->device, &command_allocation, &command_buffer) != VK_SUCCESS) {
        return false;
    }
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(command_buffer, &begin) != VK_SUCCESS) {
        return false;
    }
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, implementation_->pipeline);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                             implementation_->pipeline_layout, 0U, 1U, &descriptor_set, 0U, nullptr);
    const std::uint32_t dimensions[3] = {static_cast<std::uint32_t>(rows),
                                         static_cast<std::uint32_t>(inner),
                                         static_cast<std::uint32_t>(columns)};
    vkCmdPushConstants(command_buffer, implementation_->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0U, sizeof(dimensions), dimensions);
    vkCmdDispatch(command_buffer, static_cast<std::uint32_t>((columns - 1U) / 16U + 1U),
                  static_cast<std::uint32_t>((rows - 1U) / 16U + 1U), 1U);
    if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS) {
        return false;
    }
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1U;
    submit.pCommandBuffers = &command_buffer;
    if (vkQueueSubmit(implementation_->queue, 1U, &submit, VK_NULL_HANDLE) != VK_SUCCESS ||
        vkQueueWaitIdle(implementation_->queue) != VK_SUCCESS) {
        return false;
    }
    if (vkMapMemory(implementation_->device, output_buffer.memory, 0U, VK_WHOLE_SIZE, 0U, &mapped) != VK_SUCCESS) {
        return false;
    }
    std::memcpy(output, mapped, output_count * sizeof(float));
    vkUnmapMemory(implementation_->device, output_buffer.memory);
    return true;
#endif
}

bool VulkanComputeBackend::elementwise(ElementwiseOperation operation,
                                       const float* left,
                                       const float* right,
                                       std::size_t count,
                                       float* output) const {
#if !defined(AGENTARI_HAS_VULKAN) || !AGENTARI_HAS_VULKAN
    (void)operation;
    (void)left;
    (void)right;
    (void)count;
    (void)output;
    return false;
#else
    if (!available() || implementation_->elementwise_pipeline == VK_NULL_HANDLE ||
        left == nullptr || output == nullptr || count == 0U ||
        count > std::numeric_limits<std::uint32_t>::max() ||
        static_cast<std::uint32_t>(operation) >
            static_cast<std::uint32_t>(ElementwiseOperation::sigmoid)) {
        return false;
    }
    if (vkResetDescriptorPool(implementation_->device, implementation_->descriptor_pool, 0U) != VK_SUCCESS ||
        vkResetCommandPool(implementation_->device, implementation_->command_pool, 0U) != VK_SUCCESS) {
        return false;
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        return false;
    }
    const VkDeviceSize buffer_size = static_cast<VkDeviceSize>(count * sizeof(float));
    Buffer left_buffer;
    Buffer right_buffer;
    Buffer output_buffer;
    if (!make_buffer(implementation_->physical_device, implementation_->device, buffer_size,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, left_buffer) ||
        !make_buffer(implementation_->physical_device, implementation_->device, buffer_size,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, right_buffer) ||
        !make_buffer(implementation_->physical_device, implementation_->device, buffer_size,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, output_buffer)) {
        return false;
    }
    void* mapped = nullptr;
    if (vkMapMemory(implementation_->device, left_buffer.memory, 0U, VK_WHOLE_SIZE, 0U, &mapped) != VK_SUCCESS) {
        return false;
    }
    std::memcpy(mapped, left, count * sizeof(float));
    vkUnmapMemory(implementation_->device, left_buffer.memory);
    if (vkMapMemory(implementation_->device, right_buffer.memory, 0U, VK_WHOLE_SIZE, 0U, &mapped) != VK_SUCCESS) {
        return false;
    }
    std::memcpy(mapped, right != nullptr ? right : left, count * sizeof(float));
    vkUnmapMemory(implementation_->device, right_buffer.memory);

    VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocation.descriptorPool = implementation_->descriptor_pool;
    allocation.descriptorSetCount = 1U;
    allocation.pSetLayouts = &implementation_->descriptor_set_layout;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(implementation_->device, &allocation, &descriptor_set) != VK_SUCCESS) {
        return false;
    }
    VkDescriptorBufferInfo descriptor_buffers[3]{};
    descriptor_buffers[0].buffer = left_buffer.buffer;
    descriptor_buffers[0].range = VK_WHOLE_SIZE;
    descriptor_buffers[1].buffer = right_buffer.buffer;
    descriptor_buffers[1].range = VK_WHOLE_SIZE;
    descriptor_buffers[2].buffer = output_buffer.buffer;
    descriptor_buffers[2].range = VK_WHOLE_SIZE;
    VkWriteDescriptorSet writes[3]{};
    for (std::uint32_t index = 0U; index < 3U; ++index) {
        writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[index].dstSet = descriptor_set;
        writes[index].dstBinding = index;
        writes[index].descriptorCount = 1U;
        writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[index].pBufferInfo = &descriptor_buffers[index];
    }
    vkUpdateDescriptorSets(implementation_->device, 3U, writes, 0U, nullptr);

    VkCommandBufferAllocateInfo command_allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_allocation.commandPool = implementation_->command_pool;
    command_allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_allocation.commandBufferCount = 1U;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(implementation_->device, &command_allocation, &command_buffer) != VK_SUCCESS) {
        return false;
    }
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(command_buffer, &begin) != VK_SUCCESS) {
        return false;
    }
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      implementation_->elementwise_pipeline);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            implementation_->pipeline_layout, 0U, 1U, &descriptor_set, 0U, nullptr);
    const std::uint32_t parameters[3] = {
        static_cast<std::uint32_t>(count),
        static_cast<std::uint32_t>(operation),
        0U,
    };
    vkCmdPushConstants(command_buffer, implementation_->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0U, sizeof(parameters), parameters);
    vkCmdDispatch(command_buffer, static_cast<std::uint32_t>((count - 1U) / 256U + 1U), 1U, 1U);
    if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS) {
        return false;
    }
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1U;
    submit.pCommandBuffers = &command_buffer;
    if (vkQueueSubmit(implementation_->queue, 1U, &submit, VK_NULL_HANDLE) != VK_SUCCESS ||
        vkQueueWaitIdle(implementation_->queue) != VK_SUCCESS) {
        return false;
    }
    if (vkMapMemory(implementation_->device, output_buffer.memory, 0U, VK_WHOLE_SIZE, 0U, &mapped) != VK_SUCCESS) {
        return false;
    }
    std::memcpy(output, mapped, count * sizeof(float));
    vkUnmapMemory(implementation_->device, output_buffer.memory);
    return true;
#endif
}

}  // namespace agentari::gpu

#if defined(AGENTARI_HAS_VULKAN) && AGENTARI_HAS_VULKAN
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
#endif
