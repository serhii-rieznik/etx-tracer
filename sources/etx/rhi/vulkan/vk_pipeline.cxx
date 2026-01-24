#include <etx/rhi/vulkan/vk_pipeline.hxx>
#include <etx/rhi/vulkan/vk_shader.hxx>
#include <etx/rhi/vulkan/vk_utils.hxx>

#include <etx/core/log.hxx>

namespace etx {

static VkCompareOp convert_compare_op(RHICompareOp op) {
  switch (op) {
    case RHICompareOp::Never:
      return VK_COMPARE_OP_NEVER;
    case RHICompareOp::Less:
      return VK_COMPARE_OP_LESS;
    case RHICompareOp::Equal:
      return VK_COMPARE_OP_EQUAL;
    case RHICompareOp::LessOrEqual:
      return VK_COMPARE_OP_LESS_OR_EQUAL;
    case RHICompareOp::Greater:
      return VK_COMPARE_OP_GREATER;
    case RHICompareOp::NotEqual:
      return VK_COMPARE_OP_NOT_EQUAL;
    case RHICompareOp::GreaterOrEqual:
      return VK_COMPARE_OP_GREATER_OR_EQUAL;
    case RHICompareOp::Always:
      return VK_COMPARE_OP_ALWAYS;
    default:
      return VK_COMPARE_OP_LESS;
  }
}

static VkBlendFactor convert_blend_factor(RHIBlendFactor factor) {
  switch (factor) {
    case RHIBlendFactor::Zero:
      return VK_BLEND_FACTOR_ZERO;
    case RHIBlendFactor::One:
      return VK_BLEND_FACTOR_ONE;
    case RHIBlendFactor::SrcColor:
      return VK_BLEND_FACTOR_SRC_COLOR;
    case RHIBlendFactor::OneMinusSrcColor:
      return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
    case RHIBlendFactor::DstColor:
      return VK_BLEND_FACTOR_DST_COLOR;
    case RHIBlendFactor::OneMinusDstColor:
      return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
    case RHIBlendFactor::SrcAlpha:
      return VK_BLEND_FACTOR_SRC_ALPHA;
    case RHIBlendFactor::OneMinusSrcAlpha:
      return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    case RHIBlendFactor::DstAlpha:
      return VK_BLEND_FACTOR_DST_ALPHA;
    case RHIBlendFactor::OneMinusDstAlpha:
      return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
    case RHIBlendFactor::ConstantColor:
      return VK_BLEND_FACTOR_CONSTANT_COLOR;
    case RHIBlendFactor::OneMinusConstantColor:
      return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
    case RHIBlendFactor::ConstantAlpha:
      return VK_BLEND_FACTOR_CONSTANT_ALPHA;
    case RHIBlendFactor::OneMinusConstantAlpha:
      return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
    case RHIBlendFactor::SrcAlphaSaturate:
      return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
    default:
      return VK_BLEND_FACTOR_ONE;
  }
}

static VkBlendOp convert_blend_op(RHIBlendOp op) {
  switch (op) {
    case RHIBlendOp::Add:
      return VK_BLEND_OP_ADD;
    case RHIBlendOp::Subtract:
      return VK_BLEND_OP_SUBTRACT;
    case RHIBlendOp::ReverseSubtract:
      return VK_BLEND_OP_REVERSE_SUBTRACT;
    case RHIBlendOp::Min:
      return VK_BLEND_OP_MIN;
    case RHIBlendOp::Max:
      return VK_BLEND_OP_MAX;
    default:
      return VK_BLEND_OP_ADD;
  }
}

VKPipeline::VKPipeline(VkDevice device, VkDescriptorSetLayout bindless_layout)
  : _device(device)
  , _bindless_layout(bindless_layout) {
}

VKPipeline::~VKPipeline() {
  if (_pipeline != VK_NULL_HANDLE && _device != VK_NULL_HANDLE) {
    vkDestroyPipeline(_device, _pipeline, nullptr);
    _pipeline = VK_NULL_HANDLE;
  }

  if (_pipeline_layout != VK_NULL_HANDLE && _device != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(_device, _pipeline_layout, nullptr);
    _pipeline_layout = VK_NULL_HANDLE;
  }
}

bool VKPipeline::create_bindless_pipeline_layout() {
  VkPipelineLayoutCreateInfo layout_info = {};
  layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layout_info.setLayoutCount = 1;
  layout_info.pSetLayouts = &_bindless_layout;

  VkPushConstantRange push_constants = {};
  push_constants.stageFlags = VK_SHADER_STAGE_ALL;
  push_constants.offset = 0;
  push_constants.size = kVKMaxPushConstantsSize;

  layout_info.pushConstantRangeCount = 1;
  layout_info.pPushConstantRanges = &push_constants;

  VkResult result = vkCreatePipelineLayout(_device, &layout_info, nullptr, &_pipeline_layout);
  if (result != VK_SUCCESS) {
    log::error("Failed to create bindless pipeline layout: %d", static_cast<int>(result));
    return false;
  }

  return true;
}

VKComputePipeline::VKComputePipeline(VkDevice device, VkDescriptorSetLayout bindless_layout)
  : VKPipeline(device, bindless_layout) {
}

bool VKComputePipeline::create_compute_pipeline(const RHIComputePipelineDesc& desc) {
  if (!create_bindless_pipeline_layout()) {
    log::error("Failed to create pipeline layout for compute pipeline");
    return false;
  }

  VKShader compute_shader(_device, desc.compute_shader);
  if (!compute_shader.is_valid()) {
    log::error("Failed to create compute shader module: %s", compute_shader.get_last_error().c_str());
    return false;
  }

  VkPipelineShaderStageCreateInfo shader_stage = {};
  shader_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  shader_stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  shader_stage.module = compute_shader.get_vk_shader_module();
  shader_stage.pName = desc.entry_point.c_str();

  VkComputePipelineCreateInfo pipeline_info = {};
  pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipeline_info.stage = shader_stage;
  pipeline_info.layout = _pipeline_layout;
  pipeline_info.basePipelineHandle = VK_NULL_HANDLE;
  pipeline_info.basePipelineIndex = -1;

  VkResult result = vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &_pipeline);
  if (result != VK_SUCCESS) {
    log::error("Failed to create compute pipeline: %d", static_cast<int>(result));
    return false;
  }

  return true;
}

VKGraphicsPipeline::VKGraphicsPipeline(VkDevice device, VkDescriptorSetLayout bindless_layout)
  : VKPipeline(device, bindless_layout) {
}

bool VKGraphicsPipeline::create_graphics_pipeline(const RHIGraphicsPipelineDesc& desc) {
  if (!create_bindless_pipeline_layout()) {
    log::error("Failed to create pipeline layout for graphics pipeline");
    return false;
  }

  VKShader vertex_shader(_device, desc.vertex_shader);
  if (!vertex_shader.is_valid()) {
    log::error("Failed to create vertex shader module");
    return false;
  }

  VKShader fragment_shader(_device, desc.fragment_shader);
  if (!fragment_shader.is_valid()) {
    log::error("Failed to create fragment shader module");
    return false;
  }

  VkPipelineShaderStageCreateInfo shader_stages[] = {{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                       .stage = VK_SHADER_STAGE_VERTEX_BIT,
                                                       .module = vertex_shader.get_vk_shader_module(),
                                                       .pName = desc.vertex_entry_point.c_str()},
    {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
      .module = fragment_shader.get_vk_shader_module(),
      .pName = desc.fragment_entry_point.c_str()}};

  std::vector<VkVertexInputBindingDescription> vertex_bindings;
  std::vector<VkVertexInputAttributeDescription> vertex_attributes;

  for (uint32_t i = 0; i < desc.vertex_binding_count; ++i) {
    const auto& binding = desc.vertex_bindings[i];
    VkVertexInputBindingDescription vk_binding = {};
    vk_binding.binding = binding.binding;
    vk_binding.stride = binding.stride;
    vk_binding.inputRate = binding.input_rate == RHIVertexInputRate::Vertex ? VK_VERTEX_INPUT_RATE_VERTEX : VK_VERTEX_INPUT_RATE_INSTANCE;
    vertex_bindings.push_back(vk_binding);
  }

  for (uint32_t i = 0; i < desc.vertex_attribute_count; ++i) {
    const auto& attr = desc.vertex_attributes[i];
    VkVertexInputAttributeDescription vk_attr = {};
    vk_attr.location = attr.location;
    vk_attr.binding = attr.binding;

    switch (attr.format) {
      case RHIVertexFormat::Float2:
        vk_attr.format = VK_FORMAT_R32G32_SFLOAT;
        break;
      case RHIVertexFormat::Float3:
        vk_attr.format = VK_FORMAT_R32G32B32_SFLOAT;
        break;
      case RHIVertexFormat::Float4:
        vk_attr.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        break;
      default:
        vk_attr.format = VK_FORMAT_R32G32_SFLOAT;
        break;
    }

    vk_attr.offset = attr.offset;
    vertex_attributes.push_back(vk_attr);
  }

  VkPipelineVertexInputStateCreateInfo vertex_input = {};
  vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertex_input.vertexBindingDescriptionCount = static_cast<uint32_t>(vertex_bindings.size());
  vertex_input.pVertexBindingDescriptions = vertex_bindings.data();
  vertex_input.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertex_attributes.size());
  vertex_input.pVertexAttributeDescriptions = vertex_attributes.data();

  VkPipelineInputAssemblyStateCreateInfo input_assembly = {};
  input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  input_assembly.topology = desc.primitive_topology == RHIPrimitiveTopology::TriangleList ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
  input_assembly.primitiveRestartEnable = VK_FALSE;

  VkPipelineViewportStateCreateInfo viewport_state = {};
  viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewport_state.viewportCount = 1;
  viewport_state.scissorCount = 1;

  VkPipelineRasterizationStateCreateInfo rasterizer = {};
  rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.depthClampEnable = desc.rasterization.depth_clamp_enable ? VK_TRUE : VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
  rasterizer.lineWidth = desc.rasterization.line_width;
  rasterizer.cullMode = VK_CULL_MODE_NONE;
  rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
  rasterizer.depthBiasEnable = VK_FALSE;

  VkPipelineDepthStencilStateCreateInfo depth_stencil = {};
  depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depth_stencil.depthTestEnable = desc.depth_stencil.depth_test_enable ? VK_TRUE : VK_FALSE;
  depth_stencil.depthWriteEnable = desc.depth_stencil.depth_write_enable ? VK_TRUE : VK_FALSE;
  depth_stencil.depthCompareOp = convert_compare_op(desc.depth_stencil.depth_compare_op);
  depth_stencil.depthBoundsTestEnable = desc.depth_stencil.depth_bounds_test_enable ? VK_TRUE : VK_FALSE;
  depth_stencil.minDepthBounds = desc.depth_stencil.min_depth_bounds;
  depth_stencil.maxDepthBounds = desc.depth_stencil.max_depth_bounds;
  depth_stencil.stencilTestEnable = VK_FALSE;
  depth_stencil.front = {};
  depth_stencil.back = {};

  VkPipelineMultisampleStateCreateInfo multisampling = {};
  multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineColorBlendAttachmentState color_blend_attachment = {};
  color_blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  color_blend_attachment.blendEnable = desc.blend.blend_enable ? VK_TRUE : VK_FALSE;
  if (desc.blend.blend_enable) {
    color_blend_attachment.srcColorBlendFactor = convert_blend_factor(desc.blend.src_color_blend_factor);
    color_blend_attachment.dstColorBlendFactor = convert_blend_factor(desc.blend.dst_color_blend_factor);
    color_blend_attachment.colorBlendOp = convert_blend_op(desc.blend.color_blend_op);
    color_blend_attachment.srcAlphaBlendFactor = convert_blend_factor(desc.blend.src_alpha_blend_factor);
    color_blend_attachment.dstAlphaBlendFactor = convert_blend_factor(desc.blend.dst_alpha_blend_factor);
    color_blend_attachment.alphaBlendOp = convert_blend_op(desc.blend.alpha_blend_op);
  }

  VkPipelineColorBlendStateCreateInfo color_blending = {};
  color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  color_blending.logicOpEnable = VK_FALSE;
  color_blending.attachmentCount = 1;
  color_blending.pAttachments = &color_blend_attachment;

  VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dynamic_state = {};
  dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamic_state.dynamicStateCount = 2;
  dynamic_state.pDynamicStates = dynamic_states;

  VkFormat color_format = VK_FORMAT_B8G8R8A8_SRGB;
  if (desc.color_attachment_count > 0) {
    color_format = convert_rhi_format_to_vk(desc.color_formats[0]);
    if (color_format == VK_FORMAT_UNDEFINED) {
      color_format = VK_FORMAT_B8G8R8A8_SRGB;
    }
  }

  VkAttachmentDescription color_attachment = {};
  color_attachment.format = color_format;
  color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
  color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

  VkAttachmentReference color_attachment_ref = {};
  color_attachment_ref.attachment = 0;
  color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass = {};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &color_attachment_ref;

  VkSubpassDependency dependencies[2] = {};

  dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
  dependencies[0].dstSubpass = 0;
  dependencies[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependencies[0].srcAccessMask = 0;
  dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

  dependencies[1].srcSubpass = 0;
  dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
  dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
  dependencies[1].dstAccessMask = 0;

  VkRenderPassCreateInfo render_pass_info = {};
  render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  render_pass_info.attachmentCount = 1;
  render_pass_info.pAttachments = &color_attachment;
  render_pass_info.subpassCount = 1;
  render_pass_info.pSubpasses = &subpass;
  render_pass_info.dependencyCount = 2;
  render_pass_info.pDependencies = dependencies;

  VkRenderPass temp_render_pass;
  VkResult rp_result = vkCreateRenderPass(_device, &render_pass_info, nullptr, &temp_render_pass);
  if (rp_result != VK_SUCCESS) {
    log::error("Failed to create temporary render pass for pipeline: %d", static_cast<int>(rp_result));
    return false;
  }

  VkGraphicsPipelineCreateInfo pipeline_info = {};
  pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipeline_info.stageCount = 2;
  pipeline_info.pStages = shader_stages;
  pipeline_info.pVertexInputState = &vertex_input;
  pipeline_info.pInputAssemblyState = &input_assembly;
  pipeline_info.pViewportState = &viewport_state;
  pipeline_info.pRasterizationState = &rasterizer;
  pipeline_info.pMultisampleState = &multisampling;
  pipeline_info.pDepthStencilState = &depth_stencil;
  pipeline_info.pColorBlendState = &color_blending;
  pipeline_info.pDynamicState = &dynamic_state;
  pipeline_info.layout = _pipeline_layout;
  pipeline_info.renderPass = temp_render_pass;
  pipeline_info.subpass = 0;
  pipeline_info.basePipelineHandle = VK_NULL_HANDLE;

  VkResult result = vkCreateGraphicsPipelines(_device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &_pipeline);
  if (result != VK_SUCCESS) {
    log::error("Failed to create graphics pipeline: %d", static_cast<int>(result));
    vkDestroyRenderPass(_device, temp_render_pass, nullptr);
    return false;
  }

  vkDestroyRenderPass(_device, temp_render_pass, nullptr);

  return true;
}

}  // namespace etx
