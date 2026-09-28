// blessed: ray query / acceleration structure helper -- see blessed_rt.h
#include "blessed_rt.h"

#include "../dxvk_context.h"
#include "../dxvk_cmdlist.h"

#include "../../util/util_math.h"
#include "../../util/util_env.h"
#include "../../util/util_string.h"
#include "../../util/log/log.h"

namespace dxvk {

  BlessedAccelStruct::BlessedAccelStruct(
          DxvkDevice*                 device,
          Rc<DxvkBuffer>              buffer,
          VkAccelerationStructureKHR  handle)
  : m_device(device), m_buffer(std::move(buffer)), m_handle(handle) {
    if (m_handle) {
      VkAccelerationStructureDeviceAddressInfoKHR addressInfo =
        { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR };
      addressInfo.accelerationStructure = m_handle;

      // blessed: raw -- no dxvk wrapper for acceleration structure entry points
      m_address = m_device->vkd()->vkGetAccelerationStructureDeviceAddressKHR(
        m_device->handle(), &addressInfo);
    }
  }


  BlessedAccelStruct::~BlessedAccelStruct() {
    destroy();
  }


  void BlessedAccelStruct::destroy() {
    if (m_handle) {
      // blessed: raw
      m_device->vkd()->vkDestroyAccelerationStructureKHR(
        m_device->handle(), m_handle, nullptr);
      m_handle = VK_NULL_HANDLE;
    }
  }


  BlessedRt::BlessedRt(DxvkDevice* device)
  : m_device(device) {

  }


  BlessedRt::~BlessedRt() {

  }


  Rc<BlessedAccelStruct> BlessedRt::createAccelStruct(
          VkAccelerationStructureTypeKHR type,
          VkDeviceSize                   size) {
    DxvkBufferCreateInfo bufferInfo = { };
    bufferInfo.size    = size;
    bufferInfo.usage   = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                        | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bufferInfo.stages  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    bufferInfo.access  = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR
                        | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    bufferInfo.debugName = "blessed rt accel struct";

    Rc<DxvkBuffer> buffer = m_device->createBuffer(bufferInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    DxvkResourceBufferInfo slice = buffer->getSliceInfo();

    VkAccelerationStructureCreateInfoKHR createInfo =
      { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
    createInfo.buffer = slice.buffer;
    createInfo.offset = slice.offset;
    createInfo.size   = size;
    createInfo.type   = type;

    VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
    // blessed: raw -- no dxvk wrapper for acceleration structure entry points
    VkResult vr = m_device->vkd()->vkCreateAccelerationStructureKHR(
      m_device->handle(), &createInfo, nullptr, &handle);

    if (vr != VK_SUCCESS) {
      Logger::err(str::format("blessed: vkCreateAccelerationStructureKHR failed: ", vr));
      return new BlessedAccelStruct(m_device, std::move(buffer), VK_NULL_HANDLE);
    }

    return new BlessedAccelStruct(m_device, std::move(buffer), handle);
  }


  VkDeviceAddress BlessedRt::ensureScratch(DxvkContext* ctx,
    const Rc<DxvkCommandList>& cmd, VkDeviceSize size) {
    VkDeviceSize alignment = m_device->properties().khrAccelerationStructure
      .minAccelerationStructureScratchOffsetAlignment;
    VkDeviceSize alignedSize = align(size, alignment);

    if (m_scratch == nullptr || m_scratch->info().size < alignedSize) {
      DxvkBufferCreateInfo bufferInfo = { };
      bufferInfo.size    = alignedSize;
      bufferInfo.usage   = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                          | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
      bufferInfo.stages  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
      bufferInfo.access  = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
      bufferInfo.debugName = "blessed rt scratch";

      // blessed: dropping the old m_scratch Rc here is safe even if a
      // previous, still-in-flight command list built against its address --
      // that command list tracked it below on its own turn through here.
      m_scratch = m_device->createBuffer(bufferInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      ctx->ensureBufferAddress(m_scratch);
    }

    // blessed: keep the buffer alive until the GPU finishes the build cmd
    // is about to record against the address returned below.
    cmd->track(m_scratch);

    return m_scratch->getSliceInfo().gpuAddress;
  }


  void blessedBuildBarrier(DxvkCommandList* cmd) {
    VkMemoryBarrier2 barrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    barrier.srcStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    barrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR
                          | VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    barrier.dstStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR
                          | VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;

    VkDependencyInfo dep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dep.memoryBarrierCount = 1u;
    dep.pMemoryBarriers    = &barrier;

    // blessed: raw
    cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &dep);
  }


  Rc<BlessedAccelStruct> BlessedRt::recordBuildBlas(
          DxvkContext*         ctx,
    const Rc<DxvkCommandList>& cmd,
          VkDeviceAddress      vertexAddress) {
    VkAccelerationStructureGeometryTrianglesDataKHR triangles =
      { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR };
    triangles.vertexFormat             = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexData.deviceAddress = vertexAddress;
    triangles.vertexStride             = sizeof(float) * 3u;
    triangles.maxVertex                = 2u;
    triangles.indexType                = VK_INDEX_TYPE_NONE_KHR;

    VkAccelerationStructureGeometryKHR geometry =
      { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
    geometry.geometryType       = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.geometry.triangles = triangles;
    geometry.flags              = VK_GEOMETRY_OPAQUE_BIT_KHR;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo =
      { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
    buildInfo.type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.geometryCount = 1u;
    buildInfo.pGeometries   = &geometry;

    const uint32_t primitiveCount = 1u;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo =
      { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
    // blessed: raw
    m_device->vkd()->vkGetAccelerationStructureBuildSizesKHR(m_device->handle(),
      VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &primitiveCount, &sizeInfo);

    Rc<BlessedAccelStruct> blas = createAccelStruct(
      VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, sizeInfo.accelerationStructureSize);

    buildInfo.dstAccelerationStructure  = blas->handle();
    buildInfo.scratchData.deviceAddress = ensureScratch(ctx, cmd, sizeInfo.buildScratchSize);

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo = { };
    rangeInfo.primitiveCount = primitiveCount;
    const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &rangeInfo;

    blessedBuildBarrier(cmd.ptr());
    cmd->cmdBuildAccelerationStructures(DxvkCmdBuffer::ExecBuffer, 1u, &buildInfo, &pRangeInfo);

    // blessed: keep the blas alive until this build finishes on the GPU;
    // a tlas built from it (or the caller directly) must track it again
    // for however much longer it needs to outlive this one submission.
    cmd->track(blas);

    return blas;
  }


  Rc<BlessedAccelStruct> BlessedRt::recordBuildTlas(
          DxvkContext*         ctx,
    const Rc<DxvkCommandList>& cmd,
          VkDeviceAddress      instanceBufferAddress,
          uint32_t             count) {
    VkAccelerationStructureGeometryInstancesDataKHR instances =
      { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR };
    instances.arrayOfPointers    = VK_FALSE;
    instances.data.deviceAddress = instanceBufferAddress;

    VkAccelerationStructureGeometryKHR geometry =
      { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
    geometry.geometryType      = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.geometry.instances = instances;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo =
      { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
    buildInfo.type          = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.geometryCount = 1u;
    buildInfo.pGeometries   = &geometry;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo =
      { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
    // blessed: raw
    m_device->vkd()->vkGetAccelerationStructureBuildSizesKHR(m_device->handle(),
      VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &count, &sizeInfo);

    Rc<BlessedAccelStruct> tlas = createAccelStruct(
      VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, sizeInfo.accelerationStructureSize);

    buildInfo.dstAccelerationStructure  = tlas->handle();
    buildInfo.scratchData.deviceAddress = ensureScratch(ctx, cmd, sizeInfo.buildScratchSize);

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo = { };
    rangeInfo.primitiveCount = count;
    const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &rangeInfo;

    blessedBuildBarrier(cmd.ptr());
    cmd->cmdBuildAccelerationStructures(DxvkCmdBuffer::ExecBuffer, 1u, &buildInfo, &pRangeInfo);

    // blessed: keep the tlas alive until this build finishes on the GPU;
    // whoever later reads it (a ray query dispatch) must track it again --
    // see BlessedScene::endFrame and DxvkContext::blessedRunShadowPass.
    cmd->track(tlas);

    return tlas;
  }


  void BlessedRt::runSelfTestOnce() {
    if (m_selfTestDone)
      return;

    if (env::getEnvVar("BLESSED_RT_SELFTEST") != "1")
      return;

    m_selfTestDone = true;

    // blessed: a dedicated throwaway context/command list, never the
    // immediate context's own -- the selftest must not disturb whatever
    // frame is actually being recorded when Present fires.
    Rc<DxvkContext> ctx = m_device->createContext();
    ctx->beginRecording(m_device->createCommandList());

    // blessedRunSelfTest records, ends recording, submits and waits for
    // device idle before returning, so the GPU is guaranteed done with
    // every scratch buffer and accel struct by the time we get back here.
    ctx->blessedRunSelfTest();
  }

}
