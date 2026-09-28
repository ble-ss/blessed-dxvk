// blessed: cascade-cache -- DxvkContext::blessedRunCascadeBounds, one raw dispatch measuring mesh bounds from the game's own buffers
#include "blessed_cascade_cache.h"

#include "../dxvk_cmdlist.h"
#include "../dxvk_context.h"
#include "../dxvk_device.h"

#include <cstring>

#include <blessed_cascade_bounds.h>

namespace dxvk {

  namespace {

    // must match Job in blessed_cascade_bounds.comp (scalar layout)
    struct BoundsJob {
      uint64_t ib;
      uint64_t pos;
      uint64_t idx;
      uint64_t wt;
      uint64_t result;
      uint32_t index32;
      uint32_t indexCount;
      int32_t  baseVertex;
      uint32_t posStride;
      uint32_t posFormat;
      uint32_t idxStride;
      uint32_t wtStride;
      uint32_t firstGroup;
    };

    static_assert(sizeof(BoundsJob) == 72, "scalar layout of Job in blessed_cascade_bounds.comp");

    // must match the push block in blessed_cascade_bounds.comp
    struct BoundsPush {
      uint64_t table;
      uint32_t jobCount;
      uint32_t groupCount;
    };

    // blessed: raw pipeline, no descriptors (buffer references only), the
    // same model as the compute skinning's BlessedSkinObjects
    class BoundsObjects {
    public:

      explicit BoundsObjects(DxvkDevice* device) {
        m_layout = device->createBuiltInPipelineLayout(DxvkPipelineLayoutFlags(),
          VK_SHADER_STAGE_COMPUTE_BIT, sizeof(BoundsPush), 0u, nullptr);

        util::DxvkBuiltInShaderStage shader(blessed_cascade_bounds, nullptr);
        m_pipeline = device->createBuiltInComputePipeline(m_layout, shader);
      }

      void dispatch(const Rc<DxvkCommandList>& cmd, const BoundsPush& push) {
        cmd->cmdBindPipeline(DxvkCmdBuffer::ExecBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
        cmd->bindResources(DxvkCmdBuffer::ExecBuffer, m_layout, 0u, nullptr, sizeof(push), &push);
        cmd->cmdDispatch(DxvkCmdBuffer::ExecBuffer, push.groupCount, 1u, 1u);
      }

    private:

      const DxvkPipelineLayout* m_layout   = nullptr;
      VkPipeline                m_pipeline = VK_NULL_HANDLE;
    };

    BoundsObjects* Bounds(DxvkDevice* device) {
      static BoundsObjects s_instance(device);
      return &s_instance;
    }

    uint64_t AddressOf(DxvkContext* ctx, const Rc<DxvkBuffer>& buffer, VkDeviceSize offset) {
      if (buffer == nullptr)
        return 0u;

      ctx->ensureBufferAddress(buffer);
      return buffer->getSliceInfo().gpuAddress + offset;
    }

  }


  void DxvkContext::blessedRunCascadeBounds(
    const BlessedCascadeBoundsArgs& args) {
    if (args.requests.empty() || args.results == nullptr)
      return;

    std::vector<BoundsJob> jobs;
    jobs.reserve(args.requests.size());

    uint32_t groups = 0u;
    uint64_t results = AddressOf(this, args.results, 0u);

    for (const auto& r : args.requests) {
      if (r.ib == nullptr || !r.indexCount)
        continue;

      BoundsJob j = { };
      j.ib         = AddressOf(this, r.ib, r.ibOffset);
      j.pos        = AddressOf(this, r.pos, r.posOffset);
      j.idx        = AddressOf(this, r.idx, r.idxOffset);
      j.wt         = AddressOf(this, r.wt, r.wtOffset);
      j.result     = results + VkDeviceSize(r.slot) * sizeof(BlessedCascadeBoundsResult);
      j.index32    = r.index32 ? 1u : 0u;
      j.indexCount = r.indexCount;
      j.baseVertex = r.baseVertex;
      j.posStride  = r.posStride;
      j.posFormat  = r.posFormat;
      j.idxStride  = r.idxStride;
      j.wtStride   = r.wtStride;
      j.firstGroup = groups;
      groups += (r.indexCount + 63u) / 64u;
      jobs.push_back(j);
    }

    if (jobs.empty())
      return;

    DxvkBufferCreateInfo info = { };
    info.size      = sizeof(BoundsJob) * jobs.size();
    info.usage     = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    info.stages    = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    info.access    = VK_ACCESS_2_SHADER_READ_BIT;
    info.debugName = "blessed cascade bounds jobs";

    Rc<DxvkBuffer> table = m_device->createBuffer(info,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    std::memcpy(table->getSliceInfo().mapPtr, jobs.data(), size_t(info.size));

    this->endCurrentPass(true);
    this->flushBarriers();

    // blessed: the game's vertex and index buffers were written by uploads
    // or earlier passes, the result slots by the host: everything before,
    // visible to this compute read
    VkMemoryBarrier2 toCompute = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    toCompute.srcStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_2_HOST_BIT;
    toCompute.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_HOST_WRITE_BIT;
    toCompute.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    toCompute.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;

    VkDependencyInfo dep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dep.memoryBarrierCount = 1u;
    dep.pMemoryBarriers    = &toCompute;
    m_cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &dep);

    BoundsPush push = { };
    push.table      = AddressOf(this, table, 0u);
    push.jobCount   = uint32_t(jobs.size());
    push.groupCount = groups;
    Bounds(m_device.ptr())->dispatch(m_cmd, push);

    // the host reads the results a few frames later
    VkMemoryBarrier2 toHost = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    toHost.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    toHost.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    toHost.dstStageMask  = VK_PIPELINE_STAGE_2_HOST_BIT;
    toHost.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    dep.pMemoryBarriers  = &toHost;
    m_cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &dep);

    // blessed: every buffer read through an address stays alive until the
    // gpu is done with this command list
    m_cmd->track(table, DxvkAccess::Read);
    m_cmd->track(args.results, DxvkAccess::Write);

    for (const auto& r : args.requests) {
      if (r.ib != nullptr)  m_cmd->track(r.ib,  DxvkAccess::Read);
      if (r.pos != nullptr) m_cmd->track(r.pos, DxvkAccess::Read);
      if (r.idx != nullptr) m_cmd->track(r.idx, DxvkAccess::Read);
      if (r.wt != nullptr)  m_cmd->track(r.wt,  DxvkAccess::Read);
    }

    // blessed: state-audit -- raw pipeline and push data, as upstream's meta ops
    this->invalidateState();
  }

}
