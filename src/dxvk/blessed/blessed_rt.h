// blessed: ray query / acceleration structure helper, the plumbing every later tracer stage stands on.
#pragma once

#include <vector>

#include "../dxvk_buffer.h"
#include "../dxvk_device.h"

namespace dxvk {

  class DxvkContext;
  class DxvkCommandList;

  /**
   * \brief Owns one Vulkan acceleration structure plus its backing buffer.
   *
   * The buffer is created with ACCELERATION_STRUCTURE_STORAGE_BIT_KHR and
   * SHADER_DEVICE_ADDRESS_BIT; the VkAccelerationStructureKHR handle lives
   * on top of it. \ref address is only valid once the structure has been
   * built.
   *
   * blessed: ref-counted (derives \c RcObject) and always held as
   * \c Rc<BlessedAccelStruct> so \c DxvkCommandList::track() can pin it --
   * the destructor destroys the handle, so a tracked command list keeps it
   * alive until the GPU submission that reads or builds it has finished.
   * See the rt-lifetime seat report for why this replaced eager destruction.
   */
  /**
   * rief Orders a build after every earlier build on the queue
   *
   * All builds share one scratch buffer, and a later build may read an
   * earlier one's output. Without this, validation reports scratch
   * write-after-write hazards and whiterun lost the device (2026-09-22).
   * The first sync scope of a pipeline barrier includes earlier
   * submissions on the same queue, so this covers cross-frame reuse too.
   */
  void blessedBuildBarrier(DxvkCommandList* cmd);

  class BlessedAccelStruct : public RcObject {

  public:

    BlessedAccelStruct() = default;
    BlessedAccelStruct(DxvkDevice* device,
      Rc<DxvkBuffer> buffer, VkAccelerationStructureKHR handle);

    BlessedAccelStruct             (const BlessedAccelStruct&) = delete;
    BlessedAccelStruct& operator = (const BlessedAccelStruct&) = delete;

    ~BlessedAccelStruct();

    VkAccelerationStructureKHR handle() const { return m_handle; }

    VkDeviceAddress address() const { return m_address; }

    const Rc<DxvkBuffer>& buffer() const { return m_buffer; }

    /**
     * \brief Pins the BLASes a TLAS was built from
     *
     * Meaningful only on a TLAS. Tracking the TLAS's own \c Rc (see
     * blessed_scene.cpp / blessed_shadow.cpp) then transitively keeps every
     * BLAS it references alive for as long as the GPU may still read the
     * TLAS -- independent of the scene cache's own eviction of its BLAS
     * entries.
     */
    void setReferencedBlases(std::vector<Rc<BlessedAccelStruct>> blases) {
      m_referencedBlases = std::move(blases);
    }

  private:

    DxvkDevice*                          m_device  = nullptr;
    Rc<DxvkBuffer>                       m_buffer;
    VkAccelerationStructureKHR           m_handle  = VK_NULL_HANDLE;
    VkDeviceAddress                      m_address = 0u;
    std::vector<Rc<BlessedAccelStruct>>  m_referencedBlases;

    void destroy();

  };


  /**
   * \brief Ray query plumbing: blas/tlas builders plus a self-test.
   *
   * Owned by \ref DxvkDevice, created only when \ref DxvkDevice::supportsRayQuery
   * returns \c true. Everything here costs nothing when that helper does not
   * exist, which is the common case until the tracer proper lands.
   */
  class BlessedRt {

  public:

    explicit BlessedRt(DxvkDevice* device);
    ~BlessedRt();

    /**
     * \brief Creates an acceleration structure and its backing buffer.
     *
     * The buffer is sized exactly as returned by
     * vkGetAccelerationStructureBuildSizesKHR::accelerationStructureSize.
     */
    Rc<BlessedAccelStruct> createAccelStruct(
            VkAccelerationStructureTypeKHR type,
            VkDeviceSize                   size);

    /**
     * \brief Records a one-triangle, non-indexed BLAS build.
     *
     * Grows and pins the shared scratch buffer as needed (see \ref ensureScratch),
     * which is why this takes \p ctx rather than just a raw command buffer.
     * Tracks both the returned BLAS and the scratch buffer against \p cmd
     * before returning, so the caller does not have to.
     *
     * \param [in] ctx Context whose buffer-pinning API to use for the scratch buffer
     * \param [in] cmd Command list to record into (\p ctx's currently active one)
     * \param [in] vertexAddress Device address of 3 tightly packed
     *   VK_FORMAT_R32G32B32_SFLOAT vertices (one triangle, no index buffer)
     * \returns The built BLAS, already tracked against \p cmd. Caller must
     *   still track it (or a TLAS holding it via \ref BlessedAccelStruct::setReferencedBlases)
     *   against every later command list that reads it.
     */
    Rc<BlessedAccelStruct> recordBuildBlas(
            DxvkContext*         ctx,
      const Rc<DxvkCommandList>& cmd,
            VkDeviceAddress      vertexAddress);

    /**
     * \brief Records a TLAS build over \p count instances.
     *
     * Tracks both the returned TLAS and the scratch buffer against \p cmd
     * before returning. Does not track the instances' BLASes -- callers
     * that build multi-instance TLASes must call
     * \ref BlessedAccelStruct::setReferencedBlases on the result.
     *
     * \param [in] ctx Context whose buffer-pinning API to use for the scratch buffer
     * \param [in] cmd Command list to record into
     * \param [in] instanceBufferAddress Device address of a tightly packed
     *   array of \c VkAccelerationStructureInstanceKHR
     * \param [in] count Instance count
     */
    Rc<BlessedAccelStruct> recordBuildTlas(
            DxvkContext*         ctx,
      const Rc<DxvkCommandList>& cmd,
            VkDeviceAddress      instanceBufferAddress,
            uint32_t             count);

    /**
     * \brief Runs the rt selftest: one triangle, four rays, readback.
     *
     * No-op unless BLESSED_RT_SELFTEST=1 is set. Records into its own
     * throwaway context/command list so it never disturbs whatever the
     * real immediate context is doing this frame. Safe to call every
     * present; only fires once.
     */
    void runSelfTestOnce();

  private:

    DxvkDevice*                   m_device;
    Rc<DxvkBuffer>                m_scratch;
    bool                          m_selfTestDone = false;

    // blessed: returns m_scratch's address, growing it first if needed, and
    // tracks whichever buffer is current against cmd -- see blessed_rt.cpp.
    // Replacing m_scratch on growth is then safe even if an older command
    // list has not finished on the GPU: that command list holds its own
    // tracked reference to the old buffer already.
    VkDeviceAddress ensureScratch(DxvkContext* ctx,
      const Rc<DxvkCommandList>& cmd, VkDeviceSize size);

  };

}
