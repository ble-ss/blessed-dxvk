// blessed: actor-skinning diagnostics -- BLESSED_SCENE_SKIN_TRACE readback and the skinned_gpu_ms timestamps
#pragma once

#include <atomic>
#include <fstream>
#include <vector>

#include "../dxvk_buffer.h"
#include "../dxvk_device.h"

namespace dxvk {

  class DxvkContext;
  class DxvkCommandList;
  struct BlessedSceneSkinnedDraw;

  /**
   * \brief One skinned entry handed to \ref BlessedSkinTrace::record
   */
  struct BlessedSkinTraceInput {
    const BlessedSceneSkinnedDraw* draw = nullptr;
    Rc<DxvkBuffer>                 outPositions;
    uint32_t                       occurrence = 0;
  };

  /**
   * \brief BLESSED_SCENE_SKIN_TRACE=1: one-shot readback of the first 16 skinned entries
   *
   * cs thread only. \ref record copies each entry's compute output, its
   * index range and the raw position/index/weight bytes of its first four
   * vertices into one host-visible buffer. \ref poll, called every frame,
   * waits (without blocking) until that copy has completed on the gpu,
   * then writes <BLESSED_PROBE_DIR>/skin-trace.log: per entry the bounding
   * box of the vertices the draw actually indexes, the first four gpu
   * vertices next to a cpu re-skin of the same inputs, the decoded
   * attributes, the pivot and the first bone. Fires once per process.
   */
  class BlessedSkinTrace {

  public:

    static bool enabled();

    // BLESSED_SCENE_SKIN_TRACE_FRAME=<n> (default 300): the first scene
    // frame at or after n that has skinned draws is the one recorded, so
    // loading-screen frames are not what gets traced
    static uint64_t startFrame();

    bool wantsRecord(uint64_t frameIndex) const {
      return m_state == State::Idle && frameIndex >= startFrame();
    }

    void record(
            DxvkDevice*                         device,
            DxvkContext*                        ctx,
      const Rc<DxvkCommandList>&                cmd,
      const std::vector<BlessedSkinTraceInput>& inputs);

    void poll();

  private:

    enum class State : uint32_t { Idle, Pending, Done };

    struct Entry {
      // copied from the draw at record time
      uint32_t     vertexCount = 0;
      uint32_t     indexCount  = 0;
      uint32_t     startIndex  = 0;
      int32_t      baseVertex  = 0;
      bool         index32     = false;
      uint32_t     occurrence  = 0;
      VkFormat     posFormat   = VK_FORMAT_UNDEFINED;
      uint32_t     posStride   = 0;
      uint32_t     idxStride   = 0;
      uint32_t     wtStride    = 0;
      uint32_t     rawCount    = 0;
      uint64_t     posVb = 0, idxVb = 0, wtVb = 0, ib = 0;
      VkDeviceSize posVbOffset = 0, idxVbOffset = 0, wtVbOffset = 0;
      float        pivot[3]    = { };
      float        tagCam[3]   = { };
      bool         hasTagCam   = false;
      uint32_t     bonesUsage  = 0;
      float        bones[240 * 4];

      // offsets into the readback buffer
      VkDeviceSize outOffset = 0;
      VkDeviceSize ibOffset  = 0;
      VkDeviceSize posOffset = 0;
      VkDeviceSize idxOffset = 0;
      VkDeviceSize wtOffset  = 0;
    };

    State              m_state = State::Idle;
    std::vector<Entry> m_entries;
    Rc<DxvkBuffer>     m_readback;
    uint32_t           m_pollFrames = 0;

    void writeLog();

  };


  /**
   * \brief GPU time of the skin dispatches + blas builds/refits, per frame
   *
   * Same non-blocking ring as the shadow pass's timer: a pair of timestamps
   * per frame, collected when its slot comes round again. Results are summed
   * into the two counters handed to the constructor (read by scene.jsonl).
   */
  class BlessedSkinTimer {

  public:

    BlessedSkinTimer(
            DxvkDevice*             device,
            std::atomic<uint64_t>*  nsSum,
            std::atomic<uint32_t>*  samples);

    ~BlessedSkinTimer();

    uint32_t begin(const Rc<DxvkCommandList>& cmd);

    void end(const Rc<DxvkCommandList>& cmd, uint32_t base);

  private:

    static constexpr uint32_t PairCount = 4u;

    DxvkDevice*            m_device;
    std::atomic<uint64_t>* m_nsSum;
    std::atomic<uint32_t>* m_samples;
    VkQueryPool            m_pool = VK_NULL_HANDLE;
    uint32_t               m_next = 0u;
    bool                   m_used[PairCount] = { };

    void collect(uint32_t base);

  };

}
