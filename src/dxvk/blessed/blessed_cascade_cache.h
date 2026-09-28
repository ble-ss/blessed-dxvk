// blessed: cascade-cache -- gpu half of the cached static sun cascades: composite of a cached depth slice into the live one, and the verify compare
#pragma once

#include <cstdint>
#include <vector>

#include "../dxvk_buffer.h"
#include "../dxvk_image.h"
#include "../dxvk_shader.h"

namespace dxvk {

  class DxvkContext;
  class DxvkDevice;

  /**
   * \brief Counters the verify compare writes (host-visible, read later)
   *
   * All four are texel counts over one cascade slice except \c maxDiff,
   * the largest difference in d16 steps.
   */
  struct BlessedCascadeVerifyCounters {
    uint32_t differ  = 0u;  ///< cached != fresh
    uint32_t ghost   = 0u;  ///< cached nearer than fresh: a caster the cache has and this frame lacks
    uint32_t missing = 0u;  ///< fresh nearer than cached: a caster this frame drew that the cache lacks
    uint32_t maxDiff = 0u;
  };

  /**
   * \brief One mesh's bounds, as blessed_cascade_bounds.comp writes them
   *
   * 64 bytes per slot of the results buffer. Positions are object space,
   * as ordered float bits (see OrderedToFloat on the d3d11 side); the
   * host fills the slot with the empty values before the dispatch.
   */
  struct BlessedCascadeBoundsResult {
    uint32_t minPos[3];
    uint32_t maxPos[3];
    uint32_t minIndex;
    uint32_t maxIndex;
    uint32_t boneMask[3];
    uint32_t done;
    uint32_t pad[4];
  };

  static_assert(sizeof(BlessedCascadeBoundsResult) == 64);

  /**
   * \brief One mesh to measure: a draw's index range over its streams
   *
   * Offsets are byte offsets into the buffers: \c ibOffset of the draw's
   * first index, the others of vertex 0's attribute. A null \c pos means
   * the position stream is dynamic (the host reads it itself); a null
   * \c idx means not skinned.
   */
  struct BlessedCascadeBoundsRequest {
    Rc<DxvkBuffer>  ib;
    VkDeviceSize    ibOffset    = 0u;
    bool            index32     = false;
    uint32_t        indexCount  = 0u;
    int32_t         baseVertex  = 0;
    Rc<DxvkBuffer>  pos;
    VkDeviceSize    posOffset   = 0u;
    uint32_t        posStride   = 0u;
    uint32_t        posFormat   = 0u;   // 0 rgb32f, 1 rgba32f, 2 rgba16f
    Rc<DxvkBuffer>  idx;
    VkDeviceSize    idxOffset   = 0u;
    uint32_t        idxStride   = 0u;
    Rc<DxvkBuffer>  wt;
    VkDeviceSize    wtOffset    = 0u;
    uint32_t        wtStride    = 0u;
    uint32_t        slot        = 0u;   // BlessedCascadeBoundsResult index
  };

  /// DxvkContext::blessedRunCascadeBounds: one dispatch for all of them
  struct BlessedCascadeBoundsArgs {
    std::vector<BlessedCascadeBoundsRequest> requests;
    Rc<DxvkBuffer>  results;
  };

  /**
   * \brief The cached-cascade gpu passes (BLESSED_CASCADE_CACHE)
   *
   * Two passes, both recorded through the ordinary DxvkContext api so dxvk
   * keeps doing its own layout and barrier tracking:
   *
   * - \c composite: a fullscreen triangle that writes the cached depth
   *   through \c gl_FragDepth with a LESS_OR_EQUAL test and depth writes
   *   on, so the live slice ends as min(live, cached). Used on the frame
   *   that rebuilds the cache: the static casters went into the cache,
   *   the moving ones into the live slice, and this merges them. Depth
   *   testing is order-independent, so the result is exactly what one
   *   target with every draw would hold.
   * - \c verify: a compute pass that compares two d16 slices texel by
   *   texel and adds the counts into a host-visible buffer.
   *
   * Both leave the context's graphics or compute state changed. The d3d11
   * side resets its command list state before and restores it after, the
   * way the video blit does.
   */
  class BlessedCascadeCachePass : public RcObject {

  public:

    explicit BlessedCascadeCachePass(DxvkDevice* device);

    ~BlessedCascadeCachePass();

    /**
     * \brief dst = min(dst, src), over the whole of \c extent
     *
     * \param [in] dstDepth Depth view of the live slice (one layer)
     * \param [in] srcSampled Sampled view of the cached slice (depth aspect)
     */
    void composite(
            DxvkContext*          ctx,
      const Rc<DxvkImageView>&    dstDepth,
      const Rc<DxvkImageView>&    srcSampled,
            VkExtent2D            extent,
            VkCompareOp           compareOp = VK_COMPARE_OP_LESS_OR_EQUAL);  // ALWAYS: a plain overwrite (RESTORE=draw)

    /**
     * \brief Counts differing texels of two d16 slices into \c counters
     *
     * \c counters must be host-visible storage with a device address;
     * the pass adds into it and never clears it.
     */
    void verify(
            DxvkContext*          ctx,
      const Rc<DxvkImageView>&    cachedSampled,
      const Rc<DxvkImageView>&    freshSampled,
      const Rc<DxvkBuffer>&       counters,
            VkExtent2D            extent);

  private:

    DxvkDevice*     m_device;

    Rc<DxvkShader>  m_vs;
    Rc<DxvkShader>  m_fs;
    Rc<DxvkShader>  m_verify;

  };

}
