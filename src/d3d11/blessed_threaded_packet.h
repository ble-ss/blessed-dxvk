// blessed: threaded-fe-2 -- stage 3: draw packets (compact ops sharing one record) and the game-side shadow that folds redundant binds
#pragma once

#include <cstring>

#include "blessed_threaded_context.h"

#include "d3d11_buffer.h"

namespace dxvk {

  /**
   * \brief Ops a draw packet carries
   *
   * The hot per-draw sequence (a cb-ring map, one-slot binds, the draw)
   * as 8 to 24 byte ops behind one record header, instead of one record
   * with its own 8-byte thunk each.
   */
  enum class FeOp : uint8_t {
    CbRename,
    SetCb,
    SetSrv,
    SetSampler,
    SetShader,
    SetVb,
    SetIb,
    SetInputLayout,
    SetTopology,
    Draw,
    DrawIndexed,
    DrawInstanced,
    DrawIndexedInstanced,
  };

  /// Shader stages as the ops number them
  enum class FeStage : uint8_t { VS, HS, DS, GS, PS, CS, Count };

  struct FeOpHeader {
    FeOp      op;
    FeStage   stage;
    uint16_t  slot;   // slot, or an index buffer's format
    uint32_t  arg;    // the op's first 32-bit argument
  };

  /// CbRename: arg = chunk offset
  //
  // blessed: cb-mirror -- deliberately does not carry the chunk's mirror
  // (BlessedCbRingChunk::mirror): every field here costs 8 bytes on every
  // cb-ring map in the hot compact-packet path, mirrored or not, and this
  // is perf-max's common path (the threaded front end's compact mode).
  // Replay resolves the mirror instead, from the block already carried
  // here, via BlessedCbRing::findMirror (D3D11ImmediateContext::
  // BlessedFindCbMirror) -- see the FeOp::CbRename case in
  // blessed_threaded_record.cpp. The slow record path (FeCbRenameRec
  // below) never needed this: it already carries the whole chunk, mirror
  // included, since it copies BlessedCbRingChunk by value.
  struct FeOpCbRename {
    FeOpHeader                h;
    D3D11Buffer*              buffer;
    DxvkResourceAllocation*   block;
  };

  /// SetCb, SetSrv, SetSampler (with a slot), SetShader, SetInputLayout
  struct FeOpPtr {
    FeOpHeader  h;
    void*       ptr;
  };

  /// SetVb: slot = slot, arg = stride
  struct FeOpVb {
    FeOpHeader    h;
    ID3D11Buffer* buffer;
    UINT          offset;
    UINT          pad;
  };

  /// SetIb: slot = format, arg = offset
  struct FeOpIb {
    FeOpHeader    h;
    ID3D11Buffer* buffer;
  };

  /// SetTopology: arg = topology
  struct FeOpTopology {
    FeOpHeader  h;
  };

  /// Draw: arg = vertex count. DrawIndexed: arg = index count.
  struct FeOpDraw {
    FeOpHeader  h;
    UINT        start;
    INT         base;
  };

  /// DrawInstanced, DrawIndexedInstanced: arg = count per instance
  struct FeOpDrawInstanced {
    FeOpHeader  h;
    UINT        instances;
    UINT        start;
    INT         base;
    UINT        startInstance;
  };

  static_assert(sizeof(FeOpCbRename) == 24u && sizeof(FeOpPtr) == 16u
    && sizeof(FeOpVb) == 24u && sizeof(FeOpIb) == 16u && sizeof(FeOpTopology) == 8u
    && sizeof(FeOpDraw) == 16u && sizeof(FeOpDrawInstanced) == 24u,
    "draw packet ops are 8-byte multiples");


  /**
   * \brief A draw packet: header, then ops back to back
   *
   * Open (growing) only while it is the ring's last record and
   * unpublished; every publish closes it.
   */
  struct FePacketRec {
    BlessedFeThunk  fn;
    uint32_t        size;   // bytes, header included
    uint32_t        count;  // ops
    // blessed: cb-mirror -- set directly (not through PushOp: this is the
    // packet header, not an op) when this packet's compact CbRename op
    // retired a ring block. Written only before this packet publishes,
    // read only after -- same rule as size/count. Explicitly initialized
    // in PushOpSlow (this is raw ring memory, reinterpreted, never
    // constructed, so a default member initializer here would be a lie).
    // See D3D11ThreadedContext::Map's cb-ring case and the end of Replay.
    bool            mirrorEarlyFlush;

    static size_t Replay(D3D11ImmediateContext* ctx, void* pRecord);
  };


  /**
   * \brief blessed: fe-crash-2 -- the packet the front end is replaying
   *
   * Written at the start of each packet's replay (three plain stores),
   * read only by the crash report on the same thread. \c size and
   * \c count are the values the replay itself went by; the report
   * compares them to what the packet header holds by then.
   */
  struct FeReplayProbe {
    const FePacketRec*  rec;
    uint32_t            size;
    uint32_t            count;
  };

  extern FeReplayProbe g_feReplayProbe;


  /**
   * \brief What the replay side has bound, as recorded so far
   *
   * Game side only. A setter whose arguments equal what this holds for
   * its slot is not recorded: replaying it twice in a row is a no-op in
   * dxvk, and dxvk itself never changes these bindings (its hazard
   * tracking unbinds only views). Anything that may change them outside
   * a recorded setter (ClearState, command lists, state swaps, drained
   * setters) calls \ref invalidate, which no real argument matches.
   */
  struct FeShadow {
    constexpr static UINT FullRange = 0xfffffffeu;

    struct Cb {
      ID3D11Buffer* buffer;
      UINT          first;  // FullRange: the whole buffer
      UINT          num;
    };

    struct Vb {
      ID3D11Buffer* buffer;
      UINT          stride;
      UINT          offset;
    };

    Cb                        cb[uint32_t(FeStage::Count)][D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT];
    Vb                        vb[D3D11_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT];
    void*                     shader[uint32_t(FeStage::Count)];
    ID3D11Buffer*             ib;
    UINT                      ibFormat;
    UINT                      ibOffset;
    ID3D11InputLayout*        il;
    UINT                      topology;
    ID3D11BlendState*         blend;
    FLOAT                     blendFactor[4];
    UINT                      sampleMask;
    ID3D11DepthStencilState*  ds;
    UINT                      stencilRef;
    ID3D11RasterizerState*    rs;

    // blessed: fe-getters -- the output merger, game-side, so
    // OMGetRenderTargets(AndUnorderedAccessViews) can answer without a
    // drain (docs/research/threaded-frontend.md section 5). Layout and
    // slot rules mirror D3D11ContextStateOM (d3d11_context_state.h):
    // omRtv/omDsv follow d3d11_context.cpp's SetRenderTargetsAndUnordered-
    // AccessViews, omUav follows its [minUav, maxUav) range. Kept up to
    // date only by D3D11ThreadedContext::UpdateOmShadow; every other path
    // that can change OM bindings (ClearState, SwapDeviceContextState,
    // ExecuteCommandList, a bad-argument fallback that drains) instead
    // calls InvalidateShadow(), which sets omUnknown along with the rest.
    // Raw pointers, no references: a view named here is bound in dxvk (or
    // in a record not yet replayed, ahead of its deferred final release),
    // so it is alive for as long as the shadow matches dxvk. That is why
    // UpdateOmShadow mirrors dxvk's rejections and hazard unbinds exactly.
    constexpr static UINT MaxOmRtv = D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT;
    constexpr static UINT MaxOmUav = D3D11_1_UAV_SLOT_COUNT;

    ID3D11RenderTargetView*     omRtv[MaxOmRtv];
    UINT                        omMaxRtv;    // omRtv[0..omMaxRtv) are live, the rest null
    ID3D11DepthStencilView*     omDsv;
    ID3D11UnorderedAccessView*  omUav[MaxOmUav];
    UINT                        omMinUav;    // omUav[omMinUav..omMaxUav) may be live, else null
    UINT                        omMaxUav;

    // Nonzero after invalidate() (every byte 0xff): the OM fields above may
    // not match what will be bound once replay catches up, so a getter must
    // drain instead of trusting them. UpdateOmShadow clears it once it has
    // recomputed every field above.
    uint8_t                     omUnknown;

    /// All bits set: no pointer or topology a caller passes matches, and
    /// omUnknown reads true
    void invalidate() {
      std::memset(this, 0xff, sizeof(*this));
    }
  };


  template<typename Op>
  Op* D3D11ThreadedContext::PushOp() {
    static_assert(sizeof(Op) % 8u == 0u);

    uint64_t pos = m_writePos;

    // blessed: fe-crash-2 -- m_writeLimit never lies past the ring end
    // of the last byte written (UpdateWriteLimit), so the open packet,
    // which ends at pos, cannot grow across it
    if (likely(pos == m_packetEnd && pos + sizeof(Op) <= m_writeLimit)) {
      // blessed: fe-crash -- catches a second thread extending the same
      // open packet (see CheckProducerThread); cheap (one DWORD compare
      // once cached) next to the writeLimit check already here.
      CheckProducerThread();

      // reserve the slot, but leave m_packet->size and count untouched
      // until Recorded() sees the op's fields are stored (m_packetPending);
      // a publish must never expose an op the caller has not finished
      // writing yet.
      m_writePos      = pos + sizeof(Op);
      m_packetEnd     = pos + sizeof(Op);
      m_packetPending = uint32_t(sizeof(Op));
      m_packetOps    += 1u;
      return reinterpret_cast<Op*>(m_ring + (pos & (RingSize - 1u)));
    }

    return static_cast<Op*>(PushOpSlow(sizeof(Op)));
  }


  // blessed: fe-crash -- lives here, not in blessed_threaded_context.h,
  // because m_packet->size/count need FePacketRec's complete definition
  inline void D3D11ThreadedContext::Recorded() {
    // Fold in a pending op reservation now that its fields are stored
    // (see PushOp/PushOpSlow above): every op push is followed by exactly
    // this call once its fields are written, so m_packet->size/count only
    // ever grow to include ops the front end can safely replay.
    if (unlikely(m_packetPending)) {
      m_packet->size  += m_packetPending;
      m_packet->count += 1u;
      m_packetPending  = 0u;
    }

    if (unlikely(++m_unpublished >= m_publishEvery))
      Publish();
  }

}
