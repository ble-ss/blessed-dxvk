// blessed: threaded-fe -- the facade's entry points: what each immediate-context call records, answers on the game thread, or drains
#include <atomic>
#include <cstring>
#include <string>
#include <tuple>

#include "blessed_autoinstance.h"
#include "blessed_dump.h"
#include "blessed_threaded_context.h"
#include "blessed_threaded_packet.h"
#include "blessed_threaded_fe_diag.h" // blessed: fe-getters

#include "d3d11_buffer.h"
#include "d3d11_device.h"
#include "d3d11_query.h"
#include "d3d11_texture.h"

#include "../dxvk/dxvk_util.h"
#include "../dxvk/blessed/blessed_cb_mirror.h"
#include "../util/util_env.h"

// Every entry: the census stamp (only while the probe times the frame)
// and the facade's own multithread lock (a no-op unless the app turns
// multithread protection on). The front end never takes this lock.
#define FE_ENTRY(name) \
  blessed::FeScope feScope(blessed::FeCall::name); \
  D3D10DeviceLock feLock = m_multithread.AcquireLock()

// Wait for the front end to go idle; the caller then runs dxvk's method
#define FE_DRAIN(name, reason) \
  FE_ENTRY(name); \
  Drain(blessed::FeDrain::reason, blessed::FeCall::name)

namespace dxvk {

  static constexpr size_t BlessedFeAlign(size_t Size) {
    return (Size + 7u) & ~size_t(7u);
  }


  // --- callers: a non-virtual call into the real context ---

#define FE_CALLER(Method) \
  struct FeCall_##Method { \
    template<typename... A> \
    static void Call(D3D11ImmediateContext* ctx, A... a) { \
      ctx->D3D11ImmediateContext::Method(a...); \
    } \
  };

  FE_CALLER(ClearState)
  FE_CALLER(Draw)
  FE_CALLER(DrawIndexed)
  FE_CALLER(DrawInstanced)
  FE_CALLER(DrawIndexedInstanced)
  FE_CALLER(DrawIndexedInstancedIndirect)
  FE_CALLER(DrawInstancedIndirect)
  FE_CALLER(DrawAuto)
  FE_CALLER(Dispatch)
  FE_CALLER(DispatchIndirect)
  FE_CALLER(IASetInputLayout)
  FE_CALLER(IASetPrimitiveTopology)
  FE_CALLER(IASetIndexBuffer)
  FE_CALLER(OMSetDepthStencilState)
  FE_CALLER(RSSetState)
  FE_CALLER(SetPredication)
  FE_CALLER(SetResourceMinLOD)
  FE_CALLER(CopyResource)
  FE_CALLER(CopyStructureCount)
  FE_CALLER(ClearDepthStencilView)
  FE_CALLER(GenerateMips)
  FE_CALLER(ResolveSubresource)
  FE_CALLER(ExecuteCommandList)
  FE_CALLER(Flush)
  FE_CALLER(Flush1)
  FE_CALLER(Signal)
  FE_CALLER(Wait)
  FE_CALLER(DiscardResource)
  FE_CALLER(DiscardView)
  FE_CALLER(VSSetConstantBuffers)
  FE_CALLER(HSSetConstantBuffers)
  FE_CALLER(DSSetConstantBuffers)
  FE_CALLER(GSSetConstantBuffers)
  FE_CALLER(PSSetConstantBuffers)
  FE_CALLER(CSSetConstantBuffers)
  FE_CALLER(VSSetShaderResources)
  FE_CALLER(HSSetShaderResources)
  FE_CALLER(DSSetShaderResources)
  FE_CALLER(GSSetShaderResources)
  FE_CALLER(PSSetShaderResources)
  FE_CALLER(CSSetShaderResources)
  FE_CALLER(VSSetSamplers)
  FE_CALLER(HSSetSamplers)
  FE_CALLER(DSSetSamplers)
  FE_CALLER(GSSetSamplers)
  FE_CALLER(PSSetSamplers)
  FE_CALLER(CSSetSamplers)
  FE_CALLER(VSSetConstantBuffers1)
  FE_CALLER(HSSetConstantBuffers1)
  FE_CALLER(DSSetConstantBuffers1)
  FE_CALLER(GSSetConstantBuffers1)
  FE_CALLER(PSSetConstantBuffers1)
  FE_CALLER(CSSetConstantBuffers1)

#undef FE_CALLER

  // SetShader with no class instances
#define FE_SHADER_CALLER(Stage, Iface) \
  struct FeCall_##Stage##SetShader { \
    static void Call(D3D11ImmediateContext* ctx, Iface* pShader) { \
      ctx->D3D11ImmediateContext::Stage##SetShader(pShader, nullptr, 0u); \
    } \
  };

  FE_SHADER_CALLER(VS, ID3D11VertexShader)
  FE_SHADER_CALLER(HS, ID3D11HullShader)
  FE_SHADER_CALLER(DS, ID3D11DomainShader)
  FE_SHADER_CALLER(GS, ID3D11GeometryShader)
  FE_SHADER_CALLER(PS, ID3D11PixelShader)
  FE_SHADER_CALLER(CS, ID3D11ComputeShader)

#undef FE_SHADER_CALLER

  struct FeCall_RSSetViewports {
    static void Call(D3D11ImmediateContext* ctx, UINT, UINT Count, const D3D11_VIEWPORT* pItems) {
      ctx->D3D11ImmediateContext::RSSetViewports(Count, Count ? pItems : nullptr);
    }
  };

  struct FeCall_RSSetScissorRects {
    static void Call(D3D11ImmediateContext* ctx, UINT, UINT Count, const D3D11_RECT* pItems) {
      ctx->D3D11ImmediateContext::RSSetScissorRects(Count, Count ? pItems : nullptr);
    }
  };


  // --- records ---

  struct FeFloat4 { FLOAT v[4]; };
  struct FeUint4  { UINT  v[4]; };

  /// A call with scalar arguments, stored by value
  template<typename Caller, typename... Args>
  struct FeCallRec {
    BlessedFeThunk      fn;
    std::tuple<Args...> args;

    static size_t Replay(D3D11ImmediateContext* ctx, void* pRecord) {
      auto rec = static_cast<FeCallRec*>(pRecord);

      std::apply([ctx] (Args... a) {
        Caller::Call(ctx, a...);
      }, rec->args);

      return BlessedFeAlign(sizeof(FeCallRec));
    }
  };


  /// A call with one argument, a count and an inline array
  template<typename Caller, typename T>
  struct FeArrayRec {
    BlessedFeThunk  fn;
    UINT            arg;
    UINT            count;
    T               items[1];

    static size_t Size(UINT Count) {
      return BlessedFeAlign(sizeof(FeArrayRec) + sizeof(T) * (Count ? Count - 1u : 0u));
    }

    static size_t Replay(D3D11ImmediateContext* ctx, void* pRecord) {
      auto rec = static_cast<FeArrayRec*>(pRecord);
      Caller::Call(ctx, rec->arg, rec->count, rec->count ? rec->items : nullptr);
      return Size(rec->count);
    }
  };


  /// XXSetConstantBuffers1: buffers, then the two optional ranges
  template<typename Caller>
  struct FeCb1Rec {
    BlessedFeThunk  fn;
    UINT            start;
    UINT            count;
    uint32_t        hasFirst;
    uint32_t        hasNum;

    static size_t Size(UINT Count) {
      return BlessedFeAlign(sizeof(FeCb1Rec) + Count * (sizeof(ID3D11Buffer*) + 2u * sizeof(UINT)));
    }

    ID3D11Buffer** buffers() { return reinterpret_cast<ID3D11Buffer**>(this + 1); }
    UINT* first() { return reinterpret_cast<UINT*>(buffers() + count); }
    UINT* num() { return first() + count; }

    static size_t Replay(D3D11ImmediateContext* ctx, void* pRecord) {
      auto rec = static_cast<FeCb1Rec*>(pRecord);
      UINT n = rec->count;

      Caller::Call(ctx, rec->start, n,
        n ? rec->buffers() : nullptr,
        n && rec->hasFirst ? rec->first() : nullptr,
        n && rec->hasNum ? rec->num() : nullptr);
      return Size(n);
    }
  };


  struct FeVertexBuffersRec {
    BlessedFeThunk  fn;
    UINT            start;
    UINT            count;

    static size_t Size(UINT Count) {
      return BlessedFeAlign(sizeof(FeVertexBuffersRec) + Count * (sizeof(ID3D11Buffer*) + 2u * sizeof(UINT)));
    }

    ID3D11Buffer** buffers() { return reinterpret_cast<ID3D11Buffer**>(this + 1); }
    UINT* strides() { return reinterpret_cast<UINT*>(buffers() + count); }
    UINT* offsets() { return strides() + count; }

    static size_t Replay(D3D11ImmediateContext* ctx, void* pRecord) {
      auto rec = static_cast<FeVertexBuffersRec*>(pRecord);
      UINT n = rec->count;

      ctx->D3D11ImmediateContext::IASetVertexBuffers(rec->start, n,
        n ? rec->buffers() : nullptr,
        n ? rec->strides() : nullptr,
        n ? rec->offsets() : nullptr);
      return Size(n);
    }
  };


  /// OMSetRenderTargets and OMSetRenderTargetsAndUnorderedAccessViews
  struct FeRenderTargetsRec {
    BlessedFeThunk            fn;
    ID3D11DepthStencilView*   dsv;
    UINT                      numRtvs;    // as passed, may be a KEEP value
    UINT                      rtvCount;   // entries stored
    UINT                      uavStart;
    UINT                      numUavs;    // as passed, may be a KEEP value
    UINT                      uavCount;   // entries stored
    uint32_t                  hasCounts;
    uint32_t                  withUavs;
    uint32_t                  pad;

    static size_t Size(UINT RtvCount, UINT UavCount) {
      return BlessedFeAlign(sizeof(FeRenderTargetsRec)
        + RtvCount * sizeof(ID3D11RenderTargetView*)
        + UavCount * (sizeof(ID3D11UnorderedAccessView*) + sizeof(UINT)));
    }

    ID3D11RenderTargetView** rtvs() { return reinterpret_cast<ID3D11RenderTargetView**>(this + 1); }
    ID3D11UnorderedAccessView** uavs() { return reinterpret_cast<ID3D11UnorderedAccessView**>(rtvs() + rtvCount); }
    UINT* counts() { return reinterpret_cast<UINT*>(uavs() + uavCount); }

    static size_t Replay(D3D11ImmediateContext* ctx, void* pRecord) {
      auto rec = static_cast<FeRenderTargetsRec*>(pRecord);

      if (rec->withUavs) {
        ctx->D3D11ImmediateContext::OMSetRenderTargetsAndUnorderedAccessViews(
          rec->numRtvs, rec->rtvCount ? rec->rtvs() : nullptr, rec->dsv,
          rec->uavStart, rec->numUavs, rec->uavCount ? rec->uavs() : nullptr,
          rec->uavCount && rec->hasCounts ? rec->counts() : nullptr);
      } else {
        ctx->D3D11ImmediateContext::OMSetRenderTargets(
          rec->numRtvs, rec->rtvCount ? rec->rtvs() : nullptr, rec->dsv);
      }

      return Size(rec->rtvCount, rec->uavCount);
    }
  };


  struct FeComputeUavsRec {
    BlessedFeThunk  fn;
    UINT            start;
    UINT            count;
    uint32_t        hasCounts;
    uint32_t        pad;

    static size_t Size(UINT Count) {
      return BlessedFeAlign(sizeof(FeComputeUavsRec) + Count * (sizeof(ID3D11UnorderedAccessView*) + sizeof(UINT)));
    }

    ID3D11UnorderedAccessView** uavs() { return reinterpret_cast<ID3D11UnorderedAccessView**>(this + 1); }
    UINT* counts() { return reinterpret_cast<UINT*>(uavs() + count); }

    static size_t Replay(D3D11ImmediateContext* ctx, void* pRecord) {
      auto rec = static_cast<FeComputeUavsRec*>(pRecord);
      UINT n = rec->count;

      ctx->D3D11ImmediateContext::CSSetUnorderedAccessViews(rec->start, n,
        n ? rec->uavs() : nullptr, n && rec->hasCounts ? rec->counts() : nullptr);
      return Size(n);
    }
  };


  struct FeSoTargetsRec {
    BlessedFeThunk  fn;
    UINT            count;
    uint32_t        hasOffsets;

    static size_t Size(UINT Count) {
      return BlessedFeAlign(sizeof(FeSoTargetsRec) + Count * (sizeof(ID3D11Buffer*) + sizeof(UINT)));
    }

    ID3D11Buffer** buffers() { return reinterpret_cast<ID3D11Buffer**>(this + 1); }
    UINT* offsets() { return reinterpret_cast<UINT*>(buffers() + count); }

    static size_t Replay(D3D11ImmediateContext* ctx, void* pRecord) {
      auto rec = static_cast<FeSoTargetsRec*>(pRecord);
      UINT n = rec->count;

      ctx->D3D11ImmediateContext::SOSetTargets(n, n ? rec->buffers() : nullptr,
        n && rec->hasOffsets ? rec->offsets() : nullptr);
      return Size(n);
    }
  };


  struct FeClearViewRec {
    BlessedFeThunk  fn;
    ID3D11View*     view;
    FLOAT           color[4];
    UINT            count;
    uint32_t        pad;

    static size_t Size(UINT Count) {
      return BlessedFeAlign(sizeof(FeClearViewRec) + Count * sizeof(D3D11_RECT));
    }

    D3D11_RECT* rects() { return reinterpret_cast<D3D11_RECT*>(this + 1); }

    static size_t Replay(D3D11ImmediateContext* ctx, void* pRecord) {
      auto rec = static_cast<FeClearViewRec*>(pRecord);

      ctx->D3D11ImmediateContext::ClearView(rec->view, rec->color,
        rec->count ? rec->rects() : nullptr, rec->count);
      return Size(rec->count);
    }
  };


  struct FeCopyRegionRec {
    BlessedFeThunk    fn;
    ID3D11Resource*   dst;
    ID3D11Resource*   src;
    UINT              dstSub;
    UINT              dstX, dstY, dstZ;
    UINT              srcSub;
    UINT              copyFlags;
    D3D11_BOX         box;
    uint32_t          hasBox;
    uint32_t          version1;

    static size_t Replay(D3D11ImmediateContext* ctx, void* pRecord) {
      auto rec = static_cast<FeCopyRegionRec*>(pRecord);
      const D3D11_BOX* box = rec->hasBox ? &rec->box : nullptr;

      if (rec->version1) {
        ctx->D3D11ImmediateContext::CopySubresourceRegion1(rec->dst, rec->dstSub,
          rec->dstX, rec->dstY, rec->dstZ, rec->src, rec->srcSub, box, rec->copyFlags);
      } else {
        ctx->D3D11ImmediateContext::CopySubresourceRegion(rec->dst, rec->dstSub,
          rec->dstX, rec->dstY, rec->dstZ, rec->src, rec->srcSub, box);
      }

      return BlessedFeAlign(sizeof(FeCopyRegionRec));
    }
  };


  /// UpdateSubresource(1) on anything the game side does not answer.
  /// The source bytes follow inline, or live in a side block above
  /// MaxInlineData that the replay frees.
  struct FeUpdateRec {
    BlessedFeThunk    fn;
    ID3D11Resource*   dst;
    void*             side;
    size_t            size;
    UINT              dstSub;
    UINT              rowPitch;
    UINT              depthPitch;
    UINT              copyFlags;
    D3D11_BOX         box;
    uint32_t          hasBox;
    uint32_t          version1;

    static size_t Size(size_t Inline) {
      return BlessedFeAlign(sizeof(FeUpdateRec) + Inline);
    }

    static size_t Replay(D3D11ImmediateContext* ctx, void* pRecord) {
      auto rec = static_cast<FeUpdateRec*>(pRecord);
      const void* data = rec->side ? rec->side : static_cast<const void*>(rec + 1);
      const D3D11_BOX* box = rec->hasBox ? &rec->box : nullptr;

      if (rec->version1) {
        ctx->D3D11ImmediateContext::UpdateSubresource1(rec->dst, rec->dstSub, box,
          data, rec->rowPitch, rec->depthPitch, rec->copyFlags);
      } else {
        ctx->D3D11ImmediateContext::UpdateSubresource(rec->dst, rec->dstSub, box,
          data, rec->rowPitch, rec->depthPitch);
      }

      size_t size = Size(rec->side ? 0u : rec->size);
      std::free(rec->side);
      return size;
    }
  };


  /// Replay half of a game-side Map(WRITE_DISCARD)
  struct FeRenameRec {
    BlessedFeThunk    fn;
    D3D11Buffer*      buffer;
    alignas(Rc<DxvkResourceAllocation>)
    unsigned char     allocation[sizeof(Rc<DxvkResourceAllocation>)];
    VkDeviceSize      size;

    static size_t Replay(D3D11ImmediateContext* ctx, void* pRecord) {
      auto rec = static_cast<FeRenameRec*>(pRecord);
      auto stored = reinterpret_cast<Rc<DxvkResourceAllocation>*>(rec->allocation);

      Rc<DxvkResourceAllocation> slice = std::move(*stored);
      stored->~Rc<DxvkResourceAllocation>();

      rec->buffer->BlessedReplayRename(slice);

      D3D11ThreadedContext::ReplayEmitCs(ctx, [
        cBuffer      = rec->buffer->GetBuffer(),
        cBufferSlice = std::move(slice)
      ] (DxvkContext* c) mutable {
        c->invalidateBuffer(cBuffer, std::move(cBufferSlice));
      });

      if (unlikely(rec->size > DxvkPageAllocator::PageSize))
        D3D11ThreadedContext::ReplayThrottleDiscard(ctx, rec->size);

      return BlessedFeAlign(sizeof(FeRenameRec));
    }
  };


  /// Replay half of a cb-ring map
  struct FeCbRenameRec {
    BlessedFeThunk      fn;
    D3D11Buffer*        buffer;
    BlessedCbRingChunk  chunk;

    static size_t Replay(D3D11ImmediateContext* ctx, void* pRecord) {
      auto rec = static_cast<FeCbRenameRec*>(pRecord);

      rec->buffer->BlessedSetMapPtr(rec->chunk.mapPtr);
      D3D11ThreadedContext::ReplayCbRename(ctx, rec->buffer, rec->chunk);

      // blessed: cb-mirror -- this chunk already carries blockRetired
      // (a plain struct copy), unlike the compact op; see FeMirrorEarlyFlushRec
      if (unlikely(rec->chunk.blockRetired))
        D3D11ThreadedContext::ReplayCbMirrorEarlyFlush(ctx);

      return BlessedFeAlign(sizeof(FeCbRenameRec));
    }
  };


  /// gpu half of Begin
  struct FeQueryBeginRec {
    BlessedFeThunk  fn;
    D3D11Query*     query;

    static size_t Replay(D3D11ImmediateContext* ctx, void* pRecord) {
      auto rec = static_cast<FeQueryBeginRec*>(pRecord);

      BlessedAutoInstance::OnBarrier(); // blessed: auto-instancing census, a query splits a segment

      D3D11ThreadedContext::ReplayEmitCs(ctx, [cQuery = Com<D3D11Query, false>(rec->query)]
      (DxvkContext* c) {
        cQuery->Begin(c);
      });

      return BlessedFeAlign(sizeof(FeQueryBeginRec));
    }
  };


  /// gpu half of End, with the flush the stall heuristic chose game-side
  struct FeQueryEndRec {
    BlessedFeThunk  fn;
    D3D11Query*     query;
    uint32_t        needsBegin;
    uint32_t        flush; // 0 none, 1 execute, 2 consider

    static size_t Replay(D3D11ImmediateContext* ctx, void* pRecord) {
      auto rec = static_cast<FeQueryEndRec*>(pRecord);

      BlessedAutoInstance::OnBarrier(); // blessed: auto-instancing census, a query splits a segment

      if (unlikely(rec->needsBegin)) {
        D3D11ThreadedContext::ReplayEmitCs(ctx, [cQuery = Com<D3D11Query, false>(rec->query)]
        (DxvkContext* c) {
          cQuery->Begin(c);
        });
      }

      D3D11ThreadedContext::ReplayEmitCs(ctx, [cQuery = Com<D3D11Query, false>(rec->query)]
      (DxvkContext* c) {
        cQuery->End(c);
      });

      if (rec->flush == 1u)
        D3D11ThreadedContext::ReplayExecuteFlush(ctx, GpuFlushType::ImplicitSynchronization);
      else if (rec->flush == 2u)
        D3D11ThreadedContext::ReplayConsiderFlush(ctx, GpuFlushType::ImplicitStrongHint);

      return BlessedFeAlign(sizeof(FeQueryEndRec));
    }
  };


  /// GetData found the query pending: keep the gpu busy
  struct FeFlushHintRec {
    BlessedFeThunk  fn;

    static size_t Replay(D3D11ImmediateContext* ctx, void*) {
      D3D11ThreadedContext::ReplayConsiderFlush(ctx, GpuFlushType::ImplicitSynchronization);
      return BlessedFeAlign(sizeof(FeFlushHintRec));
    }
  };


  /// Annotation event or marker, label copied
  struct FeAnnotationRec {
    BlessedFeThunk  fn;
    D3DCOLOR        color;
    uint32_t        kind;   // 0 begin, 1 end, 2 marker
    uint32_t        length; // wchar_t count, terminator included
    uint32_t        pad;

    static size_t Size(uint32_t Length) {
      return BlessedFeAlign(sizeof(FeAnnotationRec) + Length * sizeof(WCHAR));
    }

    static size_t Replay(D3D11ImmediateContext* ctx, void* pRecord) {
      auto rec = static_cast<FeAnnotationRec*>(pRecord);
      auto label = reinterpret_cast<const WCHAR*>(rec + 1);

      if (rec->kind == 0u)
        D3D11ThreadedContext::ReplayAnnotation(ctx)->BeginEvent(rec->color, label);
      else if (rec->kind == 1u)
        D3D11ThreadedContext::ReplayAnnotation(ctx)->EndEvent();
      else
        D3D11ThreadedContext::ReplayAnnotation(ctx)->SetMarker(rec->color, label);

      return Size(rec->length);
    }
  };


  // --- stage 3: draw packets ---

  static void FeReplaySetCb(D3D11ImmediateContext* ctx, FeStage Stage, UINT Slot, ID3D11Buffer* pBuffer) {
    switch (Stage) {
      case FeStage::VS: ctx->D3D11ImmediateContext::VSSetConstantBuffers(Slot, 1u, &pBuffer); break;
      case FeStage::HS: ctx->D3D11ImmediateContext::HSSetConstantBuffers(Slot, 1u, &pBuffer); break;
      case FeStage::DS: ctx->D3D11ImmediateContext::DSSetConstantBuffers(Slot, 1u, &pBuffer); break;
      case FeStage::GS: ctx->D3D11ImmediateContext::GSSetConstantBuffers(Slot, 1u, &pBuffer); break;
      case FeStage::PS: ctx->D3D11ImmediateContext::PSSetConstantBuffers(Slot, 1u, &pBuffer); break;
      case FeStage::CS: ctx->D3D11ImmediateContext::CSSetConstantBuffers(Slot, 1u, &pBuffer); break;
      default: break;
    }
  }


  static void FeReplaySetSrv(D3D11ImmediateContext* ctx, FeStage Stage, UINT Slot, ID3D11ShaderResourceView* pView) {
    switch (Stage) {
      case FeStage::VS: ctx->D3D11ImmediateContext::VSSetShaderResources(Slot, 1u, &pView); break;
      case FeStage::HS: ctx->D3D11ImmediateContext::HSSetShaderResources(Slot, 1u, &pView); break;
      case FeStage::DS: ctx->D3D11ImmediateContext::DSSetShaderResources(Slot, 1u, &pView); break;
      case FeStage::GS: ctx->D3D11ImmediateContext::GSSetShaderResources(Slot, 1u, &pView); break;
      case FeStage::PS: ctx->D3D11ImmediateContext::PSSetShaderResources(Slot, 1u, &pView); break;
      case FeStage::CS: ctx->D3D11ImmediateContext::CSSetShaderResources(Slot, 1u, &pView); break;
      default: break;
    }
  }


  static void FeReplaySetSampler(D3D11ImmediateContext* ctx, FeStage Stage, UINT Slot, ID3D11SamplerState* pSampler) {
    switch (Stage) {
      case FeStage::VS: ctx->D3D11ImmediateContext::VSSetSamplers(Slot, 1u, &pSampler); break;
      case FeStage::HS: ctx->D3D11ImmediateContext::HSSetSamplers(Slot, 1u, &pSampler); break;
      case FeStage::DS: ctx->D3D11ImmediateContext::DSSetSamplers(Slot, 1u, &pSampler); break;
      case FeStage::GS: ctx->D3D11ImmediateContext::GSSetSamplers(Slot, 1u, &pSampler); break;
      case FeStage::PS: ctx->D3D11ImmediateContext::PSSetSamplers(Slot, 1u, &pSampler); break;
      case FeStage::CS: ctx->D3D11ImmediateContext::CSSetSamplers(Slot, 1u, &pSampler); break;
      default: break;
    }
  }


  static void FeReplaySetShader(D3D11ImmediateContext* ctx, FeStage Stage, void* pShader) {
    switch (Stage) {
      case FeStage::VS: ctx->D3D11ImmediateContext::VSSetShader(static_cast<ID3D11VertexShader*>(pShader), nullptr, 0u); break;
      case FeStage::HS: ctx->D3D11ImmediateContext::HSSetShader(static_cast<ID3D11HullShader*>(pShader), nullptr, 0u); break;
      case FeStage::DS: ctx->D3D11ImmediateContext::DSSetShader(static_cast<ID3D11DomainShader*>(pShader), nullptr, 0u); break;
      case FeStage::GS: ctx->D3D11ImmediateContext::GSSetShader(static_cast<ID3D11GeometryShader*>(pShader), nullptr, 0u); break;
      case FeStage::PS: ctx->D3D11ImmediateContext::PSSetShader(static_cast<ID3D11PixelShader*>(pShader), nullptr, 0u); break;
      case FeStage::CS: ctx->D3D11ImmediateContext::CSSetShader(static_cast<ID3D11ComputeShader*>(pShader), nullptr, 0u); break;
      default: break;
    }
  }


  // blessed: fe-crash -- BLESSED_FE_CHECK=1 validates every packet op
  // before it runs: a known op code, the pointers CbRename always sets,
  // that the op stays inside the packet's declared end, and that the
  // switch below advances by exactly what this table says. One cached
  // bool when off, so the no-check path stays free.
  static bool BlessedFeCheckEnabled() {
    static const bool s_enabled = env::getEnvVar("BLESSED_FE_CHECK") == "1";
    return s_enabled;
  }

  // The authoritative op -> size table, independent of the switch in
  // Replay() below: a push/replay size desync (the wrong FeOpXxx type at
  // either end) shows up as a mismatch against this instead of silently
  // decoding the next op's bytes as part of this one's payload.
  static uint32_t BlessedFeOpSize(FeOp op) {
    switch (op) {
      case FeOp::CbRename:             return uint32_t(sizeof(FeOpCbRename));
      case FeOp::SetCb:
      case FeOp::SetSrv:
      case FeOp::SetSampler:
      case FeOp::SetShader:
      case FeOp::SetInputLayout:       return uint32_t(sizeof(FeOpPtr));
      case FeOp::SetVb:                return uint32_t(sizeof(FeOpVb));
      case FeOp::SetIb:                return uint32_t(sizeof(FeOpIb));
      case FeOp::SetTopology:          return uint32_t(sizeof(FeOpTopology));
      case FeOp::Draw:
      case FeOp::DrawIndexed:          return uint32_t(sizeof(FeOpDraw));
      case FeOp::DrawInstanced:
      case FeOp::DrawIndexedInstanced: return uint32_t(sizeof(FeOpDrawInstanced));
    }

    return 0u;
  }

  // The last few ops replayed before a check fails, for the log line: a
  // desync usually reads as garbage several ops after where it started.
  struct BlessedFeOpHistory {
    struct Entry { FeOp op; uint32_t offset; };

    Entry     entries[3] = {};
    uint32_t  count      = 0u;

    void push(FeOp op, uint32_t offset) {
      if (count < 3u) {
        entries[count++] = { op, offset };
      } else {
        entries[0] = entries[1];
        entries[1] = entries[2];
        entries[2] = { op, offset };
      }
    }

    std::string format() const {
      std::string out;

      for (uint32_t i = 0; i < count; i++)
        out += str::format(i ? ", op " : "op ", uint32_t(entries[i].op), "@", entries[i].offset);

      return out;
    }
  };

  static void BlessedFeCheckFail(const FePacketRec* rec, size_t offset, const char* what,
      const BlessedFeOpHistory& history) {
    static std::atomic<bool> s_logged = { false };

    if (!s_logged.exchange(true, std::memory_order_relaxed)) {
      Logger::err(str::format("d3d11.blessedFeCheck: bad op at packet offset ", offset,
        " (packet size ", rec->size, ", count ", rec->count, "): ", what,
        "; preceding ops: ", history.count ? history.format() : std::string("(none, first op in packet)")));
    }

    throw DxvkError(str::format("d3d11.blessedFeCheck: ", what));
  }

  // Checked before the switch runs: the op code, CbRename's pointers,
  // and that its declared size does not run past the packet.
  static void BlessedFeCheckOp(const FePacketRec* rec, const char* ptr, const char* end,
      const FeOpHeader* h, const BlessedFeOpHistory& history) {
    size_t offset = size_t(ptr - reinterpret_cast<const char*>(rec));

    if (uint32_t(h->op) > uint32_t(FeOp::DrawIndexedInstanced))
      BlessedFeCheckFail(rec, offset, "unknown op code", history);

    uint32_t size = BlessedFeOpSize(h->op);

    if (ptr + size > end)
      BlessedFeCheckFail(rec, offset, "op's declared size runs past the packet's end", history);

    if (h->op == FeOp::CbRename) {
      auto op = reinterpret_cast<const FeOpCbRename*>(ptr);

      if (!op->block)
        BlessedFeCheckFail(rec, offset, "CbRename with a null block", history);
      else if (!op->buffer)
        BlessedFeCheckFail(rec, offset, "CbRename with a null buffer", history);
    }
  }

  // Checked after the switch has advanced ptr: the size table above
  // must agree with what the switch actually did, or the next op is
  // read from the wrong offset even though this one looked fine.
  static void BlessedFeCheckAdvance(const FePacketRec* rec, const char* opStart, const char* ptr,
      const FeOpHeader* h, const BlessedFeOpHistory& history) {
    uint32_t advanced = uint32_t(ptr - opStart);
    uint32_t expected = BlessedFeOpSize(h->op);

    if (advanced != expected) {
      size_t offset = size_t(opStart - reinterpret_cast<const char*>(rec));

      BlessedFeCheckFail(rec, offset, str::format("replay advanced ", advanced,
        " bytes for op ", uint32_t(h->op), ", the size table says ", expected).c_str(), history);
    }
  }

  // Checked once Replay's loop exits: the last op must land exactly on
  // rec->size, or the packet is under/over-declared relative to what
  // was actually walked.
  static void BlessedFeCheckPacketEnd(const FePacketRec* rec, const char* ptr, const char* end,
      const BlessedFeOpHistory& history) {
    if (ptr == end)
      return;

    static std::atomic<bool> s_logged = { false };
    ptrdiff_t drift = ptr - end;

    if (!s_logged.exchange(true, std::memory_order_relaxed)) {
      Logger::err(str::format("d3d11.blessedFeCheck: packet ended ", drift,
        " bytes off its declared size ", rec->size, " (count ", rec->count,
        "); preceding ops: ", history.count ? history.format() : std::string("(none)")));
    }
  }


  size_t FePacketRec::Replay(D3D11ImmediateContext* ctx, void* pRecord) {
    auto rec = static_cast<FePacketRec*>(pRecord);

    // blessed: fe-crash-2 -- size is read once: the walk below and the
    // advance returned at the end go by the same value, so a header that
    // changed under us (it must not) cannot also misplace the next record
    uint32_t size = rec->size;
    g_feReplayProbe.rec   = rec;
    g_feReplayProbe.size  = size;
    g_feReplayProbe.count = rec->count;

    auto ptr = reinterpret_cast<const char*>(rec + 1);
    auto end = reinterpret_cast<const char*>(rec) + size;

    BlessedFeOpHistory history;
    bool checking = BlessedFeCheckEnabled();

    while (ptr < end) {
      auto h = reinterpret_cast<const FeOpHeader*>(ptr);
      auto opStart = ptr;

      if (unlikely(checking))
        BlessedFeCheckOp(rec, ptr, end, h, history);

      switch (h->op) {
        case FeOp::CbRename: {
          auto op = reinterpret_cast<const FeOpCbRename*>(ptr);

          BlessedCbRingChunk chunk;
          chunk.block  = op->block;
          chunk.offset = op->h.arg;
          chunk.mapPtr = static_cast<char*>(op->block->mapPtr()) + op->h.arg;

          // blessed: cb-mirror -- this compact op doesn't carry the mirror
          // (see the comment on FeOpCbRename, blessed_threaded_packet.h);
          // resolve it from the ring's side table instead, off the game
          // thread that owns it (findMirror is lock-free). Free when
          // mirroring is off: mirrorEnabled() is a plain cached bool.
          if (D3D11ThreadedContext::ReplayCbMirrorEnabled(ctx)) {
            chunk.mirror = D3D11ThreadedContext::ReplayFindCbMirror(ctx, op->block);

            blessed::g_cbMirrorStats.compact.fetch_add(1u, std::memory_order_relaxed);
            if (!chunk.mirror)
              blessed::g_cbMirrorStats.compactMiss.fetch_add(1u, std::memory_order_relaxed);
          }

          op->buffer->BlessedSetMapPtr(chunk.mapPtr);
          D3D11ThreadedContext::ReplayCbRename(ctx, op->buffer, chunk);
          ptr += sizeof(*op);
        } break;

        case FeOp::SetCb: {
          auto op = reinterpret_cast<const FeOpPtr*>(ptr);
          FeReplaySetCb(ctx, op->h.stage, op->h.slot, static_cast<ID3D11Buffer*>(op->ptr));
          ptr += sizeof(*op);
        } break;

        case FeOp::SetSrv: {
          auto op = reinterpret_cast<const FeOpPtr*>(ptr);
          FeReplaySetSrv(ctx, op->h.stage, op->h.slot, static_cast<ID3D11ShaderResourceView*>(op->ptr));
          ptr += sizeof(*op);
        } break;

        case FeOp::SetSampler: {
          auto op = reinterpret_cast<const FeOpPtr*>(ptr);
          FeReplaySetSampler(ctx, op->h.stage, op->h.slot, static_cast<ID3D11SamplerState*>(op->ptr));
          ptr += sizeof(*op);
        } break;

        case FeOp::SetShader: {
          auto op = reinterpret_cast<const FeOpPtr*>(ptr);
          FeReplaySetShader(ctx, op->h.stage, op->ptr);
          ptr += sizeof(*op);
        } break;

        case FeOp::SetVb: {
          auto op = reinterpret_cast<const FeOpVb*>(ptr);
          ID3D11Buffer* buffer = op->buffer;
          UINT stride = op->h.arg;
          UINT offset = op->offset;
          ctx->D3D11ImmediateContext::IASetVertexBuffers(op->h.slot, 1u, &buffer, &stride, &offset);
          ptr += sizeof(*op);
        } break;

        case FeOp::SetIb: {
          auto op = reinterpret_cast<const FeOpIb*>(ptr);
          ctx->D3D11ImmediateContext::IASetIndexBuffer(op->buffer, DXGI_FORMAT(op->h.slot), op->h.arg);
          ptr += sizeof(*op);
        } break;

        case FeOp::SetInputLayout: {
          auto op = reinterpret_cast<const FeOpPtr*>(ptr);
          ctx->D3D11ImmediateContext::IASetInputLayout(static_cast<ID3D11InputLayout*>(op->ptr));
          ptr += sizeof(*op);
        } break;

        case FeOp::SetTopology: {
          auto op = reinterpret_cast<const FeOpTopology*>(ptr);
          ctx->D3D11ImmediateContext::IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY(op->h.arg));
          ptr += sizeof(*op);
        } break;

        case FeOp::Draw: {
          auto op = reinterpret_cast<const FeOpDraw*>(ptr);
          ctx->D3D11ImmediateContext::Draw(op->h.arg, op->start);
          ptr += sizeof(*op);
        } break;

        case FeOp::DrawIndexed: {
          auto op = reinterpret_cast<const FeOpDraw*>(ptr);
          ctx->D3D11ImmediateContext::DrawIndexed(op->h.arg, op->start, op->base);
          ptr += sizeof(*op);
        } break;

        case FeOp::DrawInstanced: {
          auto op = reinterpret_cast<const FeOpDrawInstanced*>(ptr);
          ctx->D3D11ImmediateContext::DrawInstanced(op->h.arg, op->instances, op->start, op->startInstance);
          ptr += sizeof(*op);
        } break;

        case FeOp::DrawIndexedInstanced: {
          auto op = reinterpret_cast<const FeOpDrawInstanced*>(ptr);
          ctx->D3D11ImmediateContext::DrawIndexedInstanced(op->h.arg, op->instances, op->start, op->base, op->startInstance);
          ptr += sizeof(*op);
        } break;

        default:
          throw DxvkError(str::format("d3d11.blessedFrontEndCompact: unknown op ", uint32_t(h->op)));
      }

      if (unlikely(checking)) {
        BlessedFeCheckAdvance(rec, opStart, ptr, h, history);
        history.push(h->op, uint32_t(opStart - reinterpret_cast<const char*>(rec)));
      }
    }

    if (unlikely(checking))
      BlessedFeCheckPacketEnd(rec, ptr, end, history);

    // blessed: cb-mirror -- this packet's compact CbRename retired a ring
    // block; its dirty range is final, so flush it now instead of at this
    // chunk's natural end. See D3D11ThreadedContext::Map's cb-ring case.
    if (unlikely(rec->mirrorEarlyFlush))
      D3D11ThreadedContext::ReplayCbMirrorEarlyFlush(ctx);

    return size;
  }


  // --- record helpers ---

  template<typename Caller, typename... Args>
  void D3D11ThreadedContext::RecordCall(Args... args) {
    using Rec = FeCallRec<Caller, Args...>;

    Rec* rec = Push<Rec>();
    new (&rec->args) std::tuple<Args...>(args...);
    Recorded();
  }


  template<typename Caller, typename T>
  void D3D11ThreadedContext::RecordArray(
          UINT                        Arg,
          UINT                        Count,
    const T*                          pItems) {
    using Rec = FeArrayRec<Caller, T>;

    auto rec = static_cast<Rec*>(AllocRecord(Rec::Size(Count)));
    rec->fn    = &Rec::Replay;
    rec->arg   = Arg;
    rec->count = Count;

    for (UINT i = 0; i < Count; i++)
      rec->items[i] = pItems[i];

    Recorded();
  }


  void D3D11ThreadedContext::RecordShader(
          FeStage                     Stage,
          void*                       pShader) {
    void*& bound = m_shadow->shader[uint32_t(Stage)];

    if (bound == pShader) {
      m_folded += 1u;
      return;
    }

    bound = pShader;

    auto op = PushOp<FeOpPtr>();
    op->h   = { FeOp::SetShader, Stage, 0u, 0u };
    op->ptr = pShader;
    Recorded();
  }


  void D3D11ThreadedContext::RecordSlotOp(
          FeOp                        Op,
          FeStage                     Stage,
          UINT                        Slot,
          void*                       pObject) {
    auto op = PushOp<FeOpPtr>();
    op->h   = { Op, Stage, uint16_t(std::min(Slot, 0xffffu)), 0u };
    op->ptr = pObject;
    Recorded();
  }


  bool D3D11ThreadedContext::RecordConstantBuffersCompact(
          FeStage                     Stage,
          UINT                        StartSlot,
          UINT                        NumBuffers,
          ID3D11Buffer* const*        ppConstantBuffers,
    const UINT*                       pFirstConstant,
    const UINT*                       pNumConstants) {
    // Returns true when the call is fully handled (folded, or recorded as
    // a packet op); false leaves the recording to the caller, with the
    // shadow already updated
    if (unlikely(StartSlot + NumBuffers > D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT)) {
      InvalidateShadow();
      return false;
    }

    // Without both ranges dxvk binds the whole buffer, as the plain call does
    bool ranges = pFirstConstant && pNumConstants;
    FeShadow::Cb* slots = &m_shadow->cb[uint32_t(Stage)][StartSlot];
    bool same = true;

    for (UINT i = 0; i < NumBuffers; i++) {
      UINT first = ranges ? pFirstConstant[i] : FeShadow::FullRange;
      UINT num   = ranges ? pNumConstants[i]  : FeShadow::FullRange;

      // dxvk skips a slot with a count above the limit: no change there
      if (ranges && num > D3D11_REQ_CONSTANT_BUFFER_ELEMENT_COUNT)
        continue;

      same &= slots[i].buffer == ppConstantBuffers[i]
           && slots[i].first  == first
           && slots[i].num    == num;
    }

    if (same) {
      m_folded += 1u;
      return true;
    }

    for (UINT i = 0; i < NumBuffers; i++) {
      UINT first = ranges ? pFirstConstant[i] : FeShadow::FullRange;
      UINT num   = ranges ? pNumConstants[i]  : FeShadow::FullRange;

      if (!ranges || num <= D3D11_REQ_CONSTANT_BUFFER_ELEMENT_COUNT)
        slots[i] = { ppConstantBuffers[i], first, num };
    }

    // One slot, whole buffer: the per-draw case, a packet op
    if (NumBuffers == 1u && !ranges) {
      auto op = PushOp<FeOpPtr>();
      op->h   = { FeOp::SetCb, Stage, uint16_t(StartSlot), 0u };
      op->ptr = ppConstantBuffers[0];
      Recorded();
      return true;
    }

    return false;
  }


  template<typename Caller>
  void D3D11ThreadedContext::RecordConstantBuffers1(
          UINT                        StartSlot,
          UINT                        NumBuffers,
          ID3D11Buffer* const*        ppConstantBuffers,
    const UINT*                       pFirstConstant,
    const UINT*                       pNumConstants) {
    using Rec = FeCb1Rec<Caller>;

    auto rec = static_cast<Rec*>(AllocRecord(Rec::Size(NumBuffers)));
    rec->fn       = &Rec::Replay;
    rec->start    = StartSlot;
    rec->count    = NumBuffers;
    rec->hasFirst = pFirstConstant != nullptr;
    rec->hasNum   = pNumConstants != nullptr;

    for (UINT i = 0; i < NumBuffers; i++) {
      rec->buffers()[i] = ppConstantBuffers[i];
      rec->first()[i]   = pFirstConstant ? pFirstConstant[i] : 0u;
      rec->num()[i]     = pNumConstants ? pNumConstants[i] : 0u;
    }

    Recorded();
  }

}


namespace dxvk {

  // --- ID3D11DeviceContext: basics ---

  D3D11_DEVICE_CONTEXT_TYPE STDMETHODCALLTYPE D3D11ThreadedContext::GetType() {
    blessed::FeScope feScope(blessed::FeCall::GetType);
    return D3D11_DEVICE_CONTEXT_IMMEDIATE;
  }


  UINT STDMETHODCALLTYPE D3D11ThreadedContext::GetContextFlags() {
    blessed::FeScope feScope(blessed::FeCall::GetContextFlags);
    return m_ctx->GetContextFlags();
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::ClearState() {
    FE_ENTRY(ClearState);
    RecordCall<FeCall_ClearState>();
    InvalidateShadow();
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::DiscardResource(ID3D11Resource* pResource) {
    FE_ENTRY(DiscardResource);

    if (!pResource)
      return;

    D3D11_RESOURCE_DIMENSION resType = D3D11_RESOURCE_DIMENSION_UNKNOWN;
    pResource->GetType(&resType);

    if (resType == D3D11_RESOURCE_DIMENSION_BUFFER) {
      // dxvk discards a mappable buffer as Map(WRITE_DISCARD) + Unmap,
      // so it takes the same game-side path
      auto buffer = static_cast<D3D11Buffer*>(pResource);

      if (buffer->GetMapMode() != D3D11_COMMON_BUFFER_MAP_MODE_NONE) {
        D3D11_MAPPED_SUBRESOURCE sr;
        MapBuffer(buffer, D3D11_MAP_WRITE_DISCARD, 0u, &sr);
      }
    } else if (GetCommonTexture(pResource)->GetMapMode() == D3D11_COMMON_TEXTURE_MAP_MODE_NONE) {
      RecordCall<FeCall_DiscardResource>(pResource);
    } else {
      // maps the texture: that runs game-side, after a drain
      Drain(blessed::FeDrain::Discard, blessed::FeCall::DiscardResource);
      m_ctx->DiscardResource(pResource);
    }
  }


  bool D3D11ThreadedContext::CanRecordView(ID3D11View* pView) {
    // DiscardView maps the view's texture when it is mappable; a buffer
    // view is a no-op in dxvk, and so is recording it
    Com<ID3D11Resource> resource;
    pView->GetResource(&resource);

    D3D11_RESOURCE_DIMENSION resType = D3D11_RESOURCE_DIMENSION_UNKNOWN;
    resource->GetType(&resType);

    return resType == D3D11_RESOURCE_DIMENSION_BUFFER
        || GetCommonTexture(resource.ptr())->GetMapMode() == D3D11_COMMON_TEXTURE_MAP_MODE_NONE;
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::DiscardView(ID3D11View* pResourceView) {
    FE_ENTRY(DiscardView);

    if (!pResourceView)
      return;

    if (CanRecordView(pResourceView)) {
      RecordCall<FeCall_DiscardView>(pResourceView);
    } else {
      Drain(blessed::FeDrain::Discard, blessed::FeCall::DiscardView);
      m_ctx->DiscardView(pResourceView);
    }
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::DiscardView1(
          ID3D11View*                      pResourceView,
    const D3D11_RECT*                      pRects,
          UINT                             NumRects) {
    FE_ENTRY(DiscardView1);

    // dxvk does not discard rectangles: a no-op, as upstream
    if (!pResourceView || (NumRects && pRects))
      return;

    if (CanRecordView(pResourceView)) {
      RecordCall<FeCall_DiscardView>(pResourceView);
    } else {
      Drain(blessed::FeDrain::Discard, blessed::FeCall::DiscardView1);
      m_ctx->DiscardView1(pResourceView, pRects, NumRects);
    }
  }


  // --- copies, clears, updates ---

  void STDMETHODCALLTYPE D3D11ThreadedContext::CopySubresourceRegion(
          ID3D11Resource*                   pDstResource,
          UINT                              DstSubresource,
          UINT                              DstX,
          UINT                              DstY,
          UINT                              DstZ,
          ID3D11Resource*                   pSrcResource,
          UINT                              SrcSubresource,
    const D3D11_BOX*                        pSrcBox) {
    FE_ENTRY(CopySubresourceRegion);

    auto rec = Push<FeCopyRegionRec>();
    rec->dst       = pDstResource;
    rec->src       = pSrcResource;
    rec->dstSub    = DstSubresource;
    rec->dstX      = DstX;
    rec->dstY      = DstY;
    rec->dstZ      = DstZ;
    rec->srcSub    = SrcSubresource;
    rec->copyFlags = 0u;
    rec->box       = pSrcBox ? *pSrcBox : D3D11_BOX();
    rec->hasBox    = pSrcBox != nullptr;
    rec->version1  = 0u;
    Recorded();
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::CopySubresourceRegion1(
          ID3D11Resource*                   pDstResource,
          UINT                              DstSubresource,
          UINT                              DstX,
          UINT                              DstY,
          UINT                              DstZ,
          ID3D11Resource*                   pSrcResource,
          UINT                              SrcSubresource,
    const D3D11_BOX*                        pSrcBox,
          UINT                              CopyFlags) {
    FE_ENTRY(CopySubresourceRegion1);

    auto rec = Push<FeCopyRegionRec>();
    rec->dst       = pDstResource;
    rec->src       = pSrcResource;
    rec->dstSub    = DstSubresource;
    rec->dstX      = DstX;
    rec->dstY      = DstY;
    rec->dstZ      = DstZ;
    rec->srcSub    = SrcSubresource;
    rec->copyFlags = CopyFlags;
    rec->box       = pSrcBox ? *pSrcBox : D3D11_BOX();
    rec->hasBox    = pSrcBox != nullptr;
    rec->version1  = 1u;
    Recorded();
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::CopyResource(
          ID3D11Resource*                   pDstResource,
          ID3D11Resource*                   pSrcResource) {
    FE_ENTRY(CopyResource);
    RecordCall<FeCall_CopyResource>(pDstResource, pSrcResource);
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::CopyStructureCount(
          ID3D11Buffer*                     pDstBuffer,
          UINT                              DstAlignedByteOffset,
          ID3D11UnorderedAccessView*        pSrcView) {
    FE_ENTRY(CopyStructureCount);
    RecordCall<FeCall_CopyStructureCount>(pDstBuffer, DstAlignedByteOffset, pSrcView);
  }


  struct FeCall_ClearRenderTargetView {
    static void Call(D3D11ImmediateContext* ctx, ID3D11RenderTargetView* pView, FeFloat4 Color) {
      ctx->D3D11ImmediateContext::ClearRenderTargetView(pView, Color.v);
    }
  };

  struct FeCall_ClearUnorderedAccessViewUint {
    static void Call(D3D11ImmediateContext* ctx, ID3D11UnorderedAccessView* pView, FeUint4 Values) {
      ctx->D3D11ImmediateContext::ClearUnorderedAccessViewUint(pView, Values.v);
    }
  };

  struct FeCall_ClearUnorderedAccessViewFloat {
    static void Call(D3D11ImmediateContext* ctx, ID3D11UnorderedAccessView* pView, FeFloat4 Values) {
      ctx->D3D11ImmediateContext::ClearUnorderedAccessViewFloat(pView, Values.v);
    }
  };


  void STDMETHODCALLTYPE D3D11ThreadedContext::ClearRenderTargetView(
          ID3D11RenderTargetView*           pRenderTargetView,
    const FLOAT                             ColorRGBA[4]) {
    FE_ENTRY(ClearRenderTargetView);

    FeFloat4 color;
    std::memcpy(color.v, ColorRGBA, sizeof(color.v));
    RecordCall<FeCall_ClearRenderTargetView>(pRenderTargetView, color);
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::ClearUnorderedAccessViewUint(
          ID3D11UnorderedAccessView*        pUnorderedAccessView,
    const UINT                              Values[4]) {
    FE_ENTRY(ClearUnorderedAccessViewUint);

    FeUint4 values;
    std::memcpy(values.v, Values, sizeof(values.v));
    RecordCall<FeCall_ClearUnorderedAccessViewUint>(pUnorderedAccessView, values);
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::ClearUnorderedAccessViewFloat(
          ID3D11UnorderedAccessView*        pUnorderedAccessView,
    const FLOAT                             Values[4]) {
    FE_ENTRY(ClearUnorderedAccessViewFloat);

    FeFloat4 values;
    std::memcpy(values.v, Values, sizeof(values.v));
    RecordCall<FeCall_ClearUnorderedAccessViewFloat>(pUnorderedAccessView, values);
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::ClearDepthStencilView(
          ID3D11DepthStencilView*           pDepthStencilView,
          UINT                              ClearFlags,
          FLOAT                             Depth,
          UINT8                             Stencil) {
    FE_ENTRY(ClearDepthStencilView);
    RecordCall<FeCall_ClearDepthStencilView>(pDepthStencilView, ClearFlags, Depth, Stencil);
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::ClearView(
          ID3D11View*                       pView,
    const FLOAT                             Color[4],
    const D3D11_RECT*                       pRect,
          UINT                              NumRects) {
    FE_ENTRY(ClearView);

    if (unlikely(NumRects > MaxArray || (NumRects && !pRect))) {
      Drain(blessed::FeDrain::TooMany, blessed::FeCall::ClearView);
      m_ctx->ClearView(pView, Color, pRect, NumRects);
      return;
    }

    auto rec = static_cast<FeClearViewRec*>(AllocRecord(FeClearViewRec::Size(NumRects)));
    rec->fn    = &FeClearViewRec::Replay;
    rec->view  = pView;
    rec->count = NumRects;
    rec->pad   = 0u;
    std::memcpy(rec->color, Color, sizeof(rec->color));

    for (UINT i = 0; i < NumRects; i++)
      rec->rects()[i] = pRect[i];

    Recorded();
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::GenerateMips(
          ID3D11ShaderResourceView*         pShaderResourceView) {
    FE_ENTRY(GenerateMips);
    RecordCall<FeCall_GenerateMips>(pShaderResourceView);
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::ResolveSubresource(
          ID3D11Resource*                   pDstResource,
          UINT                              DstSubresource,
          ID3D11Resource*                   pSrcResource,
          UINT                              SrcSubresource,
          DXGI_FORMAT                       Format) {
    FE_ENTRY(ResolveSubresource);
    RecordCall<FeCall_ResolveSubresource>(pDstResource, DstSubresource, pSrcResource, SrcSubresource, Format);
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::UpdateSubresource(
          ID3D11Resource*                   pDstResource,
          UINT                              DstSubresource,
    const D3D11_BOX*                        pDstBox,
    const void*                             pSrcData,
          UINT                              SrcRowPitch,
          UINT                              SrcDepthPitch) {
    FE_ENTRY(UpdateSubresource);
    UpdateResource(pDstResource, DstSubresource, pDstBox, pSrcData, SrcRowPitch, SrcDepthPitch, ~0u);
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::UpdateSubresource1(
          ID3D11Resource*                   pDstResource,
          UINT                              DstSubresource,
    const D3D11_BOX*                        pDstBox,
    const void*                             pSrcData,
          UINT                              SrcRowPitch,
          UINT                              SrcDepthPitch,
          UINT                              CopyFlags) {
    FE_ENTRY(UpdateSubresource1);
    UpdateResource(pDstResource, DstSubresource, pDstBox, pSrcData, SrcRowPitch, SrcDepthPitch, CopyFlags);
  }


  void D3D11ThreadedContext::UpdateResource(
          ID3D11Resource*                   pDstResource,
          UINT                              DstSubresource,
    const D3D11_BOX*                        pDstBox,
    const void*                             pSrcData,
          UINT                              SrcRowPitch,
          UINT                              SrcDepthPitch,
          UINT                              CopyFlags) {
    // CopyFlags ~0u marks the UpdateSubresource entry (flags 0 in dxvk)
    bool version1 = CopyFlags != ~0u;
    UINT flags = version1 ? CopyFlags : 0u;

    if (!pDstResource)
      return;

    if (unlikely(BlessedDump::IsCapturing())) {
      // the dump logs the update in call order; run it where the dump is
      Drain(blessed::FeDrain::Capture, version1 ? blessed::FeCall::UpdateSubresource1 : blessed::FeCall::UpdateSubresource);

      if (version1)
        m_ctx->UpdateSubresource1(pDstResource, DstSubresource, pDstBox, pSrcData, SrcRowPitch, SrcDepthPitch, CopyFlags);
      else
        m_ctx->UpdateSubresource(pDstResource, DstSubresource, pDstBox, pSrcData, SrcRowPitch, SrcDepthPitch);

      SyncAppMapPtr(pDstResource);
      return;
    }

    D3D11_RESOURCE_DIMENSION resourceType = D3D11_RESOURCE_DIMENSION_UNKNOWN;
    pDstResource->GetType(&resourceType);

    if (likely(resourceType == D3D11_RESOURCE_DIMENSION_BUFFER)) {
      // The mapped-buffer fast path of D3D11CommonContext::UpdateResource,
      // game-side: it renames, and every rename starts here.
      auto buffer = static_cast<D3D11Buffer*>(pDstResource);
      uint64_t bufferSize = buffer->Desc()->ByteWidth;
      bool direct = buffer->GetMapMode() == D3D11_COMMON_BUFFER_MAP_MODE_DIRECT;

      uint64_t offset = 0u;
      uint64_t length = bufferSize;

      if (pDstBox) {
        offset = pDstBox->left;
        length = pDstBox->right - offset;
      }

      if (likely(direct)) {
        UINT mappedFlags = flags & (D3D11_COPY_DISCARD | D3D11_COPY_NO_OVERWRITE);

        if (!pDstBox || ((offset + length <= bufferSize) && (length == bufferSize || mappedFlags))) {
          if (pDstBox)
            flags = mappedFlags;
          else
            offset = 0u, length = bufferSize, flags = 0u;

          void* mapPtr = flags != D3D11_COPY_NO_OVERWRITE
            ? DiscardBuffer(buffer)
            : buffer->BlessedAppMapPtr();

          std::memcpy(reinterpret_cast<char*>(mapPtr) + offset, pSrcData, length);
          return;
        }
      }

      // Recorded: a gpu copy of exactly these bytes, or nothing
      size_t size = (offset + length <= bufferSize) ? size_t(length) : 0u;
      RecordUpdate(pDstResource, DstSubresource, pDstBox, pSrcData, SrcRowPitch, SrcDepthPitch, CopyFlags, size);
    } else {
      size_t size = 0u;

      if (!ComputeUpdateSize(pDstResource, DstSubresource, pDstBox, SrcRowPitch, SrcDepthPitch, &size)) {
        Drain(blessed::FeDrain::UpdateSubresource, version1 ? blessed::FeCall::UpdateSubresource1 : blessed::FeCall::UpdateSubresource);

        if (version1)
          m_ctx->UpdateSubresource1(pDstResource, DstSubresource, pDstBox, pSrcData, SrcRowPitch, SrcDepthPitch, CopyFlags);
        else
          m_ctx->UpdateSubresource(pDstResource, DstSubresource, pDstBox, pSrcData, SrcRowPitch, SrcDepthPitch);
        return;
      }

      RecordUpdate(pDstResource, DstSubresource, pDstBox, pSrcData, SrcRowPitch, SrcDepthPitch, CopyFlags, size);
    }
  }


  bool D3D11ThreadedContext::ComputeUpdateSize(
          ID3D11Resource*             pDstResource,
          UINT                        DstSubresource,
    const D3D11_BOX*                  pDstBox,
          UINT                        SrcRowPitch,
          UINT                        SrcDepthPitch,
          size_t*                     pSize) {
    // Exactly the bytes D3D11CommonContext::UpdateTexture will read from
    // the app's pointer: its early outs read nothing, and packImageData
    // copies either one block or row by row. Planar and multi-aspect
    // formats drain instead.
    D3D11CommonTexture* texture = GetCommonTexture(pDstResource);
    *pSize = 0u;

    if (DstSubresource >= texture->CountSubresources())
      return true;

    VkFormat packedFormat = texture->GetPackedFormat();
    auto formatInfo = lookupFormatInfo(packedFormat);

    if (formatInfo->flags.test(DxvkFormatFlag::MultiPlane)
     || (formatInfo->aspectMask & (formatInfo->aspectMask - 1u)))
      return false;

    auto subresource = texture->GetSubresourceFromIndex(formatInfo->aspectMask, DstSubresource);
    VkExtent3D mipExtent = texture->MipLevelExtent(subresource.mipLevel);

    VkOffset3D offset = { 0, 0, 0 };
    VkExtent3D extent = mipExtent;

    if (pDstBox) {
      if (pDstBox->left >= pDstBox->right
       || pDstBox->top >= pDstBox->bottom
       || pDstBox->front >= pDstBox->back)
        return true;

      offset.x = pDstBox->left;
      offset.y = pDstBox->top;
      offset.z = pDstBox->front;

      extent.width  = pDstBox->right - pDstBox->left;
      extent.height = pDstBox->bottom - pDstBox->top;
      extent.depth  = pDstBox->back - pDstBox->front;
    }

    if (!util::isBlockAligned(offset, extent, formatInfo->blockSize, mipExtent))
      return true;

    VkExtent3D blocks = util::computeBlockCount(extent, formatInfo->blockSize);

    VkDeviceSize bytesPerRow   = blocks.width  * formatInfo->elementSize;
    VkDeviceSize bytesPerSlice = blocks.height * bytesPerRow;
    VkDeviceSize bytesTotal    = blocks.depth  * bytesPerSlice;

    bool directCopy = (bytesPerRow   == SrcRowPitch   || blocks.height == 1u)
                   && (bytesPerSlice == SrcDepthPitch || blocks.depth  == 1u);

    *pSize = directCopy ? size_t(bytesTotal)
      : size_t(VkDeviceSize(blocks.depth - 1u) * SrcDepthPitch
             + VkDeviceSize(blocks.height - 1u) * SrcRowPitch
             + bytesPerRow);
    return true;
  }


  void D3D11ThreadedContext::RecordUpdate(
          ID3D11Resource*             pDstResource,
          UINT                        DstSubresource,
    const D3D11_BOX*                  pDstBox,
    const void*                       pSrcData,
          UINT                        SrcRowPitch,
          UINT                        SrcDepthPitch,
          UINT                        CopyFlags,
          size_t                      DataSize) {
    bool side = DataSize > MaxInlineData;

    auto rec = static_cast<FeUpdateRec*>(AllocRecord(FeUpdateRec::Size(side ? 0u : DataSize)));
    rec->fn         = &FeUpdateRec::Replay;
    rec->dst        = pDstResource;
    rec->side       = side ? std::malloc(DataSize) : nullptr;
    rec->size       = DataSize;
    rec->dstSub     = DstSubresource;
    rec->rowPitch   = SrcRowPitch;
    rec->depthPitch = SrcDepthPitch;
    rec->copyFlags  = CopyFlags != ~0u ? CopyFlags : 0u;
    rec->box        = pDstBox ? *pDstBox : D3D11_BOX();
    rec->hasBox     = pDstBox != nullptr;
    rec->version1   = CopyFlags != ~0u;

    if (DataSize) {
      void* dst = side ? rec->side : static_cast<void*>(rec + 1);

      if (unlikely(!dst))
        throw DxvkError("d3d11.blessedThreadedFrontEnd: out of memory for an UpdateSubresource copy");

      std::memcpy(dst, pSrcData, DataSize);
    }

    Recorded();
  }


  // --- draws and dispatches: record, then publish ---

  void STDMETHODCALLTYPE D3D11ThreadedContext::DrawAuto() {
    FE_ENTRY(DrawAuto);
    RecordCall<FeCall_DrawAuto>();
    Publish();
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::Draw(
          UINT            VertexCount,
          UINT            StartVertexLocation) {
    FE_ENTRY(Draw);

    if (likely(m_compact)) {
      auto op = PushOp<FeOpDraw>();
      op->h     = { FeOp::Draw, FeStage::VS, 0u, VertexCount };
      op->start = StartVertexLocation;
      op->base  = 0;
      Recorded();
      return;
    }

    RecordCall<FeCall_Draw>(VertexCount, StartVertexLocation);
    Publish();
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::DrawIndexed(
          UINT            IndexCount,
          UINT            StartIndexLocation,
          INT             BaseVertexLocation) {
    FE_ENTRY(DrawIndexed);

    if (likely(m_compact)) {
      auto op = PushOp<FeOpDraw>();
      op->h     = { FeOp::DrawIndexed, FeStage::VS, 0u, IndexCount };
      op->start = StartIndexLocation;
      op->base  = BaseVertexLocation;
      Recorded();
      return;
    }

    RecordCall<FeCall_DrawIndexed>(IndexCount, StartIndexLocation, BaseVertexLocation);
    Publish();
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::DrawInstanced(
          UINT            VertexCountPerInstance,
          UINT            InstanceCount,
          UINT            StartVertexLocation,
          UINT            StartInstanceLocation) {
    FE_ENTRY(DrawInstanced);

    if (likely(m_compact)) {
      auto op = PushOp<FeOpDrawInstanced>();
      op->h             = { FeOp::DrawInstanced, FeStage::VS, 0u, VertexCountPerInstance };
      op->instances     = InstanceCount;
      op->start         = StartVertexLocation;
      op->base          = 0;
      op->startInstance = StartInstanceLocation;
      Recorded();
      return;
    }

    RecordCall<FeCall_DrawInstanced>(VertexCountPerInstance, InstanceCount,
      StartVertexLocation, StartInstanceLocation);
    Publish();
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::DrawIndexedInstanced(
          UINT            IndexCountPerInstance,
          UINT            InstanceCount,
          UINT            StartIndexLocation,
          INT             BaseVertexLocation,
          UINT            StartInstanceLocation) {
    FE_ENTRY(DrawIndexedInstanced);

    if (likely(m_compact)) {
      auto op = PushOp<FeOpDrawInstanced>();
      op->h             = { FeOp::DrawIndexedInstanced, FeStage::VS, 0u, IndexCountPerInstance };
      op->instances     = InstanceCount;
      op->start         = StartIndexLocation;
      op->base          = BaseVertexLocation;
      op->startInstance = StartInstanceLocation;
      Recorded();
      return;
    }

    RecordCall<FeCall_DrawIndexedInstanced>(IndexCountPerInstance, InstanceCount,
      StartIndexLocation, BaseVertexLocation, StartInstanceLocation);
    Publish();
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::DrawIndexedInstancedIndirect(
          ID3D11Buffer*   pBufferForArgs,
          UINT            AlignedByteOffsetForArgs) {
    FE_ENTRY(DrawIndexedInstancedIndirect);
    RecordCall<FeCall_DrawIndexedInstancedIndirect>(pBufferForArgs, AlignedByteOffsetForArgs);
    Publish();
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::DrawInstancedIndirect(
          ID3D11Buffer*   pBufferForArgs,
          UINT            AlignedByteOffsetForArgs) {
    FE_ENTRY(DrawInstancedIndirect);
    RecordCall<FeCall_DrawInstancedIndirect>(pBufferForArgs, AlignedByteOffsetForArgs);
    Publish();
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::Dispatch(
          UINT            ThreadGroupCountX,
          UINT            ThreadGroupCountY,
          UINT            ThreadGroupCountZ) {
    FE_ENTRY(Dispatch);
    RecordCall<FeCall_Dispatch>(ThreadGroupCountX, ThreadGroupCountY, ThreadGroupCountZ);
    Publish();
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::DispatchIndirect(
          ID3D11Buffer*   pBufferForArgs,
          UINT            AlignedByteOffsetForArgs) {
    FE_ENTRY(DispatchIndirect);
    RecordCall<FeCall_DispatchIndirect>(pBufferForArgs, AlignedByteOffsetForArgs);
    Publish();
  }


  // --- input assembler ---

  void STDMETHODCALLTYPE D3D11ThreadedContext::IASetInputLayout(ID3D11InputLayout* pInputLayout) {
    FE_ENTRY(IASetInputLayout);

    if (likely(m_compact)) {
      if (m_shadow->il == pInputLayout) {
        m_folded += 1u;
        return;
      }

      m_shadow->il = pInputLayout;

      auto op = PushOp<FeOpPtr>();
      op->h   = { FeOp::SetInputLayout, FeStage::VS, 0u, 0u };
      op->ptr = pInputLayout;
      Recorded();
      return;
    }

    RecordCall<FeCall_IASetInputLayout>(pInputLayout);
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY Topology) {
    FE_ENTRY(IASetPrimitiveTopology);

    if (likely(m_compact)) {
      if (m_shadow->topology == UINT(Topology)) {
        m_folded += 1u;
        return;
      }

      m_shadow->topology = UINT(Topology);

      auto op = PushOp<FeOpTopology>();
      op->h = { FeOp::SetTopology, FeStage::VS, 0u, UINT(Topology) };
      Recorded();
      return;
    }

    RecordCall<FeCall_IASetPrimitiveTopology>(Topology);
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::IASetVertexBuffers(
          UINT                              StartSlot,
          UINT                              NumBuffers,
          ID3D11Buffer* const*              ppVertexBuffers,
    const UINT*                             pStrides,
    const UINT*                             pOffsets) {
    FE_ENTRY(IASetVertexBuffers);

    if (unlikely(NumBuffers > MaxArray || (NumBuffers && (!ppVertexBuffers || !pStrides || !pOffsets)))) {
      Drain(blessed::FeDrain::TooMany, blessed::FeCall::IASetVertexBuffers);
      m_ctx->IASetVertexBuffers(StartSlot, NumBuffers, ppVertexBuffers, pStrides, pOffsets);
      InvalidateShadow();
      return;
    }

    if (likely(m_compact)) {
      if (unlikely(StartSlot + NumBuffers > D3D11_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT)) {
        // out of range: dxvk ignores the call, record it as it came
        InvalidateShadow();
      } else if (NumBuffers == 1u) {
        FeShadow::Vb& vb = m_shadow->vb[StartSlot];

        if (vb.buffer == ppVertexBuffers[0] && vb.stride == pStrides[0] && vb.offset == pOffsets[0]) {
          m_folded += 1u;
          return;
        }

        vb = { ppVertexBuffers[0], pStrides[0], pOffsets[0] };

        auto op = PushOp<FeOpVb>();
        op->h      = { FeOp::SetVb, FeStage::VS, uint16_t(StartSlot), pStrides[0] };
        op->buffer = ppVertexBuffers[0];
        op->offset = pOffsets[0];
        op->pad    = 0u;
        Recorded();
        return;
      } else {
        for (UINT i = 0; i < NumBuffers; i++)
          m_shadow->vb[StartSlot + i] = { ppVertexBuffers[i], pStrides[i], pOffsets[i] };
      }
    }

    auto rec = static_cast<FeVertexBuffersRec*>(AllocRecord(FeVertexBuffersRec::Size(NumBuffers)));
    rec->fn    = &FeVertexBuffersRec::Replay;
    rec->start = StartSlot;
    rec->count = NumBuffers;

    for (UINT i = 0; i < NumBuffers; i++) {
      rec->buffers()[i] = ppVertexBuffers[i];
      rec->strides()[i] = pStrides[i];
      rec->offsets()[i] = pOffsets[i];
    }

    Recorded();
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::IASetIndexBuffer(
          ID3D11Buffer*                     pIndexBuffer,
          DXGI_FORMAT                       Format,
          UINT                              Offset) {
    FE_ENTRY(IASetIndexBuffer);

    if (likely(m_compact && uint32_t(Format) <= 0xffffu)) {
      if (m_shadow->ib == pIndexBuffer && m_shadow->ibFormat == UINT(Format) && m_shadow->ibOffset == Offset) {
        m_folded += 1u;
        return;
      }

      m_shadow->ib       = pIndexBuffer;
      m_shadow->ibFormat = UINT(Format);
      m_shadow->ibOffset = Offset;

      auto op = PushOp<FeOpIb>();
      op->h      = { FeOp::SetIb, FeStage::VS, uint16_t(Format), Offset };
      op->buffer = pIndexBuffer;
      Recorded();
      return;
    }

    if (m_compact)
      InvalidateShadow();

    RecordCall<FeCall_IASetIndexBuffer>(pIndexBuffer, Format, Offset);
  }


  // --- shader stages ---

#define FE_STAGE(Stage, Iface) \
  void STDMETHODCALLTYPE D3D11ThreadedContext::Stage##SetShader( \
          Iface*                            pShader, \
          ID3D11ClassInstance* const*       ppClassInstances, \
          UINT                              NumClassInstances) { \
    FE_ENTRY(Stage##SetShader); \
    if (unlikely(NumClassInstances)) { \
      Drain(blessed::FeDrain::ClassInstances, blessed::FeCall::Stage##SetShader); \
      m_ctx->Stage##SetShader(pShader, ppClassInstances, NumClassInstances); \
      InvalidateShadow(); \
      return; \
    } \
    if (likely(m_compact)) { \
      RecordShader(FeStage::Stage, pShader); \
      return; \
    } \
    RecordCall<FeCall_##Stage##SetShader>(pShader); \
  } \
  \
  void STDMETHODCALLTYPE D3D11ThreadedContext::Stage##SetConstantBuffers( \
          UINT                              StartSlot, \
          UINT                              NumBuffers, \
          ID3D11Buffer* const*              ppConstantBuffers) { \
    FE_ENTRY(Stage##SetConstantBuffers); \
    if (unlikely(NumBuffers > MaxArray || (NumBuffers && !ppConstantBuffers))) { \
      Drain(blessed::FeDrain::TooMany, blessed::FeCall::Stage##SetConstantBuffers); \
      m_ctx->Stage##SetConstantBuffers(StartSlot, NumBuffers, ppConstantBuffers); \
      InvalidateShadow(); \
      return; \
    } \
    if (likely(m_compact) && RecordConstantBuffersCompact(FeStage::Stage, StartSlot, NumBuffers, \
        ppConstantBuffers, nullptr, nullptr)) \
      return; \
    RecordArray<FeCall_##Stage##SetConstantBuffers>(StartSlot, NumBuffers, ppConstantBuffers); \
  } \
  \
  void STDMETHODCALLTYPE D3D11ThreadedContext::Stage##SetConstantBuffers1( \
          UINT                              StartSlot, \
          UINT                              NumBuffers, \
          ID3D11Buffer* const*              ppConstantBuffers, \
    const UINT*                             pFirstConstant, \
    const UINT*                             pNumConstants) { \
    FE_ENTRY(Stage##SetConstantBuffers1); \
    if (unlikely(NumBuffers > MaxArray || (NumBuffers && !ppConstantBuffers))) { \
      Drain(blessed::FeDrain::TooMany, blessed::FeCall::Stage##SetConstantBuffers1); \
      m_ctx->Stage##SetConstantBuffers1(StartSlot, NumBuffers, ppConstantBuffers, pFirstConstant, pNumConstants); \
      InvalidateShadow(); \
      return; \
    } \
    if (likely(m_compact) && RecordConstantBuffersCompact(FeStage::Stage, StartSlot, NumBuffers, \
        ppConstantBuffers, pFirstConstant, pNumConstants)) \
      return; \
    RecordConstantBuffers1<FeCall_##Stage##SetConstantBuffers1>(StartSlot, NumBuffers, \
      ppConstantBuffers, pFirstConstant, pNumConstants); \
  } \
  \
  void STDMETHODCALLTYPE D3D11ThreadedContext::Stage##SetShaderResources( \
          UINT                              StartSlot, \
          UINT                              NumViews, \
          ID3D11ShaderResourceView* const*  ppShaderResourceViews) { \
    FE_ENTRY(Stage##SetShaderResources); \
    if (unlikely(NumViews > MaxArray || (NumViews && !ppShaderResourceViews))) { \
      Drain(blessed::FeDrain::TooMany, blessed::FeCall::Stage##SetShaderResources); \
      m_ctx->Stage##SetShaderResources(StartSlot, NumViews, ppShaderResourceViews); \
      return; \
    } \
    if (likely(m_compact) && NumViews == 1u) { \
      RecordSlotOp(FeOp::SetSrv, FeStage::Stage, StartSlot, ppShaderResourceViews[0]); \
      return; \
    } \
    RecordArray<FeCall_##Stage##SetShaderResources>(StartSlot, NumViews, ppShaderResourceViews); \
  } \
  \
  void STDMETHODCALLTYPE D3D11ThreadedContext::Stage##SetSamplers( \
          UINT                              StartSlot, \
          UINT                              NumSamplers, \
          ID3D11SamplerState* const*        ppSamplers) { \
    FE_ENTRY(Stage##SetSamplers); \
    if (unlikely(NumSamplers > MaxArray || (NumSamplers && !ppSamplers))) { \
      Drain(blessed::FeDrain::TooMany, blessed::FeCall::Stage##SetSamplers); \
      m_ctx->Stage##SetSamplers(StartSlot, NumSamplers, ppSamplers); \
      return; \
    } \
    if (likely(m_compact) && NumSamplers == 1u) { \
      RecordSlotOp(FeOp::SetSampler, FeStage::Stage, StartSlot, ppSamplers[0]); \
      return; \
    } \
    RecordArray<FeCall_##Stage##SetSamplers>(StartSlot, NumSamplers, ppSamplers); \
  } \
  \
  void STDMETHODCALLTYPE D3D11ThreadedContext::Stage##GetShader( \
          Iface**                           ppShader, \
          ID3D11ClassInstance**             ppClassInstances, \
          UINT*                             pNumClassInstances) { \
    FE_DRAIN(Stage##GetShader, Getter); \
    m_ctx->Stage##GetShader(ppShader, ppClassInstances, pNumClassInstances); \
  } \
  \
  void STDMETHODCALLTYPE D3D11ThreadedContext::Stage##GetConstantBuffers( \
          UINT                              StartSlot, \
          UINT                              NumBuffers, \
          ID3D11Buffer**                    ppConstantBuffers) { \
    FE_DRAIN(Stage##GetConstantBuffers, Getter); \
    m_ctx->Stage##GetConstantBuffers(StartSlot, NumBuffers, ppConstantBuffers); \
  } \
  \
  void STDMETHODCALLTYPE D3D11ThreadedContext::Stage##GetConstantBuffers1( \
          UINT                              StartSlot, \
          UINT                              NumBuffers, \
          ID3D11Buffer**                    ppConstantBuffers, \
          UINT*                             pFirstConstant, \
          UINT*                             pNumConstants) { \
    FE_DRAIN(Stage##GetConstantBuffers1, Getter); \
    m_ctx->Stage##GetConstantBuffers1(StartSlot, NumBuffers, ppConstantBuffers, pFirstConstant, pNumConstants); \
  } \
  \
  void STDMETHODCALLTYPE D3D11ThreadedContext::Stage##GetShaderResources( \
          UINT                              StartSlot, \
          UINT                              NumViews, \
          ID3D11ShaderResourceView**        ppShaderResourceViews) { \
    FE_DRAIN(Stage##GetShaderResources, Getter); \
    m_ctx->Stage##GetShaderResources(StartSlot, NumViews, ppShaderResourceViews); \
  } \
  \
  void STDMETHODCALLTYPE D3D11ThreadedContext::Stage##GetSamplers( \
          UINT                              StartSlot, \
          UINT                              NumSamplers, \
          ID3D11SamplerState**              ppSamplers) { \
    FE_DRAIN(Stage##GetSamplers, Getter); \
    m_ctx->Stage##GetSamplers(StartSlot, NumSamplers, ppSamplers); \
  }

  FE_STAGE(VS, ID3D11VertexShader)
  FE_STAGE(HS, ID3D11HullShader)
  FE_STAGE(DS, ID3D11DomainShader)
  FE_STAGE(GS, ID3D11GeometryShader)
  FE_STAGE(PS, ID3D11PixelShader)
  FE_STAGE(CS, ID3D11ComputeShader)

#undef FE_STAGE


  void STDMETHODCALLTYPE D3D11ThreadedContext::CSSetUnorderedAccessViews(
          UINT                              StartSlot,
          UINT                              NumUAVs,
          ID3D11UnorderedAccessView* const* ppUnorderedAccessViews,
    const UINT*                             pUAVInitialCounts) {
    FE_ENTRY(CSSetUnorderedAccessViews);

    if (unlikely(NumUAVs > MaxArray || (NumUAVs && !ppUnorderedAccessViews))) {
      Drain(blessed::FeDrain::TooMany, blessed::FeCall::CSSetUnorderedAccessViews);
      m_ctx->CSSetUnorderedAccessViews(StartSlot, NumUAVs, ppUnorderedAccessViews, pUAVInitialCounts);
      return;
    }

    auto rec = static_cast<FeComputeUavsRec*>(AllocRecord(FeComputeUavsRec::Size(NumUAVs)));
    rec->fn        = &FeComputeUavsRec::Replay;
    rec->start     = StartSlot;
    rec->count     = NumUAVs;
    rec->hasCounts = pUAVInitialCounts != nullptr;
    rec->pad       = 0u;

    for (UINT i = 0; i < NumUAVs; i++) {
      rec->uavs()[i]   = ppUnorderedAccessViews[i];
      rec->counts()[i] = pUAVInitialCounts ? pUAVInitialCounts[i] : 0u;
    }

    Recorded();
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::CSGetUnorderedAccessViews(
          UINT                              StartSlot,
          UINT                              NumUAVs,
          ID3D11UnorderedAccessView**       ppUnorderedAccessViews) {
    FE_DRAIN(CSGetUnorderedAccessViews, Getter);
    m_ctx->CSGetUnorderedAccessViews(StartSlot, NumUAVs, ppUnorderedAccessViews);
  }


  // --- output merger ---

  // blessed: fe-getters -- see the declaration in blessed_threaded_context.h
  void D3D11ThreadedContext::UpdateOmShadow(
          UINT                              NumRTVs,
          ID3D11RenderTargetView* const*    ppRenderTargetViews,
          ID3D11DepthStencilView*           pDepthStencilView,
          UINT                              UAVStartSlot,
          UINT                              NumUAVs,
          ID3D11UnorderedAccessView* const* ppUnorderedAccessViews) {
    FeShadow& s = *m_shadow;

    // blessed: with the getter shadow off (the default) nothing reads this;
    // skip the app-thread validation it costs (seen in a whiterun cpu sample)
    if (m_noOmShadow) {
      s.omUnknown = 1u;
      return;
    }

    bool keepRtv = NumRTVs == D3D11_KEEP_RENDER_TARGETS_AND_DEPTH_STENCIL;
    bool keepUav = NumUAVs == D3D11_KEEP_UNORDERED_ACCESS_VIEWS;

    // A UAV range past the slot array indexes m_state.om.uavs out of
    // bounds in dxvk too: nothing defined to mirror, so stop trusting it.
    // With NumUAVs = 0 dxvk never reads UAVStartSlot.
    if (unlikely(!keepUav && NumUAVs && (UAVStartSlot > FeShadow::MaxOmUav
     || NumUAVs > FeShadow::MaxOmUav - UAVStartSlot))) {
      s.omUnknown = 1u;
      return;
    }

    // The two early-outs of SetRenderTargetsAndUnorderedAccessViews: when
    // either rejects the call, dxvk's bindings stay as they were, so ours
    // must too. Both read only the arguments' view metadata, fixed at view
    // creation, never the context, so they are safe on this thread while
    // the front end replays. Skipping them would leave the shadow naming a
    // view dxvk never bound, which the app may release and the getter
    // would then AddRef after its final release ran.
    if (unlikely(m_ctx->TestRtvUavHazards(NumRTVs, ppRenderTargetViews, NumUAVs, ppUnorderedAccessViews)))
      return;

    if (!keepRtv && unlikely(!m_ctx->ValidateRenderTargets(NumRTVs, ppRenderTargetViews, pDepthStencilView)))
      return;

    if (unlikely(s.omUnknown)) {
      // A KEEP half leaves bindings this shadow never saw
      if (keepRtv || keepUav)
        return;

      // Both halves are replaced outright, and dxvk keeps every UAV slot
      // outside [minUav, maxUav) null, so the result does not depend on
      // what was bound before: start from the empty state
      for (UINT i = 0; i < FeShadow::MaxOmRtv; i++)
        s.omRtv[i] = nullptr;

      for (UINT i = 0; i < FeShadow::MaxOmUav; i++)
        s.omUav[i] = nullptr;

      s.omDsv     = nullptr;
      s.omMaxRtv  = 0u;
      s.omMinUav  = FeShadow::MaxOmUav;
      s.omMaxUav  = 0u;
      s.omUnknown = 0u;
    }

    if (!keepRtv) {
      for (UINT i = 0; i < FeShadow::MaxOmRtv; i++) {
        auto rtv = i < NumRTVs ? ppRenderTargetViews[i] : nullptr;

        if (s.omRtv[i] != rtv) {
          s.omRtv[i] = rtv;

          // ResolveOmUavHazards: a new RTV unbinds overlapping OM UAVs,
          // only when the call keeps the UAVs
          auto view = static_cast<D3D11RenderTargetView*>(rtv);

          if (keepUav && view && view->HasBindFlag(D3D11_BIND_UNORDERED_ACCESS)) {
            for (UINT j = 0; j < s.omMaxUav; j++) {
              if (CheckViewOverlap(view, static_cast<D3D11UnorderedAccessView*>(s.omUav[j])))
                s.omUav[j] = nullptr;
            }
          }
        }
      }

      s.omDsv    = pDepthStencilView;
      s.omMaxRtv = NumRTVs;
    }

    if (!keepUav && (NumUAVs || s.omMaxUav)) {
      UINT newMinUav = NumUAVs ? UAVStartSlot : FeShadow::MaxOmUav;
      UINT newMaxUav = NumUAVs ? UAVStartSlot + NumUAVs : 0u;

      UINT lo = std::min(s.omMinUav, newMinUav);
      UINT hi = std::max(s.omMaxUav, newMaxUav);

      s.omMinUav = newMinUav;
      s.omMaxUav = newMaxUav;

      for (UINT i = lo; i < hi; i++) {
        auto uav = (i >= UAVStartSlot && i < UAVStartSlot + NumUAVs)
          ? ppUnorderedAccessViews[i - UAVStartSlot] : nullptr;

        if (s.omUav[i] != uav) {
          s.omUav[i] = uav;

          // ResolveOmRtvHazards: a new UAV unbinds an overlapping DSV and
          // RTVs, only when the call keeps the render targets
          auto view = static_cast<D3D11UnorderedAccessView*>(uav);

          if (keepRtv && view && view->HasBindFlag(D3D11_BIND_RENDER_TARGET)) {
            if (CheckViewOverlap(view, static_cast<D3D11DepthStencilView*>(s.omDsv)))
              s.omDsv = nullptr;

            for (UINT j = 0; j < s.omMaxRtv; j++) {
              if (CheckViewOverlap(view, static_cast<D3D11RenderTargetView*>(s.omRtv[j])))
                s.omRtv[j] = nullptr;
            }
          }
        }
      }
    }
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::OMSetRenderTargets(
          UINT                              NumViews,
          ID3D11RenderTargetView* const*    ppRenderTargetViews,
          ID3D11DepthStencilView*           pDepthStencilView) {
    FE_ENTRY(OMSetRenderTargets);

    if (unlikely(NumViews > D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT || (NumViews && !ppRenderTargetViews))) {
      Drain(blessed::FeDrain::TooMany, blessed::FeCall::OMSetRenderTargets);
      m_ctx->OMSetRenderTargets(NumViews, ppRenderTargetViews, pDepthStencilView);
      InvalidateShadow(); // blessed: fe-getters -- the real call above may have changed OM state our shadow never saw
      return;
    }

    auto rec = static_cast<FeRenderTargetsRec*>(AllocRecord(FeRenderTargetsRec::Size(NumViews, 0u)));
    rec->fn        = &FeRenderTargetsRec::Replay;
    rec->dsv       = pDepthStencilView;
    rec->numRtvs   = NumViews;
    rec->rtvCount  = NumViews;
    rec->uavStart  = 0u;
    rec->numUavs   = 0u;
    rec->uavCount  = 0u;
    rec->hasCounts = 0u;
    rec->withUavs  = 0u;
    rec->pad       = 0u;

    for (UINT i = 0; i < NumViews; i++)
      rec->rtvs()[i] = ppRenderTargetViews[i];

    // blessed: fe-getters -- OMSetRenderTargets translates to
    // SetRenderTargetsAndUnorderedAccessViews(NumViews, ..., NumViews, 0,
    // nullptr, nullptr) (d3d11_context.cpp:3059): NumUAVs=0, not KEEP, so
    // this also unbinds every UAV, same as the real call the record replays.
    UpdateOmShadow(NumViews, ppRenderTargetViews, pDepthStencilView, NumViews, 0u, nullptr);

    Recorded();
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::OMSetRenderTargetsAndUnorderedAccessViews(
          UINT                              NumRTVs,
          ID3D11RenderTargetView* const*    ppRenderTargetViews,
          ID3D11DepthStencilView*           pDepthStencilView,
          UINT                              UAVStartSlot,
          UINT                              NumUAVs,
          ID3D11UnorderedAccessView* const* ppUnorderedAccessViews,
    const UINT*                             pUAVInitialCounts) {
    FE_ENTRY(OMSetRenderTargetsAndUnorderedAccessViews);

    // The KEEP values leave that half alone; nothing to copy for it
    UINT rtvCount = NumRTVs == D3D11_KEEP_RENDER_TARGETS_AND_DEPTH_STENCIL ? 0u : NumRTVs;
    UINT uavCount = NumUAVs == D3D11_KEEP_UNORDERED_ACCESS_VIEWS ? 0u : NumUAVs;

    if (unlikely(rtvCount > D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT || uavCount > MaxArray
     || (rtvCount && !ppRenderTargetViews) || (uavCount && !ppUnorderedAccessViews))) {
      Drain(blessed::FeDrain::TooMany, blessed::FeCall::OMSetRenderTargetsAndUnorderedAccessViews);
      m_ctx->OMSetRenderTargetsAndUnorderedAccessViews(NumRTVs, ppRenderTargetViews,
        pDepthStencilView, UAVStartSlot, NumUAVs, ppUnorderedAccessViews, pUAVInitialCounts);
      InvalidateShadow(); // blessed: fe-getters -- ditto OMSetRenderTargets above
      return;
    }

    auto rec = static_cast<FeRenderTargetsRec*>(AllocRecord(FeRenderTargetsRec::Size(rtvCount, uavCount)));
    rec->fn        = &FeRenderTargetsRec::Replay;
    rec->dsv       = pDepthStencilView;
    rec->numRtvs   = NumRTVs;
    rec->rtvCount  = rtvCount;
    rec->uavStart  = UAVStartSlot;
    rec->numUavs   = NumUAVs;
    rec->uavCount  = uavCount;
    rec->hasCounts = pUAVInitialCounts != nullptr;
    rec->withUavs  = 1u;
    rec->pad       = 0u;

    for (UINT i = 0; i < rtvCount; i++)
      rec->rtvs()[i] = ppRenderTargetViews[i];

    for (UINT i = 0; i < uavCount; i++) {
      rec->uavs()[i]   = ppUnorderedAccessViews[i];
      rec->counts()[i] = pUAVInitialCounts ? pUAVInitialCounts[i] : 0u;
    }

    // blessed: fe-getters -- raw NumRTVs/NumUAVs (KEEP sentinels included),
    // not rtvCount/uavCount: UpdateOmShadow tells KEEP from a real 0 itself
    UpdateOmShadow(NumRTVs, ppRenderTargetViews, pDepthStencilView,
      UAVStartSlot, NumUAVs, ppUnorderedAccessViews);

    Recorded();
  }


  struct FeCall_OMSetBlendState {
    static void Call(D3D11ImmediateContext* ctx, ID3D11BlendState* pState, FeFloat4 Factor, BOOL HasFactor, UINT SampleMask) {
      ctx->D3D11ImmediateContext::OMSetBlendState(pState, HasFactor ? Factor.v : nullptr, SampleMask);
    }
  };


  void STDMETHODCALLTYPE D3D11ThreadedContext::OMSetBlendState(
          ID3D11BlendState*                 pBlendState,
    const FLOAT                             BlendFactor[4],
          UINT                              SampleMask) {
    FE_ENTRY(OMSetBlendState);

    // A null factor leaves the factor as it is (dxvk, as d3d11)
    if (likely(m_compact)) {
      if (m_shadow->blend == pBlendState && m_shadow->sampleMask == SampleMask
       && (!BlendFactor || !std::memcmp(m_shadow->blendFactor, BlendFactor, sizeof(m_shadow->blendFactor)))) {
        m_folded += 1u;
        return;
      }

      m_shadow->blend      = pBlendState;
      m_shadow->sampleMask = SampleMask;

      if (BlendFactor)
        std::memcpy(m_shadow->blendFactor, BlendFactor, sizeof(m_shadow->blendFactor));
    }

    FeFloat4 factor = { };

    if (BlendFactor)
      std::memcpy(factor.v, BlendFactor, sizeof(factor.v));

    RecordCall<FeCall_OMSetBlendState>(pBlendState, factor, BOOL(BlendFactor != nullptr), SampleMask);
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::OMSetDepthStencilState(
          ID3D11DepthStencilState*          pDepthStencilState,
          UINT                              StencilRef) {
    FE_ENTRY(OMSetDepthStencilState);

    if (likely(m_compact)) {
      if (m_shadow->ds == pDepthStencilState && m_shadow->stencilRef == StencilRef) {
        m_folded += 1u;
        return;
      }

      m_shadow->ds         = pDepthStencilState;
      m_shadow->stencilRef = StencilRef;
    }

    RecordCall<FeCall_OMSetDepthStencilState>(pDepthStencilState, StencilRef);
  }


  // blessed: fe-getters -- BLESSED_FE_SHADOW_VERIFY: drains, then compares
  // the whole OM half of the shadow with dxvk's own bindings, not only the
  // slots the app asked for, so a divergence shows at its first getter.
  // Reached only with the shadow known.
  void D3D11ThreadedContext::VerifyOmShadow(blessed::FeCall Call) {
    ID3D11RenderTargetView*    rtv[FeShadow::MaxOmRtv] = { };
    ID3D11DepthStencilView*    dsv = nullptr;
    ID3D11UnorderedAccessView* uav[FeShadow::MaxOmUav] = { };

    // Reading the real context while the front end replays is the race
    // the front end exists to prevent: drain first. This puts back the
    // cost the shadow removed, on purpose, while verifying.
    Drain(blessed::FeDrain::Getter, Call);
    m_ctx->OMGetRenderTargetsAndUnorderedAccessViews(FeShadow::MaxOmRtv, rtv, &dsv,
      0u, FeShadow::MaxOmUav, uav);

    for (UINT i = 0; i < FeShadow::MaxOmRtv; i++) {
      blessed::FeShadowVerifyRtv(Call, i, m_shadow->omRtv[i], rtv[i]);

      if (rtv[i])
        rtv[i]->Release();
    }

    blessed::FeShadowVerifyDsv(Call, m_shadow->omDsv, dsv);

    if (dsv)
      dsv->Release();

    for (UINT i = 0; i < FeShadow::MaxOmUav; i++) {
      blessed::FeShadowVerifyUav(Call, i, m_shadow->omUav[i], uav[i]);

      if (uav[i])
        uav[i]->Release();
    }

    blessed::FeShadowVerifyCallDone();
  }


  // blessed: fe-getters -- answers from FeShadow's OM half when it is known
  // good (docs/research/threaded-frontend.md section 5), instead of
  // draining. BLESSED_FE_SHADOW_VERIFY=1 additionally drains and compares;
  // see blessed_threaded_fe_diag.h.
  void STDMETHODCALLTYPE D3D11ThreadedContext::OMGetRenderTargets(
          UINT                              NumViews,
          ID3D11RenderTargetView**          ppRenderTargetViews,
          ID3D11DepthStencilView**          ppDepthStencilView) {
    FE_ENTRY(OMGetRenderTargets);

    if (unlikely(m_shadow->omUnknown || m_noOmShadow)) {
      Drain(blessed::FeDrain::Getter, blessed::FeCall::OMGetRenderTargets);
      m_ctx->OMGetRenderTargets(NumViews, ppRenderTargetViews, ppDepthStencilView);
      return;
    }

    if (ppRenderTargetViews) {
      for (UINT i = 0; i < NumViews; i++) {
        ppRenderTargetViews[i] = (i < FeShadow::MaxOmRtv && i < m_shadow->omMaxRtv)
          ? ref(m_shadow->omRtv[i]) : nullptr;
      }
    }

    if (ppDepthStencilView)
      *ppDepthStencilView = ref(m_shadow->omDsv);

    if (unlikely(m_verifyOmShadow))
      VerifyOmShadow(blessed::FeCall::OMGetRenderTargets);
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::OMGetRenderTargetsAndUnorderedAccessViews(
          UINT                              NumRTVs,
          ID3D11RenderTargetView**          ppRenderTargetViews,
          ID3D11DepthStencilView**          ppDepthStencilView,
          UINT                              UAVStartSlot,
          UINT                              NumUAVs,
          ID3D11UnorderedAccessView**       ppUnorderedAccessViews) {
    FE_ENTRY(OMGetRenderTargetsAndUnorderedAccessViews);

    if (unlikely(m_shadow->omUnknown || m_noOmShadow)) {
      Drain(blessed::FeDrain::Getter, blessed::FeCall::OMGetRenderTargetsAndUnorderedAccessViews);
      m_ctx->OMGetRenderTargetsAndUnorderedAccessViews(NumRTVs, ppRenderTargetViews,
        ppDepthStencilView, UAVStartSlot, NumUAVs, ppUnorderedAccessViews);
      return;
    }

    if (ppRenderTargetViews) {
      for (UINT i = 0; i < NumRTVs; i++) {
        ppRenderTargetViews[i] = (i < FeShadow::MaxOmRtv && i < m_shadow->omMaxRtv)
          ? ref(m_shadow->omRtv[i]) : nullptr;
      }
    }

    if (ppDepthStencilView)
      *ppDepthStencilView = ref(m_shadow->omDsv);

    if (ppUnorderedAccessViews) {
      // Indexes the array directly, exactly like D3D11CommonContext::
      // OMGetRenderTargetsAndUnorderedAccessViews does against m_state.om.
      // uavs (d3d11_context.cpp:3168): entries outside [omMinUav, omMaxUav)
      // are already null, kept that way by UpdateOmShadow.
      for (UINT i = 0; i < NumUAVs; i++) {
        UINT slot = UAVStartSlot + i;
        ppUnorderedAccessViews[i] = slot < FeShadow::MaxOmUav ? ref(m_shadow->omUav[slot]) : nullptr;
      }
    }

    if (unlikely(m_verifyOmShadow))
      VerifyOmShadow(blessed::FeCall::OMGetRenderTargetsAndUnorderedAccessViews);
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::OMGetBlendState(
          ID3D11BlendState**                ppBlendState,
          FLOAT                             BlendFactor[4],
          UINT*                             pSampleMask) {
    FE_DRAIN(OMGetBlendState, Getter);
    m_ctx->OMGetBlendState(ppBlendState, BlendFactor, pSampleMask);
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::OMGetDepthStencilState(
          ID3D11DepthStencilState**         ppDepthStencilState,
          UINT*                             pStencilRef) {
    FE_DRAIN(OMGetDepthStencilState, Getter);
    m_ctx->OMGetDepthStencilState(ppDepthStencilState, pStencilRef);
  }


  // --- rasterizer ---

  void STDMETHODCALLTYPE D3D11ThreadedContext::RSSetState(ID3D11RasterizerState* pRasterizerState) {
    FE_ENTRY(RSSetState);

    if (likely(m_compact)) {
      if (m_shadow->rs == pRasterizerState) {
        m_folded += 1u;
        return;
      }

      m_shadow->rs = pRasterizerState;
    }

    RecordCall<FeCall_RSSetState>(pRasterizerState);
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::RSSetViewports(
          UINT                              NumViewports,
    const D3D11_VIEWPORT*                   pViewports) {
    FE_ENTRY(RSSetViewports);

    if (unlikely(NumViewports > D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE || (NumViewports && !pViewports))) {
      Drain(blessed::FeDrain::TooMany, blessed::FeCall::RSSetViewports);
      m_ctx->RSSetViewports(NumViewports, pViewports);
      return;
    }

    RecordArray<FeCall_RSSetViewports>(0u, NumViewports, pViewports);
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::RSSetScissorRects(
          UINT                              NumRects,
    const D3D11_RECT*                       pRects) {
    FE_ENTRY(RSSetScissorRects);

    if (unlikely(NumRects > D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE || (NumRects && !pRects))) {
      Drain(blessed::FeDrain::TooMany, blessed::FeCall::RSSetScissorRects);
      m_ctx->RSSetScissorRects(NumRects, pRects);
      return;
    }

    RecordArray<FeCall_RSSetScissorRects>(0u, NumRects, pRects);
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::RSGetState(ID3D11RasterizerState** ppRasterizerState) {
    FE_DRAIN(RSGetState, Getter);
    m_ctx->RSGetState(ppRasterizerState);
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::RSGetViewports(
          UINT*                             pNumViewports,
          D3D11_VIEWPORT*                   pViewports) {
    FE_DRAIN(RSGetViewports, Getter);
    m_ctx->RSGetViewports(pNumViewports, pViewports);
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::RSGetScissorRects(
          UINT*                             pNumRects,
          D3D11_RECT*                       pRects) {
    FE_DRAIN(RSGetScissorRects, Getter);
    m_ctx->RSGetScissorRects(pNumRects, pRects);
  }


  // --- stream output ---

  void STDMETHODCALLTYPE D3D11ThreadedContext::SOSetTargets(
          UINT                              NumBuffers,
          ID3D11Buffer* const*              ppSOTargets,
    const UINT*                             pOffsets) {
    FE_ENTRY(SOSetTargets);

    if (unlikely(NumBuffers > D3D11_SO_BUFFER_SLOT_COUNT || (NumBuffers && !ppSOTargets))) {
      Drain(blessed::FeDrain::TooMany, blessed::FeCall::SOSetTargets);
      m_ctx->SOSetTargets(NumBuffers, ppSOTargets, pOffsets);
      return;
    }

    auto rec = static_cast<FeSoTargetsRec*>(AllocRecord(FeSoTargetsRec::Size(NumBuffers)));
    rec->fn         = &FeSoTargetsRec::Replay;
    rec->count      = NumBuffers;
    rec->hasOffsets = pOffsets != nullptr;

    for (UINT i = 0; i < NumBuffers; i++) {
      rec->buffers()[i] = ppSOTargets[i];
      rec->offsets()[i] = pOffsets ? pOffsets[i] : 0u;
    }

    Recorded();
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::SOGetTargets(
          UINT                              NumBuffers,
          ID3D11Buffer**                    ppSOTargets) {
    FE_DRAIN(SOGetTargets, Getter);
    m_ctx->SOGetTargets(NumBuffers, ppSOTargets);
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::SOGetTargetsWithOffsets(
          UINT                              NumBuffers,
          ID3D11Buffer**                    ppSOTargets,
          UINT*                             pOffsets) {
    FE_DRAIN(SOGetTargets, Getter);
    m_ctx->SOGetTargetsWithOffsets(NumBuffers, ppSOTargets, pOffsets);
  }


  // --- input assembler getters ---

  void STDMETHODCALLTYPE D3D11ThreadedContext::IAGetInputLayout(ID3D11InputLayout** ppInputLayout) {
    FE_DRAIN(IAGetInputLayout, Getter);
    m_ctx->IAGetInputLayout(ppInputLayout);
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::IAGetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY* pTopology) {
    FE_DRAIN(IAGetPrimitiveTopology, Getter);
    m_ctx->IAGetPrimitiveTopology(pTopology);
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::IAGetVertexBuffers(
          UINT                              StartSlot,
          UINT                              NumBuffers,
          ID3D11Buffer**                    ppVertexBuffers,
          UINT*                             pStrides,
          UINT*                             pOffsets) {
    FE_DRAIN(IAGetVertexBuffers, Getter);
    m_ctx->IAGetVertexBuffers(StartSlot, NumBuffers, ppVertexBuffers, pStrides, pOffsets);
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::IAGetIndexBuffer(
          ID3D11Buffer**                    ppIndexBuffer,
          DXGI_FORMAT*                      pFormat,
          UINT*                             pOffset) {
    FE_DRAIN(IAGetIndexBuffer, Getter);
    m_ctx->IAGetIndexBuffer(ppIndexBuffer, pFormat, pOffset);
  }


  // --- predication, min lod ---

  void STDMETHODCALLTYPE D3D11ThreadedContext::SetPredication(
          ID3D11Predicate*                  pPredicate,
          BOOL                              PredicateValue) {
    FE_ENTRY(SetPredication);
    RecordCall<FeCall_SetPredication>(pPredicate, PredicateValue);
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::GetPredication(
          ID3D11Predicate**                 ppPredicate,
          BOOL*                             pPredicateValue) {
    FE_DRAIN(GetPredication, Getter);
    m_ctx->GetPredication(ppPredicate, pPredicateValue);
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::SetResourceMinLOD(
          ID3D11Resource*                   pResource,
          FLOAT                             MinLOD) {
    FE_ENTRY(SetResourceMinLOD);
    RecordCall<FeCall_SetResourceMinLOD>(pResource, MinLOD);
  }


  FLOAT STDMETHODCALLTYPE D3D11ThreadedContext::GetResourceMinLOD(ID3D11Resource* pResource) {
    FE_DRAIN(GetResourceMinLOD, Getter);
    return m_ctx->GetResourceMinLOD(pResource);
  }


  // --- tiled resources: drained, rare ---

  void STDMETHODCALLTYPE D3D11ThreadedContext::CopyTiles(
          ID3D11Resource*                   pTiledResource,
    const D3D11_TILED_RESOURCE_COORDINATE*  pTileRegionStartCoordinate,
    const D3D11_TILE_REGION_SIZE*           pTileRegionSize,
          ID3D11Buffer*                     pBuffer,
          UINT64                            BufferStartOffsetInBytes,
          UINT                              Flags) {
    FE_DRAIN(CopyTiles, Tiles);
    m_ctx->CopyTiles(pTiledResource, pTileRegionStartCoordinate, pTileRegionSize,
      pBuffer, BufferStartOffsetInBytes, Flags);
  }


  HRESULT STDMETHODCALLTYPE D3D11ThreadedContext::CopyTileMappings(
          ID3D11Resource*                   pDestTiledResource,
    const D3D11_TILED_RESOURCE_COORDINATE*  pDestRegionCoordinate,
          ID3D11Resource*                   pSourceTiledResource,
    const D3D11_TILED_RESOURCE_COORDINATE*  pSourceRegionCoordinate,
    const D3D11_TILE_REGION_SIZE*           pTileRegionSize,
          UINT                              Flags) {
    FE_DRAIN(CopyTileMappings, Tiles);
    return m_ctx->CopyTileMappings(pDestTiledResource, pDestRegionCoordinate,
      pSourceTiledResource, pSourceRegionCoordinate, pTileRegionSize, Flags);
  }


  HRESULT STDMETHODCALLTYPE D3D11ThreadedContext::ResizeTilePool(
          ID3D11Buffer*                     pTilePool,
          UINT64                            NewSizeInBytes) {
    FE_DRAIN(ResizeTilePool, Tiles);
    return m_ctx->ResizeTilePool(pTilePool, NewSizeInBytes);
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::TiledResourceBarrier(
          ID3D11DeviceChild*                pTiledResourceOrViewAccessBeforeBarrier,
          ID3D11DeviceChild*                pTiledResourceOrViewAccessAfterBarrier) {
    FE_DRAIN(TiledResourceBarrier, Tiles);
    m_ctx->TiledResourceBarrier(pTiledResourceOrViewAccessBeforeBarrier, pTiledResourceOrViewAccessAfterBarrier);
  }


  HRESULT STDMETHODCALLTYPE D3D11ThreadedContext::UpdateTileMappings(
          ID3D11Resource*                   pTiledResource,
          UINT                              NumRegions,
    const D3D11_TILED_RESOURCE_COORDINATE*  pRegionCoordinates,
    const D3D11_TILE_REGION_SIZE*           pRegionSizes,
          ID3D11Buffer*                     pTilePool,
          UINT                              NumRanges,
    const UINT*                             pRangeFlags,
    const UINT*                             pRangeTileOffsets,
    const UINT*                             pRangeTileCounts,
          UINT                              Flags) {
    FE_DRAIN(UpdateTileMappings, Tiles);
    return m_ctx->UpdateTileMappings(pTiledResource, NumRegions, pRegionCoordinates,
      pRegionSizes, pTilePool, NumRanges, pRangeFlags, pRangeTileOffsets, pRangeTileCounts, Flags);
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::UpdateTiles(
          ID3D11Resource*                   pDestTiledResource,
    const D3D11_TILED_RESOURCE_COORDINATE*  pDestTileRegionStartCoordinate,
    const D3D11_TILE_REGION_SIZE*           pDestTileRegionSize,
    const void*                             pSourceTileData,
          UINT                              Flags) {
    FE_DRAIN(UpdateTiles, Tiles);
    m_ctx->UpdateTiles(pDestTiledResource, pDestTileRegionStartCoordinate,
      pDestTileRegionSize, pSourceTileData, Flags);
  }


  // --- annotations (context2), hardware protection ---

  BOOL STDMETHODCALLTYPE D3D11ThreadedContext::IsAnnotationEnabled() {
    blessed::FeScope feScope(blessed::FeCall::IsAnnotationEnabled);
    return m_annotation.GetStatus();
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::SetMarkerInt(
          LPCWSTR                           pLabel,
          INT                               Data) {
    // a no-op in dxvk
    blessed::FeScope feScope(blessed::FeCall::SetMarkerInt);
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::BeginEventInt(
          LPCWSTR                           pLabel,
          INT                               Data) {
    // a no-op in dxvk
    blessed::FeScope feScope(blessed::FeCall::BeginEventInt);
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::EndEvent() {
    // a no-op in dxvk
    blessed::FeScope feScope(blessed::FeCall::EndEvent);
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::GetHardwareProtectionState(
          BOOL*                             pHwProtectionEnable) {
    FE_DRAIN(GetHardwareProtectionState, Getter);
    m_ctx->GetHardwareProtectionState(pHwProtectionEnable);
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::SetHardwareProtectionState(
          BOOL                              HwProtectionEnable) {
    FE_DRAIN(SetHardwareProtectionState, Other);
    m_ctx->SetHardwareProtectionState(HwProtectionEnable);
  }


  // --- queries: state on the game thread, gpu work recorded ---

  HRESULT STDMETHODCALLTYPE D3D11ThreadedContext::GetData(
          ID3D11Asynchronous*               pAsync,
          void*                             pData,
          UINT                              DataSize,
          UINT                              GetDataFlags) {
    FE_ENTRY(GetData);

    if (!pAsync || (DataSize && !pData))
      return E_INVALIDARG;

    if (DataSize && DataSize != pAsync->GetDataSize())
      return E_INVALIDARG;

    pData = DataSize ? pData : nullptr;

    // DoEnd ran game-side, so the state is current: S_FALSE until the cs
    // thread has executed the End, whatever the front end's lag
    auto query = static_cast<D3D11Query*>(pAsync);
    HRESULT hr = query->GetData(pData, GetDataFlags);

    if (hr == S_FALSE) {
      if (!(GetDataFlags & D3D11_ASYNC_GETDATA_DONOTFLUSH))
        query->NotifyStall();

      // Keep the gpu busy while the app spins; one hint per spin run.
      // Publish now: the End being waited for may still be unpublished.
      if (m_flushHint != m_writePos) {
        Push<FeFlushHintRec>();
        m_flushHint = m_writePos;
      }

      Publish();
    }

    return hr;
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::Begin(ID3D11Asynchronous* pAsync) {
    FE_ENTRY(Begin);

    if (unlikely(!pAsync))
      return;

    auto query = static_cast<D3D11Query*>(pAsync);

    if (unlikely(!query->DoBegin()))
      return;

    Push<FeQueryBeginRec>()->query = query;
    Recorded();
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::End(ID3D11Asynchronous* pAsync) {
    FE_ENTRY(End);

    if (unlikely(!pAsync))
      return;

    auto query = static_cast<D3D11Query*>(pAsync);

    // Query state and the stall heuristic, game-side: GetData reads the
    // same fields from this thread
    bool began = query->DoEnd();
    uint32_t flush = 0u;

    if (unlikely(query->TrackStalls())) {
      query->NotifyEnd();

      if (query->IsStalling())
        flush = 1u;
      else if (query->IsEvent())
        flush = 2u;
    }

    auto rec = Push<FeQueryEndRec>();
    rec->query      = query;
    rec->needsBegin = !began;
    rec->flush      = flush;
    Recorded();
  }


  // --- flushes, fences, command lists ---

  void STDMETHODCALLTYPE D3D11ThreadedContext::Flush() {
    FE_ENTRY(Flush);
    RecordCall<FeCall_Flush>();
    Publish();
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::Flush1(
          D3D11_CONTEXT_TYPE          ContextType,
          HANDLE                      hEvent) {
    FE_ENTRY(Flush1);
    RecordCall<FeCall_Flush1>(ContextType, hEvent);
    Publish();
  }


  HRESULT STDMETHODCALLTYPE D3D11ThreadedContext::Signal(
          ID3D11Fence*                pFence,
          UINT64                      Value) {
    FE_ENTRY(Signal);

    if (!pFence)
      return E_INVALIDARG;

    RecordCall<FeCall_Signal>(pFence, Value);
    Publish();
    return S_OK;
  }


  HRESULT STDMETHODCALLTYPE D3D11ThreadedContext::Wait(
          ID3D11Fence*                pFence,
          UINT64                      Value) {
    FE_ENTRY(Wait);

    if (!pFence)
      return E_INVALIDARG;

    RecordCall<FeCall_Wait>(pFence, Value);
    Publish();
    return S_OK;
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::ExecuteCommandList(
          ID3D11CommandList*  pCommandList,
          BOOL                RestoreContextState) {
    FE_ENTRY(ExecuteCommandList);
    RecordCall<FeCall_ExecuteCommandList>(pCommandList, RestoreContextState);
    InvalidateShadow();
    Publish();
  }


  HRESULT STDMETHODCALLTYPE D3D11ThreadedContext::FinishCommandList(
          BOOL                RestoreDeferredContextState,
          ID3D11CommandList   **ppCommandList) {
    // an error on an immediate context; touches no state
    blessed::FeScope feScope(blessed::FeCall::FinishCommandList);
    return m_ctx->FinishCommandList(RestoreDeferredContextState, ppCommandList);
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::SwapDeviceContextState(
          ID3DDeviceContextState*           pState,
          ID3DDeviceContextState**          ppPreviousState) {
    FE_DRAIN(SwapDeviceContextState, StateSwap);
    m_ctx->SwapDeviceContextState(pState, ppPreviousState);
    InvalidateShadow();
  }


  // --- maps: buffers answered game-side, the rest drained ---

  HRESULT STDMETHODCALLTYPE D3D11ThreadedContext::Map(
          ID3D11Resource*             pResource,
          UINT                        Subresource,
          D3D11_MAP                   MapType,
          UINT                        MapFlags,
          D3D11_MAPPED_SUBRESOURCE*   pMappedResource) {
    FE_ENTRY(Map);

    if (unlikely(!pResource))
      return E_INVALIDARG;

    D3D11_RESOURCE_DIMENSION resourceDim = D3D11_RESOURCE_DIMENSION_UNKNOWN;
    pResource->GetType(&resourceDim);

    bool isBuffer = resourceDim == D3D11_RESOURCE_DIMENSION_BUFFER;

    if (likely(isBuffer && (MapType == D3D11_MAP_WRITE_DISCARD || MapType == D3D11_MAP_WRITE_NO_OVERWRITE)
     && !BlessedDump::IsCapturing()))
      return MapBuffer(static_cast<D3D11Buffer*>(pResource), MapType, MapFlags, pMappedResource);

    // Reads, plain writes, textures, and every map while the dump logs
    // maps in call order: dxvk's own Map, on this thread, once idle
    Drain(BlessedDump::IsCapturing() ? blessed::FeDrain::Capture : blessed::FeDrain::MapOther);

    HRESULT hr = m_ctx->Map(pResource, Subresource, MapType, MapFlags, pMappedResource);

    if (isBuffer)
      SyncAppMapPtr(pResource);
    else
      m_imageMaps = m_ctx->m_mappedImageCount;

    return hr;
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::Unmap(
          ID3D11Resource*             pResource,
          UINT                        Subresource) {
    FE_ENTRY(Unmap);

    // Buffer unmaps do nothing in dxvk. Image maps only happen on the
    // drained path, so the count is this thread's to keep: a copy here,
    // off the real context's lines the front end writes all the time.
    if (unlikely(m_imageMaps > 0)) {
      D3D11_RESOURCE_DIMENSION resourceDim = D3D11_RESOURCE_DIMENSION_UNKNOWN;
      pResource->GetType(&resourceDim);

      if (resourceDim != D3D11_RESOURCE_DIMENSION_BUFFER) {
        Drain(blessed::FeDrain::UnmapImage, blessed::FeCall::Unmap);
        m_ctx->Unmap(pResource, Subresource);
        m_imageMaps = m_ctx->m_mappedImageCount;
      }
    }
  }


  HRESULT D3D11ThreadedContext::MapBuffer(
          D3D11Buffer*                pBuffer,
          D3D11_MAP                   MapType,
          UINT                        MapFlags,
          D3D11_MAPPED_SUBRESOURCE*   pMappedResource) {
    // Same probe buckets as D3D11ImmediateContext::MapBuffer, now timed
    // on the game thread
    blessed::MapType blessedMapType = blessed::MapType::Other;
    blessed::BindKind blessedBindKind = blessed::BindKind::Other;

    if (unlikely(blessed::active())) {
      blessedMapType = MapType == D3D11_MAP_WRITE_DISCARD
        ? (pBuffer->BlessedUsesCbRing() ? blessed::MapType::Ring : blessed::MapType::Discard)
        : blessed::MapType::NoOverwrite;
      blessedBindKind = blessed::classifyBindFlags(pBuffer->Desc()->BindFlags);
    }

    blessed::MapScope blessedProbe_MapBuffer(blessedMapType, blessedBindKind);

    if (unlikely(!pMappedResource))
      return E_INVALIDARG;

    if (unlikely(pBuffer->GetMapMode() == D3D11_COMMON_BUFFER_MAP_MODE_NONE)) {
      Logger::err("D3D11: Cannot map a device-local buffer");
      pMappedResource->pData = nullptr;
      return E_INVALIDARG;
    }

    UINT bufferSize = pBuffer->Desc()->ByteWidth;
    void* mapPtr = nullptr;

    if (likely(MapType == D3D11_MAP_WRITE_DISCARD)) {
      // cb-ring: the ring is game-side by design (blessed_cb_ring.h);
      // the rename is recorded and applied on replay
      if (pBuffer->BlessedUsesCbRing()) {
        BlessedCbRingChunk chunk;

        if (likely(m_ctx->m_blessedCbRing.alloc(pBuffer->BlessedBufferPtr(), bufferSize, &chunk))) {
          pBuffer->BlessedSetAppMapPtr(chunk.mapPtr);

          if (likely(m_compact && chunk.offset <= VkDeviceSize(~0u))) {
            // the map joins the draw packet its bind and draw go into
            auto op = PushOp<FeOpCbRename>();
            op->h      = { FeOp::CbRename, FeStage::VS, 0u, uint32_t(chunk.offset) };
            op->buffer = pBuffer;
            op->block  = chunk.block;
            // blessed: cb-mirror -- deliberately not carried in the op
            // itself, see the comment on FeOpCbRename (blessed_threaded_packet.h).
            // blockRetired instead sets a flag directly on the open packet's
            // own header (m_packet), never through PushOp: interleaving a
            // second, independent record here would break the fast path's
            // pos == m_packetEnd contiguity this packet's later ops rely on.
            if (unlikely(chunk.blockRetired && m_ctx->m_blessedCbRing.mirrorEnabled()))
              m_packet->mirrorEarlyFlush = true;
          } else {
            auto rec = Push<FeCbRenameRec>();
            rec->buffer = pBuffer;
            rec->chunk  = chunk;
          }

          Recorded();

          mapPtr = chunk.mapPtr;
        } else {
          blessed::CbRingScope blessedProbe_Fallback(blessed::CbRingEvent::Fallback);
          blessedProbe_MapBuffer.setType(blessed::MapType::Discard);
        }
      }

      if (!mapPtr)
        mapPtr = DiscardBuffer(pBuffer);
    } else {
      // Every rename starts on this thread, so the app side is current
      mapPtr = pBuffer->BlessedAppMapPtr();
    }

    pMappedResource->pData      = mapPtr;
    pMappedResource->RowPitch   = bufferSize;
    pMappedResource->DepthPitch = bufferSize;
    return S_OK;
  }


  void* D3D11ThreadedContext::DiscardBuffer(D3D11Buffer* pBuffer) {
    // The allocation is thread-safe; the context's local cache is only
    // ever used from this thread while the front end runs, since every
    // path that reaches it (buffer maps, mapped-buffer updates) is
    // answered here or drained.
    Rc<DxvkResourceAllocation> slice;

    { blessed::DiscardSliceScope blessedProbe_DiscardSlice;
      slice = pBuffer->GetBuffer()->allocateStorage(&m_ctx->m_allocationCache);
    }

    void* mapPtr = slice->mapPtr();
    pBuffer->BlessedSetAppMapPtr(mapPtr);

    auto rec = Push<FeRenameRec>();
    rec->buffer = pBuffer;
    rec->size   = pBuffer->Desc()->ByteWidth;
    new (rec->allocation) Rc<DxvkResourceAllocation>(std::move(slice));
    Recorded();

    return mapPtr;
  }


  void D3D11ThreadedContext::SyncAppMapPtr(ID3D11Resource* pResource) {
    // After dxvk's own code ran on this thread with the front end idle,
    // the replay side is the truth; the app side follows it
    D3D11_RESOURCE_DIMENSION resourceDim = D3D11_RESOURCE_DIMENSION_UNKNOWN;
    pResource->GetType(&resourceDim);

    if (resourceDim == D3D11_RESOURCE_DIMENSION_BUFFER) {
      auto buffer = static_cast<D3D11Buffer*>(pResource);
      buffer->BlessedSetAppMapPtr(buffer->GetMapPtr());
    }
  }


  // --- annotation interface ---

  ULONG STDMETHODCALLTYPE D3D11ThreadedContext::Annotation::AddRef() {
    return m_parent->AddRef();
  }


  ULONG STDMETHODCALLTYPE D3D11ThreadedContext::Annotation::Release() {
    return m_parent->Release();
  }


  HRESULT STDMETHODCALLTYPE D3D11ThreadedContext::Annotation::QueryInterface(REFIID riid, void** ppvObject) {
    return m_parent->QueryInterface(riid, ppvObject);
  }


  INT STDMETHODCALLTYPE D3D11ThreadedContext::Annotation::BeginEvent(
          D3DCOLOR                Color,
          LPCWSTR                 Name) {
    blessed::FeScope feScope(blessed::FeCall::AnnotationBeginEvent);

    if (!m_enabled || !Name)
      return -1;

    D3D10DeviceLock lock = m_parent->LockContext();
    m_parent->RecordAnnotation(0u, Color, Name);
    return m_eventDepth++;
  }


  INT STDMETHODCALLTYPE D3D11ThreadedContext::Annotation::EndEvent() {
    blessed::FeScope feScope(blessed::FeCall::AnnotationEndEvent);

    if (!m_enabled)
      return -1;

    D3D10DeviceLock lock = m_parent->LockContext();

    if (!m_eventDepth)
      return 0;

    m_parent->RecordAnnotation(1u, 0u, nullptr);
    return --m_eventDepth;
  }


  void STDMETHODCALLTYPE D3D11ThreadedContext::Annotation::SetMarker(
          D3DCOLOR                Color,
          LPCWSTR                 Name) {
    blessed::FeScope feScope(blessed::FeCall::AnnotationSetMarker);

    if (!m_enabled || !Name)
      return;

    D3D10DeviceLock lock = m_parent->LockContext();
    m_parent->RecordAnnotation(2u, Color, Name);
  }


  BOOL STDMETHODCALLTYPE D3D11ThreadedContext::Annotation::GetStatus() {
    return m_enabled;
  }


  void D3D11ThreadedContext::RecordAnnotation(
          uint32_t                Kind,
          D3DCOLOR                Color,
          LPCWSTR                 Name) {
    // Labels are capped; a longer one is cut, not dropped
    constexpr uint32_t MaxLength = 1024u;

    uint32_t length = 0u;

    if (Name) {
      while (length < MaxLength - 1u && Name[length])
        length += 1u;
    }

    auto rec = static_cast<FeAnnotationRec*>(AllocRecord(FeAnnotationRec::Size(length + 1u)));
    rec->fn     = &FeAnnotationRec::Replay;
    rec->color  = Color;
    rec->kind   = Kind;
    rec->length = length + 1u;
    rec->pad    = 0u;

    auto label = reinterpret_cast<WCHAR*>(rec + 1);

    for (uint32_t i = 0; i < length; i++)
      label[i] = Name[i];

    label[length] = 0;
    Recorded();
  }

}
