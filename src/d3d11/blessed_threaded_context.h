// blessed: threaded-fe -- the threaded d3d11 front end: a recording facade as the app's immediate context, the ring, and the thread that replays it
#pragma once

#include <atomic>
#include <vector>

#include "d3d11_context_imm.h"

#include "blessed_threaded_release.h"

#include "../util/util_blessed_probe.h"

#include "../util/sync/sync_spinlock.h"

namespace dxvk {

  class D3D11SwapChain;

  struct FePacketRec; // blessed: threaded-fe-2, see blessed_threaded_packet.h
  struct FeShadow;
  struct BlessedCrashWriter; // blessed: fe-crash-2, see blessed_crash_log.h
  enum class FeOp : uint8_t;
  enum class FeStage : uint8_t;

  /**
   * \brief blessed: the probe's frame boundary, game side
   *
   * Hands the cs thread's counters to \c blessed::onPresent. Called once
   * per app Present on the recording thread, whichever thread then runs
   * the Present body. See blessed_threaded_present.cpp.
   */
  void BlessedProbePresent(DxvkDevice* pDevice);

  /**
   * \brief Replays one record
   *
   * \param [in] pContext The real immediate context
   * \param [in] pRecord The record, 8-byte aligned
   * \returns Size of the record in bytes, or 0 for the pad record
   *    that sends the reader back to the start of the ring
   */
  using BlessedFeThunk = size_t (*)(D3D11ImmediateContext* pContext, void* pRecord);

  /**
   * \brief The threaded front end (d3d11.blessedThreadedFrontEnd)
   *
   * Handed to the app as its immediate context. Every call does one of
   * three things (design: docs/research/threaded-frontend.md, section 1):
   *
   * - recorded: draws, dispatches, setters, clears, copies, flushes,
   *   non-mappable UpdateSubresource, the gpu half of queries, and final
   *   releases. A record is a thunk pointer and its arguments, written
   *   into a 4 MiB ring and replayed later against the unchanged
   *   \c D3D11ImmediateContext.
   * - answered on the game thread: buffer Map(WRITE_DISCARD) and
   *   Map(WRITE_NO_OVERWRITE), Unmap on buffers, UpdateSubresource and
   *   DiscardResource on mappable buffers, query Begin/End state and
   *   GetData. A discard allocates here and records the rename.
   * - drained: everything else. The game thread publishes, waits until
   *   the front end has replayed every record, then runs dxvk's own
   *   method itself. That includes Present (stage 1) and every Get*.
   *
   * Replay runs on the front end thread (stage 1), or in loopback mode
   * (stage 0, d3d11.blessedThreadedFrontEndLoopback) on the recording
   * thread itself at every publish, which prices the recorder and keeps
   * the frame bit-identical without a second thread.
   *
   * Threads: the producer side (everything below that is not marked
   * front end) needs no thread identity, only one caller at a time,
   * which d3d11 already demands of an immediate context. Final releases
   * come from any thread through a locked list.
   */
  class D3D11ThreadedContext final : public D3D11DeviceChild<ID3D11DeviceContext4> {

  public:

    /// Ring size. Records never straddle its end.
    constexpr static size_t RingSize = size_t(4u) << 20;

    /// Largest UpdateSubresource payload copied into the ring itself
    constexpr static size_t MaxInlineData = size_t(64u) << 10;

    /// Calls between publishes. Draws publish too unless records are
    /// compact (d3d11.blessedFrontEndCompact, stage 3).
    constexpr static uint32_t PublishEvery = 32u;

    /// Calls between publishes with compact records: a packet stays open
    /// until the next publish, and each publish is an sfence and a store
    /// to a line the front end polls
    constexpr static uint32_t PublishEveryCompact = 128u;

    /// Largest array a setter records; larger ones drain
    constexpr static uint32_t MaxArray = 128u;

    D3D11ThreadedContext(
            D3D11Device*            pParent,
            D3D11ImmediateContext*  pContext,
            bool                    Loopback);

    ~D3D11ThreadedContext();

    ULONG STDMETHODCALLTYPE AddRef();

    ULONG STDMETHODCALLTYPE Release();

    HRESULT STDMETHODCALLTYPE QueryInterface(
            REFIID  riid,
            void**  ppvObject);

    D3D11_DEVICE_CONTEXT_TYPE STDMETHODCALLTYPE GetType();

    UINT STDMETHODCALLTYPE GetContextFlags();

    void STDMETHODCALLTYPE ClearState();

    void STDMETHODCALLTYPE DiscardResource(ID3D11Resource *pResource);

    void STDMETHODCALLTYPE DiscardView(ID3D11View* pResourceView);

    void STDMETHODCALLTYPE DiscardView1(
            ID3D11View*                      pResourceView,
      const D3D11_RECT*                      pRects,
            UINT                             NumRects);

    void STDMETHODCALLTYPE CopySubresourceRegion(
            ID3D11Resource*                   pDstResource,
            UINT                              DstSubresource,
            UINT                              DstX,
            UINT                              DstY,
            UINT                              DstZ,
            ID3D11Resource*                   pSrcResource,
            UINT                              SrcSubresource,
      const D3D11_BOX*                        pSrcBox);

    void STDMETHODCALLTYPE CopySubresourceRegion1(
            ID3D11Resource*                   pDstResource,
            UINT                              DstSubresource,
            UINT                              DstX,
            UINT                              DstY,
            UINT                              DstZ,
            ID3D11Resource*                   pSrcResource,
            UINT                              SrcSubresource,
      const D3D11_BOX*                        pSrcBox,
            UINT                              CopyFlags);

    void STDMETHODCALLTYPE CopyResource(
            ID3D11Resource*                   pDstResource,
            ID3D11Resource*                   pSrcResource);

    void STDMETHODCALLTYPE CopyStructureCount(
            ID3D11Buffer*                     pDstBuffer,
            UINT                              DstAlignedByteOffset,
            ID3D11UnorderedAccessView*        pSrcView);

    void STDMETHODCALLTYPE ClearRenderTargetView(
            ID3D11RenderTargetView*           pRenderTargetView,
      const FLOAT                             ColorRGBA[4]);

    void STDMETHODCALLTYPE ClearUnorderedAccessViewUint(
            ID3D11UnorderedAccessView*        pUnorderedAccessView,
      const UINT                              Values[4]);

    void STDMETHODCALLTYPE ClearUnorderedAccessViewFloat(
            ID3D11UnorderedAccessView*        pUnorderedAccessView,
      const FLOAT                             Values[4]);

    void STDMETHODCALLTYPE ClearDepthStencilView(
            ID3D11DepthStencilView*           pDepthStencilView,
            UINT                              ClearFlags,
            FLOAT                             Depth,
            UINT8                             Stencil);

    void STDMETHODCALLTYPE ClearView(
            ID3D11View                        *pView,
      const FLOAT                             Color[4],
      const D3D11_RECT                        *pRect,
            UINT                              NumRects);

    void STDMETHODCALLTYPE GenerateMips(
            ID3D11ShaderResourceView*         pShaderResourceView);

    void STDMETHODCALLTYPE ResolveSubresource(
            ID3D11Resource*                   pDstResource,
            UINT                              DstSubresource,
            ID3D11Resource*                   pSrcResource,
            UINT                              SrcSubresource,
            DXGI_FORMAT                       Format);

    void STDMETHODCALLTYPE UpdateSubresource(
            ID3D11Resource*                   pDstResource,
            UINT                              DstSubresource,
      const D3D11_BOX*                        pDstBox,
      const void*                             pSrcData,
            UINT                              SrcRowPitch,
            UINT                              SrcDepthPitch);

    void STDMETHODCALLTYPE UpdateSubresource1(
            ID3D11Resource*                   pDstResource,
            UINT                              DstSubresource,
      const D3D11_BOX*                        pDstBox,
      const void*                             pSrcData,
            UINT                              SrcRowPitch,
            UINT                              SrcDepthPitch,
            UINT                              CopyFlags);

    void STDMETHODCALLTYPE DrawAuto();

    void STDMETHODCALLTYPE Draw(
            UINT            VertexCount,
            UINT            StartVertexLocation);

    void STDMETHODCALLTYPE DrawIndexed(
            UINT            IndexCount,
            UINT            StartIndexLocation,
            INT             BaseVertexLocation);

    void STDMETHODCALLTYPE DrawInstanced(
            UINT            VertexCountPerInstance,
            UINT            InstanceCount,
            UINT            StartVertexLocation,
            UINT            StartInstanceLocation);

    void STDMETHODCALLTYPE DrawIndexedInstanced(
            UINT            IndexCountPerInstance,
            UINT            InstanceCount,
            UINT            StartIndexLocation,
            INT             BaseVertexLocation,
            UINT            StartInstanceLocation);

    void STDMETHODCALLTYPE DrawIndexedInstancedIndirect(
            ID3D11Buffer*   pBufferForArgs,
            UINT            AlignedByteOffsetForArgs);

    void STDMETHODCALLTYPE DrawInstancedIndirect(
            ID3D11Buffer*   pBufferForArgs,
            UINT            AlignedByteOffsetForArgs);

    void STDMETHODCALLTYPE Dispatch(
            UINT            ThreadGroupCountX,
            UINT            ThreadGroupCountY,
            UINT            ThreadGroupCountZ);

    void STDMETHODCALLTYPE DispatchIndirect(
            ID3D11Buffer*   pBufferForArgs,
            UINT            AlignedByteOffsetForArgs);

    void STDMETHODCALLTYPE IASetInputLayout(
            ID3D11InputLayout*                pInputLayout);

    void STDMETHODCALLTYPE IASetPrimitiveTopology(
            D3D11_PRIMITIVE_TOPOLOGY          Topology);

    void STDMETHODCALLTYPE IASetVertexBuffers(
            UINT                              StartSlot,
            UINT                              NumBuffers,
            ID3D11Buffer* const*              ppVertexBuffers,
      const UINT*                             pStrides,
      const UINT*                             pOffsets);

    void STDMETHODCALLTYPE IASetIndexBuffer(
            ID3D11Buffer*                     pIndexBuffer,
            DXGI_FORMAT                       Format,
            UINT                              Offset);

    void STDMETHODCALLTYPE IAGetInputLayout(
            ID3D11InputLayout**               ppInputLayout);

    void STDMETHODCALLTYPE IAGetPrimitiveTopology(
            D3D11_PRIMITIVE_TOPOLOGY*         pTopology);

    void STDMETHODCALLTYPE IAGetVertexBuffers(
            UINT                              StartSlot,
            UINT                              NumBuffers,
            ID3D11Buffer**                    ppVertexBuffers,
            UINT*                             pStrides,
            UINT*                             pOffsets);

    void STDMETHODCALLTYPE IAGetIndexBuffer(
            ID3D11Buffer**                    ppIndexBuffer,
            DXGI_FORMAT*                      pFormat,
            UINT*                             pOffset);

    void STDMETHODCALLTYPE VSSetShader(
            ID3D11VertexShader*               pVertexShader,
            ID3D11ClassInstance* const*       ppClassInstances,
            UINT                              NumClassInstances);

    void STDMETHODCALLTYPE VSSetConstantBuffers(
            UINT                              StartSlot,
            UINT                              NumBuffers,
            ID3D11Buffer* const*              ppConstantBuffers);

     void STDMETHODCALLTYPE VSSetConstantBuffers1(
            UINT                              StartSlot,
            UINT                              NumBuffers,
            ID3D11Buffer* const*              ppConstantBuffers,
      const UINT*                             pFirstConstant,
      const UINT*                             pNumConstants);

    void STDMETHODCALLTYPE VSSetShaderResources(
            UINT                              StartSlot,
            UINT                              NumViews,
            ID3D11ShaderResourceView* const*  ppShaderResourceViews);

    void STDMETHODCALLTYPE VSSetSamplers(
            UINT                              StartSlot,
            UINT                              NumSamplers,
            ID3D11SamplerState* const*        ppSamplers);

    void STDMETHODCALLTYPE VSGetShader(
            ID3D11VertexShader**              ppVertexShader,
            ID3D11ClassInstance**             ppClassInstances,
            UINT*                             pNumClassInstances);

    void STDMETHODCALLTYPE VSGetConstantBuffers(
            UINT                              StartSlot,
            UINT                              NumBuffers,
            ID3D11Buffer**                    ppConstantBuffers);

    void STDMETHODCALLTYPE VSGetConstantBuffers1(
            UINT                              StartSlot,
            UINT                              NumBuffers,
            ID3D11Buffer**                    ppConstantBuffers,
            UINT*                             pFirstConstant,
            UINT*                             pNumConstants);

    void STDMETHODCALLTYPE VSGetShaderResources(
            UINT                              StartSlot,
            UINT                              NumViews,
            ID3D11ShaderResourceView**        ppShaderResourceViews);

    void STDMETHODCALLTYPE VSGetSamplers(
            UINT                              StartSlot,
            UINT                              NumSamplers,
            ID3D11SamplerState**              ppSamplers);

    void STDMETHODCALLTYPE HSSetShader(
            ID3D11HullShader*                 pHullShader,
            ID3D11ClassInstance* const*       ppClassInstances,
            UINT                              NumClassInstances);

    void STDMETHODCALLTYPE HSSetConstantBuffers(
            UINT                              StartSlot,
            UINT                              NumBuffers,
            ID3D11Buffer* const*              ppConstantBuffers);

    void STDMETHODCALLTYPE HSSetConstantBuffers1(
            UINT                              StartSlot,
            UINT                              NumBuffers,
            ID3D11Buffer* const*              ppConstantBuffers,
      const UINT*                             pFirstConstant,
      const UINT*                             pNumConstants);

    void STDMETHODCALLTYPE HSSetShaderResources(
            UINT                              StartSlot,
            UINT                              NumViews,
            ID3D11ShaderResourceView* const*  ppShaderResourceViews);

    void STDMETHODCALLTYPE HSSetSamplers(
            UINT                              StartSlot,
            UINT                              NumSamplers,
            ID3D11SamplerState* const*        ppSamplers);

    void STDMETHODCALLTYPE HSGetShader(
            ID3D11HullShader**                ppHullShader,
            ID3D11ClassInstance**             ppClassInstances,
            UINT*                             pNumClassInstances);

    void STDMETHODCALLTYPE HSGetConstantBuffers(
            UINT                              StartSlot,
            UINT                              NumBuffers,
            ID3D11Buffer**                    ppConstantBuffers);

     void STDMETHODCALLTYPE HSGetConstantBuffers1(
            UINT                              StartSlot,
            UINT                              NumBuffers,
            ID3D11Buffer**                    ppConstantBuffers,
            UINT*                             pFirstConstant,
            UINT*                             pNumConstants);

    void STDMETHODCALLTYPE HSGetShaderResources(
            UINT                              StartSlot,
            UINT                              NumViews,
            ID3D11ShaderResourceView**        ppShaderResourceViews);

    void STDMETHODCALLTYPE HSGetSamplers(
            UINT                              StartSlot,
            UINT                              NumSamplers,
            ID3D11SamplerState**              ppSamplers);

    void STDMETHODCALLTYPE DSSetShader(
            ID3D11DomainShader*               pDomainShader,
            ID3D11ClassInstance* const*       ppClassInstances,
            UINT                              NumClassInstances);

    void STDMETHODCALLTYPE DSSetConstantBuffers(
            UINT                              StartSlot,
            UINT                              NumBuffers,
            ID3D11Buffer* const*              ppConstantBuffers);

    void STDMETHODCALLTYPE DSSetConstantBuffers1(
            UINT                              StartSlot,
            UINT                              NumBuffers,
            ID3D11Buffer* const*              ppConstantBuffers,
      const UINT*                             pFirstConstant,
      const UINT*                             pNumConstants);

    void STDMETHODCALLTYPE DSSetShaderResources(
            UINT                              StartSlot,
            UINT                              NumViews,
            ID3D11ShaderResourceView* const*  ppShaderResourceViews);

    void STDMETHODCALLTYPE DSSetSamplers(
            UINT                              StartSlot,
            UINT                              NumSamplers,
            ID3D11SamplerState* const*        ppSamplers);

    void STDMETHODCALLTYPE DSGetShader(
            ID3D11DomainShader**              ppDomainShader,
            ID3D11ClassInstance**             ppClassInstances,
            UINT*                             pNumClassInstances);

    void STDMETHODCALLTYPE DSGetConstantBuffers(
            UINT                              StartSlot,
            UINT                              NumBuffers,
            ID3D11Buffer**                    ppConstantBuffers);

     void STDMETHODCALLTYPE DSGetConstantBuffers1(
            UINT                              StartSlot,
            UINT                              NumBuffers,
            ID3D11Buffer**                    ppConstantBuffers,
            UINT*                             pFirstConstant,
            UINT*                             pNumConstants);

    void STDMETHODCALLTYPE DSGetShaderResources(
            UINT                              StartSlot,
            UINT                              NumViews,
            ID3D11ShaderResourceView**        ppShaderResourceViews);

    void STDMETHODCALLTYPE DSGetSamplers(
            UINT                              StartSlot,
            UINT                              NumSamplers,
            ID3D11SamplerState**              ppSamplers);

    void STDMETHODCALLTYPE GSSetShader(
            ID3D11GeometryShader*             pShader,
            ID3D11ClassInstance* const*       ppClassInstances,
            UINT                              NumClassInstances);

    void STDMETHODCALLTYPE GSSetConstantBuffers(
            UINT                              StartSlot,
            UINT                              NumBuffers,
            ID3D11Buffer* const*              ppConstantBuffers);

    void STDMETHODCALLTYPE GSSetConstantBuffers1(
            UINT                              StartSlot,
            UINT                              NumBuffers,
            ID3D11Buffer* const*              ppConstantBuffers,
      const UINT*                             pFirstConstant,
      const UINT*                             pNumConstants);

    void STDMETHODCALLTYPE GSSetShaderResources(
            UINT                              StartSlot,
            UINT                              NumViews,
            ID3D11ShaderResourceView* const*  ppShaderResourceViews);

    void STDMETHODCALLTYPE GSSetSamplers(
            UINT                              StartSlot,
            UINT                              NumSamplers,
            ID3D11SamplerState* const*        ppSamplers);

    void STDMETHODCALLTYPE GSGetShader(
            ID3D11GeometryShader**            ppGeometryShader,
            ID3D11ClassInstance**             ppClassInstances,
            UINT*                             pNumClassInstances);

    void STDMETHODCALLTYPE GSGetConstantBuffers(
            UINT                              StartSlot,
            UINT                              NumBuffers,
            ID3D11Buffer**                    ppConstantBuffers);

     void STDMETHODCALLTYPE GSGetConstantBuffers1(
            UINT                              StartSlot,
            UINT                              NumBuffers,
            ID3D11Buffer**                    ppConstantBuffers,
            UINT*                             pFirstConstant,
            UINT*                             pNumConstants);

    void STDMETHODCALLTYPE GSGetShaderResources(
            UINT                              StartSlot,
            UINT                              NumViews,
            ID3D11ShaderResourceView**        ppShaderResourceViews);

    void STDMETHODCALLTYPE GSGetSamplers(
            UINT                              StartSlot,
            UINT                              NumSamplers,
            ID3D11SamplerState**              ppSamplers);

    void STDMETHODCALLTYPE PSSetShader(
            ID3D11PixelShader*                pPixelShader,
            ID3D11ClassInstance* const*       ppClassInstances,
            UINT                              NumClassInstances);

    void STDMETHODCALLTYPE PSSetConstantBuffers(
            UINT                              StartSlot,
            UINT                              NumBuffers,
            ID3D11Buffer* const*              ppConstantBuffers);

    void STDMETHODCALLTYPE PSSetConstantBuffers1(
            UINT                              StartSlot,
            UINT                              NumBuffers,
            ID3D11Buffer* const*              ppConstantBuffers,
      const UINT*                             pFirstConstant,
      const UINT*                             pNumConstants);

    void STDMETHODCALLTYPE PSSetShaderResources(
            UINT                              StartSlot,
            UINT                              NumViews,
            ID3D11ShaderResourceView* const*  ppShaderResourceViews);

    void STDMETHODCALLTYPE PSSetSamplers(
            UINT                              StartSlot,
            UINT                              NumSamplers,
            ID3D11SamplerState* const*        ppSamplers);

    void STDMETHODCALLTYPE PSGetShader(
            ID3D11PixelShader**               ppPixelShader,
            ID3D11ClassInstance**             ppClassInstances,
            UINT*                             pNumClassInstances);

    void STDMETHODCALLTYPE PSGetConstantBuffers(
            UINT                              StartSlot,
            UINT                              NumBuffers,
            ID3D11Buffer**                    ppConstantBuffers);

    void STDMETHODCALLTYPE PSGetConstantBuffers1(
            UINT                              StartSlot,
            UINT                              NumBuffers,
            ID3D11Buffer**                    ppConstantBuffers,
            UINT*                             pFirstConstant,
            UINT*                             pNumConstants);

    void STDMETHODCALLTYPE PSGetShaderResources(
            UINT                              StartSlot,
            UINT                              NumViews,
            ID3D11ShaderResourceView**        ppShaderResourceViews);

    void STDMETHODCALLTYPE PSGetSamplers(
            UINT                              StartSlot,
            UINT                              NumSamplers,
            ID3D11SamplerState**              ppSamplers);

    void STDMETHODCALLTYPE CSSetShader(
            ID3D11ComputeShader*              pComputeShader,
            ID3D11ClassInstance* const*       ppClassInstances,
            UINT                              NumClassInstances);

    void STDMETHODCALLTYPE CSSetConstantBuffers(
            UINT                              StartSlot,
            UINT                              NumBuffers,
            ID3D11Buffer* const*              ppConstantBuffers);

    void STDMETHODCALLTYPE CSSetConstantBuffers1(
            UINT                              StartSlot,
            UINT                              NumBuffers,
            ID3D11Buffer* const*              ppConstantBuffers,
      const UINT*                             pFirstConstant,
      const UINT*                             pNumConstants);

    void STDMETHODCALLTYPE CSSetShaderResources(
            UINT                              StartSlot,
            UINT                              NumViews,
            ID3D11ShaderResourceView* const*  ppShaderResourceViews);

    void STDMETHODCALLTYPE CSSetSamplers(
            UINT                              StartSlot,
            UINT                              NumSamplers,
            ID3D11SamplerState* const*        ppSamplers);

    void STDMETHODCALLTYPE CSSetUnorderedAccessViews(
            UINT                              StartSlot,
            UINT                              NumUAVs,
            ID3D11UnorderedAccessView* const* ppUnorderedAccessViews,
      const UINT*                             pUAVInitialCounts);

    void STDMETHODCALLTYPE CSGetShader(
            ID3D11ComputeShader**             ppComputeShader,
            ID3D11ClassInstance**             ppClassInstances,
            UINT*                             pNumClassInstances);

    void STDMETHODCALLTYPE CSGetConstantBuffers(
            UINT                              StartSlot,
            UINT                              NumBuffers,
            ID3D11Buffer**                    ppConstantBuffers);

    void STDMETHODCALLTYPE CSGetConstantBuffers1(
            UINT                              StartSlot,
            UINT                              NumBuffers,
            ID3D11Buffer**                    ppConstantBuffers,
            UINT*                             pFirstConstant,
            UINT*                             pNumConstants);

    void STDMETHODCALLTYPE CSGetShaderResources(
            UINT                              StartSlot,
            UINT                              NumViews,
            ID3D11ShaderResourceView**        ppShaderResourceViews);

    void STDMETHODCALLTYPE CSGetSamplers(
            UINT                              StartSlot,
            UINT                              NumSamplers,
            ID3D11SamplerState**              ppSamplers);

    void STDMETHODCALLTYPE CSGetUnorderedAccessViews(
            UINT                              StartSlot,
            UINT                              NumUAVs,
            ID3D11UnorderedAccessView**       ppUnorderedAccessViews);

    void STDMETHODCALLTYPE OMSetRenderTargets(
            UINT                              NumViews,
            ID3D11RenderTargetView* const*    ppRenderTargetViews,
            ID3D11DepthStencilView*           pDepthStencilView);

    void STDMETHODCALLTYPE OMSetRenderTargetsAndUnorderedAccessViews(
            UINT                              NumRTVs,
            ID3D11RenderTargetView* const*    ppRenderTargetViews,
            ID3D11DepthStencilView*           pDepthStencilView,
            UINT                              UAVStartSlot,
            UINT                              NumUAVs,
            ID3D11UnorderedAccessView* const* ppUnorderedAccessViews,
      const UINT*                             pUAVInitialCounts);

    void STDMETHODCALLTYPE OMSetBlendState(
            ID3D11BlendState*                 pBlendState,
      const FLOAT                             BlendFactor[4],
            UINT                              SampleMask);

    void STDMETHODCALLTYPE OMSetDepthStencilState(
            ID3D11DepthStencilState*          pDepthStencilState,
            UINT                              StencilRef);

    void STDMETHODCALLTYPE OMGetRenderTargets(
            UINT                              NumViews,
            ID3D11RenderTargetView**          ppRenderTargetViews,
            ID3D11DepthStencilView**          ppDepthStencilView);

    void STDMETHODCALLTYPE OMGetRenderTargetsAndUnorderedAccessViews(
            UINT                              NumRTVs,
            ID3D11RenderTargetView**          ppRenderTargetViews,
            ID3D11DepthStencilView**          ppDepthStencilView,
            UINT                              UAVStartSlot,
            UINT                              NumUAVs,
            ID3D11UnorderedAccessView**       ppUnorderedAccessViews);

    void STDMETHODCALLTYPE OMGetBlendState(
            ID3D11BlendState**                ppBlendState,
            FLOAT                             BlendFactor[4],
            UINT*                             pSampleMask);

    void STDMETHODCALLTYPE OMGetDepthStencilState(
            ID3D11DepthStencilState**         ppDepthStencilState,
            UINT*                             pStencilRef);

    void STDMETHODCALLTYPE RSSetState(
            ID3D11RasterizerState*            pRasterizerState);

    void STDMETHODCALLTYPE RSSetViewports(
            UINT                              NumViewports,
      const D3D11_VIEWPORT*                   pViewports);

    void STDMETHODCALLTYPE RSSetScissorRects(
            UINT                              NumRects,
      const D3D11_RECT*                       pRects);

    void STDMETHODCALLTYPE RSGetState(
            ID3D11RasterizerState**           ppRasterizerState);

    void STDMETHODCALLTYPE RSGetViewports(
            UINT*                             pNumViewports,
            D3D11_VIEWPORT*                   pViewports);

    void STDMETHODCALLTYPE RSGetScissorRects(
            UINT*                             pNumRects,
            D3D11_RECT*                       pRects);

    void STDMETHODCALLTYPE SOSetTargets(
            UINT                              NumBuffers,
            ID3D11Buffer* const*              ppSOTargets,
      const UINT*                             pOffsets);

    void STDMETHODCALLTYPE SOGetTargets(
            UINT                              NumBuffers,
            ID3D11Buffer**                    ppSOTargets);

    void STDMETHODCALLTYPE SOGetTargetsWithOffsets(
            UINT                              NumBuffers,
            ID3D11Buffer**                    ppSOTargets,
            UINT*                             pOffsets);

    void STDMETHODCALLTYPE SetPredication(
            ID3D11Predicate*                  pPredicate,
            BOOL                              PredicateValue);

    void STDMETHODCALLTYPE GetPredication(
            ID3D11Predicate**                 ppPredicate,
            BOOL*                             pPredicateValue);

    void STDMETHODCALLTYPE SetResourceMinLOD(
            ID3D11Resource*                   pResource,
            FLOAT                             MinLOD);

    FLOAT STDMETHODCALLTYPE GetResourceMinLOD(
            ID3D11Resource*                   pResource);

    void STDMETHODCALLTYPE CopyTiles(
            ID3D11Resource*                   pTiledResource,
      const D3D11_TILED_RESOURCE_COORDINATE*  pTileRegionStartCoordinate,
      const D3D11_TILE_REGION_SIZE*           pTileRegionSize,
            ID3D11Buffer*                     pBuffer,
            UINT64                            BufferStartOffsetInBytes,
            UINT                              Flags);

    HRESULT STDMETHODCALLTYPE CopyTileMappings(
            ID3D11Resource*                   pDestTiledResource,
      const D3D11_TILED_RESOURCE_COORDINATE*  pDestRegionCoordinate,
            ID3D11Resource*                   pSourceTiledResource,
      const D3D11_TILED_RESOURCE_COORDINATE*  pSourceRegionCoordinate,
      const D3D11_TILE_REGION_SIZE*           pTileRegionSize,
            UINT                              Flags);

    HRESULT STDMETHODCALLTYPE ResizeTilePool(
            ID3D11Buffer*                     pTilePool,
            UINT64                            NewSizeInBytes);

    void STDMETHODCALLTYPE TiledResourceBarrier(
            ID3D11DeviceChild*                pTiledResourceOrViewAccessBeforeBarrier,
            ID3D11DeviceChild*                pTiledResourceOrViewAccessAfterBarrier);

    HRESULT STDMETHODCALLTYPE UpdateTileMappings(
            ID3D11Resource*                   pTiledResource,
            UINT                              NumRegions,
      const D3D11_TILED_RESOURCE_COORDINATE*  pRegionCoordinates,
      const D3D11_TILE_REGION_SIZE*           pRegionSizes,
            ID3D11Buffer*                     pTilePool,
            UINT                              NumRanges,
      const UINT*                             pRangeFlags,
      const UINT*                             pRangeTileOffsets,
      const UINT*                             pRangeTileCounts,
            UINT                              Flags);

    void STDMETHODCALLTYPE UpdateTiles(
            ID3D11Resource*                   pDestTiledResource,
      const D3D11_TILED_RESOURCE_COORDINATE*  pDestTileRegionStartCoordinate,
      const D3D11_TILE_REGION_SIZE*           pDestTileRegionSize,
      const void*                             pSourceTileData,
            UINT                              Flags);

    BOOL STDMETHODCALLTYPE IsAnnotationEnabled();

    void STDMETHODCALLTYPE SetMarkerInt(
            LPCWSTR                           pLabel,
            INT                               Data);

    void STDMETHODCALLTYPE BeginEventInt(
            LPCWSTR                           pLabel,
            INT                               Data);

    void STDMETHODCALLTYPE EndEvent();

    void STDMETHODCALLTYPE GetHardwareProtectionState(
            BOOL*                             pHwProtectionEnable);

    void STDMETHODCALLTYPE SetHardwareProtectionState(
            BOOL                              HwProtectionEnable);

    HRESULT STDMETHODCALLTYPE GetData(
            ID3D11Asynchronous*         pAsync,
            void*                       pData,
            UINT                        DataSize,
            UINT                        GetDataFlags);

    void STDMETHODCALLTYPE Begin(
            ID3D11Asynchronous*         pAsync);

    void STDMETHODCALLTYPE End(
            ID3D11Asynchronous*         pAsync);

    void STDMETHODCALLTYPE Flush();

    void STDMETHODCALLTYPE Flush1(
            D3D11_CONTEXT_TYPE          ContextType,
            HANDLE                      hEvent);

    HRESULT STDMETHODCALLTYPE Signal(
            ID3D11Fence*                pFence,
            UINT64                      Value);

    HRESULT STDMETHODCALLTYPE Wait(
            ID3D11Fence*                pFence,
            UINT64                      Value);

    void STDMETHODCALLTYPE ExecuteCommandList(
            ID3D11CommandList*  pCommandList,
            BOOL                RestoreContextState);

    HRESULT STDMETHODCALLTYPE FinishCommandList(
            BOOL                RestoreDeferredContextState,
            ID3D11CommandList   **ppCommandList);

    HRESULT STDMETHODCALLTYPE Map(
            ID3D11Resource*             pResource,
            UINT                        Subresource,
            D3D11_MAP                   MapType,
            UINT                        MapFlags,
            D3D11_MAPPED_SUBRESOURCE*   pMappedResource);

    void STDMETHODCALLTYPE Unmap(
            ID3D11Resource*             pResource,
            UINT                        Subresource);

    void STDMETHODCALLTYPE SwapDeviceContextState(
            ID3DDeviceContextState*           pState,
            ID3DDeviceContextState**          ppPreviousState);

    /**
     * \brief Device side: waits for the front end to go idle
     *
     * For device and swapchain code that touches the real immediate
     * context from the game side (keyed mutexes, interop, reflex
     * markers, swapchain resizes). Takes the facade's multithread lock.
     */
    void DrainFromDevice(blessed::FeDrain Reason);

    /**
     * \brief Swapchain side: frame boundary
     *
     * Drains (stage 1: Present runs on the game thread, as upstream)
     * and hands the front end's counters to the probe. Called at the
     * top of D3D11SwapChain::Present.
     */
    void BeginPresent();

    /**
     * \brief Stage 2: whether this Present is recorded
     *
     * True when d3d11.blessedThreadedPresent is on, the call is not a
     * DXGI_PRESENT_TEST, and its parameters fit a record. False sends
     * the caller down the stage 1 path (drain, run on this thread).
     */
    bool RecordsPresent(
            UINT                      PresentFlags,
      const DXGI_PRESENT_PARAMETERS*  pPresentParameters) const;

    /**
     * \brief Stage 2: records Present for the front end
     *
     * Publishes, then waits while the front end is more than one Present
     * behind: until it has replayed the Present before this one.
     * \param [out] pFrameId This Present's frame id, known or predicted,
     *    for the game-side latency sleep; 0 while not known yet
     * \returns The newest Present result known: this one's if the front
     *    end already ran it, else the previous one's (S_OK for the first)
     */
    HRESULT RecordPresent(
            D3D11SwapChain*           pSwapChain,
            UINT                      SyncInterval,
            UINT                      PresentFlags,
      const DXGI_PRESENT_PARAMETERS*  pPresentParameters,
            uint64_t*                 pFrameId);

    /**
     * \brief True on a thread that is replaying records right now
     *
     * The front end thread, or the recording thread itself inside a
     * loopback replay. A Present that sees it runs its upstream body.
     */
    static bool IsReplaying() {
      return t_replaying;
    }

    /**
     * \brief Whether the game can record a Present ahead of its replay
     *
     * Threaded (not loopback), with Present recorded (stage 2).
     */
    bool RunsAhead() const {
      return m_threadedPresent && !m_loopback;
    }

    /**
     * \brief Queues a final private release, any thread
     */
    void DeferRelease(
            void*             pObject,
            BlessedReleaseFn  pfnRelease);

    D3D10DeviceLock LockContext() {
      return m_multithread.AcquireLock();
    }

    // Replay-side reach into the real context, for the record thunks in
    // blessed_threaded_record.cpp. The facade is a friend of it.
    template<typename Cmd>
    static void ReplayEmitCs(D3D11ImmediateContext* pContext, Cmd&& Command) {
      pContext->EmitCs(std::forward<Cmd>(Command));
    }

    static void ReplayThrottleDiscard(D3D11ImmediateContext* pContext, VkDeviceSize Size) {
      pContext->ThrottleDiscard(Size);
    }

    static void ReplayExecuteFlush(D3D11ImmediateContext* pContext, GpuFlushType Type) {
      pContext->ExecuteFlush(Type, nullptr, false);
    }

    static void ReplayConsiderFlush(D3D11ImmediateContext* pContext, GpuFlushType Type) {
      pContext->ConsiderFlush(Type);
    }

    static void ReplayCbRename(D3D11ImmediateContext* pContext, D3D11Buffer* pBuffer, const BlessedCbRingChunk& Chunk) {
      pContext->BlessedEmitCbRename(pBuffer, Chunk);
    }

    // blessed: cb-mirror -- lets the compact CbRename op (FeOpCbRename)
    // resolve its mirror at replay time; see the comment on that struct.
    static bool ReplayCbMirrorEnabled(D3D11ImmediateContext* pContext) {
      return pContext->BlessedCbMirrorEnabled();
    }

    static DxvkResourceAllocation* ReplayFindCbMirror(D3D11ImmediateContext* pContext, DxvkResourceAllocation* block) {
      return pContext->BlessedFindCbMirror(block);
    }

    // blessed: cb-mirror -- a retired block's dirty range is final; see
    // FeMirrorEarlyFlushRec (blessed_threaded_record.cpp).
    static void ReplayCbMirrorEarlyFlush(D3D11ImmediateContext* pContext) {
      pContext->BlessedFlushCbMirrorEarly();
    }

    static IDXVKUserDefinedAnnotation* ReplayAnnotation(D3D11ImmediateContext* pContext) {
      return &pContext->m_annotation;
    }

    static void ReplayCbRingFrame(D3D11ImmediateContext* pContext, uint64_t Frame) {
      pContext->m_blessedCbRingPresetFrame = Frame;
    }

  private:

    /// vk ext context: every call drains, then runs on the real context
    class ExtContext : public ID3D11VkExtContext1 {
    public:
      ExtContext(D3D11ThreadedContext* pParent) : m_parent(pParent) { }

      ULONG STDMETHODCALLTYPE AddRef();
      ULONG STDMETHODCALLTYPE Release();
      HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppvObject);

      void STDMETHODCALLTYPE MultiDrawIndirect(UINT DrawCount, ID3D11Buffer* pBufferForArgs,
        UINT ByteOffsetForArgs, UINT ByteStrideForArgs);
      void STDMETHODCALLTYPE MultiDrawIndexedIndirect(UINT DrawCount, ID3D11Buffer* pBufferForArgs,
        UINT ByteOffsetForArgs, UINT ByteStrideForArgs);
      void STDMETHODCALLTYPE MultiDrawIndirectCount(UINT MaxDrawCount, ID3D11Buffer* pBufferForCount,
        UINT ByteOffsetForCount, ID3D11Buffer* pBufferForArgs, UINT ByteOffsetForArgs, UINT ByteStrideForArgs);
      void STDMETHODCALLTYPE MultiDrawIndexedIndirectCount(UINT MaxDrawCount, ID3D11Buffer* pBufferForCount,
        UINT ByteOffsetForCount, ID3D11Buffer* pBufferForArgs, UINT ByteOffsetForArgs, UINT ByteStrideForArgs);
      void STDMETHODCALLTYPE SetDepthBoundsTest(BOOL Enable, FLOAT MinDepthBounds, FLOAT MaxDepthBounds);
      void STDMETHODCALLTYPE SetBarrierControl(UINT ControlFlags);
      bool STDMETHODCALLTYPE LaunchCubinShaderNVX(IUnknown* hShader, uint32_t GridX, uint32_t GridY,
        uint32_t GridZ, const void* pParams, uint32_t ParamSize, void* const* pReadResources,
        uint32_t NumReadResources, void* const* pWriteResources, uint32_t NumWriteResources);

    private:
      D3D11ThreadedContext* m_parent;
    };

    /// Annotations: recorded, label copied; the depth is kept here
    class Annotation : public IDXVKUserDefinedAnnotation {
    public:
      Annotation(D3D11ThreadedContext* pParent, bool Enabled)
      : m_parent(pParent), m_enabled(Enabled) { }

      ULONG STDMETHODCALLTYPE AddRef();
      ULONG STDMETHODCALLTYPE Release();
      HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppvObject);

      INT STDMETHODCALLTYPE BeginEvent(D3DCOLOR Color, LPCWSTR Name);
      INT STDMETHODCALLTYPE EndEvent();
      void STDMETHODCALLTYPE SetMarker(D3DCOLOR Color, LPCWSTR Name);
      BOOL STDMETHODCALLTYPE GetStatus();

    private:
      D3D11ThreadedContext* m_parent;
      bool                  m_enabled;
      int32_t               m_eventDepth = 0;
    };

    struct DeferredRelease {
      void*             object;
      BlessedReleaseFn  release;
    };

    /// Present results the front end hands back, by present number
    struct PresentResult {
      HRESULT   hr      = S_OK;
      uint64_t  frameId = 0u;
    };

    constexpr static uint32_t PresentSlots = 4u;

    // blessed: fe-crash-2 -- crash forensics, see blessed_threaded_forensics.cpp
    constexpr static uint32_t PublishLogSize = 16u;
    constexpr static uint32_t SwitchLogSize  = 8u;

    /// One publish, as the publishing thread saw it
    struct FePublishEvent {
      uint64_t      writePos;     // what was published
      uint64_t      prevPublished;
      uint64_t      packetEnd;    // m_packetEnd before the publish closed it
      FePacketRec*  packet;       // m_packet (may be an old, closed one)
      uint32_t      size;         // m_packet->size then
      uint32_t      count;        // m_packet->count then
      uint32_t      pending;      // m_packetPending: nonzero = mid-op publish
      DWORD         thread;
    };

    /// One change of the recording thread
    struct FeSwitchEvent {
      uint64_t      writePos;
      uint64_t      publishes;
      uint64_t      packetEnd;
      DWORD         from;
      DWORD         to;
    };

    static thread_local bool t_replaying;

    // --- shared, read-mostly ---
    D3D11ImmediateContext*  m_ctx;
    const bool              m_loopback;
    const bool              m_threadedPresent;
    const bool              m_cbRing;
    const bool              m_compact;
    const uint32_t          m_publishEvery;
    char*                   m_ring = nullptr;

    D3D10Multithread        m_multithread;
    D3DDestructionNotifier  m_destructionNotifier;
    ExtContext              m_ext;
    Annotation              m_annotation;

    // --- producer (game thread) ---
    alignas(CACHE_LINE_SIZE)
    uint64_t                m_writePos    = 0u; // next byte to write
    uint64_t                m_writeLimit  = 0u; // see AllocRecord
    uint64_t                m_unpublished = 0u;
    uint64_t                m_records     = 0u;
    uint64_t                m_publishes   = 0u;
    uint64_t                m_wakes       = 0u;
    uint64_t                m_releases    = 0u;
    uint64_t                m_flushHint   = ~0ull; // write position of the last FlushHint
    uint64_t                m_publishedLocal = 0u; // our copy of m_published
    uint64_t                m_presentSeq  = 0u; // presents recorded, the last one's number
    DWORD                   m_placedFor   = 0u; // recording thread PlaceFrontEnd last saw
    uint64_t                m_folded      = 0u; // redundant binds not recorded (stage 3)
    uint64_t                m_packetOps   = 0u; // calls packed into draw packets (stage 3)
    uint64_t                m_packetEnd   = ~0ull; // write position after the open packet
    FePacketRec*            m_packet      = nullptr; // the open packet, if m_packetEnd is current
    // blessed: fe-crash -- bytes PushOp reserved for the newest op but has
    // not yet counted into m_packet->size/count; Recorded() applies it once
    // the caller has stored the op's fields, so a publish (and the front
    // end reading rec->size) never sees a reserved-but-unwritten op.
    uint32_t                m_packetPending = 0u;
    // blessed: fe-crash -- the thread that opened or last extended the
    // open packet (see CheckProducerThread). PlaceFrontEnd's own comment
    // notes the recording thread does change (a loading screen hands off
    // to the render thread); m_multithread's lock is a no-op unless the
    // app opts into multithread protection, so nothing else would catch
    // two threads touching m_writePos/m_packet across that handoff.
    DWORD                   m_producerThreadId = 0u;
    // blessed: fe-crash-2 -- forensics for the crash report (see
    // blessed_threaded_forensics.cpp): the last publishes, and every
    // change of the thread that records. Plain stores, producer side.
    uint32_t                m_threadSwitches = 0u;   // CheckProducerThread saw a new thread
    uint32_t                m_foreignPublishes = 0u; // Publish ran on a thread other than the recorder
    FePublishEvent          m_publishLog[PublishLogSize] = { };
    FeSwitchEvent           m_switchLog[SwitchLogSize] = { };
    FeShadow*               m_shadow      = nullptr; // bound state as recorded (stage 3)
    bool                    m_verifyOmShadow = false; // blessed: fe-getters -- BLESSED_FE_SHADOW_VERIFY, read once
    bool                    m_noOmShadow = false; // blessed: fe-getters -- the shadow runs only with BLESSED_FE_OM_SHADOW=1
    uint32_t                m_imageMaps   = 0u; // the real context's mapped image count, seen game side

    // --- written by the producer, read by the front end ---
    alignas(CACHE_LINE_SIZE)
    std::atomic<uint64_t>   m_published = { 0u };

    // --- written by the front end, read by the producer ---
    alignas(CACHE_LINE_SIZE)
    std::atomic<uint64_t>   m_replayed  = { 0u }; // every record before it has run
    std::atomic<uint64_t>   m_idleUs    = { 0u };

    // --- front end only: the range it is replaying, for the crash report
    // (blessed: fe-crash-2). Two plain stores per ReplayRange. ---
    alignas(CACHE_LINE_SIZE)
    uint64_t                m_feBegin   = 0u;
    uint64_t                m_feEnd     = 0u;

    // --- stage 2: written by the front end at the end of each Present,
    // read by the producer after it records the next one ---
    alignas(CACHE_LINE_SIZE)
    std::atomic<uint64_t>   m_presentDone = { 0u }; // present number replayed last
    std::atomic<uint32_t>   m_presentWaiting = { 0u }; // producer asleep on m_presentEvent
    PresentResult           m_presentResults[PresentSlots];

    // --- rarely written: read by the producer at every publish ---
    alignas(CACHE_LINE_SIZE)
    std::atomic<uint32_t>   m_sleeping  = { 0u };
    std::atomic<bool>       m_stop      = { false };

    // --- final releases, from any thread ---
    alignas(CACHE_LINE_SIZE)
    std::atomic<bool>       m_releasePending = { false };
    sync::Spinlock          m_releaseLock;
    std::vector<DeferredRelease> m_releaseList;
    std::vector<DeferredRelease> m_releaseTake;

    HANDLE                  m_wakeEvent = nullptr;
    HANDLE                  m_presentEvent = nullptr;
    dxvk::thread            m_thread;

    // ring, producer side
    void* AllocRecord(size_t Size) {
      uint64_t pos = m_writePos;

      if (likely(pos + Size <= m_writeLimit)) {
        // blessed: fe-crash-2 -- plain records count toward the tripwire
        // too: a second thread that never packs an op (a copy, an update)
        // would otherwise pass unseen
        CheckProducerThread();

        m_writePos = pos + Size;
        m_records += 1u;
        return m_ring + (pos & (RingSize - 1u));
      }

      return AllocRecordSlow(Size);
    }

    void* AllocRecordSlow(size_t Size);

    void UpdateWriteLimit(uint64_t Replayed);

    void WaitForReplayed(uint64_t Target);

    void Publish();

    void FlushReleases();

    // blessed: fe-getters -- Call feeds BLESSED_FE_DRAIN_STATS's by-call
    // breakdown only (FeCall::Count skips it); every other effect of Drain
    // is unchanged. Defaulted so most call sites (FE_DRAIN, DrainFromDevice,
    // BeginPresent) need only pass their own name once.
    void Drain(blessed::FeDrain Reason, blessed::FeCall Call = blessed::FeCall::Count);

    void WakeFrontEnd();

    void TakeSnapshot();

    void PlaceFrontEnd();

    void WaitForPresent(uint64_t Target);

    // stage 2, see blessed_threaded_present.cpp
    struct PresentRec;

    void FinishPresent(uint64_t Seq, HRESULT Hr, uint64_t FrameId);

    // ring, consumer side
    uint64_t ReplayRange(uint64_t Begin, uint64_t End);

    void FrontEndThread();

    ID3D11VkExtContext1* InnerExt();

    // record helpers, see blessed_threaded_record.cpp
    template<typename Rec>
    Rec* Push(size_t Extra = 0u) {
      auto rec = static_cast<Rec*>(AllocRecord(AlignRecord(sizeof(Rec) + Extra)));
      rec->fn = &Rec::Replay;
      return rec;
    }

    static constexpr size_t AlignRecord(size_t Size) {
      return (Size + 7u) & ~size_t(7u);
    }

    // blessed: fe-crash -- defined in blessed_threaded_packet.h, which has
    // the complete FePacketRec type m_packet->size/count needs
    void Recorded();

    // blessed: fe-crash -- catches a second thread mutating the open
    // packet's bookkeeping (see m_producerThreadId). Inline so the common
    // case (same thread) costs one TEB read and a predicted-taken
    // compare; the actual report is out of line since it is the rare
    // path and needs FePacketRec's complete type to log size/count.
    void CheckProducerThread() {
      DWORD thread = GetCurrentThreadId();

      if (likely(thread == m_producerThreadId))
        return;

      ReportProducerThreadChange(thread);
    }

    void ReportProducerThreadChange(DWORD thread);

    // blessed: fe-crash-2 -- forensics, see blessed_threaded_forensics.cpp
    void ReportForeignPublish(DWORD thread);

    static void WriteCrashForensics(void* pUser, BlessedCrashWriter& Writer, uint32_t ThreadId, const char* pThreadName);

    void WriteCrashForensics(BlessedCrashWriter& Writer, uint32_t ThreadId, const char* pThreadName);

    // stage 3, see blessed_threaded_packet.h
    template<typename Op>
    Op* PushOp();

    void* PushOpSlow(size_t Size);

    void InvalidateShadow();

    // blessed: fe-getters -- recomputes the OM half of the shadow (FeShadow
    // ::omRtv/omDsv/omUav) the same way D3D11CommonContext::SetRenderTargets
    // AndUnorderedAccessViews updates m_state.om (d3d11_context.cpp:6653),
    // step for step: the same two rejections (TestRtvUavHazards,
    // ValidateRenderTargets) leave it unchanged, RTV/DSV replace outright
    // unless NumRTVs is the KEEP sentinel, UAVs replace their range
    // (clearing the previously live one) unless NumUAVs is, and a KEEP half
    // gets the same hazard unbinds (ResolveOmRtvHazards/ResolveOmUavHazards).
    // From omUnknown only a call that sets both halves makes it known.
    // Called only from a path already validated against the record-size
    // caps (NumRTVs <= D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT).
    void UpdateOmShadow(
            UINT                              NumRTVs,
            ID3D11RenderTargetView* const*    ppRenderTargetViews,
            ID3D11DepthStencilView*           pDepthStencilView,
            UINT                              UAVStartSlot,
            UINT                              NumUAVs,
            ID3D11UnorderedAccessView* const* ppUnorderedAccessViews);

    // blessed: fe-getters -- BLESSED_FE_SHADOW_VERIFY's drain and compare
    void VerifyOmShadow(blessed::FeCall Call);

    // game-side halves
    HRESULT MapBuffer(
            D3D11Buffer*                pBuffer,
            D3D11_MAP                   MapType,
            UINT                        MapFlags,
            D3D11_MAPPED_SUBRESOURCE*   pMappedResource);

    void* DiscardBuffer(
            D3D11Buffer*                pBuffer);

    void UpdateResource(
            ID3D11Resource*             pDstResource,
            UINT                        DstSubresource,
      const D3D11_BOX*                  pDstBox,
      const void*                       pSrcData,
            UINT                        SrcRowPitch,
            UINT                        SrcDepthPitch,
            UINT                        CopyFlags);

    void RecordUpdate(
            ID3D11Resource*             pDstResource,
            UINT                        DstSubresource,
      const D3D11_BOX*                  pDstBox,
      const void*                       pSrcData,
            UINT                        SrcRowPitch,
            UINT                        SrcDepthPitch,
            UINT                        CopyFlags,
            size_t                      DataSize);

    void RecordAnnotation(
            uint32_t                    Kind,
            D3DCOLOR                    Color,
            LPCWSTR                     Name);

    bool ComputeUpdateSize(
            ID3D11Resource*             pDstResource,
            UINT                        DstSubresource,
      const D3D11_BOX*                  pDstBox,
            UINT                        SrcRowPitch,
            UINT                        SrcDepthPitch,
            size_t*                     pSize);

    void SyncAppMapPtr(
            ID3D11Resource*             pResource);

    bool CanRecordView(
            ID3D11View*                 pView);

    template<typename Caller, typename T>
    void RecordArray(
            UINT                        Arg,
            UINT                        Count,
      const T*                          pItems);

    template<typename Caller>
    void RecordConstantBuffers1(
            UINT                        StartSlot,
            UINT                        NumBuffers,
            ID3D11Buffer* const*        ppConstantBuffers,
      const UINT*                       pFirstConstant,
      const UINT*                       pNumConstants);

    template<typename Caller, typename... Args>
    void RecordCall(Args... args);

    // stage 3, see blessed_threaded_record.cpp
    void RecordShader(
            FeStage                     Stage,
            void*                       pShader);

    void RecordSlotOp(
            FeOp                        Op,
            FeStage                     Stage,
            UINT                        Slot,
            void*                       pObject);

    bool RecordConstantBuffersCompact(
            FeStage                     Stage,
            UINT                        StartSlot,
            UINT                        NumBuffers,
            ID3D11Buffer* const*        ppConstantBuffers,
      const UINT*                       pFirstConstant,
      const UINT*                       pNumConstants);

  };

  /**
   * \brief blessed: threaded-fe -- times Present and drains the front end
   *
   * RAII, at the top of D3D11SwapChain::Present, on the paths that run
   * the upstream body: stage 1, PRESENT_TEST, and the front end's replay
   * of a recorded Present (which neither drains nor counts in the game
   * thread's census). With no front end on the device it costs one
   * pointer check, plus the census stamp while the probe times the frame.
   */
  class BlessedPresentScope {
  public:
    BlessedPresentScope(D3D11ThreadedContext* pFrontEnd, bool Replay)
    : m_scope(blessed::FeCall::Present, !Replay) {
      if (unlikely(pFrontEnd != nullptr) && !Replay)
        pFrontEnd->BeginPresent();
    }

    BlessedPresentScope             (const BlessedPresentScope&) = delete;
    BlessedPresentScope& operator = (const BlessedPresentScope&) = delete;
  private:
    blessed::FeScope m_scope;
  };

}
