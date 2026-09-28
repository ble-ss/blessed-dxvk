// blessed: threaded-fe -- the census list: every immediate-context entry the threaded front end's facade counts, and why it drains
#pragma once

#include <cstdint>

// One X(name) per facade entry point, in ID3D11DeviceContext4 order, then
// the entries counted outside the facade (Present, device-side context
// users, the annotation and vk-ext interfaces the facade hands out).
#define BLESSED_FE_CALLS(X) \
  X(VSSetConstantBuffers) X(PSSetShaderResources) X(PSSetShader) X(PSSetSamplers) \
  X(VSSetShader) X(DrawIndexed) X(Draw) X(Map) X(Unmap) X(PSSetConstantBuffers) \
  X(IASetInputLayout) X(IASetVertexBuffers) X(IASetIndexBuffer) X(DrawIndexedInstanced) \
  X(DrawInstanced) X(GSSetConstantBuffers) X(GSSetShader) X(IASetPrimitiveTopology) \
  X(VSSetShaderResources) X(VSSetSamplers) X(Begin) X(End) X(GetData) X(SetPredication) \
  X(GSSetShaderResources) X(GSSetSamplers) X(OMSetRenderTargets) \
  X(OMSetRenderTargetsAndUnorderedAccessViews) X(OMSetBlendState) X(OMSetDepthStencilState) \
  X(SOSetTargets) X(DrawAuto) X(DrawIndexedInstancedIndirect) X(DrawInstancedIndirect) \
  X(Dispatch) X(DispatchIndirect) X(RSSetState) X(RSSetViewports) X(RSSetScissorRects) \
  X(CopySubresourceRegion) X(CopyResource) X(UpdateSubresource) X(CopyStructureCount) \
  X(ClearRenderTargetView) X(ClearUnorderedAccessViewUint) X(ClearUnorderedAccessViewFloat) \
  X(ClearDepthStencilView) X(GenerateMips) X(SetResourceMinLOD) X(GetResourceMinLOD) \
  X(ResolveSubresource) X(ExecuteCommandList) X(HSSetShaderResources) X(HSSetShader) \
  X(HSSetSamplers) X(HSSetConstantBuffers) X(DSSetShaderResources) X(DSSetShader) \
  X(DSSetSamplers) X(DSSetConstantBuffers) X(CSSetShaderResources) \
  X(CSSetUnorderedAccessViews) X(CSSetShader) X(CSSetSamplers) X(CSSetConstantBuffers) \
  X(VSGetConstantBuffers) X(PSGetShaderResources) X(PSGetShader) X(PSGetSamplers) \
  X(VSGetShader) X(PSGetConstantBuffers) X(IAGetInputLayout) X(IAGetVertexBuffers) \
  X(IAGetIndexBuffer) X(GSGetConstantBuffers) X(GSGetShader) X(IAGetPrimitiveTopology) \
  X(VSGetShaderResources) X(VSGetSamplers) X(GetPredication) X(GSGetShaderResources) \
  X(GSGetSamplers) X(OMGetRenderTargets) X(OMGetRenderTargetsAndUnorderedAccessViews) \
  X(OMGetBlendState) X(OMGetDepthStencilState) X(SOGetTargets) X(RSGetState) \
  X(RSGetViewports) X(RSGetScissorRects) X(HSGetShaderResources) X(HSGetShader) \
  X(HSGetSamplers) X(HSGetConstantBuffers) X(DSGetShaderResources) X(DSGetShader) \
  X(DSGetSamplers) X(DSGetConstantBuffers) X(CSGetShaderResources) \
  X(CSGetUnorderedAccessViews) X(CSGetShader) X(CSGetSamplers) X(CSGetConstantBuffers) \
  X(ClearState) X(Flush) X(GetType) X(GetContextFlags) X(FinishCommandList) \
  X(CopySubresourceRegion1) X(UpdateSubresource1) X(DiscardResource) X(DiscardView) \
  X(VSSetConstantBuffers1) X(HSSetConstantBuffers1) X(DSSetConstantBuffers1) \
  X(GSSetConstantBuffers1) X(PSSetConstantBuffers1) X(CSSetConstantBuffers1) \
  X(VSGetConstantBuffers1) X(HSGetConstantBuffers1) X(DSGetConstantBuffers1) \
  X(GSGetConstantBuffers1) X(PSGetConstantBuffers1) X(CSGetConstantBuffers1) \
  X(SwapDeviceContextState) X(ClearView) X(DiscardView1) X(UpdateTileMappings) \
  X(CopyTileMappings) X(CopyTiles) X(UpdateTiles) X(ResizeTilePool) X(TiledResourceBarrier) \
  X(IsAnnotationEnabled) X(SetMarkerInt) X(BeginEventInt) X(EndEvent) X(Flush1) \
  X(SetHardwareProtectionState) X(GetHardwareProtectionState) X(Signal) X(Wait) \
  X(QueryInterface) X(AnnotationBeginEvent) X(AnnotationEndEvent) X(AnnotationSetMarker) \
  X(ExtContext) X(Present) X(DeviceInternal) X(CreateDeferredContext) X(GetImmediateContext)

// Why the game thread waited for the front end to go idle and then ran
// dxvk's own code itself.
#define BLESSED_FE_DRAINS(X) \
  X(Present) X(Getter) X(MapOther) X(UnmapImage) X(UpdateSubresource) X(Discard) \
  X(ClassInstances) X(Tiles) X(StateSwap) X(ExtContext) X(Device) X(Capture) \
  X(TooMany) X(Other)

namespace dxvk::blessed {

  enum class FeCall : uint32_t {
#define BLESSED_FE_ENUM(name) name,
    BLESSED_FE_CALLS(BLESSED_FE_ENUM)
    Count
  };

  enum class FeDrain : uint32_t {
    BLESSED_FE_DRAINS(BLESSED_FE_ENUM)
    Count
#undef BLESSED_FE_ENUM
  };

}
