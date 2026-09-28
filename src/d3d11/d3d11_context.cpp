#include <algorithm>
#include <type_traits>

#include "blessed_ao.h" // blessed: rtao, traced ao before the sao composite draw
#include "blessed_cascades.h" // blessed: sun shadow cascade learner + skip
#include "blessed_cascade_cache.h" // blessed: cascade-cache, cached static sun cascades
#include "blessed_dump.h" // blessed: frame draw-classification dumper
#include "blessed_gi.h" // blessed: ray-traced gi ambient patch (pre-draw hook)
#include "blessed_hook.h" // blessed: generic post-draw pixel-shader hook
#include "blessed_point_shadow.h" // blessed: point-lights
#include "blessed_look.h" // blessed: look-post
#include "blessed_scene_capture.h" // blessed: scene-capture selector + config
#include "blessed_volumetrics.h" // blessed: volumetrics hooks (pass 138, sun mask)
#include "blessed_vanilla_halfrate.h" // blessed: perf-halfrate
#include "blessed_skip_replaced.h" // blessed: gpu track step 2 -- skip vanilla work our features replace
#include "blessed_autoinstance.h" // blessed: auto-instancing census
#include "blessed_halfrate.h" // blessed: half-rate far field
#include "blessed_reflect_halfrate.h" // blessed: refl-harden
#include "blessed_shader_replace.h" // blessed: shader-replace, twin-draw verifier
#include "blessed_vol_collapse.h" // blessed: vol-collapse
#include "blessed_vol_async.h" // blessed: vanilla-vol-async
#include "blessed_dispatch_rewrite.h" // blessed: shader-replace, per-hash dispatch-size override
#include "blessed_vol_async_verify.h" // blessed: vanilla-vol-async, BLESSED_VOL_ASYNC_VERIFY
#include "blessed_zero_copy.h" // blessed: zero-copy-present
#include "d3d11_context.h"
#include "d3d11_context_def.h"
#include "d3d11_context_imm.h"

// blessed: render-thread timing probe, see src/util/util_blessed_probe.h
#include "../util/util_blessed_probe.h"

// blessed: scene capture's cs-thread half (BlessedSceneDraw, DxvkContext::blessedSceneAddDraw)
#include "../dxvk/blessed/blessed_scene.h"

namespace dxvk {

  template<typename ContextType>
  D3D11CommonContext<ContextType>::D3D11CommonContext(
          D3D11Device*            pParent,
    const Rc<DxvkDevice>&         Device,
          UINT                    ContextFlags,
          DxvkCsChunkFlags        CsFlags)
  : D3D11DeviceChild<ID3D11DeviceContext4>(pParent),
    m_contextExt(GetTypedContext()),
    m_annotation(GetTypedContext(), Device),
    m_device    (Device),
    m_flags     (ContextFlags),
    m_staging   (Device, StagingBufferSize),
    m_csFlags   (CsFlags),
    m_csChunk   (AllocCsChunk()) {
    // Create local allocation cache with the same properties
    // that we will use for common dynamic buffer types
    uint32_t cachedDynamic = pParent->GetOptions()->cachedDynamicResources;

    VkMemoryPropertyFlags memoryFlags =
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    if (cachedDynamic & D3D11_BIND_CONSTANT_BUFFER) {
      memoryFlags &= ~VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
      cachedDynamic = 0;
    }

    VkBufferUsageFlags bufferUsage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

    if (!(cachedDynamic & D3D11_BIND_SHADER_RESOURCE)) {
      bufferUsage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                  |  VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT;
    }

    if (!(cachedDynamic & D3D11_BIND_VERTEX_BUFFER))
      bufferUsage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;

    if (!(cachedDynamic & D3D11_BIND_INDEX_BUFFER))
      bufferUsage |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;

    m_allocationCache = m_device->createAllocationCache(bufferUsage, memoryFlags);

    // Determine maximum tess factor based on device options
    if (!IsDeferred) {
      int32_t maxTessFactor = int32_t(std::min(m_device->properties().core.properties.limits.maxTessellationGenerationLevel, 64u));

      if (m_parent->GetOptions()->maxTessFactor > 0)
        maxTessFactor = std::min(maxTessFactor, m_parent->GetOptions()->maxTessFactor);

      EmitCs([
        cMaxTessFactor = maxTessFactor
      ] (DxvkContext* ctx) {
        D3D11HsPushData pushData = {};
        pushData.maxTessFactor = float(cMaxTessFactor);

        ctx->pushData(VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
          0, sizeof(pushData), &pushData);
      });
    }
  }


  template<typename ContextType>
  D3D11CommonContext<ContextType>::~D3D11CommonContext() {

  }


  template<typename ContextType>
  HRESULT STDMETHODCALLTYPE D3D11CommonContext<ContextType>::QueryInterface(REFIID riid, void** ppvObject) {
    if (ppvObject == nullptr)
      return E_POINTER;

    *ppvObject = nullptr;

    if (riid == __uuidof(IUnknown)
     || riid == __uuidof(ID3D11DeviceChild)
     || riid == __uuidof(ID3D11DeviceContext)
     || riid == __uuidof(ID3D11DeviceContext1)
     || riid == __uuidof(ID3D11DeviceContext2)
     || riid == __uuidof(ID3D11DeviceContext3)
     || riid == __uuidof(ID3D11DeviceContext4)) {
      *ppvObject = ref(this);
      return S_OK;
    }

    if (riid == __uuidof(ID3D11VkExtContext)
     || riid == __uuidof(ID3D11VkExtContext1)) {
      *ppvObject = ref(&m_contextExt);
      return S_OK;
    }

    if (riid == __uuidof(ID3DUserDefinedAnnotation)
     || riid == __uuidof(IDXVKUserDefinedAnnotation)) {
      *ppvObject = ref(&m_annotation);
      return S_OK;
    }

    if (logQueryInterfaceError(__uuidof(ID3D11DeviceContext), riid)) {
      Logger::warn("D3D11DeviceContext::QueryInterface: Unknown interface query");
      Logger::warn(str::format(riid));
    }

    return E_NOINTERFACE;
  }


  template<typename ContextType>
  D3D11_DEVICE_CONTEXT_TYPE STDMETHODCALLTYPE D3D11CommonContext<ContextType>::GetType() {
    return IsDeferred
      ? D3D11_DEVICE_CONTEXT_DEFERRED
      : D3D11_DEVICE_CONTEXT_IMMEDIATE;
  }


  template<typename ContextType>
  UINT STDMETHODCALLTYPE D3D11CommonContext<ContextType>::GetContextFlags() {
    return m_flags;
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::ClearState() {
    D3D10DeviceLock lock = LockContext();

    ResetCommandListState();
    ResetContextState();
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::DiscardResource(ID3D11Resource* pResource) {
    D3D10DeviceLock lock = LockContext();

    if (!pResource)
      return;

    D3D11_RESOURCE_DIMENSION resType = D3D11_RESOURCE_DIMENSION_UNKNOWN;
    pResource->GetType(&resType);

    if (resType == D3D11_RESOURCE_DIMENSION_BUFFER) {
      DiscardBuffer(pResource);
    } else {
      auto texture = GetCommonTexture(pResource);
      auto image = texture->GetImage();

      for (uint32_t i = 0; i < texture->CountSubresources(); i++)
        DiscardTexture(pResource, i);

      if (image) {
        EmitCs([cImage = std::move(image)] (DxvkContext* ctx) {
          ctx->discardImage(cImage);
        });
      }
    }
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::DiscardView(ID3D11View* pResourceView) {
    DiscardViewBase(pResourceView, nullptr, 0);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::DiscardView1(
          ID3D11View*              pResourceView,
    const D3D11_RECT*              pRects,
          UINT                     NumRects) {
    DiscardViewBase(pResourceView, pRects, NumRects);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::DiscardViewBase(
          ID3D11View*              pResourceView,
    const D3D11_RECT*              pRects,
          UINT                     NumRects) {
    D3D10DeviceLock lock = LockContext();

    // We don't support discarding individual rectangles
    if (!pResourceView || (NumRects && pRects))
      return;

    // ID3D11View has no methods to query the exact type of
    // the view, so we'll have to check each possible class
    auto dsv = dynamic_cast<D3D11DepthStencilView*>(pResourceView);
    auto rtv = dynamic_cast<D3D11RenderTargetView*>(pResourceView);
    auto uav = dynamic_cast<D3D11UnorderedAccessView*>(pResourceView);

    if (rtv && rtv->GetBufferView())
      return;

    Rc<DxvkImageView> view;
    if (dsv) view = dsv->GetImageView();
    if (rtv) view = rtv->GetImageView();
    if (uav) view = uav->GetImageView();

    if (view == nullptr)
      return;

    // Get information about underlying resource
    Com<ID3D11Resource> resource;
    pResourceView->GetResource(&resource);

    uint32_t mipCount = GetCommonTexture(resource.ptr())->Desc()->MipLevels;

    // Discard mip levels one by one
    VkImageSubresourceRange sr = view->subresources();

    for (uint32_t layer = 0; layer < sr.layerCount; layer++) {
      for (uint32_t mip = 0; mip < sr.levelCount; mip++) {
        DiscardTexture(resource.ptr(), D3D11CalcSubresource(
          sr.baseMipLevel + mip, sr.baseArrayLayer + layer, mipCount));
      }
    }

    if (rtv || dsv) {
      EmitCs([cView = view] (DxvkContext* ctx) {
        DxvkAttachment attachment = {};
        attachment.view = cView;

        ctx->clearRenderTarget(attachment, 0, VkClearValue(), cView->info().aspects);
      });
    }
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::CopySubresourceRegion(
          ID3D11Resource*                   pDstResource,
          UINT                              DstSubresource,
          UINT                              DstX,
          UINT                              DstY,
          UINT                              DstZ,
          ID3D11Resource*                   pSrcResource,
          UINT                              SrcSubresource,
    const D3D11_BOX*                        pSrcBox) {
    BLESSED_PROBE_CALL(ContextType, CopySubresourceRegion);
    CopySubresourceRegionBase(
      pDstResource, DstSubresource, DstX, DstY, DstZ,
      pSrcResource, SrcSubresource, pSrcBox, 0);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::CopySubresourceRegion1(
          ID3D11Resource*                   pDstResource,
          UINT                              DstSubresource,
          UINT                              DstX,
          UINT                              DstY,
          UINT                              DstZ,
          ID3D11Resource*                   pSrcResource,
          UINT                              SrcSubresource,
    const D3D11_BOX*                        pSrcBox,
          UINT                              CopyFlags) {
    CopySubresourceRegionBase(
      pDstResource, DstSubresource, DstX, DstY, DstZ,
      pSrcResource, SrcSubresource, pSrcBox, CopyFlags);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::CopySubresourceRegionBase(
          ID3D11Resource*                   pDstResource,
          UINT                              DstSubresource,
          UINT                              DstX,
          UINT                              DstY,
          UINT                              DstZ,
          ID3D11Resource*                   pSrcResource,
          UINT                              SrcSubresource,
    const D3D11_BOX*                        pSrcBox,
          UINT                              CopyFlags) {
    D3D10DeviceLock lock = LockContext();

    if (!pDstResource || !pSrcResource)
      return;

    if (pSrcBox
     && (pSrcBox->left  >= pSrcBox->right
      || pSrcBox->top   >= pSrcBox->bottom
      || pSrcBox->front >= pSrcBox->back))
      return;

    BlessedGpuPassEvent(BlessedGpuPassKind::Xfer); // blessed: gpu-pass-timing

    // blessed: probe dump hook, immediate context only
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedDump::IsCapturing()))
        BlessedDump::RecordCopySubresourceRegion(pDstResource, DstSubresource, pSrcResource, SrcSubresource);
    }

    D3D11_RESOURCE_DIMENSION dstResourceDim = D3D11_RESOURCE_DIMENSION_UNKNOWN;
    D3D11_RESOURCE_DIMENSION srcResourceDim = D3D11_RESOURCE_DIMENSION_UNKNOWN;

    pDstResource->GetType(&dstResourceDim);
    pSrcResource->GetType(&srcResourceDim);

    if (dstResourceDim == D3D11_RESOURCE_DIMENSION_BUFFER && srcResourceDim == D3D11_RESOURCE_DIMENSION_BUFFER) {
      auto dstBuffer = static_cast<D3D11Buffer*>(pDstResource);
      auto srcBuffer = static_cast<D3D11Buffer*>(pSrcResource);

      VkDeviceSize dstOffset = DstX;
      VkDeviceSize srcOffset = 0;
      VkDeviceSize byteCount = -1;

      if (pSrcBox) {
        srcOffset = pSrcBox->left;
        byteCount = pSrcBox->right - pSrcBox->left;
      }

      CopyBuffer(dstBuffer, dstOffset, srcBuffer, srcOffset, byteCount);
    } else if (dstResourceDim != D3D11_RESOURCE_DIMENSION_BUFFER && srcResourceDim != D3D11_RESOURCE_DIMENSION_BUFFER) {
      auto dstTexture = GetCommonTexture(pDstResource);
      auto srcTexture = GetCommonTexture(pSrcResource);

      // blessed: zero-copy-present -- a copy from or into the back buffer
      if constexpr (!IsDeferred) {
        if (unlikely(BlessedZeroCopy::IsEnabled())) {
          BlessedZeroCopy::OnResourceAccess(GetTypedContext(), pSrcResource, "copy source");
          BlessedZeroCopy::OnResourceAccess(GetTypedContext(), pDstResource, "copy destination");
        }
      }

      if (DstSubresource >= dstTexture->CountSubresources()
       || SrcSubresource >= srcTexture->CountSubresources())
        return;

      auto dstFormatInfo = lookupFormatInfo(dstTexture->GetPackedFormat());
      auto srcFormatInfo = lookupFormatInfo(srcTexture->GetPackedFormat());

      auto dstLayers = vk::makeSubresourceLayers(dstTexture->GetSubresourceFromIndex(dstFormatInfo->aspectMask, DstSubresource));
      auto srcLayers = vk::makeSubresourceLayers(srcTexture->GetSubresourceFromIndex(srcFormatInfo->aspectMask, SrcSubresource));

      VkOffset3D srcOffset = { 0, 0, 0 };
      VkOffset3D dstOffset = { int32_t(DstX), int32_t(DstY), int32_t(DstZ) };

      VkExtent3D srcExtent = srcTexture->MipLevelExtent(srcLayers.mipLevel);

      if (pSrcBox) {
        srcOffset.x = pSrcBox->left;
        srcOffset.y = pSrcBox->top;
        srcOffset.z = pSrcBox->front;

        srcExtent.width  = pSrcBox->right -  pSrcBox->left;
        srcExtent.height = pSrcBox->bottom - pSrcBox->top;
        srcExtent.depth  = pSrcBox->back -   pSrcBox->front;
      }

      CopyImage(
        dstTexture, &dstLayers, dstOffset,
        srcTexture, &srcLayers, srcOffset,
        srcExtent);
    }
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::CopyResource(
          ID3D11Resource*                   pDstResource,
          ID3D11Resource*                   pSrcResource) {
    BLESSED_PROBE_CALL(ContextType, CopyResource);
    D3D10DeviceLock lock = LockContext();

    if (!pDstResource || !pSrcResource || (pDstResource == pSrcResource))
      return;

    // blessed: probe dump hook, immediate context only
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedDump::IsCapturing()))
        BlessedDump::RecordCopyResource(pDstResource, pSrcResource);
    }

    BlessedGpuPassEvent(BlessedGpuPassKind::Xfer); // blessed: gpu-pass-timing

    D3D11_RESOURCE_DIMENSION dstResourceDim = D3D11_RESOURCE_DIMENSION_UNKNOWN;
    D3D11_RESOURCE_DIMENSION srcResourceDim = D3D11_RESOURCE_DIMENSION_UNKNOWN;

    pDstResource->GetType(&dstResourceDim);
    pSrcResource->GetType(&srcResourceDim);

    if (dstResourceDim != srcResourceDim)
      return;

    if (dstResourceDim == D3D11_RESOURCE_DIMENSION_BUFFER) {
      auto dstBuffer = static_cast<D3D11Buffer*>(pDstResource);
      auto srcBuffer = static_cast<D3D11Buffer*>(pSrcResource);

      if (dstBuffer->Desc()->ByteWidth != srcBuffer->Desc()->ByteWidth)
        return;

      CopyBuffer(dstBuffer, 0, srcBuffer, 0, -1);
    } else {
      auto dstTexture = GetCommonTexture(pDstResource);
      auto srcTexture = GetCommonTexture(pSrcResource);

      // blessed: zero-copy-present -- a copy from or into the back buffer
      if constexpr (!IsDeferred) {
        if (unlikely(BlessedZeroCopy::IsEnabled())) {
          BlessedZeroCopy::OnResourceAccess(GetTypedContext(), pSrcResource, "copy source");
          BlessedZeroCopy::OnResourceAccess(GetTypedContext(), pDstResource, "copy destination");
        }
      }

      auto dstDesc = dstTexture->Desc();
      auto srcDesc = srcTexture->Desc();

      // The subresource count must match as well
      if (dstDesc->ArraySize != srcDesc->ArraySize
       || dstDesc->MipLevels != srcDesc->MipLevels)
        return;

      auto dstFormatInfo = lookupFormatInfo(dstTexture->GetPackedFormat());
      auto srcFormatInfo = lookupFormatInfo(srcTexture->GetPackedFormat());

      for (uint32_t i = 0; i < dstDesc->MipLevels; i++) {
        VkImageSubresourceLayers dstLayers = { dstFormatInfo->aspectMask, i, 0, dstDesc->ArraySize };
        VkImageSubresourceLayers srcLayers = { srcFormatInfo->aspectMask, i, 0, srcDesc->ArraySize };

        CopyImage(
          dstTexture, &dstLayers, VkOffset3D(),
          srcTexture, &srcLayers, VkOffset3D(),
          srcTexture->MipLevelExtent(i));
      }
    }
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::CopyStructureCount(
          ID3D11Buffer*                     pDstBuffer,
          UINT                              DstAlignedByteOffset,
          ID3D11UnorderedAccessView*        pSrcView) {
    D3D10DeviceLock lock = LockContext();

    auto buf = static_cast<D3D11Buffer*>(pDstBuffer);
    auto uav = static_cast<D3D11UnorderedAccessView*>(pSrcView);

    if (!buf || !uav)
      return;

    auto counterView = uav->GetCounterView();

    if (counterView == nullptr)
      return;

    AddCost(GpuCostEstimate::Transfer);

    EmitCs([
      cDstSlice = buf->GetBufferSlice(DstAlignedByteOffset),
      cSrcSlice = DxvkBufferSlice(counterView)
    ] (DxvkContext* ctx) {
      ctx->copyBuffer(
        cDstSlice.buffer(),
        cDstSlice.offset(),
        cSrcSlice.buffer(),
        cSrcSlice.offset(),
        sizeof(uint32_t));
    });

    if (buf->HasSequenceNumber())
      GetTypedContext()->TrackBufferSequenceNumber(buf);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::ClearRenderTargetView(
          ID3D11RenderTargetView*           pRenderTargetView,
    const FLOAT                             ColorRGBA[4]) {
    BLESSED_PROBE_CALL(ContextType, ClearRenderTargetView);
    D3D10DeviceLock lock = LockContext();

    auto rtv = static_cast<D3D11RenderTargetView*>(pRenderTargetView);

    if (!rtv)
      return;

    // blessed: half-rate far field -- closes the main pass (capture)
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedHalfRate::IsWatching())
       && BlessedHalfRate::OnClearRtv(static_cast<D3D11ImmediateContext*>(this), rtv))
        return;
      // blessed: refl-harden -- a skipped water cube face drops its clear
      if (unlikely(BlessedReflectHalfRate::IsWatching())
       && BlessedReflectHalfRate::OnClearRtv(rtv))
        return;
    }

    // blessed: probe dump hook, immediate context only
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedDump::IsCapturing()))
        BlessedDump::RecordClearRtv(pRenderTargetView, ColorRGBA); // blessed: + clear colour
    }

    BlessedGpuPassEvent(BlessedGpuPassKind::Xfer); // blessed: gpu-pass-timing

    AddCost(GpuCostEstimate::Transfer);

    DxvkAttachment attachment = {};
    attachment.view = rtv->GetImageView();
    attachment.shadow = rtv->GetBufferView();

    // blessed: zero-copy-present -- a clear of the back buffer lands on the
    // acquired swap chain image (and may be what starts the redirect)
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedZeroCopy::IsEnabled())) {
        Rc<DxvkImageView> redirect = BlessedZeroCopy::RedirectRenderTarget(
          GetTypedContext(), attachment.view, true);

        if (redirect != nullptr)
          attachment.view = std::move(redirect);
      }
    }

    EmitCs([
      cClearValue = ConvertColorValue(ColorRGBA, attachment.view->formatInfo()),
      cAttachment = std::move(attachment)
    ] (DxvkContext* ctx) {
      ctx->clearRenderTarget(cAttachment,
        VK_IMAGE_ASPECT_COLOR_BIT, cClearValue, 0u);
    });
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::ClearUnorderedAccessViewUint(
          ID3D11UnorderedAccessView*        pUnorderedAccessView,
    const UINT                              Values[4]) {
    D3D10DeviceLock lock = LockContext();

    if (!pUnorderedAccessView)
      return;

    Com<ID3D11UnorderedAccessView> qiUav;

    if (FAILED(pUnorderedAccessView->QueryInterface(IID_PPV_ARGS(&qiUav))))
      return;

    BlessedGpuPassEvent(BlessedGpuPassKind::Xfer); // blessed: gpu-pass-timing

    AddCost(GpuCostEstimate::Transfer);

    auto uav = static_cast<D3D11UnorderedAccessView*>(qiUav.ptr());

    // Gather UAV format info. We'll use this to determine
    // whether we need to create a temporary view or not.
    D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc;
    uav->GetDesc(&uavDesc);

    VkFormat uavFormat = m_parent->LookupFormat(uavDesc.Format, DXGI_VK_FORMAT_MODE_ANY).Format;
    VkFormat rawFormat = m_parent->LookupFormat(uavDesc.Format, DXGI_VK_FORMAT_MODE_RAW).Format;

    if (uavDesc.Format == DXGI_FORMAT_A8_UNORM)
      rawFormat = uavFormat;

    if (uavFormat && !rawFormat) {
      Logger::err(str::format("D3D11: ClearUnorderedAccessViewUint: No raw format found for ", uavFormat));
      return;
    }

    VkClearValue clearValue = { };

    if (uavDesc.Format == DXGI_FORMAT_R11G11B10_FLOAT) {
      // R11G11B10 is a special case since there's no corresponding
      // integer format with the same bit layout. Use R32 instead.
      clearValue.color.uint32[0] = ((Values[0] & 0x7FF) <<  0)
                                 | ((Values[1] & 0x7FF) << 11)
                                 | ((Values[2] & 0x3FF) << 22);
      clearValue.color.uint32[1] = 0;
      clearValue.color.uint32[2] = 0;
      clearValue.color.uint32[3] = 0;
    } else if (uavDesc.Format == DXGI_FORMAT_A8_UNORM) {
      // Use the unorm format itself to execute the clear, regardless
      // of whether we use A8 or emulate the format with R8. This is
      // necessary because we cannot create R8_UINT views for A8.
      float a = float(Values[3] & 0xff) / 255.0f;

      clearValue.color.float32[0] = a;
      clearValue.color.float32[1] = a;
      clearValue.color.float32[2] = a;
      clearValue.color.float32[3] = a;
    } else {
      clearValue.color.uint32[0] = Values[0];
      clearValue.color.uint32[1] = Values[1];
      clearValue.color.uint32[2] = Values[2];
      clearValue.color.uint32[3] = Values[3];
    }

    if (uav->GetResourceType() == D3D11_RESOURCE_DIMENSION_BUFFER) {
      // In case of raw and structured buffers as well as typed
      // buffers that can be used for atomic operations, we can
      // use the fast Vulkan buffer clear function.
      Rc<DxvkBufferView> bufferView = uav->GetBufferView();

      if (bufferView->info().format == VK_FORMAT_R32_UINT
       || bufferView->info().format == VK_FORMAT_R32_SINT
       || bufferView->info().format == VK_FORMAT_R32_SFLOAT
       || bufferView->info().format == VK_FORMAT_B10G11R11_UFLOAT_PACK32) {
        EmitCs([
          cClearValue = clearValue.color.uint32[0],
          cDstSlice   = DxvkBufferSlice(bufferView)
        ] (DxvkContext* ctx) {
          ctx->clearBuffer(
            cDstSlice.buffer(),
            cDstSlice.offset(),
            cDstSlice.length(),
            cClearValue);
        });
      } else {
        // Create a view with an integer format if necessary
        if (uavFormat != rawFormat)  {
          DxvkBufferViewKey info = bufferView->info();
          info.format = rawFormat;

          bufferView = bufferView->buffer()->createView(info);
        }

        EmitCs([
          cClearValue = clearValue,
          cDstView    = bufferView
        ] (DxvkContext* ctx) {
          ctx->clearBufferView(
            cDstView, 0,
            cDstView->elementCount(),
            cClearValue.color);
        });
      }
    } else {
      Rc<DxvkImageView> imageView = uav->GetImageView();

      // If the clear value is zero, we can use the original view regardless of
      // the format since the bit pattern will not change in any supported format.
      bool isZeroClearValue = !(clearValue.color.uint32[0] | clearValue.color.uint32[1]
                              | clearValue.color.uint32[2] | clearValue.color.uint32[3]);

      EmitCs([
        cClearValue = clearValue,
        cDstView    = imageView,
        cDstFormat  = isZeroClearValue ? uavFormat : rawFormat
      ] (DxvkContext* ctx) {
        // Ensure that we can write to the image with the given format
        DxvkImageUsageInfo imageUsage = { };
        imageUsage.usage = VK_IMAGE_USAGE_STORAGE_BIT;
        imageUsage.viewFormatCount = 1;
        imageUsage.viewFormats = &cDstFormat;

        if (!ctx->ensureImageCompatibility(cDstView->image(), imageUsage)) {
          Logger::err("D3D11: Failed to recreate image for ClearUAV");
          return;
        }

        // If necessary, recreate the view
        Rc<DxvkImageView> view = cDstView;

        if (view->info().format != cDstFormat) {
          DxvkImageViewKey key = cDstView->info();
          key.format = cDstFormat;

          view = cDstView->image()->createView(key);
        }

        ctx->clearImageView(view,
          VkOffset3D { 0, 0, 0 },
          cDstView->mipLevelExtent(0),
          VK_IMAGE_ASPECT_COLOR_BIT,
          cClearValue);
      });
    }
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::ClearUnorderedAccessViewFloat(
          ID3D11UnorderedAccessView*        pUnorderedAccessView,
    const FLOAT                             Values[4]) {
    D3D10DeviceLock lock = LockContext();

    auto uav = static_cast<D3D11UnorderedAccessView*>(pUnorderedAccessView);

    if (!uav)
      return;

    auto imgView = uav->GetImageView();
    auto bufView = uav->GetBufferView();

    const DxvkFormatInfo* info = nullptr;
    if (imgView != nullptr) info = imgView->formatInfo();
    if (bufView != nullptr) info = bufView->formatInfo();

    if (!info || info->flags.any(DxvkFormatFlag::SampledSInt, DxvkFormatFlag::SampledUInt))
      return;

    BlessedGpuPassEvent(BlessedGpuPassKind::Xfer); // blessed: gpu-pass-timing

    AddCost(GpuCostEstimate::Transfer);

    VkClearValue clearValue;
    clearValue.color.float32[0] = Values[0];
    clearValue.color.float32[1] = Values[1];
    clearValue.color.float32[2] = Values[2];
    clearValue.color.float32[3] = Values[3];

    if (uav->GetResourceType() == D3D11_RESOURCE_DIMENSION_BUFFER) {
      EmitCs([
        cClearValue = clearValue,
        cDstView    = std::move(bufView)
      ] (DxvkContext* ctx) {
        ctx->clearBufferView(
          cDstView, 0,
          cDstView->elementCount(),
          cClearValue.color);
      });
    } else {
      EmitCs([
        cClearValue = clearValue,
        cDstView    = std::move(imgView)
      ] (DxvkContext* ctx) {
        ctx->clearImageView(cDstView,
          VkOffset3D { 0, 0, 0 },
          cDstView->mipLevelExtent(0),
          VK_IMAGE_ASPECT_COLOR_BIT,
          cClearValue);
      });
    }
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::ClearDepthStencilView(
          ID3D11DepthStencilView*           pDepthStencilView,
          UINT                              ClearFlags,
          FLOAT                             Depth,
          UINT8                             Stencil) {
    BLESSED_PROBE_CALL(ContextType, ClearDepthStencilView);
    D3D10DeviceLock lock = LockContext();

    auto dsv = static_cast<D3D11DepthStencilView*>(pDepthStencilView);

    if (!dsv)
      return;

    // blessed: half-rate far field -- same as ClearRenderTargetView
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedHalfRate::IsWatching())
       && BlessedHalfRate::OnClearDsv(static_cast<D3D11ImmediateContext*>(this), dsv))
        return;
    }

    // blessed: BLESSED_SKIP_CASCADES, immediate context only -- don't even
    // clear a cascade depth target once the ray-traced mask has replaced
    // it. See blessed_cascades.h.
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedCascadeSkip::ShouldSkipClear(dsv))) {
        BlessedCascadeSkip::RecordSkippedClear();
        return;
      }
      if (unlikely(BlessedCascadeCache::IsEnabled()) // blessed: cascade-cache
       && BlessedCascadeCache::OnClearDsv(static_cast<D3D11ImmediateContext*>(this), dsv, ClearFlags, Depth))
        return;
    }

    // blessed: probe dump hook, immediate context only
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedDump::IsCapturing()))
        BlessedDump::RecordClearDsv(pDepthStencilView);
    }

    BlessedGpuPassEvent(BlessedGpuPassKind::Xfer); // blessed: gpu-pass-timing

    // Figure out which aspects to clear based on
    // the image view properties and clear flags.
    VkImageAspectFlags aspectMask = 0;

    if (ClearFlags & D3D11_CLEAR_DEPTH)
      aspectMask |= VK_IMAGE_ASPECT_DEPTH_BIT;

    if (ClearFlags & D3D11_CLEAR_STENCIL)
      aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;

    aspectMask &= dsv->GetWritableAspectMask();

    if (!aspectMask)
      return;

    AddCost(GpuCostEstimate::Transfer);

    VkClearValue clearValue;
    clearValue.depthStencil.depth   = Depth;
    clearValue.depthStencil.stencil = Stencil;

    DxvkAttachment attachment = {};
    attachment.view = dsv->GetImageView();

    EmitCs([
      cClearValue = clearValue,
      cAspectMask = aspectMask,
      cAttachment = std::move(attachment)
    ] (DxvkContext* ctx) {
      ctx->clearRenderTarget(cAttachment,
        cAspectMask, cClearValue, 0u);
    });
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::ClearView(
          ID3D11View*                       pView,
    const FLOAT                             Color[4],
    const D3D11_RECT*                       pRect,
          UINT                              NumRects) {
    D3D10DeviceLock lock = LockContext();

    if (NumRects && !pRect)
      return;

    BlessedGpuPassEvent(BlessedGpuPassKind::Xfer); // blessed: gpu-pass-timing

    AddCost(GpuCostEstimate::Transfer);

    // ID3D11View has no methods to query the exact type of
    // the view, so we'll have to check each possible class
    auto dsv = dynamic_cast<D3D11DepthStencilView*>(pView);
    auto rtv = dynamic_cast<D3D11RenderTargetView*>(pView);
    auto uav = dynamic_cast<D3D11UnorderedAccessView*>(pView);
    auto vov = dynamic_cast<D3D11VideoProcessorOutputView*>(pView);

    // Retrieve underlying resource view
    if (dsv) {
      Rc<DxvkImageView> imgView = dsv->GetImageView();

      if (imgView)
        ClearImageView(std::move(imgView), Color, pRect, NumRects);
    } else if (rtv) {
      Rc<DxvkImageView> imgView = rtv->GetImageView();
      Rc<DxvkBufferView> bufView = rtv->GetBufferView();

      if (bufView) {
        Logger::err("D3D11: ClearView on buffer RTV not supported.");
        return;
      }

      // blessed: refl-harden -- a skipped water cube face drops its clear
      if constexpr (!IsDeferred) {
        if (unlikely(BlessedReflectHalfRate::IsWatching())
         && BlessedReflectHalfRate::OnClearRtv(rtv))
          return;
      }

      // blessed: zero-copy-present -- same as ClearRenderTargetView
      if constexpr (!IsDeferred) {
        if (unlikely(BlessedZeroCopy::IsEnabled()) && imgView != nullptr) {
          Rc<DxvkImageView> redirect = BlessedZeroCopy::RedirectRenderTarget(
            GetTypedContext(), imgView, true);

          if (redirect != nullptr)
            imgView = std::move(redirect);
        }
      }

      if (imgView)
        ClearImageView(std::move(imgView), Color, pRect, NumRects);
    } else if (uav) {
      Rc<DxvkImageView> imgView = uav->GetImageView();
      Rc<DxvkBufferView> bufView = uav->GetBufferView();

      if (imgView)
        ClearImageView(std::move(imgView), Color, pRect, NumRects);

      if (bufView)
        ClearBufferView(std::move(bufView), Color, pRect, NumRects);
    } else if (vov) {
      auto views = vov->GetCommon().GetViews();
      auto shadow = vov->GetCommon().GetShadow();

      // If we have to assume that the image is only partially cleared,
      // make sure to properly sync it with the shadow image.
      VkImageSubresourceLayers imageLayers = vov->GetCommon().GetImageSubresource();

      VkImageSubresourceLayers shadowLayers = { };
      shadowLayers.aspectMask = imageLayers.aspectMask;
      shadowLayers.layerCount = imageLayers.layerCount;

      if (shadow && NumRects)
        SyncImage(shadow, shadowLayers, vov->GetCommon().GetImage(), imageLayers);

      // Assume that planar video formats use Y - Cb - Cr
      // order for the purpose of mapping color components.
      uint32_t component = 0u;
      FLOAT planeColor[4] = { };

      for (uint32_t i = 0u; i < views.size(); i++) {
        if (!views[i])
          break;

        // Extract relevant color components from the clear color
        // and shift the input array accordingly.
        uint32_t n = bit::popcnt(views[i]->formatInfo()->componentMask);

        for (uint32_t c = 0u; c < 4u; c++)
          planeColor[c] = c < n && component + c < 4u ? Color[c] : 0.0f;

        component += n;

        // Perform the actual clear. Rects will be adjusted by called method.
        ClearImageView(std::move(views[i]), planeColor, pRect, NumRects);
      }

      // If the video view has a shadow image, copy it back to the base image
      if (shadow)
        SyncImage(vov->GetCommon().GetImage(), imageLayers, shadow, shadowLayers);
    }
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::GenerateMips(ID3D11ShaderResourceView* pShaderResourceView) {
    D3D10DeviceLock lock = LockContext();

    auto view = static_cast<D3D11ShaderResourceView*>(pShaderResourceView);

    if (!view || view->GetResourceType() == D3D11_RESOURCE_DIMENSION_BUFFER)
      return;

    D3D11_COMMON_RESOURCE_DESC resourceDesc = view->GetResourceDesc();

    if (!(resourceDesc.MiscFlags & D3D11_RESOURCE_MISC_GENERATE_MIPS))
      return;

    AddCost(GpuCostEstimate::Transfer);

    EmitCs([cDstImageView = view->GetImageView()]
    (DxvkContext* ctx) {
      ctx->generateMipmaps(cDstImageView, VK_FILTER_LINEAR);
    });
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::ResolveSubresource(
          ID3D11Resource*                   pDstResource,
          UINT                              DstSubresource,
          ID3D11Resource*                   pSrcResource,
          UINT                              SrcSubresource,
          DXGI_FORMAT                       Format) {
    D3D10DeviceLock lock = LockContext();

    bool isSameSubresource = pDstResource   == pSrcResource
                          && DstSubresource == SrcSubresource;

    if (!pDstResource || !pSrcResource || isSameSubresource)
      return;

    D3D11_RESOURCE_DIMENSION dstResourceType;
    D3D11_RESOURCE_DIMENSION srcResourceType;

    pDstResource->GetType(&dstResourceType);
    pSrcResource->GetType(&srcResourceType);

    if (dstResourceType != D3D11_RESOURCE_DIMENSION_TEXTURE2D
     || srcResourceType != D3D11_RESOURCE_DIMENSION_TEXTURE2D)
      return;

    BlessedGpuPassEvent(BlessedGpuPassKind::Xfer); // blessed: gpu-pass-timing

    // blessed: zero-copy-present -- a resolve from or into the back buffer
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedZeroCopy::IsEnabled())) {
        BlessedZeroCopy::OnResourceAccess(GetTypedContext(), pSrcResource, "resolve source");
        BlessedZeroCopy::OnResourceAccess(GetTypedContext(), pDstResource, "resolve destination");
      }
    }

    auto dstTexture = static_cast<D3D11Texture2D*>(pDstResource);
    auto srcTexture = static_cast<D3D11Texture2D*>(pSrcResource);

    D3D11_TEXTURE2D_DESC dstDesc;
    D3D11_TEXTURE2D_DESC srcDesc;

    dstTexture->GetDesc(&dstDesc);
    srcTexture->GetDesc(&srcDesc);

    if (dstDesc.SampleDesc.Count != 1)
      return;

    D3D11CommonTexture* dstTextureInfo = GetCommonTexture(pDstResource);
    D3D11CommonTexture* srcTextureInfo = GetCommonTexture(pSrcResource);

    const DXGI_VK_FORMAT_INFO dstFormatInfo = m_parent->LookupFormat(dstDesc.Format, DXGI_VK_FORMAT_MODE_ANY);
    const DXGI_VK_FORMAT_INFO srcFormatInfo = m_parent->LookupFormat(srcDesc.Format, DXGI_VK_FORMAT_MODE_ANY);

    auto dstVulkanFormatInfo = lookupFormatInfo(dstFormatInfo.Format);
    auto srcVulkanFormatInfo = lookupFormatInfo(srcFormatInfo.Format);

    if (DstSubresource >= dstTextureInfo->CountSubresources()
     || SrcSubresource >= srcTextureInfo->CountSubresources())
      return;

    AddCost(GpuCostEstimate::Transfer);

    const VkImageSubresource dstSubresource =
      dstTextureInfo->GetSubresourceFromIndex(
        dstVulkanFormatInfo->aspectMask, DstSubresource);

    const VkImageSubresource srcSubresource =
      srcTextureInfo->GetSubresourceFromIndex(
        srcVulkanFormatInfo->aspectMask, SrcSubresource);

    const VkImageSubresourceLayers dstSubresourceLayers = {
      dstSubresource.aspectMask,
      dstSubresource.mipLevel,
      dstSubresource.arrayLayer, 1 };

    const VkImageSubresourceLayers srcSubresourceLayers = {
      srcSubresource.aspectMask,
      srcSubresource.mipLevel,
      srcSubresource.arrayLayer, 1 };

    if (srcDesc.SampleDesc.Count == 1 || m_parent->GetOptions()->disableMsaa) {
      EmitCs([
        cDstImage  = dstTextureInfo->GetImage(),
        cSrcImage  = srcTextureInfo->GetImage(),
        cDstLayers = dstSubresourceLayers,
        cSrcLayers = srcSubresourceLayers
      ] (DxvkContext* ctx) {
        ctx->copyImage(
          cDstImage, cDstLayers, VkOffset3D { 0, 0, 0 },
          cSrcImage, cSrcLayers, VkOffset3D { 0, 0, 0 },
          cDstImage->mipLevelExtent(cDstLayers.mipLevel));
      });
    } else {
      VkFormat format = m_parent->LookupFormat(
        Format, DXGI_VK_FORMAT_MODE_ANY).Format;

      EmitCs([
        cDstImage  = dstTextureInfo->GetImage(),
        cSrcImage  = srcTextureInfo->GetImage(),
        cDstSubres = dstSubresourceLayers,
        cSrcSubres = srcSubresourceLayers,
        cFormat    = format
      ] (DxvkContext* ctx) {
        VkFormat format = cFormat ? cFormat : cSrcImage->info().format;

        VkImageResolve region;
        region.srcSubresource = cSrcSubres;
        region.srcOffset      = VkOffset3D { 0, 0, 0 };
        region.dstSubresource = cDstSubres;
        region.dstOffset      = VkOffset3D { 0, 0, 0 };
        region.extent         = cDstImage->mipLevelExtent(cDstSubres.mipLevel);

        ctx->resolveImage(cDstImage, cSrcImage, region, format,
          getDefaultResolveMode(format), VK_RESOLVE_MODE_NONE);
      });
    }

    if constexpr (!IsDeferred)
      GetTypedContext()->NotifyResolve();

    if (dstTextureInfo->HasSequenceNumber())
      GetTypedContext()->TrackTextureSequenceNumber(dstTextureInfo, DstSubresource);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::UpdateSubresource(
          ID3D11Resource*                   pDstResource,
          UINT                              DstSubresource,
    const D3D11_BOX*                        pDstBox,
    const void*                             pSrcData,
          UINT                              SrcRowPitch,
          UINT                              SrcDepthPitch) {
    BLESSED_PROBE_CALL(ContextType, UpdateSubresource);
    if (IsDeferred && unlikely(pDstBox != nullptr) && unlikely(!m_parent->GetOptions()->exposeDriverCommandLists)) {
      // If called from a deferred context and native command list support is not
      // exposed, we need to apply the destination box to the source pointer. This
      // only applies to UpdateSubresource, not to UpdateSubresource1. See MSDN:
      // https://msdn.microsoft.com/en-us/library/windows/desktop/ff476486(v=vs.85).aspx)
      size_t srcOffset = pDstBox->left;

      // For textures, the offset logic needs to take the format into account.
      // Ignore that multi-planar images exist, this is hairy enough already.
      D3D11CommonTexture* dstTexture = GetCommonTexture(pDstResource);

      if (dstTexture) {
        auto dstFormat = dstTexture->GetPackedFormat();
        auto dstFormatInfo = lookupFormatInfo(dstFormat);

        size_t blockSize = dstFormatInfo->elementSize;

        VkOffset3D offset;
        offset.x = pDstBox->left / dstFormatInfo->blockSize.width;
        offset.y = pDstBox->top / dstFormatInfo->blockSize.height;
        offset.z = pDstBox->front / dstFormatInfo->blockSize.depth;

        srcOffset = offset.x * blockSize + offset.y * SrcRowPitch + offset.z * SrcDepthPitch;
      }

      pSrcData = reinterpret_cast<const char*>(pSrcData) + srcOffset;
    }

    UpdateResource(pDstResource, DstSubresource, pDstBox,
      pSrcData, SrcRowPitch, SrcDepthPitch, 0);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::UpdateSubresource1(
          ID3D11Resource*                   pDstResource,
          UINT                              DstSubresource,
    const D3D11_BOX*                        pDstBox,
    const void*                             pSrcData,
          UINT                              SrcRowPitch,
          UINT                              SrcDepthPitch,
          UINT                              CopyFlags) {
    BLESSED_PROBE_CALL(ContextType, UpdateSubresource1);
    UpdateResource(pDstResource, DstSubresource, pDstBox,
      pSrcData, SrcRowPitch, SrcDepthPitch, CopyFlags);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::DrawAuto() {
    D3D10DeviceLock lock = LockContext();

    // blessed: vanilla-vol-async -- every draw entry point closes the
    // window first, if still open. See blessed_vol_async.h.
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedVolAsync::IsEnabled()))
        BlessedVolAsync::OnDrawPre(static_cast<D3D11ImmediateContext*>(this), m_state);
    }

    // blessed: refl-harden -- a skipped water cube face drops all its draws
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedReflectHalfRate::IsWatching()) // blessed: refl-harden
       && BlessedReflectHalfRate::OnDraw(m_state))
        return;
    }

    D3D11Buffer* buffer = m_state.ia.vertexBuffers[0].buffer.ptr();

    if (!buffer)
      return;

    DxvkBufferSlice vtxBuf = buffer->GetBufferSlice();
    DxvkBufferSlice ctrBuf = buffer->GetSOCounter();

    if (!ctrBuf.defined())
      return;

    if constexpr (!IsDeferred) { // blessed: cascade-cache, an indirect or auto draw may end a cascade
      if (unlikely(BlessedCascadeCache::IsEnabled()))
        BlessedCascadeCache::OnDraw(static_cast<D3D11ImmediateContext*>(this), m_state, BlessedCascadeDrawKind::Other, 0u, 0u, 0);
    }

    BlessedGpuPassEvent(BlessedGpuPassKind::Draw); // blessed: gpu-pass-timing

    if (unlikely(HasDirtyGraphicsBindings()))
      ApplyDirtyGraphicsBindings();

    // We bind the SO counter as an indirect count buffer,
    // so reset any tracking we may have been doing here.
    m_state.id.reset();

    EmitCs([=] (DxvkContext* ctx) mutable {
      ctx->bindDrawBuffers(DxvkBufferSlice(),
        Forwarder::move(ctrBuf));

      ctx->drawIndirectXfb(0u,
        vtxBuf.buffer()->getXfbVertexStride(),
        vtxBuf.offset());

      // Reset draw buffer right away so we don't
      // keep the SO counter alive indefinitely
      ctx->bindDrawBuffers(DxvkBufferSlice(),
        DxvkBufferSlice());
    });
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::Draw(
          UINT            VertexCount,
          UINT            StartVertexLocation) {
    BLESSED_PROBE_CALL(ContextType, Draw);
    D3D10DeviceLock lock = LockContext();

    if (unlikely(!VertexCount))
      return;

    // blessed: vanilla-vol-async -- every draw entry point closes the
    // window first, if still open, before anything else. See
    // blessed_vol_async.h.
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedVolAsync::IsEnabled()))
        BlessedVolAsync::OnDrawPre(static_cast<D3D11ImmediateContext*>(this), m_state);
    }

    // blessed: half-rate far field -- skip far draws on off frames and
    // track the main lit pass. See blessed_halfrate.h.
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedHalfRate::IsWatching())
       && BlessedHalfRate::OnDraw(static_cast<D3D11ImmediateContext*>(this), m_state, m_device.ptr()))
        return;
      if (unlikely(BlessedReflectHalfRate::IsWatching()) // blessed: refl-harden
       && BlessedReflectHalfRate::OnDraw(m_state))
        return;
    }

    // blessed: BLESSED_SKIP_CASCADES, immediate context only -- a cascade
    // depth draw is dropped before it ever reaches BatchDraw. See
    // blessed_cascades.h.
    if constexpr (!IsDeferred) {
      // blessed: perf-halfrate -- pass 138 on off frames (BLESSED_VOL_HALFRATE=2).
      // See blessed_vanilla_halfrate.h.
      if (unlikely(BlessedVolHalfRate::ShouldSkipDraw(m_state)))
        return;
      if (unlikely(BlessedCascadeSkip::ShouldSkipDraw(m_state))) {
        BlessedCascadeSkip::RecordSkippedDraw();
        return;
      }
      if (unlikely(BlessedCascadeCache::IsEnabled()) && BlessedCascadeCache::OnDraw( // blessed: cascade-cache, never cached
            static_cast<D3D11ImmediateContext*>(this), m_state, BlessedCascadeDrawKind::Other, 0u, 0u, 0))
        return;
      if (unlikely(BlessedPointShadow::ShouldSkipDraw(static_cast<D3D11ImmediateContext*>(this), m_state))) // blessed: point-lights
        return;
      // blessed: gpu track step 2 -- skip vanilla work our replacements
      // already overwrite. See blessed_skip_replaced.h.
      if (unlikely(BlessedSkipAo::ShouldSkipDraw(m_state))) {
        BlessedSkipAo::RecordSkipped();
        return;
      }
      if (unlikely(BlessedSkipVolumetrics::ShouldSkipDraw(m_state))) {
        BlessedSkipVolumetrics::RecordSkippedDraw();
        // blessed: vol-2 -- our trace runs from this draw's hook; the
        // target comes from m_state's bound rtv, so skip only the draw
        BlessedVolumetrics::OnSkippedDraw(static_cast<D3D11ImmediateContext*>(this), m_state);
        return;
      }
      if (unlikely(BlessedSkipBloom::ShouldSkipDraw(m_state))) {
        BlessedSkipBloom::RecordSkipped();
        return;
      }
      if (unlikely(BlessedSkipShadowMask::ShouldSkipDraw(m_state))) {
        BlessedSkipShadowMask::RecordSkipped();
        // blessed: the raster mask draw is dropped, but our traced shadow
        // (and the point-light watcher) is launched from this draw's
        // post-draw hook: run the hooks, skip only the draw itself.
        if (unlikely(BlessedHook::IsEnabled()))
          BlessedHook::OnDraw(static_cast<D3D11ImmediateContext*>(this), m_state);
        if (unlikely(BlessedPointShadow::IsEnabled()))
          BlessedPointShadow::OnDraw(static_cast<D3D11ImmediateContext*>(this), m_state);
        if (unlikely(BlessedVolumetrics::IsEnabled())) // blessed: vol-2, the sun capture
          BlessedVolumetrics::OnDraw(static_cast<D3D11ImmediateContext*>(this), m_state);
        return;
      }
    }

    // blessed: probe dump hook, immediate context only
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedDump::IsCapturing())) {
        BlessedDrawCounts counts;
        counts.vertexOrIndexCount = VertexCount;
        counts.startVertexOrIndex = INT(StartVertexLocation);
        BlessedDump::RecordDraw(m_state, BlessedDumpOp::Draw, counts);
      }
      BlessedAutoInstance::OnDraw(m_state, { false, VertexCount, StartVertexLocation, 0, 1u, 0u }); // blessed: auto-instancing census
    }

    BlessedGpuPassEvent(BlessedGpuPassKind::Draw); // blessed: gpu-pass-timing

    // blessed: look-post, pre-draw -- patches the tonemap pass's ps b2
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedLook::IsEnabled()))
        BlessedLook::OnPreDraw(m_state);
    }

    // blessed: gi ambient patch, pre-draw -- OnDraw only decides; the patch
    // itself runs on the cs thread right before the draw is recorded, via
    // BlessedBatchDraw*Gi (gi-cs, see blessed_gi.h). Immediate context only,
    // same as the post-draw hooks below.
    bool blessedGiPatch = false; // blessed: gi-cs, true: patch on the cs thread
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedGi::IsEnabled()))
        blessedGiPatch = BlessedGi::OnDraw(static_cast<D3D11ImmediateContext*>(this), m_state, m_device.ptr());
    }

    // blessed: rtao, pre-draw -- see blessed_ao.h
    if constexpr (!IsDeferred) {
      // blessed: async-compute -- the vol kick goes first, so rtao's pass at
      // the same draw already overlaps the async trace
      if (unlikely(BlessedVolumetrics::IsAsyncKickEnabled()))
        BlessedVolumetrics::OnDrawPre(static_cast<D3D11ImmediateContext*>(this), m_state);
      if (unlikely(BlessedAo::IsEnabled()))
        BlessedAo::OnDrawPre(static_cast<D3D11ImmediateContext*>(this), m_state);
    }

    VkDrawIndirectCommand draw = { };
    draw.vertexCount   = VertexCount;
    draw.instanceCount = 1u;
    draw.firstVertex   = StartVertexLocation;
    draw.firstInstance = 0u;

    if (unlikely(blessedGiPatch)) // blessed: gi-cs
      BlessedBatchDrawGi(draw);
    else
      BatchDraw(draw);

    // blessed: shader-replace, twin-draw check of a replaced shader
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedShaderVerify::IsEnabled()))
        BlessedShaderVerify::OnDraw(static_cast<D3D11ImmediateContext*>(this), m_state, [&] { BatchDraw(draw); });
    }

    // blessed: post-draw pixel-shader hook, immediate context only
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedHook::IsEnabled()))
        BlessedHook::OnDraw(static_cast<D3D11ImmediateContext*>(this), m_state);
      if (unlikely(BlessedPointShadow::IsEnabled())) // blessed: point-lights
        BlessedPointShadow::OnDraw(static_cast<D3D11ImmediateContext*>(this), m_state);
      if (unlikely(BlessedVolumetrics::IsEnabled())) // blessed: volumetrics
        BlessedVolumetrics::OnDraw(static_cast<D3D11ImmediateContext*>(this), m_state);
      if (unlikely(BlessedLook::IsEnabled())) // blessed: look-post
        BlessedLook::OnDraw(static_cast<D3D11ImmediateContext*>(this), m_state, m_device.ptr());
    }
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::DrawIndexed(
          UINT            IndexCount,
          UINT            StartIndexLocation,
          INT             BaseVertexLocation) {
    BLESSED_PROBE_CALL(ContextType, DrawIndexed);
    D3D10DeviceLock lock = LockContext();

    bool blessedScene = false; // blessed: scene-cs, true: static capture on the cs thread

    if (unlikely(!IndexCount))
      return;

    // blessed: vanilla-vol-async -- every draw entry point closes the
    // window first, if still open, and waits at the configured wait
    // shader (pass 138). See blessed_vol_async.h.
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedVolAsync::IsEnabled()))
        BlessedVolAsync::OnDrawPre(static_cast<D3D11ImmediateContext*>(this), m_state);
    }

    // blessed: half-rate far field -- skip far draws on off frames and
    // track the main lit pass. See blessed_halfrate.h.
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedHalfRate::IsWatching())
       && BlessedHalfRate::OnDraw(static_cast<D3D11ImmediateContext*>(this), m_state, m_device.ptr()))
        return;
      if (unlikely(BlessedReflectHalfRate::IsWatching()) // blessed: refl-harden
       && BlessedReflectHalfRate::OnDraw(m_state))
        return;
    }

    // blessed: BLESSED_SKIP_CASCADES, immediate context only -- a cascade
    // depth draw is dropped before it ever reaches BatchDrawIndexed. See
    // blessed_cascades.h.
    if constexpr (!IsDeferred) {
      BLESSED_PROBE_CALL(ContextType, DrawIndexedCascadeCheck);
      // blessed: perf-halfrate -- pass 138 on off frames (BLESSED_VOL_HALFRATE=2).
      // See blessed_vanilla_halfrate.h.
      if (unlikely(BlessedVolHalfRate::ShouldSkipDraw(m_state)))
        return;
      if (unlikely(BlessedCascadeSkip::ShouldSkipDraw(m_state))) {
        BlessedCascadeSkip::RecordSkippedDraw();
        return;
      }
      if (unlikely(BlessedCascadeCache::IsEnabled()) && BlessedCascadeCache::OnDraw( // blessed: cascade-cache
            static_cast<D3D11ImmediateContext*>(this), m_state, BlessedCascadeDrawKind::Indexed,
            IndexCount, StartIndexLocation, BaseVertexLocation))
        return;
      if (unlikely(BlessedPointShadow::ShouldSkipDraw(static_cast<D3D11ImmediateContext*>(this), m_state))) // blessed: point-lights
        return;
      // blessed: gpu track step 2 -- skip vanilla work our replacements
      // already overwrite. See blessed_skip_replaced.h.
      if (unlikely(BlessedSkipAo::ShouldSkipDraw(m_state))) {
        BlessedSkipAo::RecordSkipped();
        return;
      }
      if (unlikely(BlessedSkipVolumetrics::ShouldSkipDraw(m_state))) {
        BlessedSkipVolumetrics::RecordSkippedDraw();
        // blessed: vol-2 -- our trace runs from this draw's hook; the
        // target comes from m_state's bound rtv, so skip only the draw
        BlessedVolumetrics::OnSkippedDraw(static_cast<D3D11ImmediateContext*>(this), m_state);
        return;
      }
      if (unlikely(BlessedSkipBloom::ShouldSkipDraw(m_state))) {
        BlessedSkipBloom::RecordSkipped();
        return;
      }
      if (unlikely(BlessedSkipShadowMask::ShouldSkipDraw(m_state))) {
        BlessedSkipShadowMask::RecordSkipped();
        // blessed: the raster mask draw is dropped, but our traced shadow
        // (and the point-light watcher) is launched from this draw's
        // post-draw hook: run the hooks, skip only the draw itself.
        if (unlikely(BlessedHook::IsEnabled()))
          BlessedHook::OnDraw(static_cast<D3D11ImmediateContext*>(this), m_state);
        if (unlikely(BlessedPointShadow::IsEnabled()))
          BlessedPointShadow::OnDraw(static_cast<D3D11ImmediateContext*>(this), m_state);
        if (unlikely(BlessedVolumetrics::IsEnabled())) // blessed: vol-2, the sun capture
          BlessedVolumetrics::OnDraw(static_cast<D3D11ImmediateContext*>(this), m_state);
        return;
      }
    }

    // blessed: probe dump hook, immediate context only
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedDump::IsCapturing())) {
        BlessedDrawCounts counts;
        counts.vertexOrIndexCount = IndexCount;
        counts.startVertexOrIndex = INT(StartIndexLocation);
        counts.baseVertex         = BaseVertexLocation;
        BlessedDump::RecordDraw(m_state, BlessedDumpOp::DrawIndexed, counts);
      }
      BlessedAutoInstance::OnDraw(m_state, { true, IndexCount, StartIndexLocation, BaseVertexLocation, 1u, 0u }); // blessed: auto-instancing census

      BlessedGpuPassEvent(BlessedGpuPassKind::Draw); // blessed: gpu-pass-timing

      // blessed: scene-capture hook, immediate context only
      if (unlikely(BlessedSceneCapture::IsEnabled())) {
        BLESSED_PROBE_CALL(ContextType, DrawIndexedSceneCapture);
        blessedScene = BlessedSceneCaptureDraw(IndexCount, StartIndexLocation, BaseVertexLocation, &m_blessedSceneNext); // blessed: scene-cs
      }
    }

    // blessed: look-post, pre-draw -- patches the tonemap pass's ps b2
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedLook::IsEnabled()))
        BlessedLook::OnPreDraw(m_state);
    }

    // blessed: gi ambient patch, pre-draw -- see the comment on Draw() above.
    // gi v1: this draw's own index-buffer params, so OnDraw can key the same
    // CacheKey scene-capture would for this mesh (see BlessedGiDrawIndices).
    bool blessedGiPatch = false; // blessed: gi-cs, true: patch on the cs thread
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedGi::IsEnabled())) {
        BLESSED_PROBE_CALL(ContextType, DrawIndexedGi);
        BlessedGiDrawIndices giIndices;
        giIndices.valid      = true;
        giIndices.indexCount = IndexCount;
        giIndices.startIndex = StartIndexLocation;
        giIndices.baseVertex = BaseVertexLocation;
        blessedGiPatch = BlessedGi::OnDraw(static_cast<D3D11ImmediateContext*>(this), m_state, m_device.ptr(), giIndices, &m_blessedGiNext); // blessed: gi-bounds
      }
    }

    VkDrawIndexedIndirectCommand draw = { };
    draw.indexCount    = IndexCount;
    draw.instanceCount = 1u;
    draw.firstIndex    = StartIndexLocation;
    draw.vertexOffset  = BaseVertexLocation;
    draw.firstInstance = 0u;

    // blessed: rtao, pre-draw -- see blessed_ao.h
    if constexpr (!IsDeferred) {
      // blessed: async-compute -- the vol kick goes first, so rtao's pass at
      // the same draw already overlaps the async trace
      if (unlikely(BlessedVolumetrics::IsAsyncKickEnabled()))
        BlessedVolumetrics::OnDrawPre(static_cast<D3D11ImmediateContext*>(this), m_state);
      if (unlikely(BlessedAo::IsEnabled()))
        BlessedAo::OnDrawPre(static_cast<D3D11ImmediateContext*>(this), m_state);
    }

    { BLESSED_PROBE_CALL(ContextType, DrawIndexedCore);
      BlessedBatchDrawIndexedAny(draw, blessedGiPatch, blessedScene); // blessed: gi-cs, scene-cs
    }

    // blessed: shader-replace, twin-draw check of a replaced shader
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedShaderVerify::IsEnabled()))
        BlessedShaderVerify::OnDraw(static_cast<D3D11ImmediateContext*>(this), m_state, [&] { BatchDrawIndexed(draw); });
    }

    // blessed: post-draw pixel-shader hook, immediate context only
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedHook::IsEnabled())) {
        BLESSED_PROBE_CALL(ContextType, DrawIndexedPostHook);
        BlessedHook::OnDraw(static_cast<D3D11ImmediateContext*>(this), m_state);
      }
      if (unlikely(BlessedPointShadow::IsEnabled())) // blessed: point-lights
        BlessedPointShadow::OnDraw(static_cast<D3D11ImmediateContext*>(this), m_state);
      if (unlikely(BlessedVolumetrics::IsEnabled())) // blessed: volumetrics
        BlessedVolumetrics::OnDraw(static_cast<D3D11ImmediateContext*>(this), m_state);
      if (unlikely(BlessedLook::IsEnabled())) // blessed: look-post
        BlessedLook::OnDraw(static_cast<D3D11ImmediateContext*>(this), m_state, m_device.ptr());
    }
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::DrawInstanced(
          UINT            VertexCountPerInstance,
          UINT            InstanceCount,
          UINT            StartVertexLocation,
          UINT            StartInstanceLocation) {
    BLESSED_PROBE_CALL(ContextType, DrawInstanced);
    D3D10DeviceLock lock = LockContext();

    if (unlikely(!VertexCountPerInstance || !InstanceCount))
      return;

    // blessed: vanilla-vol-async -- every draw entry point closes the
    // window first, if still open. See blessed_vol_async.h.
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedVolAsync::IsEnabled()))
        BlessedVolAsync::OnDrawPre(static_cast<D3D11ImmediateContext*>(this), m_state);
    }

    // blessed: half-rate far field -- skip far draws on off frames and
    // track the main lit pass. See blessed_halfrate.h.
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedHalfRate::IsWatching())
       && BlessedHalfRate::OnDraw(static_cast<D3D11ImmediateContext*>(this), m_state, m_device.ptr()))
        return;
      if (unlikely(BlessedReflectHalfRate::IsWatching()) // blessed: refl-harden
       && BlessedReflectHalfRate::OnDraw(m_state))
        return;
    }

    // blessed: BLESSED_SKIP_CASCADES, immediate context only -- a cascade
    // depth draw is dropped before it ever reaches BatchDraw. See
    // blessed_cascades.h.
    if constexpr (!IsDeferred) {
      // blessed: perf-halfrate -- pass 138 on off frames (BLESSED_VOL_HALFRATE=2).
      // See blessed_vanilla_halfrate.h.
      if (unlikely(BlessedVolHalfRate::ShouldSkipDraw(m_state)))
        return;
      if (unlikely(BlessedCascadeSkip::ShouldSkipDraw(m_state))) {
        BlessedCascadeSkip::RecordSkippedDraw();
        return;
      }
      if (unlikely(BlessedCascadeCache::IsEnabled()) && BlessedCascadeCache::OnDraw( // blessed: cascade-cache, never cached
            static_cast<D3D11ImmediateContext*>(this), m_state, BlessedCascadeDrawKind::Other, 0u, 0u, 0))
        return;
      if (unlikely(BlessedPointShadow::ShouldSkipDraw(static_cast<D3D11ImmediateContext*>(this), m_state))) // blessed: point-lights
        return;
      // blessed: gpu track step 2 -- skip vanilla work our replacements
      // already overwrite. See blessed_skip_replaced.h.
      if (unlikely(BlessedSkipAo::ShouldSkipDraw(m_state))) {
        BlessedSkipAo::RecordSkipped();
        return;
      }
      if (unlikely(BlessedSkipVolumetrics::ShouldSkipDraw(m_state))) {
        BlessedSkipVolumetrics::RecordSkippedDraw();
        // blessed: vol-2 -- our trace runs from this draw's hook; the
        // target comes from m_state's bound rtv, so skip only the draw
        BlessedVolumetrics::OnSkippedDraw(static_cast<D3D11ImmediateContext*>(this), m_state);
        return;
      }
      if (unlikely(BlessedSkipBloom::ShouldSkipDraw(m_state))) {
        BlessedSkipBloom::RecordSkipped();
        return;
      }
      if (unlikely(BlessedSkipShadowMask::ShouldSkipDraw(m_state))) {
        BlessedSkipShadowMask::RecordSkipped();
        // blessed: the raster mask draw is dropped, but our traced shadow
        // (and the point-light watcher) is launched from this draw's
        // post-draw hook: run the hooks, skip only the draw itself.
        if (unlikely(BlessedHook::IsEnabled()))
          BlessedHook::OnDraw(static_cast<D3D11ImmediateContext*>(this), m_state);
        if (unlikely(BlessedPointShadow::IsEnabled()))
          BlessedPointShadow::OnDraw(static_cast<D3D11ImmediateContext*>(this), m_state);
        if (unlikely(BlessedVolumetrics::IsEnabled())) // blessed: vol-2, the sun capture
          BlessedVolumetrics::OnDraw(static_cast<D3D11ImmediateContext*>(this), m_state);
        return;
      }
    }

    // blessed: probe dump hook, immediate context only
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedDump::IsCapturing())) {
        BlessedDrawCounts counts;
        counts.vertexOrIndexCount = VertexCountPerInstance;
        counts.instanceCount      = InstanceCount;
        counts.startVertexOrIndex = INT(StartVertexLocation);
        counts.startInstance      = StartInstanceLocation;
        BlessedDump::RecordDraw(m_state, BlessedDumpOp::DrawInstanced, counts);
      }
      BlessedAutoInstance::OnDraw(m_state, { false, VertexCountPerInstance, StartVertexLocation, 0, InstanceCount, StartInstanceLocation }); // blessed: auto-instancing census
    }

    BlessedGpuPassEvent(BlessedGpuPassKind::Draw); // blessed: gpu-pass-timing

    // blessed: look-post, pre-draw -- patches the tonemap pass's ps b2
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedLook::IsEnabled()))
        BlessedLook::OnPreDraw(m_state);
    }

    // blessed: gi ambient patch, pre-draw -- see the comment on Draw() above.
    bool blessedGiPatch = false; // blessed: gi-cs, true: patch on the cs thread
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedGi::IsEnabled()))
        blessedGiPatch = BlessedGi::OnDraw(static_cast<D3D11ImmediateContext*>(this), m_state, m_device.ptr());
    }

    VkDrawIndirectCommand draw = { };
    draw.vertexCount   = VertexCountPerInstance;
    draw.instanceCount = InstanceCount;
    draw.firstVertex   = StartVertexLocation;
    draw.firstInstance = StartInstanceLocation;

    if (unlikely(blessedGiPatch)) // blessed: gi-cs
      BlessedBatchDrawGi(draw);
    else
      BatchDraw(draw);

    // blessed: shader-replace, twin-draw check of a replaced shader
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedShaderVerify::IsEnabled()))
        BlessedShaderVerify::OnDraw(static_cast<D3D11ImmediateContext*>(this), m_state, [&] { BatchDraw(draw); });
    }

    // blessed: post-draw pixel-shader hook, immediate context only
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedHook::IsEnabled()))
        BlessedHook::OnDraw(static_cast<D3D11ImmediateContext*>(this), m_state);
      if (unlikely(BlessedPointShadow::IsEnabled())) // blessed: point-lights
        BlessedPointShadow::OnDraw(static_cast<D3D11ImmediateContext*>(this), m_state);
      if (unlikely(BlessedVolumetrics::IsEnabled())) // blessed: volumetrics
        BlessedVolumetrics::OnDraw(static_cast<D3D11ImmediateContext*>(this), m_state);
      if (unlikely(BlessedLook::IsEnabled())) // blessed: look-post
        BlessedLook::OnDraw(static_cast<D3D11ImmediateContext*>(this), m_state, m_device.ptr());
    }
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::DrawIndexedInstanced(
          UINT            IndexCountPerInstance,
          UINT            InstanceCount,
          UINT            StartIndexLocation,
          INT             BaseVertexLocation,
          UINT            StartInstanceLocation) {
    BLESSED_PROBE_CALL(ContextType, DrawIndexedInstanced);
    D3D10DeviceLock lock = LockContext();

    bool blessedScene = false; // blessed: scene-cs, true: static capture on the cs thread

    if (unlikely(!IndexCountPerInstance || !InstanceCount))
      return;

    // blessed: vanilla-vol-async -- every draw entry point closes the
    // window first, if still open. See blessed_vol_async.h.
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedVolAsync::IsEnabled()))
        BlessedVolAsync::OnDrawPre(static_cast<D3D11ImmediateContext*>(this), m_state);
    }

    // blessed: half-rate far field -- skip far draws on off frames and
    // track the main lit pass. See blessed_halfrate.h.
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedHalfRate::IsWatching())
       && BlessedHalfRate::OnDraw(static_cast<D3D11ImmediateContext*>(this), m_state, m_device.ptr()))
        return;
      if (unlikely(BlessedReflectHalfRate::IsWatching()) // blessed: refl-harden
       && BlessedReflectHalfRate::OnDraw(m_state))
        return;
    }

    // blessed: BLESSED_SKIP_CASCADES, immediate context only -- a cascade
    // depth draw is dropped before it ever reaches BatchDrawIndexed. See
    // blessed_cascades.h.
    if constexpr (!IsDeferred) {
      // blessed: perf-halfrate -- pass 138 on off frames (BLESSED_VOL_HALFRATE=2).
      // See blessed_vanilla_halfrate.h.
      if (unlikely(BlessedVolHalfRate::ShouldSkipDraw(m_state)))
        return;
      if (unlikely(BlessedCascadeSkip::ShouldSkipDraw(m_state))) {
        BlessedCascadeSkip::RecordSkippedDraw();
        return;
      }
      if (unlikely(BlessedCascadeCache::IsEnabled()) && BlessedCascadeCache::OnDraw( // blessed: cascade-cache, never cached
            static_cast<D3D11ImmediateContext*>(this), m_state, BlessedCascadeDrawKind::Other, 0u, 0u, 0))
        return;
      if (unlikely(BlessedPointShadow::ShouldSkipDraw(static_cast<D3D11ImmediateContext*>(this), m_state))) // blessed: point-lights
        return;
      // blessed: gpu track step 2 -- skip vanilla work our replacements
      // already overwrite. See blessed_skip_replaced.h.
      if (unlikely(BlessedSkipAo::ShouldSkipDraw(m_state))) {
        BlessedSkipAo::RecordSkipped();
        return;
      }
      if (unlikely(BlessedSkipVolumetrics::ShouldSkipDraw(m_state))) {
        BlessedSkipVolumetrics::RecordSkippedDraw();
        // blessed: vol-2 -- our trace runs from this draw's hook; the
        // target comes from m_state's bound rtv, so skip only the draw
        BlessedVolumetrics::OnSkippedDraw(static_cast<D3D11ImmediateContext*>(this), m_state);
        return;
      }
      if (unlikely(BlessedSkipBloom::ShouldSkipDraw(m_state))) {
        BlessedSkipBloom::RecordSkipped();
        return;
      }
      if (unlikely(BlessedSkipShadowMask::ShouldSkipDraw(m_state))) {
        BlessedSkipShadowMask::RecordSkipped();
        // blessed: the raster mask draw is dropped, but our traced shadow
        // (and the point-light watcher) is launched from this draw's
        // post-draw hook: run the hooks, skip only the draw itself.
        if (unlikely(BlessedHook::IsEnabled()))
          BlessedHook::OnDraw(static_cast<D3D11ImmediateContext*>(this), m_state);
        if (unlikely(BlessedPointShadow::IsEnabled()))
          BlessedPointShadow::OnDraw(static_cast<D3D11ImmediateContext*>(this), m_state);
        if (unlikely(BlessedVolumetrics::IsEnabled())) // blessed: vol-2, the sun capture
          BlessedVolumetrics::OnDraw(static_cast<D3D11ImmediateContext*>(this), m_state);
        return;
      }
    }

    // blessed: probe dump hook, immediate context only
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedDump::IsCapturing())) {
        BlessedDrawCounts counts;
        counts.vertexOrIndexCount = IndexCountPerInstance;
        counts.instanceCount      = InstanceCount;
        counts.startVertexOrIndex = INT(StartIndexLocation);
        counts.baseVertex         = BaseVertexLocation;
        counts.startInstance      = StartInstanceLocation;
        BlessedDump::RecordDraw(m_state, BlessedDumpOp::DrawIndexedInstanced, counts);
      }
      BlessedAutoInstance::OnDraw(m_state, { true, IndexCountPerInstance, StartIndexLocation, BaseVertexLocation, InstanceCount, StartInstanceLocation }); // blessed: auto-instancing census

      BlessedGpuPassEvent(BlessedGpuPassKind::Draw); // blessed: gpu-pass-timing

      // blessed: scene-capture hook, immediate context only. Multi-instance
      // draws would need a transform per instance, which this seat doesn't
      // read (BLESSED_SCENE_XFORM points at one cbuffer slot/offset for the
      // whole draw) -- so only the InstanceCount == 1 case is captured; see
      // the seat report for why this stayed out of scope.
      if (unlikely(BlessedSceneCapture::IsEnabled()) && InstanceCount == 1u)
        blessedScene = BlessedSceneCaptureDraw(IndexCountPerInstance, StartIndexLocation, BaseVertexLocation, &m_blessedSceneNext); // blessed: scene-cs
    }

    // blessed: look-post, pre-draw -- patches the tonemap pass's ps b2
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedLook::IsEnabled()))
        BlessedLook::OnPreDraw(m_state);
    }

    // blessed: gi ambient patch, pre-draw -- see the comment on Draw() above.
    // gi v1: same InstanceCount == 1 restriction as the scene-capture hook
    // above -- a multi-instance draw's mesh/albedo association would need a
    // transform per instance this seat doesn't read.
    bool blessedGiPatch = false; // blessed: gi-cs, true: patch on the cs thread
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedGi::IsEnabled())) {
        BlessedGiDrawIndices giIndices;
        if (InstanceCount == 1u) {
          giIndices.valid      = true;
          giIndices.indexCount = IndexCountPerInstance;
          giIndices.startIndex = StartIndexLocation;
          giIndices.baseVertex = BaseVertexLocation;
        }
        blessedGiPatch = BlessedGi::OnDraw(static_cast<D3D11ImmediateContext*>(this), m_state, m_device.ptr(), giIndices, &m_blessedGiNext); // blessed: gi-cs, gi-bounds
      }
    }

    VkDrawIndexedIndirectCommand draw = { };
    draw.indexCount    = IndexCountPerInstance;
    draw.instanceCount = InstanceCount;
    draw.firstIndex    = StartIndexLocation;
    draw.vertexOffset  = BaseVertexLocation;
    draw.firstInstance = StartInstanceLocation;

    BlessedBatchDrawIndexedAny(draw, blessedGiPatch, blessedScene); // blessed: gi-cs, scene-cs

    // blessed: shader-replace, twin-draw check of a replaced shader
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedShaderVerify::IsEnabled()))
        BlessedShaderVerify::OnDraw(static_cast<D3D11ImmediateContext*>(this), m_state, [&] { BatchDrawIndexed(draw); });
    }

    // blessed: post-draw pixel-shader hook, immediate context only
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedHook::IsEnabled()))
        BlessedHook::OnDraw(static_cast<D3D11ImmediateContext*>(this), m_state);
      if (unlikely(BlessedPointShadow::IsEnabled())) // blessed: point-lights
        BlessedPointShadow::OnDraw(static_cast<D3D11ImmediateContext*>(this), m_state);
      if (unlikely(BlessedVolumetrics::IsEnabled())) // blessed: volumetrics
        BlessedVolumetrics::OnDraw(static_cast<D3D11ImmediateContext*>(this), m_state);
      if (unlikely(BlessedLook::IsEnabled())) // blessed: look-post
        BlessedLook::OnDraw(static_cast<D3D11ImmediateContext*>(this), m_state, m_device.ptr());
    }
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::DrawIndexedInstancedIndirect(
          ID3D11Buffer*   pBufferForArgs,
          UINT            AlignedByteOffsetForArgs) {
    D3D10DeviceLock lock = LockContext();

    // blessed: vanilla-vol-async -- every draw entry point closes the
    // window first, if still open. See blessed_vol_async.h.
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedVolAsync::IsEnabled()))
        BlessedVolAsync::OnDrawPre(static_cast<D3D11ImmediateContext*>(this), m_state);
    }

    // blessed: refl-harden -- a skipped water cube face drops all its draws
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedReflectHalfRate::IsWatching()) // blessed: refl-harden
       && BlessedReflectHalfRate::OnDraw(m_state))
        return;
    }

    if constexpr (!IsDeferred) { // blessed: cascade-cache, an indirect or auto draw may end a cascade (before SetDrawBuffers: ending one resets the context state)
      if (unlikely(BlessedCascadeCache::IsEnabled()))
        BlessedCascadeCache::OnDraw(static_cast<D3D11ImmediateContext*>(this), m_state, BlessedCascadeDrawKind::Other, 0u, 0u, 0);
    }

    SetDrawBuffers(pBufferForArgs, nullptr);

    BlessedGpuPassEvent(BlessedGpuPassKind::Draw); // blessed: gpu-pass-timing

    if (unlikely(HasDirtyGraphicsBindings()))
      ApplyDirtyGraphicsBindings();

    // If possible, batch multiple indirect draw calls into one single multidraw call
    if (m_csDataType == D3D11CmdType::DrawIndirectIndexed) {
      auto cmdData = static_cast<D3D11CmdDrawIndirectData*>(m_csData->first());
      auto stride = GetIndirectCommandStride(cmdData, AlignedByteOffsetForArgs, sizeof(VkDrawIndexedIndirectCommand));

      if (stride) {
        cmdData->count += 1;
        cmdData->stride = stride;
        return;
      }
    }

    // Need to start a new draw sequence
    EmitCsCmd<D3D11CmdDrawIndirectData>(D3D11CmdType::DrawIndirectIndexed, 1u,
      [] (DxvkContext* ctx, const D3D11CmdDrawIndirectData* data, size_t) {
        ctx->drawIndexedIndirect(data->offset, data->count, data->stride, true);
      });

    auto cmdData = new (m_csData->first()) D3D11CmdDrawIndirectData();
    cmdData->offset = AlignedByteOffsetForArgs;
    cmdData->count  = 1;
    cmdData->stride = 0;
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::DrawInstancedIndirect(
          ID3D11Buffer*   pBufferForArgs,
          UINT            AlignedByteOffsetForArgs) {
    D3D10DeviceLock lock = LockContext();

    // blessed: vanilla-vol-async -- every draw entry point closes the
    // window first, if still open. See blessed_vol_async.h.
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedVolAsync::IsEnabled()))
        BlessedVolAsync::OnDrawPre(static_cast<D3D11ImmediateContext*>(this), m_state);
    }

    // blessed: refl-harden -- a skipped water cube face drops all its draws
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedReflectHalfRate::IsWatching()) // blessed: refl-harden
       && BlessedReflectHalfRate::OnDraw(m_state))
        return;
    }

    if constexpr (!IsDeferred) { // blessed: cascade-cache, an indirect or auto draw may end a cascade (before SetDrawBuffers: ending one resets the context state)
      if (unlikely(BlessedCascadeCache::IsEnabled()))
        BlessedCascadeCache::OnDraw(static_cast<D3D11ImmediateContext*>(this), m_state, BlessedCascadeDrawKind::Other, 0u, 0u, 0);
    }

    SetDrawBuffers(pBufferForArgs, nullptr);

    BlessedGpuPassEvent(BlessedGpuPassKind::Draw); // blessed: gpu-pass-timing

    if (unlikely(HasDirtyGraphicsBindings()))
      ApplyDirtyGraphicsBindings();

    // If possible, batch multiple indirect draw calls into one single multidraw call
    if (m_csDataType == D3D11CmdType::DrawIndirect) {
      auto cmdData = static_cast<D3D11CmdDrawIndirectData*>(m_csData->first());
      auto stride = GetIndirectCommandStride(cmdData, AlignedByteOffsetForArgs, sizeof(VkDrawIndirectCommand));

      if (stride) {
        cmdData->count += 1;
        cmdData->stride = stride;
        return;
      }
    }

    // Need to start a new draw sequence
    EmitCsCmd<D3D11CmdDrawIndirectData>(D3D11CmdType::DrawIndirect, 1u,
      [] (DxvkContext* ctx, const D3D11CmdDrawIndirectData* data, size_t) {
        ctx->drawIndirect(data->offset, data->count, data->stride, true);
      });

    auto cmdData = new (m_csData->first()) D3D11CmdDrawIndirectData();
    cmdData->offset = AlignedByteOffsetForArgs;
    cmdData->count  = 1;
    cmdData->stride = 0;
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::Dispatch(
          UINT            ThreadGroupCountX,
          UINT            ThreadGroupCountY,
          UINT            ThreadGroupCountZ) {
    BLESSED_PROBE_CALL(ContextType, Dispatch);
    D3D10DeviceLock lock = LockContext();

    if (unlikely(!ThreadGroupCountX || !ThreadGroupCountY || !ThreadGroupCountZ))
      return;

    // blessed: half-rate far field -- a dispatch closes the main pass
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedHalfRate::IsWatching()))
        BlessedHalfRate::OnDispatch(static_cast<D3D11ImmediateContext*>(this));
      // blessed: refl-harden -- the volumetric generate's camera, before
      // anything below may drop that dispatch
      if (unlikely(BlessedReflectHalfRate::IsWatching()))
        BlessedReflectHalfRate::OnDispatch(m_state);
    }

    // blessed: gpu track step 2, immediate context only, opt-in
    // (BLESSED_VOL_SKIP_RAYMARCH=1) -- see blessed_skip_replaced.h for why
    // this one stays off by default.
    if constexpr (!IsDeferred) {
      // blessed: perf-halfrate -- vanilla volumetric light at half rate
      if (unlikely(BlessedVolHalfRate::ShouldSkipDispatch(m_state)))
        return;
      if (unlikely(BlessedSkipVolumetrics::ShouldSkipDispatch(m_state))) {
        BlessedSkipVolumetrics::RecordSkippedDispatch();
        return;
      }
    }

    // blessed: vanilla-vol-async -- opens the async-queue recording window
    // at the froxel generate dispatch (after the skip checks above, so a
    // half-rate/skipped frame never opens it for a dispatch that will not
    // run). See blessed_vol_async.h.
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedVolAsync::IsEnabled()))
        BlessedVolAsync::OnDispatchPre(static_cast<D3D11ImmediateContext*>(this), m_state);
    }

    // blessed: vol-collapse -- the volumetric z-integration chain as one
    // dispatch, the rest of the chain skipped. See blessed_vol_collapse.h.
    if constexpr (!IsDeferred) {
      // blessed: vol-async-3 -- vol-async-verify's replay of a window
      // reproduces exactly what ran; collapse must not see it again
      if (unlikely(BlessedVolCollapse::IsEnabled()) && !BlessedVolAsyncVerify::IsReplaying()) {
        if (BlessedVolCollapse::OnDispatch(static_cast<D3D11ImmediateContext*>(this), m_state,
            ThreadGroupCountX, ThreadGroupCountY, ThreadGroupCountZ))
          return;
      }
    }

    // blessed: probe dump hook, immediate context only
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedDump::IsCapturing()))
        BlessedDump::RecordDispatch(m_state, ThreadGroupCountX, ThreadGroupCountY, ThreadGroupCountZ);
    }

    BlessedGpuPassEvent(BlessedGpuPassKind::Compute); // blessed: gpu-pass-timing

    AddCost(GpuCostEstimate::Dispatch);

    if (unlikely(HasDirtyComputeBindings()))
      ApplyDirtyComputeBindings();

    // blessed: shader-replace, dispatch-size override for a narrowly matched
    // compacted replacement. Only the real dispatch below is rewritten --
    // the probe dump above and the shader-verify twin-check below still see
    // (and replay) the game's own ThreadGroupCount*, since a rewritten
    // replacement shader tolerates being over-dispatched at the vanilla
    // shape (extra groups return once col/row is out of range) but the real
    // vanilla bytecode used in a twin-check does not.
    UINT dispatchX = ThreadGroupCountX;
    UINT dispatchY = ThreadGroupCountY;
    UINT dispatchZ = ThreadGroupCountZ;

    if constexpr (!IsDeferred) {
      if (unlikely(BlessedDispatchRewrite::IsEnabled()))
        BlessedDispatchRewrite::Adjust(m_state, dispatchX, dispatchY, dispatchZ);
    }

    EmitCs([=] (DxvkContext* ctx) {
      ctx->dispatch(
        dispatchX,
        dispatchY,
        dispatchZ);
    });

    // blessed: vol-collapse, post-dispatch (the verify diff after the chain's last dispatch)
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedVolCollapse::IsEnabled()) && !BlessedVolAsyncVerify::IsReplaying()) // blessed: vol-async-3
        BlessedVolCollapse::OnDispatchDone(static_cast<D3D11ImmediateContext*>(this), m_state);
    }

    // blessed: vanilla-vol-async, post-dispatch (BLESSED_VOL_ASYNC_VERIFY's
    // shadow replay; see blessed_vol_async_verify.h)
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedVolAsyncVerify::IsEnabled())) {
        // the counts that actually ran: a reshaped replacement (volgen) covers
        // its volume only at the rewritten shape, and a rewritten shape never
        // matches a rewrite rule again
        BlessedVolAsyncVerify::OnDispatchDone(static_cast<D3D11ImmediateContext*>(this), m_state,
          dispatchX, dispatchY, dispatchZ);
      }
    }

    // blessed: shader-replace, twin-draw check of a replaced shader
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedShaderVerify::IsEnabled()))
        BlessedShaderVerify::OnDispatch(static_cast<D3D11ImmediateContext*>(this), m_state, [&] { if (unlikely(HasDirtyComputeBindings())) ApplyDirtyComputeBindings(); EmitCs([=] (DxvkContext* ctx) { ctx->dispatch(ThreadGroupCountX, ThreadGroupCountY, ThreadGroupCountZ); }); });
    }
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::DispatchIndirect(
          ID3D11Buffer*   pBufferForArgs,
          UINT            AlignedByteOffsetForArgs) {
    D3D10DeviceLock lock = LockContext();
    SetDrawBuffers(pBufferForArgs, nullptr);

    AddCost(GpuCostEstimate::DispatchIndirect);

    BlessedGpuPassEvent(BlessedGpuPassKind::Compute); // blessed: gpu-pass-timing

    if (unlikely(HasDirtyComputeBindings()))
      ApplyDirtyComputeBindings();

    EmitCs([cOffset = AlignedByteOffsetForArgs]
    (DxvkContext* ctx) {
      ctx->dispatchIndirect(cOffset);
    });
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::IASetInputLayout(ID3D11InputLayout* pInputLayout) {
    BLESSED_PROBE_CALL(ContextType, IASetInputLayout);
    D3D10DeviceLock lock = LockContext();

    auto inputLayout = static_cast<D3D11InputLayout*>(pInputLayout);

    if (m_state.ia.inputLayout != inputLayout) {
      bool equal = false;

      // Some games (e.g. Grim Dawn) create lots and lots of
      // identical input layouts, so we'll only apply the state
      // if the input layouts has actually changed between calls.
      if (m_state.ia.inputLayout != nullptr && inputLayout != nullptr)
        equal = m_state.ia.inputLayout->Compare(inputLayout);

      m_state.ia.inputLayout = inputLayout;

      if (!equal)
        ApplyInputLayout();
    }
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY Topology) {
    BLESSED_PROBE_CALL(ContextType, IASetPrimitiveTopology);
    D3D10DeviceLock lock = LockContext();

    if (m_state.ia.primitiveTopology != Topology) {
      m_state.ia.primitiveTopology = Topology;
      ApplyPrimitiveTopology();
    }
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::IASetVertexBuffers(
          UINT                              StartSlot,
          UINT                              NumBuffers,
          ID3D11Buffer* const*              ppVertexBuffers,
    const UINT*                             pStrides,
    const UINT*                             pOffsets) {
    BLESSED_PROBE_CALL(ContextType, IASetVertexBuffers);
    D3D10DeviceLock lock = LockContext();

    for (uint32_t i = 0; i < NumBuffers; i++) {
      auto newBuffer = static_cast<D3D11Buffer*>(ppVertexBuffers[i]);

      if (m_state.ia.vertexBuffers[StartSlot + i].buffer != newBuffer) {
        m_state.ia.vertexBuffers[StartSlot + i].buffer = newBuffer;
        m_state.ia.vertexBuffers[StartSlot + i].offset = pOffsets[i];
        m_state.ia.vertexBuffers[StartSlot + i].stride = pStrides[i];

        BindVertexBuffer(StartSlot + i, newBuffer, pOffsets[i], pStrides[i]);
      } else if (m_state.ia.vertexBuffers[StartSlot + i].offset != pOffsets[i]
              || m_state.ia.vertexBuffers[StartSlot + i].stride != pStrides[i]) {
        m_state.ia.vertexBuffers[StartSlot + i].offset = pOffsets[i];
        m_state.ia.vertexBuffers[StartSlot + i].stride = pStrides[i];

        BindVertexBufferRange(StartSlot + i, newBuffer, pOffsets[i], pStrides[i]);
      }
    }

    m_state.ia.maxVbCount = std::clamp(StartSlot + NumBuffers,
      m_state.ia.maxVbCount, uint32_t(m_state.ia.vertexBuffers.size()));
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::IASetIndexBuffer(
          ID3D11Buffer*                     pIndexBuffer,
          DXGI_FORMAT                       Format,
          UINT                              Offset) {
    BLESSED_PROBE_CALL(ContextType, IASetIndexBuffer);
    D3D10DeviceLock lock = LockContext();

    auto newBuffer = static_cast<D3D11Buffer*>(pIndexBuffer);

    if (m_state.ia.indexBuffer.buffer != newBuffer) {
      m_state.ia.indexBuffer.buffer = newBuffer;
      m_state.ia.indexBuffer.offset = Offset;
      m_state.ia.indexBuffer.format = Format;

      BindIndexBuffer(newBuffer, Offset, Format);
    } else if (m_state.ia.indexBuffer.offset != Offset
            || m_state.ia.indexBuffer.format != Format) {
      m_state.ia.indexBuffer.offset = Offset;
      m_state.ia.indexBuffer.format = Format;

      BindIndexBufferRange(newBuffer, Offset, Format);
    }
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::IAGetInputLayout(ID3D11InputLayout** ppInputLayout) {
    D3D10DeviceLock lock = LockContext();

    *ppInputLayout = m_state.ia.inputLayout.ref();
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::IAGetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY* pTopology) {
    D3D10DeviceLock lock = LockContext();

    *pTopology = m_state.ia.primitiveTopology;
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::IAGetVertexBuffers(
          UINT                              StartSlot,
          UINT                              NumBuffers,
          ID3D11Buffer**                    ppVertexBuffers,
          UINT*                             pStrides,
          UINT*                             pOffsets) {
    D3D10DeviceLock lock = LockContext();

    for (uint32_t i = 0; i < NumBuffers; i++) {
      const bool inRange = StartSlot + i < m_state.ia.vertexBuffers.size();

      if (ppVertexBuffers) {
        ppVertexBuffers[i] = inRange
          ? m_state.ia.vertexBuffers[StartSlot + i].buffer.ref()
          : nullptr;
      }

      if (pStrides) {
        pStrides[i] = inRange
          ? m_state.ia.vertexBuffers[StartSlot + i].stride
          : 0u;
      }

      if (pOffsets) {
        pOffsets[i] = inRange
          ? m_state.ia.vertexBuffers[StartSlot + i].offset
          : 0u;
      }
    }
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::IAGetIndexBuffer(
          ID3D11Buffer**                    ppIndexBuffer,
          DXGI_FORMAT*                      pFormat,
          UINT*                             pOffset) {
    D3D10DeviceLock lock = LockContext();

    if (ppIndexBuffer)
      *ppIndexBuffer = m_state.ia.indexBuffer.buffer.ref();

    if (pFormat)
      *pFormat = m_state.ia.indexBuffer.format;

    if (pOffset)
      *pOffset = m_state.ia.indexBuffer.offset;
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::VSSetShader(
          ID3D11VertexShader*               pVertexShader,
          ID3D11ClassInstance* const*       ppClassInstances,
          UINT                              NumClassInstances) {
    BLESSED_PROBE_CALL(ContextType, SetShader);
    D3D10DeviceLock lock = LockContext();

    auto shader = static_cast<D3D11VertexShader*>(pVertexShader);
    SetClassInstances<D3D11ShaderType::eVertex>(
      shader ? shader->GetCommonShader() : nullptr, ppClassInstances, NumClassInstances);

    if (m_state.vs != shader) {
      m_state.vs = shader;

      BindShader<D3D11ShaderType::eVertex>(GetCommonShader(shader));
    }
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::VSSetConstantBuffers(
          UINT                              StartSlot,
          UINT                              NumBuffers,
          ID3D11Buffer* const*              ppConstantBuffers) {
    BLESSED_PROBE_CALL(ContextType, SetConstantBuffers);
    D3D10DeviceLock lock = LockContext();

    SetConstantBuffers<D3D11ShaderType::eVertex>(
      StartSlot, NumBuffers, ppConstantBuffers);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::VSSetConstantBuffers1(
          UINT                              StartSlot,
          UINT                              NumBuffers,
          ID3D11Buffer* const*              ppConstantBuffers,
    const UINT*                             pFirstConstant,
    const UINT*                             pNumConstants) {
    BLESSED_PROBE_CALL(ContextType, SetConstantBuffers);
    D3D10DeviceLock lock = LockContext();

    SetConstantBuffers1<D3D11ShaderType::eVertex>(
      StartSlot, NumBuffers, ppConstantBuffers,
      pFirstConstant, pNumConstants);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::VSSetShaderResources(
          UINT                              StartSlot,
          UINT                              NumViews,
          ID3D11ShaderResourceView* const*  ppShaderResourceViews) {
    BLESSED_PROBE_CALL(ContextType, SetShaderResources);
    D3D10DeviceLock lock = LockContext();

    SetShaderResources<D3D11ShaderType::eVertex>(
      StartSlot, NumViews, ppShaderResourceViews);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::VSSetSamplers(
          UINT                              StartSlot,
          UINT                              NumSamplers,
          ID3D11SamplerState* const*        ppSamplers) {
    BLESSED_PROBE_CALL(ContextType, SetSamplers);
    D3D10DeviceLock lock = LockContext();

    SetSamplers<D3D11ShaderType::eVertex>(
      StartSlot, NumSamplers, ppSamplers);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::VSGetShader(
          ID3D11VertexShader**              ppVertexShader,
          ID3D11ClassInstance**             ppClassInstances,
          UINT*                             pNumClassInstances) {
    D3D10DeviceLock lock = LockContext();

    if (ppVertexShader)
      *ppVertexShader = m_state.vs.ref();

    GetClassInstances<D3D11ShaderType::eVertex>(ppClassInstances, pNumClassInstances);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::VSGetConstantBuffers(
          UINT                              StartSlot,
          UINT                              NumBuffers,
          ID3D11Buffer**                    ppConstantBuffers) {
    D3D10DeviceLock lock = LockContext();

    GetConstantBuffers<D3D11ShaderType::eVertex>(
      StartSlot, NumBuffers, ppConstantBuffers,
      nullptr, nullptr);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::VSGetConstantBuffers1(
          UINT                              StartSlot,
          UINT                              NumBuffers,
          ID3D11Buffer**                    ppConstantBuffers,
          UINT*                             pFirstConstant,
          UINT*                             pNumConstants) {
    D3D10DeviceLock lock = LockContext();

    GetConstantBuffers<D3D11ShaderType::eVertex>(
      StartSlot, NumBuffers, ppConstantBuffers,
      pFirstConstant, pNumConstants);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::VSGetShaderResources(
          UINT                              StartSlot,
          UINT                              NumViews,
          ID3D11ShaderResourceView**        ppShaderResourceViews) {
    D3D10DeviceLock lock = LockContext();

    GetShaderResources<D3D11ShaderType::eVertex>(
      StartSlot, NumViews, ppShaderResourceViews);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::VSGetSamplers(
          UINT                              StartSlot,
          UINT                              NumSamplers,
          ID3D11SamplerState**              ppSamplers) {
    D3D10DeviceLock lock = LockContext();

    GetSamplers<D3D11ShaderType::eVertex>(
      StartSlot, NumSamplers, ppSamplers);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::HSSetShader(
          ID3D11HullShader*                 pHullShader,
          ID3D11ClassInstance* const*       ppClassInstances,
          UINT                              NumClassInstances) {
    BLESSED_PROBE_CALL(ContextType, SetShader);
    D3D10DeviceLock lock = LockContext();

    auto shader = static_cast<D3D11HullShader*>(pHullShader);
    SetClassInstances<D3D11ShaderType::eHull>(
      shader ? shader->GetCommonShader() : nullptr, ppClassInstances, NumClassInstances);

    if (m_state.hs != shader) {
      m_state.hs = shader;

      BindShader<D3D11ShaderType::eHull>(GetCommonShader(shader));
    }
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::HSSetConstantBuffers(
          UINT                              StartSlot,
          UINT                              NumBuffers,
          ID3D11Buffer* const*              ppConstantBuffers) {
    BLESSED_PROBE_CALL(ContextType, SetConstantBuffers);
    D3D10DeviceLock lock = LockContext();

    SetConstantBuffers<D3D11ShaderType::eHull>(
      StartSlot, NumBuffers, ppConstantBuffers);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::HSSetConstantBuffers1(
          UINT                              StartSlot,
          UINT                              NumBuffers,
          ID3D11Buffer* const*              ppConstantBuffers,
    const UINT*                             pFirstConstant,
    const UINT*                             pNumConstants) {
    BLESSED_PROBE_CALL(ContextType, SetConstantBuffers);
    D3D10DeviceLock lock = LockContext();

    SetConstantBuffers1<D3D11ShaderType::eHull>(
      StartSlot, NumBuffers, ppConstantBuffers,
      pFirstConstant, pNumConstants);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::HSSetShaderResources(
          UINT                              StartSlot,
          UINT                              NumViews,
          ID3D11ShaderResourceView* const*  ppShaderResourceViews) {
    BLESSED_PROBE_CALL(ContextType, SetShaderResources);
    D3D10DeviceLock lock = LockContext();

    SetShaderResources<D3D11ShaderType::eHull>(
      StartSlot, NumViews, ppShaderResourceViews);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::HSSetSamplers(
          UINT                              StartSlot,
          UINT                              NumSamplers,
          ID3D11SamplerState* const*        ppSamplers) {
    BLESSED_PROBE_CALL(ContextType, SetSamplers);
    D3D10DeviceLock lock = LockContext();

    SetSamplers<D3D11ShaderType::eHull>(
      StartSlot, NumSamplers, ppSamplers);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::HSGetShader(
          ID3D11HullShader**                ppHullShader,
          ID3D11ClassInstance**             ppClassInstances,
          UINT*                             pNumClassInstances) {
    D3D10DeviceLock lock = LockContext();

    if (ppHullShader)
      *ppHullShader = m_state.hs.ref();

    GetClassInstances<D3D11ShaderType::eHull>(ppClassInstances, pNumClassInstances);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::HSGetConstantBuffers(
          UINT                              StartSlot,
          UINT                              NumBuffers,
          ID3D11Buffer**                    ppConstantBuffers) {
    D3D10DeviceLock lock = LockContext();

    GetConstantBuffers<D3D11ShaderType::eHull>(
      StartSlot, NumBuffers, ppConstantBuffers,
      nullptr, nullptr);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::HSGetConstantBuffers1(
          UINT                              StartSlot,
          UINT                              NumBuffers,
          ID3D11Buffer**                    ppConstantBuffers,
          UINT*                             pFirstConstant,
          UINT*                             pNumConstants) {
    D3D10DeviceLock lock = LockContext();

    GetConstantBuffers<D3D11ShaderType::eHull>(
      StartSlot, NumBuffers, ppConstantBuffers,
      pFirstConstant, pNumConstants);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::HSGetShaderResources(
          UINT                              StartSlot,
          UINT                              NumViews,
          ID3D11ShaderResourceView**        ppShaderResourceViews) {
    D3D10DeviceLock lock = LockContext();

    GetShaderResources<D3D11ShaderType::eHull>(
      StartSlot, NumViews, ppShaderResourceViews);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::HSGetSamplers(
          UINT                              StartSlot,
          UINT                              NumSamplers,
          ID3D11SamplerState**              ppSamplers) {
    D3D10DeviceLock lock = LockContext();

    GetSamplers<D3D11ShaderType::eHull>(
      StartSlot, NumSamplers, ppSamplers);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::DSSetShader(
          ID3D11DomainShader*               pDomainShader,
          ID3D11ClassInstance* const*       ppClassInstances,
          UINT                              NumClassInstances) {
    BLESSED_PROBE_CALL(ContextType, SetShader);
    D3D10DeviceLock lock = LockContext();

    auto shader = static_cast<D3D11DomainShader*>(pDomainShader);
    SetClassInstances<D3D11ShaderType::eDomain>(
      shader ? shader->GetCommonShader() : nullptr, ppClassInstances, NumClassInstances);

    if (m_state.ds != shader) {
      m_state.ds = shader;

      BindShader<D3D11ShaderType::eDomain>(GetCommonShader(shader));
    }
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::DSSetConstantBuffers(
          UINT                              StartSlot,
          UINT                              NumBuffers,
          ID3D11Buffer* const*              ppConstantBuffers) {
    BLESSED_PROBE_CALL(ContextType, SetConstantBuffers);
    D3D10DeviceLock lock = LockContext();

    SetConstantBuffers<D3D11ShaderType::eDomain>(
      StartSlot, NumBuffers, ppConstantBuffers);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::DSSetConstantBuffers1(
          UINT                              StartSlot,
          UINT                              NumBuffers,
          ID3D11Buffer* const*              ppConstantBuffers,
    const UINT*                             pFirstConstant,
    const UINT*                             pNumConstants) {
    BLESSED_PROBE_CALL(ContextType, SetConstantBuffers);
    D3D10DeviceLock lock = LockContext();

    SetConstantBuffers1<D3D11ShaderType::eDomain>(
      StartSlot, NumBuffers, ppConstantBuffers,
      pFirstConstant, pNumConstants);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::DSSetShaderResources(
          UINT                              StartSlot,
          UINT                              NumViews,
          ID3D11ShaderResourceView* const*  ppShaderResourceViews) {
    BLESSED_PROBE_CALL(ContextType, SetShaderResources);
    D3D10DeviceLock lock = LockContext();

    SetShaderResources<D3D11ShaderType::eDomain>(
      StartSlot, NumViews, ppShaderResourceViews);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::DSSetSamplers(
          UINT                              StartSlot,
          UINT                              NumSamplers,
          ID3D11SamplerState* const*        ppSamplers) {
    BLESSED_PROBE_CALL(ContextType, SetSamplers);
    D3D10DeviceLock lock = LockContext();

    SetSamplers<D3D11ShaderType::eDomain>(
      StartSlot, NumSamplers, ppSamplers);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::DSGetShader(
          ID3D11DomainShader**              ppDomainShader,
          ID3D11ClassInstance**             ppClassInstances,
          UINT*                             pNumClassInstances) {
    D3D10DeviceLock lock = LockContext();

    if (ppDomainShader)
      *ppDomainShader = m_state.ds.ref();

    GetClassInstances<D3D11ShaderType::eDomain>(ppClassInstances, pNumClassInstances);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::DSGetConstantBuffers(
          UINT                              StartSlot,
          UINT                              NumBuffers,
          ID3D11Buffer**                    ppConstantBuffers) {
    D3D10DeviceLock lock = LockContext();

    GetConstantBuffers<D3D11ShaderType::eDomain>(
      StartSlot, NumBuffers, ppConstantBuffers,
      nullptr, nullptr);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::DSGetConstantBuffers1(
          UINT                              StartSlot,
          UINT                              NumBuffers,
          ID3D11Buffer**                    ppConstantBuffers,
          UINT*                             pFirstConstant,
          UINT*                             pNumConstants) {
    D3D10DeviceLock lock = LockContext();

    GetConstantBuffers<D3D11ShaderType::eDomain>(
      StartSlot, NumBuffers, ppConstantBuffers,
      pFirstConstant, pNumConstants);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::DSGetShaderResources(
          UINT                              StartSlot,
          UINT                              NumViews,
          ID3D11ShaderResourceView**        ppShaderResourceViews) {
    D3D10DeviceLock lock = LockContext();

    GetShaderResources<D3D11ShaderType::eDomain>(
      StartSlot, NumViews, ppShaderResourceViews);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::DSGetSamplers(
          UINT                              StartSlot,
          UINT                              NumSamplers,
          ID3D11SamplerState**              ppSamplers) {
    D3D10DeviceLock lock = LockContext();

    GetSamplers<D3D11ShaderType::eDomain>(
      StartSlot, NumSamplers, ppSamplers);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::GSSetShader(
          ID3D11GeometryShader*             pShader,
          ID3D11ClassInstance* const*       ppClassInstances,
          UINT                              NumClassInstances) {
    BLESSED_PROBE_CALL(ContextType, SetShader);
    D3D10DeviceLock lock = LockContext();

    auto shader = static_cast<D3D11GeometryShader*>(pShader);
    SetClassInstances<D3D11ShaderType::eGeometry>(
      shader ? shader->GetCommonShader() : nullptr, ppClassInstances, NumClassInstances);

    if (m_state.gs != shader) {
      m_state.gs = shader;

      BindShader<D3D11ShaderType::eGeometry>(GetCommonShader(shader));
    }
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::GSSetConstantBuffers(
          UINT                              StartSlot,
          UINT                              NumBuffers,
          ID3D11Buffer* const*              ppConstantBuffers) {
    BLESSED_PROBE_CALL(ContextType, SetConstantBuffers);
    D3D10DeviceLock lock = LockContext();

    SetConstantBuffers<D3D11ShaderType::eGeometry>(
      StartSlot, NumBuffers, ppConstantBuffers);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::GSSetConstantBuffers1(
          UINT                              StartSlot,
          UINT                              NumBuffers,
          ID3D11Buffer* const*              ppConstantBuffers,
    const UINT*                             pFirstConstant,
    const UINT*                             pNumConstants) {  
    BLESSED_PROBE_CALL(ContextType, SetConstantBuffers);
    D3D10DeviceLock lock = LockContext();

    SetConstantBuffers1<D3D11ShaderType::eGeometry>(
      StartSlot, NumBuffers, ppConstantBuffers,
      pFirstConstant, pNumConstants);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::GSSetShaderResources(
          UINT                              StartSlot,
          UINT                              NumViews,
          ID3D11ShaderResourceView* const*  ppShaderResourceViews) {
    BLESSED_PROBE_CALL(ContextType, SetShaderResources);
    D3D10DeviceLock lock = LockContext();

    SetShaderResources<D3D11ShaderType::eGeometry>(
      StartSlot, NumViews, ppShaderResourceViews);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::GSSetSamplers(
          UINT                              StartSlot,
          UINT                              NumSamplers,
          ID3D11SamplerState* const*        ppSamplers) {
    BLESSED_PROBE_CALL(ContextType, SetSamplers);
    D3D10DeviceLock lock = LockContext();

    SetSamplers<D3D11ShaderType::eGeometry>(
      StartSlot, NumSamplers, ppSamplers);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::GSGetShader(
          ID3D11GeometryShader**            ppGeometryShader,
          ID3D11ClassInstance**             ppClassInstances,
          UINT*                             pNumClassInstances) {
    D3D10DeviceLock lock = LockContext();

    if (ppGeometryShader)
      *ppGeometryShader = m_state.gs.ref();

    GetClassInstances<D3D11ShaderType::eGeometry>(ppClassInstances, pNumClassInstances);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::GSGetConstantBuffers(
          UINT                              StartSlot,
          UINT                              NumBuffers,
          ID3D11Buffer**                    ppConstantBuffers) {
    D3D10DeviceLock lock = LockContext();

    GetConstantBuffers<D3D11ShaderType::eGeometry>(
      StartSlot, NumBuffers, ppConstantBuffers,
      nullptr, nullptr);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::GSGetConstantBuffers1(
          UINT                              StartSlot,
          UINT                              NumBuffers,
          ID3D11Buffer**                    ppConstantBuffers,
          UINT*                             pFirstConstant,
          UINT*                             pNumConstants) {
    D3D10DeviceLock lock = LockContext();

    GetConstantBuffers<D3D11ShaderType::eGeometry>(
      StartSlot, NumBuffers, ppConstantBuffers,
      pFirstConstant, pNumConstants);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::GSGetShaderResources(
          UINT                              StartSlot,
          UINT                              NumViews,
          ID3D11ShaderResourceView**        ppShaderResourceViews) {
    D3D10DeviceLock lock = LockContext();

    GetShaderResources<D3D11ShaderType::eGeometry>(
      StartSlot, NumViews, ppShaderResourceViews);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::GSGetSamplers(
          UINT                              StartSlot,
          UINT                              NumSamplers,
          ID3D11SamplerState**              ppSamplers) {
    D3D10DeviceLock lock = LockContext();

    GetSamplers<D3D11ShaderType::eGeometry>(
      StartSlot, NumSamplers, ppSamplers);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::PSSetShader(
          ID3D11PixelShader*                pPixelShader,
          ID3D11ClassInstance* const*       ppClassInstances,
          UINT                              NumClassInstances) {
    BLESSED_PROBE_CALL(ContextType, SetShader);
    D3D10DeviceLock lock = LockContext();

    auto shader = static_cast<D3D11PixelShader*>(pPixelShader);
    SetClassInstances<D3D11ShaderType::ePixel>(
      shader ? shader->GetCommonShader() : nullptr, ppClassInstances, NumClassInstances);

    if (m_state.ps != shader) {
      m_state.ps = shader;

      BindShader<D3D11ShaderType::ePixel>(GetCommonShader(shader));
    }
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::PSSetConstantBuffers(
          UINT                              StartSlot,
          UINT                              NumBuffers,
          ID3D11Buffer* const*              ppConstantBuffers) {
    BLESSED_PROBE_CALL(ContextType, SetConstantBuffers);
    D3D10DeviceLock lock = LockContext();

    SetConstantBuffers<D3D11ShaderType::ePixel>(
      StartSlot, NumBuffers, ppConstantBuffers);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::PSSetConstantBuffers1(
          UINT                              StartSlot,
          UINT                              NumBuffers,
          ID3D11Buffer* const*              ppConstantBuffers,
    const UINT*                             pFirstConstant,
    const UINT*                             pNumConstants) {
    BLESSED_PROBE_CALL(ContextType, SetConstantBuffers);
    D3D10DeviceLock lock = LockContext();

    SetConstantBuffers1<D3D11ShaderType::ePixel>(
      StartSlot, NumBuffers, ppConstantBuffers,
      pFirstConstant, pNumConstants);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::PSSetShaderResources(
          UINT                              StartSlot,
          UINT                              NumViews,
          ID3D11ShaderResourceView* const*  ppShaderResourceViews) {
    BLESSED_PROBE_CALL(ContextType, SetShaderResources);
    D3D10DeviceLock lock = LockContext();

    SetShaderResources<D3D11ShaderType::ePixel>(
      StartSlot, NumViews, ppShaderResourceViews);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::PSSetSamplers(
          UINT                              StartSlot,
          UINT                              NumSamplers,
          ID3D11SamplerState* const*        ppSamplers) {
    BLESSED_PROBE_CALL(ContextType, SetSamplers);
    D3D10DeviceLock lock = LockContext();

    SetSamplers<D3D11ShaderType::ePixel>(
      StartSlot, NumSamplers, ppSamplers);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::PSGetShader(
          ID3D11PixelShader**               ppPixelShader,
          ID3D11ClassInstance**             ppClassInstances,
          UINT*                             pNumClassInstances) {
    D3D10DeviceLock lock = LockContext();

    if (ppPixelShader)
      *ppPixelShader = m_state.ps.ref();

    GetClassInstances<D3D11ShaderType::ePixel>(ppClassInstances, pNumClassInstances);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::PSGetConstantBuffers(
          UINT                              StartSlot,
          UINT                              NumBuffers,
          ID3D11Buffer**                    ppConstantBuffers) {
    D3D10DeviceLock lock = LockContext();

    GetConstantBuffers<D3D11ShaderType::ePixel>(
      StartSlot, NumBuffers, ppConstantBuffers,
      nullptr, nullptr);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::PSGetConstantBuffers1(
          UINT                              StartSlot,
          UINT                              NumBuffers,
          ID3D11Buffer**                    ppConstantBuffers,
          UINT*                             pFirstConstant,
          UINT*                             pNumConstants) {
    D3D10DeviceLock lock = LockContext();

    GetConstantBuffers<D3D11ShaderType::ePixel>(
      StartSlot, NumBuffers, ppConstantBuffers,
      pFirstConstant, pNumConstants);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::PSGetShaderResources(
          UINT                              StartSlot,
          UINT                              NumViews,
          ID3D11ShaderResourceView**        ppShaderResourceViews) {
    D3D10DeviceLock lock = LockContext();

    GetShaderResources<D3D11ShaderType::ePixel>(
      StartSlot, NumViews, ppShaderResourceViews);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::PSGetSamplers(
          UINT                              StartSlot,
          UINT                              NumSamplers,
          ID3D11SamplerState**              ppSamplers) {
    D3D10DeviceLock lock = LockContext();

    GetSamplers<D3D11ShaderType::ePixel>(
      StartSlot, NumSamplers, ppSamplers);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::CSSetShader(
          ID3D11ComputeShader*              pComputeShader,
          ID3D11ClassInstance* const*       ppClassInstances,
          UINT                              NumClassInstances) {
    BLESSED_PROBE_CALL(ContextType, SetShader);
    D3D10DeviceLock lock = LockContext();

    auto shader = static_cast<D3D11ComputeShader*>(pComputeShader);
    SetClassInstances<D3D11ShaderType::eCompute>(
      shader ? shader->GetCommonShader() : nullptr, ppClassInstances, NumClassInstances);

    if (m_state.cs != shader) {
      m_state.cs = shader;

      BindShader<D3D11ShaderType::eCompute>(GetCommonShader(shader));
    }
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::CSSetConstantBuffers(
          UINT                              StartSlot,
          UINT                              NumBuffers,
          ID3D11Buffer* const*              ppConstantBuffers) {
    BLESSED_PROBE_CALL(ContextType, SetConstantBuffers);
    D3D10DeviceLock lock = LockContext();

    SetConstantBuffers<D3D11ShaderType::eCompute>(
      StartSlot, NumBuffers, ppConstantBuffers);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::CSSetConstantBuffers1(
          UINT                              StartSlot,
          UINT                              NumBuffers,
          ID3D11Buffer* const*              ppConstantBuffers,
    const UINT*                             pFirstConstant,
    const UINT*                             pNumConstants) {
    BLESSED_PROBE_CALL(ContextType, SetConstantBuffers);
    D3D10DeviceLock lock = LockContext();

    SetConstantBuffers1<D3D11ShaderType::eCompute>(
      StartSlot, NumBuffers, ppConstantBuffers,
      pFirstConstant, pNumConstants);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::CSSetShaderResources(
          UINT                              StartSlot,
          UINT                              NumViews,
          ID3D11ShaderResourceView* const*  ppShaderResourceViews) {
    BLESSED_PROBE_CALL(ContextType, SetShaderResources);
    D3D10DeviceLock lock = LockContext();

    SetShaderResources<D3D11ShaderType::eCompute>(
      StartSlot, NumViews, ppShaderResourceViews);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::CSSetSamplers(
          UINT                              StartSlot,
          UINT                              NumSamplers,
          ID3D11SamplerState* const*        ppSamplers) {
    BLESSED_PROBE_CALL(ContextType, SetSamplers);
    D3D10DeviceLock lock = LockContext();

    SetSamplers<D3D11ShaderType::eCompute>(
      StartSlot, NumSamplers, ppSamplers);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::CSSetUnorderedAccessViews(
          UINT                              StartSlot,
          UINT                              NumUAVs,
          ID3D11UnorderedAccessView* const* ppUnorderedAccessViews,
    const UINT*                             pUAVInitialCounts) {
    D3D10DeviceLock lock = LockContext();

    if (TestRtvUavHazards(0, nullptr, NumUAVs, ppUnorderedAccessViews))
      return;

    // Unbind previously bound conflicting UAVs
    int32_t uavId = m_state.uav.mask.findNext(0);

    while (uavId >= 0) {
      if (uint32_t(uavId) < StartSlot || uint32_t(uavId) >= StartSlot + NumUAVs) {
        for (uint32_t i = 0; i < NumUAVs; i++) {
          auto uav = static_cast<D3D11UnorderedAccessView*>(ppUnorderedAccessViews[i]);

          if (CheckViewOverlap(uav, m_state.uav.views[uavId].ptr())) {
            m_state.uav.views[uavId] = nullptr;
            m_state.uav.mask.clr(uavId);

            if (!DirtyComputeUnorderedAccessView(uavId, true))
              BindUnorderedAccessView(D3D11ShaderType::eCompute, uavId, nullptr);
          }
        }

        uavId = m_state.uav.mask.findNext(uavId + 1);
      } else {
        uavId = m_state.uav.mask.findNext(StartSlot + NumUAVs);
      }
    }

    // Actually bind the given UAVs
    for (uint32_t i = 0; i < NumUAVs; i++) {
      auto uav = static_cast<D3D11UnorderedAccessView*>(ppUnorderedAccessViews[i]);
      auto ctr = pUAVInitialCounts ? pUAVInitialCounts[i] : ~0u;

      if (ctr != ~0u && uav && uav->HasCounter())
        UpdateUnorderedAccessViewCounter(uav, ctr);

      if (m_state.uav.views[StartSlot + i] != uav) {
        m_state.uav.views[StartSlot + i] = uav;
        m_state.uav.mask.set(StartSlot + i, uav != nullptr);

        if (!DirtyComputeUnorderedAccessView(StartSlot + i, !uav))
          BindUnorderedAccessView(D3D11ShaderType::eCompute, StartSlot + i, uav);

        ResolveCsSrvHazards(uav);
      }
    }

    m_state.uav.maxCount = std::clamp(StartSlot + NumUAVs,
      m_state.uav.maxCount, uint32_t(m_state.uav.views.size()));
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::CSGetShader(
          ID3D11ComputeShader**             ppComputeShader,
          ID3D11ClassInstance**             ppClassInstances,
          UINT*                             pNumClassInstances) {
    D3D10DeviceLock lock = LockContext();

    if (ppComputeShader)
      *ppComputeShader = m_state.cs.ref();

    GetClassInstances<D3D11ShaderType::eCompute>(ppClassInstances, pNumClassInstances);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::CSGetConstantBuffers(
          UINT                              StartSlot,
          UINT                              NumBuffers,
          ID3D11Buffer**                    ppConstantBuffers) {
    D3D10DeviceLock lock = LockContext();

    GetConstantBuffers<D3D11ShaderType::eCompute>(
      StartSlot, NumBuffers, ppConstantBuffers,
      nullptr, nullptr);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::CSGetConstantBuffers1(
          UINT                              StartSlot,
          UINT                              NumBuffers,
          ID3D11Buffer**                    ppConstantBuffers,
          UINT*                             pFirstConstant,
          UINT*                             pNumConstants) {
    D3D10DeviceLock lock = LockContext();

    GetConstantBuffers<D3D11ShaderType::eCompute>(
      StartSlot, NumBuffers, ppConstantBuffers,
      pFirstConstant, pNumConstants);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::CSGetShaderResources(
          UINT                              StartSlot,
          UINT                              NumViews,
          ID3D11ShaderResourceView**        ppShaderResourceViews) {
    D3D10DeviceLock lock = LockContext();

    GetShaderResources<D3D11ShaderType::eCompute>(
      StartSlot, NumViews, ppShaderResourceViews);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::CSGetSamplers(
          UINT                              StartSlot,
          UINT                              NumSamplers,
          ID3D11SamplerState**              ppSamplers) {
    D3D10DeviceLock lock = LockContext();

    GetSamplers<D3D11ShaderType::eCompute>(
      StartSlot, NumSamplers, ppSamplers);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::CSGetUnorderedAccessViews(
          UINT                              StartSlot,
          UINT                              NumUAVs,
          ID3D11UnorderedAccessView**       ppUnorderedAccessViews) {
    D3D10DeviceLock lock = LockContext();

    for (uint32_t i = 0; i < NumUAVs; i++) {
      ppUnorderedAccessViews[i] = StartSlot + i < m_state.uav.views.size()
        ? m_state.uav.views[StartSlot + i].ref()
        : nullptr;
    }
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::OMSetRenderTargets(
          UINT                              NumViews,
          ID3D11RenderTargetView* const*    ppRenderTargetViews,
          ID3D11DepthStencilView*           pDepthStencilView) {
    BLESSED_PROBE_CALL(ContextType, OMSetRenderTargets);
    D3D10DeviceLock lock = LockContext();

    SetRenderTargetsAndUnorderedAccessViews(
      NumViews, ppRenderTargetViews, pDepthStencilView,
      NumViews, 0, nullptr, nullptr);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::OMSetRenderTargetsAndUnorderedAccessViews(
          UINT                              NumRTVs,
          ID3D11RenderTargetView* const*    ppRenderTargetViews,
          ID3D11DepthStencilView*           pDepthStencilView,
          UINT                              UAVStartSlot,
          UINT                              NumUAVs,
          ID3D11UnorderedAccessView* const* ppUnorderedAccessViews,
    const UINT*                             pUAVInitialCounts) {
    BLESSED_PROBE_CALL(ContextType, OMSetRenderTargets);
    D3D10DeviceLock lock = LockContext();

    SetRenderTargetsAndUnorderedAccessViews(
      NumRTVs, ppRenderTargetViews, pDepthStencilView,
      UAVStartSlot, NumUAVs, ppUnorderedAccessViews, pUAVInitialCounts);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::OMSetBlendState(
          ID3D11BlendState*                 pBlendState,
    const FLOAT                             BlendFactor[4],
          UINT                              SampleMask) {
    BLESSED_PROBE_CALL(ContextType, OMSetBlendState);
    D3D10DeviceLock lock = LockContext();

    auto blendState = static_cast<D3D11BlendState*>(pBlendState);

    if (m_state.om.cbState    != blendState
     || m_state.om.sampleMask != SampleMask) {
      m_state.om.cbState    = blendState;
      m_state.om.sampleMask = SampleMask;

      ApplyBlendState();
    }

    if (BlendFactor != nullptr) {
      for (uint32_t i = 0; i < 4; i++)
        m_state.om.blendFactor[i] = BlendFactor[i];

      ApplyBlendFactor();
    }
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::OMSetDepthStencilState(
          ID3D11DepthStencilState*          pDepthStencilState,
          UINT                              StencilRef) {
    BLESSED_PROBE_CALL(ContextType, OMSetDepthStencilState);
    D3D10DeviceLock lock = LockContext();

    auto depthStencilState = static_cast<D3D11DepthStencilState*>(pDepthStencilState);

    if (m_state.om.dsState != depthStencilState) {
      m_state.om.dsState = depthStencilState;
      m_state.om.blessedOmGeneration++; // blessed: hook-cpu -- dsState feeds MatchesDepthOnly
      ApplyDepthStencilState();
    }

    // The D3D11 runtime only appears to store the low 8 bits,
    // and some games rely on this behaviour. Do the same here.
    StencilRef &= 0xFF;

    if (m_state.om.stencilRef != StencilRef) {
      m_state.om.stencilRef = StencilRef;
      ApplyStencilRef();
    }
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::OMGetRenderTargets(
          UINT                              NumViews,
          ID3D11RenderTargetView**          ppRenderTargetViews,
          ID3D11DepthStencilView**          ppDepthStencilView) {
    OMGetRenderTargetsAndUnorderedAccessViews(
      NumViews, ppRenderTargetViews, ppDepthStencilView,
      NumViews, 0, nullptr);
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::OMGetRenderTargetsAndUnorderedAccessViews(
          UINT                              NumRTVs,
          ID3D11RenderTargetView**          ppRenderTargetViews,
          ID3D11DepthStencilView**          ppDepthStencilView,
          UINT                              UAVStartSlot,
          UINT                              NumUAVs,
          ID3D11UnorderedAccessView**       ppUnorderedAccessViews) {
    D3D10DeviceLock lock = LockContext();

    if (ppRenderTargetViews) {
      for (UINT i = 0; i < NumRTVs; i++) {
        ppRenderTargetViews[i] = i < m_state.om.rtvs.size()
          ? m_state.om.rtvs[i].ref()
          : nullptr;
      }
    }

    if (ppDepthStencilView)
      *ppDepthStencilView = m_state.om.dsv.ref();

    if (ppUnorderedAccessViews) {
      for (UINT i = 0; i < NumUAVs; i++) {
        ppUnorderedAccessViews[i] = UAVStartSlot + i < m_state.om.uavs.size()
          ? m_state.om.uavs[UAVStartSlot + i].ref()
          : nullptr;
      }
    }
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::OMGetBlendState(
          ID3D11BlendState**                ppBlendState,
          FLOAT                             BlendFactor[4],
          UINT*                             pSampleMask) {
    D3D10DeviceLock lock = LockContext();

    if (ppBlendState)
      *ppBlendState = m_state.om.cbState.ref();

    if (BlendFactor)
      std::memcpy(BlendFactor, m_state.om.blendFactor, sizeof(FLOAT) * 4);

    if (pSampleMask)
      *pSampleMask = m_state.om.sampleMask;
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::OMGetDepthStencilState(
          ID3D11DepthStencilState**         ppDepthStencilState,
          UINT*                             pStencilRef) {
    D3D10DeviceLock lock = LockContext();

    if (ppDepthStencilState)
      *ppDepthStencilState = m_state.om.dsState.ref();

    if (pStencilRef)
      *pStencilRef = m_state.om.stencilRef;
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::RSSetState(ID3D11RasterizerState* pRasterizerState) {
    BLESSED_PROBE_CALL(ContextType, RSSetState);
    D3D10DeviceLock lock = LockContext();

    auto newRasterizerState = static_cast<D3D11RasterizerState*>(pRasterizerState);

    if (m_state.rs.state != newRasterizerState) {
      // Need to keep the previous rasterizer state object alive for the time being
      auto oldRasterizerState = std::move(m_state.rs.state);

      m_state.rs.state = newRasterizerState;
      ApplyRasterizerState();

      // If necessary, update the rasterizer sample count push constant
      uint32_t oldSampleCount = oldRasterizerState ? oldRasterizerState->Desc().ForcedSampleCount : 0;
      uint32_t newSampleCount = newRasterizerState ? newRasterizerState->Desc().ForcedSampleCount : 0;

      if (oldSampleCount != newSampleCount)
        ApplyRasterizerSampleCount();

      // In D3D11, the rasterizer state defines whether the scissor test is
      // enabled, so if that changes, we need to update scissor rects as well.
      bool oldScissorEnable = oldRasterizerState && oldRasterizerState->Desc().ScissorEnable;
      bool newScissorEnable = newRasterizerState && newRasterizerState->Desc().ScissorEnable;

      if (oldScissorEnable != newScissorEnable)
        ApplyViewportState();
    }
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::RSSetViewports(
          UINT                              NumViewports,
    const D3D11_VIEWPORT*                   pViewports) {
    BLESSED_PROBE_CALL(ContextType, RSSetViewports);
    D3D10DeviceLock lock = LockContext();

    if (unlikely(NumViewports > m_state.rs.viewports.size()))
      return;

    for (uint32_t i = 0; i < NumViewports; i++) {
      const D3D11_VIEWPORT& vp = pViewports[i];

      bool valid = vp.Width >= 0.0f && vp.Height >= 0.0f
                && vp.MinDepth >= 0.0f && vp.MaxDepth <= 1.0f
                && vp.MinDepth <= vp.MaxDepth;

      if (!valid)
        return;
    }

    bool dirty = m_state.rs.numViewports != NumViewports;
    m_state.rs.numViewports = NumViewports;

    for (uint32_t i = 0; i < NumViewports; i++) {
      const D3D11_VIEWPORT& vp = m_state.rs.viewports[i];

      dirty |= vp.TopLeftX != pViewports[i].TopLeftX
            || vp.TopLeftY != pViewports[i].TopLeftY
            || vp.Width    != pViewports[i].Width
            || vp.Height   != pViewports[i].Height
            || vp.MinDepth != pViewports[i].MinDepth
            || vp.MaxDepth != pViewports[i].MaxDepth;
      
      m_state.rs.viewports[i] = pViewports[i];
    }

    if (dirty)
      ApplyViewportState();
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::RSSetScissorRects(
          UINT                              NumRects,
    const D3D11_RECT*                       pRects) {
    BLESSED_PROBE_CALL(ContextType, RSSetScissorRects);
    D3D10DeviceLock lock = LockContext();

    if (unlikely(NumRects > m_state.rs.scissors.size()))
      return;

    bool dirty = m_state.rs.numScissors != NumRects;
    m_state.rs.numScissors = NumRects;

    for (uint32_t i = 0; i < NumRects; i++) {
      if (pRects[i].bottom >= pRects[i].top
       && pRects[i].right  >= pRects[i].left) {
        const D3D11_RECT& sr = m_state.rs.scissors[i];

        dirty |= sr.top    != pRects[i].top
              || sr.left   != pRects[i].left
              || sr.bottom != pRects[i].bottom
              || sr.right  != pRects[i].right;

        m_state.rs.scissors[i] = pRects[i];
      }
    }

    if (dirty && m_state.rs.state && m_state.rs.state->Desc().ScissorEnable)
      ApplyViewportState();
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::RSGetState(ID3D11RasterizerState** ppRasterizerState) {
    D3D10DeviceLock lock = LockContext();

    if (ppRasterizerState)
      *ppRasterizerState = m_state.rs.state.ref();
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::RSGetViewports(
          UINT*                             pNumViewports,
          D3D11_VIEWPORT*                   pViewports) {
    D3D10DeviceLock lock = LockContext();
    uint32_t numWritten = m_state.rs.numViewports;

    if (pViewports) {
      numWritten = std::min(numWritten, *pNumViewports);

      for (uint32_t i = 0; i < *pNumViewports; i++) {
        if (i < m_state.rs.numViewports) {
          pViewports[i] = m_state.rs.viewports[i];
        } else {
          pViewports[i].TopLeftX = 0.0f;
          pViewports[i].TopLeftY = 0.0f;
          pViewports[i].Width    = 0.0f;
          pViewports[i].Height   = 0.0f;
          pViewports[i].MinDepth = 0.0f;
          pViewports[i].MaxDepth = 0.0f;
        }
      }
    }

    *pNumViewports = numWritten;
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::RSGetScissorRects(
          UINT*                             pNumRects,
          D3D11_RECT*                       pRects) {
    D3D10DeviceLock lock = LockContext();
    uint32_t numWritten = m_state.rs.numScissors;

    if (pRects) {
      numWritten = std::min(numWritten, *pNumRects);

      for (uint32_t i = 0; i < *pNumRects; i++) {
        if (i < m_state.rs.numScissors) {
          pRects[i] = m_state.rs.scissors[i];
        } else {
          pRects[i].left   = 0;
          pRects[i].top    = 0;
          pRects[i].right  = 0;
          pRects[i].bottom = 0;
        }
      }
    }

    *pNumRects = m_state.rs.numScissors;
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::SOSetTargets(
          UINT                              NumBuffers,
          ID3D11Buffer* const*              ppSOTargets,
    const UINT*                             pOffsets) {
    D3D10DeviceLock lock = LockContext();

    for (uint32_t i = 0; i < NumBuffers; i++) {
      D3D11Buffer* buffer = static_cast<D3D11Buffer*>(ppSOTargets[i]);
      UINT         offset = pOffsets != nullptr ? pOffsets[i] : 0;

      m_state.so.targets[i].buffer = buffer;
      m_state.so.targets[i].offset = offset;
    }

    for (uint32_t i = NumBuffers; i < D3D11_SO_BUFFER_SLOT_COUNT; i++) {
      m_state.so.targets[i].buffer = nullptr;
      m_state.so.targets[i].offset = 0;
    }

    for (uint32_t i = 0; i < D3D11_SO_BUFFER_SLOT_COUNT; i++) {
      BindXfbBuffer(i,
        m_state.so.targets[i].buffer.ptr(),
        m_state.so.targets[i].offset);
    }
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::SOGetTargets(
          UINT                              NumBuffers,
          ID3D11Buffer**                    ppSOTargets) {
    D3D10DeviceLock lock = LockContext();

    for (uint32_t i = 0; i < NumBuffers; i++) {
      ppSOTargets[i] = i < m_state.so.targets.size()
        ? m_state.so.targets[i].buffer.ref()
        : nullptr;
    }
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::SOGetTargetsWithOffsets(
          UINT                              NumBuffers,
          ID3D11Buffer**                    ppSOTargets,
          UINT*                             pOffsets) {
    D3D10DeviceLock lock = LockContext();

    for (uint32_t i = 0; i < NumBuffers; i++) {
      const bool inRange = i < m_state.so.targets.size();

      if (ppSOTargets) {
        ppSOTargets[i] = inRange
          ? m_state.so.targets[i].buffer.ref()
          : nullptr;
      }

      if (pOffsets) {
        pOffsets[i] = inRange
          ? m_state.so.targets[i].offset
          : 0u;
      }
    }
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::SetPredication(
          ID3D11Predicate*                  pPredicate,
          BOOL                              PredicateValue) {
    D3D10DeviceLock lock = LockContext();

    auto predicate = D3D11Query::FromPredicate(pPredicate);
    m_state.pr.predicateObject = predicate;
    m_state.pr.predicateValue  = PredicateValue;

    static bool s_errorShown = false;

    if (pPredicate && !std::exchange(s_errorShown, true))
      Logger::err("D3D11DeviceContext::SetPredication: Stub");
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::GetPredication(
          ID3D11Predicate**                 ppPredicate,
          BOOL*                             pPredicateValue) {
    D3D10DeviceLock lock = LockContext();

    if (ppPredicate)
      *ppPredicate = D3D11Query::AsPredicate(m_state.pr.predicateObject.ref());

    if (pPredicateValue)
      *pPredicateValue = m_state.pr.predicateValue;
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::SetResourceMinLOD(
          ID3D11Resource*                   pResource,
          FLOAT                             MinLOD) {
    bool s_errorShown = false;

    if (std::exchange(s_errorShown, true))
      Logger::err("D3D11DeviceContext::SetResourceMinLOD: Not implemented");
  }
  
  
  template<typename ContextType>
  FLOAT STDMETHODCALLTYPE D3D11CommonContext<ContextType>::GetResourceMinLOD(ID3D11Resource* pResource) {
    bool s_errorShown = false;

    if (std::exchange(s_errorShown, true))
      Logger::err("D3D11DeviceContext::GetResourceMinLOD: Not implemented");

    return 0.0f;
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::CopyTiles(
          ID3D11Resource*                   pTiledResource,
    const D3D11_TILED_RESOURCE_COORDINATE*  pTileRegionStartCoordinate,
    const D3D11_TILE_REGION_SIZE*           pTileRegionSize,
          ID3D11Buffer*                     pBuffer,
          UINT64                            BufferStartOffsetInBytes,
          UINT                              Flags) {
    D3D10DeviceLock lock = LockContext();

    if (!pTiledResource || !pBuffer)
      return;

    auto buffer = static_cast<D3D11Buffer*>(pBuffer);

    // Get buffer slice and just forward the call
    VkDeviceSize bufferSize = pTileRegionSize->NumTiles * SparseMemoryPageSize;

    if (buffer->Desc()->ByteWidth < BufferStartOffsetInBytes + bufferSize)
      return;

    DxvkBufferSlice slice = buffer->GetBufferSlice(BufferStartOffsetInBytes, bufferSize);

    CopyTiledResourceData(pTiledResource,
      pTileRegionStartCoordinate,
      pTileRegionSize, slice, Flags);

    if (buffer->HasSequenceNumber())
      GetTypedContext()->TrackBufferSequenceNumber(buffer);
  }


  template<typename ContextType>
  HRESULT STDMETHODCALLTYPE D3D11CommonContext<ContextType>::CopyTileMappings(
          ID3D11Resource*                   pDestTiledResource,
    const D3D11_TILED_RESOURCE_COORDINATE*  pDestRegionCoordinate,
          ID3D11Resource*                   pSourceTiledResource,
    const D3D11_TILED_RESOURCE_COORDINATE*  pSourceRegionCoordinate,
    const D3D11_TILE_REGION_SIZE*           pTileRegionSize,
          UINT                              Flags) {
    D3D10DeviceLock lock = LockContext();

    if (!pDestTiledResource || !pSourceTiledResource)
      return E_INVALIDARG;

    if constexpr (!IsDeferred)
      GetTypedContext()->ConsiderFlush(GpuFlushType::ImplicitWeakHint);

    DxvkSparseBindInfo bindInfo;
    bindInfo.dstResource = GetPagedResource(pDestTiledResource);
    bindInfo.srcResource = GetPagedResource(pSourceTiledResource);

    auto dstPageTable = bindInfo.dstResource->getSparsePageTable();
    auto srcPageTable = bindInfo.srcResource->getSparsePageTable();

    if (!dstPageTable || !srcPageTable)
      return E_INVALIDARG;

    if (pDestRegionCoordinate->Subresource >= dstPageTable->getSubresourceCount()
     || pSourceRegionCoordinate->Subresource >= srcPageTable->getSubresourceCount())
      return E_INVALIDARG;

    VkOffset3D dstRegionOffset = {
      int32_t(pDestRegionCoordinate->X),
      int32_t(pDestRegionCoordinate->Y),
      int32_t(pDestRegionCoordinate->Z) };

    VkOffset3D srcRegionOffset = {
      int32_t(pSourceRegionCoordinate->X),
      int32_t(pSourceRegionCoordinate->Y),
      int32_t(pSourceRegionCoordinate->Z) };

    VkExtent3D regionExtent = {
      uint32_t(pTileRegionSize->Width),
      uint32_t(pTileRegionSize->Height),
      uint32_t(pTileRegionSize->Depth) };

    for (uint32_t i = 0; i < pTileRegionSize->NumTiles; i++) {
      // We don't know the current tile mappings of either resource since
      // this may be called on a deferred context and tile mappings are
      // updated on the CS thread, so just resolve the copy in the backend
      uint32_t dstTile = dstPageTable->computePageIndex(
        pDestRegionCoordinate->Subresource, dstRegionOffset,
        regionExtent, !pTileRegionSize->bUseBox, i);

      uint32_t srcTile = srcPageTable->computePageIndex(
        pSourceRegionCoordinate->Subresource, srcRegionOffset,
        regionExtent, !pTileRegionSize->bUseBox, i);

      if (dstTile >= dstPageTable->getPageCount()
       || srcTile >= srcPageTable->getPageCount())
        return E_INVALIDARG;

      DxvkSparseBind bind;
      bind.mode = DxvkSparseBindMode::Copy;
      bind.dstPage = dstTile;
      bind.srcPage = srcTile;

      bindInfo.binds.push_back(bind);
    }

    DxvkSparseBindFlags flags = (Flags & D3D11_TILE_MAPPING_NO_OVERWRITE)
      ? DxvkSparseBindFlags(DxvkSparseBindFlag::SkipSynchronization)
      : DxvkSparseBindFlags();

    EmitCs([
      cBindInfo = std::move(bindInfo),
      cFlags    = flags
    ] (DxvkContext* ctx) {
      ctx->updatePageTable(cBindInfo, cFlags);
    });

    return S_OK;
  }


  template<typename ContextType>
  HRESULT STDMETHODCALLTYPE D3D11CommonContext<ContextType>::ResizeTilePool(
          ID3D11Buffer*                     pTilePool,
          UINT64                            NewSizeInBytes) {
    D3D10DeviceLock lock = LockContext();

    if (NewSizeInBytes % SparseMemoryPageSize)
      return E_INVALIDARG;

    auto buffer = static_cast<D3D11Buffer*>(pTilePool);

    if (!buffer->IsTilePool())
      return E_INVALIDARG;

    // Perform the resize operation. This is somewhat trivialized
    // since all lifetime tracking is done by the backend.
    EmitCs([
      cAllocator  = buffer->GetSparseAllocator(),
      cPageCount  = NewSizeInBytes / SparseMemoryPageSize
    ] (DxvkContext* ctx) {
      cAllocator->setCapacity(cPageCount);
    });

    return S_OK;
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::TiledResourceBarrier(
          ID3D11DeviceChild*                pTiledResourceOrViewAccessBeforeBarrier,
          ID3D11DeviceChild*                pTiledResourceOrViewAccessAfterBarrier) {
    D3D10DeviceLock lock = LockContext();

    DxvkGlobalPipelineBarrier srcBarrier = GetTiledResourceDependency(pTiledResourceOrViewAccessBeforeBarrier);
    DxvkGlobalPipelineBarrier dstBarrier = GetTiledResourceDependency(pTiledResourceOrViewAccessAfterBarrier);

    if (srcBarrier.stages && dstBarrier.stages) {
      EmitCs([
        cSrcBarrier = srcBarrier,
        cDstBarrier = dstBarrier
      ] (DxvkContext* ctx) {
        ctx->emitGraphicsBarrier(
          cSrcBarrier.stages, cSrcBarrier.access,
          cDstBarrier.stages, cDstBarrier.access);
      });
    }
  }


  template<typename ContextType>
  HRESULT STDMETHODCALLTYPE D3D11CommonContext<ContextType>::UpdateTileMappings(
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
    D3D10DeviceLock lock = LockContext();

    if (!pTiledResource || !NumRegions || !NumRanges)
      return E_INVALIDARG;

    if constexpr (!IsDeferred)
      GetTypedContext()->ConsiderFlush(GpuFlushType::ImplicitWeakHint);

    // Find sparse allocator if the tile pool is defined
    DxvkSparseBindInfo bindInfo;

    if (pTilePool) {
      auto tilePool = static_cast<D3D11Buffer*>(pTilePool);
      bindInfo.srcAllocator = tilePool->GetSparseAllocator();

      if (bindInfo.srcAllocator == nullptr)
        return E_INVALIDARG;
    }

    // Find resource and sparse page table for the given resource
    bindInfo.dstResource = GetPagedResource(pTiledResource);
    auto pageTable = bindInfo.dstResource->getSparsePageTable();

    if (!pageTable)
      return E_INVALIDARG;

    // Lookup table in case the app tries to bind the same
    // page multiple times. We should resolve that here and
    // only consider the last bind to any given page.
    std::vector<uint32_t> bindIndices(pageTable->getPageCount(), ~0u);

    // This function allows pretty much every parameter to be nullptr
    // in some way, so initialize some defaults as necessary
    D3D11_TILED_RESOURCE_COORDINATE regionCoord = { };
    D3D11_TILE_REGION_SIZE regionSize = { };

    if (!pRegionSizes) {
      regionSize.NumTiles = pRegionCoordinates
        ? 1 : pageTable->getPageCount();
    }

    uint32_t rangeFlag = 0u;
    uint32_t rangeTileOffset = 0u;
    uint32_t rangeTileCount = ~0u;

    // For now, just generate a simple list of tile index to
    // page index mappings, and let the backend optimize later
    uint32_t regionIdx = 0u;
    uint32_t regionTile = 0u;
    uint32_t rangeIdx = 0u;
    uint32_t rangeTile = 0u;

    while (regionIdx < NumRegions && rangeIdx < NumRanges) {
      if (!regionTile) {
        if (pRegionCoordinates)
          regionCoord = pRegionCoordinates[regionIdx];

        if (pRegionSizes)
          regionSize = pRegionSizes[regionIdx];
      }

      if (!rangeTile) {
        if (pRangeFlags)
          rangeFlag = pRangeFlags[rangeIdx];

        if (pRangeTileOffsets)
          rangeTileOffset = pRangeTileOffsets[rangeIdx];

        if (pRangeTileCounts)
          rangeTileCount = pRangeTileCounts[rangeIdx];
      }

      if (!(rangeFlag & D3D11_TILE_RANGE_SKIP)) {
        if (regionCoord.Subresource >= pageTable->getSubresourceCount())
          return E_INVALIDARG;

        if (regionSize.bUseBox && regionSize.NumTiles !=
            regionSize.Width * regionSize.Height * regionSize.Depth)
          return E_INVALIDARG;

        VkOffset3D regionOffset = {
          int32_t(regionCoord.X),
          int32_t(regionCoord.Y),
          int32_t(regionCoord.Z) };

        VkExtent3D regionExtent = {
          uint32_t(regionSize.Width),
          uint32_t(regionSize.Height),
          uint32_t(regionSize.Depth) };

        uint32_t resourceTile = pageTable->computePageIndex(regionCoord.Subresource,
          regionOffset, regionExtent, !regionSize.bUseBox, regionTile);

        // Fill in bind info for the current tile
        DxvkSparseBind bind = { };
        bind.dstPage = resourceTile;

        if (rangeFlag & D3D11_TILE_RANGE_NULL) {
          bind.mode = DxvkSparseBindMode::Null;
        } else if (pTilePool) {
          bind.mode = DxvkSparseBindMode::Bind;
          bind.srcPage = rangeFlag & D3D11_TILE_RANGE_REUSE_SINGLE_TILE
            ? rangeTileOffset
            : rangeTileOffset + rangeTile;
        } else {
          return E_INVALIDARG;
        }

        // Add bind info to the bind list, overriding
        // any existing bind for the same resource page
        if (resourceTile < pageTable->getPageCount()) {
          if (bindIndices[resourceTile] < bindInfo.binds.size())
            bindInfo.binds[bindIndices[resourceTile]] = bind;
          else
            bindInfo.binds.push_back(bind);
        }
      }

      if (++regionTile == regionSize.NumTiles) {
        regionTile = 0;
        regionIdx += 1;
      }

      if (++rangeTile == rangeTileCount) {
        rangeTile = 0;
        rangeIdx += 1;
      }
    }

    // Translate flags. The backend benefits from NO_OVERWRITE since
    // otherwise we have to serialize execution of the current command
    // buffer, the sparse binding operation, and subsequent commands.
    // With NO_OVERWRITE, we can execute it more or less asynchronously.
    DxvkSparseBindFlags flags = (Flags & D3D11_TILE_MAPPING_NO_OVERWRITE)
      ? DxvkSparseBindFlags(DxvkSparseBindFlag::SkipSynchronization)
      : DxvkSparseBindFlags();

    EmitCs([
      cBindInfo = std::move(bindInfo),
      cFlags    = flags
    ] (DxvkContext* ctx) {
      ctx->updatePageTable(cBindInfo, cFlags);
    });

    return S_OK;
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::UpdateTiles(
          ID3D11Resource*                   pDestTiledResource,
    const D3D11_TILED_RESOURCE_COORDINATE*  pDestTileRegionStartCoordinate,
    const D3D11_TILE_REGION_SIZE*           pDestTileRegionSize,
    const void*                             pSourceTileData,
          UINT                              Flags) {
    D3D10DeviceLock lock = LockContext();

    if (!pDestTiledResource || !pSourceTileData)
      return;

    // Allocate staging memory and copy source data into it, at a
    // 64k page granularity. It is not clear whether this behaviour
    // is correct in case we're writing to incmplete pages.
    VkDeviceSize bufferSize = pDestTileRegionSize->NumTiles * SparseMemoryPageSize;

    DxvkBufferSlice slice = AllocStagingBuffer(bufferSize);
    std::memcpy(slice.mapPtr(0), pSourceTileData, bufferSize);

    // Fix up flags. The runtime probably validates this in some
    // way but our internal function relies on correct flags anyway.
    Flags &= D3D11_TILE_MAPPING_NO_OVERWRITE;
    Flags |= D3D11_TILE_COPY_LINEAR_BUFFER_TO_SWIZZLED_TILED_RESOURCE;

    CopyTiledResourceData(pDestTiledResource,
      pDestTileRegionStartCoordinate,
      pDestTileRegionSize, slice, Flags);

    if constexpr (!IsDeferred)
      static_cast<ContextType*>(this)->ThrottleAllocation();
  }


  template<typename ContextType>
  BOOL STDMETHODCALLTYPE D3D11CommonContext<ContextType>::IsAnnotationEnabled() {
    return m_annotation.GetStatus();
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::SetMarkerInt(
          LPCWSTR                           pLabel,
          INT                               Data) {
    // Not implemented in the backend, ignore
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::BeginEventInt(
          LPCWSTR                           pLabel,
          INT                               Data) {
    // Not implemented in the backend, ignore
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::EndEvent() {
    // Not implemented in the backend, ignore
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::GetHardwareProtectionState(
          BOOL*                             pHwProtectionEnable) {
    static bool s_errorShown = false;

    if (!std::exchange(s_errorShown, true))
      Logger::err("D3D11DeviceContext::GetHardwareProtectionState: Not implemented");

    if (pHwProtectionEnable)
      *pHwProtectionEnable = FALSE;
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::SetHardwareProtectionState(
          BOOL                              HwProtectionEnable) {
    static bool s_errorShown = false;

    if (!std::exchange(s_errorShown, true))
      Logger::err("D3D11DeviceContext::SetHardwareProtectionState: Not implemented");
  }


  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11CommonContext<ContextType>::TransitionSurfaceLayout(
          IDXGIVkInteropSurface*    pSurface,
    const VkImageSubresourceRange*  pSubresources,
          VkImageLayout             OldLayout,
          VkImageLayout             NewLayout) {
    D3D10DeviceLock lock = LockContext();

    // Get the underlying D3D11 resource
    Com<ID3D11Resource> resource;

    pSurface->QueryInterface(__uuidof(ID3D11Resource),
      reinterpret_cast<void**>(&resource));

    // Get the texture from that resource
    D3D11CommonTexture* texture = GetCommonTexture(resource.ptr());

    EmitCs([
      cImage        = texture->GetImage(),
      cSubresources = *pSubresources,
      cOldLayout    = OldLayout,
      cNewLayout    = NewLayout
    ] (DxvkContext* ctx) {
      ctx->transformImage(
        cImage, cSubresources,
        cOldLayout, cNewLayout);
    });
  }


  template<typename ContextType>
  DxvkCsChunkRef D3D11CommonContext<ContextType>::AllocCsChunk() {
    return m_parent->AllocCsChunk(m_csFlags);
  }


  template<typename ContextType>
  DxvkBufferSlice D3D11CommonContext<ContextType>::AllocStagingBuffer(
          VkDeviceSize                      Size) {
    return m_staging.alloc(Size);
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::ApplyDirtyConstantBuffers(
          D3D11ShaderType                   Stage,
    const D3D11BindingMask&                 BoundMask,
          D3D11BindingMask&                 DirtyMask) {
    uint32_t bindMask = BoundMask.cbvMask & DirtyMask.cbvMask;

    if (!bindMask)
      return;

    // Need to clear dirty bits before binding
    const auto& state = m_state.cbv[Stage];
    DirtyMask.cbvMask -= bindMask;

    for (uint32_t slot : bit::BitMask(bindMask)) {
      const auto& cbv = state.buffers[slot];

      BindConstantBuffer(Stage, slot, cbv.buffer.ptr(),
        cbv.constantOffset, cbv.constantBound);
    }
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::ApplyDirtySamplers(
          D3D11ShaderType                   Stage,
    const D3D11BindingMask&                 BoundMask,
          D3D11BindingMask&                 DirtyMask) {
    uint32_t bindMask = BoundMask.samplerMask & DirtyMask.samplerMask;

    if (!bindMask)
      return;

    // Need to clear dirty bits before binding
    const auto& state = m_state.samplers[Stage];
    DirtyMask.samplerMask -= bindMask;

    for (uint32_t slot : bit::BitMask(bindMask))
      BindSampler(Stage, slot, state.samplers[slot].ptr());
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::ApplyDirtyShaderResources(
          D3D11ShaderType                   Stage,
    const D3D11BindingMask&                 BoundMask,
          D3D11BindingMask&                 DirtyMask) {
    const auto& state = m_state.srv[Stage];

    for (uint32_t i = 0; i < state.maxCount; i += 64u) {
      uint32_t maskIndex = i / 64u;
      uint64_t bindMask = BoundMask.srvMask[maskIndex] & DirtyMask.srvMask[maskIndex];

      if (!bindMask)
        continue;

    // Need to clear dirty bits before binding
      DirtyMask.srvMask[maskIndex] -= bindMask;

      for (uint32_t slot : bit::BitMask(bindMask))
        BindShaderResource(Stage, slot + i, state.views[slot + i].ptr());
    }
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::ApplyDirtyUnorderedAccessViews(
          D3D11ShaderType                   Stage,
    const D3D11BindingMask&                 BoundMask,
          D3D11BindingMask&                 DirtyMask) {
    uint64_t bindMask = BoundMask.uavMask & DirtyMask.uavMask;

    if (!bindMask)
      return;

    const auto& views = Stage == D3D11ShaderType::eCompute
      ? m_state.uav.views
      : m_state.om.uavs;

    // Need to clear dirty bits before binding
    DirtyMask.uavMask -= bindMask;

    for (uint32_t slot : bit::BitMask(bindMask))
      BindUnorderedAccessView(Stage, slot, views[slot].ptr());
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::ApplyDirtyGraphicsBindings() {
    auto dirtyMask = m_state.lazy.shadersDirty & m_state.lazy.shadersUsed;
    dirtyMask.clr(D3D11ShaderType::eCompute);

    if (unlikely(!(dirtyMask & m_state.lazy.graphicsUavShaders).isClear())) {
      D3D11ShaderType stage = D3D11ShaderType::ePixel;

      auto& boundMask = m_state.lazy.bindingsUsed[stage];
      auto& dirtyMask = m_state.lazy.bindingsDirty[stage];

      ApplyDirtyUnorderedAccessViews(stage, boundMask, dirtyMask);
    }

    for (uint32_t stageIndex : bit::BitMask(uint32_t(dirtyMask.raw()))) {
      D3D11ShaderType stage = D3D11ShaderType(stageIndex);

      auto& boundMask = m_state.lazy.bindingsUsed[stage];
      auto& dirtyMask = m_state.lazy.bindingsDirty[stage];

      ApplyDirtySamplers(stage, boundMask, dirtyMask);
      ApplyDirtyConstantBuffers(stage, boundMask, dirtyMask);
      ApplyDirtyShaderResources(stage, boundMask, dirtyMask);

      m_state.lazy.shadersDirty.clr(stage);
    }
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::ApplyDirtyComputeBindings() {
    D3D11ShaderType stage = D3D11ShaderType::eCompute;

    auto& boundMask = m_state.lazy.bindingsUsed[stage];
    auto& dirtyMask = m_state.lazy.bindingsDirty[stage];

    ApplyDirtySamplers(stage, boundMask, dirtyMask);
    ApplyDirtyConstantBuffers(stage, boundMask, dirtyMask);
    ApplyDirtyShaderResources(stage, boundMask, dirtyMask);
    ApplyDirtyUnorderedAccessViews(stage, boundMask, dirtyMask);

    m_state.lazy.shadersDirty.clr(stage);
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::ApplyInputLayout() {
    if (likely(m_state.ia.inputLayout != nullptr)) {
      uint32_t attributeCount = m_state.ia.inputLayout->GetAttributeCount();
      uint32_t bindingCount = m_state.ia.inputLayout->GetBindingCount();

      EmitCsCmd<DxvkVertexInput>(D3D11CmdType::None, attributeCount + bindingCount, [
        cAttributeCount   = attributeCount,
        cBindingCount     = bindingCount
      ] (DxvkContext* ctx, const DxvkVertexInput* layout, size_t) {
        ctx->setInputLayout(cAttributeCount, &layout[0],
          cBindingCount, &layout[cAttributeCount]);
      });

      for (uint32_t i = 0; i < attributeCount + bindingCount; i++)
        new (m_csData->at(i)) DxvkVertexInput(m_state.ia.inputLayout->GetInput(i));
    } else {
      EmitCs([] (DxvkContext* ctx) {
        ctx->setInputLayout(0, nullptr, 0, nullptr);
      });
    }
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::ApplyPrimitiveTopology() {
    D3D11_PRIMITIVE_TOPOLOGY topology = m_state.ia.primitiveTopology;
    DxvkInputAssemblyState iaState(VK_PRIMITIVE_TOPOLOGY_MAX_ENUM, false);

    if (topology <= D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP_ADJ) {
      static const std::array<DxvkInputAssemblyState, 14> s_iaStates = {{
        DxvkInputAssemblyState(VK_PRIMITIVE_TOPOLOGY_MAX_ENUM, false),
        // Regular topologies
        DxvkInputAssemblyState(VK_PRIMITIVE_TOPOLOGY_POINT_LIST,     false),
        DxvkInputAssemblyState(VK_PRIMITIVE_TOPOLOGY_LINE_LIST,      false),
        DxvkInputAssemblyState(VK_PRIMITIVE_TOPOLOGY_LINE_STRIP,     true),
        DxvkInputAssemblyState(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,  false),
        DxvkInputAssemblyState(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, true),
        // Gap. This includes triangle fan which isn't supported in D3D11
        DxvkInputAssemblyState(VK_PRIMITIVE_TOPOLOGY_MAX_ENUM, false),
        DxvkInputAssemblyState(VK_PRIMITIVE_TOPOLOGY_MAX_ENUM, false),
        DxvkInputAssemblyState(VK_PRIMITIVE_TOPOLOGY_MAX_ENUM, false),
        DxvkInputAssemblyState(VK_PRIMITIVE_TOPOLOGY_MAX_ENUM, false),
        // Adjacency topologies
        DxvkInputAssemblyState(VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY,       false),
        DxvkInputAssemblyState(VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY,      true),
        DxvkInputAssemblyState(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY,   false),
        DxvkInputAssemblyState(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY,  true),
      }};

      iaState = s_iaStates[uint32_t(topology)];
    } else if (topology >= D3D11_PRIMITIVE_TOPOLOGY_1_CONTROL_POINT_PATCHLIST
            && topology <= D3D11_PRIMITIVE_TOPOLOGY_32_CONTROL_POINT_PATCHLIST) {
      // The number of control points per patch can be inferred from the enum value in D3D11
      iaState = DxvkInputAssemblyState(VK_PRIMITIVE_TOPOLOGY_PATCH_LIST, false);
      iaState.setPatchVertexCount(topology - D3D11_PRIMITIVE_TOPOLOGY_1_CONTROL_POINT_PATCHLIST + 1);
    }

    EmitCs([iaState] (DxvkContext* ctx) {
      ctx->setInputAssemblyState(iaState);
    });
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::ApplyBlendState() {
    if (m_state.om.cbState != nullptr) {
      EmitCs([
        cBlendState = m_state.om.cbState->GetBlendState(),
        cMsState    = m_state.om.cbState->GetMsState(m_state.om.sampleMask),
        cLoState    = m_state.om.cbState->GetLoState()
      ] (DxvkContext* ctx) {
        for (uint32_t i = 0; i < cBlendState.size(); i++)
          ctx->setBlendMode(i, cBlendState[i]);

        ctx->setMultisampleState(cMsState);
        ctx->setLogicOpState(cLoState);
      });
    } else {
      EmitCs([
        cSampleMask = m_state.om.sampleMask
      ] (DxvkContext* ctx) {
        DxvkBlendMode cbState = InitDefaultBlendState();

        for (uint32_t i = 0; i < D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT; i++)
          ctx->setBlendMode(i, cbState);

        ctx->setMultisampleState(InitDefaultMultisampleState(cSampleMask));
        ctx->setLogicOpState(InitDefaultLogicOpState());
      });
    }
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::ApplyBlendFactor() {
    EmitCs([
      cBlendConstants = DxvkBlendConstants {
        m_state.om.blendFactor[0], m_state.om.blendFactor[1],
        m_state.om.blendFactor[2], m_state.om.blendFactor[3] }
    ] (DxvkContext* ctx) {
      ctx->setBlendConstants(cBlendConstants);
    });
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::ApplyDepthStencilState() {
    if (m_state.om.dsState != nullptr) {
      EmitCs([
        cState = m_state.om.dsState->GetState()
      ] (DxvkContext* ctx) {
        ctx->setDepthStencilState(cState);
      });
    } else {
      EmitCs([] (DxvkContext* ctx) {
        ctx->setDepthStencilState(InitDefaultDepthStencilState());
      });
    }
  }
  
  
  template<typename ContextType>
  void D3D11CommonContext<ContextType>::ApplyStencilRef() {
    EmitCs([
      cStencilRef = m_state.om.stencilRef
    ] (DxvkContext* ctx) {
      ctx->setStencilReference(cStencilRef);
    });
  }
  
  
  template<typename ContextType>
  void D3D11CommonContext<ContextType>::ApplyRasterizerState() {
    if (m_state.rs.state != nullptr) {
      EmitCs([
        cState      = m_state.rs.state->GetState(),
        cDepthBias  = m_state.rs.state->GetDepthBias()
      ] (DxvkContext* ctx) {
        ctx->setRasterizerState(cState);
        ctx->setDepthBias(cDepthBias);
      });
    } else {
      EmitCs([] (DxvkContext* ctx) {
        ctx->setRasterizerState(InitDefaultRasterizerState());
        ctx->setDepthBias(DxvkDepthBias());
      });
    }
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::ApplyRasterizerSampleCount() {
    D3D11SpecData specData = {};
    specData.sampleCount = m_state.om.sampleCount;

    if (unlikely(!specData.sampleCount)) {
      specData.sampleCount = m_state.rs.state
        ? m_state.rs.state->Desc().ForcedSampleCount
        : 0u;

      if (!specData.sampleCount)
        specData.sampleCount = 1u;
    }

    EmitCs([
      cSpecData = specData
    ] (DxvkContext* ctx) {
      ctx->setSpecConstants(VK_PIPELINE_BIND_POINT_GRAPHICS,
        0u, sizeof(cSpecData), &cSpecData);
    });
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::ApplyViewportState() {
    uint32_t viewportCount = m_state.rs.numViewports;

    if (likely(viewportCount)) {
      EmitCsCmd<DxvkViewport>(D3D11CmdType::None, viewportCount,
        [] (DxvkContext* ctx, const DxvkViewport* viewports, size_t count) {
          ctx->setViewports(count, viewports);
        });

      // Vulkan does not provide an easy way to disable the scissor test,
      // Set scissor rects that are at least as large as the framebuffer.
      bool enableScissorTest = m_state.rs.state && m_state.rs.state->Desc().ScissorEnable;

      // D3D11's coordinate system has its origin in the bottom left,
      // but the viewport coordinates are aligned to the top-left
      // corner so we can get away with flipping the viewport.
      for (uint32_t i = 0; i < viewportCount; i++) {
        const auto& vp = m_state.rs.viewports[i];

        auto* dst = new (m_csData->at(i)) DxvkViewport();
        dst->viewport.x = vp.TopLeftX;
        dst->viewport.y = vp.Height + vp.TopLeftY;
        dst->viewport.width = vp.Width;
        dst->viewport.height = -vp.Height;
        dst->viewport.minDepth = vp.MinDepth;
        dst->viewport.maxDepth = vp.MaxDepth;

        if (!enableScissorTest) {
          dst->scissor.offset = VkOffset2D { 0, 0 };
          dst->scissor.extent = VkExtent2D {
            D3D11_VIEWPORT_BOUNDS_MAX,
            D3D11_VIEWPORT_BOUNDS_MAX };
        } else if (i < m_state.rs.numScissors) {
          const auto& sr = m_state.rs.scissors[i];

          dst->scissor.offset = VkOffset2D { sr.left, sr.top };
          dst->scissor.extent = VkExtent2D {
            uint32_t(std::max(sr.left, sr.right) - sr.left),
            uint32_t(std::max(sr.top, sr.bottom) - sr.top) };
        }
      }
    } else {
      // The backend can't handle a viewport count of zero,
      // so we should at least specify one empty viewport
      EmitCs([] (DxvkContext* ctx) {
        DxvkViewport viewport = { };
        ctx->setViewports(1, &viewport);
      });
    }
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::BatchDraw(
    const VkDrawIndirectCommand&            draw) {
    if (unlikely(HasDirtyGraphicsBindings()))
      ApplyDirtyGraphicsBindings();

    // Batch consecutive draws if there are no state changes
    if (m_csDataType == D3D11CmdType::Draw) {
      auto* drawInfo = m_csChunk->pushData(m_csData, 1u);

      if (likely(drawInfo)) {
        new (drawInfo) VkDrawIndirectCommand(draw);
        return;
      }
    }

    EmitCsCmd<VkDrawIndirectCommand>(D3D11CmdType::Draw, 1u,
      [] (DxvkContext* ctx, const VkDrawIndirectCommand* draws, size_t count) {
        ctx->draw(count, draws);
      });

    new (m_csData->first()) VkDrawIndirectCommand(draw);
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::BatchDrawIndexed(
    const VkDrawIndexedIndirectCommand&     draw) {
    if (unlikely(HasDirtyGraphicsBindings()))
      ApplyDirtyGraphicsBindings();

    // Batch consecutive draws if there are no state changes
    if (m_csDataType == D3D11CmdType::DrawIndexed) {
      auto* drawInfo = m_csChunk->pushData(m_csData, 1u);

      if (likely(drawInfo)) {
        new (drawInfo) VkDrawIndexedIndirectCommand(draw);
        return;
      }
    }

    EmitCsCmd<VkDrawIndexedIndirectCommand>(D3D11CmdType::DrawIndexed, 1u,
      [] (DxvkContext* ctx, const VkDrawIndexedIndirectCommand* draws, size_t count) {
        ctx->drawIndexed(count, draws);
      });

    new (m_csData->first()) VkDrawIndexedIndirectCommand(draw);
  }


  // blessed: gi-cs -- see the declaration. A batch only ever grows while
  // no other command is emitted in between (EmitCs resets m_csDataType), so
  // every draw in one batch sees the same bound cbuffer slices and one
  // patch before the batch covers all of them.
  template<typename ContextType>
  void D3D11CommonContext<ContextType>::BlessedBatchDrawGi(
    const VkDrawIndirectCommand&            draw) {
    if (unlikely(HasDirtyGraphicsBindings()))
      ApplyDirtyGraphicsBindings();

    if (m_csDataType == D3D11CmdType::BlessedDrawGi) {
      auto* drawInfo = m_csChunk->pushData(m_csData, 1u);

      if (likely(drawInfo)) {
        new (drawInfo) VkDrawIndirectCommand(draw);
        return;
      }
    }

    EmitCsCmd<VkDrawIndirectCommand>(D3D11CmdType::BlessedDrawGi, 1u,
      [] (DxvkContext* ctx, const VkDrawIndirectCommand* draws, size_t count) {
        BlessedGi::PatchOnCs(ctx, count);
        ctx->draw(uint32_t(count), draws);
      });

    new (m_csData->first()) VkDrawIndirectCommand(draw);
  }


  // blessed: gi-cs, scene-cs -- picks the batch for an indexed draw. The two
  // claims never meet in practice (scene capture matches the depth
  // prepass, gi the main lit pass); if they ever do, the capture runs from
  // its own cs command just before the gi batch, with the bindings applied.
  template<typename ContextType>
  void D3D11CommonContext<ContextType>::BlessedBatchDrawIndexedAny(
    const VkDrawIndexedIndirectCommand&     draw,
          bool                              gi,
          bool                              scene) {
    if (likely(!(gi | scene))) {
      BatchDrawIndexed(draw);
    } else if (!scene) {
      BlessedBatchDrawIndexedGi(draw);
    } else if (!gi) {
      BlessedBatchDrawIndexedScene(draw, m_blessedSceneNext);
    } else {
      if (unlikely(HasDirtyGraphicsBindings()))
        ApplyDirtyGraphicsBindings();

      EmitCs([cRecord = m_blessedSceneNext, cDraw = draw] (DxvkContext* ctx) {
        BlessedSceneCapture::CaptureOnCs(ctx, cRecord, &cDraw, 1u);
      });

      BlessedBatchDrawIndexedGi(draw);
    }
  }


  // blessed: scene-cs -- see the declaration. Same batching as
  // BatchDrawIndexed, and a claimed draw only joins the open batch while
  // its record is equal, so one record covers every draw of a batch.
  template<typename ContextType>
  void D3D11CommonContext<ContextType>::BlessedBatchDrawIndexedScene(
    const VkDrawIndexedIndirectCommand&     draw,
    const BlessedSceneCsRecord&             record) {
    if (unlikely(HasDirtyGraphicsBindings()))
      ApplyDirtyGraphicsBindings();

    if (m_csDataType == D3D11CmdType::BlessedDrawIndexedScene && m_blessedSceneRecord == record) {
      auto* drawInfo = m_csChunk->pushData(m_csData, 1u);

      if (likely(drawInfo)) {
        new (drawInfo) VkDrawIndexedIndirectCommand(draw);
        return;
      }
    }

    m_blessedSceneRecord = record;

    EmitCsCmd<VkDrawIndexedIndirectCommand>(D3D11CmdType::BlessedDrawIndexedScene, 1u,
      [cRecord = record] (DxvkContext* ctx, const VkDrawIndexedIndirectCommand* draws, size_t count) {
        BlessedSceneCapture::CaptureOnCs(ctx, cRecord, draws, count);
        ctx->drawIndexed(uint32_t(count), draws);
      });

    new (m_csData->first()) VkDrawIndexedIndirectCommand(draw);
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::BlessedBatchDrawIndexedGi(
    const VkDrawIndexedIndirectCommand&     draw) {
    // blessed: gi-bounds -- only ever valid under BLESSED_GI_SAMPLE=bounds
    if (unlikely(m_blessedGiNext.valid)) {
      BlessedBatchDrawIndexedGiBounds(draw);
      return;
    }

    if (unlikely(HasDirtyGraphicsBindings()))
      ApplyDirtyGraphicsBindings();

    if (m_csDataType == D3D11CmdType::BlessedDrawIndexedGi) {
      auto* drawInfo = m_csChunk->pushData(m_csData, 1u);

      if (likely(drawInfo)) {
        new (drawInfo) VkDrawIndexedIndirectCommand(draw);
        return;
      }
    }

    EmitCsCmd<VkDrawIndexedIndirectCommand>(D3D11CmdType::BlessedDrawIndexedGi, 1u,
      [] (DxvkContext* ctx, const VkDrawIndexedIndirectCommand* draws, size_t count) {
        BlessedGi::PatchOnCs(ctx, count);
        ctx->drawIndexed(uint32_t(count), draws);
      });

    new (m_csData->first()) VkDrawIndexedIndirectCommand(draw);
  }


  // blessed: gi-bounds -- see the declaration. Same batching as
  // BlessedBatchDrawIndexedGi, joined only while the record is equal (the
  // same pattern as BlessedBatchDrawIndexedScene).
  template<typename ContextType>
  void D3D11CommonContext<ContextType>::BlessedBatchDrawIndexedGiBounds(
    const VkDrawIndexedIndirectCommand&     draw) {
    if (unlikely(HasDirtyGraphicsBindings()))
      ApplyDirtyGraphicsBindings();

    if (m_csDataType == D3D11CmdType::BlessedDrawIndexedGiBounds && m_blessedGiRecord == m_blessedGiNext) {
      auto* drawInfo = m_csChunk->pushData(m_csData, 1u);

      if (likely(drawInfo)) {
        new (drawInfo) VkDrawIndexedIndirectCommand(draw);
        return;
      }
    }

    m_blessedGiRecord = m_blessedGiNext;

    EmitCsCmd<VkDrawIndexedIndirectCommand>(D3D11CmdType::BlessedDrawIndexedGiBounds, 1u,
      [cRecord = m_blessedGiNext] (DxvkContext* ctx, const VkDrawIndexedIndirectCommand* draws, size_t count) {
        BlessedGi::PatchOnCsBounds(ctx, cRecord, draws, count);
        ctx->drawIndexed(uint32_t(count), draws);
      });

    new (m_csData->first()) VkDrawIndexedIndirectCommand(draw);
  }


  template<typename ContextType>
  bool D3D11CommonContext<ContextType>::BlessedSceneCaptureDraw(
          UINT                              IndexCount,
          UINT                              StartIndexLocation,
          INT                               BaseVertexLocation,
          BlessedSceneCsRecord*             pRecord) {
    if (!BlessedSceneCapture::MatchesSelector(m_state))
      return false;

    // blessed: hook-cpu -- everything below is the "we matched the pass,
    // now resolve/validate/capture this draw" path; MatchesSelector above
    // is the (now generation-cached, see blessed_scene_capture.cpp) part
    // that runs on every draw regardless of whether it matches.
    BLESSED_PROBE_CALL(ContextType, DrawIndexedSceneCaptureResolve);

    // 1. find the position stream (see D3D11InputLayout::SetBlessedPosition)
    auto* layout = m_state.ia.inputLayout.ptr();

    // blessed: actor-skinning -- route skinned draws (BLENDINDICES0/
    // BLENDWEIGHT0 in the input layout) entirely to their own path when
    // enabled. Zero cost when BLESSED_SCENE_SKINNED is unset (one cached
    // bool): the static path below is completely unchanged in that case,
    // and unchanged for every draw this doesn't claim either way.
    if (BlessedSceneCapture::SkinnedEnabled() && layout && layout->HasBlessedSkinning()) {
      BlessedSceneCaptureSkinnedDraw(IndexCount, StartIndexLocation, BaseVertexLocation, layout);
      return false;
    }

    BLESSED_PROBE_CALL(ContextType, DrawIndexedSceneStatic); // blessed: hook-cpu-2

    // blessed: scene-cs -- only the checks that need nothing but the input
    // layout and this context's own D3D11 state run here (no buffer Desc(),
    // no map pointers, no slices). The rest, in the same order, runs on the
    // cs thread in BlessedSceneCapture::CaptureOnCs against the bound dxvk
    // state, right before the draw is recorded.
    if (!layout || !layout->HasBlessedPosition()) {
      BlessedSceneCapture::RecordSkipped(BlessedSceneSkip::NoPosition);
      return false;
    }

    const DxvkVertexAttribute& posAttr = layout->GetBlessedPosition();

    if (posAttr.format != VK_FORMAT_R32G32B32_SFLOAT
     && posAttr.format != VK_FORMAT_R32G32B32A32_SFLOAT
     && posAttr.format != VK_FORMAT_R16G16B16A16_SFLOAT) {
      BlessedSceneCapture::RecordSkipped(BlessedSceneSkip::BadPositionFormat);
      return false;
    }

    if (posAttr.binding >= m_state.ia.maxVbCount) {
      BlessedSceneCapture::RecordSkipped(BlessedSceneSkip::NoPosition);
      return false;
    }

    const auto& vb = m_state.ia.vertexBuffers[posAttr.binding];

    if (!vb.buffer.ptr() || vb.stride == 0) {
      BlessedSceneCapture::RecordSkipped(BlessedSceneSkip::NoPosition);
      return false;
    }

    // (5. dynamic vertex buffer -- cs thread)

    // 6./7. topology, index format: decided here, recorded by the cs thread
    // after its dynamic-vb check, where the old code recorded them
    const auto& ib = m_state.ia.indexBuffer;

    BlessedSceneCsRecord record;
    record.posFormat  = posAttr.format;
    record.posBinding = posAttr.binding;
    record.posOffset  = posAttr.offset;
    record.passIndex  = BlessedSceneCapture::FramePassIndex();

    if (m_state.ia.primitiveTopology != D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST) {
      record.deferredA = BlessedSceneSkip::BadTopology;
    } else if (!ib.buffer.ptr()
     || (ib.format != DXGI_FORMAT_R16_UINT && ib.format != DXGI_FORMAT_R32_UINT)) {
      record.deferredA = BlessedSceneSkip::BadIndexFormat;
    } else {
      // (8. dynamic index buffer -- cs thread)

      // 9. blessed: rt-lifetime bounds audit -- indexData's device address
      // is ib.buffer's slice address + ib.offset; that address must itself
      // be aligned to the index element size, or the BLAS build is a
      // VU-violating out-of-bounds-shaped read that can device-lost the GPU.
      // primitiveOffset (StartIndexLocation * elementSize, in
      // blessed_scene.cpp) is already a multiple of elementSize on its own,
      // so only ib.offset needs checking here.
      UINT ibElementSize = ib.format == DXGI_FORMAT_R16_UINT ? 2u : 4u;

      if ((ib.offset % ibElementSize) != 0)
        record.deferredB = BlessedSceneSkip::MisalignedIndexOffset;
    }

    // (10. negative base vertex, 11.-13. transform, lod scale and camera,
    // 14. base vertex range, and the capture itself -- cs thread)
    *pRecord = record;
    return true;
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::BlessedSceneCaptureSkinnedDraw(
          UINT                              IndexCount,
          UINT                              StartIndexLocation,
          INT                               BaseVertexLocation,
          D3D11InputLayout*                 pLayout) {
    BLESSED_PROBE_CALL(ContextType, DrawIndexedSceneSkinned); // blessed: hook-cpu-2

    // 1. position stream -- same acceptance as the static path (see
    // BlessedSceneCaptureDraw above).
    if (!pLayout->HasBlessedPosition()) {
      BlessedSceneCapture::RecordSkinnedSkipped(BlessedSkinSkip::NoSkinPosition);
      return;
    }

    const DxvkVertexAttribute& posAttr = pLayout->GetBlessedPosition();

    if (posAttr.format != VK_FORMAT_R32G32B32_SFLOAT
     && posAttr.format != VK_FORMAT_R32G32B32A32_SFLOAT
     && posAttr.format != VK_FORMAT_R16G16B16A16_SFLOAT) {
      BlessedSceneCapture::RecordSkinnedSkipped(BlessedSkinSkip::BadSkinPositionFormat);
      return;
    }

    if (posAttr.binding >= m_state.ia.maxVbCount) {
      BlessedSceneCapture::RecordSkinnedSkipped(BlessedSkinSkip::NoSkinPosition);
      return;
    }

    const auto& posVbState = m_state.ia.vertexBuffers[posAttr.binding];

    if (!posVbState.buffer.ptr() || posVbState.stride == 0) {
      BlessedSceneCapture::RecordSkinnedSkipped(BlessedSkinSkip::NoSkinPosition);
      return;
    }

    if (posVbState.buffer->Desc()->Usage == D3D11_USAGE_DYNAMIC) {
      BlessedSceneCapture::RecordSkinnedSkipped(BlessedSkinSkip::DynamicSkinBuffer);
      return;
    }

    // 2. BLENDINDICES0 / BLENDWEIGHT0 streams -- see D3D11InputLayout::SetBlessedSkinning.
    // pLayout->HasBlessedSkinning() is already guaranteed true by the caller.
    const DxvkVertexAttribute& idxAttr = pLayout->GetBlessedSkinIndices();
    const DxvkVertexAttribute& wtAttr  = pLayout->GetBlessedSkinWeights();

    if (idxAttr.format != VK_FORMAT_R8G8B8A8_UNORM
     || wtAttr.format  != VK_FORMAT_R16G16B16A16_SFLOAT) {
      BlessedSceneCapture::RecordSkinnedSkipped(BlessedSkinSkip::BadSkinPositionFormat);
      return;
    }

    if (idxAttr.binding >= m_state.ia.maxVbCount || wtAttr.binding >= m_state.ia.maxVbCount) {
      BlessedSceneCapture::RecordSkinnedSkipped(BlessedSkinSkip::NoSkinPosition);
      return;
    }

    const auto& idxVbState = m_state.ia.vertexBuffers[idxAttr.binding];
    const auto& wtVbState  = m_state.ia.vertexBuffers[wtAttr.binding];

    if (!idxVbState.buffer.ptr() || idxVbState.stride == 0
     || !wtVbState.buffer.ptr()  || wtVbState.stride == 0) {
      BlessedSceneCapture::RecordSkinnedSkipped(BlessedSkinSkip::NoSkinPosition);
      return;
    }

    if (idxVbState.buffer->Desc()->Usage == D3D11_USAGE_DYNAMIC
     || wtVbState.buffer->Desc()->Usage == D3D11_USAGE_DYNAMIC) {
      BlessedSceneCapture::RecordSkinnedSkipped(BlessedSkinSkip::DynamicSkinBuffer);
      return;
    }

    if (m_state.ia.primitiveTopology != D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST) {
      BlessedSceneCapture::RecordSkinnedSkipped(BlessedSkinSkip::BadTopology);
      return;
    }

    const auto& ib = m_state.ia.indexBuffer;

    if (!ib.buffer.ptr()
     || (ib.format != DXGI_FORMAT_R16_UINT && ib.format != DXGI_FORMAT_R32_UINT)) {
      BlessedSceneCapture::RecordSkinnedSkipped(BlessedSkinSkip::BadIndexFormat);
      return;
    }

    if (ib.buffer->Desc()->Usage == D3D11_USAGE_DYNAMIC) {
      BlessedSceneCapture::RecordSkinnedSkipped(BlessedSkinSkip::DynamicSkinBuffer);
      return;
    }

    UINT ibElementSize = ib.format == DXGI_FORMAT_R16_UINT ? 2u : 4u;

    if ((ib.offset % ibElementSize) != 0) {
      BlessedSceneCapture::RecordSkinnedSkipped(BlessedSkinSkip::BadIndexFormat);
      return;
    }

    if (BaseVertexLocation < 0) {
      BlessedSceneCapture::RecordSkinnedSkipped(BlessedSkinSkip::NegativeBaseVertex);
      return;
    }

    // 3. no b2 (world matrix) -- see docs/research/no-b2-draws.md's
    // classification: a skinned depth draw has no per-draw object transform
    // at all, everything lives in the bone buffer below.
    const auto& vsCbv = m_state.cbv[D3D11ShaderType::eVertex];

    if (2u < vsCbv.maxCount && vsCbv.buffers[2].buffer.ptr()) {
      BlessedSceneCapture::RecordSkinnedSkipped(BlessedSkinSkip::HasWorldMatrix);
      return;
    }

    // 4. bones (b10), 3,840 bytes, honouring the bound constant offset --
    // same GetMapPtr()/constantOffset pattern as the static path's b2 read.
    constexpr uint32_t BonesSlot = 10u;
    constexpr uint32_t BonesSize = sizeof(float) * 240u * 4u; // 3,840 bytes

    if (BonesSlot >= vsCbv.maxCount) {
      BlessedSceneCapture::RecordSkinnedSkipped(BlessedSkinSkip::NoBonesBuffer);
      return;
    }

    const auto& bonesCb = vsCbv.buffers[BonesSlot];
    D3D11Buffer* bonesBuffer = bonesCb.buffer.ptr();
    void* bonesMapPtr = bonesBuffer ? bonesBuffer->GetMapPtr() : nullptr;

    if (!bonesMapPtr) {
      BlessedSceneCapture::RecordSkinnedSkipped(BlessedSkinSkip::NoBonesBuffer);
      return;
    }

    UINT bonesByteOffset = bonesCb.constantOffset * 16u;

    if (bonesBuffer->Desc()->ByteWidth != BonesSize
     || bonesByteOffset + BonesSize > bonesBuffer->Desc()->ByteWidth) {
      BlessedSceneCapture::RecordSkinnedSkipped(BlessedSkinSkip::BadBonesSize);
      return;
    }

    // blessed: hook-cpu-2 -- the bones are never read here: the draw keeps
    // the discard allocation b10 is mapped to (D3D11Buffer remembers it for
    // bones-shaped buffers), and the skin dispatch reads it by address.
    // Always set for such a buffer; the check only guards the impossible.
    const Rc<DxvkResourceAllocation>& bonesAllocation = bonesBuffer->BlessedMappedAllocation();

    if (!bonesAllocation || bonesAllocation->mapPtr() != bonesMapPtr) {
      BlessedSceneCapture::RecordSkinnedSkipped(BlessedSkinSkip::NoBonesBuffer);
      return;
    }

    // 5. pivot -- blessed: skin-repair. Skinned::GetBoneTransformMatrix
    // subtracts the vertex shader's OWN b12 byte 640 (BLESSED_SCENE_SKIN_PIVOT,
    // default vs:12:640), whatever it holds in this pass. This used to read
    // BLESSED_SCENE_CAMPOS, which tags the tlas camera and is not the same
    // quantity (in the prepass vs b12:640 is not the camera position).
    BlessedSceneCbufferSlot pivotCfg = BlessedSceneCapture::SkinPivotConfig();
    const auto& pivotCbv = m_state.cbv[pivotCfg.stage];

    if (!pivotCfg.valid || pivotCfg.slot >= pivotCbv.maxCount) {
      BlessedSceneCapture::RecordSkinnedSkipped(BlessedSkinSkip::NoCamPos);
      return;
    }

    const auto& pivotCb = pivotCbv.buffers[pivotCfg.slot];
    D3D11Buffer* pivotBuffer = pivotCb.buffer.ptr();
    void* pivotMapPtr = pivotBuffer ? pivotBuffer->GetMapPtr() : nullptr;
    if (pivotBuffer) // blessed: perf-halfrate -- read per skinned draw: keep it cached
      pivotBuffer->GetBuffer()->blessedMarkCpuRead();

    if (!pivotMapPtr) {
      BlessedSceneCapture::RecordSkinnedSkipped(BlessedSkinSkip::NoCamPos);
      return;
    }

    UINT pivotByteOffset = pivotCb.constantOffset * 16u + pivotCfg.offset;

    if (pivotByteOffset + 12u > pivotBuffer->Desc()->ByteWidth) {
      BlessedSceneCapture::RecordSkinnedSkipped(BlessedSkinSkip::NoCamPos);
      return;
    }

    // blessed: hook-cpu-2 -- the pose dedup (same mesh + same bones twice in
    // one frame) no longer hashes the bones here: the cs side compares the
    // arena snapshots directly, and only when a mesh repeats.

    // 6. vertex counts per stream (each stream's own binding may have a
    // different byte width/stride) -- bound by the smallest, so the compute
    // shader never reads past any one of the three buffers.
    auto StreamVertexCount = [] (const D3D11VertexBufferBinding& vbState, const DxvkVertexAttribute& attr) -> UINT {
      UINT byteWidth  = vbState.buffer->Desc()->ByteWidth;
      UINT absOffset  = vbState.offset + attr.offset;
      UINT elemSize   = UINT(lookupFormatInfo(attr.format)->elementSize);
      return absOffset + elemSize <= byteWidth ? (byteWidth - absOffset - elemSize) / vbState.stride + 1u : 0u;
    };

    UINT posVertexCount = StreamVertexCount(posVbState, posAttr);
    UINT idxVertexCount = StreamVertexCount(idxVbState, idxAttr);
    UINT wtVertexCount  = StreamVertexCount(wtVbState,  wtAttr);
    UINT vertexCount    = std::min({ posVertexCount, idxVertexCount, wtVertexCount });

    if (UINT(BaseVertexLocation) >= vertexCount) {
      BlessedSceneCapture::RecordSkinnedSkipped(BlessedSkinSkip::BaseVertexOutOfRange);
      return;
    }

    // blessed: hook-cpu-2 -- a draw from a depth-only pass after the mask
    // pass can never be the tracer's pass (see SkinnedDrawIsLate). Every
    // check above still ran, so the captured/skipped counts are unchanged;
    // only the copy and the hand-off are skipped.
    if (BlessedSceneCapture::SkinnedDrawIsLate()) {
      BlessedSceneCapture::CountLateSkinnedDraw();
      BlessedSceneCapture::RecordSkinnedCaptured();
      return;
    }

    BLESSED_PROBE_CALL(ContextType, DrawIndexedSceneSkinnedStage); // blessed: hook-cpu-2

    BlessedSceneSkinnedDraw sceneDraw;
    std::memcpy(sceneDraw.pivot,
      reinterpret_cast<const uint8_t*>(pivotMapPtr) + pivotByteOffset, 12u);

    // blessed: skin-repair -- BLESSED_SCENE_SKIN_TRACE only
    if (unlikely(BlessedSceneCapture::SkinTraceEnabled())) {
      sceneDraw.bonesUsage = uint32_t(bonesBuffer->Desc()->Usage);

      BlessedSceneCbufferSlot camCfg = BlessedSceneCapture::CamPosConfig();
      const auto& camCbv = m_state.cbv[camCfg.stage];

      if (camCfg.valid && camCfg.slot < camCbv.maxCount) {
        const auto& camCb = camCbv.buffers[camCfg.slot];
        D3D11Buffer* camBuffer = camCb.buffer.ptr();
        void* camMapPtr = camBuffer ? camBuffer->GetMapPtr() : nullptr;
        UINT camByteOffset = camCb.constantOffset * 16u + camCfg.offset;

        if (camMapPtr && camByteOffset + 12u <= camBuffer->Desc()->ByteWidth) {
          std::memcpy(sceneDraw.tagCam,
            reinterpret_cast<const uint8_t*>(camMapPtr) + camByteOffset, 12u);
          sceneDraw.hasTagCam = true;
        }
      }
    }

    // 7. package the rest -- attribute offsets are baked into each slice's
    // own offset (see BlessedSceneSkinnedDraw), matching how the static
    // path folds POSITION0's offset into sceneDraw.vb above.
    { BLESSED_PROBE_CALL(ContextType, DrawIndexedSceneSkinnedSlices); // blessed: hook-cpu-2
      sceneDraw.posVb     = posVbState.buffer->GetBufferSlice(
        posVbState.offset + posAttr.offset, VkDeviceSize(vertexCount) * posVbState.stride);
      sceneDraw.posStride = posVbState.stride;
      sceneDraw.posFormat = posAttr.format;

      sceneDraw.idxVb     = idxVbState.buffer->GetBufferSlice(
        idxVbState.offset + idxAttr.offset, VkDeviceSize(vertexCount) * idxVbState.stride);
      sceneDraw.idxStride = idxVbState.stride;

      sceneDraw.wtVb      = wtVbState.buffer->GetBufferSlice(
        wtVbState.offset + wtAttr.offset, VkDeviceSize(vertexCount) * wtVbState.stride);
      sceneDraw.wtStride  = wtVbState.stride;

      sceneDraw.vertexCount = vertexCount;

      sceneDraw.ib         = ib.buffer->GetBufferSlice(ib.offset);
      sceneDraw.indexType  = ib.format == DXGI_FORMAT_R16_UINT ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
      sceneDraw.indexCount = IndexCount;
      sceneDraw.startIndex = StartIndexLocation;
      sceneDraw.baseVertex = BaseVertexLocation;

      sceneDraw.bonesAllocation = bonesAllocation;
      sceneDraw.bonesOffset     = bonesByteOffset;
    }

    // blessed: skin-v2 -- which depth-only pass this draw came from; the cs
    // side keeps only the pass before the mask draw
    sceneDraw.pass = BlessedSceneCapture::CurrentPass();

    // blessed: hook-cpu-2 -- staged for the frame, not emitted per draw:
    // the batch goes to the cs thread with this frame's endFrame
    BlessedSceneCapture::StageSkinnedDraw(sceneDraw);

    BlessedSceneCapture::RecordSkinnedCaptured();
  }


  template<typename ContextType>
  template<D3D11ShaderType ShaderStage>
  void D3D11CommonContext<ContextType>::BindShader(
    const D3D11CommonShader*    pShaderModule) {
    uint64_t oldUavMask = m_state.lazy.bindingsUsed[ShaderStage].uavMask;

    if (pShaderModule) {
      auto buffer = pShaderModule->GetIcb();
      auto shader = pShaderModule->GetShader();

      if (unlikely(shader->needsCompile()))
        m_device->requestCompileShader(shader);

      // If this shader activates any bindings that have not yet been applied,
      // mark the shader stage as dirty so it gets applied on the next draw.
      // Don't apply it right away since any dirty bindings are likely redundant.
      m_state.lazy.shadersUsed.set(ShaderStage);
      m_state.lazy.bindingsUsed[ShaderStage] = pShaderModule->GetBindingMask();

      if (!m_state.lazy.shadersDirty.test(ShaderStage) && (DebugLazyBinding != Tristate::False)) {
        if (!(m_state.lazy.bindingsDirty[ShaderStage] & m_state.lazy.bindingsUsed[ShaderStage]).empty())
          m_state.lazy.shadersDirty.set(ShaderStage);
      }

      EmitCs([
        cBuffer = std::move(buffer),
        cShader = std::move(shader)
      ] (DxvkContext* ctx) mutable {
        constexpr VkShaderStageFlagBits stage = GetShaderStage(ShaderStage);

        uint32_t slotId = D3D11ShaderResourceMapping::computeCbvBinding(ShaderStage,
          D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT);

        ctx->bindShader<stage>(
          Forwarder::move(cShader));
        ctx->bindUniformBuffer(stage, slotId,
          Forwarder::move(cBuffer));
      });
    } else {
      // Mark shader stage as inactive and clean since we'll have no active
      // bindings. This works because if the app changes any binding at all
      // for this stage, it will get flagged as dirty, and if another shader
      // gets bound, it will check for any dirty bindings again.
      m_state.lazy.shadersUsed.clr(ShaderStage);
      m_state.lazy.shadersDirty.clr(ShaderStage);

      m_state.lazy.bindingsUsed[ShaderStage].reset();

      EmitCs([] (DxvkContext* ctx) {
        constexpr VkShaderStageFlagBits stage = GetShaderStage(ShaderStage);

        uint32_t slotId = D3D11ShaderResourceMapping::computeCbvBinding(ShaderStage,
          D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT);

        ctx->bindShader<stage>(nullptr);
        ctx->bindUniformBuffer(stage, slotId, DxvkBufferSlice());
      });
    }

    // On graphics, UAVs are available to all stages, but we treat them as part
    // of the pixel shader binding set. Re-compute the active UAV mask. We don't
    // need to set the PS as active or dirty here though since the UAV update
    // code will mark all other stages that access UAVs as dirty, too.
    uint64_t newUavMask = m_state.lazy.bindingsUsed[ShaderStage].uavMask;

    if (ShaderStage != D3D11ShaderType::eCompute && oldUavMask != newUavMask) {
      constexpr D3D11ShaderType ps = D3D11ShaderType::ePixel;

      // Since dirty UAVs are only tracked on the PS mask, we need to mark the
      // stage as dirty if any of the used UAVs overlap with the dirty PS mask.
      if (m_state.lazy.bindingsDirty[ps].uavMask & newUavMask)
        m_state.lazy.shadersDirty.set(ShaderStage);

      // Accumulate graphics UAV mask and write it back to the pixel shader mask.
      m_state.lazy.graphicsUavShaders.clr(ShaderStage);

      for (uint32_t stageIndex : bit::BitMask(uint32_t(m_state.lazy.graphicsUavShaders.raw())))
        newUavMask |= m_state.lazy.bindingsUsed[D3D11ShaderType(stageIndex)].uavMask;

      m_state.lazy.bindingsUsed[ps].uavMask = newUavMask;

      // Update bit mask of shaders actively accessing graphics UAVs
      if (newUavMask)
        m_state.lazy.graphicsUavShaders.set(ShaderStage);
    }
  }


  static VkDepthBiasRepresentationEXT FormatToDepthBiasRepresentation(DXGI_FORMAT format) {
    switch (format) {
      default:
      case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
      case DXGI_FORMAT_D32_FLOAT:
        return VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORMAT_EXT;

      case DXGI_FORMAT_D24_UNORM_S8_UINT:
      case DXGI_FORMAT_D16_UNORM:
        return VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORCE_UNORM_EXT;
    }
  }

  template<typename ContextType>
  void D3D11CommonContext<ContextType>::BindFramebuffer() {
    DxvkDepthBiasRepresentation depthBiasRepresentation =
      { VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORMAT_EXT,
        m_device->features().extDepthBiasControl.depthBiasExact };
    DxvkRenderTargets attachments;
    uint32_t sampleCount = 0;

    // D3D11 doesn't have the concept of a framebuffer object,
    // so we'll just create a new one every time the render
    // target bindings are updated. Set up the attachments.
    for (UINT i = 0; i < m_state.om.rtvs.size(); i++) {
      if (m_state.om.rtvs[i] != nullptr) {
        Rc<DxvkImageView> view = m_state.om.rtvs[i]->GetImageView();

        // blessed: zero-copy-present -- a back-buffer view renders into the
        // acquired swap chain image instead, see blessed_zero_copy.h.
        // Immediate context only: a deferred command list may replay into
        // a later frame (ExecuteCommandList ends the redirect instead).
        if constexpr (!IsDeferred) {
          if (unlikely(BlessedZeroCopy::IsEnabled())) {
            Rc<DxvkImageView> redirect = BlessedZeroCopy::RedirectRenderTarget(
              GetTypedContext(), view, false);

            if (redirect != nullptr)
              view = std::move(redirect);
          }
        }

        attachments.color[i].view = view;
        attachments.color[i].shadow = m_state.om.rtvs[i]->GetBufferView();
        sampleCount = m_state.om.rtvs[i]->GetSampleCount();
      }
    }

    if (m_state.om.dsv != nullptr) {
      attachments.depth.view = m_state.om.dsv->GetImageView();
      sampleCount = m_state.om.dsv->GetSampleCount();

      if (m_device->features().extDepthBiasControl.leastRepresentableValueForceUnormRepresentation)
        depthBiasRepresentation.depthBiasRepresentation = FormatToDepthBiasRepresentation(m_state.om.dsv->GetViewFormat());
    }

    // Create and bind the framebuffer object to the context
    EmitCs([
      cAttachments    = std::move(attachments),
      cRepresentation = depthBiasRepresentation
    ] (DxvkContext* ctx) mutable {
      ctx->setDepthBiasRepresentation(cRepresentation);
      ctx->bindRenderTargets(Forwarder::move(cAttachments), 0u);
    });

    // If necessary, update push constant for the sample count
    if (m_state.om.sampleCount != sampleCount) {
      m_state.om.sampleCount = sampleCount;
      ApplyRasterizerSampleCount();
    }
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::BindDrawBuffers(
          D3D11Buffer*                     pBufferForArgs,
          D3D11Buffer*                     pBufferForCount) {
    EmitCs([
      cArgBuffer = pBufferForArgs  ? pBufferForArgs->GetBufferSlice()  : DxvkBufferSlice(),
      cCntBuffer = pBufferForCount ? pBufferForCount->GetBufferSlice() : DxvkBufferSlice()
    ] (DxvkContext* ctx) mutable {
      ctx->bindDrawBuffers(
        Forwarder::move(cArgBuffer),
        Forwarder::move(cCntBuffer));
    });
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::BindVertexBuffer(
          UINT                              Slot,
          D3D11Buffer*                      pBuffer,
          UINT                              Offset,
          UINT                              Stride) {
    if (pBuffer) {
      EmitCs([
        cSlotId       = Slot,
        cBufferSlice  = pBuffer->GetBufferSlice(Offset),
        cStride       = Stride
      ] (DxvkContext* ctx) mutable {
        ctx->bindVertexBuffer(cSlotId,
          Forwarder::move(cBufferSlice),
          cStride);
      });
    } else {
      EmitCs([
        cSlotId       = Slot
      ] (DxvkContext* ctx) {
        ctx->bindVertexBuffer(cSlotId, DxvkBufferSlice(), 0);
      });
    }
  }
  
  
  template<typename ContextType>
  void D3D11CommonContext<ContextType>::BindVertexBufferRange(
          UINT                              Slot,
          D3D11Buffer*                      pBuffer,
          UINT                              Offset,
          UINT                              Stride) {
    if (pBuffer) {
      VkDeviceSize offset = Offset;
      VkDeviceSize length = pBuffer->GetRemainingSize(Offset);

      EmitCs([
        cSlotId       = Slot,
        cBufferOffset = offset,
        cBufferLength = length,
        cStride       = Stride
      ] (DxvkContext* ctx) mutable {
        ctx->bindVertexBufferRange(cSlotId,
          cBufferOffset, cBufferLength, cStride);
      });
    }
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::BindIndexBuffer(
          D3D11Buffer*                      pBuffer,
          UINT                              Offset,
          DXGI_FORMAT                       Format) {
    VkIndexType indexType = Format == DXGI_FORMAT_R16_UINT
      ? VK_INDEX_TYPE_UINT16
      : VK_INDEX_TYPE_UINT32;

    if (pBuffer) {
      EmitCs([
        cBufferSlice  = pBuffer->GetBufferSlice(Offset),
        cIndexType    = indexType
      ] (DxvkContext* ctx) mutable {
        ctx->bindIndexBuffer(
          Forwarder::move(cBufferSlice),
          cIndexType);
      });
    } else {
      EmitCs([
        cIndexType    = indexType
      ] (DxvkContext* ctx) {
        ctx->bindIndexBuffer(DxvkBufferSlice(), cIndexType);
      });
    }
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::BindIndexBufferRange(
          D3D11Buffer*                      pBuffer,
          UINT                              Offset,
          DXGI_FORMAT                       Format) {
    if (pBuffer) {
      VkIndexType indexType = Format == DXGI_FORMAT_R16_UINT
        ? VK_INDEX_TYPE_UINT16
        : VK_INDEX_TYPE_UINT32;

      VkDeviceSize offset = Offset;
      VkDeviceSize length = pBuffer->GetRemainingSize(Offset);

      EmitCs([
        cBufferOffset = offset,
        cBufferLength = length,
        cIndexType    = indexType
      ] (DxvkContext* ctx) mutable {
        ctx->bindIndexBufferRange(
          cBufferOffset, cBufferLength,
          cIndexType);
      });
    }
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::BindXfbBuffer(
          UINT                              Slot,
          D3D11Buffer*                      pBuffer,
          UINT                              Offset) {
    if (pBuffer) {
      EmitCs([
        cSlotId       = Slot,
        cOffset       = Offset,
        cBufferSlice  = pBuffer->GetBufferSlice(),
        cCounterSlice = pBuffer->GetSOCounter()
      ] (DxvkContext* ctx) mutable {
        if (cCounterSlice.defined() && cOffset != ~0u) {
          ctx->updateBuffer(
            cCounterSlice.buffer(),
            cCounterSlice.offset(),
            sizeof(cOffset),
            &cOffset);
        }

        ctx->bindXfbBuffer(cSlotId,
          Forwarder::move(cBufferSlice),
          Forwarder::move(cCounterSlice));
      });
    } else {
      EmitCs([
        cSlotId       = Slot
      ] (DxvkContext* ctx) {
        ctx->bindXfbBuffer(cSlotId,
          DxvkBufferSlice(),
          DxvkBufferSlice());
      });
    }
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::BindConstantBuffer(
          D3D11ShaderType                   ShaderStage,
          UINT                              Slot,
          D3D11Buffer*                      pBuffer,
          UINT                              Offset,
          UINT                              Length) {
    uint32_t slotId = D3D11ShaderResourceMapping::computeCbvBinding(ShaderStage, Slot);

    if (pBuffer) {
      EmitCs([
        cSlotId      = slotId,
        cStage       = GetShaderStage(ShaderStage),
        cBufferSlice = pBuffer->GetBufferSlice(16 * Offset, 16 * Length)
      ] (DxvkContext* ctx) mutable {
        ctx->bindUniformBuffer(cStage, cSlotId,
          Forwarder::move(cBufferSlice));
      });
    } else {
      EmitCs([
        cSlotId      = slotId,
        cStage       = GetShaderStage(ShaderStage)
      ] (DxvkContext* ctx) {
        ctx->bindUniformBuffer(cStage, cSlotId, DxvkBufferSlice());
      });
    }
  }
  
  
  template<typename ContextType>
  void D3D11CommonContext<ContextType>::BindConstantBufferRange(
          D3D11ShaderType                   ShaderStage,
          UINT                              Slot,
          UINT                              Offset,
          UINT                              Length) {
    uint32_t slotId = D3D11ShaderResourceMapping::computeCbvBinding(ShaderStage, Slot);

    EmitCs([
      cSlotId = slotId,
      cStage  = GetShaderStage(ShaderStage),
      cOffset = 16u * Offset,
      cLength = 16u * Length
    ] (DxvkContext* ctx) {
      ctx->bindUniformBufferRange(cStage, cSlotId, cOffset, cLength);
    });
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::BindSampler(
          D3D11ShaderType                   ShaderStage,
          UINT                              Slot,
          D3D11SamplerState*                pSampler) {
    uint32_t slotId = D3D11ShaderResourceMapping::computeSamplerBinding(ShaderStage, Slot);

    if (pSampler) {
      EmitCs([
        cSlotId   = slotId,
        cStage    = GetShaderStage(ShaderStage),
        cSampler  = pSampler->GetDXVKSampler()
      ] (DxvkContext* ctx) mutable {
        ctx->bindResourceSampler(cStage, cSlotId,
          Forwarder::move(cSampler));
      });
    } else {
      EmitCs([
        cSlotId   = slotId,
        cStage    = GetShaderStage(ShaderStage)
      ] (DxvkContext* ctx) {
        ctx->bindResourceSampler(cStage, cSlotId, nullptr);
      });
    }
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::BindShaderResource(
          D3D11ShaderType                   ShaderStage,
          UINT                              Slot,
          D3D11ShaderResourceView*          pResource) {
    uint32_t slotId = D3D11ShaderResourceMapping::computeSrvBinding(ShaderStage, Slot);

    if (pResource) {
      if (pResource->GetViewInfo().Dimension != D3D11_RESOURCE_DIMENSION_BUFFER) {
        EmitCs([
          cSlotId = slotId,
          cStage  = GetShaderStage(ShaderStage),
          cView   = pResource->GetImageView()
        ] (DxvkContext* ctx) mutable {
          ctx->bindResourceImageView(cStage, cSlotId,
            Forwarder::move(cView));
        });
      } else {
        EmitCs([
          cSlotId = slotId,
          cStage  = GetShaderStage(ShaderStage),
          cView   = pResource->GetBufferView()
        ] (DxvkContext* ctx) mutable {
          ctx->bindResourceBufferView(cStage, cSlotId,
            Forwarder::move(cView));
        });
      }
    } else {
      EmitCs([
        cSlotId = slotId,
        cStage  = GetShaderStage(ShaderStage)
      ] (DxvkContext* ctx) {
        ctx->bindResourceImageView(cStage, cSlotId, nullptr);
      });
    }
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::BindUnorderedAccessView(
          D3D11ShaderType                   ShaderStage,
          UINT                              Slot,
          D3D11UnorderedAccessView*         pUav) {
    uint32_t uavSlotId = D3D11ShaderResourceMapping::computeUavBinding(ShaderStage, Slot);
    uint32_t ctrSlotId = D3D11ShaderResourceMapping::computeUavCounterBinding(ShaderStage, Slot);

    VkShaderStageFlags stages = ShaderStage == D3D11ShaderType::eCompute
      ? VK_SHADER_STAGE_COMPUTE_BIT
      : VK_SHADER_STAGE_ALL_GRAPHICS;

    if (pUav) {
      if (pUav->GetViewInfo().Dimension == D3D11_RESOURCE_DIMENSION_BUFFER) {
        EmitCs([
          cUavSlotId    = uavSlotId,
          cCtrSlotId    = ctrSlotId,
          cStages       = stages,
          cBufferView   = pUav->GetBufferView(),
          cCounterView  = pUav->GetCounterView()
        ] (DxvkContext* ctx) mutable {
          ctx->bindResourceBufferView(cStages, cUavSlotId,
            Forwarder::move(cBufferView));
          ctx->bindResourceBufferView(cStages, cCtrSlotId,
            Forwarder::move(cCounterView));
        });
      } else {
        EmitCs([
          cUavSlotId    = uavSlotId,
          cCtrSlotId    = ctrSlotId,
          cStages       = stages,
          cImageView    = pUav->GetImageView()
        ] (DxvkContext* ctx) mutable {
          ctx->bindResourceImageView(cStages, cUavSlotId,
            Forwarder::move(cImageView));
          ctx->bindResourceBufferView(cStages, cCtrSlotId, nullptr);
        });
      }
    } else {
      EmitCs([
        cUavSlotId    = uavSlotId,
        cCtrSlotId    = ctrSlotId,
        cStages       = stages
      ] (DxvkContext* ctx) {
        ctx->bindResourceImageView(cStages, cUavSlotId, nullptr);
        ctx->bindResourceBufferView(cStages, cCtrSlotId, nullptr);
      });
    }
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::ClearImageView(
          Rc<DxvkImageView>                 View,
    const FLOAT                             Color[4],
    const D3D11_RECT*                       pRects,
          UINT                              NumRects) {
    // 3D views are unsupported
    if (View->info().viewType == VK_IMAGE_VIEW_TYPE_3D)
      return;

    // Convert clear value
    auto clearValue = ConvertColorValue(Color, View->formatInfo());

    VkExtent3D extent3D = View->mipLevelExtent(0);
    VkExtent2D extent2D = { extent3D.width, extent3D.height };

    // Figure out which plane we're clearing for subsampling
    const DxvkPlaneFormatInfo* plane = nullptr;
    auto imageFormatInfo = View->image()->formatInfo();

    if (imageFormatInfo->flags.test(DxvkFormatFlag::MultiPlane))
      plane = &imageFormatInfo->planes[vk::getPlaneIndex(View->info().aspects)];

    // Clear all non-empty rectangles
    EmitCsCmd<VkRect2D>(D3D11CmdType::None, std::max(NumRects, 1u), [
      cView       = std::move(View),
      cClearValue = clearValue
    ] (DxvkContext* ctx, const VkRect2D* rects, size_t count) {
      VkImageAspectFlags clearAspect = cView->formatInfo()->aspectMask & (VK_IMAGE_ASPECT_COLOR_BIT | VK_IMAGE_ASPECT_DEPTH_BIT);

      for (size_t i = 0; i < count; i++) {
        VkOffset3D offset = { rects[i].offset.x, rects[i].offset.y, 0 };
        VkExtent3D extent = { rects[i].extent.width, rects[i].extent.height, 1u };

        if (extent.width && extent.height)
          ctx->clearImageView(cView, offset, extent, clearAspect, cClearValue);
      }
    });

    if (NumRects) {
      for (uint32_t i = 0; i < NumRects; i++) {
        D3D11_RECT subsampledRect = pRects[i];

        if (plane) {
          subsampledRect.left   /= plane->blockSize.width;
          subsampledRect.top    /= plane->blockSize.height;
          subsampledRect.right  /= plane->blockSize.width;
          subsampledRect.bottom /= plane->blockSize.height;
        }

        new (m_csData->at(i)) VkRect2D(ConvertRect(subsampledRect, extent2D));
      }
    } else {
      auto vkRect = new (m_csData->first()) VkRect2D();
      vkRect->offset = VkOffset2D { 0, 0 };
      vkRect->extent = extent2D;
    }
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::ClearBufferView(
          Rc<DxvkBufferView>                View,
    const FLOAT                             Color[4],
    const D3D11_RECT*                       pRects,
          UINT                              NumRects) {
    // Convert clear value
    auto formatInfo = View->formatInfo();
    auto clearValue = ConvertColorValue(Color, formatInfo);

    // Just pass the rectangles through, even though we only need one dimension
    VkExtent2D extent2D = { uint32_t(View->info().size / formatInfo->elementSize), 1u };

    EmitCsCmd<VkRect2D>(D3D11CmdType::None, std::max(NumRects, 1u), [
      cView       = std::move(View),
      cClearValue = clearValue
    ] (DxvkContext* ctx, const VkRect2D* rects, size_t count) {
      for (size_t i = 0; i < count; i++) {
        if (rects[i].extent.width) {
          ctx->clearBufferView(cView,
            rects[i].offset.x, rects[i].extent.width,
            cClearValue.color);
        }
      }
    });

    if (NumRects) {
      for (uint32_t i = 0; i < NumRects; i++)
        new (m_csData->at(i)) VkRect2D(ConvertRect(pRects[i], extent2D));
    } else {
      auto vkRect = new (m_csData->first()) VkRect2D();
      vkRect->offset = VkOffset2D { 0, 0 };
      vkRect->extent = extent2D;
    }
  }


  template<typename ContextType>
  VkClearValue D3D11CommonContext<ContextType>::ConvertColorValue(
    const FLOAT                             Color[4],
    const DxvkFormatInfo*                   pFormatInfo) {
    VkClearValue result;

    if (pFormatInfo->aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) {
      for (uint32_t i = 0; i < 4; i++) {
        if (pFormatInfo->flags.test(DxvkFormatFlag::SampledUInt))
          result.color.uint32[i] = uint32_t(std::max(0.0f, Color[i]));
        else if (pFormatInfo->flags.test(DxvkFormatFlag::SampledSInt))
          result.color.int32[i] = int32_t(Color[i]);
        else
          result.color.float32[i] = Color[i];
      }
    } else {
      result.depthStencil.depth = Color[0];
      result.depthStencil.stencil = 0;
    }

    return result;
  }


  template<typename ContextType>
  VkRect2D D3D11CommonContext<ContextType>::ConvertRect(
          D3D11_RECT                        Rect,
          VkExtent2D                        Extent) {
    Rect.left   = std::max<int32_t>(Rect.left,   0);
    Rect.top    = std::max<int32_t>(Rect.top,    0);
    Rect.right  = std::min<int32_t>(Rect.right,  Extent.width);
    Rect.bottom = std::min<int32_t>(Rect.bottom, Extent.height);

    if (Rect.left >= Rect.right || Rect.top >= Rect.bottom)
      return VkRect2D();

    VkRect2D result = { };
    result.offset.x = Rect.left;
    result.offset.y = Rect.top;
    result.extent.width = Rect.right - Rect.left;
    result.extent.height = Rect.bottom - Rect.top;
    return result;
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::CopyBuffer(
          D3D11Buffer*                      pDstBuffer,
          VkDeviceSize                      DstOffset,
          D3D11Buffer*                      pSrcBuffer,
          VkDeviceSize                      SrcOffset,
          VkDeviceSize                      ByteCount) {
    // Clamp copy region to prevent out-of-bounds access
    VkDeviceSize dstLength = pDstBuffer->Desc()->ByteWidth;
    VkDeviceSize srcLength = pSrcBuffer->Desc()->ByteWidth;

    if (SrcOffset >= srcLength || DstOffset >= dstLength || !ByteCount)
      return;

    AddCost(GpuCostEstimate::Transfer);

    ByteCount = std::min(dstLength - DstOffset, ByteCount);
    ByteCount = std::min(srcLength - SrcOffset, ByteCount);

    EmitCs([
      cDstBuffer = pDstBuffer->GetBufferSlice(DstOffset, ByteCount),
      cSrcBuffer = pSrcBuffer->GetBufferSlice(SrcOffset, ByteCount)
    ] (DxvkContext* ctx) {
      if (cDstBuffer.buffer() != cSrcBuffer.buffer()) {
        ctx->copyBuffer(
          cDstBuffer.buffer(),
          cDstBuffer.offset(),
          cSrcBuffer.buffer(),
          cSrcBuffer.offset(),
          cSrcBuffer.length());
      } else {
        ctx->copyBufferRegion(
          cDstBuffer.buffer(),
          cDstBuffer.offset(),
          cSrcBuffer.offset(),
          cSrcBuffer.length());
      }
    });

    if (pDstBuffer->HasSequenceNumber())
      GetTypedContext()->TrackBufferSequenceNumber(pDstBuffer);
    if (pSrcBuffer->HasSequenceNumber())
      GetTypedContext()->TrackBufferSequenceNumber(pSrcBuffer);
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::CopyImage(
          D3D11CommonTexture*               pDstTexture,
    const VkImageSubresourceLayers*         pDstLayers,
          VkOffset3D                        DstOffset,
          D3D11CommonTexture*               pSrcTexture,
    const VkImageSubresourceLayers*         pSrcLayers,
          VkOffset3D                        SrcOffset,
          VkExtent3D                        SrcExtent) {
    // Image formats must be size-compatible
    auto dstFormatInfo = lookupFormatInfo(pDstTexture->GetPackedFormat());
    auto srcFormatInfo = lookupFormatInfo(pSrcTexture->GetPackedFormat());

    if (dstFormatInfo->elementSize != srcFormatInfo->elementSize)
      return;

    // Sample counts must match
    if (pDstTexture->Desc()->SampleDesc.Count != pSrcTexture->Desc()->SampleDesc.Count)
      return;

    // Obviously, the copy region must not be empty
    VkExtent3D dstMipExtent = pDstTexture->MipLevelExtent(pDstLayers->mipLevel);
    VkExtent3D srcMipExtent = pSrcTexture->MipLevelExtent(pSrcLayers->mipLevel);

    if (uint32_t(DstOffset.x) >= dstMipExtent.width
     || uint32_t(DstOffset.y) >= dstMipExtent.height
     || uint32_t(DstOffset.z) >= dstMipExtent.depth)
      return;

    if (uint32_t(SrcOffset.x) >= srcMipExtent.width
     || uint32_t(SrcOffset.y) >= srcMipExtent.height
     || uint32_t(SrcOffset.z) >= srcMipExtent.depth)
      return;

    // Don't perform the copy if the offsets aren't block-aligned
    if (!util::isBlockAligned(SrcOffset, srcFormatInfo->blockSize)
     || !util::isBlockAligned(DstOffset, dstFormatInfo->blockSize))
      return;

    // Clamp the image region in order to avoid out-of-bounds access
    VkExtent3D blockCount    = util::computeBlockCount(SrcExtent, srcFormatInfo->blockSize);
    VkExtent3D dstBlockCount = util::computeMaxBlockCount(DstOffset, dstMipExtent, dstFormatInfo->blockSize);
    VkExtent3D srcBlockCount = util::computeMaxBlockCount(SrcOffset, srcMipExtent, srcFormatInfo->blockSize);

    blockCount = util::minExtent3D(blockCount, dstBlockCount);
    blockCount = util::minExtent3D(blockCount, srcBlockCount);

    SrcExtent = util::computeBlockExtent(blockCount, srcFormatInfo->blockSize);
    SrcExtent = util::snapExtent3D(SrcOffset, SrcExtent, srcMipExtent);

    if (!SrcExtent.width || !SrcExtent.height || !SrcExtent.depth)
      return;

    AddCost(GpuCostEstimate::Transfer);

    // While copying between 2D and 3D images is allowed in CopySubresourceRegion,
    // copying more than one slice at a time is not suppoted. Layer counts are 1.
    if ((pDstTexture->GetVkImageType() == VK_IMAGE_TYPE_3D)
     != (pSrcTexture->GetVkImageType() == VK_IMAGE_TYPE_3D))
      SrcExtent.depth = 1;

    // Certain types of copies require us to pass the destination extent to
    // the backend. This may be different when copying between compressed
    // and uncompressed image formats.
    VkExtent3D dstExtent = util::computeBlockExtent(blockCount, dstFormatInfo->blockSize);
    dstExtent = util::snapExtent3D(DstOffset, dstExtent, dstMipExtent);

    // It is possible for any of the given images to be a staging image with
    // no actual image, so we need to account for all possibilities here.
    bool dstIsImage = pDstTexture->HasImage();
    bool srcIsImage = pSrcTexture->HasImage();

    if (dstIsImage && srcIsImage) {
      EmitCs([
        cDstImage  = pDstTexture->GetImage(),
        cSrcImage  = pSrcTexture->GetImage(),
        cDstLayers = *pDstLayers,
        cSrcLayers = *pSrcLayers,
        cDstOffset = DstOffset,
        cSrcOffset = SrcOffset,
        cExtent    = SrcExtent
      ] (DxvkContext* ctx) {
        // CopyResource can only copy between different images, and
        // CopySubresourceRegion can only copy data from one single
        // subresource at a time, so this check is safe.
        if (cDstImage != cSrcImage || cDstLayers != cSrcLayers) {
          ctx->copyImage(
            cDstImage, cDstLayers, cDstOffset,
            cSrcImage, cSrcLayers, cSrcOffset,
            cExtent);
        } else {
          ctx->copyImageRegion(
            cDstImage, cDstLayers, cDstOffset,
            cSrcOffset, cExtent);
        }
      });
    } else {
      // Since each subresource uses a dedicated buffer, we are going
      // to need one call per subresource for staging resource copies
      for (uint32_t i = 0; i < pDstLayers->layerCount; i++) {
        uint32_t dstSubresource = D3D11CalcSubresource(pDstLayers->mipLevel, pDstLayers->baseArrayLayer + i, pDstTexture->Desc()->MipLevels);
        uint32_t srcSubresource = D3D11CalcSubresource(pSrcLayers->mipLevel, pSrcLayers->baseArrayLayer + i, pSrcTexture->Desc()->MipLevels);

        // For multi-plane image data stored in a buffer, the backend
        // assumes that the second plane immediately follows the first
        // plane in memory, which is only true if we copy the full image.
        uint32_t planeCount = 1;

        if (dstFormatInfo->flags.test(DxvkFormatFlag::MultiPlane)) {
          bool needsSeparateCopies = !dstIsImage && !srcIsImage;

          if (!dstIsImage)
            needsSeparateCopies |= pDstTexture->MipLevelExtent(pDstLayers->mipLevel) != SrcExtent;
          if (!srcIsImage)
            needsSeparateCopies |= pSrcTexture->MipLevelExtent(pSrcLayers->mipLevel) != SrcExtent;

          if (needsSeparateCopies)
            planeCount = vk::getPlaneCount(srcFormatInfo->aspectMask);
        }

        for (uint32_t j = 0; j < planeCount; j++) {
          VkImageAspectFlags dstAspectMask = dstFormatInfo->aspectMask;
          VkImageAspectFlags srcAspectMask = srcFormatInfo->aspectMask;

          if (planeCount > 1) {
            dstAspectMask = vk::getPlaneAspect(j);
            srcAspectMask = dstAspectMask;
          }

          if (dstIsImage) {
            VkImageSubresourceLayers dstLayer = { dstAspectMask,
              pDstLayers->mipLevel, pDstLayers->baseArrayLayer + i, 1 };

            EmitCs([
              cDstImage   = pDstTexture->GetImage(),
              cDstLayers  = dstLayer,
              cDstOffset  = DstOffset,
              cDstExtent  = dstExtent,
              cDstFormat  = pDstTexture->GetPackedFormat(),
              cSrcBuffer  = pSrcTexture->GetMappedBuffer(srcSubresource),
              cSrcLayout  = pSrcTexture->GetSubresourceLayout(srcAspectMask, srcSubresource),
              cSrcOffset  = pSrcTexture->ComputeMappedOffset(srcSubresource, j, SrcOffset),
              cSrcCoord   = SrcOffset,
              cSrcExtent  = srcMipExtent
            ] (DxvkContext* ctx) {
              ctx->copyBufferToImage(cDstImage, cDstLayers, cDstOffset, cDstExtent,
                cSrcBuffer, cSrcOffset, cSrcLayout.RowPitch, cSrcLayout.DepthPitch,
                cDstFormat);
            });
          } else if (srcIsImage) {
            VkImageSubresourceLayers srcLayer = { srcAspectMask,
              pSrcLayers->mipLevel, pSrcLayers->baseArrayLayer + i, 1 };

            EmitCs([
              cSrcImage   = pSrcTexture->GetImage(),
              cSrcFormat  = pSrcTexture->GetPackedFormat(),
              cSrcLayers  = srcLayer,
              cSrcOffset  = SrcOffset,
              cSrcExtent  = SrcExtent,
              cDstBuffer  = pDstTexture->GetMappedBuffer(dstSubresource),
              cDstLayout  = pDstTexture->GetSubresourceLayout(dstAspectMask, dstSubresource),
              cDstOffset  = pDstTexture->ComputeMappedOffset(dstSubresource, j, DstOffset),
              cDstCoord   = DstOffset,
              cDstExtent  = dstMipExtent
            ] (DxvkContext* ctx) {
              ctx->copyImageToBuffer(cDstBuffer, cDstOffset, cDstLayout.RowPitch,
                cDstLayout.DepthPitch, cSrcFormat, cSrcImage, cSrcLayers, cSrcOffset, cSrcExtent);
            });
          } else {
            // The backend is not aware of image metadata in this case,
            // so we need to handle image planes and block sizes here
            VkDeviceSize elementSize = dstFormatInfo->elementSize;
            VkExtent3D dstBlockSize = dstFormatInfo->blockSize;
            VkExtent3D srcBlockSize = srcFormatInfo->blockSize;
            VkExtent3D planeBlockSize = { 1u, 1u, 1u };

            if (planeCount > 1) {
              auto plane = &dstFormatInfo->planes[j];
              dstBlockSize.width  *= plane->blockSize.width;
              dstBlockSize.height *= plane->blockSize.height;
              srcBlockSize.width  *= plane->blockSize.width;
              srcBlockSize.height *= plane->blockSize.height;

              planeBlockSize.width  = plane->blockSize.width;
              planeBlockSize.height = plane->blockSize.height;
              elementSize = plane->elementSize;
            }

            EmitCs([
              cPixelSize = elementSize,
              cSrcBuffer = pSrcTexture->GetMappedBuffer(srcSubresource),
              cSrcStart  = pSrcTexture->GetSubresourceLayout(srcAspectMask, srcSubresource).Offset,
              cSrcOffset = util::computeBlockOffset(SrcOffset, srcBlockSize),
              cSrcSize   = util::computeBlockCount(srcMipExtent, srcBlockSize),
              cDstBuffer = pDstTexture->GetMappedBuffer(dstSubresource),
              cDstStart  = pDstTexture->GetSubresourceLayout(dstAspectMask, dstSubresource).Offset,
              cDstOffset = util::computeBlockOffset(DstOffset, dstBlockSize),
              cDstSize   = util::computeBlockCount(dstMipExtent, dstBlockSize),
              cExtent    = util::computeBlockCount(blockCount, planeBlockSize)
            ] (DxvkContext* ctx) {
              ctx->copyPackedBufferImage(
                cDstBuffer, cDstStart, cDstOffset, cDstSize,
                cSrcBuffer, cSrcStart, cSrcOffset, cSrcSize,
                cExtent, cPixelSize);
            });
          }
        }
      }
    }

    if (pDstTexture->HasSequenceNumber()) {
      for (uint32_t i = 0; i < pDstLayers->layerCount; i++) {
        GetTypedContext()->TrackTextureSequenceNumber(pDstTexture, D3D11CalcSubresource(
          pDstLayers->mipLevel, pDstLayers->baseArrayLayer + i, pDstTexture->Desc()->MipLevels));
      }
    }

    if (pSrcTexture->HasSequenceNumber()) {
      for (uint32_t i = 0; i < pSrcLayers->layerCount; i++) {
        GetTypedContext()->TrackTextureSequenceNumber(pSrcTexture, D3D11CalcSubresource(
          pSrcLayers->mipLevel, pSrcLayers->baseArrayLayer + i, pSrcTexture->Desc()->MipLevels));
      }
    }
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::CopyTiledResourceData(
          ID3D11Resource*                   pResource,
    const D3D11_TILED_RESOURCE_COORDINATE*  pRegionCoordinate,
    const D3D11_TILE_REGION_SIZE*           pRegionSize,
          DxvkBufferSlice                   BufferSlice,
          UINT                              Flags) {
    Rc<DxvkPagedResource> resource = GetPagedResource(pResource);

    // Do some validation based on page table properties
    auto pageTable = resource->getSparsePageTable();

    if (!pageTable)
      return;

    if (pRegionSize->bUseBox && pRegionSize->NumTiles !=
        pRegionSize->Width * pRegionSize->Height * pRegionSize->Depth)
      return;

    if (pRegionSize->NumTiles > pageTable->getPageCount())
      return;

    // Ignore call if buffer access would be out of bounds
    VkDeviceSize bufferSize = pRegionSize->NumTiles * SparseMemoryPageSize;

    if (BufferSlice.length() < bufferSize)
      return;

    // Compute list of tile indices to copy
    std::vector<uint32_t> tiles(pRegionSize->NumTiles);

    for (uint32_t i = 0; i < pRegionSize->NumTiles; i++) {
      VkOffset3D regionOffset = {
        int32_t(pRegionCoordinate->X),
        int32_t(pRegionCoordinate->Y),
        int32_t(pRegionCoordinate->Z) };

      VkExtent3D regionExtent = {
        uint32_t(pRegionSize->Width),
        uint32_t(pRegionSize->Height),
        uint32_t(pRegionSize->Depth) };

      uint32_t tile = pageTable->computePageIndex(
        pRegionCoordinate->Subresource, regionOffset,
        regionExtent, !pRegionSize->bUseBox, i);

      // Check that the tile is valid and not part of the mip tail
      auto tileInfo = pageTable->getPageInfo(tile);

      if (tileInfo.type != DxvkSparsePageType::Buffer
       && tileInfo.type != DxvkSparsePageType::Image)
        return;

      tiles[i] = tile;
    }

    AddCost(GpuCostEstimate::Transfer);

    // If D3D12 is anything to go by, not passing this flag will trigger
    // the other code path, regardless of whether TO_LINEAR_BUFFER is set.
    if (Flags & D3D11_TILE_COPY_LINEAR_BUFFER_TO_SWIZZLED_TILED_RESOURCE) {
      EmitCs([
        cResource = std::move(resource),
        cTiles    = std::move(tiles),
        cBuffer   = std::move(BufferSlice)
      ] (DxvkContext* ctx) {
        ctx->copySparsePagesFromBuffer(
          cResource,
          cTiles.size(),
          cTiles.data(),
          cBuffer.buffer(),
          cBuffer.offset());
      });
    } else {
      EmitCs([
        cResource = std::move(resource),
        cTiles    = std::move(tiles),
        cBuffer   = std::move(BufferSlice)
      ] (DxvkContext* ctx) {
        ctx->copySparsePagesToBuffer(
          cBuffer.buffer(),
          cBuffer.offset(),
          cResource,
          cTiles.size(),
          cTiles.data());
      });
    }
  }


  template<typename ContextType>
  template<typename T>
  bool D3D11CommonContext<ContextType>::DirtyBindingGeneric(
          D3D11ShaderType                   ShaderStage,
          T                                 BoundMask,
          T&                                DirtyMask,
          T                                 DirtyBit,
          bool                              IsNull) {
    // Forward immediately if lazy binding is forced off
    if (DebugLazyBinding == Tristate::False)
      return false;

    if ((BoundMask & ~DirtyMask) & DirtyBit) {
      // If we're binding a non-null resource to an active slot that has not been
      // marked for lazy binding yet, forward the call immediately in order to
      // avoid tracking overhead. This is by far the most common case.
      if (likely(!IsNull && DebugLazyBinding != Tristate::True))
        return false;

      // If we are binding a null resource to an active slot, the app will likely
      // either bind something else or bind a shader that does not use this slot.
      // In that case, avoid likely redundant CS traffic and apply the binding on
      // the next draw.
      m_state.lazy.shadersDirty.set(ShaderStage);
    }

    // Binding is either inactive or already dirty. In the inactive case, there
    // is no need to mark the shader stage as dirty since binding a shader that
    // activates the binding will implicitly do so.
    DirtyMask |= DirtyBit;
    return true;
  }


  template<typename ContextType>
  bool D3D11CommonContext<ContextType>::DirtyConstantBuffer(
          D3D11ShaderType                   ShaderStage,
          uint32_t                          Slot,
          bool                              IsNull) {
    return DirtyBindingGeneric(ShaderStage,
      m_state.lazy.bindingsUsed[ShaderStage].cbvMask,
      m_state.lazy.bindingsDirty[ShaderStage].cbvMask,
      1u << Slot, IsNull);
  }


  template<typename ContextType>
  bool D3D11CommonContext<ContextType>::DirtySampler(
          D3D11ShaderType                   ShaderStage,
          uint32_t                          Slot,
          bool                              IsNull) {
    return DirtyBindingGeneric(ShaderStage,
      m_state.lazy.bindingsUsed[ShaderStage].samplerMask,
      m_state.lazy.bindingsDirty[ShaderStage].samplerMask,
      1u << Slot, IsNull);
  }


  template<typename ContextType>
  bool D3D11CommonContext<ContextType>::DirtyShaderResource(
          D3D11ShaderType                   ShaderStage,
          uint32_t                          Slot,
          bool                              IsNull) {
    uint32_t idx = Slot / 64u;

    return DirtyBindingGeneric(ShaderStage,
      m_state.lazy.bindingsUsed[ShaderStage].srvMask[idx],
      m_state.lazy.bindingsDirty[ShaderStage].srvMask[idx],
      uint64_t(1u) << Slot, IsNull);
  }


  template<typename ContextType>
  bool D3D11CommonContext<ContextType>::DirtyComputeUnorderedAccessView(
          uint32_t                          Slot,
          bool                              IsNull) {
    constexpr D3D11ShaderType ShaderStage = D3D11ShaderType::eCompute;

    return DirtyBindingGeneric(ShaderStage,
      m_state.lazy.bindingsUsed[ShaderStage].uavMask,
      m_state.lazy.bindingsDirty[ShaderStage].uavMask,
      uint64_t(1u) << Slot, IsNull);
  }


  template<typename ContextType>
  bool D3D11CommonContext<ContextType>::DirtyGraphicsUnorderedAccessView(
          uint32_t                          Slot) {
    constexpr D3D11ShaderType ShaderStage = D3D11ShaderType::ePixel;

    if (DebugLazyBinding == Tristate::False)
      return false;

    // Use different logic here and always use lazy binding for graphics UAVs.
    // Since graphics UAVs are generally bound together with render targets,
    // looking at the active binding mask doesn't really help us here.
    uint64_t dirtyBit = uint64_t(1u) << Slot;

    if (m_state.lazy.bindingsUsed[ShaderStage].uavMask & dirtyBit) {
      // Need to mark all graphics stages that use UAVs as dirty here to
      // make sure that bindings actually get reapplied properly. There
      // may be no pixel shader bound in this case, even though we do
      // all the tracking on the pixel shader bit mask.
      m_state.lazy.shadersDirty.set(m_state.lazy.graphicsUavShaders);
    }

    m_state.lazy.bindingsDirty[ShaderStage].uavMask |= dirtyBit;
    return true;
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::DiscardBuffer(
          ID3D11Resource*                   pResource) {
    auto buffer = static_cast<D3D11Buffer*>(pResource);

    if (buffer->GetMapMode() != D3D11_COMMON_BUFFER_MAP_MODE_NONE) {
      D3D11_MAPPED_SUBRESOURCE sr;

      Map(pResource, 0, D3D11_MAP_WRITE_DISCARD, 0, &sr);
      Unmap(pResource, 0);
    }
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::DiscardTexture(
          ID3D11Resource*                   pResource,
          UINT                              Subresource) {
    auto texture = GetCommonTexture(pResource);

    if (texture->GetMapMode() != D3D11_COMMON_TEXTURE_MAP_MODE_NONE) {
      D3D11_MAPPED_SUBRESOURCE sr;

      Map(pResource, Subresource, D3D11_MAP_WRITE_DISCARD, 0, &sr);
      Unmap(pResource, Subresource);
    }
  }


  template<typename ContextType>
  template<D3D11ShaderType ShaderStage>
  void D3D11CommonContext<ContextType>::GetConstantBuffers(
          UINT                              StartSlot,
          UINT                              NumBuffers,
          ID3D11Buffer**                    ppConstantBuffers,
          UINT*                             pFirstConstant,
          UINT*                             pNumConstants) {
    const auto& bindings = m_state.cbv[ShaderStage];

    for (uint32_t i = 0; i < NumBuffers; i++) {
      const bool inRange = StartSlot + i < bindings.buffers.size();

      if (ppConstantBuffers) {
        ppConstantBuffers[i] = inRange
          ? bindings.buffers[StartSlot + i].buffer.ref()
          : nullptr;
      }

      if (pFirstConstant) {
        pFirstConstant[i] = inRange
          ? bindings.buffers[StartSlot + i].constantOffset
          : 0u;
      }

      if (pNumConstants) {
        pNumConstants[i] = inRange
          ? bindings.buffers[StartSlot + i].constantCount
          : 0u;
      }
    }
  }


  template<typename ContextType>
  template<D3D11ShaderType ShaderStage>
  void D3D11CommonContext<ContextType>::GetShaderResources(
          UINT                              StartSlot,
          UINT                              NumViews,
          ID3D11ShaderResourceView**        ppShaderResourceViews) {
    const auto& bindings = m_state.srv[ShaderStage];

    for (uint32_t i = 0; i < NumViews; i++) {
      ppShaderResourceViews[i] = StartSlot + i < bindings.views.size()
        ? bindings.views[StartSlot + i].ref()
        : nullptr;
    }
  }


  template<typename ContextType>
  template<D3D11ShaderType ShaderStage>
  void D3D11CommonContext<ContextType>::GetSamplers(
          UINT                              StartSlot,
          UINT                              NumSamplers,
          ID3D11SamplerState**              ppSamplers) {
    const auto& bindings = m_state.samplers[ShaderStage];

    for (uint32_t i = 0; i < NumSamplers; i++) {
      ppSamplers[i] = StartSlot + i < bindings.samplers.size()
        ? bindings.samplers[StartSlot + i].ref()
        : nullptr;
    }
  }


  template<typename ContextType>
  DxvkGlobalPipelineBarrier D3D11CommonContext<ContextType>::GetTiledResourceDependency(
          ID3D11DeviceChild*                pObject) {
    if (!pObject) {
      DxvkGlobalPipelineBarrier result;
      result.stages = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
      result.access = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
      return result;
    } else {
      Com<ID3D11Resource> resource;

      if (FAILED(pObject->QueryInterface(IID_PPV_ARGS(&resource)))) {
        Com<ID3D11View> view;

        if (FAILED(pObject->QueryInterface(IID_PPV_ARGS(&view))))
          return DxvkGlobalPipelineBarrier();

        view->GetResource(&resource);
      }

      D3D11CommonTexture* texture = GetCommonTexture(resource.ptr());

      if (texture) {
        Rc<DxvkImage> image = texture->GetImage();

        DxvkGlobalPipelineBarrier result;
        result.stages = image->info().stages;
        result.access = image->info().access;
        return result;
      } else {
        Rc<DxvkBuffer> buffer = static_cast<D3D11Buffer*>(resource.ptr())->GetBuffer();

        if (buffer == nullptr)
          return DxvkGlobalPipelineBarrier();

        DxvkGlobalPipelineBarrier result;
        result.stages = buffer->info().stages;
        result.access = buffer->info().access;
        return result;
      }
    }
  }


  template<typename ContextType>
  D3D11MaxUsedBindings D3D11CommonContext<ContextType>::GetMaxUsedBindings() {
    D3D11MaxUsedBindings result;

    for (uint32_t i = 0; i < result.stages.size(); i++) {
      auto stage = D3D11ShaderType(i);

      result.stages[i].cbvCount = m_state.cbv[stage].maxCount;
      result.stages[i].srvCount = m_state.srv[stage].maxCount;
      result.stages[i].uavCount = 0;
      result.stages[i].samplerCount = m_state.samplers[stage].maxCount;
      result.stages[i].reserved = 0;
    }

    result.stages[uint32_t(D3D11ShaderType::ePixel)].uavCount = m_state.om.maxUav;
    result.stages[uint32_t(D3D11ShaderType::eCompute)].uavCount = m_state.uav.maxCount;

    result.vbCount = m_state.ia.maxVbCount;
    result.soCount = D3D11_SO_BUFFER_SLOT_COUNT;
    return result;
  }


  template<typename ContextType>
  bool D3D11CommonContext<ContextType>::HasDirtyComputeBindings() {
    return m_state.lazy.shadersDirty.test(D3D11ShaderType::eCompute);
  }


  template<typename ContextType>
  bool D3D11CommonContext<ContextType>::HasDirtyGraphicsBindings() {
    return (m_state.lazy.shadersDirty & m_state.lazy.shadersUsed).any(
      D3D11ShaderType::eVertex, D3D11ShaderType::eGeometry,
      D3D11ShaderType::eHull,   D3D11ShaderType::eDomain,
      D3D11ShaderType::ePixel);
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::ResetCommandListState() {
    EmitCs([
      cUsedBindings  = GetMaxUsedBindings()
    ] (DxvkContext* ctx) {
      // Reset render targets
      ctx->bindRenderTargets(DxvkRenderTargets(), 0u);

      // Reset vertex input state
      ctx->setInputLayout(0, nullptr, 0, nullptr);

      // Reset render states
      ctx->setInputAssemblyState(InitDefaultPrimitiveTopology());
      ctx->setDepthStencilState(InitDefaultDepthStencilState());
      ctx->setRasterizerState(InitDefaultRasterizerState());
      ctx->setDepthBias(DxvkDepthBias());
      ctx->setLogicOpState(InitDefaultLogicOpState());
      ctx->setMultisampleState(InitDefaultMultisampleState(D3D11_DEFAULT_SAMPLE_MASK));

      DxvkBlendMode cbState = InitDefaultBlendState();

      for (uint32_t i = 0; i < D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT; i++)
        ctx->setBlendMode(i, cbState);

      // Reset dynamic states
      ctx->setBlendConstants(DxvkBlendConstants { 1.0f, 1.0f, 1.0f, 1.0f });
      ctx->setStencilReference(D3D11_DEFAULT_STENCIL_REFERENCE);

      // Reset viewports
      DxvkViewport viewport = { };
      ctx->setViewports(1, &viewport);

      // Unbind indirect draw buffer
      ctx->bindDrawBuffers(DxvkBufferSlice(), DxvkBufferSlice());

      // Unbind index and vertex buffers
      ctx->bindIndexBuffer(DxvkBufferSlice(), VK_INDEX_TYPE_UINT32);

      for (uint32_t i = 0; i < cUsedBindings.vbCount; i++)
        ctx->bindVertexBuffer(i, DxvkBufferSlice(), 0);

      // Unbind transform feedback buffers
      for (uint32_t i = 0; i < cUsedBindings.soCount; i++)
        ctx->bindXfbBuffer(i, DxvkBufferSlice(), DxvkBufferSlice());

      // Unbind all shaders
      ctx->bindShader<VK_SHADER_STAGE_VERTEX_BIT>(nullptr);
      ctx->bindShader<VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT>(nullptr);
      ctx->bindShader<VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT>(nullptr);
      ctx->bindShader<VK_SHADER_STAGE_GEOMETRY_BIT>(nullptr);
      ctx->bindShader<VK_SHADER_STAGE_FRAGMENT_BIT>(nullptr);
      ctx->bindShader<VK_SHADER_STAGE_COMPUTE_BIT>(nullptr);

      // Unbind per-shader stage resources
      for (uint32_t i = 0; i < 6; i++) {
        auto programType = D3D11ShaderType(i);
        auto stage = GetShaderStage(programType);

        // Unbind constant buffers, including the shader's ICB and instance data buffer
        auto cbSlotId = D3D11ShaderResourceMapping::computeCbvBinding(programType, 0);
        ctx->bindUniformBuffer(stage, cbSlotId + D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT + 0u, DxvkBufferSlice());
        ctx->bindUniformBuffer(stage, cbSlotId + D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT + 1u, DxvkBufferSlice());

        for (uint32_t j = 0; j < cUsedBindings.stages[i].cbvCount; j++)
          ctx->bindUniformBuffer(stage, cbSlotId + j, DxvkBufferSlice());

        // Unbind shader resource views
        auto srvSlotId = D3D11ShaderResourceMapping::computeSrvBinding(programType, 0);

        for (uint32_t j = 0; j < cUsedBindings.stages[i].srvCount; j++)
          ctx->bindResourceImageView(stage, srvSlotId + j, nullptr);

        // Unbind texture samplers
        auto samplerSlotId = D3D11ShaderResourceMapping::computeSamplerBinding(programType, 0);

        for (uint32_t j = 0; j < cUsedBindings.stages[i].samplerCount; j++)
          ctx->bindResourceSampler(stage, samplerSlotId + j, nullptr);

        // Unbind UAVs for supported stages
        if (programType == D3D11ShaderType::ePixel
         || programType == D3D11ShaderType::eCompute) {
          VkShaderStageFlags stages = programType == D3D11ShaderType::ePixel
            ? VK_SHADER_STAGE_ALL_GRAPHICS
            : VK_SHADER_STAGE_COMPUTE_BIT;

          auto uavSlotId = D3D11ShaderResourceMapping::computeUavBinding(programType, 0);
          auto ctrSlotId = D3D11ShaderResourceMapping::computeUavCounterBinding(programType, 0);

          for (uint32_t j = 0; j < cUsedBindings.stages[i].uavCount; j++) {
            ctx->bindResourceImageView(stages, uavSlotId, nullptr);
            ctx->bindResourceBufferView(stages, ctrSlotId, nullptr);
          }
        }
      }
    });
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::ResetContextState() {
    // Reset shaders
    m_state.vs = nullptr;
    m_state.hs = nullptr;
    m_state.ds = nullptr;
    m_state.gs = nullptr;
    m_state.ps = nullptr;
    m_state.cs = nullptr;

    // Reset render state
    m_state.id.reset();
    m_state.ia.reset();
    m_state.om.reset();
    m_state.rs.reset();
    m_state.so.reset();
    m_state.pr.reset();

    // Reset resource bindings
    m_state.cbv.reset();
    m_state.srv.reset();
    m_state.uav.reset();
    m_state.samplers.reset();

    // Reset dirty tracking
    m_state.lazy.reset();

    // Reset class instances
    m_state.instances.reset();
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::ResetDirtyTracking() {
    // Must only be called when all bindings are guaranteed to get applied
    // to the DXVK context before the next draw or dispatch command.
    m_state.lazy.bindingsDirty.reset();
    m_state.lazy.shadersDirty = 0u;
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::ResetStagingBuffer() {
    m_staging.reset();
  }


  template<typename ContextType>
  template<D3D11ShaderType ShaderStage, typename T>
  void D3D11CommonContext<ContextType>::ResolveSrvHazards(
          T*                                pView) {
    auto& bindings = m_state.srv[ShaderStage];
    int32_t srvId = bindings.hazardous.findNext(0);

    while (srvId >= 0) {
      auto srv = bindings.views[srvId].ptr();

      if (likely(srv && srv->TestHazards())) {
        bool hazard = CheckViewOverlap(pView, srv);

        if (unlikely(hazard)) {
          bindings.views[srvId] = nullptr;
          bindings.hazardous.clr(srvId);

          if (!DirtyShaderResource(ShaderStage, srvId, true))
            BindShaderResource(ShaderStage, srvId, nullptr);
        }
      } else {
        // Avoid further redundant iterations
        bindings.hazardous.clr(srvId);
      }

      srvId = bindings.hazardous.findNext(srvId + 1);
    }
  }


  template<typename ContextType>
  template<typename T>
  void D3D11CommonContext<ContextType>::ResolveCsSrvHazards(
          T*                                pView) {
    if (!pView) return;
    ResolveSrvHazards<D3D11ShaderType::eCompute>(pView);
  }


  template<typename ContextType>
  template<typename T>
  void D3D11CommonContext<ContextType>::ResolveOmSrvHazards(
          T*                                pView) {
    if (!pView) return;
    ResolveSrvHazards<D3D11ShaderType::eVertex>(pView);
    ResolveSrvHazards<D3D11ShaderType::eHull>(pView);
    ResolveSrvHazards<D3D11ShaderType::eDomain>(pView);
    ResolveSrvHazards<D3D11ShaderType::eGeometry>(pView);
    ResolveSrvHazards<D3D11ShaderType::ePixel>(pView);
  }


  template<typename ContextType>
  bool D3D11CommonContext<ContextType>::ResolveOmRtvHazards(
          D3D11UnorderedAccessView*         pView) {
    if (!pView || !pView->HasBindFlag(D3D11_BIND_RENDER_TARGET))
      return false;

    bool hazard = false;

    if (CheckViewOverlap(pView, m_state.om.dsv.ptr())) {
      m_state.om.dsv = nullptr;
      hazard = true;
    }

    for (uint32_t i = 0; i < m_state.om.maxRtv; i++) {
      if (CheckViewOverlap(pView, m_state.om.rtvs[i].ptr())) {
        m_state.om.rtvs[i] = nullptr;
        hazard = true;
      }
    }

    return hazard;
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::ResolveOmUavHazards(
          D3D11RenderTargetView*            pView) {
    if (!pView || !pView->HasBindFlag(D3D11_BIND_UNORDERED_ACCESS))
      return;

    for (uint32_t i = 0; i < m_state.om.maxUav; i++) {
      if (CheckViewOverlap(pView, m_state.om.uavs[i].ptr())) {
        m_state.om.uavs[i] = nullptr;

        if (!DirtyGraphicsUnorderedAccessView(i))
          BindUnorderedAccessView(D3D11ShaderType::ePixel, i, nullptr);
      }
    }
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::RestoreCommandListState() {
    BindFramebuffer();

    BindShader<D3D11ShaderType::eVertex>(GetCommonShader(m_state.vs.ptr()));
    BindShader<D3D11ShaderType::eHull>(GetCommonShader(m_state.hs.ptr()));
    BindShader<D3D11ShaderType::eDomain>(GetCommonShader(m_state.ds.ptr()));
    BindShader<D3D11ShaderType::eGeometry>(GetCommonShader(m_state.gs.ptr()));
    BindShader<D3D11ShaderType::ePixel>(GetCommonShader(m_state.ps.ptr()));
    BindShader<D3D11ShaderType::eCompute>(GetCommonShader(m_state.cs.ptr()));

    ApplyInputLayout();
    ApplyPrimitiveTopology();
    ApplyBlendState();
    ApplyBlendFactor();
    ApplyDepthStencilState();
    ApplyStencilRef();
    ApplyRasterizerState();
    ApplyRasterizerSampleCount();
    ApplyViewportState();

    BindIndexBuffer(
      m_state.ia.indexBuffer.buffer.ptr(),
      m_state.ia.indexBuffer.offset,
      m_state.ia.indexBuffer.format);

    for (uint32_t i = 0; i < m_state.ia.maxVbCount; i++) {
      BindVertexBuffer(i,
        m_state.ia.vertexBuffers[i].buffer.ptr(),
        m_state.ia.vertexBuffers[i].offset,
        m_state.ia.vertexBuffers[i].stride);
    }

    for (uint32_t i = 0; i < m_state.so.targets.size(); i++)
      BindXfbBuffer(i, m_state.so.targets[i].buffer.ptr(), ~0u);

    // Reset dirty binding and shader masks before applying
    // bindings to avoid implicit null binding overrids.
    ResetDirtyTracking();

    for (uint32_t i = 0; i < D3D11ShaderTypeCount; i++) {
      auto stage = D3D11ShaderType(i);

      RestoreConstantBuffers(stage);
      RestoreShaderResources(stage);
      RestoreSamplers(stage);
    }

    RestoreUnorderedAccessViews(D3D11ShaderType::ePixel);
    RestoreUnorderedAccessViews(D3D11ShaderType::eCompute);

    // Draw buffer bindings aren't persistent at the API level, and
    // we can't meaningfully track them. Just reset this state here
    // and reapply on the next indirect draw.
    SetDrawBuffers(nullptr, nullptr);
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::RestoreConstantBuffers(
          D3D11ShaderType                   Stage) {
    const auto& bindings = m_state.cbv[Stage];

    for (uint32_t i = 0; i < bindings.maxCount; i++) {
      BindConstantBuffer(Stage, i, bindings.buffers[i].buffer.ptr(),
        bindings.buffers[i].constantOffset, bindings.buffers[i].constantBound);
    }
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::RestoreSamplers(
          D3D11ShaderType                   Stage) {
    const auto& bindings = m_state.samplers[Stage];

    for (uint32_t i = 0; i < bindings.maxCount; i++)
      BindSampler(Stage, i, bindings.samplers[i].ptr());
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::RestoreShaderResources(
          D3D11ShaderType                   Stage) {
    const auto& bindings = m_state.srv[Stage];

    for (uint32_t i = 0; i < bindings.maxCount; i++)
      BindShaderResource(Stage, i, bindings.views[i].ptr());
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::RestoreUnorderedAccessViews(
          D3D11ShaderType                   Stage) {
    const auto& views = Stage == D3D11ShaderType::eCompute
      ? m_state.uav.views
      : m_state.om.uavs;

    uint32_t maxCount = Stage == D3D11ShaderType::eCompute
      ? m_state.uav.maxCount
      : m_state.om.maxUav;

    for (uint32_t i = 0; i < maxCount; i++)
      BindUnorderedAccessView(Stage, i, views[i].ptr());
  }


  template<typename ContextType>
  template<D3D11ShaderType ShaderStage>
  void D3D11CommonContext<ContextType>::SetConstantBuffers(
          UINT                              StartSlot,
          UINT                              NumBuffers,
          ID3D11Buffer* const*              ppConstantBuffers) {
    auto& bindings = m_state.cbv[ShaderStage];

    for (uint32_t i = 0; i < NumBuffers; i++) {
      auto newBuffer = static_cast<D3D11Buffer*>(ppConstantBuffers[i]);

      uint32_t constantCount = newBuffer
        ? std::min(newBuffer->Desc()->ByteWidth / 16, UINT(D3D11_REQ_CONSTANT_BUFFER_ELEMENT_COUNT))
        : 0u;

      if (bindings.buffers[StartSlot + i].buffer         != newBuffer
       || bindings.buffers[StartSlot + i].constantOffset != 0
       || bindings.buffers[StartSlot + i].constantCount  != constantCount) {
        bindings.buffers[StartSlot + i].buffer         = newBuffer;
        bindings.buffers[StartSlot + i].constantOffset = 0;
        bindings.buffers[StartSlot + i].constantCount  = constantCount;
        bindings.buffers[StartSlot + i].constantBound  = constantCount;

        if (!DirtyConstantBuffer(ShaderStage, StartSlot + i, !newBuffer))
          BindConstantBuffer(ShaderStage, StartSlot + i, newBuffer, 0, constantCount);
      }
    }

    bindings.maxCount = std::clamp(StartSlot + NumBuffers,
      bindings.maxCount, uint32_t(bindings.buffers.size()));
  }


  template<typename ContextType>
  template<D3D11ShaderType ShaderStage>
  void D3D11CommonContext<ContextType>::SetConstantBuffers1(
          UINT                              StartSlot,
          UINT                              NumBuffers,
          ID3D11Buffer* const*              ppConstantBuffers,
    const UINT*                             pFirstConstant,
    const UINT*                             pNumConstants) {
    auto& bindings = m_state.cbv[ShaderStage];

    for (uint32_t i = 0; i < NumBuffers; i++) {
      auto newBuffer = static_cast<D3D11Buffer*>(ppConstantBuffers[i]);

      UINT constantOffset;
      UINT constantCount;
      UINT constantBound;

      if (likely(newBuffer != nullptr)) {
        UINT bufferConstantsCount = newBuffer->Desc()->ByteWidth / 16;
        constantBound = std::min(bufferConstantsCount, UINT(D3D11_REQ_CONSTANT_BUFFER_ELEMENT_COUNT));

        if (likely(pFirstConstant && pNumConstants)) {
          constantOffset  = pFirstConstant[i];
          constantCount   = pNumConstants [i];

          if (unlikely(constantCount > D3D11_REQ_CONSTANT_BUFFER_ELEMENT_COUNT))
            continue;

          constantBound = (constantOffset + constantCount > bufferConstantsCount)
            ? bufferConstantsCount - std::min(constantOffset, bufferConstantsCount)
            : constantCount;
        } else {
          constantOffset  = 0;
          constantCount   = constantBound;
        }
      } else {
        constantOffset  = 0;
        constantCount   = 0;
        constantBound   = 0;
      }

      // Do a full rebind if either the buffer changes
      if (bindings.buffers[StartSlot + i].buffer != newBuffer) {
        bindings.buffers[StartSlot + i].buffer = newBuffer;
        bindings.buffers[StartSlot + i].constantOffset = constantOffset;
        bindings.buffers[StartSlot + i].constantCount  = constantCount;
        bindings.buffers[StartSlot + i].constantBound  = constantBound;

        if (!DirtyConstantBuffer(ShaderStage, StartSlot + i, !newBuffer))
          BindConstantBuffer(ShaderStage, StartSlot + i, newBuffer, constantOffset, constantBound);
      } else if (bindings.buffers[StartSlot + i].constantOffset != constantOffset
              || bindings.buffers[StartSlot + i].constantCount  != constantCount) {
        bindings.buffers[StartSlot + i].constantOffset = constantOffset;
        bindings.buffers[StartSlot + i].constantCount  = constantCount;
        bindings.buffers[StartSlot + i].constantBound  = constantBound;

        if (!DirtyConstantBuffer(ShaderStage, StartSlot + i, !newBuffer))
          BindConstantBufferRange(ShaderStage, StartSlot + i, constantOffset, constantBound);
      }
    }

    bindings.maxCount = std::clamp(StartSlot + NumBuffers,
      bindings.maxCount, uint32_t(bindings.buffers.size()));
  }


  template<typename ContextType>
  template<D3D11ShaderType ShaderStage>
  void D3D11CommonContext<ContextType>::SetShaderResources(
          UINT                              StartSlot,
          UINT                              NumResources,
          ID3D11ShaderResourceView* const*  ppResources) {
    auto& bindings = m_state.srv[ShaderStage];

    for (uint32_t i = 0; i < NumResources; i++) {
      auto resView = static_cast<D3D11ShaderResourceView*>(ppResources[i]);

      if (bindings.views[StartSlot + i] != resView) {
        if (likely(resView != nullptr)) {
          if (unlikely(resView->TestHazards())) {
            if (TestSrvHazards<ShaderStage>(resView))
              resView = nullptr;

            // blessed: zero-copy-present -- an SRV of the back buffer (it is
            // always a render target, so always hazard-tested); a bind
            // nulled by the hazard check above reads nothing
            if constexpr (!IsDeferred) {
              if (unlikely(BlessedZeroCopy::IsEnabled()) && resView != nullptr) {
                BlessedZeroCopy::OnResourceAccess(GetTypedContext(),
                  resView->GetViewInfo().pResource, "shader resource view");
              }
            }

            // Only set if necessary, but don't reset it on every
            // bind as this would be more expensive than a few
            // redundant checks in OMSetRenderTargets and friends.
            bindings.hazardous.set(StartSlot + i, resView);
          }
        }

        bindings.views[StartSlot + i] = resView;

        if (!DirtyShaderResource(ShaderStage, StartSlot + i, !resView))
          BindShaderResource(ShaderStage, StartSlot + i, resView);
      }
    }

    bindings.maxCount = std::clamp(StartSlot + NumResources,
      bindings.maxCount, uint32_t(bindings.views.size()));
  }


  template<typename ContextType>
  template<D3D11ShaderType ShaderStage>
  void D3D11CommonContext<ContextType>::SetSamplers(
          UINT                              StartSlot,
          UINT                              NumSamplers,
          ID3D11SamplerState* const*        ppSamplers) {
    auto& bindings = m_state.samplers[ShaderStage];

    for (uint32_t i = 0; i < NumSamplers; i++) {
      auto sampler = static_cast<D3D11SamplerState*>(ppSamplers[i]);

      if (bindings.samplers[StartSlot + i] != sampler) {
        bindings.samplers[StartSlot + i] = sampler;

        if (!DirtySampler(ShaderStage, StartSlot + i, !sampler))
          BindSampler(ShaderStage, StartSlot + i, sampler);
      }
    }

    bindings.maxCount = std::clamp(StartSlot + NumSamplers,
      bindings.maxCount, uint32_t(bindings.samplers.size()));
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::SetRenderTargetsAndUnorderedAccessViews(
          UINT                              NumRTVs,
          ID3D11RenderTargetView* const*    ppRenderTargetViews,
          ID3D11DepthStencilView*           pDepthStencilView,
          UINT                              UAVStartSlot,
          UINT                              NumUAVs,
          ID3D11UnorderedAccessView* const* ppUnorderedAccessViews,
    const UINT*                             pUAVInitialCounts) {
    if (TestRtvUavHazards(NumRTVs, ppRenderTargetViews, NumUAVs, ppUnorderedAccessViews))
      return;

    bool needsUpdate = false;
    bool isMultisampled = false;

    if (likely(NumRTVs != D3D11_KEEP_RENDER_TARGETS_AND_DEPTH_STENCIL)) {
      // Native D3D11 does not change the render targets if
      // the parameters passed to this method are invalid.
      if (!ValidateRenderTargets(NumRTVs, ppRenderTargetViews, pDepthStencilView))
        return;

      for (uint32_t i = 0; i < m_state.om.rtvs.size(); i++) {
        auto rtv = i < NumRTVs
          ? static_cast<D3D11RenderTargetView*>(ppRenderTargetViews[i])
          : nullptr;

        if (m_state.om.rtvs[i] != rtv) {
          m_state.om.rtvs[i] = rtv;
          needsUpdate = true;
          ResolveOmSrvHazards(rtv);

          if (NumUAVs == D3D11_KEEP_UNORDERED_ACCESS_VIEWS)
            ResolveOmUavHazards(rtv);
        }

        isMultisampled = isMultisampled || (rtv && rtv->GetSampleCount() > 1u);
      }

      auto dsv = static_cast<D3D11DepthStencilView*>(pDepthStencilView);

      if (m_state.om.dsv != dsv) {
        m_state.om.dsv = dsv;
        needsUpdate = true;
        ResolveOmSrvHazards(dsv);
      }

      m_state.om.maxRtv = NumRTVs;
    }

    if (unlikely(NumUAVs || m_state.om.maxUav)) {
      if (likely(NumUAVs != D3D11_KEEP_UNORDERED_ACCESS_VIEWS)) {
        uint32_t newMinUav = NumUAVs ? UAVStartSlot : D3D11_1_UAV_SLOT_COUNT;
        uint32_t newMaxUav = NumUAVs ? UAVStartSlot + NumUAVs : 0u;

        uint32_t oldMinUav = std::exchange(m_state.om.minUav, newMinUav);
        uint32_t oldMaxUav = std::exchange(m_state.om.maxUav, newMaxUav);

        for (uint32_t i = std::min(oldMinUav, newMinUav);
                      i < std::max(oldMaxUav, newMaxUav); i++) {
          D3D11UnorderedAccessView* uav = nullptr;
          uint32_t                  ctr = ~0u;

          if (i >= UAVStartSlot && i < UAVStartSlot + NumUAVs) {
            uav = static_cast<D3D11UnorderedAccessView*>(ppUnorderedAccessViews[i - UAVStartSlot]);
            ctr = pUAVInitialCounts ? pUAVInitialCounts[i - UAVStartSlot] : ~0u;
          }

          if (ctr != ~0u && uav && uav->HasCounter())
            UpdateUnorderedAccessViewCounter(uav, ctr);

          if (m_state.om.uavs[i] != uav) {
            m_state.om.uavs[i] = uav;

            if (!DirtyGraphicsUnorderedAccessView(i))
              BindUnorderedAccessView(D3D11ShaderType::ePixel, i, uav);

            ResolveOmSrvHazards(uav);

            if (NumRTVs == D3D11_KEEP_RENDER_TARGETS_AND_DEPTH_STENCIL)
              needsUpdate |= ResolveOmRtvHazards(uav);
          }
        }
      }
    }

    if (needsUpdate) {
      m_state.om.blessedOmGeneration++; // blessed: hook-cpu -- rtvs/dsv actually changed
      AddCost(GpuCostEstimate::RenderPass);
      BindFramebuffer();

      if constexpr (!IsDeferred)
        GetTypedContext()->NotifyRenderPassBoundary(isMultisampled);
    }
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::SetDrawBuffers(
          ID3D11Buffer*                     pBufferForArgs,
          ID3D11Buffer*                     pBufferForCount) {
    auto argBuffer = static_cast<D3D11Buffer*>(pBufferForArgs);
    auto cntBuffer = static_cast<D3D11Buffer*>(pBufferForCount);

    auto argBufferCookie = argBuffer ? argBuffer->GetCookie() : 0u;
    auto cntBufferCookie = cntBuffer ? cntBuffer->GetCookie() : 0u;

    if (m_state.id.argBufferCookie != argBufferCookie
     || m_state.id.cntBufferCookie != cntBufferCookie) {
      m_state.id.argBufferCookie = argBufferCookie;
      m_state.id.cntBufferCookie = cntBufferCookie;

      BindDrawBuffers(argBuffer, cntBuffer);
    }
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::SyncImage(
    const Rc<DxvkImage>&                    DstImage,
    const VkImageSubresourceLayers&         DstLayers,
    const Rc<DxvkImage>&                    SrcImage,
    const VkImageSubresourceLayers&         SrcLayers) {
    EmitCs([
      cDstImage = DstImage,
      cDstLayers = DstLayers,
      cSrcImage = SrcImage,
      cSrcLayers = SrcLayers
    ] (DxvkContext* ctx) {
      ctx->copyImage(
        cDstImage, cDstLayers, VkOffset3D(),
        cSrcImage, cSrcLayers, VkOffset3D(),
        cDstImage->mipLevelExtent(cDstLayers.mipLevel));
    });
  }


  template<typename ContextType>
  bool D3D11CommonContext<ContextType>::TestRtvUavHazards(
          UINT                              NumRTVs,
          ID3D11RenderTargetView* const*    ppRTVs,
          UINT                              NumUAVs,
          ID3D11UnorderedAccessView* const* ppUAVs) {
    if (NumRTVs == D3D11_KEEP_RENDER_TARGETS_AND_DEPTH_STENCIL) NumRTVs = 0;
    if (NumUAVs == D3D11_KEEP_UNORDERED_ACCESS_VIEWS)           NumUAVs = 0;

    for (uint32_t i = 0; i < NumRTVs; i++) {
      auto rtv = static_cast<D3D11RenderTargetView*>(ppRTVs[i]);

      if (!rtv)
        continue;

      for (uint32_t j = 0; j < i; j++) {
        if (CheckViewOverlap(rtv, static_cast<D3D11RenderTargetView*>(ppRTVs[j])))
          return true;
      }

      if (rtv->HasBindFlag(D3D11_BIND_UNORDERED_ACCESS)) {
        for (uint32_t j = 0; j < NumUAVs; j++) {
          if (CheckViewOverlap(rtv, static_cast<D3D11UnorderedAccessView*>(ppUAVs[j])))
            return true;
        }
      }
    }

    for (uint32_t i = 0; i < NumUAVs; i++) {
      auto uav = static_cast<D3D11UnorderedAccessView*>(ppUAVs[i]);

      if (!uav)
        continue;

      for (uint32_t j = 0; j < i; j++) {
        if (CheckViewOverlap(uav, static_cast<D3D11UnorderedAccessView*>(ppUAVs[j])))
          return true;
      }
    }

    return false;
  }


  template<typename ContextType>
  template<D3D11ShaderType ShaderStage>
  bool D3D11CommonContext<ContextType>::TestSrvHazards(
          D3D11ShaderResourceView*          pView) {
    bool hazard = false;

    if (ShaderStage == D3D11ShaderType::eCompute) {
      int32_t uav = m_state.uav.mask.findNext(0);

      while (uav >= 0 && !hazard) {
        hazard = CheckViewOverlap(pView, m_state.uav.views[uav].ptr());
        uav = m_state.uav.mask.findNext(uav + 1);
      }
    } else {
      hazard = CheckViewOverlap(pView, m_state.om.dsv.ptr());

      for (uint32_t i = 0; !hazard && i < m_state.om.maxRtv; i++)
        hazard = CheckViewOverlap(pView, m_state.om.rtvs[i].ptr());

      for (uint32_t i = 0; !hazard && i < m_state.om.maxUav; i++)
        hazard = CheckViewOverlap(pView, m_state.om.uavs[i].ptr());
    }

    return hazard;
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::TrackResourceSequenceNumber(
          ID3D11Resource*             pResource) {
    if (!pResource)
      return;

    D3D11CommonTexture* texture = GetCommonTexture(pResource);

    if (texture) {
      if (texture->HasSequenceNumber()) {
        for (uint32_t i = 0; i < texture->CountSubresources(); i++)
          GetTypedContext()->TrackTextureSequenceNumber(texture, i);
      }
    } else {
      D3D11Buffer* buffer = static_cast<D3D11Buffer*>(pResource);

      if (buffer->HasSequenceNumber())
        GetTypedContext()->TrackBufferSequenceNumber(buffer);
    }
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::UpdateBuffer(
          D3D11Buffer*                      pDstBuffer,
          UINT                              Offset,
          UINT                              Length,
    const void*                             pSrcData) {
    constexpr uint32_t MaxDirectUpdateSize = 64u;

    DxvkBufferSlice bufferSlice = pDstBuffer->GetBufferSlice(Offset, Length);

    if (Length <= MaxDirectUpdateSize && !((Offset | Length) & 0x3)) {
      // The backend has special code paths for small buffer updates,
      // however both offset and size must be aligned to four bytes.
      // Write the data directly to the CS chunk.
      uint32_t dwordCount = Length / sizeof(uint32_t);

      EmitCsCmd<uint32_t>(D3D11CmdType::None, dwordCount, [
        cBufferSlice = std::move(bufferSlice)
      ] (DxvkContext* ctx, const uint32_t* data, size_t) {
        ctx->updateBuffer(
          cBufferSlice.buffer(),
          cBufferSlice.offset(),
          cBufferSlice.length(), data);
      });

      // Compiler should be able to vectorize here, but GCC only does
      // if we cast the destination pointer to the correct type first
      auto src = reinterpret_cast<const uint32_t*>(pSrcData);
      auto dst = reinterpret_cast<uint32_t*>(m_csData->first());

      for (uint32_t i = 0; i < dwordCount; i++)
        new (dst + i) uint32_t(src[i]);
    } else {
      // Write directly to a staging buffer and dispatch a copy
      DxvkBufferSlice stagingSlice = AllocStagingBuffer(Length);
      std::memcpy(stagingSlice.mapPtr(0), pSrcData, Length);

      EmitCs([
        cStagingSlice = std::move(stagingSlice),
        cBufferSlice  = std::move(bufferSlice)
      ] (DxvkContext* ctx) {
        ctx->copyBuffer(
          cBufferSlice.buffer(),
          cBufferSlice.offset(),
          cStagingSlice.buffer(),
          cStagingSlice.offset(),
          cBufferSlice.length());
      });
    }

    if (pDstBuffer->HasSequenceNumber())
      GetTypedContext()->TrackBufferSequenceNumber(pDstBuffer);

    if constexpr (!IsDeferred)
      static_cast<ContextType*>(this)->ThrottleAllocation();
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::UpdateTexture(
          D3D11CommonTexture*               pDstTexture,
          UINT                              DstSubresource,
    const D3D11_BOX*                        pDstBox,
    const void*                             pSrcData,
          UINT                              SrcRowPitch,
          UINT                              SrcDepthPitch) {
    if (DstSubresource >= pDstTexture->CountSubresources())
      return;

    VkFormat packedFormat = pDstTexture->GetPackedFormat();

    auto formatInfo = lookupFormatInfo(packedFormat);
    auto subresource = pDstTexture->GetSubresourceFromIndex(
        formatInfo->aspectMask, DstSubresource);

    VkExtent3D mipExtent = pDstTexture->MipLevelExtent(subresource.mipLevel);

    VkOffset3D offset = { 0, 0, 0 };
    VkExtent3D extent = mipExtent;

    if (pDstBox != nullptr) {
      if (pDstBox->left >= pDstBox->right
        || pDstBox->top >= pDstBox->bottom
        || pDstBox->front >= pDstBox->back)
        return;  // no-op, but legal

      offset.x = pDstBox->left;
      offset.y = pDstBox->top;
      offset.z = pDstBox->front;

      extent.width  = pDstBox->right - pDstBox->left;
      extent.height = pDstBox->bottom - pDstBox->top;
      extent.depth  = pDstBox->back - pDstBox->front;
    }

    if (!util::isBlockAligned(offset, extent, formatInfo->blockSize, mipExtent))
      return;

    auto stagingSlice = AllocStagingBuffer(util::computeImageDataSize(packedFormat, extent));

    util::packImageData(stagingSlice.mapPtr(0),
      pSrcData, SrcRowPitch, SrcDepthPitch, 0, 0,
      pDstTexture->GetVkImageType(), extent, 1,
      formatInfo, formatInfo->aspectMask);

    UpdateImage(pDstTexture, &subresource,
      offset, extent, std::move(stagingSlice));

    if constexpr (!IsDeferred)
      static_cast<ContextType*>(this)->ThrottleAllocation();
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::UpdateImage(
          D3D11CommonTexture*               pDstTexture,
    const VkImageSubresource*               pDstSubresource,
          VkOffset3D                        DstOffset,
          VkExtent3D                        DstExtent,
          DxvkBufferSlice                   StagingBuffer) {
    bool dstIsImage = pDstTexture->HasImage();

    uint32_t dstSubresource = D3D11CalcSubresource(pDstSubresource->mipLevel,
      pDstSubresource->arrayLayer, pDstTexture->Desc()->MipLevels);

    if (dstIsImage) {
      EmitCs([
        cDstImage         = pDstTexture->GetImage(),
        cDstLayers        = vk::makeSubresourceLayers(*pDstSubresource),
        cDstOffset        = DstOffset,
        cDstExtent        = DstExtent,
        cStagingSlice     = std::move(StagingBuffer),
        cPackedFormat     = pDstTexture->GetPackedFormat()
      ] (DxvkContext* ctx) {
        ctx->copyBufferToImage(cDstImage,
          cDstLayers, cDstOffset, cDstExtent,
          cStagingSlice.buffer(),
          cStagingSlice.offset(), 0, 0,
          cPackedFormat);
      });
    } else {
      // If the destination image is backed only by a buffer, we need to use
      // the packed buffer copy function which does not know about planes and
      // format metadata, so deal with it manually here.
      VkExtent3D dstMipExtent = pDstTexture->MipLevelExtent(pDstSubresource->mipLevel);

      auto dstFormat = pDstTexture->GetPackedFormat();
      auto dstFormatInfo = lookupFormatInfo(dstFormat);

      uint32_t planeCount = 1;

      if (dstFormatInfo->flags.test(DxvkFormatFlag::MultiPlane))
        planeCount = vk::getPlaneCount(dstFormatInfo->aspectMask);

      // The source data isn't stored in an image so we'll also need to
      // track the offset for that while iterating over the planes.
      VkDeviceSize srcPlaneOffset = 0;

      for (uint32_t i = 0; i < planeCount; i++) {
        VkImageAspectFlags dstAspectMask = dstFormatInfo->aspectMask;
        VkDeviceSize elementSize = dstFormatInfo->elementSize;
        VkExtent3D blockSize = dstFormatInfo->blockSize;

        if (dstFormatInfo->flags.test(DxvkFormatFlag::MultiPlane)) {
          dstAspectMask = vk::getPlaneAspect(i);

          auto plane = &dstFormatInfo->planes[i];
          blockSize.width  *= plane->blockSize.width;
          blockSize.height *= plane->blockSize.height;
          elementSize = plane->elementSize;
        }

        VkExtent3D blockCount = util::computeBlockCount(DstExtent, blockSize);

        EmitCs([
          cDstBuffer      = pDstTexture->GetMappedBuffer(dstSubresource),
          cDstStart       = pDstTexture->GetSubresourceLayout(dstAspectMask, dstSubresource).Offset,
          cDstOffset      = util::computeBlockOffset(DstOffset, blockSize),
          cDstSize        = util::computeBlockCount(dstMipExtent, blockSize),
          cDstExtent      = blockCount,
          cSrcBuffer      = StagingBuffer.buffer(),
          cSrcStart       = StagingBuffer.offset() + srcPlaneOffset,
          cPixelSize      = elementSize
        ] (DxvkContext* ctx) {
          ctx->copyPackedBufferImage(
            cDstBuffer, cDstStart, cDstOffset, cDstSize,
            cSrcBuffer, cSrcStart, VkOffset3D(), cDstExtent,
            cDstExtent, cPixelSize);
        });

        srcPlaneOffset += util::flattenImageExtent(blockCount) * elementSize;
      }
    }

    if (pDstTexture->HasSequenceNumber())
      GetTypedContext()->TrackTextureSequenceNumber(pDstTexture, dstSubresource);
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::UpdateResource(
          ID3D11Resource*                   pDstResource,
          UINT                              DstSubresource,
    const D3D11_BOX*                        pDstBox,
    const void*                             pSrcData,
          UINT                              SrcRowPitch,
          UINT                              SrcDepthPitch,
          UINT                              CopyFlags) {
    auto context = static_cast<ContextType*>(this);
    D3D10DeviceLock lock = context->LockContext();

    if (!pDstResource)
      return;

    // blessed: probe dump hook, immediate context only
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedDump::IsCapturing()))
        BlessedDump::RecordUpdateSubresource(pDstResource, DstSubresource);
    }

    // We need a different code path for buffers
    D3D11_RESOURCE_DIMENSION resourceType;
    pDstResource->GetType(&resourceType);

    // blessed: zero-copy-present -- an update of the back buffer
    if constexpr (!IsDeferred) {
      if (unlikely(BlessedZeroCopy::IsEnabled()) && resourceType != D3D11_RESOURCE_DIMENSION_BUFFER)
        BlessedZeroCopy::OnResourceAccess(GetTypedContext(), pDstResource, "update subresource");
    }

    if (likely(resourceType == D3D11_RESOURCE_DIMENSION_BUFFER)) {
      const auto bufferResource = static_cast<D3D11Buffer*>(pDstResource);
      uint64_t bufferSize = bufferResource->Desc()->ByteWidth;

      // Provide a fast path for mapped buffer updates since some
      // games use UpdateSubresource to update constant buffers.
      if (likely(bufferResource->GetMapMode() == D3D11_COMMON_BUFFER_MAP_MODE_DIRECT) && likely(!pDstBox)) {
        context->UpdateMappedBuffer(bufferResource, 0, bufferSize, pSrcData, 0);
        return;
      }

      // Validate buffer range to update
      uint64_t offset = 0;
      uint64_t length = bufferSize;

      if (pDstBox) {
        offset = pDstBox->left;
        length = pDstBox->right - offset;
      }

      if (unlikely(offset + length > bufferSize))
        return;

      // Still try to be fast if a box is provided but we update the full buffer
      if (likely(bufferResource->GetMapMode() == D3D11_COMMON_BUFFER_MAP_MODE_DIRECT)) {
        CopyFlags &= D3D11_COPY_DISCARD | D3D11_COPY_NO_OVERWRITE;

        if (likely(length == bufferSize) || unlikely(CopyFlags != 0)) {
          context->UpdateMappedBuffer(bufferResource, offset, length, pSrcData, CopyFlags);
          return;
        }
      }

      // Otherwise we can't really do anything fancy, so just do a GPU copy
      if (likely(length))
        context->UpdateBuffer(bufferResource, offset, length, pSrcData);
    } else {
      D3D11CommonTexture* textureResource = GetCommonTexture(pDstResource);

      context->UpdateTexture(textureResource,
        DstSubresource, pDstBox, pSrcData, SrcRowPitch, SrcDepthPitch);
    }
  }


  template<typename ContextType>
  void D3D11CommonContext<ContextType>::UpdateUnorderedAccessViewCounter(
          D3D11UnorderedAccessView*         pUav,
          uint32_t                          CounterValue) {
    EmitCs([
      cView    = pUav->GetCounterView(),
      cCounter = CounterValue
    ] (DxvkContext* ctx) {
      ctx->updateBuffer(cView->buffer(),
        cView->info().offset, sizeof(cCounter), &cCounter);
    });
  }


  template<typename ContextType>
  bool D3D11CommonContext<ContextType>::ValidateRenderTargets(
          UINT                              NumViews,
          ID3D11RenderTargetView* const*    ppRenderTargetViews,
          ID3D11DepthStencilView*           pDepthStencilView) {
    Rc<DxvkImageView> refView;

    VkExtent3D dsvExtent = { 0u, 0u, 0u };
    VkExtent3D rtvExtent = { 0u, 0u, 0u };

    if (pDepthStencilView != nullptr) {
      refView = static_cast<D3D11DepthStencilView*>(
        pDepthStencilView)->GetImageView();
      dsvExtent = refView->mipLevelExtent(0);
    }

    for (uint32_t i = 0; i < NumViews; i++) {
      if (ppRenderTargetViews[i] != nullptr) {
        auto curView = static_cast<D3D11RenderTargetView*>(
          ppRenderTargetViews[i])->GetImageView();

        if (!rtvExtent.width)
          rtvExtent = curView->mipLevelExtent(0);

        if (refView != nullptr) {
          // Render target views must all have the same sample count,
          // layer count, and type. The size can mismatch under certain
          // conditions, the D3D11 documentation is wrong here.
          if (curView->info().viewType != refView->info().viewType
           || curView->info().layerCount != refView->info().layerCount)
            return false;

          if (curView->image()->info().sampleCount
           != refView->image()->info().sampleCount)
            return false;

          // Color targets must all be the same size
          VkExtent3D curExtent = curView->mipLevelExtent(0);

          if (curExtent.width  != rtvExtent.width
           || curExtent.height != rtvExtent.height)
            return false;
        } else {
          // Set reference view. All remaining views
          // must be compatible to the reference view.
          refView = curView;
        }
      }
    }

    // Based on testing, the depth-stencil target is allowed
    // to be larger than all color targets, but not smaller
    if (rtvExtent.width && dsvExtent.width) {
      if (rtvExtent.width  > dsvExtent.width
       || rtvExtent.height > dsvExtent.height)
        return false;
    }

    return true;
  }


  template<typename ContextType>
  template<D3D11ShaderType ShaderStage>
  void D3D11CommonContext<ContextType>::SetClassInstances(
    const D3D11CommonShader*                pShader,
          ID3D11ClassInstance* const*       ppClassInstances,
          UINT                              NumClassInstances) {
    auto& state = m_state.instances[ShaderStage];

    if (likely(!NumClassInstances) && likely(!state.instanceCount))
      return;

    // Not sure if this is the right thing to do, but if we don't
    // have a shader we also don't really have anything to bind
    if (!pShader)
      NumClassInstances = 0u;

    // Assign class instances for state tracking and populate
    // constant buffer containing class instance data */
    for (uint32_t i = 0u; i < NumClassInstances && i < D3D11ClassInstanceState::MaxInstances; i++)
      state.instances[i] = static_cast<D3D11ClassInstance*>(ppClassInstances[i]);

    // Unset previously bound class instances, if any
    for (uint32_t i = NumClassInstances; i < state.instanceCount; i++)
      state.instances[i] = nullptr;

    uint32_t slotId = D3D11ShaderResourceMapping::computeCbvBinding(ShaderStage,
      D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT + 1u);

    if (NumClassInstances) {
      auto buffer = AllocInstanceDataBuffer(ShaderStage);
      auto slice = buffer->allocateStorage();

      auto data = reinterpret_cast<D3D11InstanceData*>(slice->mapPtr());

      for (uint32_t i = 0u; i < D3D11ClassInstanceState::MaxInstances; i++)
        data[i] = pShader->GetClassInstanceData(i, state.instances[i].ptr());

      EmitCs([
        cSlotId = slotId,
        cStage  = GetShaderStage(ShaderStage),
        cBuffer = std::move(buffer),
        cSlice  = std::move(slice)
      ] (DxvkContext* ctx) mutable {
        ctx->invalidateBuffer(cBuffer, Forwarder::move(cSlice));
        ctx->bindUniformBuffer(cStage, cSlotId, DxvkBufferSlice(Forwarder::move(cBuffer)));
      });
    } else {
      EmitCs([
        cSlotId = slotId,
        cStage  = GetShaderStage(ShaderStage)
      ] (DxvkContext* ctx) mutable {
        ctx->bindUniformBuffer(cStage, cSlotId, DxvkBufferSlice());
      });
    }
  }


  template<typename ContextType>
  template<D3D11ShaderType ShaderStage>
  void D3D11CommonContext<ContextType>::GetClassInstances(
          ID3D11ClassInstance**             ppClassInstances,
          UINT*                             pNumClassInstances) {
    if (unlikely(pNumClassInstances)) {
      const auto& state = m_state.instances[ShaderStage];

      if (ppClassInstances) {
        for (uint32_t i = 0u; i < *pNumClassInstances; i++) {
          ppClassInstances[i] = i < state.instanceCount
            ? state.instances[i].ref()
            : nullptr;
        }
      }

      *pNumClassInstances = state.instanceCount;
    }
  }


  template<typename ContextType>
  Rc<DxvkBuffer> D3D11CommonContext<ContextType>::AllocInstanceDataBuffer(
          D3D11ShaderType                   ShaderStage) {
    if (!m_instanceData[ShaderStage]) {
      DxvkBufferCreateInfo bufferInfo = { };
      bufferInfo.size = sizeof(D3D11InstanceData) * D3D11ClassInstanceState::MaxInstances;
      bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
      bufferInfo.stages = GetShaderStage(ShaderStage);
      bufferInfo.access = VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
      bufferInfo.debugName = "Instance data";

      m_instanceData[ShaderStage] = m_device->createBuffer(bufferInfo,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }

    return m_instanceData[ShaderStage];
  }


  template<typename ContextType>
  DxvkInputAssemblyState D3D11CommonContext<ContextType>::InitDefaultPrimitiveTopology() {
    return DxvkInputAssemblyState(VK_PRIMITIVE_TOPOLOGY_MAX_ENUM, false);
  }


  template<typename ContextType>
  DxvkRasterizerState D3D11CommonContext<ContextType>::InitDefaultRasterizerState() {
    DxvkRasterizerState rsState = { };
    rsState.setPolygonMode(VK_POLYGON_MODE_FILL);
    rsState.setCullMode(VK_CULL_MODE_BACK_BIT);
    rsState.setFrontFace(VK_FRONT_FACE_CLOCKWISE);
    rsState.setDepthClip(true);
    rsState.setConservativeMode(VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT);
    rsState.setSampleCount(0);
    rsState.setFlatShading(false);
    rsState.setLineMode(VK_LINE_RASTERIZATION_MODE_DEFAULT_EXT);
    return rsState;
  }


  template<typename ContextType>
  DxvkDepthStencilState D3D11CommonContext<ContextType>::InitDefaultDepthStencilState() {
    DxvkDepthStencilState dsState = { };
    dsState.setDepthTest(true);
    dsState.setDepthWrite(true);
    dsState.setDepthCompareOp(VK_COMPARE_OP_LESS);
    return dsState;
  }


  template<typename ContextType>
  DxvkMultisampleState D3D11CommonContext<ContextType>::InitDefaultMultisampleState(
          UINT                              SampleMask) {
    DxvkMultisampleState msState = { };
    msState.setSampleMask(SampleMask);
    return msState;
  }


  template<typename ContextType>
  DxvkLogicOpState D3D11CommonContext<ContextType>::InitDefaultLogicOpState() {
    DxvkLogicOpState loState = { };
    loState.setLogicOp(false, VK_LOGIC_OP_NO_OP);
    return loState;
  }


  template<typename ContextType>
  DxvkBlendMode D3D11CommonContext<ContextType>::InitDefaultBlendState() {
    DxvkBlendMode cbState = { };
    cbState.setBlendEnable(false);
    cbState.setColorOp(VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD);
    cbState.setAlphaOp(VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD);
    cbState.setWriteMask(VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                       | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT);
    return cbState;
  }

  // Explicitly instantiate here
  template class D3D11CommonContext<D3D11DeferredContext>;
  template class D3D11CommonContext<D3D11ImmediateContext>;

}
