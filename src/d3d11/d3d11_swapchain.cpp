#include "blessed_autoinstance.h" // blessed: auto-instancing census
#include "blessed_cascades.h" // blessed: sun shadow cascade learner + skip
#include "blessed_cascade_cache.h" // blessed: cascade-cache
#include "blessed_dump.h" // blessed: frame draw-classification dumper
#include "blessed_gi.h" // blessed: ray-traced gi ambient patch
#include "blessed_gpu_passes.h" // blessed: gpu-pass-timing
#include "blessed_hook.h" // blessed: generic post-draw pixel-shader hook
#include "blessed_point_shadow.h" // blessed: point-lights
#include "blessed_vanilla_halfrate.h" // blessed: perf-halfrate
#include "blessed_crash_log.h" // blessed: crash-log
#include "blessed_skip_replaced.h" // blessed: gpu track step 2 -- skip vanilla work our features replace
#include "blessed_volumetrics.h" // blessed: async-compute, the vol kick frame boundary
#include "blessed_look.h" // blessed: look-post
#include "blessed_halfrate.h" // blessed: half-rate far field
#include "blessed_reflect_halfrate.h" // blessed: refl-harden
#include "blessed_midframe_split.h" // blessed: between-passes, BLESSED_MIDFRAME_SPLIT
#include "../dxvk/blessed/blessed_present.h" // blessed: present-idle, BLESSED_PRESENT=bridge
#include "../dxvk/blessed/blessed_fse.h" // blessed: present-fse-appcontrolled, BLESSED_FSE=app
#include "blessed_scene_capture.h" // blessed: scene-capture config + BLESSED_SCENE_LOG
#include "blessed_threaded_context.h" // blessed: threaded-fe
#include "blessed_zero_copy.h" // blessed: zero-copy-present
#include "d3d11_context_imm.h"
#include "d3d11_device.h"
#include "d3d11_swapchain.h"

#include "../util/util_likely.h"

#include "../dxvk/dxvk_latency_builtin.h"
#include "../dxvk/dxvk_shader_spirv.h"

// blessed: rt selftest, fired once on the first present -- see blessed_rt.h
#include "../dxvk/blessed/blessed_rt.h"
#include "../dxvk/blessed/blessed_scene.h" // blessed: hook-cpu-2, BlessedSceneSkinBatch

#include "../util/util_win32_compat.h"

// blessed: render-thread timing probe, see src/util/util_blessed_probe.h
#include "../util/util_blessed_probe.h"

#include <d3d11_composition_vert.h>
#include <d3d11_composition_frag.h>

namespace dxvk {

  static uint16_t MapGammaControlPoint(float x) {
    if (x < 0.0f) x = 0.0f;
    if (x > 1.0f) x = 1.0f;
    return uint16_t(65535.0f * x);
  }

  static VkColorSpaceKHR ConvertColorSpace(DXGI_COLOR_SPACE_TYPE colorspace) {
    switch (colorspace) {
      case DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709:    return VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
      case DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020: return VK_COLOR_SPACE_HDR10_ST2084_EXT;
      case DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709:    return VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT;
      default:
        Logger::warn(str::format("DXGI: ConvertColorSpace: Unknown colorspace ", colorspace));
        return VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    }
  }

  static VkXYColorEXT ConvertXYColor(const UINT16 (&dxgiColor)[2]) {
    return VkXYColorEXT{ float(dxgiColor[0]) / 50000.0f, float(dxgiColor[1]) / 50000.0f };
  }

  static float ConvertMaxLuminance(UINT dxgiLuminance) {
    return float(dxgiLuminance);
  }

  static float ConvertMinLuminance(UINT dxgiLuminance) {
    return float(dxgiLuminance) * 0.0001f;
  }

  static float ConvertLevel(UINT16 dxgiLevel) {
    return float(dxgiLevel);
  }

  static VkHdrMetadataEXT ConvertHDRMetadata(const DXGI_HDR_METADATA_HDR10& dxgiMetadata) {
    VkHdrMetadataEXT vkMetadata = { VK_STRUCTURE_TYPE_HDR_METADATA_EXT };
    vkMetadata.displayPrimaryRed         = ConvertXYColor(dxgiMetadata.RedPrimary);
    vkMetadata.displayPrimaryGreen       = ConvertXYColor(dxgiMetadata.GreenPrimary);
    vkMetadata.displayPrimaryBlue        = ConvertXYColor(dxgiMetadata.BluePrimary);
    vkMetadata.whitePoint                = ConvertXYColor(dxgiMetadata.WhitePoint);
    vkMetadata.maxLuminance              = ConvertMaxLuminance(dxgiMetadata.MaxMasteringLuminance);
    vkMetadata.minLuminance              = ConvertMinLuminance(dxgiMetadata.MinMasteringLuminance);
    vkMetadata.maxContentLightLevel      = ConvertLevel(dxgiMetadata.MaxContentLightLevel);
    vkMetadata.maxFrameAverageLightLevel = ConvertLevel(dxgiMetadata.MaxFrameAverageLightLevel);
    return vkMetadata;
  }


  D3D11SwapChain::D3D11SwapChain(
          D3D11DXGIDevice*        pContainer,
          D3D11Device*            pDevice,
          IDXGIVkSurfaceFactory*  pSurfaceFactory,
    const DXGI_SWAP_CHAIN_DESC1*  pDesc)
  : m_dxgiDevice(pContainer),
    m_parent(pDevice),
    m_surfaceFactory(pSurfaceFactory),
    m_desc(*pDesc),
    m_device(pDevice->GetDXVKDevice()),
    m_frameLatencyCap(pDevice->GetOptions()->maxFrameLatency) {
    CreateFrameLatencyEvent();
    CreatePresenter();
    CreateBackBuffers();
    CreateBlitter();
  }


  D3D11SwapChain::~D3D11SwapChain() {
    // Avoids hanging when in this state, see comment
    // in DxvkDevice::~DxvkDevice.
    if (this_thread::isInModuleDetachment())
      return;

    // blessed: threaded-fe-2 -- no recorded Present may outlive us
    m_parent->BlessedDrainFrontEnd(blessed::FeDrain::Device);

    // blessed: zero-copy-present -- an early acquire must reach a present,
    // or destroyResources waits for it forever
    BlessedReleaseEarlyAcquire();
    BlessedZeroCopy::OnSwapChainDestroyed(this);

    m_presenter->destroyResources();
    
    DestroyFrameLatencyEvent();
    DestroyLatencyTracker();
  }


  HRESULT STDMETHODCALLTYPE D3D11SwapChain::QueryInterface(
          REFIID                  riid,
          void**                  ppvObject) {
    if (ppvObject == nullptr)
      return E_POINTER;

    InitReturnPtr(ppvObject);

    if (riid == __uuidof(IUnknown)
     || riid == __uuidof(IDXGIVkSwapChain)
     || riid == __uuidof(IDXGIVkSwapChain1)
     || riid == __uuidof(IDXGIVkSwapChain2)
     || riid == __uuidof(IDXGIVkSwapChain3)) {
      *ppvObject = ref(this);
      return S_OK;
    }

    if (logQueryInterfaceError(__uuidof(IDXGIVkSwapChain), riid)) {
      Logger::warn("D3D11SwapChain::QueryInterface: Unknown interface query");
      Logger::warn(str::format(riid));
    }

    return E_NOINTERFACE;
  }


  HRESULT STDMETHODCALLTYPE D3D11SwapChain::GetDesc(
          DXGI_SWAP_CHAIN_DESC1*    pDesc) {
    *pDesc = m_desc;
    return S_OK;
  }


  HRESULT STDMETHODCALLTYPE D3D11SwapChain::GetAdapter(
          REFIID                    riid,
          void**                    ppvObject) {
    return m_dxgiDevice->GetParent(riid, ppvObject);
  }


  HRESULT STDMETHODCALLTYPE D3D11SwapChain::GetDevice(
          REFIID                    riid,
          void**                    ppDevice) {
    return m_dxgiDevice->QueryInterface(riid, ppDevice);
  }


  HRESULT STDMETHODCALLTYPE D3D11SwapChain::GetImage(
          UINT                      BufferId,
          REFIID                    riid,
          void**                    ppBuffer) {
    InitReturnPtr(ppBuffer);

    if (BufferId >= m_backBuffers.size()) {
      Logger::err("D3D11: GetImage: Invalid buffer ID");
      return DXGI_ERROR_UNSUPPORTED;
    }

    return m_backBuffers[BufferId]->QueryInterface(riid, ppBuffer);
  }


  UINT STDMETHODCALLTYPE D3D11SwapChain::GetImageIndex() {
    return 0;
  }


  UINT STDMETHODCALLTYPE D3D11SwapChain::GetFrameLatency() {
    return m_frameLatency;
  }


  HANDLE STDMETHODCALLTYPE D3D11SwapChain::GetFrameLatencyEvent() {
    HANDLE result = nullptr;
    HANDLE processHandle = GetCurrentProcess();

    if (!DuplicateHandle(processHandle, m_frameLatencyEvent,
        processHandle, &result, 0, FALSE, DUPLICATE_SAME_ACCESS)) {
      Logger::err("DxgiSwapChain::GetFrameLatencyWaitableObject: DuplicateHandle failed");
      return nullptr;
    }

    return result;
  }


  HRESULT STDMETHODCALLTYPE D3D11SwapChain::ChangeProperties(
    const DXGI_SWAP_CHAIN_DESC1*    pDesc,
    const UINT*                     pNodeMasks,
          IUnknown* const*          ppPresentQueues) {
    m_parent->BlessedDrainFrontEnd(blessed::FeDrain::Device); // blessed: threaded-fe

    if (m_desc.Format != pDesc->Format)
      m_presenter->setSurfaceFormat(GetSurfaceFormat(pDesc->Format));

    if (m_desc.Width != pDesc->Width || m_desc.Height != pDesc->Height)
      m_presenter->setSurfaceExtent({ m_desc.Width, m_desc.Height });

    m_desc = *pDesc;
    CreateBackBuffers();
    return S_OK;
  }


  HRESULT STDMETHODCALLTYPE D3D11SwapChain::SetPresentRegion(
    const RECT*                     pRegion) {
    Logger::err("D3D11SwapChain::SetPresentRegion: Stub");
    return E_NOTIMPL;
  }


  HRESULT STDMETHODCALLTYPE D3D11SwapChain::SetGammaControl(
          UINT                      NumControlPoints,
    const DXGI_RGB*                 pControlPoints) {
    m_parent->BlessedDrainFrontEnd(blessed::FeDrain::Device); // blessed: threaded-fe-2
    bool isIdentity = true;

    if (NumControlPoints > 1) {
      std::array<DxvkGammaCp, 1025> cp;

      if (NumControlPoints > cp.size())
        return E_INVALIDARG;
      
      for (uint32_t i = 0; i < NumControlPoints; i++) {
        uint16_t identity = MapGammaControlPoint(float(i) / float(NumControlPoints - 1));

        cp[i].r = MapGammaControlPoint(pControlPoints[i].Red);
        cp[i].g = MapGammaControlPoint(pControlPoints[i].Green);
        cp[i].b = MapGammaControlPoint(pControlPoints[i].Blue);
        cp[i].a = 0;

        isIdentity &= cp[i].r == identity
                   && cp[i].g == identity
                   && cp[i].b == identity;
      }

      if (!isIdentity)
        m_blitter->setGammaRamp(NumControlPoints, cp.data());
    }

    if (isIdentity)
      m_blitter->setGammaRamp(0, nullptr);

    m_hasGammaRamp = !isIdentity; // blessed: zero-copy-present

    return S_OK;
  }


  HRESULT STDMETHODCALLTYPE D3D11SwapChain::SetFrameLatency(
          UINT                      MaxLatency) {
    m_parent->BlessedDrainFrontEnd(blessed::FeDrain::Device); // blessed: threaded-fe-2
    if (MaxLatency == 0 || MaxLatency > DXGI_MAX_SWAP_CHAIN_BUFFERS)
      return DXGI_ERROR_INVALID_CALL;

    if (m_frameLatencyEvent) {
      // Windows DXGI does not seem to handle the case where the new maximum
      // latency is less than the current value, and some games relying on
      // this behaviour will hang if we attempt to decrement the semaphore.
      // Thus, only increment the semaphore as necessary.
      if (MaxLatency > m_frameLatency)
        ReleaseSemaphore(m_frameLatencyEvent, MaxLatency - m_frameLatency, nullptr);
    }

    m_frameLatency = MaxLatency;
    return S_OK;
  }


  HRESULT STDMETHODCALLTYPE D3D11SwapChain::Present(
          UINT                      SyncInterval,
          UINT                      PresentFlags,
    const DXGI_PRESENT_PARAMETERS*  pPresentParameters) {
    // blessed: threaded-fe -- stage 2 records Present for the front end,
    // whose replay comes back through here and runs the body below.
    // Stage 1 (and PRESENT_TEST) drains first and runs on the game
    // thread. The census times the game-side call.
    D3D11ThreadedContext* blessedFe = m_parent->BlessedFrontEnd();
    bool blessedReplay = unlikely(blessedFe != nullptr) && D3D11ThreadedContext::IsReplaying();

    if (unlikely(blessedFe != nullptr) && !blessedReplay
     && blessedFe->RecordsPresent(PresentFlags, pPresentParameters))
      return BlessedRecordPresent(blessedFe, SyncInterval, PresentFlags, pPresentParameters);

    BlessedPresentScope blessedFePresent(blessedFe, blessedReplay);

    // blessed: frame boundary for the render-thread probe, game side only
    if (blessed::enabled() && !blessedReplay)
      BlessedProbePresent(m_device.ptr());

    HRESULT hr = S_OK;

    // blessed: one cached-pointer-guarded branch when disabled or already
    // run; the selftest itself only ever fires once, gated on
    // BLESSED_RT_SELFTEST=1.
    BlessedRt* blessedRt = m_device->blessedRt();
    if (unlikely(blessedRt != nullptr))
      blessedRt->runSelfTestOnce();

    if (m_device->getDeviceStatus() != VK_SUCCESS)
      hr = DXGI_ERROR_DEVICE_RESET;

    if (PresentFlags & DXGI_PRESENT_TEST) {
      if (hr != S_OK)
        return hr;

      VkResult status = m_presenter->checkSwapChainStatus();
      return status == VK_SUCCESS ? S_OK : DXGI_STATUS_OCCLUDED;
    }

    if (hr != S_OK) {
      SyncFrameLatency();
      return hr;
    }

    // blessed: scene-capture end-of-frame blas/tlas build. Injected as its
    // own ordered cs chunk so it lands after every draw this frame already
    // queued but still before PresentImage's own ExecuteFlush below --
    // see blessed_scene.h's endFrame. One cached-bool branch when disabled.
    if (unlikely(BlessedSceneCapture::IsEnabled())) {
      BlessedSceneCapture::NotifySwapchainExtent(m_desc.Width, m_desc.Height);

      // blessed: skin-v2 -- the depth-only pass before this frame's mask draw
      uint32_t maskPass = BlessedSceneCapture::TakeMaskPass();

      // blessed: hook-cpu-2 -- this frame's skinned draws travel with it
      std::unique_ptr<BlessedSceneSkinBatch> skinBatch = BlessedSceneCapture::TakeSkinBatch();

      m_parent->GetContext()->InjectCs(DxvkCsQueue::Ordered,
        [maskPass, cSkinBatch = std::move(skinBatch)] (DxvkContext* ctx) mutable {
          BlessedSceneCapture::EndFrameOnCs(); // blessed: scene-cs, static per-pass counts
          ctx->blessedSceneEndFrame(maskPass, std::move(cSkinBatch));
        });
    }

    // blessed: gi -- own copy of the swapchain extent (see blessed_gi.cpp's
    // IsMainLitPassBound), refreshed every present regardless of scene
    // capture's own enable state.
    if (unlikely(BlessedGi::IsEnabled())) {
      BlessedGi::NotifySwapchainExtent(m_desc.Width, m_desc.Height);

      // blessed: probe trace, same ordered cs chunk as blessedSceneEndFrame
      // and right after it, so this frame's scene tlas is already current
      // when BlessedGiState::runTrace reads it (see blessed_gi.h). A cheap
      // no-op (one pointer check) when BLESSED_GI=const rather than probes.
      m_parent->GetContext()->InjectCs(DxvkCsQueue::Ordered,
        [] (DxvkContext* ctx) {
          ctx->blessedRunGiTrace();
        });
    }

    // blessed: BLESSED_SKIP_CASCADES -- clears the learned cascade-image set
    // on a resolution change. Cheap no-op otherwise.
    if (unlikely(BlessedCascadeSkip::IsEnabled()))
      BlessedCascadeSkip::NotifySwapchainExtent(m_desc.Width, m_desc.Height);

    // blessed: cascade-cache -- ends a cascade still open, reads back
    // verify counters, cascade_cache.jsonl windows
    if (unlikely(BlessedCascadeCache::IsEnabled())) {
      BlessedCascadeCache::NotifySwapchainExtent(m_desc.Width, m_desc.Height);
      BlessedCascadeCache::OnPresent(m_parent->GetContext());
    }

    try {
      hr = PresentImage(SyncInterval, pPresentParameters);
    } catch (const DxvkError& e) {
      Logger::err(e.message());
      hr = E_FAIL;
    }

    // blessed: gpu-pass-timing frame boundary: close, read back, write windows
    if (unlikely(BlessedGpuPasses::IsEnabled()))
      BlessedGpuPasses::OnPresent();
    // blessed: between-passes -- resets the BLESSED_MIDFRAME_SPLIT pass counter
    if (unlikely(BlessedMidframeSplit::IsEnabled()))
      BlessedMidframeSplit::OnPresent();

    // Ensure to synchronize and release the frame latency semaphore
    // even if presentation failed with STATUS_OCCLUDED, or otherwise
    // applications using the semaphore may deadlock. This works because
    // we do not increment the frame ID in those situations.
    SyncFrameLatency();

    // Ignore latency stuff if presentation failed
    DxvkLatencyStats latencyStats = { };

    if (hr == S_OK && m_latency) {
      latencyStats = m_latency->getStatistics(m_frameId);

      // blessed: threaded-fe-2 -- a recorded Present sleeps game side
      if (!blessedReplay)
        m_latency->sleepAndBeginFrame(m_frameId + 1, std::abs(m_targetFrameRate));
    }

    if (m_latencyHud)
      m_latencyHud->accumulateStats(latencyStats);

    // blessed: present-to-present frame boundary for the probe dumper
    if (unlikely(BlessedDump::IsEnabled()) && hr == S_OK)
      BlessedDump::OnPresent();

    // blessed: present-to-present frame boundary for the post-draw hook's log
    if (unlikely(BlessedHook::IsEnabled()) && hr == S_OK)
      BlessedHook::OnPresent();
    if (unlikely(BlessedLook::IsEnabled()) && hr == S_OK) // blessed: look-post
      BlessedLook::OnPresent();
    if (unlikely(BlessedHalfRate::IsAvailable())) // blessed: half-rate far field, the live switch
      BlessedHalfRate::OnPresent();
    if (unlikely(BlessedReflectHalfRate::IsEnabled())) // blessed: refl-harden
      BlessedReflectHalfRate::OnPresent();
    // blessed: gi.jsonl window boundary
    if (unlikely(BlessedGi::IsEnabled()) && hr == S_OK)
      BlessedGi::OnPresent(m_device.ptr());
    // blessed: BLESSED_SCENE_LOG window boundary
    if (unlikely(BlessedSceneCapture::IsEnabled()) && hr == S_OK)
      BlessedSceneCapture::OnPresent(m_device.ptr());
    // blessed: BLESSED_SKIP_CASCADES window boundary (cascades.jsonl)
    if (unlikely(BlessedCascadeSkip::IsEnabled()) && hr == S_OK)
      BlessedCascadeSkip::OnPresent();
    // blessed: async-compute -- a volumetrics kick never outlives its frame
    if (unlikely(BlessedVolumetrics::IsAsyncKickEnabled()))
      BlessedVolumetrics::OnPresent();
    // blessed: point-lights -- channel claims, learned maps, pointshadow.jsonl
    if (unlikely(BlessedPointShadow::IsEnabled()) && hr == S_OK)
      BlessedPointShadow::OnPresent();
    BlessedCrashLogOnPresent(); // blessed: crash-log -- re-arm the unhandled-exception filter
    // blessed: perf-halfrate -- per-frame skip state resets on every present
    if (unlikely(BlessedVolHalfRate::IsEnabled()))
      BlessedVolHalfRate::OnPresent();
    // blessed: gpu track step 2 -- one jsonl window boundary per feature
    if (hr == S_OK) {
      if (unlikely(BlessedSkipAo::IsEnabled()))
        BlessedSkipAo::OnPresent();
      if (unlikely(BlessedSkipVolumetrics::IsDrawSkipEnabled() || BlessedSkipVolumetrics::IsDispatchSkipEnabled()))
        BlessedSkipVolumetrics::OnPresent();
      if (unlikely(BlessedSkipBloom::IsEnabled()))
        BlessedSkipBloom::OnPresent();
      if (unlikely(BlessedSkipShadowMask::IsEnabled()))
        BlessedSkipShadowMask::OnPresent();
      if (unlikely(BlessedAutoInstance::IsCensusEnabled())) // blessed: auto-instancing census
        BlessedAutoInstance::OnPresent();
    }

    return hr;
  }


  UINT STDMETHODCALLTYPE D3D11SwapChain::CheckColorSpaceSupport(
          DXGI_COLOR_SPACE_TYPE     ColorSpace) {
    UINT supportFlags = 0;

    VkColorSpaceKHR vkColorSpace = ConvertColorSpace(ColorSpace);

    if (m_presenter->supportsColorSpace(vkColorSpace))
      supportFlags |= DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT;

    return supportFlags;
  }


  HRESULT STDMETHODCALLTYPE D3D11SwapChain::SetColorSpace(
          DXGI_COLOR_SPACE_TYPE     ColorSpace) {
    m_parent->BlessedDrainFrontEnd(blessed::FeDrain::Device); // blessed: threaded-fe-2
    VkColorSpaceKHR colorSpace = ConvertColorSpace(ColorSpace);

    if (!m_presenter->supportsColorSpace(colorSpace))
      return E_INVALIDARG;

    m_colorSpace = colorSpace;

    m_presenter->setSurfaceFormat(GetSurfaceFormat(m_desc.Format));
    return S_OK;
  }


  HRESULT STDMETHODCALLTYPE D3D11SwapChain::SetHDRMetaData(
    const DXGI_VK_HDR_METADATA*     pMetaData) {
    m_parent->BlessedDrainFrontEnd(blessed::FeDrain::Device); // blessed: threaded-fe-2
    // For some reason this call always seems to succeed on Windows
    if (pMetaData->Type == DXGI_HDR_METADATA_TYPE_HDR10)
      m_presenter->setHdrMetadata(ConvertHDRMetadata(pMetaData->HDR10));

    return S_OK;
  }


  void STDMETHODCALLTYPE D3D11SwapChain::GetLastPresentCount(
          UINT64*                   pLastPresentCount) {
    *pLastPresentCount = UINT64(m_frameId - DXGI_MAX_SWAP_CHAIN_BUFFERS);
  }


  void STDMETHODCALLTYPE D3D11SwapChain::GetFrameStatistics(
          DXGI_VK_FRAME_STATISTICS* pFrameStatistics) {
    std::lock_guard<dxvk::mutex> lock(m_frameStatisticsLock);
    *pFrameStatistics = m_frameStatistics;
  }


  void STDMETHODCALLTYPE D3D11SwapChain::SetTargetFrameRate(
          double                    FrameRate) {
    m_parent->BlessedDrainFrontEnd(blessed::FeDrain::Device); // blessed: threaded-fe-2
    m_targetFrameRate = FrameRate;

    if (m_presenter != nullptr)
      m_presenter->setFrameRateLimit(m_targetFrameRate, GetActualFrameLatency());
  }


  HRESULT STDMETHODCALLTYPE D3D11SwapChain::SetBackgroundColor(
    const DXGI_RGBA*                pColor) {
    m_parent->BlessedDrainFrontEnd(blessed::FeDrain::Device); // blessed: threaded-fe-2
    m_clearColor.float32[0] = pColor->r;
    m_clearColor.float32[1] = pColor->g;
    m_clearColor.float32[2] = pColor->b;
    m_clearColor.float32[3] = pColor->a;
    return S_OK;
  }


  HRESULT STDMETHODCALLTYPE D3D11SwapChain::SetRotation(
          DXGI_MODE_ROTATION        Rotation) {
    if (Rotation != DXGI_MODE_ROTATION_IDENTITY) {
      Logger::err(str::format("D3D11SwapChain::SetRotation: Rotation ", Rotation, " not supported."));
      return E_NOTIMPL;
    }

    return S_OK;
  }


  Rc<DxvkImageView> D3D11SwapChain::GetBackBufferView() {
    Rc<DxvkImage> image = GetCommonTexture(m_backBuffers[0].ptr())->GetImage();

    if (m_compositionBuffer)
      image = m_compositionBuffer;

    DxvkImageViewKey key;
    key.viewType = VK_IMAGE_VIEW_TYPE_2D;
    key.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    key.format = image->info().format;
    key.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
    key.mipIndex = 0u;
    key.mipCount = 1u;
    key.layerIndex = 0u;
    key.layerCount = 1u;

    return image->createView(key);
  }


  Rc<DxvkImage> D3D11SwapChain::BlessedTryEarlyAcquire(D3D11ImmediateContext* ctx) {
    // blessed: zero-copy-present, see blessed_zero_copy.h. Shape checks
    // first: anything that needs the blitter (HUD, colour space, gamma,
    // MSAA, more than one D3D back buffer, incremental present) or that a
    // redirect cannot undo keeps the normal path for the frame.
    if (m_zcAcquired || m_zcMismatch)
      return nullptr; // never acquire twice without a present in between

    if (m_hasHud || m_backBuffers.size() != 1u || m_compositionBuffer != nullptr)
      return nullptr;

    if (m_desc.SwapEffect == DXGI_SWAP_EFFECT_SEQUENTIAL
     || m_desc.SwapEffect == DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL)
      return nullptr;

    if (m_colorSpace != VK_COLOR_SPACE_SRGB_NONLINEAR_KHR || m_hasGammaRamp)
      return nullptr;

    Rc<DxvkImage> backBufferImage = GetCommonTexture(m_backBuffers[0].ptr())->GetImage();

    if (backBufferImage->info().sampleCount != VK_SAMPLE_COUNT_1_BIT)
      return nullptr;

    // blessed: zero-copy-present, present-mode fix -- the presenter creates
    // its swap chain at the first acquire, in the mode of the interval it
    // was last given (default 1, fifo). Before any real Present that is
    // not the game's interval, and with dynamic present modes a later
    // setSyncInterval never recreates it, so frame 1 takes the normal
    // path: its acquire runs in PresentImage after setSyncInterval. After
    // that, reapply the last real interval so a recreated swap chain
    // (resize, out of date) uses it too; a no-op when unchanged.
    if (!m_blessedSyncIntervalKnown)
      return nullptr;

    m_presenter->setSyncInterval(m_blessedSyncInterval);

    // Submit everything recorded so far with no WSI semaphore attached, so
    // the gpu starts on it while this thread may block in the acquire.
    ctx->ExecuteFlush(GpuFlushType::ImplicitStrongHint, nullptr, false);

    PresenterSync sync;
    Rc<DxvkImage> swapImage;

    VkResult status = m_presenter->acquireNextImage(sync, swapImage);

    // VK_NOT_READY (occluded, no swap chain) and errors hold no image;
    // VK_SUCCESS and VK_SUBOPTIMAL_KHR do, and must reach a present.
    if (status < 0 || status == VK_NOT_READY)
      return nullptr;

    m_zcSync = sync;
    m_zcImage = swapImage;
    m_zcAcquired = true;
    m_zcWaitEmitted = false;

    VkRect2D srcRect = ComputeSrcPresentRect();
    VkExtent2D dstSize = { swapImage->info().extent.width, swapImage->info().extent.height };
    VkRect2D dstRect = ComputeDstPresentRect(dstSize, srcRect.extent);

    bool exactFit = srcRect.offset.x == 0 && srcRect.offset.y == 0
                 && dstRect.offset.x == 0 && dstRect.offset.y == 0
                 && dstRect.extent.width  == srcRect.extent.width
                 && dstRect.extent.height == srcRect.extent.height
                 && dstSize.width  == backBufferImage->info().extent.width
                 && dstSize.height == backBufferImage->info().extent.height;

    // The presenter exposes the swap image as the sRGB half of its format
    // pair when the swap chain is mutable; the back buffer's RTV format is
    // checked per view in BlessedZeroCopy::RedirectRenderTarget.
    VkFormat swapFormat = vk::getSrgbFormatPair(swapImage->info().format).first;
    VkFormat backFormat = vk::getSrgbFormatPair(backBufferImage->info().format).first;

    bool formatMatch = swapImage->info().format == backBufferImage->info().format
                    || (swapFormat != VK_FORMAT_UNDEFINED && swapFormat == backFormat);

    // The fallback copies the acquired image back into the back buffer.
    VkImageUsageFlags needUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                                | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    bool usageMatch = (swapImage->info().usage & needUsage) == needUsage
                   && (backBufferImage->info().usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT);

    if (!exactFit || !formatMatch || !usageMatch) {
      // Stays so until the back buffers or the swap chain shape change;
      // PresentImage presents the held image through the normal blit.
      m_zcMismatch = true;

      Logger::warn(str::format("BlessedZeroCopy: swap image does not fit the back buffer (fit ",
        exactFit ? 1 : 0, ", format ", formatMatch ? 1 : 0, ", usage ", usageMatch ? 1 : 0,
        "; ", swapImage->info().format, " vs ", backBufferImage->info().format,
        "), using the present blit until the next resize"));
      return nullptr;
    }

    // Only the command list that starts here waits for the acquire; the
    // present semaphore is attached by PresentImage as usual.
    PresenterSync acquireOnly = { };
    acquireOnly.acquire = sync.acquire;

    ctx->EmitCs([cSync = acquireOnly] (DxvkContext* dxvkCtx) {
      dxvkCtx->synchronizeWsi(cSync);
    });

    m_zcWaitEmitted = true;
    return swapImage;
  }


  void D3D11SwapChain::BlessedCorrectBackBufferForRead(
          D3D11ImmediateContext*  ctx,
    const Rc<DxvkImage>&          swapImage) {
    // blessed: zero-copy-present -- a fallback while the back buffer's
    // writes were going into the acquired swap image: copy them back
    // (formats are an sRGB pair or equal, so the copy is bitwise).
    Rc<DxvkImage> backBufferImage = GetCommonTexture(m_backBuffers[0].ptr())->GetImage();

    ctx->EmitCs([
      cDst    = backBufferImage,
      cSrc    = swapImage,
      cExtent = backBufferImage->info().extent
    ] (DxvkContext* dxvkCtx) {
      VkImageSubresourceLayers layers = { VK_IMAGE_ASPECT_COLOR_BIT, 0u, 0u, 1u };

      dxvkCtx->copyImage(
        cDst, layers, VkOffset3D { 0, 0, 0 },
        cSrc, layers, VkOffset3D { 0, 0, 0 },
        cExtent);
    });
  }


  void D3D11SwapChain::BlessedReleaseEarlyAcquire() {
    // blessed: zero-copy-present -- an early acquire with no Present to
    // follow (the swap chain is going away). The presenter will not
    // acquire or tear down while an image is held, so present it as is.
    if (!m_zcAcquired)
      return;

    PresenterSync sync = m_zcSync;

    if (m_zcWaitEmitted)
      sync.acquire = VK_NULL_HANDLE;

    m_zcAcquired = false;
    m_zcWaitEmitted = false;
    m_zcImage = nullptr;
    BlessedZeroCopy::OnPresent();

    auto immediateContext = m_parent->GetContext();
    auto immediateContextLock = immediateContext->LockContext();

    immediateContext->ExecuteFlush(GpuFlushType::ExplicitFlush, nullptr, true);

    immediateContext->EmitCs([
      cDevice    = m_device,
      cPresenter = m_presenter,
      cSync      = sync,
      cFrameId   = ++m_frameId
    ] (DxvkContext* ctx) {
      ctx->synchronizeWsi(cSync);
      ctx->flushCommandList(nullptr, nullptr);

      cDevice->presentImage(cPresenter, nullptr, cFrameId, 0u, nullptr, nullptr);
    });

    immediateContext->FlushCsChunk();
  }


  HRESULT D3D11SwapChain::PresentImage(
            UINT                      SyncInterval,
      const DXGI_PRESENT_PARAMETERS*  pPresentParameters) {
    // Flush pending rendering commands before
    auto immediateContext = m_parent->GetContext();
    auto immediateContextLock = immediateContext->LockContext();

    // blessed: gpu-pass-timing, the end of the frame's last pass, before
    // the flush so it lands in the frame's own command list
    if (unlikely(BlessedGpuPasses::IsEnabled())) {
      BlessedGpuPassMark mark = BlessedGpuPasses::OnFrameEnd(m_device.ptr());

      if (mark.query != nullptr) {
        immediateContext->EmitCs([cMark = std::move(mark)] (DxvkContext* ctx) {
          BlessedGpuPasses::WriteMark(ctx, cMark);
        });
      }
    }

    immediateContext->EndFrame(m_latency);
    immediateContext->ExecuteFlush(GpuFlushType::ExplicitFlush, nullptr, true);

    // blessed: zero-copy-present, present-mode fix -- remember the interval
    // this real Present configured so the next frame's early acquire (if
    // any) applies the same one before it runs, see BlessedTryEarlyAcquire.
    m_blessedSyncIntervalKnown = true;
    m_blessedSyncInterval = SyncInterval;

    m_presenter->setSyncInterval(SyncInterval);

    // Presentation semaphores and WSI swap chain image
    if (m_latency)
      m_latency->notifyCpuPresentBegin(m_frameId + 1u);

    PresenterSync sync;
    Rc<DxvkImage> backBuffer;

    // blessed: zero-copy-present -- BlessedTryEarlyAcquire may already hold
    // this frame's image (redirected or not). Reuse it: acquireNextImage
    // blocks until the held image is presented. If the acquire semaphore
    // was already attached to an earlier command list, drop it here: a
    // binary semaphore is waited on exactly once.
    bool blessedHeld = m_zcAcquired;
    bool blessedSkipBlit = false;

    if (unlikely(blessedHeld)) {
      sync = m_zcSync;
      backBuffer = m_zcImage;

      if (m_zcWaitEmitted)
        sync.acquire = VK_NULL_HANDLE;

      blessedSkipBlit = BlessedZeroCopy::IsRedirectActive();

      m_zcAcquired = false;
      m_zcWaitEmitted = false;
      m_zcImage = nullptr;
    }

    // per-frame redirect state resets whether or not this present succeeds
    if (unlikely(BlessedZeroCopy::IsEnabled()))
      BlessedZeroCopy::OnPresent();

    if (likely(!blessedHeld)) {
      VkResult status = m_presenter->acquireNextImage(sync, backBuffer);

      if (status != VK_SUCCESS && m_latency)
        m_latency->discardTimings();

      if (status < 0)
        return E_FAIL;

      if (status == VK_NOT_READY)
        return DXGI_STATUS_OCCLUDED;
    }

    VkExtent2D dstSize = { backBuffer->info().extent.width, backBuffer->info().extent.height };

    VkRect2D srcRect = ComputeSrcPresentRect();
    VkRect2D dstRect = ComputeDstPresentRect(dstSize, srcRect.extent);

    // Incremental presentation is only supported with flip model presentation
    // on native, not 100% sure about the exact validation here.
    bool incrementalPresent = UseIncrementalPresent(pPresentParameters);

    bool sequential = m_desc.SwapEffect == DXGI_SWAP_EFFECT_SEQUENTIAL ||
                      m_desc.SwapEffect == DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;

    if (incrementalPresent && !sequential) {
      Logger::err("D3D11: Incremental present only supported with sequential present modes.");
      return DXGI_ERROR_INVALID_CALL;
    }

    DirtyRectList dirtyRects;

    if (incrementalPresent) {
      CompositeIncrementalPresent(immediateContext, pPresentParameters);

      // Redraw everything if the HUD is active since we don't
      // keep track of the exact screen areas there. Likewise,
      // nope out if there is any scaling going on.
      if (!m_hasHud && dstRect.extent != srcRect.extent) {
        VkRect2D bounds = dstRect;
        bounds.offset.x -= srcRect.offset.x;
        bounds.offset.y -= srcRect.offset.y;

        dirtyRects = NormalizeDirtyRects(pPresentParameters, bounds);
      }
    } else {
      // Nuke incremental present image out of existence to
      // save memory, also to pick the correct source image
      m_compositionBuffer = nullptr;
      m_compositionScroll = nullptr;
    }

    m_frameId += 1;

    // Present from CS thread so that we don't
    // have to synchronize with it first.
    DxvkImageViewKey viewInfo = { };
    viewInfo.viewType   = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.usage      = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    viewInfo.format     = backBuffer->info().format;
    viewInfo.aspects    = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.mipIndex   = 0u;
    viewInfo.mipCount   = 1u;
    viewInfo.layerIndex = 0u;
    viewInfo.layerCount = 1u;

    immediateContext->EmitCs([
      cDevice         = m_device,
      cBlitter        = m_blitter,
      cBackBuffer     = backBuffer->createView(viewInfo),
      cSwapImage      = GetBackBufferView(),
      cSync           = sync,
      cPresenter      = m_presenter,
      cLatency        = m_latency,
      cColorSpace     = m_colorSpace,
      cFrameId        = m_frameId,
      cDirtyRects     = std::move(dirtyRects),
      cClearColor     = m_clearColor,
      cSrcRect        = srcRect,
      cDstRect        = dstRect,
      cBlessedMark    = BlessedGpuPasses::IsEnabled() && !blessedSkipBlit // blessed: gpu-pass-timing, no mark without a blit
        ? BlessedGpuPasses::OnPresentBlit(m_device.ptr()) : BlessedGpuPassMark(),
      cSkipBlit       = blessedSkipBlit // blessed: zero-copy-present
    ] (DxvkContext* ctx) {
      // blessed: zero-copy-present -- the frame's back-buffer render
      // targets were already redirected straight into cBackBuffer (the
      // acquired swap chain image) at bind time, and nothing read the
      // back buffer since, so there is nothing left to blit.
      if (!cSkipBlit) {
        // Update back buffer color space as necessary
        if (cSwapImage->image()->info().colorSpace != cColorSpace) {
          DxvkImageUsageInfo usage = { };
          usage.colorSpace = cColorSpace;

          ctx->ensureImageCompatibility(cSwapImage->image(), usage);
        }

        // Blit the D3D back buffer onto the actual Vulkan
        // swap chain and render the HUD if we have one.
        auto contextObjects = ctx->beginExternalRendering();

        cBlitter->present(contextObjects, cClearColor,
          cBackBuffer, cDstRect, cSwapImage, cSrcRect);

        // blessed: gpu-pass-timing, the end of dxvk's own present blit
        if (unlikely(cBlessedMark.query != nullptr))
          BlessedGpuPasses::WriteMark(ctx, cBlessedMark);
      }

      // Submit current command list and present
      ctx->synchronizeWsi(cSync);
      ctx->flushCommandList(nullptr, nullptr);

      cDevice->presentImage(cPresenter, cLatency, cFrameId,
        cDirtyRects.size(), cDirtyRects.data(), nullptr);
    });

    if (m_backBuffers.size() > 1u)
      RotateBackBuffers(immediateContext);

    immediateContext->FlushCsChunk();

    if (m_latency) {
      m_latency->notifyCpuPresentEnd(m_frameId);

      if (m_latency->needsAutoMarkers()) {
        immediateContext->EmitCs([
          cLatency = m_latency,
          cFrameId = m_frameId
        ] (DxvkContext* ctx) {
          ctx->beginLatencyTracking(cLatency, cFrameId + 1u);
        });
      }
    }

    return S_OK;
  }


  void D3D11SwapChain::RotateBackBuffers(D3D11ImmediateContext* ctx) {
    small_vector<Rc<DxvkImage>, 4> images;

    for (uint32_t i = 0; i < m_backBuffers.size(); i++)
      images.push_back(GetCommonTexture(m_backBuffers[i].ptr())->GetImage());

    ctx->EmitCs([
      cImages = std::move(images)
    ] (DxvkContext* ctx) {
      auto allocation = cImages[0]->storage();

      for (size_t i = 0u; i + 1 < cImages.size(); i++) {
        ctx->invalidateImage(cImages[i], cImages[i + 1]->storage(),
          cImages[i + 1]->info().layout);
      }

      ctx->invalidateImage(cImages[cImages.size() - 1u],
        std::move(allocation), cImages[0]->info().layout);
    });
  }


  void D3D11SwapChain::CreateFrameLatencyEvent() {
    m_frameLatencySignal = new sync::CallbackFence(m_frameId);

    if (m_desc.Flags & DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT)
      m_frameLatencyEvent = CreateSemaphore(nullptr, m_frameLatency, DXGI_MAX_SWAP_CHAIN_BUFFERS, nullptr);
  }


  void D3D11SwapChain::CreatePresenter() {
    PresenterDesc presenterDesc = { };
    presenterDesc.deferSurfaceCreation = m_parent->GetOptions()->deferSurfaceCreation;

    m_presenter = new Presenter(m_device, m_frameLatencySignal, presenterDesc, [
      cAdapter  = m_device->adapter(),
      cFactory  = m_surfaceFactory
    ] (VkSurfaceKHR* surface) {
      return cFactory->CreateSurface(
        cAdapter->vki()->instance(),
        cAdapter->handle(), surface);
    });

    // blessed: present-idle -- BLESSED_PRESENT=bridge presents through our own dxgi swap chain
    if (unlikely(BlessedPresent::Mode() == BlessedPresentMode::Bridge)) {
      Com<IBlessedDXGIVkWindow> window;

      if (SUCCEEDED(m_surfaceFactory->QueryInterface(__uuidof(IBlessedDXGIVkWindow), reinterpret_cast<void**>(&window))))
        m_presenter->blessedUseBridge(window->GetWindow());
    }

    // blessed: present-fse-appcontrolled -- BLESSED_FSE=app requests exclusive fullscreen
    // on the app's own window, application-controlled instead of driver-managed
    if (unlikely(BlessedFse::Mode() == BlessedFseMode::App)) {
      Com<IBlessedDXGIVkWindow> window;

      if (SUCCEEDED(m_surfaceFactory->QueryInterface(__uuidof(IBlessedDXGIVkWindow), reinterpret_cast<void**>(&window))))
        m_presenter->blessedRequestAppControlledFse(window->GetWindow());
    }

    m_presenter->setSurfaceFormat(GetSurfaceFormat(m_desc.Format));
    m_presenter->setSurfaceExtent({ m_desc.Width, m_desc.Height });
    m_presenter->setFrameRateLimit(m_targetFrameRate, GetActualFrameLatency());

    m_latency = m_device->createLatencyTracker(m_presenter);

    Com<D3D11ReflexDevice> reflex = GetReflexDevice();
    reflex->RegisterLatencyTracker(m_latency);
  }


  void D3D11SwapChain::CreateBackBuffers() {
    // Explicitly destroy current swap image before
    // creating a new one to free up resources
    m_backBuffers.clear();

    m_compositionBuffer = nullptr;
    m_compositionScroll = nullptr;

    bool sequential = m_desc.SwapEffect == DXGI_SWAP_EFFECT_SEQUENTIAL ||
                      m_desc.SwapEffect == DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
    uint32_t backBufferCount = sequential ? m_desc.BufferCount : 1u;

    // Create new back buffer
    D3D11_COMMON_TEXTURE_DESC desc;
    desc.Width              = std::max(m_desc.Width,  1u);
    desc.Height             = std::max(m_desc.Height, 1u);
    desc.Depth              = 1;
    desc.MipLevels          = 1;
    desc.ArraySize          = 1;
    desc.Format             = m_desc.Format;
    desc.SampleDesc         = m_desc.SampleDesc;
    desc.Usage              = D3D11_USAGE_DEFAULT;
    desc.BindFlags          = 0;
    desc.CPUAccessFlags     = 0;
    desc.MiscFlags          = 0;
    desc.TextureLayout      = D3D11_TEXTURE_LAYOUT_UNDEFINED;

    if (m_desc.BufferUsage & DXGI_USAGE_RENDER_TARGET_OUTPUT)
      desc.BindFlags |= D3D11_BIND_RENDER_TARGET;

    if (m_desc.BufferUsage & DXGI_USAGE_SHADER_INPUT)
      desc.BindFlags |= D3D11_BIND_SHADER_RESOURCE;

    if (m_desc.BufferUsage & DXGI_USAGE_UNORDERED_ACCESS)
      desc.BindFlags |= D3D11_BIND_UNORDERED_ACCESS;
    
    if (m_desc.Flags & DXGI_SWAP_CHAIN_FLAG_GDI_COMPATIBLE)
      desc.MiscFlags |= D3D11_RESOURCE_MISC_GDI_COMPATIBLE;
    
    DXGI_USAGE dxgiUsage = DXGI_USAGE_BACK_BUFFER;

    for (uint32_t i = 0; i < backBufferCount; i++) {
      if (m_desc.SwapEffect == DXGI_SWAP_EFFECT_DISCARD
       || m_desc.SwapEffect == DXGI_SWAP_EFFECT_FLIP_DISCARD)
         dxgiUsage |= DXGI_USAGE_DISCARD_ON_PRESENT;

      m_backBuffers.push_back(new D3D11Texture2D(
        m_parent, this, &desc, dxgiUsage));

      dxgiUsage |= DXGI_USAGE_READ_ONLY;
    }

    small_vector<Rc<DxvkImage>, 4> images;

    for (uint32_t i = 0; i < backBufferCount; i++)
      images.push_back(GetCommonTexture(m_backBuffers[i].ptr())->GetImage());

    // Initialize images so that we can use them. Clearing
    // to black prevents garbled output for the first frame.
    m_parent->GetContext()->InjectCs(DxvkCsQueue::HighPriority, [
      cImages = std::move(images)
    ] (DxvkContext* ctx) {
      for (size_t i = 0; i < cImages.size(); i++) {
        ctx->setDebugName(cImages[i], str::format("Back buffer ", i).c_str());
        ctx->initImage(cImages[i], VK_IMAGE_LAYOUT_UNDEFINED);
      }
    });

    // blessed: zero-copy-present -- track this back buffer so render-target
    // binds can be redirected. Ineligible (null) with more than one D3D back
    // buffer (sequential, incremental present), or when the back buffer can
    // be written or read outside the hooked paths (UAV, GDI).
    m_zcMismatch = false;

    if (BlessedZeroCopy::IsEnabled()) {
      bool eligible = backBufferCount == 1u
        && !(m_desc.BufferUsage & DXGI_USAGE_UNORDERED_ACCESS)
        && !(m_desc.Flags & DXGI_SWAP_CHAIN_FLAG_GDI_COMPATIBLE);

      Rc<DxvkImage> trackedImage = eligible
        ? GetCommonTexture(m_backBuffers[0].ptr())->GetImage()
        : nullptr;

      BlessedZeroCopy::SetTrackedSwapChain(this,
        static_cast<ID3D11Resource*>(m_backBuffers[0].ptr()), trackedImage);
    }
  }


  void D3D11SwapChain::CreateBlitter() {
    Rc<hud::Hud> hud = hud::Hud::createHud(m_device);

    if (hud) {
      hud->addItem<hud::HudClientApiItem>("api", 1, GetApiName());

      if (m_latency)
        m_latencyHud = hud->addItem<hud::HudLatencyItem>("latency", 4);
    }

    m_hasHud = hud && !hud->empty();
    m_blitter = new DxvkSwapchainBlitter(m_device, std::move(hud));
  }


  void D3D11SwapChain::DestroyFrameLatencyEvent() {
    CloseHandle(m_frameLatencyEvent);
  }


  void D3D11SwapChain::DestroyLatencyTracker() {
    // Need to make sure the context stops using
    // the tracker for submissions
    m_parent->BlessedDrainFrontEnd(blessed::FeDrain::Device); // blessed: threaded-fe
    m_parent->GetContext()->InjectCs(DxvkCsQueue::Ordered, [
      cLatency = m_latency
    ] (DxvkContext* ctx) {
      ctx->endLatencyTracking(cLatency);
    });

    Com<D3D11ReflexDevice> reflex = GetReflexDevice();
    reflex->UnregisterLatencyTracker(m_latency);
  }


  void D3D11SwapChain::SyncFrameLatency() {
    // Wait for the sync event so that we respect the maximum frame latency
    // blessed: threaded-fe-2 -- one frame less on the front end: the game
    // may record one Present ahead of it, so the depth in flight is kept
    D3D11ThreadedContext* blessedFe = m_parent->BlessedFrontEnd();
    uint64_t blessedAhead = unlikely(blessedFe != nullptr) && blessedFe->RunsAhead()
      && D3D11ThreadedContext::IsReplaying() ? 1u : 0u;

    m_frameLatencySignal->wait(m_frameId - GetActualFrameLatency() + blessedAhead);

    m_frameLatencySignal->setCallback(m_frameId, [this,
      cFrameId           = m_frameId,
      cFrameLatencyEvent = m_frameLatencyEvent
    ] () {
      if (cFrameLatencyEvent)
        ReleaseSemaphore(cFrameLatencyEvent, 1, nullptr);

      std::lock_guard<dxvk::mutex> lock(m_frameStatisticsLock);
      m_frameStatistics.PresentCount = cFrameId - DXGI_MAX_SWAP_CHAIN_BUFFERS;
      m_frameStatistics.PresentQPCTime = dxvk::high_resolution_clock::get_counter();
    });
  }


  uint32_t D3D11SwapChain::GetActualFrameLatency() {
    // DXGI does not seem to implicitly synchronize waitable swap chains,
    // so in that case we should just respect the user config. For regular
    // swap chains, pick the latency from the DXGI device.
    uint32_t maxFrameLatency = DXGI_MAX_SWAP_CHAIN_BUFFERS;

    if (!(m_desc.Flags & DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT))
      m_dxgiDevice->GetMaximumFrameLatency(&maxFrameLatency);

    if (m_frameLatencyCap)
      maxFrameLatency = std::min(maxFrameLatency, m_frameLatencyCap);

    maxFrameLatency = std::min(maxFrameLatency, m_desc.BufferCount);
    return maxFrameLatency;
  }


  VkSurfaceFormatKHR D3D11SwapChain::GetSurfaceFormat(DXGI_FORMAT Format) {
    switch (Format) {
      default:
        Logger::warn(str::format("D3D11SwapChain: Unexpected format: ", m_desc.Format));
        [[fallthrough]];

      case DXGI_FORMAT_R8G8B8A8_UNORM:
      case DXGI_FORMAT_B8G8R8A8_UNORM:
        return { VK_FORMAT_R8G8B8A8_UNORM, m_colorSpace };

      case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
      case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
        return { VK_FORMAT_R8G8B8A8_SRGB, m_colorSpace };

      case DXGI_FORMAT_R10G10B10A2_UNORM:
        return { VK_FORMAT_A2B10G10R10_UNORM_PACK32, m_colorSpace };

      case DXGI_FORMAT_R16G16B16A16_FLOAT:
        return { VK_FORMAT_R16G16B16A16_SFLOAT, m_colorSpace };
    }
  }


  Com<D3D11ReflexDevice> D3D11SwapChain::GetReflexDevice() {
    Com<ID3DLowLatencyDevice> llDevice;
    m_parent->QueryInterface(__uuidof(ID3DLowLatencyDevice), reinterpret_cast<void**>(&llDevice));

    return static_cast<D3D11ReflexDevice*>(llDevice.ptr());
  }


  VkRect2D D3D11SwapChain::ComputeSrcPresentRect() const {
    VkRect2D rect = {};
    rect.extent.width = m_desc.Width;
    rect.extent.height = m_desc.Height;
    return rect;
  }


  VkRect2D D3D11SwapChain::ComputeDstPresentRect(VkExtent2D DstSize, VkExtent2D SrcSize) const {
    VkRect2D result = {};

    switch (m_desc.Scaling) {
      default:
      case DXGI_SCALING_STRETCH: {
        result.extent = DstSize;
      } break;

      case DXGI_SCALING_NONE: {
        // TODO honour WS_EX_LAYOUTRTL
        result.extent = SrcSize;
      } break;

      case DXGI_SCALING_ASPECT_RATIO_STRETCH: {
        if (DstSize.width * SrcSize.height > SrcSize.width * DstSize.height) {
          // Destination is wider than source, offset horizontally
          result.extent.width = (SrcSize.width * DstSize.height) / SrcSize.height;
          result.extent.height = DstSize.height;
          result.offset.x = int32_t(DstSize.width - result.extent.width) / 2;
        } else if (DstSize.width * SrcSize.height < SrcSize.width * DstSize.height) {
          // Destination is taller than source, offset vertically
          result.extent.width = DstSize.width;
          result.extent.height = (SrcSize.height * DstSize.width) / SrcSize.width;
          result.offset.y = int32_t(DstSize.height - result.extent.height) / 2;
        } else {
          // Aspect ratio matches, simple stretch.
          result.extent = DstSize;
        }
      } break;
    }

    return result;
  }


  void D3D11SwapChain::CompositeIncrementalPresent(
          D3D11ImmediateContext*   pContext,
    const DXGI_PRESENT_PARAMETERS* pPresentParameters) {
    Rc<DxvkImage> backBuffer = GetCommonTexture(m_backBuffers.front().ptr())->GetImage();

    if (!m_compositionVs || !m_compositionFs)
      CreateCompositionShaders();

    pContext->ResetDirtyTracking();
    pContext->ResetCommandListState();

    if (!m_compositionBuffer) {
      // Create front buffer as a shader-readable render target
      DxvkImageCreateInfo imageInfo = backBuffer->info();
      imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                      | VK_IMAGE_USAGE_SAMPLED_BIT
                      | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                      | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
      imageInfo.stages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                       | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                       | VK_PIPELINE_STAGE_TRANSFER_BIT;
      imageInfo.access = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT
                       | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                       | VK_ACCESS_SHADER_READ_BIT
                       | VK_ACCESS_TRANSFER_READ_BIT
                       | VK_ACCESS_TRANSFER_WRITE_BIT;
      imageInfo.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      imageInfo.debugName = "Composition";

      m_compositionBuffer = m_device->createImage(imageInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

      // Create another image that we can render the scroll region to
      imageInfo.debugName = "Composition (Scroll)";
      m_compositionScroll = m_device->createImage(imageInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

      // Create internal images for incremental presentation. If this is the
      // first frame to use incremental present, then we know for sure that
      // the last back buffer contains the full contents last presented, so
      // simply copy the last back buffer to it.
      pContext->EmitCs([
        cPrevImage    = GetCommonTexture(m_backBuffers.back().ptr())->GetImage(),
        cCurrImage    = m_compositionBuffer,
        cScrollImage  = m_compositionScroll
      ] (DxvkContext* ctx) {
        VkExtent3D extent = cCurrImage->info().extent;

        VkImageSubresourceLayers subresource = {};
        subresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        subresource.layerCount = 1u;

        ctx->initImage(cScrollImage, VK_IMAGE_LAYOUT_UNDEFINED);
        ctx->initImage(cCurrImage, VK_IMAGE_LAYOUT_UNDEFINED);
        ctx->copyImage(cCurrImage, subresource, VkOffset3D(),
          cPrevImage, subresource, VkOffset3D(), extent);
      });
    }

    DxvkImageViewKey renderViewInfo = {};
    renderViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    renderViewInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    renderViewInfo.format = backBuffer->info().format;
    renderViewInfo.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    renderViewInfo.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
    renderViewInfo.mipCount = 1u;
    renderViewInfo.layerCount = 1u;
    renderViewInfo.packedSwizzle = DxvkImageViewKey::packSwizzle({
      VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G,
      VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A });

    DxvkImageViewKey shaderViewInfo = renderViewInfo;
    shaderViewInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    shaderViewInfo.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // Swapchain resolution
    VkExtent2D resolution = {
      backBuffer->info().extent.width,
      backBuffer->info().extent.height };

    // Set up all render state
    pContext->EmitCs([
      cVs = m_compositionVs,
      cFs = m_compositionFs
    ] (DxvkContext* ctx) mutable {
      ctx->beginDebugLabel(vk::makeLabel(0xc6c0dc, "DXGI incremental present"));

      ctx->bindShader<VK_SHADER_STAGE_VERTEX_BIT>(std::move(cVs));
      ctx->bindShader<VK_SHADER_STAGE_FRAGMENT_BIT>(std::move(cFs));

      DxvkInputAssemblyState iaState = {};
      iaState.setPrimitiveTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP);

      ctx->setInputAssemblyState(iaState);

      DxvkRasterizerState rsState = {};
      rsState.setPolygonMode(VK_POLYGON_MODE_FILL);
      rsState.setCullMode(VK_CULL_MODE_BACK_BIT);
      rsState.setFrontFace(VK_FRONT_FACE_COUNTER_CLOCKWISE);
      rsState.setSampleCount(VK_SAMPLE_COUNT_1_BIT);

      ctx->setRasterizerState(rsState);
    });

    // Check if we have a valid, non-empty scroll region and a non-zero
    // scroll offset. If so, we need to copy the scroll area first.
    bool scroll = false;

    if (pPresentParameters->pScrollRect && pPresentParameters->pScrollOffset) {
      scroll = pPresentParameters->pScrollRect->right > pPresentParameters->pScrollRect->left
            && pPresentParameters->pScrollRect->bottom > pPresentParameters->pScrollRect->top
            && (pPresentParameters->pScrollOffset->x || pPresentParameters->pScrollOffset->y);
    }

    if (scroll) {
      pContext->EmitCs([
        cSrcView      = m_compositionBuffer->createView(shaderViewInfo),
        cDstView      = m_compositionScroll->createView(renderViewInfo),
        cScrollRect   = *pPresentParameters->pScrollRect,
        cScrollOffset = *pPresentParameters->pScrollOffset
      ] (DxvkContext* ctx) mutable {
        DxvkAttachment attachment = {};
        attachment.view = cDstView;

        ctx->clearRenderTarget(attachment, 0u, VkClearValue(), VK_IMAGE_ASPECT_COLOR_BIT);

        DxvkRenderTargets rts = {};
        rts.color[0].view = std::move(cDstView);

        ctx->bindRenderTargets(std::move(rts), 0u);
        ctx->bindResourceImageView(VK_SHADER_STAGE_FRAGMENT_BIT, 0, std::move(cSrcView));

        DxvkViewport viewport = {};
        viewport.scissor.extent.width = uint32_t(cScrollRect.right - cScrollRect.left);
        viewport.scissor.extent.height = uint32_t(cScrollRect.bottom - cScrollRect.top);
        viewport.viewport.width = float(viewport.scissor.extent.width);
        viewport.viewport.height = float(viewport.scissor.extent.height);
        viewport.viewport.maxDepth = 1.0f;

        ctx->setViewports(1u, &viewport);

        CompositionArgs args = {};
        args.srcOffset.x = cScrollRect.left - cScrollOffset.x;
        args.srcOffset.y = cScrollRect.top - cScrollOffset.y;
        args.extent = viewport.scissor.extent;
        args.resolution = viewport.scissor.extent;

        ctx->pushData(VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(args), &args);

        VkDrawIndirectCommand draw = {};
        draw.vertexCount = 4u;
        draw.instanceCount = 1u;

        ctx->draw(1u, &draw);
      });
    }

    // Bind actual front buffer for rendering now
    pContext->EmitCs([
      cDstView    = m_compositionBuffer->createView(renderViewInfo),
      cResolution = resolution
    ] (DxvkContext* ctx) mutable {
      DxvkRenderTargets rts = {};
      rts.color[0].view = std::move(cDstView);

      ctx->bindRenderTargets(std::move(rts), 0u);

      DxvkViewport viewport = {};
      viewport.scissor.extent = cResolution;
      viewport.viewport.width = float(cResolution.width);
      viewport.viewport.height = float(cResolution.height);
      viewport.viewport.maxDepth = 1.0f;

      ctx->setViewports(1u, &viewport);
    });

    if (scroll) {
      pContext->EmitCs([
        cScrollView   = m_compositionScroll->createView(shaderViewInfo),
        cBufferView   = backBuffer->createView(shaderViewInfo),
        cScrollRect   = *pPresentParameters->pScrollRect,
        cScrollOffset = *pPresentParameters->pScrollOffset,
        cResolution   = resolution
      ] (DxvkContext* ctx) mutable {
        CompositionArgs args = {};
        args.srcOffset = { cScrollRect.left - cScrollOffset.x, cScrollRect.top - cScrollOffset.y };
        args.dstOffset = { cScrollRect.left, cScrollRect.top };
        args.extent.width = cScrollRect.right - cScrollRect.left;
        args.extent.height = cScrollRect.bottom - cScrollRect.top;
        args.resolution = cResolution;

        VkDrawIndirectCommand draw = {};
        draw.vertexCount = 4u;
        draw.instanceCount = 1u;

        ctx->bindResourceImageView(VK_SHADER_STAGE_FRAGMENT_BIT, 0, std::move(cBufferView));
        ctx->pushData(VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(args), &args);
        ctx->draw(1u, &draw);

        args.srcOffset = { 0, 0 };
        args.dstOffset = { cScrollRect.left, cScrollRect.top };
        args.extent.width = cScrollRect.right - cScrollRect.left - std::abs(cScrollOffset.x);
        args.extent.height = cScrollRect.bottom - cScrollRect.top - std::abs(cScrollOffset.y);
        args.resolution = cResolution;

        if (cScrollOffset.x > 0) {
          args.srcOffset.x += cScrollOffset.x;
          args.dstOffset.x += cScrollOffset.x;
        }

        if (cScrollOffset.y > 0) {
          args.srcOffset.y += cScrollOffset.y;
          args.dstOffset.y += cScrollOffset.y;
        }

        ctx->bindResourceImageView(VK_SHADER_STAGE_FRAGMENT_BIT, 0, std::move(cScrollView));
        ctx->pushData(VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(args), &args);
        ctx->draw(1u, &draw);
      });
    }

    // Apply all the actual dirty rects
    if (pPresentParameters->DirtyRectsCount) {
      pContext->EmitCs([
        cSrcView = backBuffer->createView(shaderViewInfo)
      ] (DxvkContext* ctx) mutable {
        ctx->bindResourceImageView(VK_SHADER_STAGE_FRAGMENT_BIT, 0, std::move(cSrcView));
      });

      for (uint32_t i = 0u; i < pPresentParameters->DirtyRectsCount; i++) {
        RECT rect = pPresentParameters->pDirtyRects[i];

        if (rect.right > rect.left && rect.bottom > rect.top) {
          pContext->EmitCs([
            cRect       = rect,
            cResolution = resolution
          ] (DxvkContext* ctx) mutable {
            VkDrawIndirectCommand draw = {};
            draw.vertexCount = 4u;
            draw.instanceCount = 1u;

            CompositionArgs args = {};
            args.srcOffset.x = cRect.left;
            args.srcOffset.y = cRect.top;
            args.dstOffset.x = cRect.left;
            args.dstOffset.y = cRect.top;
            args.extent.width = cRect.right - cRect.left;
            args.extent.height = cRect.bottom - cRect.top;
            args.resolution = cResolution;

            ctx->pushData(VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(args), &args);
            ctx->draw(1u, &draw);
          });
        }
      }
    }

    // Ensure D3D context state is properly reapplied
    pContext->EmitCs([] (DxvkContext* ctx) {
      ctx->endDebugLabel();
    });

    pContext->RestoreCommandListState();
  }



  bool D3D11SwapChain::UseIncrementalPresent(
    const DXGI_PRESENT_PARAMETERS* pPresentParameters) const {
    if (!pPresentParameters)
      return false;

    if (pPresentParameters->DirtyRectsCount) {
      bool hasFullRect = false;

      RECT fullRect = {};
      fullRect.right = m_desc.Width;
      fullRect.bottom = m_desc.Height;

      for (uint32_t i = 0u; i < pPresentParameters->DirtyRectsCount; i++) {
        const auto& rect = pPresentParameters->pDirtyRects[i];

        hasFullRect = hasFullRect || (rect.left <= fullRect.left && rect.top <= fullRect.top
          && rect.right >= fullRect.right && rect.bottom >= fullRect.bottom);
      }

      if (!hasFullRect)
        return true;
    }

    if (pPresentParameters->pScrollRect && pPresentParameters->pScrollOffset) {
      const auto& rect = *pPresentParameters->pScrollRect;
      const auto& offset = *pPresentParameters->pScrollOffset;

      if (rect.left < rect.right && rect.top < rect.bottom && (offset.x || offset.y))
        return true;
    }

    return false;
  }


  void D3D11SwapChain::CreateCompositionShaders() {
    const std::array<DxvkBindingInfo, 1> fsBindings = {{
      { 0u, 0u, 0u, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1u, VK_IMAGE_VIEW_TYPE_2D, VK_ACCESS_SHADER_READ_BIT },
    }};

    DxvkSpirvShaderCreateInfo vsInfo = { };
    vsInfo.localPushData = DxvkPushDataBlock(0, sizeof(CompositionArgs), sizeof(uint32_t), 0u);
    vsInfo.debugName = "DXGI_VS";
    m_compositionVs = new DxvkSpirvShader(vsInfo, d3d11_composition_vert);

    DxvkSpirvShaderCreateInfo fsInfo = { };
    fsInfo.bindingCount = fsBindings.size();
    fsInfo.bindings = fsBindings.data();
    fsInfo.debugName = "DXGI_FS";
    m_compositionFs = new DxvkSpirvShader(fsInfo, d3d11_composition_frag);
  }


  D3D11SwapChain::DirtyRectList D3D11SwapChain::NormalizeDirtyRects(const DXGI_PRESENT_PARAMETERS* pPresentParameters, VkRect2D Bounds) const {
    DirtyRectList result;

    if (pPresentParameters->pScrollRect && pPresentParameters->pScrollOffset
     && (pPresentParameters->pScrollOffset->x || pPresentParameters->pScrollOffset->y))
      AddDirtyRect(result, *pPresentParameters->pScrollRect, Bounds);

    for (uint32_t i = 0u; i < pPresentParameters->DirtyRectsCount; i++)
      AddDirtyRect(result, pPresentParameters->pDirtyRects[i], Bounds);

    return result;
  }


  void D3D11SwapChain::AddDirtyRect(DirtyRectList& List, RECT Rect, VkRect2D Bounds) const {
    // Clamp rect to screen area, and ignore if the result is empty
    Rect.left = std::max<int32_t>(Rect.left + Bounds.offset.x, 0);
    Rect.top = std::max<int32_t>(Rect.top + Bounds.offset.y, 0);
    Rect.right = std::min<int32_t>(Rect.right + Bounds.offset.x, Bounds.extent.width);
    Rect.bottom = std::min<int32_t>(Rect.bottom + Bounds.offset.y, Bounds.extent.height);

    if (Rect.left >= Rect.right || Rect.top >= Rect.bottom)
      return;

    // Scan existing list and remove any rects that overlap,
    // merging the overlapping rectangles into the new rect.
    auto iter = List.begin();

    while (iter != List.end()) {
      RECT next = {};
      next.left = iter->offset.x;
      next.right = iter->offset.x + iter->extent.width;
      next.top = iter->offset.y;
      next.bottom = iter->offset.y + iter->extent.height;

      bool overlap = next.left < Rect.right && Rect.left < next.right
                  && next.top < Rect.bottom && Rect.top < next.bottom;

      if (overlap) {
        iter = List.erase(iter);

        Rect.left = std::min(Rect.left, next.left);
        Rect.top = std::min(Rect.top, next.top);
        Rect.right = std::max(Rect.right, next.right);
        Rect.bottom = std::max(Rect.bottom, next.bottom);
      } else {
        iter++;
      }
    }

    // No more overlapping rectangles in list, add new one.
    auto& vkRect = List.emplace_back();
    vkRect.offset.x = Rect.left;
    vkRect.offset.y = Rect.top;
    vkRect.extent.width = Rect.right - Rect.left;
    vkRect.extent.height = Rect.bottom - Rect.top;
  }


  std::string D3D11SwapChain::GetApiName() const {
    Com<IDXGIDXVKDevice> device;
    m_parent->QueryInterface(__uuidof(IDXGIDXVKDevice), reinterpret_cast<void**>(&device));

    uint32_t apiVersion = device->GetAPIVersion();
    uint32_t featureLevel = m_parent->GetFeatureLevel();

    uint32_t flHi = (featureLevel >> 12);
    uint32_t flLo = (featureLevel >> 8) & 0x7;

    bool is11On12 = m_parent->Is11on12Device();

    return str::format("D3D", apiVersion, (is11On12 ? "On12" : ""), " FL", flHi, "_", flLo);
  }

}
