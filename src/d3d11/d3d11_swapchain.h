#pragma once

#include "d3d11_texture.h"

#include "../dxvk/hud/dxvk_hud.h"

#include "../dxvk/dxvk_latency.h"
#include "../dxvk/dxvk_presenter.h" // blessed: zero-copy-present, PresenterSync
#include "../dxvk/dxvk_swapchain_blitter.h"

#include "../util/sync/sync_signal.h"

namespace dxvk {
  
  class D3D11Device;
  class D3D11DXGIDevice;
  class D3D11ThreadedContext; // blessed: threaded-fe-2
  class BlessedZeroCopy; // blessed: zero-copy-present

  class D3D11SwapChain : public ComObject<IDXGIVkSwapChain3> {
    constexpr static uint32_t DefaultFrameLatency = 1;
    friend class BlessedZeroCopy; // blessed: zero-copy-present
  public:

    D3D11SwapChain(
            D3D11DXGIDevice*          pContainer,
            D3D11Device*              pDevice,
            IDXGIVkSurfaceFactory*    pSurfaceFactory,
      const DXGI_SWAP_CHAIN_DESC1*    pDesc);
    
    ~D3D11SwapChain();

    HRESULT STDMETHODCALLTYPE QueryInterface(
            REFIID                    riid,
            void**                    ppvObject);

    HRESULT STDMETHODCALLTYPE GetDesc(
            DXGI_SWAP_CHAIN_DESC1*    pDesc);

    HRESULT STDMETHODCALLTYPE GetAdapter(
            REFIID                    riid,
            void**                    ppvObject);
    
    HRESULT STDMETHODCALLTYPE GetDevice(
            REFIID                    riid,
            void**                    ppDevice);
    
    HRESULT STDMETHODCALLTYPE GetImage(
            UINT                      BufferId,
            REFIID                    riid,
            void**                    ppBuffer);

    UINT STDMETHODCALLTYPE GetImageIndex();

    UINT STDMETHODCALLTYPE GetFrameLatency();

    HANDLE STDMETHODCALLTYPE GetFrameLatencyEvent();

    HRESULT STDMETHODCALLTYPE ChangeProperties(
      const DXGI_SWAP_CHAIN_DESC1*    pDesc,
      const UINT*                     pNodeMasks,
            IUnknown* const*          ppPresentQueues);

    HRESULT STDMETHODCALLTYPE SetPresentRegion(
      const RECT*                     pRegion);

    HRESULT STDMETHODCALLTYPE SetGammaControl(
            UINT                      NumControlPoints,
      const DXGI_RGB*                 pControlPoints);

    HRESULT STDMETHODCALLTYPE SetFrameLatency(
            UINT                      MaxLatency);

    HRESULT STDMETHODCALLTYPE Present(
            UINT                      SyncInterval,
            UINT                      PresentFlags,
      const DXGI_PRESENT_PARAMETERS*  pPresentParameters);

    UINT STDMETHODCALLTYPE CheckColorSpaceSupport(
            DXGI_COLOR_SPACE_TYPE     ColorSpace);

    HRESULT STDMETHODCALLTYPE SetColorSpace(
            DXGI_COLOR_SPACE_TYPE     ColorSpace);

    HRESULT STDMETHODCALLTYPE SetHDRMetaData(
      const DXGI_VK_HDR_METADATA*     pMetaData);

    void STDMETHODCALLTYPE GetLastPresentCount(
            UINT64*                   pLastPresentCount);

    void STDMETHODCALLTYPE GetFrameStatistics(
            DXGI_VK_FRAME_STATISTICS* pFrameStatistics);

    void STDMETHODCALLTYPE SetTargetFrameRate(
            double                    FrameRate);

    HRESULT STDMETHODCALLTYPE SetBackgroundColor(
      const DXGI_RGBA*                pColor);

    HRESULT STDMETHODCALLTYPE SetRotation(
            DXGI_MODE_ROTATION        Rotation);

    // blessed: threaded-fe-2 -- the front end's replay of a recorded
    // Present, see blessed_threaded_present.cpp
    HRESULT BlessedReplayPresent(
            UINT                      SyncInterval,
            UINT                      PresentFlags,
      const DXGI_PRESENT_PARAMETERS*  pPresentParameters,
            uint64_t*                 pFrameId);

  private:

    // blessed: threaded-fe-2 -- the game-side half of a recorded Present
    HRESULT BlessedRecordPresent(
            D3D11ThreadedContext*     pFrontEnd,
            UINT                      SyncInterval,
            UINT                      PresentFlags,
      const DXGI_PRESENT_PARAMETERS*  pPresentParameters);

    using DirtyRectList = small_vector<VkRectLayerKHR, 4>;

    enum BindingIds : uint32_t {
      Image = 0,
      Gamma = 1,
    };

    struct CompositionArgs {
      VkOffset2D srcOffset;
      VkOffset2D dstOffset;
      VkExtent2D extent;
      VkExtent2D resolution;
    };

    Com<D3D11DXGIDevice, false> m_dxgiDevice;
    
    D3D11Device*              m_parent;
    Com<IDXGIVkSurfaceFactory> m_surfaceFactory;

    DXGI_SWAP_CHAIN_DESC1     m_desc;

    Rc<DxvkDevice>            m_device;
    Rc<Presenter>             m_presenter;

    Rc<DxvkSwapchainBlitter>  m_blitter;
    Rc<DxvkLatencyTracker>    m_latency;

    small_vector<Com<D3D11Texture2D, false>, 4> m_backBuffers;

    Rc<DxvkImage>             m_compositionBuffer;
    Rc<DxvkImage>             m_compositionScroll;

    Rc<DxvkShader>            m_compositionVs;
    Rc<DxvkShader>            m_compositionFs;

    uint64_t                  m_frameId      = DXGI_MAX_SWAP_CHAIN_BUFFERS;
    uint32_t                  m_frameLatency = DefaultFrameLatency;
    uint32_t                  m_frameLatencyCap = 0;
    HANDLE                    m_frameLatencyEvent = nullptr;
    Rc<sync::CallbackFence>   m_frameLatencySignal;

    VkColorSpaceKHR           m_colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    VkClearColorValue         m_clearColor = {};

    // blessed: zero-copy-present -- SetGammaControl mirrors whether a
    // non-identity ramp is active, checked by BlessedTryEarlyAcquire
    bool                      m_hasGammaRamp = false;

    double                    m_targetFrameRate = 0.0;

    dxvk::mutex               m_frameStatisticsLock;
    DXGI_VK_FRAME_STATISTICS  m_frameStatistics = { };

    bool                      m_hasHud = false;
    Rc<hud::HudLatencyItem>   m_latencyHud;

    // blessed: zero-copy-present -- the image BlessedTryEarlyAcquire holds
    // until PresentImage (or BlessedReleaseEarlyAcquire) presents it; whether
    // its acquire semaphore is already attached to a command list; and a
    // latch, reset by CreateBackBuffers, for a swap image that cannot take
    // the redirect. See blessed_zero_copy.h.
    bool                      m_zcAcquired = false;
    bool                      m_zcWaitEmitted = false;
    bool                      m_zcMismatch = false;
    PresenterSync             m_zcSync = {};
    Rc<DxvkImage>             m_zcImage;

    // blessed: zero-copy-present, present-mode fix -- the sync interval the
    // last real Present configured on m_presenter. BlessedTryEarlyAcquire
    // declines until it is known, so the swap chain is never created
    // before the game's first Present has set its interval.
    bool                      m_blessedSyncIntervalKnown = false;
    UINT                      m_blessedSyncInterval = 0;

    Rc<DxvkImageView> GetBackBufferView();

    // blessed: zero-copy-present -- flushes, acquires the swap chain image
    // early and attaches its acquire semaphore to the command list that
    // starts here. Returns the acquired image, or null if this frame keeps
    // the normal path.
    Rc<DxvkImage> BlessedTryEarlyAcquire(D3D11ImmediateContext* ctx);

    // blessed: zero-copy-present -- copies the acquired image back into the
    // real back buffer when a redirect falls back mid-frame
    void BlessedCorrectBackBufferForRead(
            D3D11ImmediateContext*  ctx,
      const Rc<DxvkImage>&          swapImage);

    // blessed: zero-copy-present -- presents a held early acquire as is
    void BlessedReleaseEarlyAcquire();

    HRESULT PresentImage(
            UINT                      SyncInterval,
      const DXGI_PRESENT_PARAMETERS*  pPresentParameters);

    void RotateBackBuffers(D3D11ImmediateContext* ctx);

    void CreateFrameLatencyEvent();

    void CreatePresenter();

    void CreateBackBuffers();

    void CreateBlitter();

    void DestroyFrameLatencyEvent();

    void DestroyLatencyTracker();

    void SyncFrameLatency();

    uint32_t GetActualFrameLatency();

    VkSurfaceFormatKHR GetSurfaceFormat(DXGI_FORMAT Format);

    Com<D3D11ReflexDevice> GetReflexDevice();

    VkRect2D ComputeSrcPresentRect() const;

    VkRect2D ComputeDstPresentRect(VkExtent2D DstSize, VkExtent2D SrcSize) const;

    void CompositeIncrementalPresent(
            D3D11ImmediateContext*   pContext,
      const DXGI_PRESENT_PARAMETERS* pPresentParameters);

    bool UseIncrementalPresent(
      const DXGI_PRESENT_PARAMETERS* pPresentParameters) const;

    void CreateCompositionShaders();

    DirtyRectList NormalizeDirtyRects(const DXGI_PRESENT_PARAMETERS* pPresentParameters, VkRect2D Bounds) const;

    void AddDirtyRect(DirtyRectList& List, RECT Rect, VkRect2D Bounds) const;

    std::string GetApiName() const;

  };

}
