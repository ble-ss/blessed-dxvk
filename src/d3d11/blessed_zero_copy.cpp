// blessed: BLESSED_ZERO_COPY, see blessed_zero_copy.h
#include <string>

#include "blessed_hook.h"
#include "blessed_look.h"
#include "blessed_zero_copy.h"
#include "d3d11_context_imm.h"
#include "d3d11_swapchain.h"

#include "../util/util_env.h"
#include "../util/util_string.h"
#include "../util/log/log.h"

namespace dxvk {

  namespace {

    // Plain pointers only: identities, never owners, so no static
    // destructor releases a dxvk object at process exit. The back buffer
    // lives while registered (the swap chain owns it), the acquired image
    // while the presenter holds it for this frame.
    D3D11SwapChain*   g_chain           = nullptr;
    ID3D11Resource*   g_bbResource      = nullptr;
    const DxvkImage*  g_bbImage         = nullptr;
    bool              g_disabledForever = false;
    bool              g_announced       = false;

    /// acquired swap chain image this frame's redirect writes into
    DxvkImage*        g_redirectImage   = nullptr;
    /// the frame already tried to begin a redirect (success or not)
    bool              g_triedThisFrame  = false;
    /// the frame touched the back buffer other than as a render target
    bool              g_touchedThisFrame = false;

    uint64_t          g_fallbackCount   = 0u;


    void ResetFrame() {
      g_redirectImage    = nullptr;
      g_triedThisFrame   = false;
      g_touchedThisFrame = false;
    }

  }


  namespace blessed_zero_copy_detail {
    extern const bool g_enabled = env::getEnvVar("BLESSED_ZERO_COPY") == "1";
  }


  void BlessedZeroCopy::SetTrackedSwapChain(
          D3D11SwapChain*         pChain,
          ID3D11Resource*         pResource,
    const Rc<DxvkImage>&          image) {
    if (!IsEnabled() || g_disabledForever)
      return;

    if (!g_announced) {
      g_announced = true;

      if (BlessedLook::IsEnabled() || BlessedHook::IsEnabled()) {
        // Both run passes on render target 0 straight through DxvkContext,
        // past the redirect.
        g_disabledForever = true;
        Logger::warn("BlessedZeroCopy: BLESSED_LOOK or BLESSED_HOOK_PS is set, "
          "leaving BLESSED_ZERO_COPY off");
        return;
      }

      Logger::info("BlessedZeroCopy: redirecting back-buffer render targets "
        "into the acquired swap chain image (BLESSED_ZERO_COPY=1)");
    }

    ResetFrame();

    if (g_chain != nullptr && g_chain != pChain) {
      g_disabledForever = true;
      g_chain = nullptr;
      g_bbResource = nullptr;
      g_bbImage = nullptr;

      Logger::warn("BlessedZeroCopy: a second swap chain appeared, "
        "disabling BLESSED_ZERO_COPY for the rest of the process");
      return;
    }

    g_chain = pChain;
    g_bbResource = image != nullptr ? pResource : nullptr;
    g_bbImage = image.ptr();
  }


  void BlessedZeroCopy::OnSwapChainDestroyed(D3D11SwapChain* pChain) {
    if (g_chain != pChain)
      return;

    ResetFrame();

    g_chain = nullptr;
    g_bbResource = nullptr;
    g_bbImage = nullptr;
  }


  Rc<DxvkImageView> BlessedZeroCopy::RedirectRenderTarget(
          D3D11ImmediateContext*  pContext,
    const Rc<DxvkImageView>&      view,
          bool                    rebind) {
    if (view == nullptr || g_bbImage == nullptr || view->image() != g_bbImage)
      return nullptr;

    if (g_touchedThisFrame)
      return nullptr;

    if (g_redirectImage == nullptr) {
      if (g_triedThisFrame || g_chain == nullptr)
        return nullptr;

      g_triedThisFrame = true;
      g_redirectImage = g_chain->BlessedTryEarlyAcquire(pContext).ptr();

      if (g_redirectImage == nullptr)
        return nullptr;
    }

    DxvkImageViewKey key = view->info();

    if (key.mipIndex != 0u || key.layerIndex != 0u
     || !g_redirectImage->isViewCompatible(key.format)) {
      Fallback(pContext, "rtv format or subresource", rebind);
      return nullptr;
    }

    key.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    return g_redirectImage->createView(key);
  }


  void BlessedZeroCopy::OnResourceAccess(
          D3D11ImmediateContext*  pContext,
          ID3D11Resource*         pResource,
    const char*                   pReason) {
    if (pResource == nullptr || pResource != g_bbResource || g_touchedThisFrame)
      return;

    Fallback(pContext, pReason, true);
  }


  void BlessedZeroCopy::OnUnknownAccess(
          D3D11ImmediateContext*  pContext,
    const char*                   pReason) {
    if (g_bbImage == nullptr || g_touchedThisFrame)
      return;

    Fallback(pContext, pReason, true);
  }


  bool BlessedZeroCopy::IsRedirectActive() {
    return g_redirectImage != nullptr && !g_touchedThisFrame;
  }


  void BlessedZeroCopy::OnPresent() {
    ResetFrame();
  }


  void BlessedZeroCopy::Fallback(
          D3D11ImmediateContext*  pContext,
    const char*                   pReason,
          bool                    rebind) {
    bool wasActive = g_redirectImage != nullptr;

    g_touchedThisFrame = true;

    if (!wasActive)
      return;

    // The acquired image holds everything drawn since the redirect began;
    // the real back buffer must see it before the access and before the
    // present blit, which runs as usual from here on.
    g_chain->BlessedCorrectBackBufferForRead(pContext, Rc<DxvkImage>(g_redirectImage));

    // A back-buffer view may still be bound through the acquired image.
    if (rebind) {
      bool bound = false;

      for (const auto& rtv : pContext->m_state.om.rtvs) {
        if (rtv != nullptr && rtv->GetImageView()->image() == g_bbImage)
          bound = true;
      }

      if (bound)
        pContext->BindFramebuffer();
    }

    if (g_fallbackCount++ < 8u) {
      Logger::info(str::format("BlessedZeroCopy: fell back to the present "
        "blit for this frame: ", pReason));
    }
  }

}
