// blessed: BLESSED_ZERO_COPY -- redirect the frame's back-buffer render-target writes into the acquired swap chain image and skip the present blit
//
// worth about +0.73% in the traversal (c67), roughly the 0.03 ms the blit took.
//
// mechanism: the first time a frame binds (or clears) the tracked back
// buffer as a render target, D3D11SwapChain::BlessedTryEarlyAcquire flushes
// everything recorded so far (non-blocking, no WSI semaphores attached),
// acquires the vulkan swap chain image, and attaches only the acquire
// semaphore to the command list that starts right there. Every back-buffer
// render-target binding and clear for the rest of the frame then uses a view
// of the acquired image with the RTV's own format. PresentImage reuses the
// acquired image, attaches only the present semaphore, and skips the blit.
//
// fallback: any other access to the back buffer this frame (an SRV bind, a
// copy or resolve from or into it, UpdateSubresource, ClearView, Map,
// ExecuteCommandList) stops the redirect for the rest of the frame. If the
// redirect had already begun, the acquired image is first copied back into
// the real back buffer and the framebuffer is rebound, so the access and the
// normal present blit see what was drawn.
//
// threads: everything here runs on whichever thread owns the immediate
// context at that moment (the game thread, or the threaded front end's
// worker); the swap chain's setters drain the front end first.
//
// cost when disabled: one cached bool per hook.
#pragma once

#include "../dxvk/dxvk_image.h"

#include "d3d11_include.h"

namespace dxvk {

  class D3D11SwapChain;
  class D3D11ImmediateContext;

  // blessed: backing flag for BlessedZeroCopy::IsEnabled(), set once at
  // static-init time from BLESSED_ZERO_COPY alone (env only, no cross-TU
  // state), same pattern as blessed_look_detail::g_enabled
  namespace blessed_zero_copy_detail {
    extern const bool g_enabled;
  }

  class BlessedZeroCopy {

  public:

    /**
     * \brief Whether BLESSED_ZERO_COPY=1 is set
     *
     * One inline read of a constant global. BLESSED_LOOK or BLESSED_HOOK_PS
     * keep the feature idle from registration on (SetTrackedSwapChain).
     */
    static bool IsEnabled() {
      return blessed_zero_copy_detail::g_enabled;
    }

    /**
     * \brief Registers the swap chain's current back buffer
     *
     * Called from D3D11SwapChain::CreateBackBuffers. A null image marks the
     * swap chain as never eligible. A second, distinct swap chain turns the
     * feature off for the rest of the process.
     */
    static void SetTrackedSwapChain(
            D3D11SwapChain*         pChain,
            ID3D11Resource*         pResource,
      const Rc<DxvkImage>&          image);

    /**
     * \brief Clears the registration if this is the tracked swap chain
     */
    static void OnSwapChainDestroyed(D3D11SwapChain* pChain);

    /**
     * \brief Render-target redirect for one back-buffer view
     *
     * Called from BindFramebuffer and ClearRenderTargetView (immediate
     * context) for every colour view. Returns a view of the acquired swap
     * chain image to use instead, or null to use \c view unchanged.
     * \param [in] rebind False when called from inside BindFramebuffer
     */
    static Rc<DxvkImageView> RedirectRenderTarget(
            D3D11ImmediateContext*  pContext,
      const Rc<DxvkImageView>&      view,
            bool                    rebind);

    /**
     * \brief Any access to \c pResource other than as a render target
     *
     * No-op unless it is the tracked back buffer. Ends the redirect for the
     * rest of the frame (see the file comment). Must run before the access
     * itself is emitted.
     */
    static void OnResourceAccess(
            D3D11ImmediateContext*  pContext,
            ID3D11Resource*         pResource,
      const char*                   pReason);

    /**
     * \brief Ends the redirect for this frame whatever the resource
     */
    static void OnUnknownAccess(
            D3D11ImmediateContext*  pContext,
      const char*                   pReason);

    /**
     * \brief Whether the blit can be skipped at this frame's present
     */
    static bool IsRedirectActive();

    /**
     * \brief Present boundary: resets all per-frame state
     */
    static void OnPresent();

  private:

    static void Fallback(
            D3D11ImmediateContext*  pContext,
      const char*                   pReason,
            bool                    rebind);

  };

}
