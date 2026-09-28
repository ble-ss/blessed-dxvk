// blessed: threaded-fe -- deferred final release: the hook D3D11DeviceChild and D3D11StateObject call when an app's last reference goes
#pragma once

namespace dxvk {

  class D3D11Device;

  /// Runs an object's deferred private release on the front end
  using BlessedReleaseFn = void (*)(void* object);

  /**
   * \brief Set once any device runs a threaded front end
   *
   * Read on every final public release. False unless
   * d3d11.blessedThreadedFrontEnd is set, so the release
   * path costs one predictable branch on a cached bool.
   */
  extern bool g_blessedDeferRelease;

  /**
   * \brief Queues an object's private release behind the ring
   *
   * Called from any thread when the app's public reference count reaches
   * zero. Records already in the ring may still name the object, so its
   * private release runs on the front end after all of them.
   * \returns \c false if the device has no front end; the caller then
   *    releases at once, as upstream does.
   */
  bool BlessedDeferRelease(
          D3D11Device*      pDevice,
          void*             pObject,
          BlessedReleaseFn  pfnRelease);

}
