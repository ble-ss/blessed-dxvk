#pragma once

#include "../util/com/com_object.h"

#include "../vulkan/vulkan_loader.h"

#include "dxgi_interfaces.h"

namespace dxvk {

  /**
   * \brief Surface factory
   *
   * Provides a way to transparently create a
   * Vulkan surface for a given platform window.
   */
  // blessed: present-idle -- hands the factory's window to BLESSED_PRESENT=bridge
  class BlessedDxgiWindow : public ComObject<IBlessedDXGIVkWindow> {

  public:

    BlessedDxgiWindow(HWND hWnd) : m_window(hWnd) { }

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppvObject) {
      if (ppvObject == nullptr)
        return E_POINTER;

      *ppvObject = nullptr;

      if (riid == __uuidof(IUnknown) || riid == __uuidof(IBlessedDXGIVkWindow)) {
        *ppvObject = ref(this);
        return S_OK;
      }

      return E_NOINTERFACE;
    }

    HWND STDMETHODCALLTYPE GetWindow() { return m_window; }

  private:

    HWND m_window = nullptr;

  };

  class DxgiSurfaceFactory : public ComObject<IDXGIVkSurfaceFactory> {

  public:

    DxgiSurfaceFactory(
            PFN_vkGetInstanceProcAddr vulkanLoaderProc,
            HWND                      hWnd);

    ~DxgiSurfaceFactory();

    HRESULT STDMETHODCALLTYPE QueryInterface(
            REFIID                    riid,
            void**                    ppvObject);

    VkResult STDMETHODCALLTYPE CreateSurface(
            VkInstance                Instance,
            VkPhysicalDevice          Adapter,
            VkSurfaceKHR*             pSurface);

  private:

    PFN_vkGetInstanceProcAddr m_vkGetInstanceProcAddr = nullptr;
    HWND                      m_window = nullptr;
    bool                      m_ownsWindow = false;

    HWND CreateDummyWindow();

    void DestroyDummyWindow();

  };
  
}
