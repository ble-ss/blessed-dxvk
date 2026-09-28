// blessed: present-idle -- the d3d12 side of BLESSED_PRESENT=bridge: device, flip swap chain, shared images and fences
#include <array>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <deque>
#include <string>
#include <vector>

#include <d3d12.h>
#include <dxgi1_6.h>

#include "blessed_present_bridge.h"

#include "../dxvk_buffer.h"
#include "../dxvk_context.h"
#include "../dxvk_device.h"

#include "../../util/com/com_pointer.h"
#include "../../util/util_env.h"
#include "../../util/util_string.h"
#include "../../util/thread.h"
#include "../../util/log/log.h"

namespace dxvk {

  namespace {

    using PFN_CreateDXGIFactory2 = HRESULT (WINAPI*)(UINT, REFIID, void**);

    // flip-model swap chains take these four; srgb formats are views only
    DXGI_FORMAT PickFormat(VkFormat format) {
      switch (format) {
        case VK_FORMAT_R16G16B16A16_SFLOAT:
          return DXGI_FORMAT_R16G16B16A16_FLOAT;
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
          return DXGI_FORMAT_R10G10B10A2_UNORM;
        case VK_FORMAT_B8G8R8A8_UNORM:
        case VK_FORMAT_B8G8R8A8_SRGB:
          return DXGI_FORMAT_B8G8R8A8_UNORM;
        default:
          return DXGI_FORMAT_R8G8B8A8_UNORM;
      }
    }

    VkFormat ToVkFormat(DXGI_FORMAT format) {
      switch (format) {
        case DXGI_FORMAT_R16G16B16A16_FLOAT: return VK_FORMAT_R16G16B16A16_SFLOAT;
        case DXGI_FORMAT_R10G10B10A2_UNORM:  return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
        case DXGI_FORMAT_B8G8R8A8_UNORM:     return VK_FORMAT_B8G8R8A8_UNORM;
        default:                             return VK_FORMAT_R8G8B8A8_UNORM;
      }
    }

    uint32_t EnvUint(const char* name, uint32_t def, uint32_t lo, uint32_t hi) {
      std::string v = env::getEnvVar(name);

      if (v.empty())
        return def;

      long n = std::strtol(v.c_str(), nullptr, 10);
      return uint32_t(std::min<long>(std::max<long>(n, lo), hi));
    }

  }


  struct BlessedPresentBridge::Impl {
    DxvkDevice*                         device = nullptr;
    HWND                                window = nullptr;
    bool                                valid  = false;

    HMODULE                             dxgiModule = nullptr;
    HMODULE                             d3d12Module = nullptr;

    Com<IDXGIFactory4>                  factory;
    Com<ID3D12Device>                   d3d12;
    Com<ID3D12CommandQueue>             queue;
    Com<IDXGISwapChain3>                swapchain;
    bool                                tearing = false;
    uint32_t                            bufferCount = 3u;

    // vulkan signals gpuFence (imported as a timeline semaphore) after
    // the blit; the d3d12 queue signals doneFence after each present
    Com<ID3D12Fence>                    gpuFence;
    Rc<DxvkFence>                       vkFence;
    Com<ID3D12Fence>                    doneFence;
    HANDLE                              doneEvent = nullptr;
    uint64_t                            value = 0u;

    VkExtent2D                          extent = { };
    DXGI_FORMAT                         format = DXGI_FORMAT_UNKNOWN;
    bool                                zeroCopy = false;

    std::vector<Rc<DxvkImage>>          images;
    std::vector<uint64_t>               slotValue;
    std::vector<Com<ID3D12Resource>>    shared;      // copy mode only
    std::vector<Com<ID3D12CommandAllocator>> allocators;
    std::vector<Com<ID3D12GraphicsCommandList>> lists; // [bb * n + slot]

    uint64_t                            acquired = 0u;
    dxvk::mutex                         pendingMutex;
    std::deque<std::pair<uint32_t, uint64_t>> pending; // slot, value
    bool                                warnedIndex = false;

    // BLESSED_BRIDGE_WAIT=cpu: a thread waits for the blit on the cpu, then
    // presents, so the d3d12 queue gets no cross-queue gpu wait at all
    struct Job { uint32_t slot; uint64_t value; uint32_t syncInterval; };
    uint64_t                            probeFrame = 0u;
    uint64_t                            presented = 0u;
    bool                                cpuWait = false;
    dxvk::mutex                         jobMutex;
    dxvk::condition_variable            jobCond;
    std::deque<Job>                     jobs;
    bool                                stopping = false;
    HANDLE                              gpuEvent = nullptr;
    dxvk::thread                        worker;

    bool init();
    HRESULT submitFrame(const Job& job, bool gpuWait);
    void runWorker();
    bool configure(VkExtent2D ext, DXGI_FORMAT fmt);
    void releaseBuffers();
    bool shareSwapchainBuffers(std::vector<HANDLE>& handles);
    bool createSharedImages(std::vector<HANDLE>& handles);
    bool recordCopies();
    bool importImages(std::vector<HANDLE>& handles);
    bool verifyImages();
    bool readPixel(ID3D12Resource* resource, uint32_t& pixel);
    void waitDone(uint64_t v);
  };


  bool BlessedPresentBridge::Impl::init() {
    // the game folder's dxgi.dll is dxvk: load the system one by full path
    std::array<wchar_t, MAX_PATH> dir = { };
    UINT len = GetSystemDirectoryW(dir.data(), MAX_PATH);

    if (!len || len >= MAX_PATH - 16u)
      return false;

    std::wstring sys(dir.data(), len);
    dxgiModule  = LoadLibraryExW((sys + L"\\dxgi.dll").c_str(), nullptr, 0);
    d3d12Module = LoadLibraryExW((sys + L"\\d3d12.dll").c_str(), nullptr, 0);

    HMODULE self = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
      reinterpret_cast<LPCWSTR>(&EnvUint), &self);

    if (!dxgiModule || !d3d12Module || dxgiModule == self) {
      Logger::err("blessed: bridge: could not load the system dxgi.dll / d3d12.dll");
      return false;
    }

    auto createFactory = reinterpret_cast<PFN_CreateDXGIFactory2>(GetProcAddress(dxgiModule, "CreateDXGIFactory2"));
    auto createDevice  = reinterpret_cast<PFN_D3D12_CREATE_DEVICE>(GetProcAddress(d3d12Module, "D3D12CreateDevice"));

    if (!createFactory || !createDevice)
      return false;

    if (FAILED(createFactory(0, __uuidof(IDXGIFactory4), reinterpret_cast<void**>(&factory)))) {
      Logger::err("blessed: bridge: CreateDXGIFactory2 failed");
      return false;
    }

    const auto& vk11 = device->properties().vk11;

    if (!vk11.deviceLUIDValid)
      return false;

    LUID luid;
    std::memcpy(&luid, vk11.deviceLUID, sizeof(luid));

    Com<IDXGIAdapter1> adapter;

    if (FAILED(factory->EnumAdapterByLuid(luid, __uuidof(IDXGIAdapter1), reinterpret_cast<void**>(&adapter)))
     || FAILED(createDevice(adapter.ptr(), D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), reinterpret_cast<void**>(&d3d12)))) {
      Logger::err("blessed: bridge: no d3d12 device on the vulkan adapter");
      return false;
    }

    D3D12_COMMAND_QUEUE_DESC queueDesc = { };
    queueDesc.Type     = D3D12_COMMAND_LIST_TYPE_DIRECT;
    queueDesc.Priority = env::getEnvVar("BLESSED_BRIDGE_PRIORITY") == "high"
      ? D3D12_COMMAND_QUEUE_PRIORITY_HIGH : D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;

    if (FAILED(d3d12->CreateCommandQueue(&queueDesc, __uuidof(ID3D12CommandQueue), reinterpret_cast<void**>(&queue))))
      return false;

    // the fence vulkan signals: shared, imported as a timeline semaphore
    HANDLE fenceHandle = nullptr;

    if (FAILED(d3d12->CreateFence(0, D3D12_FENCE_FLAG_SHARED, __uuidof(ID3D12Fence), reinterpret_cast<void**>(&gpuFence)))
     || FAILED(d3d12->CreateSharedHandle(gpuFence.ptr(), nullptr, GENERIC_ALL, nullptr, &fenceHandle))) {
      Logger::err("blessed: bridge: shared fence failed");
      return false;
    }

    try {
      DxvkFenceCreateInfo fenceInfo = { };
      fenceInfo.initialValue = 0u;
      fenceInfo.sharedType   = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT;
      fenceInfo.sharedHandle = fenceHandle;
      vkFence = device->createFence(fenceInfo);
    } catch (const DxvkError& e) {
      Logger::err(str::format("blessed: bridge: ", e.message()));
    }

    CloseHandle(fenceHandle);

    if (vkFence == nullptr)
      return false;

    if (FAILED(d3d12->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), reinterpret_cast<void**>(&doneFence))))
      return false;

    doneEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);

    Com<IDXGIFactory5> factory5;
    BOOL allowTearing = FALSE;

    if (SUCCEEDED(factory->QueryInterface(__uuidof(IDXGIFactory5), reinterpret_cast<void**>(&factory5)))
     && SUCCEEDED(factory5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &allowTearing, sizeof(allowTearing))))
      tearing = allowTearing;

    bufferCount = EnvUint("BLESSED_BRIDGE_BUFFERS", 3u, 2u, 8u);

    cpuWait = env::getEnvVar("BLESSED_BRIDGE_WAIT") == "cpu";
    probeFrame = EnvUint("BLESSED_BRIDGE_PROBE", 0u, 0u, 1000000u);

    if (cpuWait) {
      gpuEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
      worker = dxvk::thread([this] { runWorker(); });
    }

    Logger::info(str::format("blessed: bridge: d3d12 device up, ", bufferCount,
      " buffers, tearing ", tearing ? "yes" : "no", ", wait ", cpuWait ? "cpu thread" : "gpu"));
    return true;
  }


  void BlessedPresentBridge::Impl::waitDone(uint64_t v) {
    if (!v || doneFence->GetCompletedValue() >= v)
      return;

    doneFence->SetEventOnCompletion(v, doneEvent);
    WaitForSingleObject(doneEvent, INFINITE);
  }


  void BlessedPresentBridge::Impl::releaseBuffers() {
    // no vulkan or d3d12 work may still touch the buffers
    device->waitForIdle();
    waitDone(value);

    images.clear();
    shared.clear();
    lists.clear();
    allocators.clear();
    slotValue.clear();
  }


  bool BlessedPresentBridge::Impl::shareSwapchainBuffers(std::vector<HANDLE>& handles) {
    if (env::getEnvVar("BLESSED_BRIDGE_COPY") == "1")
      return false;

    for (uint32_t i = 0; i < bufferCount; i++) {
      Com<ID3D12Resource> buffer;
      HANDLE handle = nullptr;

      if (FAILED(swapchain->GetBuffer(i, __uuidof(ID3D12Resource), reinterpret_cast<void**>(&buffer)))
       || FAILED(d3d12->CreateSharedHandle(buffer.ptr(), nullptr, GENERIC_ALL, nullptr, &handle))) {
        for (HANDLE h : handles)
          CloseHandle(h);

        handles.clear();
        return false;
      }

      handles.push_back(handle);
    }

    return true;
  }


  bool BlessedPresentBridge::Impl::createSharedImages(std::vector<HANDLE>& handles) {
    D3D12_HEAP_PROPERTIES heap = { };
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC desc = { };
    desc.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width            = extent.width;
    desc.Height           = extent.height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels        = 1;
    desc.Format           = format;
    desc.SampleDesc.Count = 1;
    desc.Layout           = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags            = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;

    for (uint32_t i = 0; i < bufferCount; i++) {
      Com<ID3D12Resource> resource;
      HANDLE handle = nullptr;

      if (FAILED(d3d12->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &desc,
            D3D12_RESOURCE_STATE_COMMON, nullptr, __uuidof(ID3D12Resource), reinterpret_cast<void**>(&resource)))
       || FAILED(d3d12->CreateSharedHandle(resource.ptr(), nullptr, GENERIC_ALL, nullptr, &handle)))
        return false;

      shared.push_back(resource);
      handles.push_back(handle);
    }

    return recordCopies();
  }


  bool BlessedPresentBridge::Impl::recordCopies() {
    for (uint32_t bb = 0; bb < bufferCount; bb++) {
      Com<ID3D12Resource> buffer;

      if (FAILED(swapchain->GetBuffer(bb, __uuidof(ID3D12Resource), reinterpret_cast<void**>(&buffer))))
        return false;

      for (uint32_t s = 0; s < bufferCount; s++) {
        Com<ID3D12CommandAllocator> alloc;
        Com<ID3D12GraphicsCommandList> list;

        if (FAILED(d3d12->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(ID3D12CommandAllocator), reinterpret_cast<void**>(&alloc)))
         || FAILED(d3d12->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.ptr(), nullptr, __uuidof(ID3D12GraphicsCommandList), reinterpret_cast<void**>(&list))))
          return false;

        std::array<D3D12_RESOURCE_BARRIER, 2> barriers = { };

        for (auto& b : barriers) {
          b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
          b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        }

        barriers[0].Transition.pResource   = buffer.ptr();
        barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
        barriers[0].Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_DEST;
        barriers[1].Transition.pResource   = shared[s].ptr();
        barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
        barriers[1].Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_SOURCE;
        list->ResourceBarrier(2, barriers.data());

        list->CopyResource(buffer.ptr(), shared[s].ptr());

        for (auto& b : barriers)
          std::swap(b.Transition.StateBefore, b.Transition.StateAfter);

        list->ResourceBarrier(2, barriers.data());

        if (FAILED(list->Close()))
          return false;

        allocators.push_back(alloc);
        lists.push_back(list);
      }
    }

    return true;
  }


  bool BlessedPresentBridge::Impl::importImages(std::vector<HANDLE>& handles) {
    bool ok = handles.size() == bufferCount;

    // dxvk quietly makes a plain, unshared image when this query fails
    DxvkFormatQuery query = { };
    query.format     = ToVkFormat(format);
    query.type       = VK_IMAGE_TYPE_2D;
    query.tiling     = VK_IMAGE_TILING_OPTIMAL;
    query.usage      = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    query.flags      = 0u;
    query.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT;

    auto limits = device->getFormatLimits(query);

    if (!limits || !(limits->externalFeatures & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT)) {
      Logger::err(str::format("blessed: bridge: the driver cannot import d3d12 resources of format ",
        query.format, " (external features ", limits ? limits->externalFeatures : 0u, ")"));
      ok = false;
    } else {
      Logger::info(str::format("blessed: bridge: d3d12 resource import supported, external features 0x",
        std::hex, limits->externalFeatures, std::dec));
    }

    for (uint32_t i = 0; i < handles.size(); i++) {
      std::string debugName = str::format("blessed bridge image ", i);

      DxvkImageCreateInfo info = { };
      info.type        = VK_IMAGE_TYPE_2D;
      info.format      = ToVkFormat(format);
      info.sampleCount = VK_SAMPLE_COUNT_1_BIT;
      info.extent      = { extent.width, extent.height, 1u };
      info.numLayers   = 1u;
      info.mipLevels   = 1u;
      info.usage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
      info.stages      = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
      info.access      = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
      info.tiling      = VK_IMAGE_TILING_OPTIMAL;
      info.layout      = VK_IMAGE_LAYOUT_GENERAL;
      info.colorSpace  = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
      info.shared      = VK_TRUE;
      info.sharing.mode   = DxvkSharedHandleMode::Import;
      info.sharing.type   = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT;
      info.sharing.handle = handles[i];
      info.debugName   = debugName.c_str();

      try {
        Rc<DxvkImage> image = device->createImage(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        images.push_back(std::move(image));
      } catch (const DxvkError& e) {
        Logger::err(str::format("blessed: bridge: buffer ", i, " import failed: ", e.message()));
        ok = false;
      }

      CloseHandle(handles[i]);
    }

    handles.clear();

    if (!ok || images.size() != bufferCount)
      return false;

    return verifyImages();
  }


  bool BlessedPresentBridge::Impl::readPixel(ID3D12Resource* resource, uint32_t& pixel) {
    D3D12_HEAP_PROPERTIES heap = { };
    heap.Type = D3D12_HEAP_TYPE_READBACK;

    D3D12_RESOURCE_DESC desc = { };
    desc.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width            = 256;
    desc.Height           = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels        = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    Com<ID3D12Resource> readback;
    Com<ID3D12CommandAllocator> alloc;
    Com<ID3D12GraphicsCommandList> list;

    if (FAILED(d3d12->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
          D3D12_RESOURCE_STATE_COPY_DEST, nullptr, __uuidof(ID3D12Resource), reinterpret_cast<void**>(&readback)))
     || FAILED(d3d12->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(ID3D12CommandAllocator), reinterpret_cast<void**>(&alloc)))
     || FAILED(d3d12->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.ptr(), nullptr, __uuidof(ID3D12GraphicsCommandList), reinterpret_cast<void**>(&list))))
      return false;

    D3D12_RESOURCE_BARRIER barrier = { };
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource   = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_SOURCE;
    list->ResourceBarrier(1, &barrier);

    D3D12_TEXTURE_COPY_LOCATION dst = { };
    dst.pResource = readback.ptr();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint.Footprint.Format   = format;
    dst.PlacedFootprint.Footprint.Width    = 1;
    dst.PlacedFootprint.Footprint.Height   = 1;
    dst.PlacedFootprint.Footprint.Depth    = 1;
    dst.PlacedFootprint.Footprint.RowPitch = 256;

    D3D12_TEXTURE_COPY_LOCATION src = { };
    src.pResource = resource;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.SubresourceIndex = 0;

    D3D12_BOX box = { extent.width / 2u, extent.height / 2u, 0u, extent.width / 2u + 1u, extent.height / 2u + 1u, 1u };
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, &box);

    std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
    list->ResourceBarrier(1, &barrier);

    if (FAILED(list->Close()))
      return false;

    ID3D12CommandList* lists[] = { list.ptr() };
    queue->ExecuteCommandLists(1, lists);

    uint64_t v = ++value;
    queue->Signal(doneFence.ptr(), v);
    waitDone(v);

    void* data = nullptr;
    D3D12_RANGE range = { 0, 4 };

    if (FAILED(readback->Map(0, &range, &data)))
      return false;

    std::memcpy(&pixel, data, sizeof(pixel));

    D3D12_RANGE none = { 0, 0 };
    readback->Unmap(0, &none);
    return true;
  }


  bool BlessedPresentBridge::Impl::verifyImages() {
    // 8-bit formats only; the others are trusted as imported
    if (format != DXGI_FORMAT_R8G8B8A8_UNORM && format != DXGI_FORMAT_B8G8R8A8_UNORM)
      return true;

    // vulkan writes one texel per buffer, each its own colour, with a plain
    // buffer-to-image copy. not a clear: a vulkan fast clear lives in the
    // compression metadata and d3d12 does not see it (proven in pi-test:
    // cleared buffers read back 0, copied texels and drawn frames match)
    Rc<DxvkContext> ctx = device->createContext();
    ctx->beginRecording(device->createCommandList());

    DxvkBufferCreateInfo bufferInfo = { };
    bufferInfo.size   = 4u * images.size();
    bufferInfo.usage  = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
    bufferInfo.access = VK_ACCESS_TRANSFER_READ_BIT;

    Rc<DxvkBuffer> texels = device->createBuffer(bufferInfo,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    for (uint32_t i = 0; i < images.size(); i++) {
      uint32_t r = 16u * (i + 1u), g = 200u, b = 40u;
      uint32_t texel = images[i]->info().format == VK_FORMAT_R8G8B8A8_UNORM
        ? (r | (g << 8) | (b << 16) | (255u << 24))
        : (b | (g << 8) | (r << 16) | (255u << 24));
      std::memcpy(reinterpret_cast<char*>(texels->mapPtr(0)) + 4u * i, &texel, 4u);

      ctx->copyBufferToImage(images[i], VkImageSubresourceLayers { VK_IMAGE_ASPECT_COLOR_BIT, 0u, 0u, 1u },
        VkOffset3D { int32_t(extent.width / 2u), int32_t(extent.height / 2u), 0 }, VkExtent3D { 1u, 1u, 1u },
        texels, 4u * i, 0u, 0u, images[i]->info().format);
    }

    ctx->flushCommandList(nullptr, nullptr);
    device->waitForIdle();

    // ... and d3d12 must read back exactly that colour
    for (uint32_t i = 0; i < images.size(); i++) {
      ID3D12Resource* resource = nullptr;
      Com<ID3D12Resource> buffer;

      if (zeroCopy) {
        if (FAILED(swapchain->GetBuffer(i, __uuidof(ID3D12Resource), reinterpret_cast<void**>(&buffer))))
          return false;

        resource = buffer.ptr();
      } else {
        resource = shared[i].ptr();
      }

      uint32_t pixel = 0u;

      if (!readPixel(resource, pixel))
        return false;

      uint32_t r = 16u * (i + 1u), g = 200u, b = 40u;
      uint32_t expected = format == DXGI_FORMAT_R8G8B8A8_UNORM
        ? (r | (g << 8) | (b << 16) | (255u << 24))
        : (b | (g << 8) | (r << 16) | (255u << 24));

      if (pixel != expected) {
        Logger::err(str::format("blessed: bridge: buffer ", i, " reads back 0x", std::hex, pixel,
          " instead of 0x", expected, std::dec, ": the import does not share memory"));
        return false;
      }
    }

    // test hook: pretend the readback failed, to exercise the fallbacks
    if (env::getEnvVar("BLESSED_BRIDGE_VERIFY_FAIL") == "1") {
      Logger::warn("blessed: bridge: BLESSED_BRIDGE_VERIFY_FAIL=1, treating the readback as failed");
      return false;
    }

    Logger::info(str::format("blessed: bridge: ", images.size(), " buffers verified by readback"));
    return true;
  }


  bool BlessedPresentBridge::Impl::configure(VkExtent2D ext, DXGI_FORMAT fmt) {
    if (swapchain != nullptr && ext == extent && fmt == format)
      return true;

    releaseBuffers();

    extent = ext;
    format = fmt;

    UINT flags = tearing ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0u;

    if (swapchain == nullptr) {
      DXGI_SWAP_CHAIN_DESC1 desc = { };
      desc.Width       = extent.width;
      desc.Height      = extent.height;
      desc.Format      = format;
      desc.SampleDesc.Count = 1;
      desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
      desc.BufferCount = bufferCount;
      desc.Scaling     = DXGI_SCALING_STRETCH;
      desc.SwapEffect  = DXGI_SWAP_EFFECT_FLIP_DISCARD;
      desc.AlphaMode   = DXGI_ALPHA_MODE_IGNORE;
      desc.Flags       = flags;

      Com<IDXGISwapChain1> swapchain1;
      HRESULT hr = factory->CreateSwapChainForHwnd(queue.ptr(), window, &desc, nullptr, nullptr, &swapchain1);

      if (FAILED(hr) || FAILED(swapchain1->QueryInterface(__uuidof(IDXGISwapChain3), reinterpret_cast<void**>(&swapchain)))) {
        Logger::err(str::format("blessed: bridge: CreateSwapChainForHwnd failed: ", hr));
        return false;
      }

      // dxvk's own dxgi owns alt+enter and window changes
      factory->MakeWindowAssociation(window, DXGI_MWA_NO_ALT_ENTER | DXGI_MWA_NO_WINDOW_CHANGES);
    } else {
      HRESULT hr = swapchain->ResizeBuffers(bufferCount, extent.width, extent.height, format, flags);

      if (FAILED(hr)) {
        Logger::err(str::format("blessed: bridge: ResizeBuffers failed: ", hr));
        return false;
      }
    }

    // zero-copy first; any buffer that fails to import or to verify
    // drops us to copy mode, and a failed copy mode drops the bridge
    std::vector<HANDLE> handles;
    zeroCopy = shareSwapchainBuffers(handles);

    if (zeroCopy && !importImages(handles)) {
      Logger::warn("blessed: bridge: zero-copy import failed, trying copy mode");
      images.clear();
      zeroCopy = false;
    }

    if (!zeroCopy) {
      if (!createSharedImages(handles) || !importImages(handles)) {
        Logger::err("blessed: bridge: copy-mode import failed, the bridge stays off");

        for (HANDLE h : handles)
          CloseHandle(h);

        images.clear();
        return false;
      }
    }

    slotValue.assign(bufferCount, 0u);
    acquired = 0u;

    Logger::info(str::format("blessed: bridge: ", extent.width, "x", extent.height,
      " format ", uint32_t(format), zeroCopy ? ", zero-copy (swap chain buffers shared)" : ", copy mode"));
    return true;
  }


  BlessedPresentBridge::BlessedPresentBridge(DxvkDevice* device, HWND window)
  : m(std::make_unique<Impl>()) {
    m->device = device;
    m->window = window;
    m->valid  = window && m->init();
  }


  BlessedPresentBridge::~BlessedPresentBridge() {
    if (m->worker.joinable()) {
      { std::lock_guard lock(m->jobMutex);
        m->stopping = true;
        m->jobCond.notify_one();
      }

      m->worker.join();
    }

    if (m->gpuEvent)
      CloseHandle(m->gpuEvent);

    if (m->valid)
      m->releaseBuffers();

    m->swapchain = nullptr;
    m->queue = nullptr;

    if (m->doneEvent)
      CloseHandle(m->doneEvent);
  }


  bool BlessedPresentBridge::valid() const {
    return m->valid;
  }


  VkResult BlessedPresentBridge::acquire(
          VkExtent2D          extent,
          VkSurfaceFormatKHR  format,
          PresenterSync&      sync,
          Rc<DxvkImage>&      image) {
    if (!m->valid || !m->configure(extent, PickFormat(format.format)))
      return VK_ERROR_SURFACE_LOST_KHR;

    uint32_t slot = uint32_t(m->acquired++ % m->bufferCount);

    // the only wait: this buffer's previous present must have left the d3d12 queue
    m->waitDone(m->slotValue[slot]);

    uint64_t v = ++m->value;
    m->slotValue[slot] = v;

    { std::lock_guard lock(m->pendingMutex);
      m->pending.emplace_back(slot, v);
    }

    sync = PresenterSync();
    sync.present = m->vkFence->handle();
    sync.blessedPresentValue = v;

    image = m->images[slot];
    return VK_SUCCESS;
  }


  VkResult BlessedPresentBridge::present(uint32_t syncInterval) {
    std::pair<uint32_t, uint64_t> next;

    { std::lock_guard lock(m->pendingMutex);

      if (m->pending.empty())
        return VK_ERROR_OUT_OF_DATE_KHR;

      next = m->pending.front();
      m->pending.pop_front();
    }

    Impl::Job job = { next.first, next.second, syncInterval };

    if (m->cpuWait) {
      std::lock_guard lock(m->jobMutex);
      m->jobs.push_back(job);
      m->jobCond.notify_one();
      return VK_SUCCESS;
    }

    HRESULT hr = m->submitFrame(job, true);

    if (FAILED(hr)) {
      Logger::err(str::format("blessed: bridge: Present failed: ", hr));
      return VK_ERROR_SURFACE_LOST_KHR;
    }

    return VK_SUCCESS;
  }


  HRESULT BlessedPresentBridge::Impl::submitFrame(const Job& job, bool gpuWait) {
    // gpu-side wait on the d3d12 queue only; vulkan has already moved on
    if (gpuWait)
      queue->Wait(gpuFence.ptr(), job.value);

    uint32_t bb = swapchain->GetCurrentBackBufferIndex();

    if (zeroCopy) {
      if (bb != job.slot && !warnedIndex) {
        warnedIndex = true;
        Logger::warn(str::format("blessed: bridge: back buffer ", bb, " != slot ", job.slot));
      }
    } else {
      ID3D12CommandList* list = lists[bb * bufferCount + job.slot].ptr();
      queue->ExecuteCommandLists(1, &list);
    }

    UINT flags = (!job.syncInterval && tearing) ? DXGI_PRESENT_ALLOW_TEARING : 0u;
    HRESULT hr = swapchain->Present(std::min(job.syncInterval, 4u), flags);

    queue->Signal(doneFence.ptr(), job.value);

    // BLESSED_BRIDGE_PROBE=N (diagnostic): the centre pixel d3d12 sees in
    // the N-th presented frame, to compare with what the app drew
    if (unlikely(probeFrame) && ++presented == probeFrame) {
      waitDone(job.value);

      Com<ID3D12Resource> buffer;
      ID3D12Resource* resource = zeroCopy ? nullptr : shared[job.slot].ptr();

      if (zeroCopy && SUCCEEDED(swapchain->GetBuffer(job.slot, __uuidof(ID3D12Resource), reinterpret_cast<void**>(&buffer))))
        resource = buffer.ptr();

      uint32_t pixel = 0u;

      if (resource && readPixel(resource, pixel))
        Logger::info(str::format("blessed: bridge: probe frame ", probeFrame, " centre pixel 0x", std::hex, pixel, std::dec));
    }

    return hr;
  }


  void BlessedPresentBridge::Impl::runWorker() {
    env::setThreadName("blessed-bridge");

    while (true) {
      Job job;

      { std::unique_lock lock(jobMutex);
        jobCond.wait(lock, [this] { return stopping || !jobs.empty(); });

        if (jobs.empty())
          return;

        job = jobs.front();
        jobs.pop_front();
      }

      if (gpuFence->GetCompletedValue() < job.value) {
        gpuFence->SetEventOnCompletion(job.value, gpuEvent);
        WaitForSingleObject(gpuEvent, INFINITE);
      }

      HRESULT hr = submitFrame(job, false);

      if (FAILED(hr))
        Logger::err(str::format("blessed: bridge: Present failed: ", hr));
    }
  }


  void BlessedPresentBridge::waitIdle() {
    if (m->valid)
      m->waitDone(m->value);
  }

}
