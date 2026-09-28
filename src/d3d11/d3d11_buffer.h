#pragma once

#include "../dxvk/dxvk_cs.h"
#include "../dxvk/dxvk_device.h"

#include "../d3d10/d3d10_buffer.h"

#include "d3d11_device_child.h"
#include "d3d11_interfaces.h"
#include "d3d11_on_12.h"
#include "d3d11_resource.h"

namespace dxvk {
  
  class D3D11Device;


  /**
   * \brief Buffer map mode
   */
  enum D3D11_COMMON_BUFFER_MAP_MODE {
    D3D11_COMMON_BUFFER_MAP_MODE_NONE,
    D3D11_COMMON_BUFFER_MAP_MODE_DIRECT,
  };


  /**
   * \brief Stream output buffer offset
   *
   * A byte offset into the buffer that
   * stores the byte offset where new
   * data will be written to.
   */
  struct D3D11SOCounter {
    uint32_t byteOffset;
  };
  
  
  class D3D11Buffer : public D3D11DeviceChild<ID3D11Buffer> {
    static constexpr VkDeviceSize BufferSliceAlignment = 64;
  public:
    
    D3D11Buffer(
            D3D11Device*                pDevice,
      const D3D11_BUFFER_DESC*          pDesc,
      const D3D11_ON_12_RESOURCE_INFO*  p11on12Info);

    ~D3D11Buffer();
    
    HRESULT STDMETHODCALLTYPE QueryInterface(
            REFIID  riid,
            void**  ppvObject) final;
    
    void STDMETHODCALLTYPE GetType(
            D3D11_RESOURCE_DIMENSION *pResourceDimension) final;
    
    UINT STDMETHODCALLTYPE GetEvictionPriority() final;
    
    void STDMETHODCALLTYPE SetEvictionPriority(UINT EvictionPriority) final;
    
    void STDMETHODCALLTYPE GetDesc(
            D3D11_BUFFER_DESC *pDesc) final;
    
    void STDMETHODCALLTYPE SetDebugName(const char* pName) final;

    bool CheckViewCompatibility(
            UINT                BindFlags,
            DXGI_FORMAT         Format) const;

    const D3D11_BUFFER_DESC* Desc() const {
      return &m_desc;
    }

    BOOL IsTilePool() const {
      return bool(m_desc.MiscFlags & D3D11_RESOURCE_MISC_TILE_POOL);
    }

    D3D11_COMMON_BUFFER_MAP_MODE GetMapMode() const {
      return m_mapMode;
    }

    uint64_t GetCookie() const {
      return m_cookie;
    }

    Rc<DxvkBuffer> GetBuffer() const {
      return m_buffer;
    }

    // blessed: perf-halfrate -- the same without the Rc copy (two locked
    // ops) for the per-map ring path; the D3D11Buffer keeps it alive
    DxvkBuffer* BlessedBufferPtr() const {
      return m_buffer.ptr();
    }

    Rc<DxvkSparsePageAllocator> GetSparseAllocator() const {
      return m_sparseAllocator;
    }
    
    DxvkBufferSlice GetBufferSlice() const {
      return DxvkBufferSlice(m_buffer, 0, m_desc.ByteWidth);
    }
    
    DxvkBufferSlice GetBufferSlice(VkDeviceSize offset) const {
      VkDeviceSize size = m_desc.ByteWidth;
      offset = std::min(offset, size);
      return DxvkBufferSlice(m_buffer, offset, size - offset);
    }
    
    DxvkBufferSlice GetBufferSlice(VkDeviceSize offset, VkDeviceSize length) const {
      VkDeviceSize size = m_desc.ByteWidth;
      offset = std::min(offset, size);
      return DxvkBufferSlice(m_buffer, offset, std::min(length, size - offset));
    }

    VkDeviceSize GetRemainingSize(VkDeviceSize offset) const {
      VkDeviceSize size = m_desc.ByteWidth;
      offset = std::min(offset, size);
      return size - offset;
    }

    DxvkBufferSlice GetSOCounter() {
      return m_soCounter != nullptr
        ? DxvkBufferSlice(m_soCounter)
        : DxvkBufferSlice();
    }
    
    Rc<DxvkResourceAllocation> AllocSlice(DxvkLocalAllocationCache* cache) {
      return m_buffer->allocateStorage(cache);
    }
    
    Rc<DxvkResourceAllocation> DiscardSlice(DxvkLocalAllocationCache* cache) {
      auto allocation = m_buffer->allocateStorage(cache);
      m_mapPtr = allocation->mapPtr();

      // blessed: hook-cpu-2 -- see BlessedMappedAllocation
      if (unlikely(m_blessedKeepAllocation))
        m_blessedAllocation = allocation;

      return allocation;
    }

    /**
     * \brief blessed: hook-cpu-2 -- the allocation GetMapPtr points into
     *
     * App thread. Kept only for buffers shaped like the skinned bones
     * cbuffer (dynamic cbuffer, 3,840 bytes, BLESSED_SCENE_SKINNED=1), so
     * scene capture can hand the gpu the bytes a draw saw without reading
     * them on the cpu; null for every other buffer.
     */
    const Rc<DxvkResourceAllocation>& BlessedMappedAllocation() const {
      return m_blessedAllocation;
    }

    void* GetMapPtr() const {
      return m_mapPtr;
    }

    /**
     * \brief blessed: cb-ring -- whether WRITE_DISCARD maps use the ring
     *
     * Decided once at creation: d3d11.blessedCbRing is set, and this is
     * a dynamic, write-only, constant-buffer-only buffer in cached host
     * memory, small enough for a ring chunk, and not one whose allocation
     * scene capture keeps (BlessedMappedAllocation).
     */
    bool BlessedUsesCbRing() const {
      return m_blessedCbRing;
    }

    /**
     * \brief blessed: cb-ring -- points GetMapPtr at a ring chunk
     *
     * App thread, immediate context only, called by the ring map path
     * in place of DiscardSlice.
     */
    void BlessedSetMapPtr(void* mapPtr) {
      m_mapPtr = mapPtr;
    }

    /**
     * rief blessed: threaded-fe -- the app side of the map state
     *
     * With the threaded front end, the game thread renames a buffer on
     * Map(WRITE_DISCARD) long before the front end replays that rename.
     * The game side answers maps from this pointer; GetMapPtr (and the
     * kept allocation) stay the replay side, set when the rename record
     * replays, so every draw-time reader sees the slice its draw saw.
     * Game thread only. Equal to GetMapPtr whenever the ring is empty.
     */
    void* BlessedAppMapPtr() const {
      return m_blessedAppMapPtr;
    }

    void BlessedSetAppMapPtr(void* mapPtr) {
      m_blessedAppMapPtr = mapPtr;
    }

    /**
     * rief blessed: threaded-fe -- replay half of a game-side discard
     *
     * Front end only: what DiscardSlice does to the replay side.
     */
    void BlessedReplayRename(const Rc<DxvkResourceAllocation>& allocation) {
      m_mapPtr = allocation->mapPtr();

      if (unlikely(m_blessedKeepAllocation))
        m_blessedAllocation = allocation;
    }

    D3D10Buffer* GetD3D10Iface() {
      return &m_d3d10;
    }

    bool HasSequenceNumber() const {
      return m_mapMode != D3D11_COMMON_BUFFER_MAP_MODE_NONE
          && !(m_desc.MiscFlags & D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS)
          && !(m_desc.BindFlags);
    }

    void TrackSequenceNumber(uint64_t Seq) {
      m_seq = Seq;
    }

    uint64_t GetSequenceNumber() {
      return HasSequenceNumber() ? m_seq
        : DxvkCsThread::SynchronizeAll;
    }

    /**
     * \brief Retrieves D3D11on12 resource info
     * \returns 11on12 resource info
     */
    D3D11_ON_12_RESOURCE_INFO Get11on12Info() const {
      return m_11on12;
    }

    /**
     * \brief Normalizes buffer description
     * 
     * \param [in] pDesc Buffer description
     * \returns \c S_OK if the parameters are valid
     */
    static HRESULT NormalizeBufferProperties(
            D3D11_BUFFER_DESC*      pDesc);

    /**
     * \brief Initializes D3D11 buffer description from D3D12
     *
     * \param [in] pResource D3D12 resource
     * \param [in] pResourceFlags D3D11 flag overrides
     * \param [out] pBufferDesc D3D11 buffer description
     * \returns \c S_OK if the parameters are valid
     */
    static HRESULT GetDescFromD3D12(
            ID3D12Resource*         pResource,
      const D3D11_RESOURCE_FLAGS*   pResourceFlags,
            D3D11_BUFFER_DESC*      pBufferDesc);

  private:
    
    D3D11_BUFFER_DESC             m_desc;
    D3D11_ON_12_RESOURCE_INFO     m_11on12;
    D3D11_COMMON_BUFFER_MAP_MODE  m_mapMode;
    
    Rc<DxvkBuffer>                m_buffer;
    uint64_t                      m_cookie = 0u;

    Rc<DxvkBuffer>                m_soCounter;
    Rc<DxvkSparsePageAllocator>   m_sparseAllocator;
    uint64_t                      m_seq = 0ull;

    void*                         m_mapPtr = nullptr;

    // blessed: hook-cpu-2 -- see BlessedMappedAllocation
    bool                          m_blessedKeepAllocation = false;
    Rc<DxvkResourceAllocation>    m_blessedAllocation;

    // blessed: threaded-fe-2 -- the game side's map state on a line of its
    // own: the front end writes m_mapPtr (and the refcounts) on replay,
    // and each map would otherwise pull the line back across cores
    // blessed: threaded-fe -- see BlessedAppMapPtr
    alignas(CACHE_LINE_SIZE)
    void*                         m_blessedAppMapPtr = nullptr;

    // blessed: cb-ring -- see BlessedUsesCbRing
    bool                          m_blessedCbRing = false;

    D3D11DXGIResource             m_resource;
    D3D10Buffer                   m_d3d10;

    D3DDestructionNotifier        m_destructionNotifier;

    BOOL CheckFormatFeatureSupport(
            VkFormat              Format,
            VkFormatFeatureFlags2 Features) const;
    
    VkMemoryPropertyFlags GetMemoryFlags() const;

    Rc<DxvkBuffer> CreateSoCounterBuffer();

    static D3D11_COMMON_BUFFER_MAP_MODE DetermineMapMode(
            VkMemoryPropertyFlags MemFlags);

  };


  /**
   * \brief Retrieves buffer from resource pointer
   * 
   * \param [in] pResource The resource to query
   * \returns Pointer to buffer, or \c nullptr
   */
  D3D11Buffer* GetCommonBuffer(
          ID3D11Resource*       pResource);
  
}
