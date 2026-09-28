#pragma once

#include "d3d11_device_child.h"

#include "../d3d10/d3d10_input_layout.h"

namespace dxvk {
  
  class D3D11Device;

  class D3D11InputLayout : public D3D11DeviceChild<ID3D11InputLayout> {
    
  public:
    
    D3D11InputLayout(
            D3D11Device*          pDevice,
            uint32_t              numAttributes,
      const DxvkVertexAttribute*  pAttributes,
            uint32_t              numBindings,
      const DxvkVertexBinding*    pBindings);
    
    ~D3D11InputLayout();
    
    HRESULT STDMETHODCALLTYPE QueryInterface(
            REFIID                riid,
            void**                ppvObject) final;

    uint32_t GetAttributeCount() const {
      return m_attributeCount;
    }

    uint32_t GetBindingCount() const {
      return m_bindingCount;
    }

    DxvkVertexInput GetInput(uint32_t Index) const {
      return m_inputs[Index];
    }

    // blessed: scene-capture needs to find the POSITION0 stream after
    // dxvk has already thrown the D3D11 semantic strings away (see
    // CreateInputLayout, which is the only place that still has them).
    bool HasBlessedPosition() const {
      return m_blessedPositionValid;
    }

    const DxvkVertexAttribute& GetBlessedPosition() const {
      return m_blessedPosition;
    }

    void SetBlessedPosition(const DxvkVertexAttribute& attribute) {
      m_blessedPosition      = attribute;
      m_blessedPositionValid = true;
    }

    // blessed: actor-skinning needs BLENDINDICES0/BLENDWEIGHT0 the same way
    // scene-capture needs POSITION0 -- see SetBlessedPosition above.
    bool HasBlessedSkinning() const {
      return m_blessedSkinningValid;
    }

    const DxvkVertexAttribute& GetBlessedSkinIndices() const {
      return m_blessedSkinIndices;
    }

    const DxvkVertexAttribute& GetBlessedSkinWeights() const {
      return m_blessedSkinWeights;
    }

    void SetBlessedSkinning(
      const DxvkVertexAttribute& indices,
      const DxvkVertexAttribute& weights) {
      m_blessedSkinIndices   = indices;
      m_blessedSkinWeights   = weights;
      m_blessedSkinningValid = true;
    }

    bool Compare(
      const D3D11InputLayout*     pOther) const;
    
    D3D10InputLayout* GetD3D10Iface() {
      return &m_d3d10;
    }
    
  private:

    uint32_t m_attributeCount = 0;
    uint32_t m_bindingCount = 0;

    std::array<DxvkVertexInput, MaxNumVertexAttributes + MaxNumVertexBindings> m_inputs = { };

    // blessed: see SetBlessedPosition
    bool                m_blessedPositionValid = false;
    DxvkVertexAttribute m_blessedPosition       = { };

    // blessed: see SetBlessedSkinning
    bool                m_blessedSkinningValid = false;
    DxvkVertexAttribute m_blessedSkinIndices    = { };
    DxvkVertexAttribute m_blessedSkinWeights    = { };

    D3D10InputLayout m_d3d10;

    D3DDestructionNotifier m_destructionNotifier;
    
  };
  
}
