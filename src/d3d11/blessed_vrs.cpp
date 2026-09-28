// blessed: vrs -- scans DXBC pixel shaders for discard and depth / coverage / stencil exports (BLESSED_VRS).
#include <atomic>

#include <dxbc/dxbc_container.h>
#include <dxbc/dxbc_parser.h>

#include "blessed_vrs.h"

#include "../dxvk/dxvk_shader.h"
#include "../dxvk/blessed/blessed_vrs.h"

#include "../util/log/log.h"
#include "../util/util_string.h"

namespace dxvk {

  namespace {

    bool IsFullRateRegister(dxbc_spv::dxbc::RegisterType type) {
      using dxbc_spv::dxbc::RegisterType;

      switch (type) {
        case RegisterType::eDepth:
        case RegisterType::eDepthGe:
        case RegisterType::eDepthLe:
        case RegisterType::eCoverageOut:
        case RegisterType::eCoverageIn:
        case RegisterType::eInnerCoverage:
        case RegisterType::eStencilRef:
          return true;

        default:
          return false;
      }
    }

  }


  void BlessedVrsTagPixelShader(
          DxvkShader*             pShader,
    const void*                   pBytecode,
          size_t                  BytecodeLength) {
    if (BlessedVrs::mode() == BlessedVrsMode::Off || !pShader)
      return;

    bool fullRate = false;
    bool parsed = false;

    try {
      dxbc_spv::dxbc::Container container(pBytecode, BytecodeLength);
      dxbc_spv::dxbc::Parser parser(container.getCodeChunk());

      while (parser && !fullRate) {
        dxbc_spv::dxbc::Instruction op = parser.parseInstruction();

        if (!op)
          break;

        if (op.getOpToken().getOpCode() == dxbc_spv::dxbc::OpCode::eDiscard) {
          fullRate = true;
          break;
        }

        // blessed: declarations and writes both name the register, so any
        // operand of these types is enough
        for (uint32_t i = 0; i < op.getDstCount() && !fullRate; i++)
          fullRate = IsFullRateRegister(op.getDst(i).getRegisterType());

        for (uint32_t i = 0; i < op.getSrcCount() && !fullRate; i++)
          fullRate = IsFullRateRegister(op.getSrc(i).getRegisterType());
      }

      parsed = true;
    } catch (...) {
      parsed = false;
    }

    // blessed: a shader we cannot read stays at 1x1, never the other way
    if (!parsed)
      fullRate = true;

    pShader->blessedSetVrsFullRate(fullRate);

    // blessed: shaders are created from any app thread
    static std::atomic<uint32_t> s_total = { 0u };
    static std::atomic<uint32_t> s_full  = { 0u };

    uint32_t full  = fullRate ? s_full.fetch_add(1u) + 1u : s_full.load();
    uint32_t total = s_total.fetch_add(1u) + 1u;

    if (!(total % 500u)) {
      Logger::info(str::format("blessed: vrs: pixel shaders scanned=", total,
        " full-rate (discard / depth / coverage / stencil)=", full));
    }
  }

}
