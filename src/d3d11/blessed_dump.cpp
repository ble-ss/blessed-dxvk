// blessed: per-draw frame dumper for offline pass/shader classification (BLESSED_PROBE_DIR)
#include "blessed_dump.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <unordered_set>
#include <vector>

#include "d3d11_buffer.h"
#include "d3d11_texture.h"

#include "../util/util_env.h"
#include "../util/log/log.h"

namespace dxvk {

  // blessed: definitions for the two flags declared extern in blessed_dump.h
  // (see the comment there). g_enabled's initializer runs once at static-init
  // time (dll load), same as blessed::g_enabled in util_blessed_probe.cpp, so
  // every later read is a plain load -- no magic-static guard.
  namespace blessed_dump_detail {
    bool g_capturing = false;
  }

  namespace {

    // blessed: minimal single-purpose json writer (no external deps in tree)
    class JsonWriter {
    public:
      explicit JsonWriter(std::string& out) : m_out(out) {
        m_needComma.push_back(false);
      }

      void objBegin() { comma(); m_out += '{'; m_needComma.push_back(false); }
      void objEnd()   { m_out += '}'; m_needComma.pop_back(); m_needComma.back() = true; }
      void arrBegin() { comma(); m_out += '['; m_needComma.push_back(false); }
      void arrEnd()   { m_out += ']'; m_needComma.pop_back(); m_needComma.back() = true; }

      void key(const char* k) {
        comma();
        m_out += '"'; m_out += k; m_out += "\":";
        m_needComma.back() = false;
      }

      void val(std::nullptr_t) { comma(); m_out += "null"; m_needComma.back() = true; }
      void val(bool v)         { comma(); m_out += v ? "true" : "false"; m_needComma.back() = true; }

      void val(int32_t v)  { comma(); m_out += std::to_string(v); m_needComma.back() = true; }
      void val(uint32_t v) { comma(); m_out += std::to_string(v); m_needComma.back() = true; }
      void val(int64_t v)  { comma(); m_out += std::to_string(v); m_needComma.back() = true; }
      void val(uint64_t v) { comma(); m_out += std::to_string(v); m_needComma.back() = true; }

      void val(float f) {
        comma();
        if (!std::isfinite(f)) {
          m_out += "null";
        } else {
          char buf[32];
          std::snprintf(buf, sizeof(buf), "%.5g", double(f));
          m_out += buf;
        }
        m_needComma.back() = true;
      }

      void val(const std::string& s) {
        comma();
        m_out += '"';
        for (char c : s) {
          switch (c) {
            case '"':  m_out += "\\\""; break;
            case '\\': m_out += "\\\\"; break;
            case '\n': m_out += "\\n"; break;
            case '\r': m_out += "\\r"; break;
            case '\t': m_out += "\\t"; break;
            default:
              if (static_cast<unsigned char>(c) < 0x20) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                m_out += buf;
              } else {
                m_out += c;
              }
          }
        }
        m_out += '"';
        m_needComma.back() = true;
      }

      void val(const char* s) { val(std::string(s)); }

      void valHex(uint64_t p) {
        comma();
        char buf[24];
        std::snprintf(buf, sizeof(buf), "\"0x%llx\"", (unsigned long long) p);
        m_out += buf;
        m_needComma.back() = true;
      }

    private:
      void comma() { if (m_needComma.back()) m_out += ','; }

      std::string&      m_out;
      std::vector<bool>  m_needComma;
    };

    struct BlessedViewInfo {
      uint64_t     res  = 0;
      DXGI_FORMAT  fmt  = DXGI_FORMAT_UNKNOWN;
      const char*  dim  = "unknown";
      uint32_t     w    = 0;
      uint32_t     h    = 0;
      uint32_t     mip  = 0;
      uint32_t     arr  = 1;
      uint32_t     slice = 0; // blessed: first array slice (point-light shadow maps are Texture2DArray slices)
    };

    void FillDim(const D3D11_VK_VIEW_INFO& vi, BlessedViewInfo& info) {
      info.res = uint64_t(vi.pResource);

      switch (vi.Dimension) {
        case D3D11_RESOURCE_DIMENSION_BUFFER:
          info.dim = "buffer";
          info.w   = uint32_t(vi.Buffer.Length);
          info.h   = 0;
          info.mip = 0;
          info.arr = 1;
          break;

        case D3D11_RESOURCE_DIMENSION_TEXTURE1D:
        case D3D11_RESOURCE_DIMENSION_TEXTURE2D:
        case D3D11_RESOURCE_DIMENSION_TEXTURE3D: {
          info.dim = vi.Dimension == D3D11_RESOURCE_DIMENSION_TEXTURE1D ? "tex1d"
                   : vi.Dimension == D3D11_RESOURCE_DIMENSION_TEXTURE2D ? "tex2d" : "tex3d";
          info.mip = vi.Image.MinLevel;
          info.arr = vi.Image.NumLayers;
          info.slice = vi.Image.MinLayer;

          if (D3D11CommonTexture* tex = GetCommonTexture(vi.pResource)) {
            VkExtent3D ext = tex->MipLevelExtent(vi.Image.MinLevel);
            info.w = ext.width;
            info.h = ext.height;
          }
          break;
        }

        default:
          break;
      }
    }

    // blessed: shared shape across rtv/dsv/srv/uav's GetViewInfo()+GetDesc()
    template<typename View, typename Desc>
    BlessedViewInfo GetViewInfoT(View* view) {
      BlessedViewInfo info;
      FillDim(view->GetViewInfo(), info);

      Desc desc = { };
      view->GetDesc(&desc);
      info.fmt = desc.Format;
      return info;
    }

    void WriteViewInfo(JsonWriter& j, const BlessedViewInfo& info) {
      j.key("res"); j.valHex(info.res);
      j.key("fmt"); j.val(int32_t(info.fmt));
      j.key("dim"); j.val(info.dim);
      j.key("w");   j.val(info.w);
      j.key("h");   j.val(info.h);
    }

    // ---- capture state (single immediate context, no locking beyond
    // the D3D10DeviceLock already held by every call site we hook into) ----

    std::string                    g_dir;
    std::vector<uint64_t>          g_dumpAt;

    uint64_t                       g_presentIndex        = 0;
    uint32_t                       g_presentsSinceCheck   = 0;

    uint64_t                       g_captureIdx  = 0;
    uint64_t                       g_seq         = 0;

    std::ofstream                  g_frameFile;
    std::ofstream                  g_resFile;
    std::unordered_set<uint64_t>   g_seenResources;

    void ParseDumpAt(const std::string& s) {
      std::string cur;
      for (size_t i = 0; i <= s.size(); i++) {
        if (i == s.size() || s[i] == ',') {
          if (!cur.empty())
            g_dumpAt.push_back(std::strtoull(cur.c_str(), nullptr, 10));
          cur.clear();
        } else {
          cur += s[i];
        }
      }
    }

    void WriteResourceLine(ID3D11Resource* pResource) {
      if (!pResource || !g_resFile.is_open())
        return;

      uint64_t id = uint64_t(pResource);

      if (!g_seenResources.insert(id).second)
        return;

      D3D11_RESOURCE_DIMENSION dim = D3D11_RESOURCE_DIMENSION_UNKNOWN;
      pResource->GetType(&dim);

      std::string line;
      JsonWriter j(line);
      j.objBegin();
      j.key("id"); j.valHex(id);

      if (dim == D3D11_RESOURCE_DIMENSION_BUFFER) {
        auto* buf = static_cast<D3D11Buffer*>(pResource);
        const D3D11_BUFFER_DESC* desc = buf->Desc();

        j.key("type");       j.val("buffer");
        j.key("byte_width"); j.val(uint32_t(desc->ByteWidth));
        j.key("usage");      j.val(int32_t(desc->Usage));
        j.key("bind_flags"); j.val(int32_t(desc->BindFlags));
        j.key("cpu_access"); j.val(int32_t(desc->CPUAccessFlags));
        j.key("misc_flags"); j.val(int32_t(desc->MiscFlags));
      } else if (D3D11CommonTexture* tex = GetCommonTexture(pResource)) {
        const D3D11_COMMON_TEXTURE_DESC* desc = tex->Desc();

        j.key("type"); j.val(
          dim == D3D11_RESOURCE_DIMENSION_TEXTURE1D ? "tex1d" :
          dim == D3D11_RESOURCE_DIMENSION_TEXTURE2D ? "tex2d" :
          dim == D3D11_RESOURCE_DIMENSION_TEXTURE3D ? "tex3d" : "unknown");
        j.key("w");           j.val(desc->Width);
        j.key("h");           j.val(desc->Height);
        j.key("depth");       j.val(desc->Depth);
        j.key("mips");        j.val(desc->MipLevels);
        j.key("array");       j.val(desc->ArraySize);
        j.key("fmt");         j.val(int32_t(desc->Format));
        j.key("usage");       j.val(int32_t(desc->Usage));
        j.key("bind_flags");  j.val(int32_t(desc->BindFlags));
        j.key("cpu_access");  j.val(int32_t(desc->CPUAccessFlags));
        j.key("misc_flags");  j.val(int32_t(desc->MiscFlags));
      } else {
        j.key("type"); j.val("unknown");
      }

      j.objEnd();
      g_resFile << line << '\n';
    }

    void TouchResource(ID3D11Resource* pResource) {
      if (pResource)
        WriteResourceLine(pResource);
    }

    void TouchBuffer(D3D11Buffer* pBuffer) {
      if (pBuffer)
        WriteResourceLine(pBuffer);
    }

    void StartCapture(uint64_t index) {
      std::error_code ec;
      std::filesystem::create_directories(g_dir, ec);

      std::string framePath = g_dir + "/frame-" + std::to_string(index) + ".jsonl";
      std::string resPath   = g_dir + "/frame-" + std::to_string(index) + "-resources.jsonl";

      g_frameFile.open(framePath, std::ios::out | std::ios::trunc);
      g_resFile.open(resPath, std::ios::out | std::ios::trunc);

      if (!g_frameFile.is_open() || !g_resFile.is_open()) {
        Logger::warn(str::format("BlessedDump: failed to open capture files for frame ", index));
        g_frameFile.close();
        g_resFile.close();
        return;
      }

      g_seenResources.clear();
      g_seq        = 0;
      blessed_dump_detail::g_capturing = true;
      g_captureIdx = index;

      Logger::info(str::format("BlessedDump: capturing frame ", index, " to ", g_dir));
    }

    void FinishCapture() {
      g_frameFile.close();
      g_resFile.close();
      blessed_dump_detail::g_capturing = false;

      Logger::info(str::format("BlessedDump: finished frame ", g_captureIdx));
    }

    // blessed: runs once, at static-init time (dll load) -- this used to be
    // the lambda inside a function-local magic static in IsEnabled(), which
    // meant every one of IsEnabled()'s ~13k-per-frame callers rechecked a
    // thread-safe init guard even long after the one-time init had run.
    // Called from the g_enabled initializer below, so it must stay textually
    // after g_dir/ParseDumpAt.
    bool InitEnabled() {
      std::string dir = env::getEnvVar("BLESSED_PROBE_DIR");

      if (dir.empty())
        return false;

      g_dir = dir;

      std::string dumpAt = env::getEnvVar("BLESSED_DUMP_AT");

      if (!dumpAt.empty())
        ParseDumpAt(dumpAt);

      return true;
    }

  }


  namespace blessed_dump_detail {
    extern const bool g_enabled = InitEnabled();
  }


  void BlessedDump::OnPresent() {
    if (!IsEnabled())
      return;

    if (blessed_dump_detail::g_capturing)
      FinishCapture();

    g_presentIndex++;

    bool startNext = false;

    for (uint64_t idx : g_dumpAt) {
      if (idx == g_presentIndex) {
        startNext = true;
        break;
      }
    }

    if (++g_presentsSinceCheck >= 30) {
      g_presentsSinceCheck = 0;

      std::error_code ec;
      std::string triggerPath = g_dir + "/dump.trigger";

      if (std::filesystem::exists(triggerPath, ec)) {
        std::filesystem::remove(triggerPath, ec);
        startNext = true;
      }
    }

    if (startNext)
      StartCapture(g_presentIndex);
  }


  void BlessedDump::RecordDraw(
    const D3D11ContextState&   state,
          BlessedDumpOp        op,
    const BlessedDrawCounts&   counts) {
    if (!IsCapturing())
      return;

    std::string line;
    JsonWriter j(line);
    j.objBegin();

    j.key("i");  j.val(g_seq++);
    j.key("op"); j.val(
      op == BlessedDumpOp::Draw                 ? "draw" :
      op == BlessedDumpOp::DrawIndexed          ? "draw_indexed" :
      op == BlessedDumpOp::DrawInstanced        ? "draw_instanced" : "draw_indexed_instanced");

    bool isIndexed = op == BlessedDumpOp::DrawIndexed || op == BlessedDumpOp::DrawIndexedInstanced;

    j.key(isIndexed ? "index_count" : "vertex_count"); j.val(counts.vertexOrIndexCount);
    j.key(isIndexed ? "start_index" : "start_vertex");  j.val(counts.startVertexOrIndex);

    if (isIndexed) {
      j.key("base_vertex"); j.val(counts.baseVertex);
    }

    j.key("instance_count"); j.val(counts.instanceCount);
    j.key("start_instance"); j.val(counts.startInstance);
    j.key("topology");       j.val(int32_t(state.ia.primitiveTopology));

    // shaders
    if (state.vs.ptr()) { j.key("vs"); j.val(state.vs->GetCommonShader()->GetName()); }
    if (state.ps.ptr()) { j.key("ps"); j.val(state.ps->GetCommonShader()->GetName()); }
    if (state.gs.ptr()) { j.key("gs"); j.val(state.gs->GetCommonShader()->GetName()); }
    if (state.hs.ptr()) { j.key("hs"); j.val(state.hs->GetCommonShader()->GetName()); }
    if (state.ds.ptr()) { j.key("ds"); j.val(state.ds->GetCommonShader()->GetName()); }

    // render targets
    j.key("rtv"); j.arrBegin();
    for (uint32_t i = 0; i < state.om.maxRtv; i++) {
      auto* rtv = state.om.rtvs[i].ptr();
      if (!rtv) continue;

      BlessedViewInfo info = GetViewInfoT<D3D11RenderTargetView, D3D11_RENDER_TARGET_VIEW_DESC>(rtv);
      TouchResource(reinterpret_cast<ID3D11Resource*>(info.res));

      j.objBegin();
      j.key("slot"); j.val(i);
      WriteViewInfo(j, info);
      j.key("mip"); j.val(info.mip);
      j.key("array"); j.val(info.arr);
      j.key("first_slice"); j.val(info.slice);
      j.objEnd();
    }
    j.arrEnd();

    // depth-stencil
    if (auto* dsv = state.om.dsv.ptr()) {
      BlessedViewInfo info = GetViewInfoT<D3D11DepthStencilView, D3D11_DEPTH_STENCIL_VIEW_DESC>(dsv);
      TouchResource(reinterpret_cast<ID3D11Resource*>(info.res));

      D3D11_DEPTH_STENCIL_DESC dsDesc = { };
      dsDesc.DepthEnable    = TRUE;
      dsDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
      dsDesc.DepthFunc      = D3D11_COMPARISON_LESS;
      dsDesc.StencilEnable  = FALSE;

      if (auto* dsState = state.om.dsState.ptr())
        dsDesc = dsState->Desc();

      j.key("dsv"); j.objBegin();
      WriteViewInfo(j, info);
      j.key("mip"); j.val(info.mip);
      j.key("array"); j.val(info.arr);
      j.key("first_slice"); j.val(info.slice);
      j.key("depth_write"); j.val(bool(dsDesc.DepthEnable && dsDesc.DepthWriteMask == D3D11_DEPTH_WRITE_MASK_ALL));
      j.key("depth_func");  j.val(int32_t(dsDesc.DepthFunc));
      j.key("stencil");     j.val(bool(dsDesc.StencilEnable));
      j.objEnd();
    } else {
      j.key("dsv"); j.val(nullptr);
    }

    // blend
    j.key("blend"); j.arrBegin();
    for (uint32_t i = 0; i < state.om.maxRtv; i++) {
      D3D11_RENDER_TARGET_BLEND_DESC1 rtDesc = { };
      rtDesc.BlendEnable    = FALSE;
      rtDesc.SrcBlend       = D3D11_BLEND_ONE;
      rtDesc.DestBlend      = D3D11_BLEND_ZERO;
      rtDesc.BlendOp        = D3D11_BLEND_OP_ADD;
      rtDesc.RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;

      if (auto* blendState = state.om.cbState.ptr()) {
        const D3D11_BLEND_DESC1& desc = blendState->Desc();
        rtDesc = desc.IndependentBlendEnable ? desc.RenderTarget[i] : desc.RenderTarget[0];
      }

      j.objBegin();
      j.key("slot");       j.val(i);
      j.key("enable");     j.val(bool(rtDesc.BlendEnable));
      j.key("src");        j.val(int32_t(rtDesc.SrcBlend));
      j.key("dst");        j.val(int32_t(rtDesc.DestBlend));
      j.key("op");         j.val(int32_t(rtDesc.BlendOp));
      j.key("write_mask"); j.val(int32_t(rtDesc.RenderTargetWriteMask));
      j.objEnd();
    }
    j.arrEnd();

    // viewport
    if (state.rs.numViewports > 0) {
      const D3D11_VIEWPORT& vp = state.rs.viewports[0];
      j.key("vp"); j.objBegin();
      j.key("x");  j.val(vp.TopLeftX);
      j.key("y");  j.val(vp.TopLeftY);
      j.key("w");  j.val(vp.Width);
      j.key("h");  j.val(vp.Height);
      j.key("min_depth"); j.val(vp.MinDepth);
      j.key("max_depth"); j.val(vp.MaxDepth);
      j.objEnd();
    } else {
      j.key("vp"); j.val(nullptr);
    }

    // shader resource views (vs, ps)
    auto writeSrvList = [&] (D3D11ShaderType stage) {
      j.arrBegin();
      const auto& srvStage = state.srv[stage];
      for (uint32_t i = 0; i < srvStage.maxCount; i++) {
        auto* srv = srvStage.views[i].ptr();
        if (!srv) continue;

        BlessedViewInfo info = GetViewInfoT<D3D11ShaderResourceView, D3D11_SHADER_RESOURCE_VIEW_DESC>(srv);
        TouchResource(reinterpret_cast<ID3D11Resource*>(info.res));

        j.objBegin();
        j.key("slot"); j.val(i);
        WriteViewInfo(j, info);
        j.key("first_slice"); j.val(info.slice);
        j.key("array_size");  j.val(info.arr);
        j.objEnd();
      }
      j.arrEnd();
    };

    j.key("ps_srv"); writeSrvList(D3D11ShaderType::ePixel);
    j.key("vs_srv"); writeSrvList(D3D11ShaderType::eVertex);

    // constant buffers (vs, ps) + first 512 bytes of b0-b2
    auto writeCbList = [&] (D3D11ShaderType stage) {
      j.arrBegin();
      const auto& cbvStage = state.cbv[stage];
      for (uint32_t i = 0; i < cbvStage.maxCount; i++) {
        const auto& cb = cbvStage.buffers[i];
        if (!cb.buffer.ptr()) continue;

        TouchBuffer(cb.buffer.ptr());

        j.objBegin();
        j.key("slot"); j.val(i);
        j.key("res");  j.valHex(uint64_t(cb.buffer.ptr()));
        j.key("size"); j.val(cb.buffer->Desc()->ByteWidth);
        j.key("first_constant"); j.val(cb.constantOffset);
        j.key("num_constants");  j.val(cb.constantCount);
        j.objEnd();
      }
      j.arrEnd();
    };

    auto writeCbData = [&] (D3D11ShaderType stage) {
      j.arrBegin();
      const auto& cbvStage = state.cbv[stage];
      for (uint32_t slot = 0; slot < 3; slot++) {
        D3D11Buffer* buffer = slot < cbvStage.maxCount ? cbvStage.buffers[slot].buffer.ptr() : nullptr;
        void* mapPtr = buffer ? buffer->GetMapPtr() : nullptr;

        if (!buffer || !mapPtr) {
          j.val(nullptr);
          continue;
        }

        UINT byteWidth   = buffer->Desc()->ByteWidth;
        UINT byteOffset  = cbvStage.buffers[slot].constantOffset * 16u;
        byteOffset = std::min(byteOffset, byteWidth);

        UINT available = byteWidth - byteOffset;
        UINT length    = std::min(available, 512u) & ~3u; // blessed: 512 reaches lighting b2 c29 and the point mask's ShadowMapProj

        const float* data = reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(mapPtr) + byteOffset);

        j.arrBegin();
        for (UINT k = 0; k < length / 4; k++)
          j.val(data[k]);
        j.arrEnd();
      }
      j.arrEnd();
    };

    j.key("cb"); j.objBegin();
    j.key("vs"); writeCbList(D3D11ShaderType::eVertex);
    j.key("ps"); writeCbList(D3D11ShaderType::ePixel);
    j.key("data"); j.objBegin();
    j.key("vs"); writeCbData(D3D11ShaderType::eVertex);
    j.key("ps"); writeCbData(D3D11ShaderType::ePixel);
    j.objEnd();
    j.objEnd();

    // input assembler
    j.key("ia"); j.objBegin();

    j.key("elements"); j.arrBegin();
    if (auto* layout = state.ia.inputLayout.ptr()) {
      for (uint32_t i = 0; i < layout->GetAttributeCount(); i++) {
        DxvkVertexAttribute attr = layout->GetInput(i).attribute();
        j.objBegin();
        // blessed: dxvk discards D3D11 semantic strings after translating
        // to shader input locations, so we report the location instead.
        j.key("location"); j.val(attr.location);
        j.key("fmt");       j.val(int32_t(attr.format));
        j.key("slot");      j.val(attr.binding);
        j.key("offset");    j.val(attr.offset);
        j.objEnd();
      }
    }
    j.arrEnd();

    j.key("vertex_buffers"); j.arrBegin();
    for (uint32_t i = 0; i < state.ia.maxVbCount; i++) {
      const auto& vb = state.ia.vertexBuffers[i];
      if (!vb.buffer.ptr()) continue;

      TouchBuffer(vb.buffer.ptr());

      j.objBegin();
      j.key("slot");   j.val(i);
      j.key("res");    j.valHex(uint64_t(vb.buffer.ptr()));
      j.key("stride"); j.val(vb.stride);
      j.key("offset"); j.val(vb.offset);
      j.objEnd();
    }
    j.arrEnd();

    if (auto* ib = state.ia.indexBuffer.buffer.ptr()) {
      TouchBuffer(ib);

      j.key("index_buffer"); j.objBegin();
      j.key("res");    j.valHex(uint64_t(ib));
      j.key("fmt");    j.val(int32_t(state.ia.indexBuffer.format));
      j.key("offset"); j.val(state.ia.indexBuffer.offset);
      j.objEnd();
    } else {
      j.key("index_buffer"); j.val(nullptr);
    }

    j.objEnd(); // ia

    j.objEnd(); // line

    g_frameFile << line << '\n';
  }


  void BlessedDump::RecordDispatch(
    const D3D11ContextState&   state,
          UINT                 ThreadGroupCountX,
          UINT                 ThreadGroupCountY,
          UINT                 ThreadGroupCountZ) {
    if (!IsCapturing())
      return;

    std::string line;
    JsonWriter j(line);
    j.objBegin();

    j.key("i");  j.val(g_seq++);
    j.key("op"); j.val("dispatch");

    if (state.cs.ptr())
      { j.key("cs"); j.val(state.cs->GetCommonShader()->GetName()); }

    j.key("group_x"); j.val(ThreadGroupCountX);
    j.key("group_y"); j.val(ThreadGroupCountY);
    j.key("group_z"); j.val(ThreadGroupCountZ);

    j.key("cs_srv"); j.arrBegin();
    const auto& srvStage = state.srv[D3D11ShaderType::eCompute];
    for (uint32_t i = 0; i < srvStage.maxCount; i++) {
      auto* srv = srvStage.views[i].ptr();
      if (!srv) continue;

      BlessedViewInfo info = GetViewInfoT<D3D11ShaderResourceView, D3D11_SHADER_RESOURCE_VIEW_DESC>(srv);
      TouchResource(reinterpret_cast<ID3D11Resource*>(info.res));

      j.objBegin();
      j.key("slot"); j.val(i);
      WriteViewInfo(j, info);
      j.objEnd();
    }
    j.arrEnd();

    j.key("cs_uav"); j.arrBegin();
    for (uint32_t i = 0; i < state.uav.maxCount; i++) {
      auto* uav = state.uav.views[i].ptr();
      if (!uav) continue;

      BlessedViewInfo info = GetViewInfoT<D3D11UnorderedAccessView, D3D11_UNORDERED_ACCESS_VIEW_DESC>(uav);
      TouchResource(reinterpret_cast<ID3D11Resource*>(info.res));

      j.objBegin();
      j.key("slot"); j.val(i);
      WriteViewInfo(j, info);
      j.objEnd();
    }
    j.arrEnd();

    j.objEnd();

    g_frameFile << line << '\n';
  }


  namespace {
    void RecordResourceOnly(const char* op, ID3D11Resource* pResource) {
      if (!BlessedDump::IsCapturing() || !pResource)
        return;

      TouchResource(pResource);

      std::string line;
      JsonWriter j(line);
      j.objBegin();
      j.key("i");   j.val(g_seq++);
      j.key("op");  j.val(op);
      j.key("res"); j.valHex(uint64_t(pResource));
      j.objEnd();

      g_frameFile << line << '\n';
    }
  }


  void BlessedDump::RecordClearRtv(ID3D11RenderTargetView* pRtv, const FLOAT color[4]) {
    if (!pRtv || !IsCapturing())
      return;

    // blessed: the clear colour decides what an unwritten mask channel reads
    auto* rtv = static_cast<D3D11RenderTargetView*>(pRtv);
    const D3D11_VK_VIEW_INFO& vi = rtv->GetViewInfo();
    TouchResource(vi.pResource);

    std::string line;
    JsonWriter j(line);
    j.objBegin();
    j.key("i");   j.val(g_seq++);
    j.key("op");  j.val("clear_rtv");
    j.key("res"); j.valHex(uint64_t(vi.pResource));
    j.key("color"); j.arrBegin();
    for (uint32_t k = 0; k < 4; k++)
      j.val(color ? color[k] : 0.0f);
    j.arrEnd();
    j.key("first_slice"); j.val(vi.Image.MinLayer);
    j.objEnd();

    g_frameFile << line << '\n';
  }


  void BlessedDump::RecordClearDsv(ID3D11DepthStencilView* pDsv) {
    if (!pDsv)
      return;

    auto* dsv = static_cast<D3D11DepthStencilView*>(pDsv);
    RecordResourceOnly("clear_dsv", reinterpret_cast<ID3D11Resource*>(dsv->GetViewInfo().pResource));
  }


  void BlessedDump::RecordCopyResource(ID3D11Resource* pDst, ID3D11Resource* pSrc) {
    if (!IsCapturing())
      return;

    TouchResource(pDst);
    TouchResource(pSrc);

    std::string line;
    JsonWriter j(line);
    j.objBegin();
    j.key("i");   j.val(g_seq++);
    j.key("op");  j.val("copy_resource");
    j.key("dst"); j.valHex(uint64_t(pDst));
    j.key("src"); j.valHex(uint64_t(pSrc));
    j.objEnd();

    g_frameFile << line << '\n';
  }


  void BlessedDump::RecordCopySubresourceRegion(
          ID3D11Resource*      pDst,
          UINT                 DstSubresource,
          ID3D11Resource*      pSrc,
          UINT                 SrcSubresource) {
    if (!IsCapturing())
      return;

    TouchResource(pDst);
    TouchResource(pSrc);

    std::string line;
    JsonWriter j(line);
    j.objBegin();
    j.key("i");       j.val(g_seq++);
    j.key("op");      j.val("copy_subresource_region");
    j.key("dst");     j.valHex(uint64_t(pDst));
    j.key("dst_sub"); j.val(int32_t(DstSubresource));
    j.key("src");     j.valHex(uint64_t(pSrc));
    j.key("src_sub"); j.val(int32_t(SrcSubresource));
    j.objEnd();

    g_frameFile << line << '\n';
  }


  void BlessedDump::RecordUpdateSubresource(ID3D11Resource* pDst, UINT DstSubresource) {
    if (!IsCapturing() || !pDst)
      return;

    TouchResource(pDst);

    std::string line;
    JsonWriter j(line);
    j.objBegin();
    j.key("i");   j.val(g_seq++);
    j.key("op");  j.val("update_subresource");
    j.key("res"); j.valHex(uint64_t(pDst));
    j.key("sub"); j.val(int32_t(DstSubresource));
    j.objEnd();

    g_frameFile << line << '\n';
  }


  void BlessedDump::RecordMap(ID3D11Resource* pResource, UINT Subresource, D3D11_MAP MapType) {
    if (!IsCapturing() || !pResource)
      return;

    TouchResource(pResource);

    std::string line;
    JsonWriter j(line);
    j.objBegin();
    j.key("i");  j.val(g_seq++);
    j.key("op"); j.val("map");
    j.key("res"); j.valHex(uint64_t(pResource));
    j.key("sub"); j.val(int32_t(Subresource));
    j.key("map_type"); j.val(int32_t(MapType));
    j.objEnd();

    g_frameFile << line << '\n';
  }

}
