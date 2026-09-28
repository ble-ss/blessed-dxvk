// blessed: vol-collapse, learns vanilla's volumetric integration chain, then replays it as one dispatch and skips the rest
#include "blessed_vol_collapse.h"
#include "blessed_vol_collapse_dxbc.h"
#include "blessed_shader_replace.h"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "d3d11_buffer.h"
#include "d3d11_context_imm.h"
#include "d3d11_shader.h"

#include "../util/util_env.h"
#include "../util/util_string.h"
#include "../util/log/log.h"

namespace dxvk {

  namespace {

    const char* GenerateName = "cs.ab674eb17a87e2a53c93b6c42ce02f7b";
    const char* ChainName    = "cs.1c4ebb626fcb949deff76ee402305459";

    // the chain cs's thread group (dcl_thread_group 32, 32, 1); the
    // collapse cs runs 8x8 groups over the same thread grid
    constexpr uint32_t ChainGroupSize    = 32u;
    constexpr uint32_t CollapseGroupSize = 8u;

    enum class Kind : uint8_t { Other, Generate, Chain };
    enum class Phase : uint8_t { Learning, Active, Off };

    struct Step {
      uint32_t        k   = 0u;
      ID3D11Resource* src = nullptr;
      ID3D11Resource* dst = nullptr;
      UINT            gx  = 0u;
      UINT            gy  = 0u;
      UINT            gz  = 0u;
    };

    uint32_t ComputeVerifyMax() {
      if (env::getEnvVar("BLESSED_VOL_COLLAPSE_VERIFY") != "1")
        return 0u;

      std::string max = env::getEnvVar("BLESSED_VOL_COLLAPSE_VERIFY_MAX");
      return max.empty() ? 8u : uint32_t(std::strtoul(max.c_str(), nullptr, 10));
    }

    struct State {
      std::recursive_mutex mutex;  // the collapse and the diff dispatch re-enter
      Phase           phase   = Phase::Learning;
      bool            busy    = false;
      bool            genSeen = false;

      std::unordered_map<const void*, Kind> kindMemo;

      // learning: the chain dispatches since the last generate
      std::vector<Step>          observed;
      std::vector<Com<ID3D11Resource>> observedRefs;
      bool                       observeOk = true;

      // what was learned
      uint32_t                   k1    = 0u;
      uint32_t                   count = 0u;
      Com<ID3D11Resource>        first;   // src of the first chain dispatch
      Com<ID3D11Resource>        second;  // dst of the first chain dispatch
      UINT                       gx = 0u, gy = 0u;

      // the current run: index of the next expected chain dispatch, 0 = none
      uint32_t                   next = 0u;
      bool                       verifyRun   = false;
      bool                       pendingDiff = false;
      uint32_t                   verifyDone  = 0u;
      uint32_t                   verifyMax   = ComputeVerifyMax();
      uint64_t                   collapsed   = 0u;

      Com<ID3D11Resource>        scratch[2];
      Com<ID3D11ComputeShader>   cs;
      Com<ID3D11Buffer>          params;
      std::unordered_map<ID3D11Resource*, Com<ID3D11UnorderedAccessView>> uavs;
    };

    State& GetState() {
      static State s;
      return s;
    }

    bool ComputeEnabled() {
      return env::getEnvVar("BLESSED_VOL_COLLAPSE") == "1";
    }

    Kind Classify(const D3D11ContextState& state) {
      auto& s = GetState();
      const D3D11CommonShader* shader = state.cs->GetCommonShader();
      const void* key = shader->GetShader().ptr();

      auto memo = s.kindMemo.find(key);
      if (memo != s.kindMemo.end())
        return memo->second;

      std::string name = shader->GetName();
      Kind kind = name == ChainName ? Kind::Chain
                : name == GenerateName ? Kind::Generate
                : Kind::Other;
      s.kindMemo.emplace(key, kind);
      return kind;
    }

    // cb0[1].x of the chain cs, read as the uint the shader uses
    bool ReadK(const D3D11ContextState& state, uint32_t* k) {
      const auto& cbv = state.cbv[D3D11ShaderType::eCompute];
      if (cbv.maxCount < 1u)
        return false;

      D3D11Buffer* buffer = cbv.buffers[0].buffer.ptr();
      if (!buffer || !buffer->GetMapPtr())
        return false;

      UINT offset = cbv.buffers[0].constantOffset * 16u + 16u;
      if (offset + 4u > buffer->Desc()->ByteWidth)
        return false;

      std::memcpy(k, reinterpret_cast<const uint8_t*>(buffer->GetMapPtr()) + offset, 4u);
      return true;
    }

    // src (srv t0) and dst (uav u0) of a chain dispatch, as raw pointers;
    // refs, when given, keep them alive
    void ReadTextures(D3D11ImmediateContext* ctx, Step* step, std::vector<Com<ID3D11Resource>>* refs) {
      Com<ID3D11ShaderResourceView> srv;
      Com<ID3D11UnorderedAccessView> uav;
      ctx->CSGetShaderResources(0u, 1u, &srv);
      ctx->CSGetUnorderedAccessViews(0u, 1u, &uav);

      Com<ID3D11Resource> src, dst;
      if (srv != nullptr)
        srv->GetResource(&src);
      if (uav != nullptr)
        uav->GetResource(&dst);

      step->src = src.ptr();
      step->dst = dst.ptr();

      if (refs) {
        refs->push_back(src);
        refs->push_back(dst);
      }
    }

    void TurnOff(const std::string& why) {
      auto& s = GetState();
      s.phase = Phase::Off;
      s.next = 0u;
      Logger::warn(str::format("BlessedVolCollapse: off (", why, "); the chain runs as vanilla"));
    }

    bool CheckFormat(ID3D11Device* dev, ID3D11Resource* res) {
      Com<ID3D11Texture3D> tex;
      if (FAILED(res->QueryInterface(__uuidof(ID3D11Texture3D), reinterpret_cast<void**>(&tex))))
        return false;

      D3D11_TEXTURE3D_DESC desc;
      tex->GetDesc(&desc);

      D3D11_FEATURE_DATA_FORMAT_SUPPORT2 support = { };
      support.InFormat = desc.Format;

      if (FAILED(dev->CheckFeatureSupport(D3D11_FEATURE_FORMAT_SUPPORT2, &support, sizeof(support))))
        return false;

      UINT need = D3D11_FORMAT_SUPPORT2_UAV_TYPED_LOAD | D3D11_FORMAT_SUPPORT2_UAV_TYPED_STORE;
      return (support.OutFormatSupport2 & need) == need
          && (desc.BindFlags & D3D11_BIND_UNORDERED_ACCESS);
    }

    // at a generate: the observed chain, if any, becomes what is learned
    void FinishLearning(D3D11ImmediateContext* ctx) {
      auto& s = GetState();

      if (s.observed.empty())
        return;

      const Step& f = s.observed[0];
      bool ok = s.observeOk && f.k >= 1u && f.src && f.dst && f.src != f.dst && f.gz == 1u;

      for (size_t i = 1u; ok && i < s.observed.size(); i++) {
        const Step& st = s.observed[i];
        bool even = !(i & 1u);
        ok = st.k == f.k + uint32_t(i)
          && st.src == (even ? f.src : f.dst)
          && st.dst == (even ? f.dst : f.src)
          && st.gx == f.gx && st.gy == f.gy && st.gz == 1u;
      }

      Com<ID3D11Device> dev;
      ctx->GetDevice(&dev);

      if (ok && (!CheckFormat(dev.ptr(), f.src) || !CheckFormat(dev.ptr(), f.dst))) {
        TurnOff("the chain's textures do not allow typed uav loads");
        return;
      }

      if (!ok) {
        Logger::warn(str::format("BlessedVolCollapse: a chain of ", s.observed.size(),
          " dispatches did not have the expected shape; still learning"));
      } else {
        s.k1     = f.k;
        s.count  = uint32_t(s.observed.size());
        s.first  = f.src;
        s.second = f.dst;
        s.gx     = f.gx;
        s.gy     = f.gy;
        s.phase  = Phase::Active;
        Logger::info(str::format("BlessedVolCollapse: learned a chain of ", s.count,
          " dispatches, k ", s.k1, "..", s.k1 + s.count - 1u, ", ", s.gx, "x", s.gy,
          " groups; collapsing from the next one",
          s.verifyMax ? str::format(" (the first ", s.verifyMax, " verified against vanilla)") : std::string()));
      }

      s.observed.clear();
      s.observedRefs.clear();
      s.observeOk = true;
    }

    ID3D11UnorderedAccessView* GetUav(ID3D11Device* dev, ID3D11Resource* res) {
      auto& s = GetState();
      auto found = s.uavs.find(res);
      if (found != s.uavs.end())
        return found->second.ptr();

      Com<ID3D11UnorderedAccessView> uav;
      if (FAILED(dev->CreateUnorderedAccessView(res, nullptr, &uav)))
        return nullptr;

      // the map holds the view, and the view holds the resource
      return s.uavs.emplace(res, uav).first->second.ptr();
    }

    // one dispatch that replays the whole chain on (first, second)
    bool RunCollapse(D3D11ImmediateContext* ctx, ID3D11Resource* first, ID3D11Resource* second) {
      auto& s = GetState();
      Com<ID3D11Device> dev;
      ctx->GetDevice(&dev);

      if (s.cs == nullptr) {
        if (FAILED(dev->CreateComputeShader(blessed_vol_collapse_cs, sizeof(blessed_vol_collapse_cs), nullptr, &s.cs)))
          return false;

        D3D11_BUFFER_DESC desc = { 16u, D3D11_USAGE_DEFAULT, D3D11_BIND_CONSTANT_BUFFER, 0u, 0u, 0u };
        if (FAILED(dev->CreateBuffer(&desc, nullptr, &s.params)))
          return false;
      }

      ID3D11UnorderedAccessView* uavs[2] = { GetUav(dev.ptr(), first), GetUav(dev.ptr(), second) };
      if (!uavs[0] || !uavs[1])
        return false;

      // save what the collapse changes
      Com<ID3D11ComputeShader>        oldCs;
      Com<ID3D11ShaderResourceView>   oldSrv;
      Com<ID3D11UnorderedAccessView>  oldUav[2];
      Com<ID3D11Buffer>               oldCb;
      ctx->CSGetShader(&oldCs, nullptr, nullptr);
      ctx->CSGetShaderResources(0u, 1u, &oldSrv);
      ctx->CSGetUnorderedAccessViews(0u, 2u, reinterpret_cast<ID3D11UnorderedAccessView**>(oldUav));
      ctx->CSGetConstantBuffers(0u, 1u, &oldCb);

      uint32_t params[4] = { s.k1, s.count, 0u, 0u };
      ctx->UpdateSubresource(s.params.ptr(), 0u, nullptr, params, 0u, 0u);

      ID3D11ShaderResourceView* nullSrv = nullptr;
      ID3D11Buffer* cb = s.params.ptr();
      UINT keep[2] = { ~0u, ~0u };
      ctx->CSSetShaderResources(0u, 1u, &nullSrv);
      ctx->CSSetUnorderedAccessViews(0u, 2u, uavs, keep);
      ctx->CSSetConstantBuffers(0u, 1u, &cb);
      ctx->CSSetShader(s.cs.ptr(), nullptr, 0u);

      s.busy = true;
      ctx->Dispatch(s.gx * (ChainGroupSize / CollapseGroupSize), s.gy * (ChainGroupSize / CollapseGroupSize), 1u);
      s.busy = false;

      ID3D11UnorderedAccessView* restoreUav[2] = { oldUav[0].ptr(), oldUav[1].ptr() };
      ID3D11ShaderResourceView* restoreSrv = oldSrv.ptr();
      ID3D11Buffer* restoreCb = oldCb.ptr();
      ctx->CSSetUnorderedAccessViews(0u, 2u, restoreUav, keep);
      ctx->CSSetShaderResources(0u, 1u, &restoreSrv);
      ctx->CSSetConstantBuffers(0u, 1u, &restoreCb);
      ctx->CSSetShader(oldCs.ptr(), nullptr, 0u);
      return true;
    }

    bool MakeScratch(D3D11ImmediateContext* ctx) {
      auto& s = GetState();

      if (s.scratch[0] != nullptr)
        return true;

      Com<ID3D11Device> dev;
      ctx->GetDevice(&dev);

      for (uint32_t i = 0u; i < 2u; i++) {
        Com<ID3D11Texture3D> tex;
        if (FAILED((i ? s.second : s.first)->QueryInterface(__uuidof(ID3D11Texture3D), reinterpret_cast<void**>(&tex))))
          return false;

        D3D11_TEXTURE3D_DESC desc;
        tex->GetDesc(&desc);
        desc.Usage          = D3D11_USAGE_DEFAULT;
        desc.BindFlags      = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        desc.CPUAccessFlags = 0u;
        desc.MiscFlags      = 0u;

        Com<ID3D11Texture3D> copy;
        if (FAILED(dev->CreateTexture3D(&desc, nullptr, &copy)))
          return false;

        s.scratch[i] = copy.ptr();
      }

      return true;
    }

    // the first chain dispatch of a run, all checks passed
    bool StartRun(D3D11ImmediateContext* ctx) {
      auto& s = GetState();
      s.genSeen = false;
      s.next = s.count > 1u ? 1u : 0u;

      if (s.verifyDone < s.verifyMax) {
        if (!MakeScratch(ctx)) {
          TurnOff("cannot create the verify copies");
          return false;
        }

        // the state the chain starts from, then the collapse on the copies;
        // the chain itself runs as vanilla on the real textures
        ctx->CopyResource(s.scratch[0].ptr(), s.first.ptr());
        ctx->CopyResource(s.scratch[1].ptr(), s.second.ptr());

        if (!RunCollapse(ctx, s.scratch[0].ptr(), s.scratch[1].ptr())) {
          TurnOff("cannot run the collapse");
          return false;
        }

        s.verifyRun   = true;
        s.pendingDiff = s.next == 0u;
        return false;
      }

      if (!RunCollapse(ctx, s.first.ptr(), s.second.ptr())) {
        TurnOff("cannot run the collapse");
        return false;
      }

      if (s.collapsed++ == 0u)
        Logger::info("BlessedVolCollapse: first chain collapsed");

      return true;
    }

  }


  namespace blessed_vol_collapse_detail {
    extern const bool g_enabled = ComputeEnabled();
  }


  bool BlessedVolCollapse::OnDispatch(
          D3D11ImmediateContext*  ctx,
    const D3D11ContextState&      state,
          UINT                    groupsX,
          UINT                    groupsY,
          UINT                    groupsZ) {
    auto& s = GetState();
    std::lock_guard<std::recursive_mutex> lock(s.mutex);

    if (s.busy || s.phase == Phase::Off || state.cs == nullptr)
      return false;

    Kind kind = Classify(state);

    if (kind == Kind::Other)
      return false;

    if (kind == Kind::Generate) {
      if (s.next) {
        TurnOff(str::format("a chain ended after ", s.next, " of ", s.count, " dispatches"));
        return false;
      }

      if (s.phase == Phase::Learning)
        FinishLearning(ctx);

      s.genSeen = true;
      return false;
    }

    // a chain dispatch
    Step step;
    step.gx = groupsX;
    step.gy = groupsY;
    step.gz = groupsZ;

    bool kOk = ReadK(state, &step.k);

    if (s.phase == Phase::Learning) {
      if (s.genSeen || !s.observed.empty()) {
        s.genSeen = false;
        s.observeOk = s.observeOk && kOk;
        ReadTextures(ctx, &step, &s.observedRefs);
        s.observed.push_back(step);
      }
      return false;
    }

    ReadTextures(ctx, &step, nullptr);

    if (s.next) {
      // inside a run: this dispatch must be the next one of the chain
      uint32_t m = s.next;
      bool even = !(m & 1u);
      bool ok = kOk && step.k == s.k1 + m
             && step.src == (even ? s.first.ptr() : s.second.ptr())
             && step.dst == (even ? s.second.ptr() : s.first.ptr())
             && step.gx == s.gx && step.gy == s.gy && step.gz == 1u;

      if (!ok) {
        TurnOff(str::format("chain dispatch ", m + 1u, " of ", s.count, " did not match (k ", step.k, ")"));
        return false;
      }

      s.next = m + 1u < s.count ? m + 1u : 0u;

      if (s.verifyRun) {
        s.pendingDiff = s.next == 0u;
        return false;
      }

      return true;
    }

    bool start = s.genSeen && kOk && step.k == s.k1
      && step.src == s.first.ptr() && step.dst == s.second.ptr()
      && step.gx == s.gx && step.gy == s.gy && step.gz == 1u;

    if (!start)
      return false;

    return StartRun(ctx);
  }


  void BlessedVolCollapse::OnDispatchDone(
          D3D11ImmediateContext*  ctx,
    const D3D11ContextState&      state) {
    auto& s = GetState();
    std::lock_guard<std::recursive_mutex> lock(s.mutex);

    if (!s.pendingDiff || s.busy)
      return;

    s.pendingDiff = false;
    s.verifyRun   = false;
    s.verifyDone += 1u;

    ID3D11Resource* real[2] = { s.first.ptr(), s.second.ptr() };
    ID3D11Resource* copy[2] = { s.scratch[0].ptr(), s.scratch[1].ptr() };

    s.busy = true;
    BlessedShaderVerify::CompareTextures(ctx, "vol-collapse", s.verifyDone, s.verifyMax, real, copy, 2u);
    s.busy = false;
  }

}
