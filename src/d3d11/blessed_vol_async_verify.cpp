// blessed: BLESSED_VOL_ASYNC_VERIFY -- app-thread half: record the window, replay it on graphics onto scratch, the two bitwise checks. See blessed_vol_async_verify.h.
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "blessed_vol_async_verify.h"
#include "blessed_vol_async.h"
#include "blessed_shader_replace.h"

#include "d3d11_buffer.h"
#include "d3d11_context_imm.h"
#include "d3d11_shader.h"

#include "../dxvk/blessed/blessed_async.h"

#include "../util/util_env.h"
#include "../util/util_string.h"
#include "../util/log/log.h"

namespace dxvk {

  namespace {

    constexpr UINT SrvSlots = 8u;
    constexpr UINT CbSlots  = 4u;
    constexpr UINT SmpSlots = 4u;
    constexpr UINT UavSlots = 2u;

    // same convention as blessed_vol_async.cpp's own MatchesToken; a
    // separate copy (this file must not depend on that .cpp's anonymous
    // namespace) against the one generate hash BLESSED_VOL_ASYNC itself
    // already matched -- BLESSED_VOL_ASYNC_GEN_CS, read again here so a
    // custom hash (a synthetic test's own shader) still lines up.
    bool MatchesToken(const std::string& name, const std::string& stagePrefix, const std::string& hexPrefix) {
      return hexPrefix.size() >= 8u
          && name.rfind(stagePrefix, 0) == 0u
          && name.compare(stagePrefix.size(), hexPrefix.size(), hexPrefix) == 0;
    }

    const std::string& GenerateCsToken() {
      static const std::string s_token = [] {
        std::string gen = env::getEnvVar("BLESSED_VOL_ASYNC_GEN_CS");
        return gen.empty() ? std::string("ab674eb1") : gen;
      }();
      return s_token;
    }

    uint32_t VerifyPeriod() {
      std::string p = env::getEnvVar("BLESSED_VOL_ASYNC_VERIFY_PERIOD");
      uint32_t v = p.empty() ? 0u : uint32_t(std::strtoul(p.c_str(), nullptr, 10));
      // at least 2: a check's blocking readback runs one window later
      return v ? std::max(v, 2u) : 60u;
    }

    uint32_t VerifyMax() {
      std::string m = env::getEnvVar("BLESSED_VOL_ASYNC_VERIFY_MAX");
      if (m.empty())
        return 200u;
      return uint32_t(std::strtoul(m.c_str(), nullptr, 10)); // 0 = unlimited
    }

    bool ComputeEnabled() {
      // not BlessedVolAsync::IsEnabled(): that reads another file's global,
      // whose dynamic initializer may not have run yet when this one runs
      // (static init order across files is unspecified; the mingw build ran
      // this one first and verify stayed off in game, c7876c05). The
      // function-local static behind VanillaVolRequested is always ready.
      return BlessedAsync::VanillaVolRequested() && env::getEnvVar("BLESSED_VOL_ASYNC_VERIFY") == "1";
    }

    // one saved cs invocation's worth of state, for save-replay-restore
    struct SavedCs {
      Com<ID3D11ComputeShader>          shader;
      Com<ID3D11ShaderResourceView>     srv[SrvSlots];
      Com<ID3D11UnorderedAccessView>    uav[UavSlots];
      Com<ID3D11Buffer>                 cb[CbSlots];
      UINT                              cbFirst[CbSlots] = { };
      UINT                              cbNum[CbSlots] = { };
      Com<ID3D11SamplerState>           smp[SmpSlots];
    };

    void SaveCs(D3D11ImmediateContext* ctx, SavedCs* s) {
      ctx->CSGetShader(&s->shader, nullptr, nullptr);
      ctx->CSGetShaderResources(0u, SrvSlots, reinterpret_cast<ID3D11ShaderResourceView**>(s->srv));
      ctx->CSGetUnorderedAccessViews(0u, UavSlots, reinterpret_cast<ID3D11UnorderedAccessView**>(s->uav));
      ctx->CSGetConstantBuffers1(0u, CbSlots, reinterpret_cast<ID3D11Buffer**>(s->cb), s->cbFirst, s->cbNum);
      ctx->CSGetSamplers(0u, SmpSlots, reinterpret_cast<ID3D11SamplerState**>(s->smp));
    }

    void RestoreCs(D3D11ImmediateContext* ctx, const SavedCs& s) {
      ID3D11ShaderResourceView* srv[SrvSlots];
      ID3D11UnorderedAccessView* uav[UavSlots];
      ID3D11Buffer* cb[CbSlots];
      ID3D11SamplerState* smp[SmpSlots];
      UINT keep[UavSlots];
      for (UINT i = 0u; i < SrvSlots; i++) srv[i] = s.srv[i].ptr();
      for (UINT i = 0u; i < UavSlots; i++) { uav[i] = s.uav[i].ptr(); keep[i] = ~0u; }
      for (UINT i = 0u; i < CbSlots; i++) cb[i] = s.cb[i].ptr();
      for (UINT i = 0u; i < SmpSlots; i++) smp[i] = s.smp[i].ptr();

      ctx->CSSetShaderResources(0u, SrvSlots, srv);
      ctx->CSSetUnorderedAccessViews(0u, UavSlots, uav, keep);
      ctx->CSSetConstantBuffers1(0u, CbSlots, cb, s.cbFirst, s.cbNum);
      ctx->CSSetSamplers(0u, SmpSlots, smp);
      ctx->CSSetShader(s.shader.ptr(), nullptr, 0u);
    }

    // one dispatch of the window, as recorded at the time it ran: which
    // tracked volume each srv/uav slot referenced (-1: neither, bind the
    // view as is), and the cpu bytes of every dynamic cbuffer it read
    struct Step {
      SavedCs  cs;
      int      srvVol[SrvSlots] = { };
      int      uavVol[UavSlots] = { };
      std::vector<uint8_t> cbBytes[CbSlots];
      UINT     gx = 0u, gy = 0u, gz = 0u;
    };

    struct State {
      std::recursive_mutex mutex; // replayed dispatches re-enter this file's own hook

      uint32_t windows      = 0u; // generate dispatches seen (the once-per-frame counter)
      uint32_t checksDone   = 0u;
      uint32_t checksMax    = VerifyMax();
      uint32_t period       = VerifyPeriod();

      bool     tracking      = false; // this window's dispatches are being recorded
      bool     replayed      = false; // recorded and replayed; waiting for the wait draw
      bool     coverageBroken = false;
      bool     busy          = false; // re-entrancy guard around our own Dispatch calls

      Com<ID3D11Resource> real[2];    // the froxel volumes' identity for this window

      Com<ID3D11Resource> chainScratch[2]; // the replay's own pair
      Com<ID3D11Resource> finalSnap[2];    // the real pair at the wait, copied on graphics

      bool     comparePending = false;     // finalSnap vs chainScratch, not read back yet
      uint32_t pendingWindow  = 0u;

      std::vector<Step> steps;
      std::vector<Com<ID3D11Buffer>> cbPool; // default-usage copies of the recorded constants

      std::unordered_map<ID3D11Resource*, Com<ID3D11UnorderedAccessView>> uavCache;
      std::unordered_map<ID3D11Resource*, Com<ID3D11ShaderResourceView>>  srvCache;
    };

    State& GetState() {
      static State s;
      return s;
    }

    ID3D11UnorderedAccessView* GetUav(ID3D11Device* dev, ID3D11Resource* res) {
      auto& s = GetState();
      auto found = s.uavCache.find(res);
      if (found != s.uavCache.end())
        return found->second.ptr();

      Com<ID3D11UnorderedAccessView> uav;
      if (FAILED(dev->CreateUnorderedAccessView(res, nullptr, &uav)))
        return nullptr;

      return s.uavCache.emplace(res, uav).first->second.ptr();
    }

    ID3D11ShaderResourceView* GetSrv(ID3D11Device* dev, ID3D11Resource* res) {
      auto& s = GetState();
      auto found = s.srvCache.find(res);
      if (found != s.srvCache.end())
        return found->second.ptr();

      Com<ID3D11ShaderResourceView> srv;
      if (FAILED(dev->CreateShaderResourceView(res, nullptr, &srv)))
        return nullptr;

      return s.srvCache.emplace(res, srv).first->second.ptr();
    }

    bool MakeMatchingScratch(ID3D11Device* dev, ID3D11Resource* like, Com<ID3D11Resource>& out) {
      Com<ID3D11Texture3D> tex3d;
      if (SUCCEEDED(like->QueryInterface(__uuidof(ID3D11Texture3D), reinterpret_cast<void**>(&tex3d)))) {
        D3D11_TEXTURE3D_DESC desc;
        tex3d->GetDesc(&desc);
        desc.Usage          = D3D11_USAGE_DEFAULT;
        desc.BindFlags      = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        desc.CPUAccessFlags = 0u;
        desc.MiscFlags      = 0u;

        Com<ID3D11Texture3D> copy;
        if (FAILED(dev->CreateTexture3D(&desc, nullptr, &copy)))
          return false;

        out = copy.ptr();
        return true;
      }

      Com<ID3D11Texture2D> tex2d;
      if (SUCCEEDED(like->QueryInterface(__uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&tex2d)))) {
        D3D11_TEXTURE2D_DESC desc;
        tex2d->GetDesc(&desc);
        desc.Usage          = D3D11_USAGE_DEFAULT;
        desc.BindFlags      = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        desc.CPUAccessFlags = 0u;
        desc.MiscFlags      = 0u;

        Com<ID3D11Texture2D> copy;
        if (FAILED(dev->CreateTexture2D(&desc, nullptr, &copy)))
          return false;

        out = copy.ptr();
        return true;
      }

      return false;
    }

    bool EnsureScratch(D3D11ImmediateContext* ctx, ID3D11Resource* real0, ID3D11Resource* real1) {
      auto& s = GetState();
      if (s.chainScratch[0] != nullptr)
        return true;

      Com<ID3D11Device> dev;
      ctx->GetDevice(&dev);

      ID3D11Resource* real[2] = { real0, real1 };

      for (uint32_t i = 0u; i < 2u; i++) {
        if (!MakeMatchingScratch(dev.ptr(), real[i], s.chainScratch[i])
         || !MakeMatchingScratch(dev.ptr(), real[i], s.finalSnap[i]))
          return false;
      }

      return true;
    }

    int TrackedIndex(ID3D11Resource* res) {
      auto& s = GetState();
      if (!res)
        return -1;
      if (res == s.real[0].ptr())
        return 0;
      if (res == s.real[1].ptr())
        return 1;
      return -1;
    }

    // record the dispatch that just ran; false if it has a shape the
    // replay cannot reproduce (a uav that is not one of the tracked pair)
    bool RecordStep(D3D11ImmediateContext* ctx, UINT gx, UINT gy, UINT gz) {
      auto& s = GetState();
      Step step;
      SaveCs(ctx, &step.cs);
      step.gx = gx;
      step.gy = gy;
      step.gz = gz;

      for (UINT i = 0u; i < UavSlots; i++) {
        step.uavVol[i] = -1;
        if (step.cs.uav[i] == nullptr)
          continue;

        Com<ID3D11Resource> res;
        step.cs.uav[i]->GetResource(&res);
        step.uavVol[i] = TrackedIndex(res.ptr());

        if (step.uavVol[i] < 0)
          return false;
      }

      for (UINT i = 0u; i < SrvSlots; i++) {
        step.srvVol[i] = -1;
        if (step.cs.srv[i] == nullptr)
          continue;

        Com<ID3D11Resource> res;
        step.cs.srv[i]->GetResource(&res);
        step.srvVol[i] = TrackedIndex(res.ptr());
      }

      // the bytes this dispatch saw: GetMapPtr is the replay side of the
      // map state (threaded-fe), set when this dispatch's own rename
      // replayed, so it is the slice the gpu reads -- straight from host
      // memory, never through the cb ring's vram mirror or a queue
      for (UINT i = 0u; i < CbSlots; i++) {
        if (step.cs.cb[i] == nullptr)
          continue;

        auto* buffer = static_cast<D3D11Buffer*>(step.cs.cb[i].ptr());

        if (buffer->Desc()->Usage != D3D11_USAGE_DYNAMIC)
          continue; // default/immutable: bound as is on replay

        const void* src = buffer->GetMapPtr();
        if (!src)
          return false;

        step.cbBytes[i].resize(buffer->Desc()->ByteWidth);
        std::memcpy(step.cbBytes[i].data(), src, step.cbBytes[i].size());
      }

      s.steps.push_back(std::move(step));
      return true;
    }

    ID3D11Buffer* PooledCb(ID3D11Device* dev, size_t index, UINT size) {
      auto& s = GetState();
      while (s.cbPool.size() <= index)
        s.cbPool.emplace_back();

      auto& cb = s.cbPool[index];

      if (cb != nullptr) {
        D3D11_BUFFER_DESC desc;
        cb->GetDesc(&desc);
        if (desc.ByteWidth != size)
          cb = nullptr;
      }

      if (cb == nullptr) {
        D3D11_BUFFER_DESC desc = { size, D3D11_USAGE_DEFAULT, D3D11_BIND_CONSTANT_BUFFER, 0u, 0u, 0u };
        if (FAILED(dev->CreateBuffer(&desc, nullptr, &cb)))
          return nullptr;
      }

      return cb.ptr();
    }

    // the independent reference: every recorded dispatch again, in order,
    // on the graphics queue (the window is closed by now), onto the
    // scratch pair, reading the recorded constants from default-usage
    // copies filled from the recorded host bytes
    bool Replay(D3D11ImmediateContext* ctx) {
      auto& s = GetState();

      Com<ID3D11Device> dev;
      ctx->GetDevice(&dev);

      SavedCs saved;
      SaveCs(ctx, &saved);

      bool ok = true;
      size_t pooled = 0u;

      for (size_t n = 0u; n < s.steps.size() && ok; n++) {
        const Step& step = s.steps[n];

        ID3D11ShaderResourceView* srv[SrvSlots];
        ID3D11UnorderedAccessView* uav[UavSlots];
        ID3D11Buffer* cb[CbSlots];
        ID3D11SamplerState* smp[SmpSlots];
        UINT keep[UavSlots] = { ~0u, ~0u };

        for (UINT i = 0u; i < SrvSlots; i++) {
          srv[i] = step.srvVol[i] < 0 ? step.cs.srv[i].ptr()
            : GetSrv(dev.ptr(), s.chainScratch[step.srvVol[i]].ptr());
          ok = ok && (step.srvVol[i] < 0 || srv[i]);
        }

        for (UINT i = 0u; i < UavSlots; i++) {
          uav[i] = step.uavVol[i] < 0 ? nullptr
            : GetUav(dev.ptr(), s.chainScratch[step.uavVol[i]].ptr());
          ok = ok && (step.uavVol[i] < 0 || uav[i]);
        }

        for (UINT i = 0u; i < CbSlots; i++) {
          cb[i] = step.cs.cb[i].ptr();

          if (!step.cbBytes[i].empty()) {
            cb[i] = PooledCb(dev.ptr(), pooled++, UINT(step.cbBytes[i].size()));
            ok = ok && cb[i];

            if (cb[i])
              ctx->UpdateSubresource(cb[i], 0u, nullptr, step.cbBytes[i].data(), 0u, 0u);
          }
        }

        for (UINT i = 0u; i < SmpSlots; i++)
          smp[i] = step.cs.smp[i].ptr();

        if (!ok)
          break;

        ctx->CSSetShader(step.cs.shader.ptr(), nullptr, 0u);
        ctx->CSSetUnorderedAccessViews(0u, UavSlots, uav, keep);
        ctx->CSSetShaderResources(0u, SrvSlots, srv);
        ctx->CSSetConstantBuffers1(0u, CbSlots, cb, step.cs.cbFirst, step.cs.cbNum);
        ctx->CSSetSamplers(0u, SmpSlots, smp);
        ctx->Dispatch(step.gx, step.gy, step.gz);

        // unbind before the next step flips srv/uav roles
        ID3D11UnorderedAccessView* nullUav[UavSlots] = { };
        ID3D11ShaderResourceView* nullSrv[SrvSlots] = { };
        ctx->CSSetUnorderedAccessViews(0u, UavSlots, nullUav, keep);
        ctx->CSSetShaderResources(0u, SrvSlots, nullSrv);
      }

      RestoreCs(ctx, saved);
      return ok;
    }

  }


  namespace blessed_vol_async_verify_detail {
    extern const bool g_enabled = ComputeEnabled();
  }


  bool BlessedVolAsyncVerify::IsReplaying() {
    return IsEnabled() && GetState().busy;
  }


  void BlessedVolAsyncVerify::OnDispatchDone(
          D3D11ImmediateContext*  ctx,
    const D3D11ContextState&      state,
          UINT                    groupsX,
          UINT                    groupsY,
          UINT                    groupsZ) {
    auto& s = GetState();
    std::lock_guard<std::recursive_mutex> lock(s.mutex);

    if (s.busy || state.cs == nullptr)
      return;

    bool isGenerate = MatchesToken(state.cs->GetCommonShader()->GetName(), "cs.", GenerateCsToken());

    if (isGenerate && (s.tracking || s.replayed)) {
      // a second generate before this window's wait draw: not the shape
      // the dump describes; drop this check rather than compare it
      Logger::warn("blessed: vol-async-verify: a generate dispatch came before the "
        "previous window's wait draw; abandoning that window's check");
      s.tracking = false;
      s.replayed = false;
      s.steps.clear();
    }

    if (!s.tracking) {
      if (!isGenerate)
        return; // between windows: nothing to record

      s.windows++;
      bool checkThis = s.period != 0u && (s.windows % s.period) == 0u
        && (s.checksMax == 0u || s.checksDone < s.checksMax);
      if (!checkThis)
        return;

      Com<ID3D11UnorderedAccessView> uav0, uav1;
      ctx->CSGetUnorderedAccessViews(0u, 1u, &uav0);
      ctx->CSGetUnorderedAccessViews(1u, 1u, &uav1);
      if (uav0 == nullptr || uav1 == nullptr)
        return; // not the shape the dump describes; skip rather than guess

      Com<ID3D11Resource> real0, real1;
      uav0->GetResource(&real0);
      uav1->GetResource(&real1);

      // the scratch is created here, inside the window, and d3d11
      // initializes a new texture through init commands of its own: a
      // window that has to create the scratch only creates it, and the
      // next checked window compares.
      bool fresh = s.chainScratch[0] == nullptr;

      if (!EnsureScratch(ctx, real0.ptr(), real1.ptr())) {
        static bool s_warned = false;
        if (!s_warned) {
          s_warned = true;
          Logger::warn("blessed: vol-async-verify: cannot create scratch (unsupported format?), staying off for this process");
        }
        return;
      }

      if (fresh) {
        s.windows = 0u; // the next checked window is a full period away
        return;
      }

      s.real[0] = real0;
      s.real[1] = real1;
      s.tracking = true;
      s.coverageBroken = false;
      s.steps.clear();

      if (!RecordStep(ctx, groupsX, groupsY, groupsZ))
        s.coverageBroken = true;
      return;
    }

    // tracking: a later dispatch inside the same window
    if (s.coverageBroken)
      return;

    if (!RecordStep(ctx, groupsX, groupsY, groupsZ)) {
      static bool s_warned = false;
      if (!s_warned) {
        s_warned = true;
        Logger::warn("blessed: vol-async-verify: a dispatch inside the window did not reference "
          "the tracked froxel pair; skipping this window's checks (coverage, not a hazard)");
      }
      s.coverageBroken = true;
    }
  }


  void BlessedVolAsyncVerify::OnWindowClose(D3D11ImmediateContext* ctx) {
    auto& s = GetState();
    std::lock_guard<std::recursive_mutex> lock(s.mutex);

    // the previous check's comparison: its readback blocks the cpu until
    // the gpu is done, which also drains the queues -- done here, a window
    // later, so the checked frame itself runs with its normal submission
    // timing (a stall inside it hid a late mirror copy in the synthetic)
    if (s.comparePending && s.windows > s.pendingWindow) {
      s.comparePending = false;
      s.checksDone += 1u;

      ID3D11Resource* real[2] = { s.finalSnap[0].ptr(), s.finalSnap[1].ptr() };
      ID3D11Resource* replay[2] = { s.chainScratch[0].ptr(), s.chainScratch[1].ptr() };

      s.busy = true;
      BlessedShaderVerify::CompareTextures(ctx, "vol-async-final", s.checksDone, s.checksMax, real, replay, 2u);
      s.busy = false;
    }

    if (!s.tracking)
      return;

    s.tracking = false;

    if (s.coverageBroken || s.steps.empty()) {
      s.steps.clear();
      s.replayed = false;
      Logger::warn(str::format("blessed: vol-async-verify: window ", s.windows,
        " not checked (coverage broken; see the earlier warning)"));
      return;
    }

    s.busy = true;
    bool ok = Replay(ctx);
    s.busy = false;

    s.steps.clear();
    s.replayed = ok;

    if (!ok) {
      Logger::warn(str::format("blessed: vol-async-verify: window ", s.windows,
        " not checked (the replay could not bind a recorded step)"));
    }
  }


  void BlessedVolAsyncVerify::OnSyncPoint(D3D11ImmediateContext* ctx) {
    auto& s = GetState();
    std::lock_guard<std::recursive_mutex> lock(s.mutex);

    if (!s.replayed)
      return;

    s.replayed = false;

    // the real pair as the wait draw sees it, copied on graphics after the
    // wait; compared (with a blocking readback) at the next window's close
    s.busy = true;
    ctx->CopyResource(s.finalSnap[0].ptr(), s.real[0].ptr());
    ctx->CopyResource(s.finalSnap[1].ptr(), s.real[1].ptr());
    s.busy = false;

    s.comparePending = true;
    s.pendingWindow = s.windows;
  }

}
