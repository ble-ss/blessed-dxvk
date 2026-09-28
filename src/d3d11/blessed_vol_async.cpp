// blessed: BLESSED_VOL_ASYNC -- app-thread half: shader matching, window open/close, the pass-138 wait. See blessed_vol_async.h.
#include <string>

#include "blessed_vol_async.h"
#include "blessed_vol_async_verify.h"

#include "d3d11_context_imm.h"
#include "d3d11_shader.h"
#include "d3d11_view_srv.h"
#include "d3d11_view_uav.h"

#include "../dxvk/blessed/blessed_async.h"
#include "../dxvk/dxvk_context.h"

#include "../util/util_env.h"
#include "../util/log/log.h"

namespace dxvk {

  namespace {

    struct VolAsyncConfig {
      std::string generateCs;
      std::string waitPs;

      static const VolAsyncConfig& Get() {
        static VolAsyncConfig s_config = [] {
          VolAsyncConfig c;
          std::string gen = env::getEnvVar("BLESSED_VOL_ASYNC_GEN_CS");
          c.generateCs = gen.empty() ? std::string("ab674eb1") : gen;
          std::string wait = env::getEnvVar("BLESSED_VOL_ASYNC_WAIT_PS");
          c.waitPs = wait.empty() ? std::string("c480e36e") : wait;
          return c;
        }();

        return s_config;
      }
    };

    // one bool, cs thread's app-side counterpart: is the async window open
    // right now (between the generate dispatch and the next draw)?
    struct VolAsyncState {
      bool windowOpen = false;
      // blessed: vol-async-3 -- BLESSED_VOL_ASYNC=2: the generate
      // dispatch's two uav images, and whether the chain kick has begun
      Rc<DxvkImage> vol[2];
      bool chainOnCompute = false;
      bool computeRefused = false; // a chain dispatch touched another image: =1 from now on
    };

    VolAsyncState g_state;

    // a full "cs.<hash>..." / "fs.<hash>..." shader name against an 8+ char
    // hex prefix, the same convention as blessed_volumetrics.cpp's MatchesAny
    bool MatchesToken(const std::string& name, const std::string& stagePrefix, const std::string& hexPrefix) {
      return hexPrefix.size() >= 8u
          && name.rfind(stagePrefix, 0) == 0u
          && name.compare(stagePrefix.size(), hexPrefix.size(), hexPrefix) == 0;
    }

    Rc<DxvkImage> ViewImage(D3D11UnorderedAccessView* uav) {
      if (!uav)
        return nullptr;
      Rc<DxvkImageView> view = uav->GetImageView();
      return view != nullptr ? view->image() : nullptr;
    }

    Rc<DxvkImage> ViewImage(D3D11ShaderResourceView* srv) {
      if (!srv)
        return nullptr;
      Rc<DxvkImageView> view = srv->GetImageView();
      return view != nullptr ? view->image() : nullptr;
    }

    // blessed: vol-async-3 -- may this dispatch run on the compute family?
    // Only if every image it binds is one of the two froxel volumes (the
    // ones the ownership transfers cover); buffers are concurrent anyway.
    bool OnlyTouchesVolumes(const D3D11ContextState& state) {
      auto ok = [] (const Rc<DxvkImage>& image) {
        return image == nullptr || image == g_state.vol[0] || image == g_state.vol[1];
      };

      const auto& srvs = state.srv[D3D11ShaderType::eCompute];
      for (uint32_t i = 0u; i < srvs.maxCount; i++) {
        if (!ok(ViewImage(srvs.views[i].ptr())))
          return false;
      }

      for (uint32_t i = 0u; i < state.uav.maxCount; i++) {
        if (!ok(ViewImage(state.uav.views[i].ptr())))
          return false;
      }

      return true;
    }

  }


  namespace blessed_vol_async_detail {
    extern const bool g_enabled = BlessedAsync::VanillaVolRequested();
  }


  void BlessedVolAsync::OnDispatchPre(D3D11ImmediateContext* ctx, const D3D11ContextState& state) {
    if (g_state.windowOpen) {
      // blessed: vol-async-3 -- BLESSED_VOL_ASYNC=2: the first chain
      // dispatch moves the rest of the window to the compute family; a
      // later one that binds another image cannot go there, so the chain
      // kick ends before it (waited for and handed back right away) and
      // the rest of this window records on graphics, in order
      if (BlessedAsync::VanillaVolMode() == 2u && !g_state.computeRefused) {
        bool fits = OnlyTouchesVolumes(state);

        if (!g_state.chainOnCompute && fits) {
          ctx->EmitCs([
            cVol0 = g_state.vol[0],
            cVol1 = g_state.vol[1]
          ] (DxvkContext* dxvkCtx) {
            dxvkCtx->blessedVolAsyncSwitch(cVol0, cVol1); // false: stays as =1
          });
          g_state.chainOnCompute = true;
        } else if (!fits) {
          g_state.computeRefused = true;
          Logger::warn("blessed: vol-async: a chain dispatch binds an image other than the two "
            "froxel volumes; BLESSED_VOL_ASYNC=2 runs as =1 from here on");

          if (g_state.chainOnCompute) {
            ctx->EmitCs([] (DxvkContext* dxvkCtx) {
              dxvkCtx->blessedVolAsyncEnd();
              dxvkCtx->blessedAsyncSync();
            });
            g_state.windowOpen = false;
          }
        }
      }

      return; // already recording this chain (a later dispatch of it)
    }

    // vol-async-verify's own replay of the generate runs on graphics
    if (unlikely(BlessedVolAsyncVerify::IsEnabled()) && BlessedVolAsyncVerify::IsReplaying())
      return;

    D3D11ComputeShader* cs = state.cs.ptr();
    if (!cs)
      return;

    if (!MatchesToken(cs->GetCommonShader()->GetName(), "cs.", VolAsyncConfig::Get().generateCs))
      return;

    ctx->EmitCs([] (DxvkContext* dxvkCtx) {
      dxvkCtx->blessedVolAsyncBegin(); // no-op (one cached bool) when unavailable
    });
    g_state.windowOpen = true;

    // blessed: vol-async-3 -- the two volumes the chain may carry to the
    // compute family: exactly generate's uav 0 and 1
    if (BlessedAsync::VanillaVolMode() == 2u) {
      g_state.vol[0] = ViewImage(state.uav.views[0].ptr());
      g_state.vol[1] = ViewImage(state.uav.views[1].ptr());
      g_state.chainOnCompute = false;

      if (g_state.vol[0] == nullptr || g_state.vol[1] == nullptr || g_state.vol[0] == g_state.vol[1])
        g_state.computeRefused = true;
    }
  }


  void BlessedVolAsync::OnDrawPre(D3D11ImmediateContext* ctx, const D3D11ContextState& state) {
    D3D11PixelShader* ps = state.ps.ptr();
    bool isWaitPs = ps != nullptr
      && MatchesToken(ps->GetCommonShader()->GetName(), "fs.", VolAsyncConfig::Get().waitPs);

    // the safety-critical half: never let a draw record into the async
    // queue's command buffer. Cheap when the window was never opened.
    if (g_state.windowOpen) {
      ctx->EmitCs([] (DxvkContext* dxvkCtx) {
        dxvkCtx->blessedVolAsyncEnd(); // a no-op if blessedVolAsyncBegin() never actually began
      });
      g_state.windowOpen = false;

      // blessed: vol-async-verify -- the window is closed: replay its
      // recorded dispatches on graphics now, compared at the wait draw.
      // See blessed_vol_async_verify.h.
      if (unlikely(BlessedVolAsyncVerify::IsEnabled()))
        BlessedVolAsyncVerify::OnWindowClose(ctx);
    }

    if (!isWaitPs)
      return;

    ctx->EmitCs([] (DxvkContext* dxvkCtx) {
      dxvkCtx->blessedAsyncSync(); // a no-op (one bool test) when nothing is outstanding
    });

    // blessed: vol-async-verify -- safe to read back and compare only now:
    // the wait above guarantees the kick's gpu work (whichever queue it
    // ran on) is done before this draw, and CompareTextures stalls on its
    // own readback besides. See blessed_vol_async_verify.h.
    if (unlikely(BlessedVolAsyncVerify::IsEnabled()))
      BlessedVolAsyncVerify::OnSyncPoint(ctx);
  }

}
