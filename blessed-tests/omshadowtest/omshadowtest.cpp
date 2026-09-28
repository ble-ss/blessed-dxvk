// fe-getters seat: a fixed OMSet/ClearState/Get sequence, run once against
// whichever d3d11.dll is on PATH (the fork's own, plain or with the
// threaded front end on), printing a trace that is stable across pointer
// identity so the same sequence run twice -- once with the threaded front
// end off, once with it on (loopback, deterministic) -- can be diffed byte
// for byte. See run.sh. No rendering: this only exercises OM bindings.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void Check(HRESULT hr, const char* what) {
  if (FAILED(hr)) { fprintf(stderr, "%s failed: 0x%08lx\n", what, hr); exit(1); }
}

static LRESULT CALLBACK WndProc(HWND h, UINT m, WPARAM w, LPARAM l) {
  return DefWindowProcA(h, m, w, l);
}

// Maps a returned view pointer back to the small stable tag it was created
// with, so the trace reads the same regardless of the run's real addresses.
// A non-null pointer that isn't one of ours is a bug, not a null slot: it
// prints loudly instead of silently matching.
struct TagTable {
  std::vector<void*> byTag; // index 0 unused (0 means "no tag" / null)

  int tag(void* p) {
    if (!p) return 0;
    for (size_t i = 0; i < byTag.size(); i++)
      if (byTag[i] == p) return int(i) + 1;
    return -1; // unrecognized, non-null: print as an error, never matches
  }

  int add(void* p) { byTag.push_back(p); return int(byTag.size()); }
};

static TagTable g_tags;

template<typename T>
static void PrintRtv(const char* label, T* p) {
  int t = g_tags.tag(p);
  if (t < 0) printf("%s=BADPTR", label);
  else if (t == 0) printf("%s=null", label);
  else printf("%s=#%d", label, t);
}

int main() {
  SetProcessDPIAware();
  WNDCLASSA wc = { }; wc.lpfnWndProc = WndProc; wc.hInstance = GetModuleHandleA(nullptr);
  wc.lpszClassName = "omshadowtest";
  RegisterClassA(&wc);
  // Not shown: this test binds no render targets to the swapchain and
  // never presents, so a visible window would only flash and confuse.
  HWND hwnd = CreateWindowExA(0, "omshadowtest", "omshadowtest", WS_OVERLAPPEDWINDOW,
    0, 0, 64, 64, nullptr, nullptr, wc.hInstance, nullptr);

  DXGI_SWAP_CHAIN_DESC sd = { };
  sd.BufferCount = 1; sd.BufferDesc.Width = 64; sd.BufferDesc.Height = 64;
  sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
  sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT; sd.OutputWindow = hwnd;
  sd.SampleDesc.Count = 1; sd.Windowed = TRUE; sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

  ID3D11Device* dev = nullptr; ID3D11DeviceContext* ctx = nullptr; IDXGISwapChain* sc = nullptr;
  D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_0;
  Check(D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, &fl, 1,
    D3D11_SDK_VERSION, &sd, &sc, &dev, nullptr, &ctx), "device");

  // Two small render-targetable textures, a depth texture, and a UAV-
  // capable buffer with two array slices' worth of UAVs (two views over
  // disjoint halves of the same buffer, so no aliasing hazard).
  ID3D11Texture2D *texA = nullptr, *texB = nullptr, *texDepth = nullptr;
  D3D11_TEXTURE2D_DESC td = { };
  td.Width = 4; td.Height = 4; td.MipLevels = 1; td.ArraySize = 1;
  td.SampleDesc.Count = 1; td.Usage = D3D11_USAGE_DEFAULT;
  td.Format = DXGI_FORMAT_R8G8B8A8_UNORM; td.BindFlags = D3D11_BIND_RENDER_TARGET;
  Check(dev->CreateTexture2D(&td, nullptr, &texA), "texA");
  Check(dev->CreateTexture2D(&td, nullptr, &texB), "texB");
  td.Format = DXGI_FORMAT_R32_TYPELESS; td.BindFlags = D3D11_BIND_DEPTH_STENCIL;
  Check(dev->CreateTexture2D(&td, nullptr, &texDepth), "texDepth");

  ID3D11Buffer* uavBuf = nullptr;
  D3D11_BUFFER_DESC bd = { };
  bd.ByteWidth = 256; bd.Usage = D3D11_USAGE_DEFAULT;
  bd.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
  bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
  Check(dev->CreateBuffer(&bd, nullptr, &uavBuf), "uavBuf");

  ID3D11RenderTargetView *rtvA = nullptr, *rtvB = nullptr;
  Check(dev->CreateRenderTargetView(texA, nullptr, &rtvA), "rtvA");
  Check(dev->CreateRenderTargetView(texB, nullptr, &rtvB), "rtvB");

  ID3D11DepthStencilView* dsv = nullptr;
  D3D11_DEPTH_STENCIL_VIEW_DESC dd = { };
  dd.Format = DXGI_FORMAT_D32_FLOAT; dd.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
  Check(dev->CreateDepthStencilView(texDepth, &dd, &dsv), "dsv");

  ID3D11UnorderedAccessView *uav0 = nullptr, *uav1 = nullptr;
  D3D11_UNORDERED_ACCESS_VIEW_DESC ud = { };
  ud.Format = DXGI_FORMAT_R32_TYPELESS; ud.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
  ud.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
  ud.Buffer.FirstElement = 0; ud.Buffer.NumElements = 32; // first half, in 32-bit elements
  Check(dev->CreateUnorderedAccessView(uavBuf, &ud, &uav0), "uav0");
  ud.Buffer.FirstElement = 32; ud.Buffer.NumElements = 32; // second half
  Check(dev->CreateUnorderedAccessView(uavBuf, &ud, &uav1), "uav1");

  // review additions: an 8x8 render target (size mismatch with the 4x4
  // ones and the 4x4 depth, so dxvk's ValidateRenderTargets rejects any
  // mix), and a 4x4 texture bindable as both RTV and UAV, for the hazard
  // unbinds a KEEP half triggers.
  ID3D11Texture2D *texBig = nullptr, *texRU = nullptr;
  td.Width = 8; td.Height = 8; td.Format = DXGI_FORMAT_R8G8B8A8_UNORM; td.BindFlags = D3D11_BIND_RENDER_TARGET;
  Check(dev->CreateTexture2D(&td, nullptr, &texBig), "texBig");
  td.Width = 4; td.Height = 4; td.Format = DXGI_FORMAT_R32_FLOAT;
  td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_UNORDERED_ACCESS;
  Check(dev->CreateTexture2D(&td, nullptr, &texRU), "texRU");

  ID3D11RenderTargetView *rtvBig = nullptr, *rtvRU = nullptr, *rtvTmp = nullptr;
  Check(dev->CreateRenderTargetView(texBig, nullptr, &rtvBig), "rtvBig");
  Check(dev->CreateRenderTargetView(texRU, nullptr, &rtvRU), "rtvRU");
  Check(dev->CreateRenderTargetView(texB, nullptr, &rtvTmp), "rtvTmp");

  ID3D11UnorderedAccessView* uavRU = nullptr;
  Check(dev->CreateUnorderedAccessView(texRU, nullptr, &uavRU), "uavRU");

  // Tags 1..9, in this exact order -- both runs create the same objects in
  // the same order, so the tags line up even though the pointers don't.
  g_tags.add(rtvA); g_tags.add(rtvB); g_tags.add(dsv); g_tags.add(uav0); g_tags.add(uav1);
  g_tags.add(rtvBig); g_tags.add(rtvRU); g_tags.add(uavRU); g_tags.add(rtvTmp);

  auto step = [&](const char* what) { printf("-- %s\n", what); };

  auto getRtv = [&](UINT n) {
    ID3D11RenderTargetView* rtv[2] = { };
    ID3D11DepthStencilView* d = nullptr;
    ctx->OMGetRenderTargets(n, rtv, &d);
    printf("OMGetRenderTargets(%u):", n);
    for (UINT i = 0; i < n; i++) { printf(" "); PrintRtv("rtv", rtv[i]); }
    printf(" "); PrintRtv("dsv", d); printf("\n");
    for (UINT i = 0; i < n; i++) if (rtv[i]) rtv[i]->Release();
    if (d) d->Release();
  };

  auto getRtvUav = [&](UINT nRtv, UINT uavStart, UINT nUav) {
    ID3D11RenderTargetView* rtv[2] = { };
    ID3D11DepthStencilView* d = nullptr;
    ID3D11UnorderedAccessView* uav[4] = { };
    ctx->OMGetRenderTargetsAndUnorderedAccessViews(nRtv, rtv, &d, uavStart, nUav, uav);
    printf("OMGetRenderTargetsAndUnorderedAccessViews(%u,%u,%u):", nRtv, uavStart, nUav);
    for (UINT i = 0; i < nRtv; i++) { printf(" "); PrintRtv("rtv", rtv[i]); }
    printf(" "); PrintRtv("dsv", d); printf("\n");
    printf("  uav:");
    for (UINT i = 0; i < nUav; i++) { printf(" "); PrintRtv("uav", uav[i]); }
    printf("\n");
    for (UINT i = 0; i < nRtv; i++) if (rtv[i]) rtv[i]->Release();
    if (d) d->Release();
    for (UINT i = 0; i < nUav; i++) if (uav[i]) uav[i]->Release();
  };

  // 1. before any bind: the shadow (if any) starts unknown, must drain to
  //    the correct all-null answer.
  step("initial");
  getRtv(2);

  // 2. a plain OMSetRenderTargets, then read it back.
  step("set rtvA+dsv");
  ID3D11RenderTargetView* setRtv1[1] = { rtvA };
  ctx->OMSetRenderTargets(1, setRtv1, dsv);
  getRtv(2);

  // 2b. (review) the first set after the initial unknown state must leave
  //     every UAV slot readable as null, not the invalidate() fill.
  step("uavs after the first set from unknown");
  getRtvUav(2, 0, 4);

  // 3. OMSetRenderTargetsAndUnorderedAccessViews binding both halves,
  //    UAVs starting at slot 1 (leaves slot 0 null on purpose).
  step("set rtvA+dsv+uav[1,2]");
  ID3D11UnorderedAccessView* setUav1[2] = { uav0, uav1 };
  ctx->OMSetRenderTargetsAndUnorderedAccessViews(1, setRtv1, dsv, 1, 2, setUav1, nullptr);
  getRtvUav(2, 0, 4);

  // 4. OMSetRenderTargets (not the AndUAV overload): d3d11 defines this as
  //    unbinding every UAV, even though this call never mentions them.
  step("set rtvB, no dsv (must clear uavs too)");
  ID3D11RenderTargetView* setRtv2[1] = { rtvB };
  ctx->OMSetRenderTargets(1, setRtv2, nullptr);
  getRtvUav(2, 0, 4);

  // 5. KEEP sentinels: rebind only the UAVs, leaving RTV/DSV as they are.
  step("uavs only via KEEP_RENDER_TARGETS_AND_DEPTH_STENCIL");
  ctx->OMSetRenderTargetsAndUnorderedAccessViews(
    D3D11_KEEP_RENDER_TARGETS_AND_DEPTH_STENCIL, nullptr, nullptr, 0, 2, setUav1, nullptr);
  getRtvUav(2, 0, 4);

  // 6. ClearState: back to the device's initial (all-null) state.
  step("ClearState");
  ctx->ClearState();
  getRtv(2);

  // 7. re-establish the shadow after the ClearState invalidation.
  step("set rtvA+dsv again");
  ctx->OMSetRenderTargets(1, setRtv1, dsv);
  getRtv(1);

  // --- review additions ---

  // 8. rejected by ValidateRenderTargets (4x4 + 8x8 colour targets): dxvk
  //    keeps rtvA+dsv bound, so the shadow must too.
  step("rejected: mixed sizes");
  ID3D11RenderTargetView* setMixed[2] = { rtvA, rtvBig };
  ctx->OMSetRenderTargets(2, setMixed, dsv);
  getRtv(2);

  // 9. rejected: 8x8 colour over a 4x4 depth.
  step("rejected: depth smaller than colour");
  ID3D11RenderTargetView* setBig[1] = { rtvBig };
  ctx->OMSetRenderTargets(1, setBig, dsv);
  getRtv(2);

  // 10. rejected by TestRtvUavHazards: the same view twice.
  step("rejected: rtv aliasing");
  ID3D11RenderTargetView* setTwice[2] = { rtvB, rtvB };
  ctx->OMSetRenderTargets(2, setTwice, nullptr);
  getRtv(2);

  // 11. KEEP render targets + a UAV over the bound RTV's texture: dxvk
  //     unbinds the RTV (ResolveOmRtvHazards).
  step("keep rtvs, uav over the bound rtv unbinds it");
  ID3D11RenderTargetView* setRU[1] = { rtvRU };
  ID3D11UnorderedAccessView* setUavRU[1] = { uavRU };
  ctx->OMSetRenderTargets(1, setRU, nullptr);
  ctx->OMSetRenderTargetsAndUnorderedAccessViews(
    D3D11_KEEP_RENDER_TARGETS_AND_DEPTH_STENCIL, nullptr, nullptr, 1, 1, setUavRU, nullptr);
  getRtvUav(2, 0, 4);

  // 12. KEEP UAVs + an RTV over the bound UAV's texture: dxvk unbinds the
  //     UAV (ResolveOmUavHazards).
  step("keep uavs, rtv over the bound uav unbinds it");
  ctx->OMSetRenderTargetsAndUnorderedAccessViews(
    1, setRU, nullptr, 0, D3D11_KEEP_UNORDERED_ACCESS_VIEWS, nullptr, nullptr);
  getRtvUav(2, 0, 4);

  // 13. after ClearState, a KEEP call must not make the shadow "known"
  //     with the other half unseen.
  step("ClearState, then keep-rtv uav set");
  ctx->OMSetRenderTargets(1, setRtv1, dsv);
  ctx->ClearState();
  ID3D11UnorderedAccessView* setUav0[1] = { uav0 };
  ctx->OMSetRenderTargetsAndUnorderedAccessViews(
    D3D11_KEEP_RENDER_TARGETS_AND_DEPTH_STENCIL, nullptr, nullptr, 2, 1, setUav0, nullptr);
  getRtvUav(2, 0, 4);

  // 14. a view the app released while bound: dxvk keeps it alive, the
  //     getter must hand it back intact.
  step("released while bound");
  ID3D11RenderTargetView* setTmp[1] = { rtvTmp };
  ctx->OMSetRenderTargets(1, setTmp, nullptr);
  rtvTmp->Release();
  getRtv(1);
  ctx->OMSetRenderTargets(0, nullptr, nullptr);
  getRtv(1);

  // 15. only the depth view requested, and only the colour ones: the
  //     unrequested outputs stay untouched, and verify must not flag them.
  step("partial outputs");
  ctx->OMSetRenderTargets(1, setRtv1, dsv);
  { ID3D11DepthStencilView* d = nullptr;
    ctx->OMGetRenderTargets(0, nullptr, &d);
    printf("OMGetRenderTargets(0,null,&dsv): "); PrintRtv("dsv", d); printf("\n");
    if (d) d->Release();
    ID3D11RenderTargetView* r = nullptr;
    ctx->OMGetRenderTargets(1, &r, nullptr);
    printf("OMGetRenderTargets(1,&rtv,null): "); PrintRtv("rtv", r); printf("\n");
    if (r) r->Release(); }

  ctx->Flush();
  printf("done\n");
  return 0;
}
