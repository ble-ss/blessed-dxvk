// zero-copy-present seat: verification harness for BLESSED_ZERO_COPY.
// see .briefs/zero-copy-present.md's "verify" clause. plain D3D11, no
// DirectComposition/IPresentationManager (that machinery is present-idle's
// and present-cursed's concern, unrelated to this feature) -- an ordinary
// flip-model-less DISCARD swap chain, loaded against dxvk's own d3d11.dll
// so BindFramebuffer/PresentImage's zero-copy path is what's under test.
//
// pattern: every frame renders one full-screen opaque quad (deterministic
// color keyed to the frame index, not wall-clock time, so two independent
// runs -- BLESSED_ZERO_COPY=0 and =1 -- draw an identical sequence) then
// OVERLAY_COUNT small alpha-blended quads on top, mirroring the traversal
// dump's shape (one opaque full-screen draw, then blended ui draws).
//
// the primary check, exercising both paths the feature has:
//   SELF-READ, every CHECK_INTERVAL frames: CopyResource + Map reads the
//   app's own D3D11 back buffer resource in-process. This is itself a
//   "read of the back buffer" and must trigger
//   BlessedCorrectBackBufferForRead's corrective copy when zero-copy is
//   active -- so this checks the FALLBACK path, exactly, frame-perfect
//   (no capture-timing slop: same process, same frame). every other
//   frame exercises the pure redirect+skip-blit path unread.
//
// round 2 (lead's ask, after the switch-off baseline itself came back
// all-black): this used to also poll IDXGIOutputDuplication in-process
// for an actual-screen-content check, the way present-cursed's
// comp_d3d11.cpp does. that can never work here: this process loads
// dxvk's OWN d3d11.dll/dxgi.dll (the whole point, to exercise the
// feature), so IDXGIOutput1::DuplicateOutput resolves to DXVK'S DXGI
// output, and dxvk's DxgiOutput::DuplicateOutput1 is a stub that always
// returns DXGI_ERROR_UNSUPPORTED (src/dxgi/dxgi_output.cpp:503-520,
// "Not implemented") -- acquireOk stayed 0 because dupl itself was
// always null, not because of anything frame-rate or window-related.
// comp_d3d11.cpp's own DDA works because IT links the system's native
// d3d11.dll (no dxvk dlls anywhere near it). the actual all-black bug
// was separate and real: see QuadRenderer's rasterizer state, below.
// external screen confirmation now comes from run.sh instead: a plain
// GDI BitBlt screenshot in a completely separate process (no dxvk dlls
// in ITS path either), the same technique pc-test/run.sh already uses.
//
// run twice (BLESSED_ZERO_COPY=0, then =1) and diff the printed verdicts,
// self-read counts, and frame/present timing; see run.sh.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

using Clock = std::chrono::steady_clock;

#define CHECK_HR(expr) do { HRESULT _hr = (expr); if (FAILED(_hr)) { \
  fprintf(stderr, "%s failed: HRESULT 0x%08lx (line %d)\n", #expr, (unsigned long)_hr, __LINE__); exit(1); } } while (0)

static bool g_quit = false;
static LRESULT CALLBACK WndProc(HWND h, UINT m, WPARAM w, LPARAM l) {
  if (m == WM_DESTROY || (m == WM_KEYDOWN && w == VK_ESCAPE)) { g_quit = true; PostQuitMessage(0); return 0; }
  return DefWindowProcA(h, m, w, l);
}

static const int OVERLAY_COUNT = 15;
static const int CHECK_INTERVAL = 30; // self-read cadence, in frames

// deterministic color for a given integer key -- shared by the producer
// (what's drawn) and every verifier (what's expected). NOT a function of
// wall-clock time: two independent process runs must draw byte-identical
// sequences when indexed by the same integer.
static void colorForKey(uint32_t key, float rgba[4]) {
  double a = key * 0.6180339887; // golden-ratio walk, decorrelates nearby keys
  rgba[0] = 0.5f + 0.5f * (float)sin(6.28318530718 * a);
  rgba[1] = 0.5f + 0.5f * (float)sin(6.28318530718 * a + 2.09439510239);
  rgba[2] = 0.5f + 0.5f * (float)sin(6.28318530718 * a + 4.18879020479);
  rgba[3] = 1.0f;
}

static const char* kQuadShaderSrc = R"(
cbuffer QuadCB : register(b0) { float4 color; float2 posMin; float2 posMax; };
struct VSOut { float4 pos : SV_POSITION; };
VSOut VSMain(uint vid : SV_VertexID) {
  // unit quad in [0,1], two triangles via a 4-vertex fan (vid 0..3, drawn
  // as a triangle strip) mapped into [posMin,posMax] in clip space.
  float2 uv = float2(float(vid & 1), float((vid >> 1) & 1));
  float2 p = lerp(posMin, posMax, uv);
  VSOut o;
  o.pos = float4(p, 0, 1);
  return o;
}
float4 PSMain(float4 pos : SV_POSITION) : SV_TARGET {
  return color;
}
)";

struct QuadCB { float color[4]; float posMin[2]; float posMax[2]; };

struct QuadRenderer {
  ID3D11VertexShader* vs = nullptr;
  ID3D11PixelShader* ps = nullptr;
  ID3D11Buffer* cb = nullptr;
  ID3D11BlendState* blendOpaque = nullptr;
  ID3D11BlendState* blendAlpha = nullptr;
  ID3D11RasterizerState* rs = nullptr;

  void init(ID3D11Device* dev) {
    ID3DBlob* vsBlob = nullptr; ID3DBlob* psBlob = nullptr; ID3DBlob* err = nullptr;
    HRESULT hr = D3DCompile(kQuadShaderSrc, strlen(kQuadShaderSrc), "quad", nullptr, nullptr,
      "VSMain", "vs_5_0", 0, 0, &vsBlob, &err);
    if (FAILED(hr)) { fprintf(stderr, "VS compile failed: %s\n", err ? (char*)err->GetBufferPointer() : "?"); exit(1); }
    hr = D3DCompile(kQuadShaderSrc, strlen(kQuadShaderSrc), "quad", nullptr, nullptr,
      "PSMain", "ps_5_0", 0, 0, &psBlob, &err);
    if (FAILED(hr)) { fprintf(stderr, "PS compile failed: %s\n", err ? (char*)err->GetBufferPointer() : "?"); exit(1); }
    CHECK_HR(dev->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), nullptr, &vs));
    CHECK_HR(dev->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(), nullptr, &ps));
    vsBlob->Release(); psBlob->Release();

    D3D11_BUFFER_DESC bd{}; bd.ByteWidth = sizeof(QuadCB); bd.Usage = D3D11_USAGE_DYNAMIC;
    bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER; bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    CHECK_HR(dev->CreateBuffer(&bd, nullptr, &cb));

    D3D11_BLEND_DESC bdOpaque{};
    bdOpaque.RenderTarget[0].BlendEnable = FALSE;
    bdOpaque.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    CHECK_HR(dev->CreateBlendState(&bdOpaque, &blendOpaque));

    D3D11_BLEND_DESC bdAlpha{};
    bdAlpha.RenderTarget[0].BlendEnable = TRUE;
    bdAlpha.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
    bdAlpha.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    bdAlpha.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    bdAlpha.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    bdAlpha.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
    bdAlpha.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    bdAlpha.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    CHECK_HR(dev->CreateBlendState(&bdAlpha, &blendAlpha));

    // explicit CullMode=NONE: found the hard way (round 2 of this harness --
    // the switch-off baseline itself came back all-black). D3D11's DEFAULT
    // rasterizer state culls back faces, and this was never set, so it was
    // in effect the whole time. comp_d3d11.cpp's HeavyRenderer sets this
    // for the same reason ("don't trust the default winding-vs-Y-flip
    // reasoning to necessarily hold") -- same fix here, not just for the
    // fullscreen triangle but for every quad.
    D3D11_RASTERIZER_DESC rd{}; rd.FillMode = D3D11_FILL_SOLID; rd.CullMode = D3D11_CULL_NONE; rd.DepthClipEnable = TRUE;
    CHECK_HR(dev->CreateRasterizerState(&rd, &rs));
  }

  // draws one quad. posMin/posMax are in clip space [-1,1].
  void draw(ID3D11DeviceContext* ctx, ID3D11RenderTargetView* rtv, int w, int h,
            const float color[4], float minX, float minY, float maxX, float maxY, bool blended) {
    D3D11_MAPPED_SUBRESOURCE mapped{};
    ctx->Map(cb, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
    QuadCB v{ { color[0], color[1], color[2], color[3] }, { minX, minY }, { maxX, maxY } };
    memcpy(mapped.pData, &v, sizeof(v));
    ctx->Unmap(cb, 0);
    D3D11_VIEWPORT vp{ 0, 0, (float)w, (float)h, 0, 1 };
    ctx->RSSetViewports(1, &vp);
    ctx->RSSetState(rs);
    ctx->OMSetRenderTargets(1, &rtv, nullptr);
    float blendFactor[4] = { 0, 0, 0, 0 };
    ctx->OMSetBlendState(blended ? blendAlpha : blendOpaque, blendFactor, 0xffffffff);
    ctx->IASetInputLayout(nullptr);
    ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    ctx->VSSetShader(vs, nullptr, 0);
    ctx->PSSetShader(ps, nullptr, 0);
    ctx->PSSetConstantBuffers(0, 1, &cb);
    ctx->Draw(4, 0);
  }
};

int main(int argc, char** argv) {
  // zc_test.exe <seconds> [pollMs]
  // pollMs is accepted-and-unused (round 2 dropped the in-process DDA poll
  // it used to pace -- see the file header) so run.sh's call shape doesn't
  // need to change.
  double seconds = argc > 1 ? atof(argv[1]) : 6.0;
  double pollMs  = argc > 2 ? atof(argv[2]) : 33.0;
  (void)pollMs;

  char zcEnv[8]{};
  GetEnvironmentVariableA("BLESSED_ZERO_COPY", zcEnv, sizeof(zcEnv));
  printf("BLESSED_ZERO_COPY=%s\n", zcEnv[0] ? zcEnv : "(unset)");

  const int W = 640, H = 480;
  SetProcessDPIAware();
  WNDCLASSA wc = {}; wc.lpfnWndProc = WndProc; wc.hInstance = GetModuleHandleA(nullptr);
  wc.lpszClassName = "zctest"; wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
  RegisterClassA(&wc);
  RECT wr{ 0, 0, W, H };
  AdjustWindowRect(&wr, WS_OVERLAPPEDWINDOW, FALSE);
  HWND hwnd = CreateWindowExA(0, "zctest", "zctest", WS_OVERLAPPEDWINDOW | WS_VISIBLE,
    50, 50, wr.right - wr.left, wr.bottom - wr.top, nullptr, nullptr, wc.hInstance, nullptr);
  if (!hwnd) { fprintf(stderr, "CreateWindowExA failed, GetLastError=%lu\n", GetLastError()); return 1; }
  ShowWindow(hwnd, SW_SHOW);
  UpdateWindow(hwnd);
  SetForegroundWindow(hwnd);

  DXGI_SWAP_CHAIN_DESC scd{};
  scd.BufferDesc.Width = W; scd.BufferDesc.Height = H;
  scd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
  scd.BufferDesc.RefreshRate.Numerator = 0; scd.BufferDesc.RefreshRate.Denominator = 1;
  scd.SampleDesc.Count = 1;
  scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
  scd.BufferCount = 1;
  scd.OutputWindow = hwnd;
  scd.Windowed = TRUE;
  scd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD; // single D3D back buffer, matches skyrim's shape
  scd.Flags = 0;

  ID3D11Device* dev = nullptr; ID3D11DeviceContext* ctx = nullptr; IDXGISwapChain* swapChain = nullptr;
  D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_0;
  CHECK_HR(D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, &fl, 1,
    D3D11_SDK_VERSION, &scd, &swapChain, &dev, nullptr, &ctx));

  // print which d3d11.dll actually loaded -- confirms this ran against the
  // zero-copy build, not system32's.
  HMODULE d3d11Mod = GetModuleHandleA("d3d11.dll");
  char d3d11Path[MAX_PATH]{};
  if (d3d11Mod) GetModuleFileNameA(d3d11Mod, d3d11Path, sizeof(d3d11Path));
  printf("d3d11.dll: %s\n", d3d11Path);

  QuadRenderer quad; quad.init(dev);

  // staging texture for the in-process self-read check (a), sized to the
  // whole back buffer so a partial-content bug (e.g. only some overlays
  // redirected) would show up, not just a single sample point.
  D3D11_TEXTURE2D_DESC stagingDesc{};
  stagingDesc.Width = W; stagingDesc.Height = H; stagingDesc.MipLevels = 1; stagingDesc.ArraySize = 1;
  stagingDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM; stagingDesc.SampleDesc.Count = 1;
  stagingDesc.Usage = D3D11_USAGE_STAGING; stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
  ID3D11Texture2D* selfStaging = nullptr; CHECK_HR(dev->CreateTexture2D(&stagingDesc, nullptr, &selfStaging));

  // window client-rect origin on screen -- printed so run.sh's external
  // screenshot can be cross-referenced against these coordinates by eye.
  // one point in the plain opaque area (away from any overlay), one
  // inside the first overlay's rect.
  RECT clientRect{}; GetClientRect(hwnd, &clientRect);
  POINT clientOrigin{ 0, 0 }; ClientToScreen(hwnd, &clientOrigin);
  const int opaqueSampleX = clientOrigin.x + W - 30, opaqueSampleY = clientOrigin.y + 30;
  const int overlaySampleX = clientOrigin.x + 20, overlaySampleY = clientOrigin.y + 20;
  printf("window client origin (%ld,%ld), sample points: opaque (%d,%d) overlay (%d,%d)\n",
    clientOrigin.x, clientOrigin.y, opaqueSampleX, opaqueSampleY, overlaySampleX, overlaySampleY);

  const float colorTolerance = 0.08f;

  uint32_t selfChecked = 0, selfMatched = 0;
  double sumFrameMs = 0.0, sumPresentMs = 0.0;
  uint32_t frames = 0;

  // lead's ask, round 2: cap the present rate -- uncapped DISCARD-mode
  // presentation on this card hit ~4200fps, far past the 75Hz display it
  // was meant to be watched on, and served no purpose here (the self-read
  // check is in-process and doesn't care about display refresh at all).
  // paced to ~200fps so a human or an external screenshot has a realistic
  // chance of seeing coherent, individually-presented frames.
  const double targetFrameMs = 1000.0 / 200.0;

  auto t0 = Clock::now();
  auto lastFrameTime = t0;
  auto lastFrameStart = t0;
  MSG msg;
  while (!g_quit) {
    while (PeekMessageA(&msg, nullptr, 0, 0, PM_REMOVE)) { TranslateMessage(&msg); DispatchMessageA(&msg); }
    double t = std::chrono::duration<double>(Clock::now() - t0).count();
    if (t >= seconds) break;

    ID3D11Texture2D* backTex = nullptr; CHECK_HR(swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&backTex));
    ID3D11RenderTargetView* backRtv = nullptr; CHECK_HR(dev->CreateRenderTargetView(backTex, nullptr, &backRtv));

    float opaqueColor[4]; colorForKey(frames * 1000u, opaqueColor);
    float clip0[4] = { -1, -1, 1, 1 };
    quad.draw(ctx, backRtv, W, H, opaqueColor, clip0[0], clip0[1], clip0[2], clip0[3], false);

    for (int k = 0; k < OVERLAY_COUNT; k++) {
      float ocolor[4]; colorForKey(frames * 1000u + k + 1, ocolor);
      ocolor[3] = 0.5f; // constant alpha, matches "blended ui/overlay draws"
      // spread overlays across the top-left corner in a small grid so the
      // fixed overlay sample point (20,20) is covered by every one of them,
      // stacking in submission order -- mirrors "15 blended draws on top of
      // the opaque draw" rather than 15 non-overlapping rects.
      float minX = -1.0f + 0.02f * k, minY = 1.0f - 0.12f - 0.02f * k;
      float maxX = minX + 0.10f, maxY = 1.0f - 0.02f * k;
      quad.draw(ctx, backRtv, W, H, ocolor, minX, minY, maxX, maxY, true);
    }

    // check (a): self-read, every CHECK_INTERVAL frames. This IS a read of
    // the back buffer -- exercises BlessedCorrectBackBufferForRead when a
    // redirect is active. Frame-exact: same process, no capture-timing slop.
    if (frames % CHECK_INTERVAL == 0) {
      ctx->CopyResource(selfStaging, backTex);
      D3D11_MAPPED_SUBRESOURCE m{};
      if (SUCCEEDED(ctx->Map(selfStaging, 0, D3D11_MAP_READ, 0, &m))) {
        BYTE* row = (BYTE*)m.pData + (H - 31) * m.RowPitch; // near (30, H-31) -> opaque sample area
        BYTE* px = row + (W - 30) * 4;
        float r = px[0] / 255.0f, g = px[1] / 255.0f, b = px[2] / 255.0f; // R8G8B8A8
        ctx->Unmap(selfStaging, 0);
        selfChecked++;
        bool match = fabsf(r - opaqueColor[0]) < colorTolerance
                  && fabsf(g - opaqueColor[1]) < colorTolerance
                  && fabsf(b - opaqueColor[2]) < colorTolerance;
        if (match) selfMatched++;
        else printf("SELF-READ MISMATCH frame %u: expected (%.3f,%.3f,%.3f) got (%.3f,%.3f,%.3f)\n",
          frames, opaqueColor[0], opaqueColor[1], opaqueColor[2], r, g, b);
      }
    }

    auto beforePresent = Clock::now();
    HRESULT pr = swapChain->Present(0, 0);
    auto afterPresent = Clock::now();
    if (FAILED(pr)) { fprintf(stderr, "Present failed: 0x%08lx at frame %u\n", (unsigned long)pr, frames); backRtv->Release(); backTex->Release(); break; }
    sumPresentMs += std::chrono::duration<double, std::milli>(afterPresent - beforePresent).count();
    sumFrameMs += std::chrono::duration<double, std::milli>(afterPresent - lastFrameTime).count();
    lastFrameTime = afterPresent;

    backRtv->Release(); backTex->Release();

    // lead's ask, round 2: pace to ~200fps. sleep off whatever's left of
    // the target frame budget after everything above (draws, self-read,
    // Present); never sleeps negative, never blocks past the deadline.
    double sinceFrameStart = std::chrono::duration<double, std::milli>(Clock::now() - lastFrameStart).count();
    if (sinceFrameStart < targetFrameMs)
      Sleep((DWORD)(targetFrameMs - sinceFrameStart));
    lastFrameStart = Clock::now();

    frames++;
  }

  double el = std::chrono::duration<double>(Clock::now() - t0).count();
  printf("frames %u seconds %.3f fps %.1f\n", frames, el, frames / el);
  printf("avg frame time %.4fms, avg Present() call time %.4fms\n",
    frames ? sumFrameMs / frames : 0.0, frames ? sumPresentMs / frames : 0.0);
  printf("self-read (fallback path, in-process, frame-exact): checked %u matched %u\n", selfChecked, selfMatched);
  printf("verdict: %s\n",
    (selfChecked > 0 && selfMatched == selfChecked) ? "PASS" : "FAIL (see counts above)");

  DestroyWindow(hwnd);
  return 0;
}
