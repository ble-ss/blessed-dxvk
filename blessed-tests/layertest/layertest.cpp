// perlayer seat: synthetic replay of skyrim's layered render targets (a render pass per layer with a clear, sampled afterwards)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>

// modes:
//   0 cube:    rgba16f colour array + d16 depth array, both one layer per pass (fully disjoint)
//   1 cascade: d16 depth array only, one layer per pass (the sun cascades)
//   2 water:   rgba16f colour array + one shared single-layer depth (skyrim's water cube)
//   3 mixed:   mode 0, but pass k samples layer k-1 of the same image, and layer 0 is rendered twice
//              (exercises the fall-back: the per-layer record must see the sampled and repeated layers)
static const char* g_hlsl = R"(
cbuffer C : register(b0) { float4 color; float4 ofs; };
struct V { float4 p : SV_Position; float2 uv : TEXCOORD0; };
SamplerState smp_ : register(s1);
V vsTri(uint id : SV_VertexID) {
  V o; float2 uv = float2((id << 1) & 2, id & 2);
  o.uv = uv; o.p = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1); return o; }
V vsQuad(uint id : SV_VertexID, uint inst : SV_InstanceID) {
  // a small centred quad, 1/4 of the target, depth ofs.z
  V o; float2 uv = float2((id << 1) & 2, id & 2) * 0.5;
  o.uv = uv; o.p = float4((uv - 0.5) * 0.5 + ofs.xy * 0.0, ofs.z, 1); return o; }
float4 psSolid(V i) : SV_Target { return color; }
Texture2DArray<float4> prev : register(t1);
float4 psMix(V i) : SV_Target { return color + prev.Sample(smp_, float3(i.uv, ofs.w)) * 0.0001; }
Texture2DArray<float4> arr : register(t0);
SamplerState smp : register(s0);
float4 psCompose(V i) : SV_Target {
  float4 acc = 0;
  [loop] for (uint k = 0; k < (uint)ofs.w; k++) acc += arr.Sample(smp, float3(i.uv, k));
  return acc / max(ofs.w, 1); }
)";

struct Cb { float color[4]; float ofs[4]; };

static bool g_quit = false;
static LRESULT CALLBACK WndProc(HWND h, UINT m, WPARAM w, LPARAM l) {
  if (m == WM_DESTROY || (m == WM_KEYDOWN && w == VK_ESCAPE)) { g_quit = true; PostQuitMessage(0); return 0; }
  return DefWindowProcA(h, m, w, l);
}

static ID3DBlob* Compile(const char* entry, const char* target) {
  ID3DBlob* b = nullptr; ID3DBlob* e = nullptr;
  if (FAILED(D3DCompile(g_hlsl, strlen(g_hlsl), nullptr, nullptr, nullptr, entry, target, 0, 0, &b, &e))) {
    fprintf(stderr, "compile %s: %s\n", entry, e ? (const char*)e->GetBufferPointer() : "?");
    exit(1);
  }
  return b;
}

static float ClearValue(UINT k) { return 0.05f + 0.1f * float(k); }
static float DrawValue(UINT k)  { return 0.5f + 0.05f * float(k); }

static void Check(HRESULT hr, const char* what) {
  if (FAILED(hr)) { fprintf(stderr, "%s failed: 0x%08lx\n", what, hr); exit(1); }
}

int main(int argc, char** argv) {
  // layertest <seconds> <mode> <drawsPerLayer> <layers> <size> [verifyEvery]
  double seconds = argc > 1 ? atof(argv[1]) : 10.0;
  int mode       = argc > 2 ? atoi(argv[2]) : 0;
  int draws      = argc > 3 ? atoi(argv[3]) : 40;
  UINT layers    = argc > 4 ? (UINT)atoi(argv[4]) : 6u;
  UINT size      = argc > 5 ? (UINT)atoi(argv[5]) : 512u;
  int verifyEvery = argc > 6 ? atoi(argv[6]) : 0;

  bool hasColor = mode != 1;
  bool depthArray = mode != 2;
  if (mode == 3 && layers < 2) layers = 2;

  SetProcessDPIAware();
  const UINT W = 1280, H = 720;
  WNDCLASSA wc = { }; wc.lpfnWndProc = WndProc; wc.hInstance = GetModuleHandleA(nullptr);
  wc.lpszClassName = "layertest"; wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
  RegisterClassA(&wc);
  HWND hwnd = CreateWindowExA(0, "layertest", "layertest", WS_OVERLAPPEDWINDOW | WS_VISIBLE,
    40, 40, W, H, nullptr, nullptr, wc.hInstance, nullptr);

  DXGI_SWAP_CHAIN_DESC sd = { };
  sd.BufferCount = 2; sd.BufferDesc.Width = W; sd.BufferDesc.Height = H;
  sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
  sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT; sd.OutputWindow = hwnd;
  sd.SampleDesc.Count = 1; sd.Windowed = TRUE; sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;

  ID3D11Device* dev = nullptr; ID3D11DeviceContext* ctx = nullptr; IDXGISwapChain* sc = nullptr;
  D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_0;
  Check(D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, &fl, 1,
    D3D11_SDK_VERSION, &sd, &sc, &dev, nullptr, &ctx), "device");

  ID3D11Texture2D* bb = nullptr; sc->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&bb);
  ID3D11RenderTargetView* bbRtv = nullptr; dev->CreateRenderTargetView(bb, nullptr, &bbRtv);

  // layered colour target (a cube when 6 layers, like the water reflection)
  ID3D11Texture2D* colorTex = nullptr;
  std::vector<ID3D11RenderTargetView*> colorRtv(layers, nullptr);
  ID3D11ShaderResourceView* colorSrv = nullptr;
  std::vector<ID3D11ShaderResourceView*> sliceSrv(layers, nullptr);

  if (hasColor) {
    D3D11_TEXTURE2D_DESC td = { };
    td.Width = size; td.Height = size; td.MipLevels = 1; td.ArraySize = layers;
    td.Format = DXGI_FORMAT_R16G16B16A16_FLOAT; td.SampleDesc.Count = 1;
    td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    td.MiscFlags = layers == 6 ? D3D11_RESOURCE_MISC_TEXTURECUBE : 0;
    Check(dev->CreateTexture2D(&td, nullptr, &colorTex), "color tex");

    for (UINT k = 0; k < layers; k++) {
      D3D11_RENDER_TARGET_VIEW_DESC rd = { };
      rd.Format = td.Format; rd.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DARRAY;
      rd.Texture2DArray.FirstArraySlice = k; rd.Texture2DArray.ArraySize = 1;
      Check(dev->CreateRenderTargetView(colorTex, &rd, &colorRtv[k]), "rtv");
    }

    D3D11_SHADER_RESOURCE_VIEW_DESC vd = { };
    vd.Format = td.Format; vd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
    vd.Texture2DArray.MipLevels = 1; vd.Texture2DArray.ArraySize = layers;
    Check(dev->CreateShaderResourceView(colorTex, &vd, &colorSrv), "color srv");

    for (UINT k = 0; k < layers; k++) {
      vd.Texture2DArray.FirstArraySlice = k; vd.Texture2DArray.ArraySize = 1;
      Check(dev->CreateShaderResourceView(colorTex, &vd, &sliceSrv[k]), "slice srv");
    }
  }

  // depth: an array with one layer per pass (cube, cascade) or one shared layer (water)
  UINT depthLayers = depthArray ? layers : 1u;
  ID3D11Texture2D* depthTex = nullptr;
  std::vector<ID3D11DepthStencilView*> dsv(depthLayers, nullptr);
  ID3D11ShaderResourceView* depthSrv = nullptr;
  {
    D3D11_TEXTURE2D_DESC td = { };
    td.Width = size; td.Height = size; td.MipLevels = 1; td.ArraySize = depthLayers;
    td.Format = DXGI_FORMAT_R16_TYPELESS; td.SampleDesc.Count = 1;
    td.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;
    Check(dev->CreateTexture2D(&td, nullptr, &depthTex), "depth tex");

    for (UINT k = 0; k < depthLayers; k++) {
      D3D11_DEPTH_STENCIL_VIEW_DESC dd = { };
      dd.Format = DXGI_FORMAT_D16_UNORM; dd.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DARRAY;
      dd.Texture2DArray.FirstArraySlice = k; dd.Texture2DArray.ArraySize = 1;
      Check(dev->CreateDepthStencilView(depthTex, &dd, &dsv[k]), "dsv");
    }

    D3D11_SHADER_RESOURCE_VIEW_DESC vd = { };
    vd.Format = DXGI_FORMAT_R16_UNORM; vd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
    vd.Texture2DArray.MipLevels = 1; vd.Texture2DArray.ArraySize = depthLayers;
    Check(dev->CreateShaderResourceView(depthTex, &vd, &depthSrv), "depth srv");
  }

  // staging copies for the verify step
  ID3D11Texture2D* colorStage = nullptr; ID3D11Texture2D* depthStage = nullptr;
  if (verifyEvery) {
    D3D11_TEXTURE2D_DESC td = { };
    if (hasColor) {
      colorTex->GetDesc(&td); td.BindFlags = 0; td.MiscFlags = 0;
      td.Usage = D3D11_USAGE_STAGING; td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      Check(dev->CreateTexture2D(&td, nullptr, &colorStage), "color stage");
    }
    depthTex->GetDesc(&td); td.BindFlags = 0; td.MiscFlags = 0;
    td.Usage = D3D11_USAGE_STAGING; td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    Check(dev->CreateTexture2D(&td, nullptr, &depthStage), "depth stage");
  }

  ID3DBlob* b;
  ID3D11VertexShader* vsTri; b = Compile("vsTri", "vs_5_0"); dev->CreateVertexShader(b->GetBufferPointer(), b->GetBufferSize(), nullptr, &vsTri);
  ID3D11VertexShader* vsQuad; b = Compile("vsQuad", "vs_5_0"); dev->CreateVertexShader(b->GetBufferPointer(), b->GetBufferSize(), nullptr, &vsQuad);
  ID3D11PixelShader* psSolid; b = Compile("psSolid", "ps_5_0"); dev->CreatePixelShader(b->GetBufferPointer(), b->GetBufferSize(), nullptr, &psSolid);
  ID3D11PixelShader* psMix; b = Compile("psMix", "ps_5_0"); dev->CreatePixelShader(b->GetBufferPointer(), b->GetBufferSize(), nullptr, &psMix);
  ID3D11PixelShader* psCompose; b = Compile("psCompose", "ps_5_0"); dev->CreatePixelShader(b->GetBufferPointer(), b->GetBufferSize(), nullptr, &psCompose);

  D3D11_BUFFER_DESC bd = { }; bd.ByteWidth = sizeof(Cb); bd.Usage = D3D11_USAGE_DYNAMIC;
  bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER; bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
  ID3D11Buffer* cb = nullptr; dev->CreateBuffer(&bd, nullptr, &cb);

  D3D11_SAMPLER_DESC smd = { }; smd.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
  smd.AddressU = smd.AddressV = smd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP; smd.MaxLOD = D3D11_FLOAT32_MAX;
  ID3D11SamplerState* smp = nullptr; dev->CreateSamplerState(&smd, &smp);

  D3D11_DEPTH_STENCIL_DESC dsd = { }; dsd.DepthEnable = TRUE; dsd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
  dsd.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
  ID3D11DepthStencilState* dss = nullptr; dev->CreateDepthStencilState(&dsd, &dss);
  D3D11_DEPTH_STENCIL_DESC dsdOff = { };
  ID3D11DepthStencilState* dssOff = nullptr; dev->CreateDepthStencilState(&dsdOff, &dssOff);

  D3D11_RASTERIZER_DESC rsd = { }; rsd.FillMode = D3D11_FILL_SOLID; rsd.CullMode = D3D11_CULL_NONE; rsd.DepthClipEnable = TRUE;
  ID3D11RasterizerState* rs = nullptr; dev->CreateRasterizerState(&rsd, &rs);

  // gpu timing: a timestamp before and after the layer loop, and one at frame end
  const int Ring = 6;
  ID3D11Query* qDisjoint[Ring]; ID3D11Query* qStart[Ring]; ID3D11Query* qEnd[Ring]; ID3D11Query* qFrame[Ring];
  for (int i = 0; i < Ring; i++) {
    D3D11_QUERY_DESC qd = { D3D11_QUERY_TIMESTAMP_DISJOINT, 0 }; dev->CreateQuery(&qd, &qDisjoint[i]);
    qd.Query = D3D11_QUERY_TIMESTAMP;
    dev->CreateQuery(&qd, &qStart[i]); dev->CreateQuery(&qd, &qEnd[i]); dev->CreateQuery(&qd, &qFrame[i]);
  }

  std::vector<double> loopMs, periodMs;
  UINT64 lastFrameTs = 0; bool haveLast = false;
  long frames = 0, verifyFails = 0, verifies = 0;

  auto setCb = [&](float r, float g, float bl, float a, float z, float n) {
    D3D11_MAPPED_SUBRESOURCE m; ctx->Map(cb, 0, D3D11_MAP_WRITE_DISCARD, 0, &m);
    Cb c = { { r, g, bl, a }, { 0, 0, z, n } }; memcpy(m.pData, &c, sizeof(c)); ctx->Unmap(cb, 0);
  };

  auto t0 = std::chrono::steady_clock::now();
  MSG msg;

  while (!g_quit) {
    while (PeekMessageA(&msg, nullptr, 0, 0, PM_REMOVE)) { TranslateMessage(&msg); DispatchMessageA(&msg); }
    double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (el >= seconds) break;

    int slot = frames % Ring;
    ctx->Begin(qDisjoint[slot]);
    ctx->End(qStart[slot]);

    // the layered passes: clear the layer, then tens of small draws into it
    ID3D11ShaderResourceView* nullSrv[1] = { nullptr };
    ctx->PSSetShaderResources(0, 1, nullSrv);
    ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ctx->IASetInputLayout(nullptr);
    ctx->RSSetState(rs);
    ctx->OMSetDepthStencilState(dss, 0);
    D3D11_VIEWPORT vp = { 0, 0, float(size), float(size), 0, 1 };
    ctx->RSSetViewports(1, &vp);
    ctx->VSSetShader(vsQuad, nullptr, 0);
    ctx->PSSetShader(hasColor ? psSolid : nullptr, nullptr, 0);
    ctx->VSSetConstantBuffers(0, 1, &cb);
    ctx->PSSetConstantBuffers(0, 1, &cb);

    UINT passes = mode == 3 ? layers + 1 : layers;
    ctx->PSSetSamplers(1, 1, &smp);

    for (UINT p = 0; p < passes; p++) {
      UINT k = p % layers;
      ID3D11DepthStencilView* d = dsv[depthArray ? k : 0];

      if (mode == 3) {
        ID3D11ShaderResourceView* none = nullptr;
        ctx->PSSetShaderResources(1, 1, &none);
        ctx->PSSetShader(k ? psMix : psSolid, nullptr, 0);
      }

      if (hasColor) {
        float cc[4] = { ClearValue(k), 0.f, 0.f, 1.f };
        ctx->ClearRenderTargetView(colorRtv[k], cc);
      }
      ctx->ClearDepthStencilView(d, D3D11_CLEAR_DEPTH, 1.0f, 0);
      ctx->OMSetRenderTargets(hasColor ? 1 : 0, hasColor ? &colorRtv[k] : nullptr, d);

      if (mode == 3 && k)
        ctx->PSSetShaderResources(1, 1, &sliceSrv[k - 1]);

      for (int i = 0; i < draws; i++) {
        float z = 0.9f - 0.8f * float(i) / float(std::max(draws, 1));
        setCb(DrawValue(k), 0.25f, 0.f, 1.f, z, 0.f);
        ctx->Draw(6, 0);
      }
    }

    ctx->End(qEnd[slot]);

    // sample every layer afterwards
    ctx->OMSetRenderTargets(1, &bbRtv, nullptr);
    ctx->OMSetDepthStencilState(dssOff, 0);
    D3D11_VIEWPORT bvp = { 0, 0, float(W), float(H), 0, 1 };
    ctx->RSSetViewports(1, &bvp);
    ctx->VSSetShader(vsTri, nullptr, 0);
    ctx->PSSetShader(psCompose, nullptr, 0);
    setCb(0, 0, 0, 0, 0, float(hasColor ? layers : depthLayers));
    ctx->PSSetShaderResources(1, 1, nullSrv);
    ctx->PSSetShaderResources(0, 1, hasColor ? &colorSrv : &depthSrv);
    ctx->PSSetSamplers(0, 1, &smp);
    ctx->Draw(3, 0);
    ctx->PSSetShaderResources(0, 1, nullSrv);

    ctx->End(qFrame[slot]);
    ctx->End(qDisjoint[slot]);

    // correctness: every layer holds its clear at the corner and its draw in the centre
    if (verifyEvery && frames % verifyEvery == verifyEvery - 1) {
      verifies++;
      if (hasColor) {
        ctx->CopyResource(colorStage, colorTex);
        for (UINT k = 0; k < layers; k++) {
          D3D11_MAPPED_SUBRESOURCE m;
          Check(ctx->Map(colorStage, D3D11CalcSubresource(0, k, 1), D3D11_MAP_READ, 0, &m), "map color");
          auto px = [&](UINT x, UINT y) { return ((const uint16_t*)((const char*)m.pData + y * m.RowPitch))[x * 4]; };
          uint16_t corner = px(1, 1), centre = px(size / 2, size / 2);
          // half floats: compare against a float->half of the expected values
          auto h2f = [](uint16_t h) { UINT e = (h >> 10) & 31, f = h & 1023; return e ? std::ldexp(1.f + f / 1024.f, int(e) - 15) : std::ldexp(f / 1024.f, -14); };
          if (std::fabs(h2f(corner) - ClearValue(k)) > 0.002f || std::fabs(h2f(centre) - DrawValue(k)) > 0.002f) {
            if (verifyFails < 20) fprintf(stderr, "verify: frame %ld layer %u colour corner %f centre %f\n", frames, k, h2f(corner), h2f(centre));
            verifyFails++;
          }
          ctx->Unmap(colorStage, D3D11CalcSubresource(0, k, 1));
        }
      }
      ctx->CopyResource(depthStage, depthTex);
      for (UINT k = 0; k < depthLayers; k++) {
        D3D11_MAPPED_SUBRESOURCE m;
        Check(ctx->Map(depthStage, D3D11CalcSubresource(0, k, 1), D3D11_MAP_READ, 0, &m), "map depth");
        auto px = [&](UINT x, UINT y) { return ((const uint16_t*)((const char*)m.pData + y * m.RowPitch))[x]; };
        float corner = px(1, 1) / 65535.f, centre = px(size / 2, size / 2) / 65535.f;
        float zMin = 0.9f - 0.8f * float(draws - 1) / float(std::max(draws, 1));
        if (std::fabs(corner - 1.f) > 0.001f || std::fabs(centre - zMin) > 0.001f) {
          if (verifyFails < 20) fprintf(stderr, "verify: frame %ld layer %u depth corner %f centre %f (want %f)\n", frames, k, corner, centre, zMin);
          verifyFails++;
        }
        ctx->Unmap(depthStage, D3D11CalcSubresource(0, k, 1));
      }
    }

    sc->Present(0, 0);
    frames++;

    // read back the oldest slot
    if (frames >= Ring) {
      int rs2 = frames % Ring;
      D3D11_QUERY_DATA_TIMESTAMP_DISJOINT dj;
      while (ctx->GetData(qDisjoint[rs2], &dj, sizeof(dj), 0) == S_FALSE) { }
      UINT64 a, e, f;
      ctx->GetData(qStart[rs2], &a, sizeof(a), 0);
      ctx->GetData(qEnd[rs2], &e, sizeof(e), 0);
      ctx->GetData(qFrame[rs2], &f, sizeof(f), 0);
      if (!dj.Disjoint && el > 1.0) {
        loopMs.push_back(double(e - a) * 1000.0 / double(dj.Frequency));
        if (haveLast) periodMs.push_back(double(f - lastFrameTs) * 1000.0 / double(dj.Frequency));
      }
      lastFrameTs = f; haveLast = true;
    }
  }

  auto stats = [](std::vector<double>& v, double& mean, double& med, double& p90) {
    if (v.empty()) { mean = med = p90 = 0; return; }
    double s = 0; for (double x : v) s += x; mean = s / v.size();
    std::sort(v.begin(), v.end()); med = v[v.size() / 2]; p90 = v[v.size() * 9 / 10];
  };
  double lm, lmed, lp90, pm, pmed, pp90;
  stats(loopMs, lm, lmed, lp90); stats(periodMs, pm, pmed, pp90);
  double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  printf("layertest mode=%d draws=%d layers=%u size=%u frames=%ld fps=%.1f\n", mode, draws, layers, size, frames, frames / el);
  printf("layer-loop gpu ms: mean %.4f median %.4f p90 %.4f (n=%zu)\n", lm, lmed, lp90, loopMs.size());
  printf("frame period gpu ms: mean %.4f median %.4f p90 %.4f (n=%zu)\n", pm, pmed, pp90, periodMs.size());
  printf("verify: %ld checks, %ld failures\n", verifies, verifyFails);
  return verifyFails ? 2 : 0;
}
