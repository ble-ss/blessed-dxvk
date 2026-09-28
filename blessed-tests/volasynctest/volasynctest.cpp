// volasync seat: synthetic replay of vanilla's froxel window (cascade atlas -> generate -> 90-step ping-pong chain -> raster -> consumer)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <vector>
#include <algorithm>
#include <string>

// The shape of whiterun-dump-1 frame 1920, passes 7-138:
//   cascades   draws into a 4096x4096 d16 atlas (R16_TYPELESS, sampled as R16_UNORM)
//   generate   cs, srv t0 = atlas, t1 = 512x512 tex2d, t2 = 4096 tex1d lut, t3 = 32^3 tex3d noise,
//              uav u0/u1 = the two 320x192x90 R16_FLOAT froxel volumes, 10x6x90 groups
//   chain      cs x steps, a WRITE_DISCARD map before each (the dump's map_type 4 on one cbuffer),
//              srv t0 = one volume, uav u0 = the other, 10x6x1 groups, ping-pong
//   raster     a depth clear still inside the window (the dump's clear_dsv, i=6251), then draws
//              that touch neither volume (the depth prepass, the masks, the lit pass)
//   consumer   the wait draw: ps srv t2 = the final volume (uav0 of generate)
// Every constant a dispatch reads comes through a dynamic cbuffer mapped with WRITE_DISCARD, so
// with d3d11.blessedCbRing + d3d11.blessedCbMirror they take the ring + vram mirror path. The
// generate constants are mapped at the top of the frame, before the cascades: their mirror copy
// is still pending when the async window opens (the readiness fix's case); the chain's are mapped
// inside the window (the host-block fallback's case). A default-usage cbuffer is also written with
// UpdateSubresource inside the window, right before the chain (vol-collapse's own params buffer
// does exactly this in game). All inputs depend only on the frame index,
// so a per-frame checksum of the final volume must match across sync, =1 and =2.
static const char* g_hlsl = R"(
cbuffer Gen : register(b0) { float4 g0; float4 g1; };
cbuffer Step : register(b1) { uint4 s0; float4 s1; };
cbuffer Draw : register(b2) { float4 d0; float4 d1; };
cbuffer Param : register(b3) { float4 p0; };

Texture2D<float>   atlas : register(t0);
Texture2D<float4>  aux   : register(t1);
Texture1D<float4>  lut   : register(t2);
Texture3D<float4>  noise : register(t3);
SamplerState       lin   : register(s0);
RWTexture3D<float> volA  : register(u0);
RWTexture3D<float> volB  : register(u1);

[numthreads(32, 32, 1)]
void csGenerate(uint3 id : SV_DispatchThreadID) {
  float2 uv = (float2(id.xy) + 0.5) / float2(320.0, 192.0);
  float acc = 0.0;
  [unroll] for (uint t = 0; t < 8; t++) {
    float2 o = float2(t & 3, t >> 2) * (1.0 / 4096.0) * 7.0;
    acc += atlas.SampleLevel(lin, uv + o + g0.xy, 0.0);
  }
  float zf = (float(id.z) + 0.5) / 90.0;
  float4 a = aux.SampleLevel(lin, uv, 0.0);
  float4 l = lut.SampleLevel(lin, zf + g0.z, 0.0);
  float4 n = noise.SampleLevel(lin, float3(uv * 4.0, zf + g0.w), 0.0);
  float v = acc * 0.125 * g1.x + a.x * g1.y + l.y * g1.z + n.z * g1.w + zf;
  volA[id] = v;
  volB[id] = v;
}

Texture3D<float>   src : register(t0);
RWTexture3D<float> dst : register(u0);

// step k: dst holds slices 0..k-2 from two steps ago; write k-1 (carried) and k (integrated)
[numthreads(32, 32, 1)]
void csChain(uint3 id : SV_DispatchThreadID) {
  uint k = s0.x;
  if (k == 0u) { dst[uint3(id.xy, 0)] = src[uint3(id.xy, 0)]; return; }
  float prev = src[uint3(id.xy, k - 1u)];
  dst[uint3(id.xy, k - 1u)] = prev;
  dst[uint3(id.xy, k)] = src[uint3(id.xy, k)] * p0.x + prev * s1.x;
}

struct V { float4 p : SV_Position; float2 uv : TEXCOORD0; };
V vsTri(uint id : SV_VertexID) {
  V o; float2 uv = float2((id << 1) & 2, id & 2);
  o.uv = uv; o.p = float4(uv * float2(2, -2) + float2(-1, 1), d0.z, 1); return o; }
V vsCascade(uint id : SV_VertexID) {
  // a fan of slanted triangles over one quadrant (d0.xy = quadrant origin in ndc), depth from d0.z/d0.w
  uint tri = id / 3u, c = id % 3u;
  float fx = float(tri % 8u) / 8.0, fy = float(tri / 8u) / 8.0;
  float2 p = float2(fx, fy) + (c == 0u ? float2(0, 0) : c == 1u ? float2(0.2, 0) : float2(0, 0.2));
  V o; o.uv = p; o.p = float4(d0.xy + p, saturate(d0.z + d0.w * (p.x - p.y)), 1); return o; }
float4 psRaster(V i) : SV_Target {
  float4 c = float4(i.uv, d0.w, 1);
  [loop] for (uint k = 0; k < (uint)d1.x; k++) c = frac(c * 1.0137 + c.yzwx * 0.371 + 0.01);
  return c; }

Texture2D<float4> scene : register(t0);
Texture3D<float>  fog   : register(t2);
float4 psConsumer(V i) : SV_Target {
  float f = fog.SampleLevel(lin, float3(i.uv, 0.5), 0.0) + fog.SampleLevel(lin, float3(i.uv, 0.98), 0.0);
  return scene.SampleLevel(lin, i.uv, 0.0) * 0.5 + f * 0.01; }

// the consumer-time check: right after the wait draw, what does the graphics queue see in the
// final volume? one pixel per froxel column, every slice folded in
float psCheck(V i) : SV_Target {
  uint2 c = uint2(i.p.xy);
  float acc = 0.0;
  [loop] for (uint z = 0; z < 90u; z++) acc = acc * 1.0009765625 + fog.Load(int4(c, z, 0));
  return acc; }
)";

struct Cb { float a[4]; float b[4]; };

static bool g_quit = false;
static LRESULT CALLBACK WndProc(HWND h, UINT m, WPARAM w, LPARAM l) {
  if (m == WM_DESTROY || (m == WM_KEYDOWN && w == VK_ESCAPE)) { g_quit = true; PostQuitMessage(0); return 0; }
  return DefWindowProcA(h, m, w, l);
}

static void Check(HRESULT hr, const char* what) {
  if (FAILED(hr)) { fprintf(stderr, "%s failed: 0x%08lx\n", what, hr); exit(1); }
}

static ID3DBlob* Compile(const char* entry, const char* target) {
  ID3DBlob* b = nullptr; ID3DBlob* e = nullptr;
  if (FAILED(D3DCompile(g_hlsl, strlen(g_hlsl), nullptr, nullptr, nullptr, entry, target, 0, 0, &b, &e))) {
    fprintf(stderr, "compile %s: %s\n", entry, e ? (const char*)e->GetBufferPointer() : "?");
    exit(1);
  }
  return b;
}

// dxvk names a shader "<stage>.<hex of the dxbc container hash>" (DxvkShaderHash::toString: the
// 16 checksum bytes at offset 4, in byte order)
static std::string DxvkHash(ID3DBlob* b) {
  const uint8_t* p = (const uint8_t*)b->GetBufferPointer();
  static const char* hex = "0123456789abcdef";
  std::string s;
  for (int i = 4; i < 20; i++) { s += hex[p[i] >> 4]; s += hex[p[i] & 15]; }
  return s;
}

// frame-indexed, deterministic "random" in [0,1)
static float Rnd(uint32_t a, uint32_t b) {
  uint32_t h = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA77u;
  h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
  return float(h >> 8) / 16777216.0f;
}

int main(int argc, char** argv) {
  // volasynctest <seconds> <maxFrames> <chainSteps> <rasterDraws> <rasterIters> <checkEvery> [uploadMB]
  double seconds   = argc > 1 ? atof(argv[1]) : 10.0;
  long   maxFrames = argc > 2 ? atol(argv[2]) : 0;
  UINT   steps     = argc > 3 ? (UINT)atoi(argv[3]) : 90u;
  int    rasterDraws = argc > 4 ? atoi(argv[4]) : 8;
  int    rasterIters = argc > 5 ? atoi(argv[5]) : 64;
  int    checkEvery  = argc > 6 ? atoi(argv[6]) : 0;
  // a large upload late in the frame: dxvk puts it on the transfer queue, ahead of the cb mirror
  // copies recorded when the list is flushed, so those copies land late -- the way a streaming
  // upload does in game. Without it the transfer queue is idle and a late mirror copy still beats
  // the kick, which hides a missing readiness flush.
  UINT   uploadMB    = argc > 7 ? (UINT)atoi(argv[7]) : 0u;
  steps = std::min(std::max(steps, 1u), 90u);

  ID3DBlob* bGen  = Compile("csGenerate", "cs_5_0");
  ID3DBlob* bCh   = Compile("csChain", "cs_5_0");
  ID3DBlob* bVs   = Compile("vsTri", "vs_5_0");
  ID3DBlob* bVsC  = Compile("vsCascade", "vs_5_0");
  ID3DBlob* bPsR  = Compile("psRaster", "ps_5_0");
  ID3DBlob* bPsC  = Compile("psConsumer", "ps_5_0");
  ID3DBlob* bPsK  = Compile("psCheck", "ps_5_0");

  // point the fork's hooks at this test's own shaders, unless the caller already did
  std::string genHash = DxvkHash(bGen), waitHash = DxvkHash(bPsC);
  char tmp[8];
  if (!GetEnvironmentVariableA("BLESSED_VOL_ASYNC_GEN_CS", tmp, sizeof(tmp)))
    SetEnvironmentVariableA("BLESSED_VOL_ASYNC_GEN_CS", genHash.c_str());
  if (!GetEnvironmentVariableA("BLESSED_VOL_ASYNC_WAIT_PS", tmp, sizeof(tmp)))
    SetEnvironmentVariableA("BLESSED_VOL_ASYNC_WAIT_PS", waitHash.c_str());
  printf("shaders: generate cs.%s chain cs.%s consumer fs.%s\n", genHash.c_str(), DxvkHash(bCh).c_str(), waitHash.c_str());
  fflush(stdout);

  SetProcessDPIAware();
  const UINT W = 1920, H = 1080;
  WNDCLASSA wc = { }; wc.lpfnWndProc = WndProc; wc.hInstance = GetModuleHandleA(nullptr);
  wc.lpszClassName = "volasynctest"; wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
  RegisterClassA(&wc);
  HWND hwnd = CreateWindowExA(0, "volasynctest", "volasynctest", WS_OVERLAPPEDWINDOW | WS_VISIBLE,
    40, 40, 960, 540, nullptr, nullptr, wc.hInstance, nullptr);

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

  // the cascade atlas
  ID3D11Texture2D* atlasTex = nullptr; ID3D11DepthStencilView* atlasDsv = nullptr; ID3D11ShaderResourceView* atlasSrv = nullptr;
  {
    D3D11_TEXTURE2D_DESC td = { };
    td.Width = 4096; td.Height = 4096; td.MipLevels = 1; td.ArraySize = 1;
    td.Format = DXGI_FORMAT_R16_TYPELESS; td.SampleDesc.Count = 1;
    td.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;
    Check(dev->CreateTexture2D(&td, nullptr, &atlasTex), "atlas");
    D3D11_DEPTH_STENCIL_VIEW_DESC dd = { }; dd.Format = DXGI_FORMAT_D16_UNORM; dd.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
    Check(dev->CreateDepthStencilView(atlasTex, &dd, &atlasDsv), "atlas dsv");
    D3D11_SHADER_RESOURCE_VIEW_DESC vd = { }; vd.Format = DXGI_FORMAT_R16_UNORM; vd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    vd.Texture2D.MipLevels = 1;
    Check(dev->CreateShaderResourceView(atlasTex, &vd, &atlasSrv), "atlas srv");
  }

  // static inputs: 512^2 aux, 4096 lut, 32^3 noise (immutable, filled once)
  ID3D11ShaderResourceView *auxSrv = nullptr, *lutSrv = nullptr, *noiseSrv = nullptr;
  {
    std::vector<uint16_t> data(512 * 512 * 4);
    for (size_t i = 0; i < data.size(); i++) data[i] = uint16_t(Rnd(1u, uint32_t(i)) * 65535.f);
    D3D11_TEXTURE2D_DESC td = { }; td.Width = 512; td.Height = 512; td.MipLevels = 1; td.ArraySize = 1;
    td.Format = DXGI_FORMAT_R16G16B16A16_UNORM; td.SampleDesc.Count = 1; td.Usage = D3D11_USAGE_IMMUTABLE;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA init = { data.data(), 512 * 8, 0 };
    ID3D11Texture2D* t = nullptr; Check(dev->CreateTexture2D(&td, &init, &t), "aux");
    Check(dev->CreateShaderResourceView(t, nullptr, &auxSrv), "aux srv");

    std::vector<uint16_t> l(4096 * 4);
    for (size_t i = 0; i < l.size(); i++) l[i] = uint16_t(Rnd(2u, uint32_t(i)) * 65535.f);
    D3D11_TEXTURE1D_DESC t1 = { }; t1.Width = 4096; t1.MipLevels = 1; t1.ArraySize = 1;
    t1.Format = DXGI_FORMAT_R16G16B16A16_UNORM; t1.Usage = D3D11_USAGE_IMMUTABLE; t1.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA i1 = { l.data(), 4096 * 8, 0 };
    ID3D11Texture1D* tl = nullptr; Check(dev->CreateTexture1D(&t1, &i1, &tl), "lut");
    Check(dev->CreateShaderResourceView(tl, nullptr, &lutSrv), "lut srv");

    std::vector<uint8_t> n(32 * 32 * 32 * 4);
    for (size_t i = 0; i < n.size(); i++) n[i] = uint8_t(Rnd(3u, uint32_t(i)) * 255.f);
    D3D11_TEXTURE3D_DESC t3 = { }; t3.Width = 32; t3.Height = 32; t3.Depth = 32; t3.MipLevels = 1;
    t3.Format = DXGI_FORMAT_R8G8B8A8_UNORM; t3.Usage = D3D11_USAGE_IMMUTABLE; t3.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA i3 = { n.data(), 32 * 4, 32 * 32 * 4 };
    ID3D11Texture3D* tn = nullptr; Check(dev->CreateTexture3D(&t3, &i3, &tn), "noise");
    Check(dev->CreateShaderResourceView(tn, nullptr, &noiseSrv), "noise srv");
  }

  // the two froxel volumes
  ID3D11Texture3D* vol[2] = { }; ID3D11ShaderResourceView* volSrv[2] = { }; ID3D11UnorderedAccessView* volUav[2] = { };
  for (int i = 0; i < 2; i++) {
    D3D11_TEXTURE3D_DESC td = { }; td.Width = 320; td.Height = 192; td.Depth = 90; td.MipLevels = 1;
    td.Format = DXGI_FORMAT_R16_FLOAT; td.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    Check(dev->CreateTexture3D(&td, nullptr, &vol[i]), "vol");
    Check(dev->CreateShaderResourceView(vol[i], nullptr, &volSrv[i]), "vol srv");
    Check(dev->CreateUnorderedAccessView(vol[i], nullptr, &volUav[i]), "vol uav");
  }
  ID3D11Buffer* bigBuf = nullptr; std::vector<uint8_t> bigData;
  if (uploadMB) {
    D3D11_BUFFER_DESC ud = { uploadMB << 20, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0, D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS, 0 };
    Check(dev->CreateBuffer(&ud, nullptr, &bigBuf), "upload buffer");
    bigData.resize(size_t(uploadMB) << 20, 0x5a);
  }

  ID3D11Texture3D* volStage = nullptr;
  if (checkEvery) {
    D3D11_TEXTURE3D_DESC td; vol[0]->GetDesc(&td); td.BindFlags = 0; td.Usage = D3D11_USAGE_STAGING;
    td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    Check(dev->CreateTexture3D(&td, nullptr, &volStage), "vol stage");
  }

  // the raster target
  ID3D11Texture2D* sceneTex = nullptr; ID3D11RenderTargetView* sceneRtv = nullptr; ID3D11ShaderResourceView* sceneSrv = nullptr;
  {
    D3D11_TEXTURE2D_DESC td = { }; td.Width = W; td.Height = H; td.MipLevels = 1; td.ArraySize = 1;
    td.Format = DXGI_FORMAT_R16G16B16A16_FLOAT; td.SampleDesc.Count = 1;
    td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    Check(dev->CreateTexture2D(&td, nullptr, &sceneTex), "scene");
    dev->CreateRenderTargetView(sceneTex, nullptr, &sceneRtv);
    dev->CreateShaderResourceView(sceneTex, nullptr, &sceneSrv);
  }

  ID3D11Texture2D* sceneDepth = nullptr; ID3D11DepthStencilView* sceneDsv = nullptr;
  {
    D3D11_TEXTURE2D_DESC td = { }; td.Width = W; td.Height = H; td.MipLevels = 1; td.ArraySize = 1;
    td.Format = DXGI_FORMAT_D32_FLOAT; td.SampleDesc.Count = 1; td.BindFlags = D3D11_BIND_DEPTH_STENCIL;
    Check(dev->CreateTexture2D(&td, nullptr, &sceneDepth), "scene depth");
    Check(dev->CreateDepthStencilView(sceneDepth, nullptr, &sceneDsv), "scene dsv");
  }

  ID3D11ComputeShader *csGen, *csChain;
  ID3D11VertexShader *vsTri, *vsCascade;
  ID3D11PixelShader *psRaster, *psConsumer;
  dev->CreateComputeShader(bGen->GetBufferPointer(), bGen->GetBufferSize(), nullptr, &csGen);
  dev->CreateComputeShader(bCh->GetBufferPointer(), bCh->GetBufferSize(), nullptr, &csChain);
  dev->CreateVertexShader(bVs->GetBufferPointer(), bVs->GetBufferSize(), nullptr, &vsTri);
  dev->CreateVertexShader(bVsC->GetBufferPointer(), bVsC->GetBufferSize(), nullptr, &vsCascade);
  dev->CreatePixelShader(bPsR->GetBufferPointer(), bPsR->GetBufferSize(), nullptr, &psRaster);
  dev->CreatePixelShader(bPsC->GetBufferPointer(), bPsC->GetBufferSize(), nullptr, &psConsumer);
  ID3D11PixelShader* psCheck = nullptr;
  dev->CreatePixelShader(bPsK->GetBufferPointer(), bPsK->GetBufferSize(), nullptr, &psCheck);
  ID3D11Texture2D *checkTex = nullptr, *checkStage = nullptr; ID3D11RenderTargetView* checkRtv = nullptr;
  if (checkEvery) {
    D3D11_TEXTURE2D_DESC td = { }; td.Width = 320; td.Height = 192; td.MipLevels = 1; td.ArraySize = 1;
    td.Format = DXGI_FORMAT_R32_FLOAT; td.SampleDesc.Count = 1; td.BindFlags = D3D11_BIND_RENDER_TARGET;
    Check(dev->CreateTexture2D(&td, nullptr, &checkTex), "check rt");
    dev->CreateRenderTargetView(checkTex, nullptr, &checkRtv);
    td.BindFlags = 0; td.Usage = D3D11_USAGE_STAGING; td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    Check(dev->CreateTexture2D(&td, nullptr, &checkStage), "check stage");
  }

  D3D11_BUFFER_DESC bd = { }; bd.ByteWidth = sizeof(Cb); bd.Usage = D3D11_USAGE_DYNAMIC;
  bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER; bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
  ID3D11Buffer *genCb = nullptr, *stepCb = nullptr, *drawCb = nullptr;
  dev->CreateBuffer(&bd, nullptr, &genCb); dev->CreateBuffer(&bd, nullptr, &stepCb); dev->CreateBuffer(&bd, nullptr, &drawCb);
  D3D11_BUFFER_DESC pd = { 16, D3D11_USAGE_DEFAULT, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0 };
  ID3D11Buffer* paramCb = nullptr; dev->CreateBuffer(&pd, nullptr, &paramCb);

  D3D11_SAMPLER_DESC smd = { }; smd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
  smd.AddressU = smd.AddressV = smd.AddressW = D3D11_TEXTURE_ADDRESS_WRAP; smd.MaxLOD = D3D11_FLOAT32_MAX;
  ID3D11SamplerState* smp = nullptr; dev->CreateSamplerState(&smd, &smp);

  D3D11_DEPTH_STENCIL_DESC dsd = { }; dsd.DepthEnable = TRUE; dsd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
  dsd.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
  ID3D11DepthStencilState* dss = nullptr; dev->CreateDepthStencilState(&dsd, &dss);
  D3D11_DEPTH_STENCIL_DESC dsdOff = { };
  ID3D11DepthStencilState* dssOff = nullptr; dev->CreateDepthStencilState(&dsdOff, &dssOff);
  D3D11_DEPTH_STENCIL_DESC dsdAlways = { }; dsdAlways.DepthEnable = TRUE; dsdAlways.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
  dsdAlways.DepthFunc = D3D11_COMPARISON_ALWAYS;
  ID3D11DepthStencilState* dssAlways = nullptr; dev->CreateDepthStencilState(&dsdAlways, &dssAlways);
  D3D11_RASTERIZER_DESC rsd = { }; rsd.FillMode = D3D11_FILL_SOLID; rsd.CullMode = D3D11_CULL_NONE; rsd.DepthClipEnable = TRUE;
  ID3D11RasterizerState* rs = nullptr; dev->CreateRasterizerState(&rsd, &rs);

  auto setCb = [&](ID3D11Buffer* b, const Cb& c) {
    D3D11_MAPPED_SUBRESOURCE m; Check(ctx->Map(b, 0, D3D11_MAP_WRITE_DISCARD, 0, &m), "map cb");
    memcpy(m.pData, &c, sizeof(c)); ctx->Unmap(b, 0);
  };

  const int Ring = 6;
  ID3D11Query *qDisjoint[Ring], *qStart[Ring], *qFrame[Ring];
  for (int i = 0; i < Ring; i++) {
    D3D11_QUERY_DESC qd = { D3D11_QUERY_TIMESTAMP_DISJOINT, 0 }; dev->CreateQuery(&qd, &qDisjoint[i]);
    qd.Query = D3D11_QUERY_TIMESTAMP; dev->CreateQuery(&qd, &qStart[i]); dev->CreateQuery(&qd, &qFrame[i]);
  }

  std::vector<double> frameGpuMs, periodMs, cpuMs;
  UINT64 lastFrameTs = 0; bool haveLast = false;
  long frames = 0, checks = 0, queryFails = 0;
  ID3D11ShaderResourceView* nullSrv[4] = { };
  ID3D11UnorderedAccessView* nullUav[2] = { };

  auto t0 = std::chrono::steady_clock::now();
  auto tPrev = t0;
  MSG msg;

  while (!g_quit) {
    while (PeekMessageA(&msg, nullptr, 0, 0, PM_REMOVE)) { TranslateMessage(&msg); DispatchMessageA(&msg); }
    double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (el >= seconds || (maxFrames && frames >= maxFrames)) break;

    uint32_t f = uint32_t(frames);
    int slot = frames % Ring;
    ctx->Begin(qDisjoint[slot]);
    ctx->End(qStart[slot]);

    // top of frame: the generate constants (their mirror copy is pending when the window opens)
    setCb(genCb, { { Rnd(f, 10) * 0.01f, Rnd(f, 11) * 0.01f, Rnd(f, 12), Rnd(f, 13) },
                   { 0.5f + Rnd(f, 14), Rnd(f, 15), Rnd(f, 16), Rnd(f, 17) } });

    // cascades: four quadrants of the atlas
    ctx->CSSetShaderResources(0, 4, nullSrv);
    ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ctx->IASetInputLayout(nullptr);
    ctx->RSSetState(rs);
    ctx->OMSetDepthStencilState(dss, 0);
    D3D11_VIEWPORT avp = { 0, 0, 4096.f, 4096.f, 0, 1 };
    ctx->RSSetViewports(1, &avp);
    ctx->ClearDepthStencilView(atlasDsv, D3D11_CLEAR_DEPTH, 1.0f, 0);
    ctx->OMSetRenderTargets(0, nullptr, atlasDsv);
    ctx->VSSetShader(vsCascade, nullptr, 0);
    ctx->PSSetShader(nullptr, nullptr, 0);
    ctx->VSSetConstantBuffers(2, 1, &drawCb);
    for (int q = 0; q < 4; q++) {
      setCb(drawCb, { { (q & 1) ? 0.f : -1.f, (q & 2) ? 0.f : -1.f, 0.2f + 0.6f * Rnd(f, 20 + q), 0.3f * Rnd(f, 30 + q) }, { 0, 0, 0, 0 } });
      ctx->Draw(3 * 64, 0);
    }
    ctx->OMSetRenderTargets(0, nullptr, nullptr);

    // generate
    ID3D11ShaderResourceView* genSrv[4] = { atlasSrv, auxSrv, lutSrv, noiseSrv };
    ctx->CSSetShader(csGen, nullptr, 0);
    ctx->CSSetShaderResources(0, 4, genSrv);
    ctx->CSSetUnorderedAccessViews(0, 2, volUav, nullptr);
    ctx->CSSetConstantBuffers(0, 1, &genCb);
    ctx->CSSetSamplers(0, 1, &smp);
    ctx->Dispatch(10, 6, 90);
    ctx->CSSetUnorderedAccessViews(0, 2, nullUav, nullptr);
    ctx->CSSetShaderResources(0, 4, nullSrv);

    // chain: ping-pong, one map before each step
    float param[4] = { 0.75f + 0.25f * Rnd(f, 50), 0, 0, 0 };
    ctx->UpdateSubresource(paramCb, 0, nullptr, param, 0, 0);
    ctx->CSSetShader(csChain, nullptr, 0);
    ctx->CSSetConstantBuffers(1, 1, &stepCb);
    ctx->CSSetConstantBuffers(3, 1, &paramCb);
    for (UINT k = 0; k < steps; k++) {
      int s = int(k & 1u), d = s ^ 1;
      Cb c = { }; uint32_t kk = k; memcpy(&c.a[0], &kk, 4); c.b[0] = 0.25f + 0.5f * Rnd(f, 100 + k);
      setCb(stepCb, c);
      ctx->CSSetUnorderedAccessViews(0, 1, &nullUav[0], nullptr);
      ctx->CSSetShaderResources(0, 1, &volSrv[s]);
      ctx->CSSetUnorderedAccessViews(0, 1, &volUav[d], nullptr);
      ctx->Dispatch(10, 6, 1);
    }
    ctx->CSSetUnorderedAccessViews(0, 1, &nullUav[0], nullptr);
    ctx->CSSetShaderResources(0, 1, nullSrv);
    int finalVol = int(steps & 1u); // step k writes vol[(k&1)^1]; the last is steps-1

    // raster that touches neither volume (the prepass/masks/lit stretch), after a depth
    // clear that is still inside the window
    ctx->ClearDepthStencilView(sceneDsv, D3D11_CLEAR_DEPTH, 1.0f, 0);
    D3D11_VIEWPORT svp = { 0, 0, float(W), float(H), 0, 1 };
    ctx->RSSetViewports(1, &svp);
    ctx->OMSetDepthStencilState(dssAlways, 0);
    ctx->OMSetRenderTargets(1, &sceneRtv, sceneDsv);
    ctx->VSSetShader(vsTri, nullptr, 0);
    ctx->PSSetShader(psRaster, nullptr, 0);
    ctx->PSSetConstantBuffers(2, 1, &drawCb);
    for (int r = 0; r < rasterDraws; r++) {
      setCb(drawCb, { { 0, 0, 0.5f, Rnd(f, 200 + r) }, { float(rasterIters), 0, 0, 0 } });
      ctx->Draw(3, 0);
    }

    // the consumer (the wait draw)
    setCb(drawCb, { { 0, 0, 0.5f, 0 }, { 0, 0, 0, 0 } });
    ctx->OMSetDepthStencilState(dssOff, 0);
    ctx->OMSetRenderTargets(1, &bbRtv, nullptr);
    ctx->PSSetShader(psConsumer, nullptr, 0);
    ID3D11ShaderResourceView* conSrv[3] = { sceneSrv, nullptr, volSrv[finalVol] };
    ctx->PSSetShaderResources(0, 3, conSrv);
    ctx->PSSetSamplers(0, 1, &smp);
    ctx->Draw(3, 0);

    bool checkFrame = checkEvery && (frames % checkEvery) == checkEvery - 1;
    if (checkFrame) {
      D3D11_VIEWPORT cvp = { 0, 0, 320.f, 192.f, 0, 1 };
      ctx->RSSetViewports(1, &cvp);
      ctx->OMSetRenderTargets(1, &checkRtv, nullptr);
      ctx->PSSetShader(psCheck, nullptr, 0);
      ctx->Draw(3, 0);
      ctx->RSSetViewports(1, &svp);
    }
    ctx->PSSetShaderResources(0, 3, nullSrv);

    if (bigBuf)
      ctx->UpdateSubresource(bigBuf, 0, nullptr, bigData.data(), 0, 0);

    ctx->End(qFrame[slot]);
    ctx->End(qDisjoint[slot]);

    if (checkFrame) {
      ctx->CopyResource(checkStage, checkTex);
      D3D11_MAPPED_SUBRESOURCE cm;
      Check(ctx->Map(checkStage, 0, D3D11_MAP_READ, 0, &cm), "map check");
      uint64_t ch = 1469598103934665603ull;
      for (UINT y = 0; y < 192; y++) {
        const uint8_t* row = (const uint8_t*)cm.pData + y * cm.RowPitch;
        for (UINT x = 0; x < 320 * 4; x++) { ch ^= row[x]; ch *= 1099511628211ull; }
      }
      ctx->Unmap(checkStage, 0);
      printf("cchk frame=%ld hash=%016llx\n", frames, (unsigned long long)ch);

      ctx->CopyResource(volStage, vol[finalVol]);
      D3D11_MAPPED_SUBRESOURCE m;
      Check(ctx->Map(volStage, 0, D3D11_MAP_READ, 0, &m), "map vol");
      uint64_t h = 1469598103934665603ull;
      for (UINT z = 0; z < 90; z++) for (UINT y = 0; y < 192; y++) {
        const uint8_t* row = (const uint8_t*)m.pData + z * m.DepthPitch + y * m.RowPitch;
        for (UINT x = 0; x < 320 * 2; x++) { h ^= row[x]; h *= 1099511628211ull; }
      }
      ctx->Unmap(volStage, 0);
      printf("chk frame=%ld hash=%016llx\n", frames, (unsigned long long)h);
      checks++;
    }

    sc->Present(0, 0);
    frames++;

    auto tNow = std::chrono::steady_clock::now();
    if (el > 1.0) cpuMs.push_back(std::chrono::duration<double, std::milli>(tNow - tPrev).count());
    tPrev = tNow;

    if (frames >= Ring) {
      int r2 = frames % Ring;
      D3D11_QUERY_DATA_TIMESTAMP_DISJOINT dj;
      auto tq = std::chrono::steady_clock::now();
      HRESULT hq;
      while ((hq = ctx->GetData(qDisjoint[r2], &dj, sizeof(dj), 0)) == S_FALSE
          && std::chrono::steady_clock::now() - tq < std::chrono::seconds(2)) { }
      UINT64 a = 0, e = 0;
      HRESULT ha = ctx->GetData(qStart[r2], &a, sizeof(a), 0);
      HRESULT he = ctx->GetData(qFrame[r2], &e, sizeof(e), 0);
      if (hq != S_OK || ha != S_OK || he != S_OK) {
        if (queryFails++ < 5) printf("query not ready: frame %ld (disjoint %lx start %lx end %lx)\n", frames, hq, ha, he);
      } else if (!dj.Disjoint && el > 1.0) {
        frameGpuMs.push_back(double(e - a) * 1000.0 / double(dj.Frequency));
        if (haveLast) periodMs.push_back(double(e - lastFrameTs) * 1000.0 / double(dj.Frequency));
      }
      lastFrameTs = e; haveLast = true;
    }
  }

  auto stats = [](std::vector<double> v, double& mean, double& med, double& p90) {
    if (v.empty()) { mean = med = p90 = 0; return; }
    double s = 0; for (double x : v) s += x; mean = s / v.size();
    std::sort(v.begin(), v.end()); med = v[v.size() / 2]; p90 = v[v.size() * 9 / 10];
  };
  double m1, d1, p1, m2, d2, p2, m3, d3, p3;
  stats(frameGpuMs, m1, d1, p1); stats(periodMs, m2, d2, p2); stats(cpuMs, m3, d3, p3);
  double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  printf("volasynctest steps=%u raster=%dx%d frames=%ld fps=%.1f checks=%ld\n", steps, rasterDraws, rasterIters, frames, frames / el, checks);
  printf("frame gpu ms (start->end ts): mean %.4f median %.4f p90 %.4f (n=%zu)\n", m1, d1, p1, frameGpuMs.size());
  printf("frame period gpu ms: mean %.4f median %.4f p90 %.4f (n=%zu)\n", m2, d2, p2, periodMs.size());
  printf("frame wall ms: mean %.4f median %.4f p90 %.4f (n=%zu)\n", m3, d3, p3, cpuMs.size());
  return 0;
}
