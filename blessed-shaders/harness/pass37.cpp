// blessed: shader-replace, pass 37 micro-benchmark: skyrim's soft-particle pass rebuilt with its own pixel shader, timed on native d3d11 or dxvk
//
// usage: pass37 PS.dxbc [flags...]
//   the pass: clear rtv 1 and 2, then 96 indexed draws of 20 camera-facing
//   quads each into rgba16f + rg8 + rgba16f with alpha blending (rgb write
//   mask), read-only d24s8 depth test less-equal, 4 dynamic cbuffers mapped
//   per draw, the game's pixel shader 556a3b73 (soft particles: depth copy
//   load, two samples, two discards). timed with d3d11 timestamps, median of
//   40 frames.
//   flags (each removes one ingredient, to find what dxvk pays for):
//     noclear   no rtv clears          noblend   blending off
//     nodepth   no depth buffer        mask15    write mask rgba
//     onert     rtv 0 only             nodiscard a pixel shader without discard
//     static    cbuffers default-usage, updated with UpdateSubresource
//     rwdepth   depth writes on
// the d3d11.dll found first decides the backend: put the fork's d3d11.dll and
// dxgi.dll next to the exe for dxvk, leave them out for native.
#define NOMINMAX
#include <d3d11.h>
#include <d3dcompiler.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

  std::vector<uint8_t> ReadFile(const char* path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  }

  ID3DBlob* Compile(const char* src, const char* profile) {
    ID3DBlob* blob = nullptr;
    ID3DBlob* err = nullptr;
    if (FAILED(D3DCompile(src, std::strlen(src), nullptr, nullptr, nullptr, "main", profile, 0, 0, &blob, &err))) {
      std::printf("compile failed: %s\n", err ? (const char*) err->GetBufferPointer() : "?");
      std::exit(1);
    }
    return blob;
  }

  // 20 quads per draw from SV_VertexID alone; each draw offsets its quads
  const char* g_vs = R"(
cbuffer draw : register(b12) { float4 g_draw; };  // x: draw index, y: quad half size (ndc)
cbuffer palette : register(b9) { float4 g_pal[240]; };  // like the game's 3840-byte vs cbuffers
cbuffer palette2 : register(b10) { float4 g_pal2[240]; };
cbuffer palette3 : register(b5) { float4 g_pal3[240]; };
cbuffer palette4 : register(b6) { float4 g_pal4[240]; };
struct VsOut {
  float4 pos : SV_POSITION;
  float4 uv  : TEXCOORD0;
  float4 c0  : COLOR0;
  float4 c1  : COLOR1;
  float  f   : TEXCOORD5;
};
uint hash(uint x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; }
VsOut main(uint vid : SV_VertexID) {
  uint quad = vid / 6u + uint(g_draw.x) * 20u;
  uint corner = vid % 6u;
  float2 c = float2((0x32u >> corner) & 1u, (0x2cu >> corner) & 1u);
  uint h = hash(quad);
  float2 center = float2(h & 0xffffu, h >> 16) / 65535.0f * 2.0f - 1.0f;
  float size = g_draw.y * (0.5f + float(hash(h) & 0xffu) / 255.0f);
  VsOut o;
  float4 p0 = g_pal[quad % 240u];
  float4 p1 = g_pal[(quad * 7u + corner) % 240u] + g_pal2[(quad * 3u + corner) % 240u] + g_pal3[(quad * 5u) % 240u] + g_pal4[(quad + corner) % 240u];
  o.pos = float4(center + (c * 2.0f - 1.0f) * size + (p0.xy + p1.zw) * 1e-6f, 0.5f, 1.0f);
  o.uv = float4(c, 0.5f, 0.2f);
  o.c0 = float4(1.0f, 0.9f, 0.8f, 0.6f);
  o.c1 = float4(0.2f, 0.3f, 0.4f, 0.3f);
  o.f = 0.9f;
  return o;
}
)";

  // the same work as 556a3b73 minus the two discards
  const char* g_psNoDiscard = R"(
cbuffer cb0 : register(b0) { float4 c0; };
cbuffer cb1 : register(b1) { float4 c1[3]; };
cbuffer cb2 : register(b2) { float4 c2[9]; };
Texture2D t0 : register(t0); Texture2D t3 : register(t3); Texture2D t4 : register(t4);
SamplerState s0 : register(s0); SamplerState s4 : register(s4);
struct PsIn { float4 pos : SV_POSITION; float4 uv : TEXCOORD0; float4 c0 : COLOR0; float4 c1 : COLOR1; float f : TEXCOORD5; };
struct PsOut { float4 o0 : SV_Target0; float4 o1 : SV_Target1; float4 o2 : SV_Target2; };
PsOut main(PsIn i) {
  float d = t3.Load(int3(i.pos.xy, 0)).x;
  float soft = saturate(c1[2].y / ((1 - d) * c0.z + c0.y) - i.uv.w);
  float4 t = t0.Sample(s0, i.uv.xy);
  float4 col = i.c0 * c1[0] * float4(t.yzw, i.uv.z);
  float a = col.w * soft;
  float n = t4.Sample(s4, float2(t.x, a * c2[8].w)).w;
  col.xyz = lerp(col.xyz, c2[8].xyz * col.xyz, c1[2].x);
  col.xyz = lerp(col.xyz, i.c1.xyz, i.c1.w);
  float4 r = float4(i.f * col.xyz, a + n * 0.0001f);
  PsOut o; o.o0 = r; o.o1 = float4(1, 0, 0, r.w); o.o2 = r; return o;
}
)";

  // ablation shaders, picked with ps=<name>
  const char* g_psVariants[][2] = {
    { "flat", R"(
struct PsOut { float4 o0 : SV_Target0; float4 o1 : SV_Target1; float4 o2 : SV_Target2; };
PsOut main(float4 pos : SV_POSITION) { PsOut o; o.o0 = float4(0.5, 0.4, 0.3, 0.2); o.o1 = float4(1, 0, 0, 0.2); o.o2 = o.o0; return o; }
)" },
    { "load", R"(
Texture2D t3 : register(t3);
struct PsOut { float4 o0 : SV_Target0; float4 o1 : SV_Target1; float4 o2 : SV_Target2; };
PsOut main(float4 pos : SV_POSITION) { float d = t3.Load(int3(pos.xy, 0)).x; PsOut o; o.o0 = float4(d, 0.4, 0.3, 0.2); o.o1 = float4(1, 0, 0, 0.2); o.o2 = o.o0; return o; }
)" },
    { "sample", R"(
Texture2D t0 : register(t0); SamplerState s0 : register(s0);
struct PsIn { float4 pos : SV_POSITION; float4 uv : TEXCOORD0; };
struct PsOut { float4 o0 : SV_Target0; float4 o1 : SV_Target1; float4 o2 : SV_Target2; };
PsOut main(PsIn i) { float4 t = t0.Sample(s0, i.uv.xy); PsOut o; o.o0 = t * 0.5; o.o1 = float4(1, 0, 0, t.w); o.o2 = o.o0; return o; }
)" },
    { "discard", R"(
Texture2D t0 : register(t0); SamplerState s0 : register(s0);
struct PsIn { float4 pos : SV_POSITION; float4 uv : TEXCOORD0; };
struct PsOut { float4 o0 : SV_Target0; float4 o1 : SV_Target1; float4 o2 : SV_Target2; };
PsOut main(PsIn i) { float4 t = t0.Sample(s0, i.uv.xy); if (t.w < 0.001) discard; PsOut o; o.o0 = t * 0.5; o.o1 = float4(1, 0, 0, t.w); o.o2 = o.o0; return o; }
)" },
  };

  ID3D11ShaderResourceView* MakeTex(ID3D11Device* dev, uint32_t w, uint32_t h, uint32_t seed) {
    std::vector<uint32_t> data(size_t(w) * h);
    for (size_t i = 0; i < data.size(); i++) {
      uint32_t x = uint32_t(i) * 2654435761u + seed;
      x ^= x >> 13;
      data[i] = x | 0x80000000u;
    }
    D3D11_TEXTURE2D_DESC desc = { w, h, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM, { 1, 0 },
      D3D11_USAGE_IMMUTABLE, D3D11_BIND_SHADER_RESOURCE, 0, 0 };
    D3D11_SUBRESOURCE_DATA init = { data.data(), w * 4u, 0 };
    ID3D11Texture2D* tex = nullptr;
    ID3D11ShaderResourceView* srv = nullptr;
    dev->CreateTexture2D(&desc, &init, &tex);
    dev->CreateShaderResourceView(tex, nullptr, &srv);
    return srv;
  }

  bool Has(int argc, char** argv, const char* flag) {
    for (int i = 2; i < argc; i++)
      if (!std::strcmp(argv[i], flag))
        return true;
    return false;
  }

}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: pass37 PS.dxbc [noclear|noblend|nodepth|mask15|onert|nodiscard|static|rwdepth]...\n");
    return 1;
  }

  bool noClear = Has(argc, argv, "noclear"), noBlend = Has(argc, argv, "noblend");
  bool noDepth = Has(argc, argv, "nodepth"), mask15 = Has(argc, argv, "mask15");
  bool oneRt = Has(argc, argv, "onert"), noDiscard = Has(argc, argv, "nodiscard");
  bool staticCb = Has(argc, argv, "static"), rwDepth = Has(argc, argv, "rwdepth");
  float quadSize = 0.02f;
  for (int i = 2; i < argc; i++)
    if (!std::strncmp(argv[i], "size=", 5))
      quadSize = float(std::atof(argv[i] + 5));

  HMODULE mod = LoadLibraryA("d3d11.dll");
  char modPath[MAX_PATH] = { };
  GetModuleFileNameA(mod, modPath, MAX_PATH);
  auto create = reinterpret_cast<PFN_D3D11_CREATE_DEVICE>(GetProcAddress(mod, "D3D11CreateDevice"));
  ID3D11Device* dev = nullptr;
  ID3D11DeviceContext* ctx = nullptr;
  D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_0;
  if (!create || FAILED(create(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, &fl, 1, D3D11_SDK_VERSION, &dev, nullptr, &ctx))) {
    std::printf("no device\n");
    return 1;
  }

  const UINT W = 1920, H = 1080;

  // targets
  DXGI_FORMAT fmts[3] = { DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R8G8_UNORM, DXGI_FORMAT_R16G16B16A16_FLOAT };
  UINT rtCount = 3;
  for (int i = 2; i < argc; i++) {
    if (!std::strncmp(argv[i], "rt0=", 4)) fmts[0] = DXGI_FORMAT(std::atoi(argv[i] + 4));
    if (!std::strncmp(argv[i], "rt1=", 4)) fmts[1] = DXGI_FORMAT(std::atoi(argv[i] + 4));
    if (!std::strncmp(argv[i], "rt2=", 4)) fmts[2] = DXGI_FORMAT(std::atoi(argv[i] + 4));
    if (!std::strncmp(argv[i], "rts=", 4)) rtCount = UINT(std::atoi(argv[i] + 4));
  }
  ID3D11RenderTargetView* rtv[3] = { };
  for (int i = 0; i < 3; i++) {
    D3D11_TEXTURE2D_DESC desc = { W, H, 1, 1, fmts[i], { 1, 0 }, D3D11_USAGE_DEFAULT,
      D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE, 0, 0 };
    ID3D11Texture2D* tex = nullptr;
    dev->CreateTexture2D(&desc, nullptr, &tex);
    dev->CreateRenderTargetView(tex, nullptr, &rtv[i]);
  }

  // the depth buffer and the depth copy the shader loads (both r24g8 typeless)
  ID3D11DepthStencilView* dsv = nullptr;
  ID3D11ShaderResourceView* depthCopy = nullptr;
  for (int i = 0; i < 2; i++) {
    D3D11_TEXTURE2D_DESC desc = { W, H, 1, 1, DXGI_FORMAT_R24G8_TYPELESS, { 1, 0 }, D3D11_USAGE_DEFAULT,
      D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE, 0, 0 };
    ID3D11Texture2D* tex = nullptr;
    dev->CreateTexture2D(&desc, nullptr, &tex);
    D3D11_DEPTH_STENCIL_VIEW_DESC dd = { DXGI_FORMAT_D24_UNORM_S8_UINT, D3D11_DSV_DIMENSION_TEXTURE2D, 0, { } };
    ID3D11DepthStencilView* view = nullptr;
    dev->CreateDepthStencilView(tex, &dd, &view);
    ctx->ClearDepthStencilView(view, D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 0.9f, 0);
    if (i == 0) {
      dsv = view;
    } else {
      D3D11_SHADER_RESOURCE_VIEW_DESC sd = { };
      sd.Format = DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
      sd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
      sd.Texture2D.MipLevels = 1;
      dev->CreateShaderResourceView(tex, &sd, &depthCopy);
    }
  }

  // shaders
  ID3DBlob* vsBlob = Compile(g_vs, "vs_5_0");
  ID3D11VertexShader* vs = nullptr;
  dev->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), nullptr, &vs);
  ID3D11PixelShader* ps = nullptr;
  const char* variant = nullptr;
  for (int i = 2; i < argc; i++)
    if (!std::strncmp(argv[i], "ps=", 3))
      for (auto& v : g_psVariants)
        if (!std::strcmp(v[0], argv[i] + 3))
          variant = v[1];
  if (variant) {
    ID3DBlob* psBlob = Compile(variant, "ps_5_0");
    dev->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(), nullptr, &ps);
  } else if (noDiscard) {
    ID3DBlob* psBlob = Compile(g_psNoDiscard, "ps_5_0");
    dev->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(), nullptr, &ps);
  } else {
    std::vector<uint8_t> code = ReadFile(argv[1]);
    if (FAILED(dev->CreatePixelShader(code.data(), code.size(), nullptr, &ps))) {
      std::printf("CreatePixelShader failed\n");
      return 1;
    }
  }

  // index buffer: 20 quads, 6 indices each, plain 0..119
  std::vector<uint16_t> indices(120);
  for (uint16_t i = 0; i < 120; i++)
    indices[i] = i;
  D3D11_BUFFER_DESC ibDesc = { 240, D3D11_USAGE_IMMUTABLE, D3D11_BIND_INDEX_BUFFER, 0, 0, 0 };
  D3D11_SUBRESOURCE_DATA ibInit = { indices.data(), 0, 0 };
  ID3D11Buffer* ib = nullptr;
  dev->CreateBuffer(&ibDesc, &ibInit, &ib);

  // cbuffers: slots 0, 1, 2, 11 like the game's pixel shader, vs 12 = draw
  // index, vs 9 = a 3840-byte palette like the game's vs cb9/cb10
  UINT cbSizes[9] = { 16, 48, 144, 16, 16, 3840, 3840, 3840, 3840 };
  UINT cbSlots[9] = { 0, 1, 2, 11, 12, 9, 10, 5, 6 };
  ID3D11Buffer* cbs[9] = { };
  for (int i = 0; i < 9; i++) {
    D3D11_BUFFER_DESC desc = { cbSizes[i], staticCb ? D3D11_USAGE_DEFAULT : D3D11_USAGE_DYNAMIC,
      D3D11_BIND_CONSTANT_BUFFER, staticCb ? 0u : UINT(D3D11_CPU_ACCESS_WRITE), 0, 0 };
    dev->CreateBuffer(&desc, nullptr, &cbs[i]);
  }

  static float cbData[9][960] = { };
  cbData[0][1] = 1.0f; cbData[0][2] = 0.5f;                       // cb0[0].yz: depth linearise
  for (int k = 0; k < 4; k++) cbData[1][k] = 1.0f;                // cb1[0]: tint
  cbData[1][8] = 0.5f; cbData[1][9] = 2.0f;                       // cb1[2].xy
  for (int k = 0; k < 4; k++) cbData[2][32 + k] = 0.7f;           // cb2[8]
  cbData[3][0] = 0.001f;                                          // cb11[0].x: alpha threshold

  ID3D11SamplerState* samp = nullptr;
  D3D11_SAMPLER_DESC sd = { };
  sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
  sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
  sd.MaxLOD = D3D11_FLOAT32_MAX;
  dev->CreateSamplerState(&sd, &samp);

  ID3D11ShaderResourceView* t0 = MakeTex(dev, 512, 512, 1u);
  ID3D11ShaderResourceView* t4 = MakeTex(dev, 256, 512, 2u);

  D3D11_BLEND_DESC bd = { };
  bd.IndependentBlendEnable = FALSE;
  bd.RenderTarget[0].BlendEnable = noBlend ? FALSE : TRUE;
  bd.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
  bd.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
  bd.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
  bd.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
  bd.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
  bd.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
  bd.RenderTarget[0].RenderTargetWriteMask = mask15 ? 15 : 7;
  ID3D11BlendState* blend = nullptr;
  dev->CreateBlendState(&bd, &blend);

  D3D11_DEPTH_STENCIL_DESC dsd = { };
  dsd.DepthEnable = TRUE;
  dsd.DepthWriteMask = rwDepth ? D3D11_DEPTH_WRITE_MASK_ALL : D3D11_DEPTH_WRITE_MASK_ZERO;
  dsd.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
  D3D11_RASTERIZER_DESC rd = { };
  rd.FillMode = D3D11_FILL_SOLID;
  rd.CullMode = D3D11_CULL_NONE;
  rd.DepthClipEnable = TRUE;
  ID3D11RasterizerState* raster = nullptr;
  dev->CreateRasterizerState(&rd, &raster);
  ID3D11DepthStencilState* depth = nullptr;
  dev->CreateDepthStencilState(&dsd, &depth);

  D3D11_QUERY_DESC qd = { D3D11_QUERY_TIMESTAMP_DISJOINT, 0 };
  ID3D11Query* disjoint = nullptr;
  ID3D11Query* ts[2] = { };
  ID3D11Query* stats = nullptr;
  dev->CreateQuery(&qd, &disjoint);
  qd.Query = D3D11_QUERY_TIMESTAMP;
  dev->CreateQuery(&qd, &ts[0]);
  dev->CreateQuery(&qd, &ts[1]);
  qd.Query = D3D11_QUERY_PIPELINE_STATISTICS;
  dev->CreateQuery(&qd, &stats);
  D3D11_QUERY_DATA_PIPELINE_STATISTICS lastStats = { };

  D3D11_VIEWPORT vp = { 0, 0, float(W), float(H), 0, 1 };
  std::vector<double> times;
  const FLOAT zero[4] = { 0, 0, 0, 0 };

  for (int frame = 0; frame < 60; frame++) {
    ctx->Begin(disjoint);
    ctx->Begin(stats);
    ctx->End(ts[0]);

    if (!noClear) {
      ctx->ClearRenderTargetView(rtv[1], zero);
      ctx->ClearRenderTargetView(rtv[2], zero);
    }

    ctx->OMSetRenderTargets(oneRt ? 1 : rtCount, rtv, noDepth ? nullptr : dsv);
    ctx->OMSetBlendState(blend, nullptr, 0xffffffffu);
    ctx->OMSetDepthStencilState(depth, 0);
    ctx->RSSetViewports(1, &vp);
    ctx->RSSetState(raster);
    ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ctx->IASetIndexBuffer(ib, DXGI_FORMAT_R16_UINT, 0);
    ctx->IASetInputLayout(nullptr);
    ctx->VSSetShader(vs, nullptr, 0);
    ctx->PSSetShader(ps, nullptr, 0);
    ID3D11ShaderResourceView* srvs[5] = { t0, nullptr, nullptr, depthCopy, t4 };
    ctx->PSSetShaderResources(0, 5, srvs);
    ID3D11SamplerState* samps[5] = { samp, nullptr, nullptr, nullptr, samp };
    ctx->PSSetSamplers(0, 5, samps);

    for (int draw = 0; draw < 96; draw++) {
      cbData[4][0] = float(draw);
      cbData[4][1] = quadSize;
      for (int i = 0; i < 9; i++) {
        if (staticCb) {
          ctx->UpdateSubresource(cbs[i], 0, nullptr, cbData[i], 0, 0);
        } else {
          D3D11_MAPPED_SUBRESOURCE m = { };
          ctx->Map(cbs[i], 0, D3D11_MAP_WRITE_DISCARD, 0, &m);
          std::memcpy(m.pData, cbData[i], cbSizes[i]);
          ctx->Unmap(cbs[i], 0);
        }
        if (i < 4)
          ctx->PSSetConstantBuffers(cbSlots[i], 1, &cbs[i]);
        else
          ctx->VSSetConstantBuffers(cbSlots[i], 1, &cbs[i]);
      }
      ctx->DrawIndexed(120, 0, draw * 120);
    }

    ctx->End(ts[1]);
    ctx->End(stats);
    ctx->End(disjoint);

    D3D11_QUERY_DATA_TIMESTAMP_DISJOINT dj = { };
    UINT64 a = 0, b = 0;
    while (ctx->GetData(disjoint, &dj, sizeof(dj), 0) != S_OK) { }
    while (ctx->GetData(ts[0], &a, sizeof(a), 0) != S_OK) { }
    while (ctx->GetData(ts[1], &b, sizeof(b), 0) != S_OK) { }
    while (ctx->GetData(stats, &lastStats, sizeof(lastStats), 0) != S_OK) { }

    if (frame >= 20 && !dj.Disjoint && dj.Frequency)
      times.push_back(double(b - a) * 1000.0 / double(dj.Frequency));
  }

  std::sort(times.begin(), times.end());
  std::string flags;
  for (int i = 2; i < argc; i++)
    flags += std::string(" ") + argv[i];
  char exePath[MAX_PATH] = { };
  GetModuleFileNameA(nullptr, exePath, MAX_PATH);
  std::string exeDir(exePath);
  exeDir = exeDir.substr(0, exeDir.find_last_of("\\/"));
  bool isDxvk = _strnicmp(modPath, exeDir.c_str(), exeDir.size()) == 0;
  std::printf("%-6s median %.4f ms  min %.4f  max %.4f  ps %llu prims %llu |%s\n",
    isDxvk ? "dxvk" : "native", times[times.size() / 2], times.front(), times.back(),
    (unsigned long long) lastStats.PSInvocations, (unsigned long long) lastStats.CPrimitives, flags.c_str());
  return 0;
}
