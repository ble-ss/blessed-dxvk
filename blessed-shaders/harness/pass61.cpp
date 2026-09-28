// blessed: shader-replace, pass 61 micro-benchmark: skyrim's vanilla taa resolve, timed only (no replacement shipped -- see blessed-notes/sao-taa-notes.md)
//
// usage: pass61 PS.dxbc
//   draws a fullscreen triangle with it into two rgba8 targets (no blend, no
//   depth -- matches the vanilla draw's own state), reading 6 synthetic srvs
//   shaped like the real pass (t0/t1/t5 rgba8 color-ish, t2 rg16f velocity,
//   t3 a depth srv view of an r24g8 texture, t4 rg8), timed with d3d11
//   timestamps, median of 50 frames.
// the d3d11.dll found first decides the backend: put the fork's d3d11.dll and
// dxgi.dll next to the exe for dxvk, leave them out for native.
#define NOMINMAX
#include <d3d11.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

namespace {

  std::vector<uint8_t> ReadFile(const char* path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  }

  ID3D11ShaderResourceView* MakeRgba8(ID3D11Device* dev, uint32_t w, uint32_t h, uint32_t seed) {
    std::vector<uint32_t> data(size_t(w) * h);
    for (size_t i = 0; i < data.size(); i++) {
      uint32_t x = uint32_t(i) * 2654435761u + seed;
      x ^= x >> 13; x *= 0x85ebca6bu; x ^= x >> 15;
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

  ID3D11ShaderResourceView* MakeRg8(ID3D11Device* dev, uint32_t w, uint32_t h, uint32_t seed) {
    std::vector<uint16_t> data(size_t(w) * h);
    for (size_t i = 0; i < data.size(); i++) {
      uint32_t x = uint32_t(i) * 2654435761u + seed;
      x ^= x >> 13; x *= 0x85ebca6bu; x ^= x >> 15;
      data[i] = uint16_t(x & 0xffffu);
    }
    D3D11_TEXTURE2D_DESC desc = { w, h, 1, 1, DXGI_FORMAT_R8G8_UNORM, { 1, 0 },
      D3D11_USAGE_IMMUTABLE, D3D11_BIND_SHADER_RESOURCE, 0, 0 };
    D3D11_SUBRESOURCE_DATA init = { data.data(), w * 2u, 0 };
    ID3D11Texture2D* tex = nullptr;
    ID3D11ShaderResourceView* srv = nullptr;
    dev->CreateTexture2D(&desc, &init, &tex);
    dev->CreateShaderResourceView(tex, nullptr, &srv);
    return srv;
  }

  ID3D11ShaderResourceView* MakeRg16f(ID3D11Device* dev, uint32_t w, uint32_t h) {
    std::vector<uint16_t> data(size_t(w) * h * 2u);
    for (uint32_t y = 0; y < h; y++) {
      for (uint32_t x = 0; x < w; x++) {
        float ang = 0.031f * x + 0.047f * y;
        float fx = 0.3f * std::cos(ang);
        float fy = 0.3f * std::sin(ang);
        auto toHalf = [](float v) -> uint16_t {
          uint32_t bits; std::memcpy(&bits, &v, 4);
          uint32_t sign = (bits >> 16) & 0x8000u;
          int32_t  exp  = int32_t((bits >> 23) & 0xffu) - 127 + 15;
          uint32_t mant = (bits >> 13) & 0x3ffu;
          if (exp <= 0) return uint16_t(sign);
          if (exp >= 31) return uint16_t(sign | 0x7c00u);
          return uint16_t(sign | (uint32_t(exp) << 10) | mant);
        };
        size_t idx = (size_t(y) * w + x) * 2u;
        data[idx + 0] = toHalf(fx);
        data[idx + 1] = toHalf(fy);
      }
    }
    D3D11_TEXTURE2D_DESC desc = { w, h, 1, 1, DXGI_FORMAT_R16G16_FLOAT, { 1, 0 },
      D3D11_USAGE_IMMUTABLE, D3D11_BIND_SHADER_RESOURCE, 0, 0 };
    D3D11_SUBRESOURCE_DATA init = { data.data(), w * 4u, 0 };
    ID3D11Texture2D* tex = nullptr;
    ID3D11ShaderResourceView* srv = nullptr;
    dev->CreateTexture2D(&desc, &init, &tex);
    dev->CreateShaderResourceView(tex, nullptr, &srv);
    return srv;
  }

  // a depth-typed srv (r24_unorm_x8_typeless view of an r24g8 texture), like
  // the real pass's t3
  ID3D11ShaderResourceView* MakeDepthSrv(ID3D11Device* dev, uint32_t w, uint32_t h) {
    D3D11_TEXTURE2D_DESC desc = { w, h, 1, 1, DXGI_FORMAT_R24G8_TYPELESS, { 1, 0 },
      D3D11_USAGE_DEFAULT, D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE, 0, 0 };
    ID3D11Texture2D* tex = nullptr;
    dev->CreateTexture2D(&desc, nullptr, &tex);
    // give it real depth values via a temporary dsv clear, not just zeros
    D3D11_DEPTH_STENCIL_VIEW_DESC dd = { DXGI_FORMAT_D24_UNORM_S8_UINT, D3D11_DSV_DIMENSION_TEXTURE2D, 0, { } };
    ID3D11DeviceContext* ctx = nullptr;
    dev->GetImmediateContext(&ctx);
    ID3D11DepthStencilView* dsv = nullptr;
    dev->CreateDepthStencilView(tex, &dd, &dsv);
    ctx->ClearDepthStencilView(dsv, D3D11_CLEAR_DEPTH, 0.7f, 0);
    dsv->Release();
    ctx->Release();
    D3D11_SHADER_RESOURCE_VIEW_DESC sd = { };
    sd.Format = DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
    sd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    sd.Texture2D.MipLevels = 1;
    ID3D11ShaderResourceView* srv = nullptr;
    dev->CreateShaderResourceView(tex, &sd, &srv);
    return srv;
  }

  const char* g_vs = R"(
struct VsOut { float4 pos : SV_POSITION; float2 uv : TEXCOORD0; };
VsOut main(uint vid : SV_VertexID) {
  VsOut o;
  o.uv  = float2((vid << 1) & 2, vid & 2);
  o.pos = float4(o.uv * float2(2, -2) + float2(-1, 1), 0, 1);
  return o;
}
)";

  ID3DBlob* CompileVs() {
    HMODULE d3dc = LoadLibraryA("d3dcompiler_47.dll");
    if (!d3dc) d3dc = LoadLibraryA("d3dcompiler_43.dll");
    auto compile = reinterpret_cast<HRESULT(WINAPI*)(LPCVOID, SIZE_T, LPCSTR, const D3D_SHADER_MACRO*,
      ID3DInclude*, LPCSTR, LPCSTR, UINT, UINT, ID3DBlob**, ID3DBlob**)>(GetProcAddress(d3dc, "D3DCompile"));
    ID3DBlob* blob = nullptr; ID3DBlob* err = nullptr;
    if (FAILED(compile(g_vs, std::strlen(g_vs), nullptr, nullptr, nullptr, "main", "vs_5_0", 0, 0, &blob, &err))) {
      std::printf("vs compile failed: %s\n", err ? (const char*) err->GetBufferPointer() : "?");
      std::exit(1);
    }
    return blob;
  }

}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: pass61 PS.dxbc\n");
    return 1;
  }

  HMODULE mod = LoadLibraryA("d3d11.dll");
  auto create = reinterpret_cast<PFN_D3D11_CREATE_DEVICE>(GetProcAddress(mod, "D3D11CreateDevice"));
  ID3D11Device* dev = nullptr;
  ID3D11DeviceContext* ctx = nullptr;
  D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_0;
  if (!create || FAILED(create(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, &fl, 1, D3D11_SDK_VERSION, &dev, nullptr, &ctx))) {
    std::printf("no device\n");
    return 1;
  }

  const UINT W = 1920, H = 1080;

  ID3D11Texture2D* rtTex[2] = { };
  ID3D11RenderTargetView* rtv[2] = { };
  for (int i = 0; i < 2; i++) {
    D3D11_TEXTURE2D_DESC desc = { W, H, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM, { 1, 0 },
      D3D11_USAGE_DEFAULT, D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE, 0, 0 };
    dev->CreateTexture2D(&desc, nullptr, &rtTex[i]);
    dev->CreateRenderTargetView(rtTex[i], nullptr, &rtv[i]);
  }

  ID3D11ShaderResourceView* srv[6] = {
    MakeRgba8(dev, W, H, 1u),
    MakeRgba8(dev, W, H, 2u),
    MakeRg16f(dev, W, H),
    MakeDepthSrv(dev, W, H),
    MakeRg8(dev, W, H, 3u),
    MakeRgba8(dev, W, H, 4u),
  };

  ID3DBlob* vsBlob = CompileVs();
  ID3D11VertexShader* vs = nullptr;
  dev->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), nullptr, &vs);

  std::vector<uint8_t> code = ReadFile(argv[1]);
  ID3D11PixelShader* ps = nullptr;
  if (code.empty() || FAILED(dev->CreatePixelShader(code.data(), code.size(), nullptr, &ps))) {
    std::printf("CreatePixelShader failed\n");
    return 1;
  }

  // cb12: 45 vec4s, cb2: 6 vec4s (dcl_constantbuffer CB2[6] in the vanilla
  // dxbc). non-degenerate: no zero radii/lengths anywhere division might hit.
  static float cb12Data[45 * 4] = { };
  cb12Data[43 * 4 + 0] = 1920.0f; cb12Data[43 * 4 + 1] = 1079.0f;
  cb12Data[43 * 4 + 2] = 960.0f;  cb12Data[43 * 4 + 3] = 539.0f;
  cb12Data[44 * 4 + 0] = 0.0f;    cb12Data[44 * 4 + 2] = 1919.0f; cb12Data[44 * 4 + 3] = 539.0f;
  D3D11_BUFFER_DESC bd12 = { sizeof(cb12Data), D3D11_USAGE_IMMUTABLE, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0 };
  D3D11_SUBRESOURCE_DATA bi12 = { cb12Data, 0, 0 };
  ID3D11Buffer* cb12 = nullptr;
  dev->CreateBuffer(&bd12, &bi12, &cb12);

  float cb2Data[24] = {
    1.0f / 1920, 1.0f / 1080, 0.0f, 0.0f,
    1.0f, 1.0f, 1.0f, 1.0f,
    0.2f, 0.3f, 0.5f, 0.83333f,
    1.0f / 1920, 1.0f / 1080, 1.0f, 1.0f,
    0.5f, 0.5f, 0.1f, 0.05f,
    0.8f, 0.9f, 0.5f, 1.0f,
  };
  D3D11_BUFFER_DESC bd2 = { sizeof(cb2Data), D3D11_USAGE_IMMUTABLE, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0 };
  D3D11_SUBRESOURCE_DATA bi2 = { cb2Data, 0, 0 };
  ID3D11Buffer* cb2 = nullptr;
  dev->CreateBuffer(&bd2, &bi2, &cb2);

  ID3D11SamplerState* samp = nullptr;
  D3D11_SAMPLER_DESC sd = { };
  sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
  sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
  sd.MaxLOD = D3D11_FLOAT32_MAX;
  dev->CreateSamplerState(&sd, &samp);

  D3D11_QUERY_DESC qd = { D3D11_QUERY_TIMESTAMP_DISJOINT, 0 };
  ID3D11Query* disjoint = nullptr;
  dev->CreateQuery(&qd, &disjoint);
  qd.Query = D3D11_QUERY_TIMESTAMP;
  ID3D11Query* ts0 = nullptr; ID3D11Query* ts1 = nullptr;
  dev->CreateQuery(&qd, &ts0);
  dev->CreateQuery(&qd, &ts1);

  D3D11_VIEWPORT vp = { 0, 0, float(W), float(H), 0, 1 };
  const FLOAT clear[4] = { 0, 0, 0, 0 };
  std::vector<double> times;

  for (int frame = 0; frame < 60; frame++) {
    ctx->Begin(disjoint);
    ctx->End(ts0);

    ctx->ClearRenderTargetView(rtv[0], clear);
    ctx->ClearRenderTargetView(rtv[1], clear);
    ctx->RSSetViewports(1, &vp);
    ctx->OMSetRenderTargets(2, rtv, nullptr);
    ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ctx->VSSetShader(vs, nullptr, 0);
    ctx->PSSetShader(ps, nullptr, 0);
    ctx->PSSetShaderResources(0, 6, srv);
    ID3D11SamplerState* samps6[6] = { samp, samp, samp, samp, samp, samp };
    ctx->PSSetSamplers(0, 6, samps6);
    ID3D11Buffer* cbs[13] = { };
    cbs[2] = cb2; cbs[12] = cb12;
    ctx->PSSetConstantBuffers(0, 13, cbs);
    ctx->Draw(3, 0);

    ctx->End(ts1);
    ctx->End(disjoint);

    D3D11_QUERY_DATA_TIMESTAMP_DISJOINT dj;
    while (ctx->GetData(disjoint, &dj, sizeof(dj), 0) != S_OK) { }
    UINT64 t0 = 0, t1 = 0;
    while (ctx->GetData(ts0, &t0, sizeof(t0), 0) != S_OK) { }
    while (ctx->GetData(ts1, &t1, sizeof(t1), 0) != S_OK) { }
    if (!dj.Disjoint && frame >= 10)
      times.push_back(double(t1 - t0) * 1000.0 / double(dj.Frequency));
  }

  std::sort(times.begin(), times.end());
  double median = times.empty() ? -1.0 : times[times.size() / 2];
  std::printf("%s: %.4f ms (median of 50)\n", argv[1], median);
  return 0;
}
