// blessed: shader-replace, pass 16 micro-benchmark: skyrim's vanilla sao rebuilt with its own pixel shader, timed and bit-diffed on native d3d11 or dxvk
//
// usage: pass16 PS.dxbc [PS2.dxbc]
//   with one dxbc: draws a fullscreen triangle with it into an rgba8 target
//   (no blend, no depth -- matches the vanilla draw's own state), reading 4
//   synthetic srvs shaped like the real pass (t0 r32f depth, t1 rgba8 normal,
//   t2 rgba8 gi buffer, t3 rg16f dither), timed with d3d11 timestamps, median
//   of 60 frames.
//   with two dxbc: also renders both into separate targets against the exact
//   same synthetic bindings and reads both back, reporting the max per-
//   channel absolute difference -- a same-process, same-inputs check that
//   needs no game and no D: write. it is not a substitute for the in-game
//   twin-draw verifier (BLESSED_SHADER_VERIFY): the inputs here are
//   synthetic, not the game's own depth/normal/gi buffers, so it proves the
//   two dxbc agree on this input, not on every input the game will ever
//   bind.
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

  // deterministic, plausible-range synthetic fields: no zeros where the
  // shader divides (depth, dither length), no degenerate normals
  float DepthField(uint32_t x, uint32_t y) {
    float v = 0.05f + 0.90f * (0.5f + 0.5f * std::sin(0.01f * x) * std::cos(0.017f * y));
    return v;
  }

  ID3D11ShaderResourceView* MakeDepthTex(ID3D11Device* dev, uint32_t w, uint32_t h) {
    std::vector<float> data(size_t(w) * h);
    for (uint32_t y = 0; y < h; y++)
      for (uint32_t x = 0; x < w; x++)
        data[size_t(y) * w + x] = DepthField(x, y);
    D3D11_TEXTURE2D_DESC desc = { w, h, 1, 1, DXGI_FORMAT_R32_FLOAT, { 1, 0 },
      D3D11_USAGE_IMMUTABLE, D3D11_BIND_SHADER_RESOURCE, 0, 0 };
    D3D11_SUBRESOURCE_DATA init = { data.data(), w * 4u, 0 };
    ID3D11Texture2D* tex = nullptr;
    ID3D11ShaderResourceView* srv = nullptr;
    dev->CreateTexture2D(&desc, &init, &tex);
    dev->CreateShaderResourceView(tex, nullptr, &srv);
    return srv;
  }

  ID3D11ShaderResourceView* MakeRgba8(ID3D11Device* dev, uint32_t w, uint32_t h, uint32_t seed) {
    std::vector<uint32_t> data(size_t(w) * h);
    for (size_t i = 0; i < data.size(); i++) {
      uint32_t x = uint32_t(i) * 2654435761u + seed;
      x ^= x >> 13; x *= 0x85ebca6bu; x ^= x >> 15;
      // keep every byte in [16, 239]: no exact-0/exact-255 corner cases
      uint8_t b0 = 16 + uint8_t(x & 0xffu) % 224u;
      uint8_t b1 = 16 + uint8_t((x >> 8) & 0xffu) % 224u;
      uint8_t b2 = 16 + uint8_t((x >> 16) & 0xffu) % 224u;
      uint8_t b3 = 16 + uint8_t((x >> 24) & 0xffu) % 224u;
      data[i] = uint32_t(b0) | (uint32_t(b1) << 8) | (uint32_t(b2) << 16) | (uint32_t(b3) << 24);
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

  ID3D11ShaderResourceView* MakeRg16f(ID3D11Device* dev, uint32_t w, uint32_t h) {
    std::vector<uint16_t> data(size_t(w) * h * 2u);
    for (uint32_t y = 0; y < h; y++) {
      for (uint32_t x = 0; x < w; x++) {
        // a rotating unit-ish vector, never (0,0): fp16 via a tiny manual pack
        float ang = 0.037f * x + 0.061f * y;
        float fx = 0.4f * std::cos(ang) + 0.1f;
        float fy = 0.4f * std::sin(ang) + 0.1f;
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

  struct Rig {
    ID3D11Device*        dev = nullptr;
    ID3D11DeviceContext* ctx = nullptr;
    ID3D11VertexShader*  vs  = nullptr;
    ID3D11ShaderResourceView* srv[4] = { };
    ID3D11Buffer* cb2  = nullptr;
    ID3D11Buffer* cb12 = nullptr;
    ID3D11SamplerState* samp = nullptr;
    const UINT W = 1920, H = 1080;
  };

  void SetupRig(Rig& r) {
    r.srv[0] = MakeDepthTex(r.dev, r.W, r.H);
    r.srv[1] = MakeRgba8(r.dev, r.W, r.H, 11u);
    r.srv[2] = MakeRgba8(r.dev, r.W / 2u, r.H / 2u, 22u);
    r.srv[3] = MakeRg16f(r.dev, r.W, r.H);

    ID3DBlob* vsBlob = CompileVs();
    r.dev->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), nullptr, &r.vs);

    // cb2: 4 vec4s, plausible non-degenerate values (see the .hlsl for field meaning)
    float cb2Data[16] = {
      1.0f / 1920, 1.0f / 1080, 0.5f / 1920, 0.5f / 1080,   // c0: texel size, half-texel bias
      1.0f, 40.0f, 0.02f, 0.6f,                             // c1: dither gate, radius scale, falloff bias, ao strength
      0.01f, 0.01f, 0.003f, 0.003f,                         // c2: dither seed scale, spiral step scale
      0.6f, 4.0f, 3.0f, 0.98f,                              // c3: power, max radius^2, gi sharpness, sky threshold
    };
    D3D11_BUFFER_DESC bd = { sizeof(cb2Data), D3D11_USAGE_IMMUTABLE, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0 };
    D3D11_SUBRESOURCE_DATA bi = { cb2Data, 0, 0 };
    r.dev->CreateBuffer(&bd, &bi, &r.cb2);

    static float cb12Data[45 * 4] = { };
    cb12Data[43 * 4 + 0] = 1920.0f; cb12Data[43 * 4 + 1] = 1079.0f;
    cb12Data[43 * 4 + 2] = 960.0f;  cb12Data[43 * 4 + 3] = 539.0f;
    cb12Data[44 * 4 + 0] = 0.0f;    cb12Data[44 * 4 + 2] = 1919.0f; cb12Data[44 * 4 + 3] = 539.0f;
    D3D11_BUFFER_DESC bd2 = { sizeof(cb12Data), D3D11_USAGE_IMMUTABLE, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0 };
    D3D11_SUBRESOURCE_DATA bi2 = { cb12Data, 0, 0 };
    r.dev->CreateBuffer(&bd2, &bi2, &r.cb12);

    D3D11_SAMPLER_DESC sd = { };
    sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.MaxLOD = D3D11_FLOAT32_MAX;
    r.dev->CreateSamplerState(&sd, &r.samp);
  }

  ID3D11PixelShader* LoadPs(ID3D11Device* dev, const char* path) {
    std::vector<uint8_t> code = ReadFile(path);
    ID3D11PixelShader* ps = nullptr;
    if (code.empty() || FAILED(dev->CreatePixelShader(code.data(), code.size(), nullptr, &ps))) {
      std::printf("CreatePixelShader failed for %s\n", path);
      std::exit(1);
    }
    return ps;
  }

  // renders one frame with `ps` into `rtv`, no timing, for the diff path
  void RenderOnce(Rig& r, ID3D11PixelShader* ps, ID3D11RenderTargetView* rtv) {
    const FLOAT clear[4] = { 0, 0, 0, 0 };
    r.ctx->ClearRenderTargetView(rtv, clear);
    D3D11_VIEWPORT vp = { 0, 0, float(r.W), float(r.H), 0, 1 };
    r.ctx->RSSetViewports(1, &vp);
    r.ctx->OMSetRenderTargets(1, &rtv, nullptr);
    r.ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    r.ctx->VSSetShader(r.vs, nullptr, 0);
    r.ctx->PSSetShader(ps, nullptr, 0);
    ID3D11ShaderResourceView* srvs4[4] = { r.srv[0], r.srv[1], r.srv[2], r.srv[3] };
    r.ctx->PSSetShaderResources(0, 4, srvs4);
    ID3D11SamplerState* samps4[4] = { r.samp, r.samp, r.samp, r.samp };
    r.ctx->PSSetSamplers(0, 4, samps4);
    ID3D11Buffer* cbs[13] = { };
    cbs[2] = r.cb2; cbs[12] = r.cb12;
    r.ctx->PSSetConstantBuffers(0, 13, cbs);
    r.ctx->Draw(3, 0);
  }

  double TimePass(Rig& r, ID3D11PixelShader* ps) {
    D3D11_TEXTURE2D_DESC rtDesc = { r.W, r.H, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM, { 1, 0 },
      D3D11_USAGE_DEFAULT, D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE, 0, 0 };
    ID3D11Texture2D* rtTex = nullptr;
    r.dev->CreateTexture2D(&rtDesc, nullptr, &rtTex);
    ID3D11RenderTargetView* rtv = nullptr;
    r.dev->CreateRenderTargetView(rtTex, nullptr, &rtv);

    D3D11_QUERY_DESC qd = { D3D11_QUERY_TIMESTAMP_DISJOINT, 0 };
    ID3D11Query* disjoint = nullptr;
    r.dev->CreateQuery(&qd, &disjoint);
    qd.Query = D3D11_QUERY_TIMESTAMP;
    ID3D11Query* ts0 = nullptr; ID3D11Query* ts1 = nullptr;
    r.dev->CreateQuery(&qd, &ts0);
    r.dev->CreateQuery(&qd, &ts1);

    std::vector<double> times;
    for (int frame = 0; frame < 60; frame++) {
      r.ctx->Begin(disjoint);
      r.ctx->End(ts0);
      RenderOnce(r, ps, rtv);
      r.ctx->End(ts1);
      r.ctx->End(disjoint);

      D3D11_QUERY_DATA_TIMESTAMP_DISJOINT dj;
      while (r.ctx->GetData(disjoint, &dj, sizeof(dj), 0) != S_OK) { }
      UINT64 t0 = 0, t1 = 0;
      while (r.ctx->GetData(ts0, &t0, sizeof(t0), 0) != S_OK) { }
      while (r.ctx->GetData(ts1, &t1, sizeof(t1), 0) != S_OK) { }
      if (!dj.Disjoint && frame >= 10)
        times.push_back(double(t1 - t0) * 1000.0 / double(dj.Frequency));
    }

    std::sort(times.begin(), times.end());
    rtv->Release(); rtTex->Release();
    disjoint->Release(); ts0->Release(); ts1->Release();
    return times.empty() ? -1.0 : times[times.size() / 2];
  }

  // renders `ps` once, reads the target back to a host buffer
  void RenderAndReadback(Rig& r, ID3D11PixelShader* ps, std::vector<uint8_t>& out) {
    D3D11_TEXTURE2D_DESC rtDesc = { r.W, r.H, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM, { 1, 0 },
      D3D11_USAGE_DEFAULT, D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE, 0, 0 };
    ID3D11Texture2D* rtTex = nullptr;
    r.dev->CreateTexture2D(&rtDesc, nullptr, &rtTex);
    ID3D11RenderTargetView* rtv = nullptr;
    r.dev->CreateRenderTargetView(rtTex, nullptr, &rtv);
    RenderOnce(r, ps, rtv);

    D3D11_TEXTURE2D_DESC stDesc = rtDesc;
    stDesc.Usage = D3D11_USAGE_STAGING;
    stDesc.BindFlags = 0;
    stDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ID3D11Texture2D* stage = nullptr;
    r.dev->CreateTexture2D(&stDesc, nullptr, &stage);
    r.ctx->CopyResource(stage, rtTex);

    D3D11_MAPPED_SUBRESOURCE mapped = { };
    r.ctx->Map(stage, 0, D3D11_MAP_READ, 0, &mapped);
    out.resize(size_t(r.W) * r.H * 4u);
    for (uint32_t y = 0; y < r.H; y++)
      std::memcpy(&out[size_t(y) * r.W * 4u], reinterpret_cast<const uint8_t*>(mapped.pData) + size_t(y) * mapped.RowPitch, size_t(r.W) * 4u);
    r.ctx->Unmap(stage, 0);

    rtv->Release(); rtTex->Release(); stage->Release();
  }

}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: pass16 PS.dxbc [PS2.dxbc]\n");
    return 1;
  }

  HMODULE mod = LoadLibraryA("d3d11.dll");
  auto create = reinterpret_cast<PFN_D3D11_CREATE_DEVICE>(GetProcAddress(mod, "D3D11CreateDevice"));
  Rig r;
  D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_0;
  if (!create || FAILED(create(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, &fl, 1, D3D11_SDK_VERSION, &r.dev, nullptr, &r.ctx))) {
    std::printf("no device\n");
    return 1;
  }
  SetupRig(r);

  ID3D11PixelShader* psA = LoadPs(r.dev, argv[1]);
  double timeA = TimePass(r, psA);
  std::printf("%s: %.4f ms (median of 50)\n", argv[1], timeA);

  if (argc >= 3) {
    ID3D11PixelShader* psB = LoadPs(r.dev, argv[2]);
    double timeB = TimePass(r, psB);
    std::printf("%s: %.4f ms (median of 50)\n", argv[2], timeB);

    std::vector<uint8_t> outA, outB;
    RenderAndReadback(r, psA, outA);
    RenderAndReadback(r, psB, outB);

    uint32_t maxDiff[4] = { 0, 0, 0, 0 };
    uint64_t differing = 0;
    for (size_t i = 0; i < outA.size(); i += 4) {
      for (int c = 0; c < 4; c++) {
        int d = std::abs(int(outA[i + c]) - int(outB[i + c]));
        if (d) differing++;
        maxDiff[c] = std::max(maxDiff[c], uint32_t(d));
      }
    }
    std::printf("diff (synthetic inputs, 8-bit channels): %llu differing byte(s) of %zu, max per-channel [%u,%u,%u,%u]\n",
      (unsigned long long) differing, outA.size(), maxDiff[0], maxDiff[1], maxDiff[2], maxDiff[3]);
  }

  return 0;
}
