// blessed: shader-replace, offline harness: runs a game compute shader or a test draw on dxvk without the game
//
// usage:
//   harness cs VANILLA.dxbc OUT.bin GX GY GZ   run the blur shaders' binding set (t0, t1, u0, cb0,
//                                              s0, s1) on 1920x1080 r32f inputs, dispatch 50x, dump u0
//   harness ps                                  draw a fullscreen triangle with a test pixel shader
//                                              (checks the verifier's draw path)
// the d3d11.dll next to the exe is the one used (copy the fork's d3d11.dll and
// dxgi.dll there). BLESSED_SHADER_REPLACE / BLESSED_SHADER_VERIFY apply as in game.
#define NOMINMAX
#include <d3d11.h>
#include <d3dcompiler.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <vector>

namespace {

  PFN_D3D11_CREATE_DEVICE g_create = nullptr;

  std::vector<uint8_t> ReadFile(const char* path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  }

  bool CreateDevice(ID3D11Device** dev, ID3D11DeviceContext** ctx) {
    HMODULE mod = LoadLibraryA("d3d11.dll");
    if (!mod)
      return false;
    g_create = reinterpret_cast<PFN_D3D11_CREATE_DEVICE>(GetProcAddress(mod, "D3D11CreateDevice"));
    D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_0;
    return g_create && SUCCEEDED(g_create(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, &fl, 1,
      D3D11_SDK_VERSION, dev, nullptr, ctx));
  }

  // a smooth field with hard steps, so the blur's edge test takes both paths
  float Field(uint32_t x, uint32_t y, uint32_t seed) {
    float v = 0.5f + 0.25f * std::sin(0.013f * x + seed) * std::cos(0.021f * y);
    if (((x / 37u) + (y / 53u) + seed) % 5u == 0u)
      v += 0.3f;
    uint32_t h = (x * 1973u + y * 9277u + seed * 26699u) | 1u;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    return v + float(h & 0xffu) * (1.0f / 65536.0f);
  }

  ID3D11ShaderResourceView* MakeInput(ID3D11Device* dev, uint32_t w, uint32_t h, uint32_t seed) {
    std::vector<float> data(size_t(w) * h);
    for (uint32_t y = 0; y < h; y++)
      for (uint32_t x = 0; x < w; x++)
        data[size_t(y) * w + x] = Field(x, y, seed);

    D3D11_TEXTURE2D_DESC desc = { w, h, 1, 1, DXGI_FORMAT_R32_FLOAT, { 1, 0 },
      D3D11_USAGE_IMMUTABLE, D3D11_BIND_SHADER_RESOURCE, 0, 0 };
    D3D11_SUBRESOURCE_DATA init = { data.data(), w * 4u, 0 };
    ID3D11Texture2D* tex = nullptr;
    ID3D11ShaderResourceView* srv = nullptr;
    dev->CreateTexture2D(&desc, &init, &tex);
    dev->CreateShaderResourceView(tex, nullptr, &srv);
    tex->Release();
    return srv;
  }

  int RunCs(const char* shaderPath, const char* outPath, UINT gx, UINT gy, UINT gz) {
    const uint32_t w = 1920u, h = 1080u;
    ID3D11Device* dev = nullptr;
    ID3D11DeviceContext* ctx = nullptr;
    if (!CreateDevice(&dev, &ctx)) {
      std::printf("no device\n");
      return 1;
    }

    std::vector<uint8_t> code = ReadFile(shaderPath);
    ID3D11ComputeShader* cs = nullptr;
    if (FAILED(dev->CreateComputeShader(code.data(), code.size(), nullptr, &cs))) {
      std::printf("CreateComputeShader failed\n");
      return 1;
    }

    ID3D11ShaderResourceView* srv[2] = { MakeInput(dev, w, h, 1u), MakeInput(dev, w, h, 2u) };

    D3D11_TEXTURE2D_DESC desc = { w, h, 1, 1, DXGI_FORMAT_R32_FLOAT, { 1, 0 },
      D3D11_USAGE_DEFAULT, D3D11_BIND_UNORDERED_ACCESS, 0, 0 };
    ID3D11Texture2D* out = nullptr;
    ID3D11UnorderedAccessView* uav = nullptr;
    dev->CreateTexture2D(&desc, nullptr, &out);
    dev->CreateUnorderedAccessView(out, nullptr, &uav);

    float cbData[8] = { 1.0f / w, 1.0f / h, 0, 0, 1.0f - 0.5f / w, 1.0f - 0.5f / h, 0, 0 };
    D3D11_BUFFER_DESC cbDesc = { sizeof(cbData), D3D11_USAGE_IMMUTABLE, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0 };
    D3D11_SUBRESOURCE_DATA cbInit = { cbData, 0, 0 };
    ID3D11Buffer* cb = nullptr;
    dev->CreateBuffer(&cbDesc, &cbInit, &cb);

    D3D11_SAMPLER_DESC sd = { };
    sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.MaxLOD = D3D11_FLOAT32_MAX;
    ID3D11SamplerState* samp[2] = { };
    dev->CreateSamplerState(&sd, &samp[0]);
    sd.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
    dev->CreateSamplerState(&sd, &samp[1]);

    ctx->CSSetShader(cs, nullptr, 0);
    ctx->CSSetShaderResources(0, 2, srv);
    ctx->CSSetUnorderedAccessViews(0, 1, &uav, nullptr);
    ctx->CSSetConstantBuffers(0, 1, &cb);
    ctx->CSSetSamplers(0, 2, samp);

    D3D11_QUERY_DESC qd = { D3D11_QUERY_TIMESTAMP_DISJOINT, 0 };
    ID3D11Query* disjoint = nullptr;
    ID3D11Query* ts[2] = { };
    dev->CreateQuery(&qd, &disjoint);
    qd.Query = D3D11_QUERY_TIMESTAMP;
    dev->CreateQuery(&qd, &ts[0]);
    dev->CreateQuery(&qd, &ts[1]);

    const uint32_t reps = 50u;
    double best = 1e30;

    for (uint32_t round = 0; round < 5u; round++) {
      ctx->Begin(disjoint);
      ctx->End(ts[0]);
      for (uint32_t i = 0; i < reps; i++)
        ctx->Dispatch(gx, gy, gz);
      ctx->End(ts[1]);
      ctx->End(disjoint);

      D3D11_QUERY_DATA_TIMESTAMP_DISJOINT dj = { };
      UINT64 t0 = 0, t1 = 0;
      while (ctx->GetData(disjoint, &dj, sizeof(dj), 0) != S_OK) { }
      while (ctx->GetData(ts[0], &t0, sizeof(t0), 0) != S_OK) { }
      while (ctx->GetData(ts[1], &t1, sizeof(t1), 0) != S_OK) { }

      if (!dj.Disjoint && dj.Frequency) {
        double ms = double(t1 - t0) * 1000.0 / double(dj.Frequency) / reps;
        if (ms < best)
          best = ms;
      }
    }

    std::printf("%s: %.4f ms per dispatch (best of 5 x %u)\n", shaderPath, best, reps);

    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ID3D11Texture2D* staging = nullptr;
    dev->CreateTexture2D(&desc, nullptr, &staging);
    ctx->CopyResource(staging, out);

    D3D11_MAPPED_SUBRESOURCE mapped = { };
    ctx->Map(staging, 0, D3D11_MAP_READ, 0, &mapped);
    std::ofstream f(outPath, std::ios::binary);
    for (uint32_t y = 0; y < h; y++)
      f.write(reinterpret_cast<const char*>(mapped.pData) + size_t(y) * mapped.RowPitch, w * 4u);
    ctx->Unmap(staging, 0);
    return 0;
  }

  // the vanilla volumetric integration: one z slice per dispatch, ping-pong
  // between two 3d textures, the way pass 14 runs it; then the same prefix
  // sum as one dispatch that loops over z (the collapse a fork-level
  // dispatch skip would make possible)
  const char* g_collapse =
    "Texture3D<float4> t0 : register(t0);\n"
    "RWTexture3D<float4> u0 : register(u0);\n"
    "RWTexture3D<float4> u1 : register(u1);\n"
    "[numthreads(8, 8, 1)]\n"
    "void main(uint3 id : SV_DispatchThreadID) {\n"
    "  uint w, h, d;\n"
    "  t0.GetDimensions(w, h, d);\n"
    "  if (id.x >= w || id.y >= h) return;\n"
    "  float s = t0.Load(int4(id.xy, 0, 0)).x;\n"
    "  u0[uint3(id.xy, 0)] = s.xxxx; u1[uint3(id.xy, 0)] = s.xxxx;\n"
    "  for (uint z = 1; z < d; z++) {\n"
    "    s = s + t0.Load(int4(id.xy, z, 0)).x;\n"
    "    u0[uint3(id.xy, z)] = s.xxxx; u1[uint3(id.xy, z)] = s.xxxx;\n"
    "  }\n"
    "}\n";

  ID3DBlob* Compile(const char* src, const char* profile);

  int RunChain(const char* shaderPath, UINT w, UINT h, UINT d) {
    ID3D11Device* dev = nullptr;
    ID3D11DeviceContext* ctx = nullptr;
    if (!CreateDevice(&dev, &ctx)) {
      std::printf("no device\n");
      return 1;
    }

    std::vector<uint8_t> code = ReadFile(shaderPath);
    ID3D11ComputeShader* cs = nullptr;
    if (FAILED(dev->CreateComputeShader(code.data(), code.size(), nullptr, &cs))) {
      std::printf("CreateComputeShader failed\n");
      return 1;
    }

    ID3DBlob* collapseBlob = Compile(g_collapse, "cs_5_0");
    ID3D11ComputeShader* collapse = nullptr;
    dev->CreateComputeShader(collapseBlob->GetBufferPointer(), collapseBlob->GetBufferSize(), nullptr, &collapse);

    std::vector<float> data(size_t(w) * h * d);
    for (size_t i = 0; i < data.size(); i++)
      data[i] = Field(uint32_t(i % w), uint32_t(i / w), 3u) * 0.01f;

    D3D11_TEXTURE3D_DESC desc = { w, h, d, 1, DXGI_FORMAT_R32_FLOAT, D3D11_USAGE_DEFAULT,
      D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS, 0, 0 };
    D3D11_SUBRESOURCE_DATA init = { data.data(), w * 4u, w * h * 4u };
    ID3D11Texture3D* tex[3] = { };
    ID3D11ShaderResourceView* srv[3] = { };
    ID3D11UnorderedAccessView* uav[3] = { };
    for (int i = 0; i < 3; i++) {
      dev->CreateTexture3D(&desc, &init, &tex[i]);
      dev->CreateShaderResourceView(tex[i], nullptr, &srv[i]);
      dev->CreateUnorderedAccessView(tex[i], nullptr, &uav[i]);
    }

    D3D11_BUFFER_DESC cbDesc = { 32, D3D11_USAGE_DYNAMIC, D3D11_BIND_CONSTANT_BUFFER, D3D11_CPU_ACCESS_WRITE, 0, 0 };
    ID3D11Buffer* cb = nullptr;
    dev->CreateBuffer(&cbDesc, nullptr, &cb);

    D3D11_SAMPLER_DESC sd = { };
    sd.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
    sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.MaxLOD = D3D11_FLOAT32_MAX;
    ID3D11SamplerState* samp = nullptr;
    dev->CreateSamplerState(&sd, &samp);

    D3D11_QUERY_DESC qd = { D3D11_QUERY_TIMESTAMP_DISJOINT, 0 };
    ID3D11Query* disjoint = nullptr;
    ID3D11Query* ts[3] = { };
    dev->CreateQuery(&qd, &disjoint);
    qd.Query = D3D11_QUERY_TIMESTAMP;
    for (auto& q : ts)
      dev->CreateQuery(&qd, &q);

    double bestChain = 1e30, bestCollapse = 1e30;
    ID3D11ShaderResourceView* nullSrv = nullptr;
    ID3D11UnorderedAccessView* nullUav[2] = { };

    for (uint32_t round = 0; round < 5u; round++) {
      ctx->Begin(disjoint);
      ctx->End(ts[0]);

      ctx->CSSetShader(cs, nullptr, 0);
      ctx->CSSetSamplers(0, 1, &samp);
      for (UINT k = 1; k < d; k++) {
        D3D11_MAPPED_SUBRESOURCE m = { };
        ctx->Map(cb, 0, D3D11_MAP_WRITE_DISCARD, 0, &m);
        float* f = reinterpret_cast<float*>(m.pData);
        f[0] = float(w); f[1] = float(h); f[2] = float(d); f[3] = 0.0f;
        uint32_t* u = reinterpret_cast<uint32_t*>(f + 4);
        u[0] = k; u[1] = u[2] = u[3] = 0u;
        ctx->Unmap(cb, 0);
        ID3D11ShaderResourceView* s = srv[(k & 1u) ? 0 : 1];
        ID3D11UnorderedAccessView* o = uav[(k & 1u) ? 1 : 0];
        ctx->CSSetShaderResources(0, 1, &nullSrv);
        ctx->CSSetUnorderedAccessViews(0, 1, &o, nullptr);
        ctx->CSSetShaderResources(0, 1, &s);
        ctx->CSSetConstantBuffers(0, 1, &cb);
        ctx->Dispatch((w + 31u) / 32u, (h + 31u) / 32u, 1u);
      }
      ctx->CSSetShaderResources(0, 1, &nullSrv);
      ctx->CSSetUnorderedAccessViews(0, 2, nullUav, nullptr);
      ctx->End(ts[1]);

      ctx->CSSetShader(collapse, nullptr, 0);
      ctx->CSSetShaderResources(0, 1, &srv[2]);
      ID3D11UnorderedAccessView* outs[2] = { uav[0], uav[1] };
      ctx->CSSetUnorderedAccessViews(0, 2, outs, nullptr);
      ctx->Dispatch((w + 7u) / 8u, (h + 7u) / 8u, 1u);
      ctx->CSSetShaderResources(0, 1, &nullSrv);
      ctx->CSSetUnorderedAccessViews(0, 2, nullUav, nullptr);
      ctx->End(ts[2]);
      ctx->End(disjoint);

      D3D11_QUERY_DATA_TIMESTAMP_DISJOINT dj = { };
      UINT64 t[3] = { };
      while (ctx->GetData(disjoint, &dj, sizeof(dj), 0) != S_OK) { }
      for (int i = 0; i < 3; i++)
        while (ctx->GetData(ts[i], &t[i], sizeof(t[i]), 0) != S_OK) { }

      if (!dj.Disjoint && dj.Frequency) {
        bestChain = std::min(bestChain, double(t[1] - t[0]) * 1000.0 / double(dj.Frequency));
        bestCollapse = std::min(bestCollapse, double(t[2] - t[1]) * 1000.0 / double(dj.Frequency));
      }

      // the textures now hold prefix sums; reset them for the next round
      for (int i = 0; i < 2; i++)
        ctx->UpdateSubresource(tex[i], 0, nullptr, data.data(), w * 4u, w * h * 4u);
    }

    std::printf("volume %ux%ux%u: vanilla chain (%u dispatches) %.4f ms, one-dispatch collapse %.4f ms\n",
      w, h, d, d - 1u, bestChain, bestCollapse);
    return 0;
  }

  uint16_t ToHalf(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = int32_t((x >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0)
      return uint16_t(sign);
    if (exp >= 31)
      return uint16_t(sign | 0x7c00u);
    return uint16_t(sign | (uint32_t(exp) << 10) | (mant >> 13));
  }

  // "chain2 CHAIN GEN W H D ROUNDS OUT [linear|point]": the game's pass 14
  // shape. per round: the two r16f volumes reset to the same data, a
  // generate dispatch (the vanilla generate cs, nothing bound: it only
  // marks the frame), then D dispatches of the vanilla chain cs with
  // k = 1 .. D, ping-pong A -> B first, as in the whiterun dump. the
  // chain is timed per round. after the last round both volumes are
  // written to OUT.a and OUT.b. with the fork's dll and BLESSED_VOL_COLLAPSE=1
  // round 1 is learned and later rounds are collapsed.
  int RunChain2(const char* chainPath, const char* genPath, UINT w, UINT h, UINT d,
      UINT rounds, const char* outPath, bool point) {
    ID3D11Device* dev = nullptr;
    ID3D11DeviceContext* ctx = nullptr;
    if (!CreateDevice(&dev, &ctx)) {
      std::printf("no device\n");
      return 1;
    }

    ID3D11ComputeShader* chain = nullptr;
    ID3D11ComputeShader* gen = nullptr;
    std::vector<uint8_t> chainCode = ReadFile(chainPath);
    std::vector<uint8_t> genCode = ReadFile(genPath);
    if (FAILED(dev->CreateComputeShader(chainCode.data(), chainCode.size(), nullptr, &chain))
     || FAILED(dev->CreateComputeShader(genCode.data(), genCode.size(), nullptr, &gen))) {
      std::printf("CreateComputeShader failed\n");
      return 1;
    }

    std::vector<uint16_t> data(size_t(w) * h * d);
    for (size_t i = 0; i < data.size(); i++) {
      uint32_t x = uint32_t(i % w), y = uint32_t((i / w) % h), z = uint32_t(i / (size_t(w) * h));
      data[i] = ToHalf(Field(x + 7u * z, y, 3u) * 0.02f * (1.0f + 0.01f * float(z)));
    }

    D3D11_TEXTURE3D_DESC desc = { w, h, d, 1, DXGI_FORMAT_R16_FLOAT, D3D11_USAGE_DEFAULT,
      D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS, 0, 0 };
    ID3D11Texture3D* tex[2] = { };
    ID3D11ShaderResourceView* srv[2] = { };
    ID3D11UnorderedAccessView* uav[2] = { };
    for (int i = 0; i < 2; i++) {
      dev->CreateTexture3D(&desc, nullptr, &tex[i]);
      dev->CreateShaderResourceView(tex[i], nullptr, &srv[i]);
      dev->CreateUnorderedAccessView(tex[i], nullptr, &uav[i]);
    }

    D3D11_BUFFER_DESC cbDesc = { 32, D3D11_USAGE_DYNAMIC, D3D11_BIND_CONSTANT_BUFFER, D3D11_CPU_ACCESS_WRITE, 0, 0 };
    ID3D11Buffer* cb = nullptr;
    dev->CreateBuffer(&cbDesc, nullptr, &cb);

    D3D11_SAMPLER_DESC sd = { };
    sd.Filter = point ? D3D11_FILTER_MIN_MAG_MIP_POINT : D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.MaxLOD = D3D11_FLOAT32_MAX;
    ID3D11SamplerState* samp = nullptr;
    dev->CreateSamplerState(&sd, &samp);

    D3D11_QUERY_DESC qd = { D3D11_QUERY_TIMESTAMP_DISJOINT, 0 };
    ID3D11Query* disjoint = nullptr;
    ID3D11Query* ts[2] = { };
    dev->CreateQuery(&qd, &disjoint);
    qd.Query = D3D11_QUERY_TIMESTAMP;
    dev->CreateQuery(&qd, &ts[0]);
    dev->CreateQuery(&qd, &ts[1]);

    ID3D11ShaderResourceView* nullSrv = nullptr;
    ID3D11UnorderedAccessView* nullUav[2] = { };
    std::vector<double> times;

    for (UINT round = 0; round < rounds; round++) {
      for (int i = 0; i < 2; i++)
        ctx->UpdateSubresource(tex[i], 0, nullptr, data.data(), w * 2u, w * h * 2u);

      ctx->CSSetShaderResources(0, 1, &nullSrv);
      ctx->CSSetUnorderedAccessViews(0, 2, nullUav, nullptr);
      ctx->CSSetShader(gen, nullptr, 0);
      ctx->Dispatch(1, 1, 1);

      ctx->Begin(disjoint);
      ctx->End(ts[0]);

      ctx->CSSetShader(chain, nullptr, 0);
      ctx->CSSetSamplers(0, 1, &samp);
      ctx->CSSetConstantBuffers(0, 1, &cb);

      for (UINT k = 1; k <= d; k++) {
        D3D11_MAPPED_SUBRESOURCE m = { };
        ctx->Map(cb, 0, D3D11_MAP_WRITE_DISCARD, 0, &m);
        float* f = reinterpret_cast<float*>(m.pData);
        f[0] = float(w); f[1] = float(h); f[2] = float(d); f[3] = 0.0f;
        uint32_t* u = reinterpret_cast<uint32_t*>(f + 4);
        u[0] = k; u[1] = u[2] = u[3] = 0u;
        ctx->Unmap(cb, 0);

        int src = (k & 1u) ? 0 : 1;
        ctx->CSSetShaderResources(0, 1, &nullSrv);
        ctx->CSSetUnorderedAccessViews(0, 1, &uav[1 - src], nullptr);
        ctx->CSSetShaderResources(0, 1, &srv[src]);
        ctx->Dispatch((w + 31u) / 32u, (h + 31u) / 32u, 1u);
      }

      ctx->CSSetShaderResources(0, 1, &nullSrv);
      ctx->CSSetUnorderedAccessViews(0, 1, nullUav, nullptr);
      ctx->End(ts[1]);
      ctx->End(disjoint);

      D3D11_QUERY_DATA_TIMESTAMP_DISJOINT dj = { };
      UINT64 t0 = 0, t1 = 0;
      while (ctx->GetData(disjoint, &dj, sizeof(dj), 0) != S_OK) { }
      while (ctx->GetData(ts[0], &t0, sizeof(t0), 0) != S_OK) { }
      while (ctx->GetData(ts[1], &t1, sizeof(t1), 0) != S_OK) { }
      times.push_back(dj.Disjoint || !dj.Frequency ? -1.0 : double(t1 - t0) * 1000.0 / double(dj.Frequency));
    }

    std::vector<double> later(times.begin() + 1, times.end());
    std::sort(later.begin(), later.end());
    std::printf("volume %ux%ux%u r16f, %u chain dispatches, %s sampler: round 1 %.4f ms, rounds 2-%u median %.4f ms\n",
      w, h, d, d, point ? "point" : "linear", times[0], rounds, later.empty() ? 0.0 : later[later.size() / 2]);

    for (int i = 0; i < 2; i++) {
      desc.Usage = D3D11_USAGE_STAGING;
      desc.BindFlags = 0;
      desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      ID3D11Texture3D* staging = nullptr;
      dev->CreateTexture3D(&desc, nullptr, &staging);
      ctx->CopyResource(staging, tex[i]);
      D3D11_MAPPED_SUBRESOURCE mapped = { };
      ctx->Map(staging, 0, D3D11_MAP_READ, 0, &mapped);
      std::ofstream f(std::string(outPath) + (i ? ".b" : ".a"), std::ios::binary);
      for (UINT z = 0; z < d; z++)
        for (UINT y = 0; y < h; y++)
          f.write(reinterpret_cast<const char*>(mapped.pData) + size_t(z) * mapped.DepthPitch + size_t(y) * mapped.RowPitch, w * 2u);
      ctx->Unmap(staging, 0);
    }

    return 0;
  }

  // "collapsecs CS W H D OUT": the chain2 volumes (same data), then one
  // dispatch of CS (a collapse kernel: u0 = A, u1 = B, cb0 = { k1 = 1,
  // count = D }), timed; both volumes written to OUT.a / OUT.b
  int RunCollapseCs(const char* csPath, UINT w, UINT h, UINT d, const char* outPath, UINT tx, UINT ty) {
    ID3D11Device* dev = nullptr;
    ID3D11DeviceContext* ctx = nullptr;
    if (!CreateDevice(&dev, &ctx))
      return 1;

    std::vector<uint8_t> code = ReadFile(csPath);
    ID3D11ComputeShader* cs = nullptr;
    if (FAILED(dev->CreateComputeShader(code.data(), code.size(), nullptr, &cs)))
      return 1;

    std::vector<uint16_t> data(size_t(w) * h * d);
    for (size_t i = 0; i < data.size(); i++) {
      uint32_t x = uint32_t(i % w), y = uint32_t((i / w) % h), z = uint32_t(i / (size_t(w) * h));
      data[i] = ToHalf(Field(x + 7u * z, y, 3u) * 0.02f * (1.0f + 0.01f * float(z)));
    }

    D3D11_TEXTURE3D_DESC desc = { w, h, d, 1, DXGI_FORMAT_R16_FLOAT, D3D11_USAGE_DEFAULT,
      D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS, 0, 0 };
    ID3D11Texture3D* tex[2] = { };
    ID3D11UnorderedAccessView* uav[2] = { };
    for (int i = 0; i < 2; i++) {
      dev->CreateTexture3D(&desc, nullptr, &tex[i]);
      dev->CreateUnorderedAccessView(tex[i], nullptr, &uav[i]);
    }

    uint32_t params[4] = { 1u, d, 0u, 0u };
    D3D11_BUFFER_DESC cbDesc = { 16, D3D11_USAGE_IMMUTABLE, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0 };
    D3D11_SUBRESOURCE_DATA cbInit = { params, 0, 0 };
    ID3D11Buffer* cb = nullptr;
    dev->CreateBuffer(&cbDesc, &cbInit, &cb);

    D3D11_QUERY_DESC qd = { D3D11_QUERY_TIMESTAMP_DISJOINT, 0 };
    ID3D11Query* disjoint = nullptr;
    ID3D11Query* ts[2] = { };
    dev->CreateQuery(&qd, &disjoint);
    qd.Query = D3D11_QUERY_TIMESTAMP;
    dev->CreateQuery(&qd, &ts[0]);
    dev->CreateQuery(&qd, &ts[1]);

    std::vector<double> times;
    for (int round = 0; round < 6; round++) {
      for (int i = 0; i < 2; i++)
        ctx->UpdateSubresource(tex[i], 0, nullptr, data.data(), w * 2u, w * h * 2u);
      ctx->CSSetShader(cs, nullptr, 0);
      ctx->CSSetUnorderedAccessViews(0, 2, uav, nullptr);
      ctx->CSSetConstantBuffers(0, 1, &cb);
      ctx->Begin(disjoint);
      ctx->End(ts[0]);
      ctx->Dispatch(((w + 31u) / 32u) * (32u / tx), ((h + 31u) / 32u) * (32u / ty), 1u);
      ctx->End(ts[1]);
      ctx->End(disjoint);
      D3D11_QUERY_DATA_TIMESTAMP_DISJOINT dj = { };
      UINT64 t0 = 0, t1 = 0;
      while (ctx->GetData(disjoint, &dj, sizeof(dj), 0) != S_OK) { }
      while (ctx->GetData(ts[0], &t0, sizeof(t0), 0) != S_OK) { }
      while (ctx->GetData(ts[1], &t1, sizeof(t1), 0) != S_OK) { }
      times.push_back(double(t1 - t0) * 1000.0 / double(dj.Frequency));
    }
    std::sort(times.begin(), times.end());
    std::printf("%s: %.4f ms (median of 6)\n", csPath, times[3]);

    for (int i = 0; i < 2; i++) {
      D3D11_TEXTURE3D_DESC sdesc = desc;
      sdesc.Usage = D3D11_USAGE_STAGING;
      sdesc.BindFlags = 0;
      sdesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      ID3D11Texture3D* staging = nullptr;
      dev->CreateTexture3D(&sdesc, nullptr, &staging);
      ctx->CopyResource(staging, tex[i]);
      D3D11_MAPPED_SUBRESOURCE mapped = { };
      ctx->Map(staging, 0, D3D11_MAP_READ, 0, &mapped);
      std::ofstream f(std::string(outPath) + (i ? ".b" : ".a"), std::ios::binary);
      for (UINT z = 0; z < d; z++)
        for (UINT y = 0; y < h; y++)
          f.write(reinterpret_cast<const char*>(mapped.pData) + size_t(z) * mapped.DepthPitch + size_t(y) * mapped.RowPitch, w * 2u);
      ctx->Unmap(staging, 0);
    }
    return 0;
  }

  // "volgen SHADER GX GY GZ OUT": binds the vanilla volumetric generate
  // shader's exact resource shapes (cs.ab674eb1..., a 32x32x1 vanilla
  // dispatch of 10x6x90) and dispatches it at the given GX/GY/GZ -- either
  // the game's own shape or a reshaped-group replacement's matching
  // dispatch -- with synthetic, deterministic inputs. Used to compare a
  // group-shape-only rewrite's u0/u1 output against vanilla's bitwise: same
  // shader arithmetic, same inputs, only the group declaration and dispatch
  // mapping differ, so any coordinate the game would have written must
  // still be written to the same value.
  //
  // t0, t1: texture2darray, r16_unorm, 4096x4096 / 512x512 (the far/near
  //   shadow atlas slices this shader samples with sample_l)
  // t2:     texture1d, r32_float, 4096 (a 1d lookup this shader indexes by
  //   an immediate-constant-buffer offset)
  // t3:     texture3d, r8_uint, 32x32x32 (a small noise volume)
  // u0, u1: texture3d, r16_float, 320x192x90 (the two density outputs;
  //   320*192*90 = 5,529,600 voxels matches the vanilla 10x6x90 @ 32x32x1
  //   dispatch-thread space exactly)
  // cb0:    24 float4 (384 bytes), synthetic but deterministic -- exactness
  //   here only requires the two runs to see identical inputs, not that
  //   the values are the game's actual per-frame constants.
  int RunVolgen(const char* shaderPath, UINT gx, UINT gy, UINT gz, const char* outPath) {
    ID3D11Device* dev = nullptr;
    ID3D11DeviceContext* ctx = nullptr;
    if (!CreateDevice(&dev, &ctx)) {
      std::printf("no device\n");
      return 1;
    }

    std::vector<uint8_t> code = ReadFile(shaderPath);
    ID3D11ComputeShader* cs = nullptr;
    if (FAILED(dev->CreateComputeShader(code.data(), code.size(), nullptr, &cs))) {
      std::printf("CreateComputeShader failed\n");
      return 1;
    }

    auto makeArraySrv = [&](UINT w, UINT h, DXGI_FORMAT fmt, uint32_t seed) -> ID3D11ShaderResourceView* {
      std::vector<uint16_t> data(size_t(w) * h);
      for (uint32_t y = 0; y < h; y++)
        for (uint32_t x = 0; x < w; x++)
          data[size_t(y) * w + x] = uint16_t(Field(x, y, seed) * 65535.0f);

      D3D11_TEXTURE2D_DESC desc = { w, h, 1, 1, fmt, { 1, 0 },
        D3D11_USAGE_IMMUTABLE, D3D11_BIND_SHADER_RESOURCE, 0, 0 };
      D3D11_SUBRESOURCE_DATA init = { data.data(), w * 2u, 0 };
      ID3D11Texture2D* tex = nullptr;
      dev->CreateTexture2D(&desc, &init, &tex);

      D3D11_SHADER_RESOURCE_VIEW_DESC svd = { };
      svd.Format = fmt;
      svd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
      svd.Texture2DArray.MipLevels = 1;
      svd.Texture2DArray.ArraySize = 1;
      ID3D11ShaderResourceView* srv = nullptr;
      dev->CreateShaderResourceView(tex, &svd, &srv);
      tex->Release();
      return srv;
    };

    ID3D11ShaderResourceView* t0 = makeArraySrv(4096u, 4096u, DXGI_FORMAT_R16_UNORM, 11u);
    ID3D11ShaderResourceView* t1 = makeArraySrv(512u, 512u, DXGI_FORMAT_R16_UNORM, 12u);

    ID3D11ShaderResourceView* t2 = nullptr;
    {
      const UINT n = 4096u;
      std::vector<float> data(n);
      for (UINT i = 0; i < n; i++)
        data[i] = Field(i, 0u, 13u);
      D3D11_TEXTURE1D_DESC desc = { n, 1, 1, DXGI_FORMAT_R32_FLOAT, D3D11_USAGE_IMMUTABLE,
        D3D11_BIND_SHADER_RESOURCE, 0, 0 };
      D3D11_SUBRESOURCE_DATA init = { data.data(), 0, 0 };
      ID3D11Texture1D* tex = nullptr;
      dev->CreateTexture1D(&desc, &init, &tex);
      dev->CreateShaderResourceView(tex, nullptr, &t2);
      tex->Release();
    }

    ID3D11ShaderResourceView* t3 = nullptr;
    {
      const UINT n = 32u;
      std::vector<uint8_t> data(size_t(n) * n * n);
      for (size_t i = 0; i < data.size(); i++)
        data[i] = uint8_t(Field(uint32_t(i % n), uint32_t((i / n) % n), 14u) * 255.0f);
      D3D11_TEXTURE3D_DESC desc = { n, n, n, 1, DXGI_FORMAT_R8_UINT, D3D11_USAGE_IMMUTABLE,
        D3D11_BIND_SHADER_RESOURCE, 0, 0 };
      D3D11_SUBRESOURCE_DATA init = { data.data(), n, n * n };
      ID3D11Texture3D* tex = nullptr;
      dev->CreateTexture3D(&desc, &init, &tex);
      dev->CreateShaderResourceView(tex, nullptr, &t3);
      tex->Release();
    }

    const UINT uw = 320u, uh = 192u, ud = 90u;
    D3D11_TEXTURE3D_DESC uDesc = { uw, uh, ud, 1, DXGI_FORMAT_R16_FLOAT, D3D11_USAGE_DEFAULT,
      D3D11_BIND_UNORDERED_ACCESS, 0, 0 };
    ID3D11Texture3D* uTex[2] = { };
    ID3D11UnorderedAccessView* uav[2] = { };
    for (int i = 0; i < 2; i++) {
      dev->CreateTexture3D(&uDesc, nullptr, &uTex[i]);
      dev->CreateUnorderedAccessView(uTex[i], nullptr, &uav[i]);
    }

    std::vector<float> cbData(24u * 4u);
    for (size_t i = 0; i < cbData.size(); i++)
      cbData[i] = 0.1f + 0.01f * float(i % 37u);
    D3D11_BUFFER_DESC cbDesc = { UINT(cbData.size() * 4u), D3D11_USAGE_IMMUTABLE, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0 };
    D3D11_SUBRESOURCE_DATA cbInit = { cbData.data(), 0, 0 };
    ID3D11Buffer* cb = nullptr;
    dev->CreateBuffer(&cbDesc, &cbInit, &cb);

    D3D11_SAMPLER_DESC sd = { };
    sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.MaxLOD = D3D11_FLOAT32_MAX;
    ID3D11SamplerState* samp[4] = { };
    sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    dev->CreateSamplerState(&sd, &samp[0]);
    dev->CreateSamplerState(&sd, &samp[1]);
    sd.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
    dev->CreateSamplerState(&sd, &samp[2]);
    dev->CreateSamplerState(&sd, &samp[3]);

    ID3D11ShaderResourceView* srvs[4] = { t0, t1, t2, t3 };
    ctx->CSSetShader(cs, nullptr, 0);
    ctx->CSSetShaderResources(0, 4, srvs);
    ctx->CSSetUnorderedAccessViews(0, 2, uav, nullptr);
    ctx->CSSetConstantBuffers(0, 1, &cb);
    ctx->CSSetSamplers(0, 4, samp);

    D3D11_QUERY_DESC qd = { D3D11_QUERY_TIMESTAMP_DISJOINT, 0 };
    ID3D11Query* disjoint = nullptr;
    ID3D11Query* ts[2] = { };
    dev->CreateQuery(&qd, &disjoint);
    qd.Query = D3D11_QUERY_TIMESTAMP;
    dev->CreateQuery(&qd, &ts[0]);
    dev->CreateQuery(&qd, &ts[1]);

    const uint32_t reps = 50u;
    double best = 1e30;

    for (uint32_t round = 0; round < 5u; round++) {
      ctx->Begin(disjoint);
      ctx->End(ts[0]);
      for (uint32_t i = 0; i < reps; i++)
        ctx->Dispatch(gx, gy, gz);
      ctx->End(ts[1]);
      ctx->End(disjoint);

      D3D11_QUERY_DATA_TIMESTAMP_DISJOINT dj = { };
      UINT64 t0 = 0, t1 = 0;
      while (ctx->GetData(disjoint, &dj, sizeof(dj), 0) != S_OK) { }
      while (ctx->GetData(ts[0], &t0, sizeof(t0), 0) != S_OK) { }
      while (ctx->GetData(ts[1], &t1, sizeof(t1), 0) != S_OK) { }

      if (!dj.Disjoint && dj.Frequency) {
        double ms = double(t1 - t0) * 1000.0 / double(dj.Frequency) / reps;
        if (ms < best)
          best = ms;
      }
    }

    std::printf("%s: dispatch(%u,%u,%u): %.4f ms per dispatch (best of 5 x %u)\n",
      shaderPath, gx, gy, gz, best, reps);

    for (int i = 0; i < 2; i++) {
      D3D11_TEXTURE3D_DESC sdesc = uDesc;
      sdesc.Usage = D3D11_USAGE_STAGING;
      sdesc.BindFlags = 0;
      sdesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      ID3D11Texture3D* staging = nullptr;
      dev->CreateTexture3D(&sdesc, nullptr, &staging);
      ctx->CopyResource(staging, uTex[i]);
      D3D11_MAPPED_SUBRESOURCE mapped = { };
      ctx->Map(staging, 0, D3D11_MAP_READ, 0, &mapped);
      std::ofstream f(std::string(outPath) + (i ? ".u1" : ".u0"), std::ios::binary);
      for (UINT z = 0; z < ud; z++)
        for (UINT y = 0; y < uh; y++)
          f.write(reinterpret_cast<const char*>(mapped.pData) + size_t(z) * mapped.DepthPitch + size_t(y) * mapped.RowPitch, uw * 2u);
      ctx->Unmap(staging, 0);
    }
    return 0;
  }

  // "create FILE...": creates each shader (stage from the dxbc) and exits;
  // with DXVK_SHADER_DUMP_PATH set, dxvk dumps what it made of them
  int RunCreate(int count, char** paths) {
    ID3D11Device* dev = nullptr;
    ID3D11DeviceContext* ctx = nullptr;
    if (!CreateDevice(&dev, &ctx)) {
      std::printf("no device\n");
      return 1;
    }

    for (int i = 0; i < count; i++) {
      std::vector<uint8_t> code = ReadFile(paths[i]);
      uint32_t type = 0xffffu;
      uint32_t chunks = 0u;
      std::memcpy(&chunks, code.data() + 28, 4);
      for (uint32_t c = 0; c < chunks; c++) {
        uint32_t off = 0u;
        std::memcpy(&off, code.data() + 32 + 4 * c, 4);
        if (!std::memcmp(code.data() + off, "SHEX", 4) || !std::memcmp(code.data() + off, "SHDR", 4)) {
          std::memcpy(&type, code.data() + off + 8, 4);
          type >>= 16;
        }
      }

      HRESULT hr = E_FAIL;
      if (type == 0u) {
        ID3D11PixelShader* ps = nullptr;
        hr = dev->CreatePixelShader(code.data(), code.size(), nullptr, &ps);
      } else if (type == 1u) {
        ID3D11VertexShader* vs = nullptr;
        hr = dev->CreateVertexShader(code.data(), code.size(), nullptr, &vs);
      } else if (type == 5u) {
        ID3D11ComputeShader* cs = nullptr;
        hr = dev->CreateComputeShader(code.data(), code.size(), nullptr, &cs);
      }
      std::printf("%s: type %u, hr 0x%08x\n", paths[i], type, unsigned(hr));
    }

    return 0;
  }

  const char* g_vs =
    "float4 main(uint id : SV_VertexID) : SV_Position {\n"
    "  float2 p = float2((id << 1) & 2, id & 2);\n"
    "  return float4(p * float2(2, -2) + float2(-1, 1), 0, 1);\n"
    "}\n";

  // BLESSED_TEST_PS_VARIANT picks which source this run compiles for the
  // replacement file; the vanilla one is variant 0
  const char* g_ps[3] = {
    "float4 main(float4 p : SV_Position) : SV_Target0 { return float4(frac(p.xy / 64), 0.25, 1); }\n",
    "float4 main(float4 p : SV_Position) : SV_Target0 { float2 q = p.xy / 64; return float4(q - floor(q), 0.25, 1); }\n",
    "float4 main(float4 p : SV_Position) : SV_Target0 { return float4(frac(p.xy / 64), 0.25 + (p.x > 1000 ? 0.01 : 0), 1); }\n",
  };

  ID3DBlob* Compile(const char* src, const char* profile) {
    ID3DBlob* blob = nullptr;
    ID3DBlob* err = nullptr;
    if (FAILED(D3DCompile(src, std::strlen(src), nullptr, nullptr, nullptr, "main", profile, 0, 0, &blob, &err))) {
      std::printf("compile failed: %s\n", err ? (const char*) err->GetBufferPointer() : "?");
      return nullptr;
    }
    return blob;
  }

  // "write DIR VARIANT": writes the vanilla ps's replacement file into DIR
  int WriteReplacement(const char* dir, int variant) {
    ID3DBlob* vanilla = Compile(g_ps[0], "ps_5_0");
    ID3DBlob* repl = Compile(g_ps[variant], "ps_5_0");
    const uint8_t* hash = reinterpret_cast<const uint8_t*>(vanilla->GetBufferPointer()) + 4;
    char name[512];
    int n = std::snprintf(name, sizeof(name), "%s/fs.", dir);
    for (int i = 0; i < 16; i++)
      n += std::snprintf(name + n, sizeof(name) - n, "%02x", hash[i]);
    std::snprintf(name + n, sizeof(name) - n, ".dxbc");
    std::ofstream f(name, std::ios::binary);
    f.write(reinterpret_cast<const char*>(repl->GetBufferPointer()), repl->GetBufferSize());
    std::printf("wrote %s (variant %d)\n", name, variant);
    return 0;
  }

  int RunPs() {
    ID3D11Device* dev = nullptr;
    ID3D11DeviceContext* ctx = nullptr;
    if (!CreateDevice(&dev, &ctx)) {
      std::printf("no device\n");
      return 1;
    }

    ID3DBlob* vsBlob = Compile(g_vs, "vs_5_0");
    ID3DBlob* psBlob = Compile(g_ps[0], "ps_5_0");
    ID3D11VertexShader* vs = nullptr;
    ID3D11PixelShader* ps = nullptr;
    dev->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), nullptr, &vs);
    dev->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(), nullptr, &ps);

    D3D11_TEXTURE2D_DESC desc = { 1920, 1080, 1, 1, DXGI_FORMAT_R16G16B16A16_FLOAT, { 1, 0 },
      D3D11_USAGE_DEFAULT, D3D11_BIND_RENDER_TARGET, 0, 0 };
    ID3D11Texture2D* rt = nullptr;
    ID3D11RenderTargetView* rtv = nullptr;
    dev->CreateTexture2D(&desc, nullptr, &rt);
    dev->CreateRenderTargetView(rt, nullptr, &rtv);

    D3D11_VIEWPORT vp = { 0, 0, 1920, 1080, 0, 1 };
    ctx->RSSetViewports(1, &vp);
    ctx->OMSetRenderTargets(1, &rtv, nullptr);
    ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ctx->VSSetShader(vs, nullptr, 0);
    ctx->PSSetShader(ps, nullptr, 0);

    for (int i = 0; i < 3; i++)
      ctx->Draw(3, 0);

    ctx->Flush();
    std::printf("drew 3 fullscreen triangles\n");
    return 0;
  }

}

int main(int argc, char** argv) {
  if (argc >= 7 && !std::strcmp(argv[1], "cs"))
    return RunCs(argv[2], argv[3], UINT(atoi(argv[4])), UINT(atoi(argv[5])), UINT(atoi(argv[6])));
  if (argc >= 6 && !std::strcmp(argv[1], "chain"))
    return RunChain(argv[2], UINT(atoi(argv[3])), UINT(atoi(argv[4])), UINT(atoi(argv[5])));
  if (argc >= 7 && !std::strcmp(argv[1], "collapsecs"))
    return RunCollapseCs(argv[2], UINT(atoi(argv[3])), UINT(atoi(argv[4])), UINT(atoi(argv[5])), argv[6],
      argc >= 9 ? UINT(atoi(argv[7])) : 8u, argc >= 9 ? UINT(atoi(argv[8])) : 8u);
  if (argc >= 9 && !std::strcmp(argv[1], "chain2"))
    return RunChain2(argv[2], argv[3], UINT(atoi(argv[4])), UINT(atoi(argv[5])), UINT(atoi(argv[6])),
      UINT(atoi(argv[7])), argv[8], argc >= 10 && !std::strcmp(argv[9], "point"));
  if (argc >= 3 && !std::strcmp(argv[1], "create"))
    return RunCreate(argc - 2, argv + 2);
  if (argc >= 2 && !std::strcmp(argv[1], "ps"))
    return RunPs();
  if (argc >= 4 && !std::strcmp(argv[1], "write"))
    return WriteReplacement(argv[2], atoi(argv[3]));
  if (argc >= 7 && !std::strcmp(argv[1], "volgen"))
    return RunVolgen(argv[2], UINT(atoi(argv[3])), UINT(atoi(argv[4])), UINT(atoi(argv[5])), argv[6]);
  std::printf("usage: harness cs SHADER OUT GX GY GZ | ps | write DIR VARIANT | volgen SHADER GX GY GZ OUT\n");
  return 1;
}
