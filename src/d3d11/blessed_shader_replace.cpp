// blessed: shader-replace, the replacement table (create time) and the twin-draw verifier (draw time)
#include "blessed_shader_replace.h"
#include "blessed_shader_verify_dxbc.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <dxbc/dxbc_container.h>

#include "d3d11_context_imm.h"
#include "d3d11_device.h"
#include "d3d11_shader.h"

#include "../util/util_env.h"
#include "../util/util_string.h"
#include "../util/log/log.h"

namespace dxvk {

  namespace {

    constexpr uint32_t MaxTargets = 8u;
    constexpr uint32_t RecordSize = 32u;

    struct Entry {
      std::string           name;
      std::vector<uint8_t>  code;
      bool                  validated = false;
      bool                  refused   = false;
      bool                  check     = false;

      // verify twins: the vanilla shader and the replacement, each under a
      // salted key so no blessed hook matches a twin draw
      bool                  twinsMade = false;
      Com<ID3D11PixelShader>    psVanilla;
      Com<ID3D11PixelShader>    psReplace;
      Com<ID3D11ComputeShader>  csVanilla;
      Com<ID3D11ComputeShader>  csReplace;

      uint32_t              checksDone = 0u;
      bool                  skipLogged = false;
    };

    struct Scratch {
      Com<ID3D11Resource>             resource;
      Com<ID3D11RenderTargetView>     rtv;
      Com<ID3D11UnorderedAccessView>  uav;
      Com<ID3D11ShaderResourceView>   srv;
    };

    struct State {
      std::recursive_mutex  mutex;
      bool                  loaded = false;
      std::string           dir;

      std::unordered_map<std::string, std::unique_ptr<Entry>> table;
      std::unordered_set<std::string> twinNames;

      // verify
      std::vector<std::string> checkTokens;
      uint32_t              maxChecks = 8u;
      bool                  inVerify  = false;
      std::unordered_map<const void*, Entry*> shaderMemo;
      std::unordered_map<std::string, Scratch> scratch;

      Com<ID3D11ComputeShader>  diff2d;
      Com<ID3D11ComputeShader>  diff3d;
      Com<ID3D11Buffer>         results;
      Com<ID3D11UnorderedAccessView> resultsUav;
      Com<ID3D11Buffer>         staging;
      Com<ID3D11Buffer>         params;
      bool                      diffFailed = false;

      std::ofstream         file;
      bool                  fileTried = false;
    };

    State& GetState() {
      static State s;
      return s;
    }

    bool ComputeReplaceEnabled() {
      return !env::getEnvVar("BLESSED_SHADER_REPLACE").empty();
    }

    bool ComputeVerifyEnabled() {
      return ComputeReplaceEnabled()
          && env::getEnvVar("BLESSED_SHADER_VERIFY") == "1";
    }

    bool IsHexName(const std::string& s) {
      // "<stage>.<32 hex>"
      size_t dot = s.find('.');

      if (dot == std::string::npos || s.size() - dot - 1u != 32u)
        return false;

      for (size_t i = dot + 1u; i < s.size(); i++) {
        char c = s[i];
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
          return false;
      }

      return true;
    }

    std::vector<std::string> ParseList(const std::string& csv) {
      std::vector<std::string> out;
      std::stringstream ss(csv);
      std::string tok;

      while (std::getline(ss, tok, ',')) {
        if (!tok.empty())
          out.push_back(tok);
      }

      return out;
    }

    bool MatchesCheckList(const std::string& name) {
      auto& s = GetState();

      if (s.checkTokens.empty())
        return true;

      size_t dot = name.find('.');
      std::string hex = dot == std::string::npos ? name : name.substr(dot + 1u);

      for (const auto& tok : s.checkTokens) {
        if (tok == name)
          return true;
        if (tok.size() >= 8u && hex.compare(0u, tok.size(), tok) == 0)
          return true;
      }

      return false;
    }

    void LoadTable() {
      auto& s = GetState();

      if (s.loaded)
        return;

      s.loaded = true;
      s.dir = env::getEnvVar("BLESSED_SHADER_REPLACE");

      if (ComputeVerifyEnabled()) {
        s.checkTokens = ParseList(env::getEnvVar("BLESSED_SHADER_VERIFY_LIST"));

        std::string max = env::getEnvVar("BLESSED_SHADER_VERIFY_MAX");
        if (!max.empty())
          s.maxChecks = uint32_t(std::strtoul(max.c_str(), nullptr, 10));
      }

      std::error_code ec;
      std::filesystem::path dir = str::topath(s.dir.c_str());

      for (auto it = std::filesystem::directory_iterator(dir, ec);
           !ec && it != std::filesystem::directory_iterator(); it.increment(ec)) {
        if (!it->is_regular_file(ec) || it->path().extension() != ".dxbc")
          continue;

        std::string name = it->path().stem().string();

        if (!IsHexName(name))
          continue;

        std::ifstream f(it->path(), std::ios::binary);
        std::vector<uint8_t> code((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

        if (code.size() < 32u) {
          Logger::warn(str::format("BlessedShaderReplace: ", name, ": file too small, ignored"));
          continue;
        }

        auto e = std::make_unique<Entry>();
        e->name  = name;
        e->code  = std::move(code);
        e->check = ComputeVerifyEnabled() && MatchesCheckList(name);
        s.table.emplace(name, std::move(e));
      }

      if (ec)
        Logger::warn(str::format("BlessedShaderReplace: cannot list ", s.dir, ": ", ec.message()));

      Logger::info(str::format("BlessedShaderReplace: ", s.table.size(),
        " replacement(s) in ", s.dir, ComputeVerifyEnabled() ? " (verify on)" : ""));
    }

    // dxbc program type from the SHEX/SHDR version token:
    // 0 ps, 1 vs, 2 gs, 3 hs, 4 ds, 5 cs; ~0u if not found
    uint32_t ProgramType(const uint8_t* code, size_t size) {
      if (size < 32u || std::memcmp(code, "DXBC", 4u))
        return ~0u;

      uint32_t count = 0u;
      std::memcpy(&count, code + 28u, 4u);

      for (uint32_t i = 0u; i < count && 32u + 4u * i + 4u <= size; i++) {
        uint32_t offset = 0u;
        std::memcpy(&offset, code + 32u + 4u * i, 4u);

        if (size_t(offset) + 12u > size)
          continue;

        if (!std::memcmp(code + offset, "SHEX", 4u) || !std::memcmp(code + offset, "SHDR", 4u)) {
          uint32_t version = 0u;
          std::memcpy(&version, code + offset + 8u, 4u);
          return version >> 16u;
        }
      }

      return ~0u;
    }

    uint32_t StageProgramType(VkShaderStageFlagBits stage) {
      switch (stage) {
        case VK_SHADER_STAGE_FRAGMENT_BIT:                return 0u;
        case VK_SHADER_STAGE_VERTEX_BIT:                  return 1u;
        case VK_SHADER_STAGE_GEOMETRY_BIT:                return 2u;
        case VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT:    return 3u;
        case VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT: return 4u;
        case VK_SHADER_STAGE_COMPUTE_BIT:                 return 5u;
        default:                                          return ~0u;
      }
    }

    bool Validate(Entry& e, VkShaderStageFlagBits stage) {
      dxbc_spv::dxbc::Container container(e.code.data(), e.code.size());

      if (!container.validateHash()) {
        Logger::warn(str::format("BlessedShaderReplace: ", e.name, ": dxbc checksum invalid, vanilla runs"));
        return false;
      }

      if (ProgramType(e.code.data(), e.code.size()) != StageProgramType(stage)) {
        Logger::warn(str::format("BlessedShaderReplace: ", e.name, ": program type does not match the stage, vanilla runs"));
        return false;
      }

      return true;
    }

    DxvkShaderHash SaltedKey(VkShaderStageFlagBits stage, const void* code, size_t size, uint8_t salt) {
      uint8_t hash[16];
      std::memcpy(hash, reinterpret_cast<const uint8_t*>(code) + 4u, 16u);
      hash[15] ^= salt;
      return DxvkShaderHash(stage, uint32_t(size), hash, sizeof(hash));
    }

  }


  namespace blessed_shader_replace_detail {
    extern const bool g_replaceEnabled = ComputeReplaceEnabled();
    extern const bool g_verifyEnabled  = ComputeVerifyEnabled();
  }


  bool BlessedShaderReplace::Apply(
          D3D11Device*      device,
    const DxvkShaderHash&   key,
    const void**            ppCode,
          size_t*           pSize,
    const CreateModuleFn&   create) {
    auto& s = GetState();
    std::lock_guard<std::recursive_mutex> lock(s.mutex);
    LoadTable();

    std::string name = key.toString();
    auto entry = s.table.find(name);

    if (entry == s.table.end())
      return false;

    Entry& e = *entry->second;

    if (!e.validated) {
      e.validated = true;
      e.refused = !Validate(e, key.stage());

      if (!e.refused) {
        Logger::info(str::format("BlessedShaderReplace: ", name, " replaced (vanilla ",
          *pSize, " bytes, replacement ", e.code.size(), " bytes)"));
      }
    }

    if (e.refused)
      return false;

    if (e.check && !e.twinsMade) {
      e.twinsMade = true;

      DxvkShaderHash vanillaKey = SaltedKey(key.stage(), *ppCode, *pSize, 0x5au);
      DxvkShaderHash replaceKey = SaltedKey(key.stage(), e.code.data(), e.code.size(), 0xa5u);
      s.twinNames.insert(vanillaKey.toString());
      s.twinNames.insert(replaceKey.toString());

      D3D11CommonShader vanilla, replace;
      bool ok = create(vanillaKey, *ppCode, *pSize, &vanilla)
             && create(replaceKey, e.code.data(), e.code.size(), &replace);

      if (!ok) {
        Logger::warn(str::format("BlessedShaderVerify: ", name, ": twin creation failed, not checked"));
      } else if (key.stage() == VK_SHADER_STAGE_FRAGMENT_BIT) {
        e.psVanilla = new D3D11PixelShader(device, vanilla);
        e.psReplace = new D3D11PixelShader(device, replace);
      } else if (key.stage() == VK_SHADER_STAGE_COMPUTE_BIT) {
        e.csVanilla = new D3D11ComputeShader(device, vanilla);
        e.csReplace = new D3D11ComputeShader(device, replace);
      } else {
        Logger::warn(str::format("BlessedShaderVerify: ", name, ": only pixel and compute shaders are checked"));
      }
    }

    *ppCode = e.code.data();
    *pSize  = e.code.size();
    return true;
  }


  bool BlessedShaderReplace::SkipDiskCache(const DxvkShaderHash& key) {
    if (!IsEnabled())
      return false;

    auto& s = GetState();
    std::lock_guard<std::recursive_mutex> lock(s.mutex);
    LoadTable();

    std::string name = key.toString();
    auto entry = s.table.find(name);

    if (entry != s.table.end() && !entry->second->refused)
      return true;

    return s.twinNames.find(name) != s.twinNames.end();
  }


  // ---------------------------------------------------------------------
  // verifier

  namespace {

    bool FloatReadable(DXGI_FORMAT fmt) {
      switch (fmt) {
        case DXGI_FORMAT_R32G32B32A32_FLOAT:
        case DXGI_FORMAT_R16G16B16A16_FLOAT:
        case DXGI_FORMAT_R16G16B16A16_UNORM:
        case DXGI_FORMAT_R16G16B16A16_SNORM:
        case DXGI_FORMAT_R32G32_FLOAT:
        case DXGI_FORMAT_R10G10B10A2_UNORM:
        case DXGI_FORMAT_R11G11B10_FLOAT:
        case DXGI_FORMAT_R8G8B8A8_UNORM:
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
        case DXGI_FORMAT_R8G8B8A8_SNORM:
        case DXGI_FORMAT_B8G8R8A8_UNORM:
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
        case DXGI_FORMAT_B8G8R8X8_UNORM:
        case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
        case DXGI_FORMAT_R16G16_FLOAT:
        case DXGI_FORMAT_R16G16_UNORM:
        case DXGI_FORMAT_R16G16_SNORM:
        case DXGI_FORMAT_R32_FLOAT:
        case DXGI_FORMAT_R8G8_UNORM:
        case DXGI_FORMAT_R8G8_SNORM:
        case DXGI_FORMAT_R16_FLOAT:
        case DXGI_FORMAT_R16_UNORM:
        case DXGI_FORMAT_R16_SNORM:
        case DXGI_FORMAT_R8_UNORM:
        case DXGI_FORMAT_R8_SNORM:
        case DXGI_FORMAT_A8_UNORM:
          return true;
        default:
          return false;
      }
    }

    struct Target {
      uint32_t    slot = 0u;
      DXGI_FORMAT fmt  = DXGI_FORMAT_UNKNOWN;
      uint32_t    w = 0u, h = 0u, d = 1u;
      bool        is3d = false;
      Scratch*    a = nullptr;
      Scratch*    b = nullptr;
    };

    Entry* FindEntry(const D3D11CommonShader* shader) {
      auto& s = GetState();
      const void* key = shader->GetShader().ptr();

      auto memo = s.shaderMemo.find(key);
      if (memo != s.shaderMemo.end())
        return memo->second;

      Entry* result = nullptr;
      auto entry = s.table.find(shader->GetName());

      if (entry != s.table.end() && !entry->second->refused)
        result = entry->second.get();

      s.shaderMemo.emplace(key, result);
      return result;
    }

    void LogSkip(Entry& e, const std::string& why) {
      if (e.skipLogged)
        return;

      e.skipLogged = true;
      Logger::warn(str::format("BlessedShaderVerify: ", e.name, ": not checked (", why, ")"));
    }

    Scratch* GetScratch(ID3D11Device* dev, const Target& t, char set, bool rtv) {
      auto& s = GetState();
      std::string key = str::format(set, rtv ? "r" : "u", t.is3d ? "3" : "2", ".",
        uint32_t(t.fmt), ".", t.w, "x", t.h, "x", t.d, ".", t.slot);

      auto found = s.scratch.find(key);
      if (found != s.scratch.end())
        return &found->second;

      Scratch sc;
      UINT bind = D3D11_BIND_SHADER_RESOURCE | (rtv ? D3D11_BIND_RENDER_TARGET : D3D11_BIND_UNORDERED_ACCESS);

      if (t.is3d) {
        D3D11_TEXTURE3D_DESC desc = { };
        desc.Width     = t.w;
        desc.Height    = t.h;
        desc.Depth     = t.d;
        desc.MipLevels = 1u;
        desc.Format    = t.fmt;
        desc.Usage     = D3D11_USAGE_DEFAULT;
        desc.BindFlags = bind;

        Com<ID3D11Texture3D> tex;
        if (FAILED(dev->CreateTexture3D(&desc, nullptr, &tex)))
          return nullptr;
        sc.resource = tex.ptr();
      } else {
        D3D11_TEXTURE2D_DESC desc = { };
        desc.Width      = t.w;
        desc.Height     = t.h;
        desc.MipLevels  = 1u;
        desc.ArraySize  = 1u;
        desc.Format     = t.fmt;
        desc.SampleDesc = { 1u, 0u };
        desc.Usage      = D3D11_USAGE_DEFAULT;
        desc.BindFlags  = bind;

        Com<ID3D11Texture2D> tex;
        if (FAILED(dev->CreateTexture2D(&desc, nullptr, &tex)))
          return nullptr;
        sc.resource = tex.ptr();
      }

      if (FAILED(dev->CreateShaderResourceView(sc.resource.ptr(), nullptr, &sc.srv)))
        return nullptr;

      if (rtv && FAILED(dev->CreateRenderTargetView(sc.resource.ptr(), nullptr, &sc.rtv)))
        return nullptr;

      if (!rtv && FAILED(dev->CreateUnorderedAccessView(sc.resource.ptr(), nullptr, &sc.uav)))
        return nullptr;

      return &s.scratch.emplace(key, std::move(sc)).first->second;
    }

    bool EnsureDiff(ID3D11Device* dev) {
      auto& s = GetState();

      if (s.diffFailed)
        return false;

      if (s.diff2d != nullptr)
        return true;

      bool ok = SUCCEEDED(dev->CreateComputeShader(blessed_shader_verify_2d, sizeof(blessed_shader_verify_2d), nullptr, &s.diff2d))
             && SUCCEEDED(dev->CreateComputeShader(blessed_shader_verify_3d, sizeof(blessed_shader_verify_3d), nullptr, &s.diff3d));

      D3D11_BUFFER_DESC desc = { };
      desc.ByteWidth = MaxTargets * RecordSize;
      desc.Usage     = D3D11_USAGE_DEFAULT;
      desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
      desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
      ok = ok && SUCCEEDED(dev->CreateBuffer(&desc, nullptr, &s.results));

      D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc = { };
      uavDesc.Format              = DXGI_FORMAT_R32_TYPELESS;
      uavDesc.ViewDimension       = D3D11_UAV_DIMENSION_BUFFER;
      uavDesc.Buffer.NumElements  = MaxTargets * RecordSize / 4u;
      uavDesc.Buffer.Flags        = D3D11_BUFFER_UAV_FLAG_RAW;
      ok = ok && SUCCEEDED(dev->CreateUnorderedAccessView(s.results.ptr(), &uavDesc, &s.resultsUav));

      desc.Usage          = D3D11_USAGE_STAGING;
      desc.BindFlags      = 0u;
      desc.MiscFlags      = 0u;
      desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      ok = ok && SUCCEEDED(dev->CreateBuffer(&desc, nullptr, &s.staging));

      desc.ByteWidth      = 16u;
      desc.Usage          = D3D11_USAGE_DEFAULT;
      desc.BindFlags      = D3D11_BIND_CONSTANT_BUFFER;
      desc.CPUAccessFlags = 0u;
      ok = ok && SUCCEEDED(dev->CreateBuffer(&desc, nullptr, &s.params));

      if (!ok) {
        s.diffFailed = true;
        Logger::err("BlessedShaderVerify: cannot create the diff resources, verify off");
      }

      return ok;
    }

    std::ofstream* GetFile() {
      auto& s = GetState();

      if (!s.fileTried) {
        s.fileTried = true;
        std::string dir = env::getEnvVar("BLESSED_PROBE_DIR");

        if (dir.empty()) {
          Logger::warn("BlessedShaderVerify: BLESSED_PROBE_DIR is not set; results go to the dxvk log only");
        } else {
          std::error_code ec;
          std::filesystem::create_directories(str::topath(dir.c_str()), ec);
          s.file.open(str::topath((dir + "/shader-verify.jsonl").c_str()), std::ios::app);
        }
      }

      return s.file.is_open() ? &s.file : nullptr;
    }

    // runs the diff kernel per target, reads the records back (a blocking
    // map: this is a debug mode) and logs them
    void DiffAndLog(ID3D11DeviceContext* ctx, ID3D11Device* dev, const std::string& name,
        uint32_t check, uint32_t checks, const char* kind,
        const std::vector<Target>& targets, const std::string& skipped) {
      auto& s = GetState();

      if (!EnsureDiff(dev))
        return;

      // save the compute state the diff pass touches
      Com<ID3D11ComputeShader>        oldCs;
      Com<ID3D11ShaderResourceView>   oldSrv[2];
      Com<ID3D11UnorderedAccessView>  oldUav[MaxTargets];
      Com<ID3D11Buffer>               oldCb;
      ctx->CSGetShader(&oldCs, nullptr, nullptr);
      ctx->CSGetShaderResources(0u, 2u, reinterpret_cast<ID3D11ShaderResourceView**>(oldSrv));
      ctx->CSGetUnorderedAccessViews(0u, MaxTargets, reinterpret_cast<ID3D11UnorderedAccessView**>(oldUav));
      ctx->CSGetConstantBuffers(0u, 1u, &oldCb);

      // the compared textures may still be bound as compute uavs (the last
      // dispatch of a chain wrote them): unbind every uav slot before the
      // srvs go in, or the srv binds are dropped as hazards
      ID3D11UnorderedAccessView* diffUavs[MaxTargets] = { s.resultsUav.ptr() };
      UINT keepCounts[MaxTargets];
      for (auto& c : keepCounts)
        c = ~0u;
      ctx->CSSetUnorderedAccessViews(0u, MaxTargets, diffUavs, keepCounts);

      UINT zero[4] = { 0u, 0u, 0u, 0u };
      ctx->ClearUnorderedAccessViewUint(s.resultsUav.ptr(), zero);

      for (uint32_t i = 0u; i < targets.size(); i++) {
        const Target& t = targets[i];
        uint32_t params[4] = { i, 0u, 0u, 0u };
        ctx->UpdateSubresource(s.params.ptr(), 0u, nullptr, params, 0u, 0u);

        ID3D11ShaderResourceView* srvs[2] = { t.a->srv.ptr(), t.b->srv.ptr() };
        ID3D11UnorderedAccessView* uav = s.resultsUav.ptr();
        ID3D11Buffer* cb = s.params.ptr();
        ctx->CSSetShader(t.is3d ? s.diff3d.ptr() : s.diff2d.ptr(), nullptr, 0u);
        ctx->CSSetUnorderedAccessViews(0u, 1u, &uav, nullptr);
        ctx->CSSetShaderResources(0u, 2u, srvs);
        ctx->CSSetConstantBuffers(0u, 1u, &cb);
        ctx->Dispatch((t.w + 7u) / 8u, (t.h + 7u) / 8u, t.d);
      }

      ID3D11ShaderResourceView* nullSrv[2] = { nullptr, nullptr };
      ctx->CSSetShaderResources(0u, 2u, nullSrv);

      ID3D11ShaderResourceView* restoreSrv[2] = { oldSrv[0].ptr(), oldSrv[1].ptr() };
      ID3D11Buffer* restoreCb = oldCb.ptr();
      ctx->CSSetShader(oldCs.ptr(), nullptr, 0u);
      ctx->CSSetUnorderedAccessViews(0u, MaxTargets, reinterpret_cast<ID3D11UnorderedAccessView**>(oldUav), keepCounts);
      ctx->CSSetShaderResources(0u, 2u, restoreSrv);
      ctx->CSSetConstantBuffers(0u, 1u, &restoreCb);

      ctx->CopyResource(s.staging.ptr(), s.results.ptr());

      D3D11_MAPPED_SUBRESOURCE mapped = { };
      if (FAILED(ctx->Map(s.staging.ptr(), 0u, D3D11_MAP_READ, 0u, &mapped))) {
        Logger::warn(str::format("BlessedShaderVerify: ", name, ": readback failed"));
        return;
      }

      uint32_t rec[MaxTargets][8];
      std::memcpy(rec, mapped.pData, sizeof(rec));
      ctx->Unmap(s.staging.ptr(), 0u);

      std::stringstream line;
      line << "{\"shader\":\"" << name << "\",\"kind\":\"" << kind
           << "\",\"check\":" << check << ",\"targets\":[";

      float worst = 0.0f;
      uint64_t differing = 0u;

      for (uint32_t i = 0u; i < targets.size(); i++) {
        const Target& t = targets[i];
        const uint32_t* r = rec[i];

        float maxDiff;
        std::memcpy(&maxDiff, &r[0], 4u);
        uint64_t sum = (uint64_t(r[4]) << 32u) | r[3];
        double mean = r[2] ? double(sum) / 1048576.0 / double(r[2]) : 0.0;

        if (r[1]) {
          worst = std::max(worst, maxDiff);
          differing += r[1];
        }

        std::string first = "null";
        if (r[5]) {
          uint32_t index = ~r[5];
          uint32_t x = index % t.w;
          uint32_t y = (index / t.w) % t.h;
          uint32_t z = index / (t.w * t.h);
          first = str::format("[", x, ",", y, ",", z, "]");
        }

        line << (i ? "," : "") << "{\"slot\":" << t.slot << ",\"fmt\":" << uint32_t(t.fmt)
             << ",\"w\":" << t.w << ",\"h\":" << t.h << ",\"d\":" << t.d
             << ",\"touched\":" << r[2] << ",\"differ\":" << r[1] << ",\"bad\":" << r[6]
             << ",\"max\":" << (r[1] ? maxDiff : 0.0f) << ",\"mean\":" << mean
             << ",\"first\":" << first << "}";
      }

      line << "]" << skipped << "}";

      if (auto* file = GetFile())
        *file << line.str() << std::endl;

      Logger::info(str::format("BlessedShaderVerify: ", name, " ", kind, " check ",
        check, "/", checks, ": ", targets.size(), " target(s), ",
        differing ? str::format(differing, " pixel(s) differ, max ", worst)
                  : std::string("bit-identical")));
    }

    std::string SkippedJson(const std::vector<std::pair<uint32_t, std::string>>& skipped) {
      if (skipped.empty())
        return std::string();

      std::stringstream ss;
      ss << ",\"skipped\":[";

      for (size_t i = 0u; i < skipped.size(); i++)
        ss << (i ? "," : "") << "{\"slot\":" << skipped[i].first << ",\"why\":\"" << skipped[i].second << "\"}";

      ss << "]";
      return ss.str();
    }

  }


  void BlessedShaderVerify::OnDraw(
          D3D11ImmediateContext*  ctx,
    const D3D11ContextState&      state,
    const ReplayFn&               replay) {
    auto& s = GetState();
    std::lock_guard<std::recursive_mutex> lock(s.mutex);

    if (s.inVerify || state.ps == nullptr)
      return;

    Entry* e = FindEntry(state.ps->GetCommonShader());

    if (!e || !e->check || e->checksDone >= s.maxChecks)
      return;

    if (e->psVanilla == nullptr || e->psReplace == nullptr)
      return LogSkip(*e, "no twins");

    for (uint32_t i = state.om.minUav; i < state.om.maxUav; i++) {
      if (state.om.uavs[i] != nullptr)
        return LogSkip(*e, "uavs bound on the output merger");
    }

    Com<ID3D11Device> dev;
    ctx->GetDevice(&dev);

    // the bound targets, as the twins' scratch sets
    Com<ID3D11RenderTargetView> oldRtv[MaxTargets];
    Com<ID3D11DepthStencilView> oldDsv;
    ctx->OMGetRenderTargets(MaxTargets, reinterpret_cast<ID3D11RenderTargetView**>(oldRtv), &oldDsv);

    std::vector<Target> targets;
    std::vector<std::pair<uint32_t, std::string>> skipped;
    ID3D11RenderTargetView* setA[MaxTargets] = { };
    ID3D11RenderTargetView* setB[MaxTargets] = { };
    uint32_t count = 0u;

    for (uint32_t i = 0u; i < MaxTargets; i++) {
      if (oldRtv[i] == nullptr)
        continue;

      count = i + 1u;

      D3D11_RENDER_TARGET_VIEW_DESC desc;
      oldRtv[i]->GetDesc(&desc);

      if (desc.ViewDimension != D3D11_RTV_DIMENSION_TEXTURE2D) {
        skipped.push_back({ i, "not a plain 2d target" });
        continue;
      }

      if (!FloatReadable(desc.Format)) {
        skipped.push_back({ i, str::format("format ", uint32_t(desc.Format)) });
        continue;
      }

      Com<ID3D11Resource> res;
      oldRtv[i]->GetResource(&res);
      Com<ID3D11Texture2D> tex;

      if (FAILED(res->QueryInterface(__uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&tex)))) {
        skipped.push_back({ i, "not a texture2d" });
        continue;
      }

      D3D11_TEXTURE2D_DESC texDesc;
      tex->GetDesc(&texDesc);

      if (texDesc.SampleDesc.Count > 1u) {
        skipped.push_back({ i, "msaa" });
        continue;
      }

      Target t;
      t.slot = i;
      t.fmt  = desc.Format;
      t.w    = std::max(1u, texDesc.Width  >> desc.Texture2D.MipSlice);
      t.h    = std::max(1u, texDesc.Height >> desc.Texture2D.MipSlice);
      t.a    = GetScratch(dev.ptr(), t, 'a', true);
      t.b    = GetScratch(dev.ptr(), t, 'b', true);

      if (!t.a || !t.b) {
        skipped.push_back({ i, "scratch creation failed" });
        continue;
      }

      setA[i] = t.a->rtv.ptr();
      setB[i] = t.b->rtv.ptr();
      targets.push_back(t);
    }

    if (targets.empty())
      return LogSkip(*e, "no comparable render target");

    s.inVerify = true;
    e->checksDone += 1u;

    // save what the twins change
    Com<ID3D11PixelShader>        oldPs;
    Com<ID3D11BlendState>         oldBlend;
    Com<ID3D11DepthStencilState>  oldDepth;
    FLOAT oldFactor[4];
    UINT  oldMask = 0u, oldRef = 0u;
    ctx->PSGetShader(&oldPs, nullptr, nullptr);
    ctx->OMGetBlendState(&oldBlend, oldFactor, &oldMask);
    ctx->OMGetDepthStencilState(&oldDepth, &oldRef);

    const FLOAT clear[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    for (uint32_t pass = 0u; pass < 2u; pass++) {
      ID3D11RenderTargetView** set = pass ? setB : setA;

      for (const auto& t : targets)
        ctx->ClearRenderTargetView(pass ? t.b->rtv.ptr() : t.a->rtv.ptr(), clear);

      ctx->OMSetRenderTargets(count, set, nullptr);
      ctx->OMSetBlendState(nullptr, nullptr, D3D11_DEFAULT_SAMPLE_MASK);
      ctx->OMSetDepthStencilState(nullptr, 0u);
      ctx->PSSetShader(pass ? e->psReplace.ptr() : e->psVanilla.ptr(), nullptr, 0u);
      replay();
    }

    ctx->PSSetShader(oldPs.ptr(), nullptr, 0u);
    ctx->OMSetBlendState(oldBlend.ptr(), oldFactor, oldMask);
    ctx->OMSetDepthStencilState(oldDepth.ptr(), oldRef);
    ctx->OMSetRenderTargets(MaxTargets, reinterpret_cast<ID3D11RenderTargetView**>(oldRtv), oldDsv.ptr());

    DiffAndLog(ctx, dev.ptr(), e->name, e->checksDone, s.maxChecks, "draw", targets, SkippedJson(skipped));
    s.inVerify = false;
  }


  void BlessedShaderVerify::OnDispatch(
          D3D11ImmediateContext*  ctx,
    const D3D11ContextState&      state,
    const ReplayFn&               replay) {
    auto& s = GetState();
    std::lock_guard<std::recursive_mutex> lock(s.mutex);

    if (s.inVerify || state.cs == nullptr)
      return;

    Entry* e = FindEntry(state.cs->GetCommonShader());

    if (!e || !e->check || e->checksDone >= s.maxChecks)
      return;

    if (e->csVanilla == nullptr || e->csReplace == nullptr)
      return LogSkip(*e, "no twins");

    Com<ID3D11Device> dev;
    ctx->GetDevice(&dev);

    uint32_t count = std::min(state.uav.maxCount, MaxTargets);
    Com<ID3D11UnorderedAccessView> oldUav[MaxTargets];
    ctx->CSGetUnorderedAccessViews(0u, count, reinterpret_cast<ID3D11UnorderedAccessView**>(oldUav));

    std::vector<Target> targets;
    std::vector<std::pair<uint32_t, std::string>> skipped;
    ID3D11UnorderedAccessView* setA[MaxTargets] = { };
    ID3D11UnorderedAccessView* setB[MaxTargets] = { };

    for (uint32_t i = 0u; i < count; i++) {
      if (oldUav[i] == nullptr)
        continue;

      D3D11_UNORDERED_ACCESS_VIEW_DESC desc;
      oldUav[i]->GetDesc(&desc);

      if (!FloatReadable(desc.Format)) {
        skipped.push_back({ i, str::format("format ", uint32_t(desc.Format)) });
        continue;
      }

      Com<ID3D11Resource> res;
      oldUav[i]->GetResource(&res);

      Target t;
      t.slot = i;
      t.fmt  = desc.Format;

      if (desc.ViewDimension == D3D11_UAV_DIMENSION_TEXTURE2D) {
        Com<ID3D11Texture2D> tex;
        if (FAILED(res->QueryInterface(__uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&tex)))) {
          skipped.push_back({ i, "not a texture2d" });
          continue;
        }

        D3D11_TEXTURE2D_DESC texDesc;
        tex->GetDesc(&texDesc);
        t.w = std::max(1u, texDesc.Width  >> desc.Texture2D.MipSlice);
        t.h = std::max(1u, texDesc.Height >> desc.Texture2D.MipSlice);
      } else if (desc.ViewDimension == D3D11_UAV_DIMENSION_TEXTURE3D) {
        Com<ID3D11Texture3D> tex;
        if (FAILED(res->QueryInterface(__uuidof(ID3D11Texture3D), reinterpret_cast<void**>(&tex)))) {
          skipped.push_back({ i, "not a texture3d" });
          continue;
        }

        D3D11_TEXTURE3D_DESC texDesc;
        tex->GetDesc(&texDesc);
        uint32_t mip = desc.Texture3D.MipSlice;
        uint32_t depth = std::max(1u, texDesc.Depth >> mip);

        if (desc.Texture3D.FirstWSlice != 0u || (desc.Texture3D.WSize != ~0u && desc.Texture3D.WSize != depth)) {
          skipped.push_back({ i, "partial 3d view" });
          continue;
        }

        t.w    = std::max(1u, texDesc.Width  >> mip);
        t.h    = std::max(1u, texDesc.Height >> mip);
        t.d    = depth;
        t.is3d = true;
      } else {
        skipped.push_back({ i, "not a 2d or 3d texture uav" });
        continue;
      }

      t.a = GetScratch(dev.ptr(), t, 'a', false);
      t.b = GetScratch(dev.ptr(), t, 'b', false);

      if (!t.a || !t.b) {
        skipped.push_back({ i, "scratch creation failed" });
        continue;
      }

      setA[i] = t.a->uav.ptr();
      setB[i] = t.b->uav.ptr();
      targets.push_back(t);
    }

    if (targets.empty())
      return LogSkip(*e, "no comparable uav");

    s.inVerify = true;
    e->checksDone += 1u;

    Com<ID3D11ComputeShader> oldCs;
    ctx->CSGetShader(&oldCs, nullptr, nullptr);

    UINT keepCounts[MaxTargets];
    for (auto& c : keepCounts)
      c = ~0u;

    const FLOAT clear[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    for (uint32_t pass = 0u; pass < 2u; pass++) {
      for (const auto& t : targets)
        ctx->ClearUnorderedAccessViewFloat(pass ? t.b->uav.ptr() : t.a->uav.ptr(), clear);

      ctx->CSSetUnorderedAccessViews(0u, count, pass ? setB : setA, keepCounts);
      ctx->CSSetShader(pass ? e->csReplace.ptr() : e->csVanilla.ptr(), nullptr, 0u);
      replay();
    }

    ctx->CSSetShader(oldCs.ptr(), nullptr, 0u);
    ctx->CSSetUnorderedAccessViews(0u, count, reinterpret_cast<ID3D11UnorderedAccessView**>(oldUav), keepCounts);

    DiffAndLog(ctx, dev.ptr(), e->name, e->checksDone, s.maxChecks, "dispatch", targets, SkippedJson(skipped));
    s.inVerify = false;
  }


  void BlessedShaderVerify::CompareTextures(
          ID3D11DeviceContext*    ctx,
    const std::string&            label,
          uint32_t                check,
          uint32_t                checks,
          ID3D11Resource* const*  a,
          ID3D11Resource* const*  b,
          uint32_t                count) {
    auto& s = GetState();
    std::lock_guard<std::recursive_mutex> lock(s.mutex);

    if (s.inVerify)
      return;

    Com<ID3D11Device> dev;
    ctx->GetDevice(&dev);

    std::vector<Scratch> views(2u * count);
    std::vector<Target> targets;
    std::vector<std::pair<uint32_t, std::string>> skipped;

    for (uint32_t i = 0u; i < count && i < MaxTargets; i++) {
      Target t;
      t.slot = i;

      Com<ID3D11Texture2D> tex2d;
      Com<ID3D11Texture3D> tex3d;

      if (SUCCEEDED(a[i]->QueryInterface(__uuidof(ID3D11Texture3D), reinterpret_cast<void**>(&tex3d)))) {
        D3D11_TEXTURE3D_DESC desc;
        tex3d->GetDesc(&desc);
        t.fmt = desc.Format;
        t.w = desc.Width;
        t.h = desc.Height;
        t.d = desc.Depth;
        t.is3d = true;
      } else if (SUCCEEDED(a[i]->QueryInterface(__uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&tex2d)))) {
        D3D11_TEXTURE2D_DESC desc;
        tex2d->GetDesc(&desc);
        t.fmt = desc.Format;
        t.w = desc.Width;
        t.h = desc.Height;
      } else {
        skipped.push_back({ i, "not a 2d or 3d texture" });
        continue;
      }

      if (!FloatReadable(t.fmt)
       || FAILED(dev->CreateShaderResourceView(a[i], nullptr, &views[2u * i].srv))
       || FAILED(dev->CreateShaderResourceView(b[i], nullptr, &views[2u * i + 1u].srv))) {
        skipped.push_back({ i, str::format("format ", uint32_t(t.fmt)) });
        continue;
      }

      t.a = &views[2u * i];
      t.b = &views[2u * i + 1u];
      targets.push_back(t);
    }

    if (targets.empty())
      return;

    s.inVerify = true;
    DiffAndLog(ctx, dev.ptr(), label, check, checks, "texture", targets, SkippedJson(skipped));
    s.inVerify = false;
  }

}
