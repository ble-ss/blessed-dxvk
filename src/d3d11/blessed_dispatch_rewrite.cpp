// blessed: shader-replace, the data-driven dispatch-size override table (see blessed_dispatch_rewrite.h)
#include "blessed_dispatch_rewrite.h"

#include "d3d11_shader.h"

#include "../util/util_env.h"
#include "../util/log/log.h"
#include "../util/util_string.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace dxvk {

  namespace {

    struct Rule {
      UINT matchX, matchY, matchZ;
      UINT newX,   newY,   newZ;
    };

    struct Table {
      // one shader name may carry more than one rule (different dispatch
      // sizes seen in different scenes); all are exact-match only.
      std::unordered_map<std::string, std::vector<Rule>> rules;
    };

    // parses every line "mx my mz -> nx ny nz" out of one sidecar file.
    // Malformed lines are skipped with a warning; a sidecar that yields no
    // rule at all is the same as not being there.
    std::vector<Rule> ParseSidecar(const std::filesystem::path& path) {
      std::vector<Rule> out;

      std::ifstream f(path);
      if (!f.good())
        return out;

      std::string line;
      while (std::getline(f, line)) {
        // trim trivial whitespace/CR so a sidecar edited on windows parses
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' ' || line.back() == '\t'))
          line.pop_back();
        if (line.empty())
          continue;

        Rule r = { };
        char arrow[3] = { };
        std::istringstream iss(line);
        if (iss >> r.matchX >> r.matchY >> r.matchZ >> arrow[0] >> arrow[1] >> r.newX >> r.newY >> r.newZ
            && arrow[0] == '-' && arrow[1] == '>') {
          out.push_back(r);
        } else {
          Logger::warn(str::format("BlessedDispatchRewrite: ", path.string(), ": unparsable line, ignored: ", line));
        }
      }

      return out;
    }

    // rules are live only when BLESSED_SHADER_REPLACE holds a sidecar
    // <name>.dispatch beside the compacted replacement's .dxbc. Keying on
    // the game's shader name alone would also reshape the dispatch of an
    // unreplaced or differently-shaped bound shader of the same name.
    Table ComputeTable() {
      Table t;
      std::string dir = env::getEnvVar("BLESSED_SHADER_REPLACE");
      if (dir.empty())
        return t;

      std::error_code ec;
      std::filesystem::path dirPath = str::topath(dir.c_str());

      for (auto it = std::filesystem::directory_iterator(dirPath, ec);
           !ec && it != std::filesystem::directory_iterator(); it.increment(ec)) {
        if (!it->is_regular_file(ec) || it->path().extension() != ".dispatch")
          continue;

        std::string name = it->path().stem().string();
        auto parsed = ParseSidecar(it->path());

        if (!parsed.empty())
          t.rules.emplace(std::move(name), std::move(parsed));
      }

      return t;
    }

    const Table g_table = ComputeTable();

    bool ComputeEnabled() {
      return !g_table.rules.empty();
    }

  }

  namespace blessed_dispatch_rewrite_detail {
    extern const bool g_enabled = ComputeEnabled();
  }

  void BlessedDispatchRewrite::Adjust(
    const D3D11ContextState& state,
          UINT&               x,
          UINT&               y,
          UINT&               z) {
    if (state.cs == nullptr)
      return;

    auto it = g_table.rules.find(state.cs->GetCommonShader()->GetName());
    if (it == g_table.rules.end())
      return;

    for (const auto& r : it->second) {
      if (x == r.matchX && y == r.matchY && z == r.matchZ) {
        x = r.newX;
        y = r.newY;
        z = r.newZ;
        return;
      }
    }
  }

}
