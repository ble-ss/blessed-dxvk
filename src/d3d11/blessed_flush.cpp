// blessed: submission policy for the immediate context, fewer mid-pass submissions (BLESSED_FLUSH)
#include <string>

#include "blessed_flush.h"

#include "../util/util_env.h"

namespace dxvk {

  namespace {

    uint32_t InitPolicy() {
      // no Logger here: static init may run before the logger exists
      std::string policy = env::getEnvVar("BLESSED_FLUSH");

      if (policy == "pass")
        return 1u;

      if (policy == "strong")
        return 2u;

      return 0u;
    }

  }

  namespace blessed_flush_detail {
    extern const uint32_t g_policy = InitPolicy();
  }

}
