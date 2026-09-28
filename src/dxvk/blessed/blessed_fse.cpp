// blessed: present-fse-appcontrolled -- BLESSED_FSE mode switch
#include <string>

#include "blessed_fse.h"

#include "../../util/util_env.h"

namespace dxvk {

  BlessedFseMode BlessedFse::Mode() {
    static const BlessedFseMode s_mode = [] {
      std::string v = env::getEnvVar("BLESSED_FSE");

      if (v == "app")
        return BlessedFseMode::App;

      return BlessedFseMode::Off;
    } ();

    return s_mode;
  }

}
