#!/bin/sh
# perlayer seat: one layertest run on the fork's dlls.
# usage: run.sh <outdir> <on|off> <vvl|novvl> <seconds> <mode> <draws> <layers> <size> [verifyEvery]
set -e
out=$1; lt=$2; vvl=$3; shift 3
here=$(cd "$(dirname "$0")" && pwd)
wt=$(cd "$here/../.." && pwd)
b=${BUILD:-build.msvc}
rm -rf "$out"; mkdir -p "$out"
cp "$here/layertest.exe" "$out/"
cp "$wt/$b/src/d3d11/d3d11.dll" "$wt/$b/src/dxgi/dxgi.dll" "$out/"
w=$(cygpath -w "$out")
export DXVK_LOG_PATH="$w" DXVK_LOG_LEVEL=info
[ "$lt" = on ] && export BLESSED_LAYER_TRACKING=1 || unset BLESSED_LAYER_TRACKING
if [ "$vvl" = vvl ]; then
  export VK_LAYER_PATH='E:\blessed_skyrim\tools\vvl' VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
  export VK_LAYER_SETTINGS_PATH="$w\vk_layer_settings.txt"
  printf 'khronos_validation.debug_action = VK_DBG_LAYER_ACTION_LOG_MSG\nkhronos_validation.log_filename = %s\nkhronos_validation.validate_sync = true\nkhronos_validation.report_flags = error,warn\n' "$w\\vvl.log" > "$out/vk_layer_settings.txt"
fi
cd "$out"
set +e
./layertest.exe "$@" > app.log 2>&1
rc=$?
echo "rc=$rc" >> app.log
cat app.log
