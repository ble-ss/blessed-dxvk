#!/bin/sh
# volasync seat: one volasynctest run on the fork's dlls.
# usage: run.sh <outdir> <sync|1|2> <vvl|novvl> <verify|noverify> <seconds> <maxFrames> <steps> <rasterDraws> <rasterIters> <checkEvery> [uploadMB]
# env: BUILD (default build.msvc), TIMING=1 adds BLESSED_ASYNC_TIMING=1, FE=0 drops the threaded front end,
#      DESC=sets forces dxvk's descriptor-set binding model: with the heap or descriptor buffers (the
#      default on this machine), vvl's sync validation cannot see what a shader reads or writes
#      through a descriptor, only copies, clears, attachments, barriers and ownership;
#      XCONF='a = b;c = d' appends dxvk.conf lines; DLLDIR=<dir> takes the dlls from there
set -e
out=$1; mode=$2; vvl=$3; ver=$4; shift 4
here=$(cd "$(dirname "$0")" && pwd)
wt=$(cd "$here/../.." && pwd)
b=${BUILD:-build.msvc}
rm -rf "$out"; mkdir -p "$out"
cp "$here/volasynctest.exe" "$out/"
if [ -n "${DLLDIR:-}" ]; then cp "$DLLDIR/d3d11.dll" "$DLLDIR/dxgi.dll" "$out/"; else cp "$wt/$b/src/d3d11/d3d11.dll" "$wt/$b/src/dxgi/dxgi.dll" "$out/"; fi
w=$(cygpath -w "$out")
# the game's own cbuffer settings (c52's dxvk_conf), plus the skyrim profile's cached dynamic cbuffers
{
  echo "d3d11.cachedDynamicResources = c"
  echo "d3d11.blessedCbRing = True"
  echo "d3d11.blessedCbMirror = True"
  [ "${FE:-1}" = 1 ] && echo "d3d11.blessedThreadedFrontEnd = True"
  [ -n "${XCONF:-}" ] && echo "$XCONF" | tr ';' '\n'
  if [ "${DESC:-}" = sets ]; then
    echo "dxvk.enableDescriptorHeap = False"
    echo "dxvk.enableDescriptorBuffer = False"
  fi
} > "$out/dxvk.conf"
export DXVK_CONFIG_FILE="$w\\dxvk.conf" DXVK_LOG_PATH="$w" DXVK_LOG_LEVEL=info BLESSED_PROBE_DIR="$w"
unset BLESSED_VOL_ASYNC BLESSED_VOL_ASYNC_DRYRUN BLESSED_VOL_ASYNC_VERIFY BLESSED_ASYNC_TIMING BLESSED_ASYNC BLESSED_ASYNC_QUEUE
[ "$mode" != sync ] && export BLESSED_VOL_ASYNC=$mode
[ "$ver" = verify ] && export BLESSED_VOL_ASYNC_VERIFY=1 BLESSED_VOL_ASYNC_VERIFY_PERIOD=${PERIOD:-7} BLESSED_VOL_ASYNC_VERIFY_MAX=0
[ "${TIMING:-0}" = 1 ] && export BLESSED_ASYNC_TIMING=1
[ "${DRYRUN:-0}" = 1 ] && export BLESSED_VOL_ASYNC_DRYRUN=1
if [ "$vvl" = vvl ]; then
  export VK_LAYER_PATH='E:\blessed_skyrim\tools\vvl' VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
  export VK_LAYER_SETTINGS_PATH="$w\vk_layer_settings.txt"
  printf 'khronos_validation.debug_action = VK_DBG_LAYER_ACTION_LOG_MSG\nkhronos_validation.log_filename = %s\nkhronos_validation.validate_sync = true\nkhronos_validation.report_flags = error,warn\n' "$w\\vvl.log" > "$out/vk_layer_settings.txt"
fi
cd "$out"
set +e
./volasynctest.exe "$@" > app.log 2>&1
rc=$?
echo "rc=$rc" >> app.log
cat app.log
