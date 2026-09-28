#!/bin/sh
# fe-getters seat: builds omshadowtest.exe if needed, then runs the fixed
# OM shadow sequence twice against the fork's own d3d11.dll/dxgi.dll --
# once with the threaded front end off, once on (loopback, deterministic
# same-thread replay) -- and diffs the two traces. Equal output is a pass:
# the shadow answered every OMGet the same way a plain context would.
#
# usage: run.sh <outdir>
# requires a vcvars64 shell (cl.exe on PATH) to build; skips the build if
# omshadowtest.exe is already next to this script.
set -e
out=$1
here=$(cd "$(dirname "$0")" && pwd)
wt=$(cd "$here/../.." && pwd)
b=${BUILD:-build.msvc}

if [ ! -f "$here/omshadowtest.exe" ]; then
  ( cd "$here" && cl /nologo /EHsc /O2 omshadowtest.cpp /link d3d11.lib dxgi.lib user32.lib /out:omshadowtest.exe ) \
    || { echo "build failed -- run this from a vcvars64 shell"; exit 1; }
fi

rm -rf "$out"; mkdir -p "$out"
cp "$here/omshadowtest.exe" "$out/"
cp "$wt/$b/src/d3d11/d3d11.dll" "$wt/$b/src/dxgi/dxgi.dll" "$out/"
w=$(cygpath -w "$out")
export DXVK_LOG_PATH="$w" DXVK_LOG_LEVEL=info

cd "$out"
set +e

unset DXVK_CONFIG
./omshadowtest.exe > off.trace 2> off.log
rc_off=$?

export DXVK_CONFIG="d3d11.blessedThreadedFrontEnd = True; d3d11.blessedThreadedFrontEndLoopback = True;"
./omshadowtest.exe > on.trace 2> on.log
rc_on=$?

# review: the same loopback run with the shadow verify on (every shadowed
# getter also drains and compares the whole OM shadow with dxvk's state),
# and a threaded (not loopback) run, whose trace must match too
mkdir -p verify threaded
cp omshadowtest.exe d3d11.dll dxgi.dll verify/
cp omshadowtest.exe d3d11.dll dxgi.dll threaded/
( cd verify && DXVK_LOG_PATH="$w\verify" BLESSED_FE_SHADOW_VERIFY=1 BLESSED_FE_DRAIN_STATS=1 ./omshadowtest.exe > on.trace 2> on.log )
rc_verify=$?

export DXVK_CONFIG="d3d11.blessedThreadedFrontEnd = True;"
( cd threaded && DXVK_LOG_PATH="$w\threaded" ./omshadowtest.exe > on.trace 2> on.log )
rc_thr=$?
unset DXVK_CONFIG

echo "off rc=$rc_off, on rc=$rc_on, verify rc=$rc_verify, threaded rc=$rc_thr"
fail=0
for t in on.trace verify/on.trace threaded/on.trace; do
  diff -u off.trace "$t" || { echo "FAIL: $t diverged from off.trace"; fail=1; }
done
mism=$(cat verify/*d3d11.log 2>/dev/null | grep -c "blessed_fe_shadow_verify: mismatch")
echo "verify: $(cat verify/*d3d11.log 2>/dev/null | grep "blessed_fe_shadow_verify: calls=" | tail -1) mismatch lines=$mism"
[ "$mism" = 0 ] || { echo "FAIL: shadow verify logged mismatches"; fail=1; }
[ $fail = 0 ] && echo "PASS: shadow matches the plain context" || exit 1
