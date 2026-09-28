#!/bin/sh
# zero-copy-present seat: runs zc_test.exe twice (BLESSED_ZERO_COPY=0 then
# =1) against the zero-copy worktree's own build, and prints both so the
# verdicts/counts can be compared by eye. usage: run.sh [seconds] [pollMs]
#
# safety: reuses present-idle's quiet gate (no campaign/skybench running),
# PLUS an explicit refusal if SkyrimSE.exe or PresentMon* is already
# running -- the brief's own wording, checked by name, not only via
# quiet.sh's campaign/skybench process scan. hard-timed with GNU
# coreutils `timeout` so a hung window can't sit there indefinitely.
set -e
here=/e/blessed_skyrim/wt/zero-copy/zc-test
wt=/e/blessed_skyrim/wt/zero-copy
secs=${1:-6}
pollms=${2:-33}

sh /e/blessed_skyrim/wt/present-idle/pi-test/quiet.sh

busy=$(powershell -NoProfile -Command "(Get-Process -Name SkyrimSE,PresentMon* -ErrorAction SilentlyContinue | Measure-Object).Count" 2>/dev/null || echo 0)
if [ "$busy" != "0" ]; then
  echo "refusing to run: SkyrimSE.exe or PresentMon* is already running"
  exit 1
fi

if [ ! -f "$here/zc_test.exe" ]; then
  echo "zc_test.exe not built -- run build.cmd first"
  exit 1
fi

# private, self-contained dll copy: the executable's own directory wins
# dll search order over system32, so this loads OUR build without ever
# touching tools/dxvk-fork or D:\.
# BUILD=build.msvc picks the msvc build; the default is the noprobe mingw one.
build=${BUILD:-build.mingw-noprobe}
for f in d3d11.dll d3d11.pdb dxgi.dll dxgi.pdb; do
  src="$wt/$build/src/d3d11/$f"
  [ -f "$src" ] || src="$wt/$build/src/dxgi/$f"
  [ -f "$src" ] && cp -f "$src" "$here/$f"
done

if [ ! -f "$here/d3d11.dll" ]; then
  echo "no built d3d11.dll found under $wt/$build -- build the worktree first"
  exit 1
fi

cd "$here"

# external, independent screen confirmation (round 2, lead's ask): zc_test.exe
# can no longer check actual screen content itself -- see zc_test.cpp's file
# header for why IDXGIOutputDuplication never works from inside a process
# that's loaded dxvk's own dxgi.dll. same technique present-cursed's
# pc-test/run.sh uses instead: a plain GDI BitBlt screenshot
# (System.Drawing.Graphics.CopyFromScreen) from a completely separate
# powershell.exe process, which never touches dxvk's dlls at all. taken at
# the run's midpoint.
screenshot() {
  label=$1; runsecs=$2
  half=$(( ${runsecs%.*} / 2 )); [ "$half" -lt 1 ] && half=1
  ( sleep "$half"; powershell.exe -NoProfile -Command "
Add-Type -AssemblyName System.Windows.Forms,System.Drawing
\$sz = [System.Windows.Forms.SystemInformation]::PrimaryMonitorSize
\$b = New-Object System.Drawing.Bitmap(\$sz.Width, \$sz.Height)
\$g = [System.Drawing.Graphics]::FromImage(\$b)
\$g.CopyFromScreen(0, 0, 0, 0, \$b.Size)
\$b.Save('$(cygpath -w "$here")\screenshot-$label.png', [System.Drawing.Imaging.ImageFormat]::Png)
\$g.Dispose(); \$b.Dispose()
" > "screenshot-$label.log" 2>&1 ) &
}

echo "== BLESSED_ZERO_COPY=0 (baseline) =="
screenshot off "$secs"
BLESSED_ZERO_COPY=0 timeout -k 5 60 ./zc_test.exe "$secs" "$pollms" > run-off.log 2>&1 || echo "(zc_test.exe off-run exit $?)"
wait
cat run-off.log
[ -f screenshot-off.png ] && echo "-- midrun screenshot: $here/screenshot-off.png --" || { echo "-- midrun screenshot FAILED --"; cat screenshot-off.log 2>/dev/null; }

echo
echo "== BLESSED_ZERO_COPY=1 (redirect) =="
screenshot on "$secs"
BLESSED_ZERO_COPY=1 timeout -k 5 60 ./zc_test.exe "$secs" "$pollms" > run-on.log 2>&1 || echo "(zc_test.exe on-run exit $?)"
wait
cat run-on.log
[ -f screenshot-on.png ] && echo "-- midrun screenshot: $here/screenshot-on.png --" || { echo "-- midrun screenshot FAILED --"; cat screenshot-on.log 2>/dev/null; }

echo
echo "== verdicts =="
grep -H "verdict:" run-off.log run-on.log
