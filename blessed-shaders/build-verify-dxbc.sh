#!/bin/sh
# regenerate src/d3d11/blessed_shader_verify_dxbc.h and blessed_vol_collapse_dxbc.h from their hlsl
set -e
here="$(cd "$(dirname "$0")/.." && pwd)"
fxc="/c/Program Files (x86)/Windows Kits/10/bin/10.0.28000.0/x64/fxc.exe"
src="$here/src/d3d11/shaders/blessed_shader_verify.hlsl"
out="$here/src/d3d11/blessed_shader_verify_dxbc.h"
tmp="$(mktemp -d)"
"$fxc" -nologo -T cs_5_0 -E main -O3 -Vn blessed_shader_verify_2d -Fh "$tmp/2d.h" "$(cygpath -w "$src")"
"$fxc" -nologo -T cs_5_0 -E main -O3 -D VERIFY_3D=1 -Vn blessed_shader_verify_3d -Fh "$tmp/3d.h" "$(cygpath -w "$src")"
{
  echo "// blessed: shader-replace verifier diff kernels, fxc output of shaders/blessed_shader_verify.hlsl (generated, see blessed-shaders/build-verify-dxbc.sh)"
  echo "#pragma once"
  echo
  sed -n '/^const BYTE/,/^};/p' "$tmp/2d.h" | sed 's/^const BYTE/static const unsigned char/'
  echo
  sed -n '/^const BYTE/,/^};/p' "$tmp/3d.h" | sed 's/^const BYTE/static const unsigned char/'
} > "$out"
src2="$here/src/d3d11/shaders/blessed_vol_collapse.hlsl"
out2="$here/src/d3d11/blessed_vol_collapse_dxbc.h"
"$fxc" -nologo -T cs_5_0 -E main -O3 -WX -Vn blessed_vol_collapse_cs -Fh "$tmp/vc.h" "$(cygpath -w "$src2")"
{
  echo "// blessed: vol-collapse chain replay kernel, fxc output of shaders/blessed_vol_collapse.hlsl (generated, see blessed-shaders/build-verify-dxbc.sh)"
  echo "#pragma once"
  echo
  sed -n '/^const BYTE/,/^};/p' "$tmp/vc.h" | sed 's/^const BYTE/static const unsigned char/'
} > "$out2"
echo "wrote $out2"
rm -rf "$tmp"
echo "wrote $out"
