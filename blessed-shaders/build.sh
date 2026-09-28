#!/bin/sh
# blessed: shader-replace, builds the replacement sets from the game's bundle and src/*.hlsl
#
#   out/identity/  vanilla's own bytecode, byte for byte (the stage 1 proof)
#   out/rewrite/   src/<name>.hlsl compiled by fxc, each gated by iface_check.py
#
# point BLESSED_SHADER_REPLACE at one of them. reads D:\ only.
set -e
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"
fxc="/c/Program Files (x86)/Windows Kits/10/bin/10.0.28000.0/x64/fxc.exe"

identity="fs.b63b3bf7d8862ae508dfbb37d874f3af cs.cf1e1a211573e943f02eeb926ccbe8c6"
rewrites="$(ls src/*.hlsl 2>/dev/null | sed 's|src/||; s|\.hlsl$||')"

need=""
for n in $identity $rewrites; do
  [ -f "vanilla/$n.dxbc" ] || need="$need $n"
done
[ -z "$need" ] || python fxp_extract.py vanilla $need

rm -rf out/identity out/rewrite
mkdir -p out/identity out/rewrite

for n in $identity; do
  cp "vanilla/$n.dxbc" "out/identity/$n.dxbc"
done
echo "identity: $(ls out/identity | wc -l) shader(s)"

fail=0
for n in $rewrites; do
  case "$n" in
    fs.*) profile=ps_5_0 ;;
    cs.*) profile=cs_5_0 ;;
    *) echo "$n: unknown stage"; fail=1; continue ;;
  esac
  "$fxc" -nologo -T $profile -E main -O3 -WX -Fo "out/rewrite/$n.dxbc" "src/$n.hlsl" >/dev/null
  if ! python iface_check.py "vanilla/$n.dxbc" "out/rewrite/$n.dxbc"; then
    rm -f "out/rewrite/$n.dxbc"
    fail=1
  fi
done
echo "rewrite: $(ls out/rewrite | wc -l) shader(s)"
exit $fail
