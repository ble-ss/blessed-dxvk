# sao/taa seat notes (2026-09-25)

worktree `wt/saotaa` from `blessed` b5a6bbd9. brief: `.briefs/sao-taa.md`, two
shaders flagged by the hotspots seat's notes (`blessed-notes/gpu-hotspots-notes.md`,
rank 10 and 11): vanilla sao `83560015` (dead under `BLESSED_RTAO`, +10%,
0.148/0.135 ms) and the taa resolve `72db6ece` (+6%, 0.143/0.135 ms).

## fs.83560015 (vanilla sao) -- replacement shipped, small measured win, one algorithmic dead end shipped separately

read from `fxc /dumpbin` of the vanilla dxbc (pulled read-only from
`Skyrim - Shaders.bsa` via `blessed-shaders/fxp_extract.py 83560015`,
disassembly kept locally as `sao.asm`, not committed -- game bytecode stays
out of git per `blessed-shaders/.gitignore`). what it does: a fullscreen pass,
no blend, no depth bound, one render target (rgba8):

- a 5-tap spiral ssao kernel against a mip-chained linear-depth buffer (t0),
  each tap's mip level chosen from its own sample radius (farther taps read a
  coarser mip)
- a hemisphere hint vector reconstructed from a 2-channel encoded normal (t1)
- occlusion accumulated per tap as `dot(delta, hint)/dist^2 * falloff^3`,
  gated by a sky mask on the tap's depth
- the ao result gamma-remapped by an artist power curve (symmetric branch
  around power==0.5), then dilated along x then y using screen derivatives of
  depth (to avoid bleeding across depth edges) and a checkerboard dither (to
  avoid the dilation aliasing on its own)
- a final edge-aware blend against a lower-resolution bounced-gi buffer (t2),
  jittered by a blue-noise-style vector (t3)
- the center pixel's own depth is separately packed into the output's y/z
  channels (two bytes, for a downstream consumer) -- independent of the ao math
- ten texture fetches total (t0 x6, t1 x1, t2 x1, t3 x2), not the ~30-tap
  kernel the hotspots seat's notes guessed from the frame capture alone (a
  guess they flagged as unconfirmed, since they didn't have time to disassemble
  it) -- correcting that estimate is this seat's first finding.

### what shipped (`blessed-shaders/src/fs.835600159503652602087a8d0f9e7204.hlsl`)

two changes, both measured on the standalone harness (`harness/pass16.cpp`,
below), not just argued:

1. the per-tap mip level's `log2(radiusStep*0.2)` is split via
   `log2(a*b)==log2(a)+log2(b)` into one `log2(radiusScale)` hoisted above the
   loop, plus a baked-constant add per iteration (`log2((i+0.5)*0.2)` for
   i in 0..4) -- four fewer `log2` calls per pixel.
2. the normal fetch's `sample_d` with explicit zero gradients (vanilla's way
   of forcing mip 0 without a screen-edge derivative discontinuity) is now an
   explicit-lod `SampleLevel(0)` -- same mip, same filtering, the cheaper
   instruction path.

interface-checked clean (`iface_check.py`, via `build.sh`), zero fxc warnings
(`-WX`).

### tried and measured, not shipped: unrolling the 5-iteration loop

the obvious first idea -- the loop is a real `loop`/`breakc`-guarded loop
over a compile-time-constant trip count of 5, so `[unroll]` should remove the
per-pixel branch/counter overhead for free. built it, timed it, and it's
**slower**, reproducibly, on both backends:

| variant | native | dxvk (fork, `tools/dxvk-fork/x64`) |
|---|---|---|
| vanilla | 0.1300-0.1331 ms | 0.1597-0.1601 ms |
| `[unroll]` | 0.1340-0.1352 ms (+3-4%) | 0.1603-0.1635 ms (+1-4%) |
| `[loop]` (kept, no other change) | -- | 0.1610 ms (parity) |
| shipped (`[loop]` + log2 split + SampleLevel) | 0.1239-0.1260 ms (-5%) | 0.1593-0.1595 ms (-0.4%) |

reproduced 3-8x per row; the dxvk shipped-vs-vanilla gap (0.1599-0.1601 vs
0.1593-0.1595) held across 8 consecutive runs with no overlap. the unroll
result says this shader is texture-fetch-bound (five dependent, LOD-varying
samples per pixel), not control-flow-bound -- removing the loop's branch
overhead doesn't help because the GPU is waiting on memory either way, and
the larger unrolled code body likely costs a little register pressure/icache
back. that also explains why the shipped ALU trims (four fewer `log2`,
one cheaper sample) land as a small, real win rather than a large one: they
shave the part of the pass that isn't the bottleneck.

**not attempted:** fp16 for the loop's accumulator math. considered and
dropped for two reasons: (1) the unroll finding above says this is a fetch-
bound shader, so fp16 ALU is unlikely to move the needle either; (2) the
shader has a real exact-equality gate right after the loop (`depthNorm==1.0`,
the sky detect just before the final `movc` into `o0.x`) and a`ge`-based sky
mask inside the loop -- both fp32, both computed well outside anything this
seat's changes touch, but a reason for caution about precision changes
anywhere in this shader's neighborhood.

### a real transcription bug caught and fixed before it shipped

first draft of the port had the normal's z-component as
`-sqrt(-nLenSq*0.5+1)`. the vanilla disassembly (`sao.asm` line 65,
`mul r4.xyz, r4.xywx, l(-1,1,-1,0)`) shows the z-component is `-r4.w`
straight, where `r4.w = -nLenSq*0.5+1` (line 62) -- **no sqrt**. the sqrt at
line 63 (`sqrt r2.z, r4.z`) feeds a *different* register (the xy scale, from
`r4.z = -nLenSq*0.25+1`, a different constant). fixed to `nZTerm =
-nLenSq*0.5+1` directly. worth recording because the harness's own bit-diff
(below) did **not** catch this on either draft -- see the caveat.

also tightened for numerical fidelity once the bug-hunt was underway (found
by the same line-by-line re-read, not measured to matter, but free to fix):
the power curve's high branch now does `1.0/denom` then multiply, matching
vanilla's explicit `div`-then-`mul` (not a single HLSL divide, which could
round once instead of twice); the dither-vector normalize is one `rsqrt`
(matching vanilla's `rsq`), not `sqrt` then a reciprocal.

### the harness (`harness/pass16.cpp`) and its bit-diff's real limits

standalone d3d11 microbenchmark, no game, no `D:\` write: a fullscreen
triangle into an rgba8 target against synthetic t0 (r32f depth), t1/t2
(rgba8), t3 (rg16f), timed with d3d11 timestamps. given two dxbc paths it
also renders both against the *identical* synthetic bindings and reads both
back, reporting max per-channel byte difference.

**both the buggy draft and the fixed version scored zero differing bytes
against real vanilla bytecode on this harness's synthetic data.** that is a
weak signal, not a strong one -- worth saying plainly rather than letting the
zero read as more than it is. the likely reason: the loop's `falloff =
max(c3.y - distSq, 0)^3` term gates almost everything the normal feeds into
(`contribution = ratio * falloff`), and this harness's synthetic depth field
and cb2/cb12 constants are not the same scale as the game's real depth
buffer and per-frame constants -- if `falloff` lands near zero for most of
the synthetic image, the normal's contribution (and so the bug) never shows
up in the 8-bit output. the harness is genuinely useful for **timing** (real
d3d11, real driver, real binding shape) and for catching gross breakage
(wrong format, wrong slot, a crash), but it is not a substitute for
`BLESSED_SHADER_VERIFY`'s twin-draw against the game's own buffers, which is
the only check that will exercise the real value ranges. **owed to the lead**,
same as the hotspots seat's items: an in-game run with
`BLESSED_SHADER_REPLACE=<out/rewrite dir> BLESSED_SHADER_VERIFY=1
BLESSED_SHADER_VERIFY_LIST=835600159503652602087a8d0f9e7204` and a look at
`shader-verify.jsonl` before this ships for real.

### how to enable

`BLESSED_SHADER_REPLACE=<path to blessed-shaders/out/rewrite>` (built via
`blessed-shaders/build.sh`, which auto-discovers `src/*.hlsl` and gates each
one through `iface_check.py`). the replacement is a drop-in swap keyed by the
vanilla hash, same as every other shader-replace entry.

## fs.72db6ece (taa resolve) -- disassembled, timed, no replacement shipped

what it does: two render targets (rgba8, no blend, no depth), six srvs (t0/t1
color-ish rgba8, t2 rg16f, t3 a depth srv (r24_unorm_x8_typeless over an r24g8
resource), t4 rg8, t5 rgba8). the core of the shader is a **3x3 neighborhood
closest-candidate search**: 9 taps of t3 (the depth srv, single channel
replicated across all four components by the srv's default mapping -- this is
also why different lines read it with different, otherwise-meaningless
component swizzles) at the center pixel and its 8 texel-aligned neighbors,
each compared and reduced via a chain of `eq`/`lt`/`movc` to find the
neighbor with the closest depth to a reference; that neighbor's matching t0
(color) and t5 (a stencil-like disocclusion flag) samples are then carried
forward. a second stage does a near-identical closest-match reduction over
the same 9 positions using a different distance metric (line ~176-231 of
`taa.asm`, not fully traced) before a final history blend gated by cb2[4]
weights and multiple disocclusion checks against t4/t5.

**no replacement shipped.** reasoning, in order of weight:

1. the closest-candidate reduction is built on **exact floating-point
   equality** between derived values (`eq` appears at least 8 times in the
   328-instruction disassembly, each one deciding which neighbor's data
   propagates). any restructuring -- even a same-semantics recompilation
   through a different codepath -- risks the compiler choosing a different
   fma/rounding somewhere upstream of one of those comparisons, which would
   not show up as a small numeric drift: it would silently change *which
   neighbor* gets selected, a correctness class of bug (visible ghosting or
   flicker), not a precision one.
2. the one lever that looked genuinely safe -- t3 is a plain single-channel
   depth srv read at texel-aligned integer offsets, a strong candidate for
   `Sample` -> `Load` (skip the sampler's address/filter path entirely, since
   a point- or degenerate-linear-filtered fetch at an exact texel center
   returns the same texel either way) -- still requires transcribing the
   entire 328-instruction shader by hand to change even one instruction
   family, since the pipeline compiles whole hlsl files, not binary patches.
   that transcription is exactly the exact-equality risk in (1), just moved
   earlier: any slip anywhere in the other ~300 instructions feeding those
   `eq` gates is the same class of bug.
3. payoff is the smallest of the two passes in the hotspots table (+6%,
   0.008 ms absolute) -- the risk/reward does not clear the bar this seat's
   time budget supports, matching the hotspots seat's own original triage of
   both these passes as "flagged for a future seat" before this brief handed
   them over.

this mirrors the hotspots seat's own precedent (`gpu-hotspots-notes.md`'s
disproven per-buffer fix): a reasoned no-ship, not a shrug.

### timed anyway (`harness/pass61.cpp`)

same approach as pass16's harness, no replacement to diff against, so timing
only: two rgba8 targets, six synthetic srvs shaped like the real pass
(rgba8/rgba8/rg16f/depth-srv/rg8/rgba8), cb12[45] + cb2[6] (the vanilla dxbc
declares `CB2[6]`, not sao's `CB2[4]`).

- native: 0.2406 ms (median of 50)
- dxvk (fork): 0.2422 ms (median of 50)

these numbers are this harness's own synthetic-binding cost, not the real
frame's 0.143/0.135 ms (same caveat pass37's harness carried for the soft-
particle pass: a standalone microbenchmark times the real binding *shape* on
real hardware, not the real frame's exact texture content or state history).
useful as a reusable timing rig for whoever revisits this pass; not a
substitute for the frame capture's own numbers.

## owed to the lead

- **sao**: an in-game `BLESSED_SHADER_VERIFY` run on `835600159503652602087a8d0f9e7204`
  before this replacement is trusted -- the standalone harness's bit-diff is
  a weak signal here (see above), not a strong one.
- **sao**: if the lead wants more than the modest measured win (~0.4% on
  dxvk, ~5% native, both real but small), the next lever is the taps
  themselves, not the ALU around them -- none of the ten fetches are
  redundant, so that would mean changing the sampling pattern itself
  (e.g. fewer, wider taps), which changes the image and needs an art call,
  not just an engineering one.
- **taa**: untouched, on purpose (see reasoning above). if someone wants to
  revisit it, `harness/pass61.cpp` and the disassembly notes above are the
  running start; the real lever (t3's 9 `Sample` calls -> `Load`) is
  identified but not attempted, specifically because attempting it means
  transcribing the whole shader and this seat judged that not worth the risk
  for a 6%/0.008ms pass.
