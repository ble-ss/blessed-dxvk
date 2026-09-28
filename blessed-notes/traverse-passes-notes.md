# traverse-passes seat notes (2026-09-25)

worktree `wt/traverse` from `blessed` 7b1dc318 (post cb-mirror merge). brief: `.briefs/traverse-passes.md`.

## summary

item 1's premise does not hold for `556a3b73` once measured against the running
mirror: its cbuffers are already fully ring- and mirror-backed, with zero
fallback. items 2 and 3 have a real, structurally-confirmed shape but no fix
I could build and trust without either a traversal-scene frame dump or an
in-game a/b, neither of which this seat can produce (no game launch). rather
than ship an unverified change -- the exact mistake the hotspots seat already
made once on this same pass and reverted -- I'm handing back the evidence and
the next concrete step for each.

**update, lead follow-up with `bench/runs/traverse-dump-2` (real traversal frames)
and the `vir` in-game a/b result (31% cpu regression, that lever is closed):**
`566c03f3` is now traced and root-caused (below) -- it isn't a cbuffer problem
at all, it's a dynamic vertex buffer, a resource class the cb-mirror never
touches. Built and shipped: `d3d11.blessedVbRebar`. Water cube (`79c17b07`)
is now traced down to the exact dxvk mechanism between faces, and it's
confirmed to sit in dxvk's general per-image (not per-subresource) write-
tracking -- real, but not a contained fix; not shipped.

## 1. `556a3b73` -- already fully mirrored; the gap is not memory locality

`bench/dumpview.py bench/runs/whiterun-dump-1/probe/frame-1920.jsonl --shader 556a3b73`:
the pass's cbuffers (vs slots 0,1,2,7,9,10,12; ps slots 0,1,2,11,12; sizes
16-3840 bytes) are all, per `frame-1920-resources.jsonl`: `usage=2` (DYNAMIC),
`bind_flags=4` (CONSTANT_BUFFER only), `cpu_access=65536` (WRITE),
`misc_flags=0`. That is exactly `D3D11Buffer`'s cb-ring eligibility gate
(`src/d3d11/d3d11_buffer.cpp:134-140`, `m_blessedCbRing`): DYNAMIC,
bind flags == CB only, no misc flags, `ByteWidth <= BlessedCbRing::MaxChunkSize`
(16 KiB; the largest here is 3840), host-cached memory, not a bones buffer.
So every one of this pass's cbuffers is ring-eligible, which makes it
mirror-eligible whenever `d3d11.blessedCbMirror` is on.

Confirmed at runtime, not just by inference: `bench/runs/c54-passes-pm-mirror-traverse-r1/SkyrimSE_d3d11.log` --
`d3d11.blessedCbMirror: last 600 frames -- renames 25200 mirrored, 0 cpu-read
... 0 lookup misses`. Zero cpu-read exclusions, zero misses, for the whole
traversal run this pass's timing came from.

And the mirror is a real substitution, not just a copy that nothing reads:
`DxvkContext::blessedRenameBuffers` (`src/dxvk/blessed/blessed_cb_ring.cpp:44-45`)
calls `buffer->blessedAssignStorageRange(useMirror ? mirror : block, ...)` --
the `DxvkBuffer`'s bound storage becomes the device-local mirror allocation
itself, not the host block. Every draw's descriptor for these cbuffers reads
vram, confirmed by the counters above (100% coverage, no fallback path taken).

So the "if they live in host memory, extend the mirror" branch of the brief
doesn't apply here: they don't live in host memory anymore, for every draw in
this traversal capture. Extending the mirror to reach `556a3b73`'s buffers is
a no-op -- it already reaches them.

This converges with the hotspots seat's own dead end
(`blessed-notes/gpu-hotspots-notes.md`, "a per-buffer fix attempted, built,
tested, and disproven"): a scoped memory-placement change on the exact same
9 buffers (`blessedSmallCbUncachedBelow`) did not reproduce the global
`cachedDynamicResources = vir` recovery either. Two independent memory-
relocation approaches (uncaching the mapping, and now vram-mirroring the
binding) both leave this pass's gap intact. That's convergent evidence the
lever isn't these buffers' memory placement at all -- something else `a`
(today's default profile) changes app-wide is the actual cost. **closed by
the lead's in-game a/b**: `DXVK_CONFIG=d3d11.cachedDynamicResources=vir` costs
31% cpu (4.7 -> 6.6 ms busy). that lever is dead; `556a3b73`'s residual gap
has no remaining candidate this seat can test.

## `566c03f3` -- root-caused: a 4 MiB dynamic vertex buffer, not a cbuffer

traced from `bench/runs/traverse-dump-2/probe/frame-{19000,20500,22000,23500,25000}.jsonl`
(real traversal frames, perf-max + cb-mirror, `SkyrimSE_dxgi.log` confirms
`cachedDynamicResources = a`, `blessedCbMirror = True`). one draw of this
shader per frame in all five: a 6-index full-screen-ish quad into
`R16G16B16A16_FLOAT` 1920x1080 + `D24_UNORM_S8_UINT` (depth-tested, no depth
write), alpha blend (`src=SRC_ALPHA, dst=ONE`), ten SRV textures (cubemap +
several BC1/BC3 2d) -- reads like a sun/lens-glare composite. its cbuffers
(vs 0,1,2,7,8,9,10,11,12; ps 0,1,2,11,12) are the same DYNAMIC/CB-only/
host-cached shape as `556a3b73`'s and are equally ring/mirror-eligible; not
the lead here.

the `ia` block is: vertex slot 0 is a tiny (112-byte) STATIC, non-cached quad
-- irrelevant. **vertex slot 1 is the real find**: a **4,194,304-byte (4 MiB)
buffer, `usage=2` (DYNAMIC), `bind_flags=3` (`VERTEX_BUFFER | INDEX_BUFFER`),
`cpu_access=65536` (WRITE)**, confirmed identically across all five frames
(different D3D11Buffer instances each time -- the engine cycles a pool of
these -- but always 4 MiB, always `bind_flags=3`). Skyrim's own stats confirm
the write pattern: `probe-frames.jsonl`'s `map.nooverwrite_vb: n=12.58/frame`
-- this is a streaming ring the engine appends into with `WRITE_NO_OVERWRITE`,
not a per-frame discard.

**why the mirror doesn't reach it:** `D3D11Buffer::m_blessedCbRing`
(`d3d11_buffer.cpp:134-140`) requires `m_desc.BindFlags == D3D11_BIND_CONSTANT_BUFFER`
exactly -- the cb-ring, and therefore `blessedCbMirror`, only ever look at
constant buffers. This 4 MiB buffer's bind flags (`VERTEX_BUFFER|INDEX_BUFFER`)
never qualify, mirror on or off. And under the default `SkyrimSE.exe` profile
(`cachedDynamicResources = a` = `~0u`, every bind flag), `D3D11Buffer::GetMemoryFlags()`
(`d3d11_buffer.cpp:391-397`) forces it into `HOST_CACHED` system memory --
the exact same "gpu reads a dynamic buffer from cached system memory" cost
class the cb-mirror exists to fix for constant buffers, just on the vertex-
buffer side, where nothing mirrors it. That is a complete, evidence-backed
answer to "why does it still run ~2x native even with every cbuffer
mirrored": the buffer actually costing this pass isn't a cbuffer.

one more thing worth noting for the record, not a lead: without the override,
`GetMemoryFlags()`'s own `D3D11_USAGE_DYNAMIC` case (line 376-382) *already*
prefers `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` (resizable-bar) for any
dynamic buffer with bind flags -- it's only the app profile's `useCached`
override that pushes it back into host-cached memory. So unlike the cbuffer
case, fixing this doesn't need a mirror/copy step at all: it needs to *not
override* dxvk's own default for this bind-flag class.

### built: `d3d11.blessedVbRebar` (off by default)

`src/d3d11/d3d11_buffer.cpp` (+13 lines, `GetMemoryFlags()`): when
`d3d11.blessedVbRebar` is on and a DYNAMIC buffer's bind flags are
`VERTEX_BUFFER` and/or `INDEX_BUFFER` **only** (never also `CONSTANT_BUFFER`,
`SHADER_RESOURCE`, or anything else -- the exact shape traced above, nothing
broader), it opts back out of the `cachedDynamicResources` cached-memory
override and keeps dxvk's own DYNAMIC default (device-local, host-visible).
`src/d3d11/d3d11_options.h` (+9), `d3d11_options.cpp` (+1) add the option,
same pattern as `blessedCbRing`/`blessedCbMirror`.

This is deliberately *not* shaped like the cb-mirror: no ring, no per-
submission copy, no barrier plumbing. It just changes which memory type the
buffer's one persistent allocation is created in, so it's unaffected by
whether the game maps it with DISCARD or NO_OVERWRITE (confirmed this buffer
uses NO_OVERWRITE above). Costs one predictable branch, at buffer creation
only -- `GetMemoryFlags()` is called once per `D3D11Buffer` constructor, never
per-frame.

**why this is a real bet, not free**: the backlog's rebar-ring dead end
(`bis-rebar`, -30% fps) is about a *different* write pattern -- ~13k small,
scattered, individually-locked cbuffer stores a frame, where each locked
instruction after a write-combined store waits ~200 ns to drain over pcie
(`scratchpad/perf-halfrate/wcbench`). A 4 MiB dynamic vb filled by
`WRITE_NO_OVERWRITE` append is normally a handful of larger, more sequential
writes -- the workload write-combined/device-local memory is actually built
for. But I have no in-game measurement that this specific engine buffer's
write pattern is in fact sequential rather than many small scattered stores,
so **owed to the lead: an in-game a/b of `d3d11.blessedVbRebar=True`**,
watching both cpu (in case it turns out to be scattered writes after all)
and gpu frame time, same shape as the `vir` test already run.

**owed to the lead, second item:** the same buffer class (DYNAMIC, VB/IB-only
bind flags) almost certainly isn't unique to `566c03f3` -- any other pass
using the engine's shared streaming geometry buffer would see the same
placement, and `blessedVbRebar` covers all of them at once, not just this
shader. Worth checking the traversal capture's other per-pass deltas against
this once the a/b is in.

## 2. water reflection cube `79c17b07` -- shape confirmed, no safe fix found

`frame-1920-resources.jsonl`: the cube target is 512x512, `array=6`, `mips=1`
(`misc_flags=4` = TEXTURECUBE, `bind_flags=40` = RENDER_TARGET|SHADER_RESOURCE).
one mip level rules out the brief's "mip generation" guess outright.

tracing the command stream around the pass (`i=148..155`): each face's 35-245
draws end with a `clear_rtv` + `clear_dsv` immediately before the next face's
draws begin, all three (rtv, dsv, and the *other* faces) sharing the same two
underlying resource pointers -- `dumpview.py`'s pass-splitting keys off
resource id, not array layer, so it correctly reads this as one clear-then-
draw pass per face, six times a frame (or a rolling subset -- this dump only
shows 2 of the loop's iterations at the frame's start). native and dxvk issue
the identical D3D11 call sequence here; the gap is not extra api traffic on
our side, it's per-face fixed overhead (render pass begin/end, framebuffer
lookup, and the layout transition round-trip: the face just rendered must
move color-attachment -> shader-read for later sampling, the next face
shader-read/undefined -> color-attachment) landing harder on dxvk than on
native's driver because the actual per-face work (512x512, tens of draws) is
small enough that the fixed cost dominates.

that is a real, structural explanation, but not a fix I'm shipping blind:
the actual lever (avoiding a full transition round-trip per face, e.g. by
keeping these targets in `VK_IMAGE_LAYOUT_GENERAL` across the read/write
switch, or reusing a cached render pass/framebuffer object across faces) is
a change to dxvk's general render-target binding path
(`src/dxvk/dxvk_context.cpp`, the `OMSetRenderTargets` bind path and its
layout-transition bookkeeping), shared by every render target in the engine.
I have no way to verify it helps this pass without regressing something else
without an in-game capture, and the brief is explicit that this seat can't
launch Skyrim. Flagging rather than guessing.

**owed to the lead:** a `BLESSED_GPU_PASSES` capture with a `VK_LAYER_KHRONOS_validation`
or gpu-trace pass over just this target (six frames, one per face) would show
whether the cost is the transition or the render-pass/framebuffer lookup --
worth knowing before touching the general bind path.

### update: the exact between-faces mechanism, traced in code

per the lead's ask, here is precisely what dxvk does on every face switch,
function by function (`src/dxvk/dxvk_context.cpp`):

1. **`clearRenderTarget`** (line 374): the game's `ClearRenderTargetView`/
   `ClearDepthStencilView` for the next face never actually clears anything
   here. Since the new face's view isn't the current framebuffer's attachment
   (`m_state.om.framebufferInfo.isFullSize`/`findAttachment` miss -- it's a
   *different* array-layer view of the same image), it calls
   `endCurrentPass(true)` (line 407) and then just records the clear as
   pending (`deferClear`, line 423). **the clear is already folded into the
   next render pass's `LOAD_OP_CLEAR`** when that pass is built --
   confirmed by the function's own comment ("2) The clear gets folded into
   render pass ops") -- so "defer the clears into loadOp" is not a lever
   here, dxvk already does it.
2. **`updateRenderTargets`** (line 7922), run lazily before the next draw:
   detects the attachment set changed, calls `endCurrentPass(true)` again
   (harmless, already ended) and `resetRenderPassOps` (line 7939) to compute
   the new pass's load/store ops from the deferred clear above.
3. **`renderPassBindFramebuffer` -> `acquireRenderTargets`** (lines
   6666/6549): for each attachment, queries its *actual current* Vulkan
   layout (`image()->queryLayout`, line 6613) and, if it doesn't already
   match `COLOR_ATTACHMENT_OPTIMAL`/`DEPTH_STENCIL_ATTACHMENT_OPTIMAL`,
   prepares a transition. Then **`prepareOutOfOrderTransition`** (line
   11018) decides which command buffer that transition can go on: it's
   allowed onto the cheap, out-of-order init buffer only if
   `!image.isTracked(m_trackingId, DxvkAccess::Write)` -- i.e. only if this
   `DxvkImage` hasn't already been written this command list.

   **this is the actual per-face cost**: `isTracked` is per-*image*, not
   per-subresource. Face 1's render marks the whole cube image as tracked-
   written for the rest of this command list, so faces 2 through 6 -- even
   though each touches a disjoint array layer -- all fail
   `prepareOutOfOrderTransition` and get routed onto the slower, in-order
   exec-buffer barrier path (line 6623-6624), with a `flushBarriers()`
   (line 6632) forcing those barriers to actually be walked before the pass
   begins. Native's driver has no equivalent "one bit per whole resource"
   tracking forcing serialization between non-overlapping subresources.
4. a fresh `vkCmdBeginRenderPass`/`EndRenderPass` (or the framebuffer/pass
   object lookup that backs it) per face, on top of (3).

So of the brief's three guesses, one is already handled by dxvk
(clears->loadOp) and mip generation doesn't apply (mips=1, confirmed
earlier); the real cost is **(3): DXVK's write-tracking is coarser than the
game's access pattern**, forcing every face after the first through
synchronized barrier handling it doesn't actually need, plus the fixed
per-face render-pass-object cost from (4).

**still not shipping a fix**: `m_trackingId`/`isTracked` is the core of
dxvk's whole hazard-tracking scheme, used by every resource in the engine,
not something scoped to this one cube. Moving to per-subresource tracking
(or special-casing cube-array attachments to bypass it) is a correctness-
critical change to shared machinery that a build seat with no in-game
testing has no business making blind -- a missed barrier here is corruption
or a driver crash, not a soft perf regression. Flagging the mechanism, not
guessing at a patch.

## 3. streaming uploads -- already on the transfer queue upstream

checked `DxvkContext::uploadImageHw` (`src/dxvk/dxvk_context.cpp:6376`): it
already issues its layout transition and `copyImageBufferData` calls against
`DxvkCmdBuffer::SdmaBuffer` -- the dedicated transfer queue, when the device
has one -- which is stock dxvk 3.1.1 behaviour, not something this fork is
missing. `uploadImageFb` (the fixed-function fallback for formats without
hardware transfer support) runs on the graphics queue by necessity: it's a
real draw, not a copy. I found nothing to move here without traversal-scene
data showing a specific upload actually landing on the graphics queue that
shouldn't be -- `docs/tuning-report.md`'s per-pass gpu numbers don't currently
break out upload/init work as its own pass, so there's nothing in hand to
target. Not attempted further; same "need a trace, not a guess" situation as
item 2.

## what's owed to the lead now

- **`d3d11.blessedVbRebar=True`**, in-game a/b (cpu + gpu frame time), same
  shape as the `vir` test: does keeping the 4 MiB streaming vertex/index
  buffer off cached memory help `566c03f3` and anything else using the same
  engine buffer, without the scattered-write cpu regression the cbuffer case
  had.
- water cube: no code to test. a gpu-passes capture bracketing just the
  6-face loop would confirm the barrier-serialization mechanism traced above
  before anyone considers touching `m_trackingId`'s per-image granularity --
  that's a much bigger, riskier change than this seat should propose blind.
- item 3 (streaming uploads): still nothing found; would need traversal-scene
  data showing a specific upload on the wrong queue to have anything to
  target.
