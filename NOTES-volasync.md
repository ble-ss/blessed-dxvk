# vanilla-vol-async: findings and scope (seat `volasync`, 2026-09-25)

## step 1: why `kicks=0` in c47

`BLESSED_ASYNC`'s existing "vol" pass (`src/d3d11/blessed_volumetrics.cpp`,
`blessed_vol_detail::g_asyncKick`) only arms when `BLESSED_VOLUMETRICS=rt`
(our own ray-traced replacement of pass 138) *and* `BLESSED_ASYNC=1` *and*
`vol` is in `BLESSED_ASYNC_PASSES`. c47/c51 (`bench/plans/unmeasured-levers-2.json`)
set `BLESSED_VOL_COLLAPSE=1` only, never `BLESSED_VOLUMETRICS=rt` -- so the
kick hook (`BlessedVolumetrics::OnKickDraw`, gated on `cfg.mode != VolMode::Rt`)
never ran. This is a different, unrelated feature from vanilla's own
generate/chain/blur compute the audit's item e is about; nothing was broken,
the flag was just never armed. No fix needed there; `BLESSED_VOL_ASYNC` below
is the actual lever for vanilla's compute.

## step 1: the exact producer/consumer graph (whiterun dump, frame 1920)

`python bench/dumpview.py bench/runs/whiterun-dump-1/probe/frame-1920.jsonl`,
cross-referenced with the raw `frame-1920.jsonl` records:

- **generate** (`cs.ab674eb1`, dispatch i=6066, pass ~17): `cs_srv` slot 0 =
  4096x4096 tex2d (the sun cascade atlas), slot 1 = 512x512 tex2d, slot 2 =
  a 4096x1 tex1d LUT, slot 3 = a 32x32 tex3d (static noise). `cs_uav` slots
  0/1 = the two 320x192x90 R16F froxel volumes. Runs immediately after the
  cascade draws (passes 7-15, last i=6065) with **zero draws** between them
  and the chain that follows.
- **chain** (`cs.1c4ebb62`, 90 dispatches un-collapsed, i=6068..~6250, passes
  18-107): ping-pongs the two froxel volumes. `BLESSED_VOL_COLLAPSE=1`
  replaces all 90 with one dispatch of its own shader, still inside the same
  dispatch-only stretch.
- Passes 17-107 in the dump are *all* `compute/copy (guess)`, draws=0: no
  draw call happens between generate and the depth prepass (pass 108,
  i=6252). This is a much longer window than the audit's "~1.1 ms of raster"
  figure named (that number only counted the three big named passes --
  prepass/mask/lit -- as a conservative estimate); the real producer-to-
  first-consumer gap is generate's own dispatch to pass 138 (i=12323), i.e.
  the *entire* rest of the opaque+lighting frame. More overlap room than
  assumed, not less.
- **pass 138** (`fs.c480e36e`, i=12323, `draw_indexed`): `ps_srv` slot 2 =
  the same froxel volume resource pointer the generate/chain wrote
  (`0x2805d96e960`). This is the read this seat closes the window before.
- **the blur pair is downstream of pass 138, not upstream**: `cf1e1a21`
  (i=12326) reads pass 138's *own render target* (`rtv[0]` of i=12323, same
  pointer `0x28013c223a0`) plus an aux tex2d, writing a scratch buffer;
  `dbb99d2e` (i=12328) reads that scratch buffer, writing
  `0x28013c22b00` -- which is exactly pass 138's `ps_srv` slot 1 (its own
  *previous-frame* blurred history). So the real chain is: pass 138 draws
  raw output -> blur1 -> blur2 -> next frame's pass 138 reads the result.
  The blur's producer-to-consumer window is therefore *this frame's pass
  138 to next frame's pass 138*, nearly a whole frame already. **Moving the
  blur pair is not the same problem as moving generate/chain**, and this
  seat did not move it -- see "not done" below.

## what got built: `BLESSED_VOL_ASYNC=1`

New, independent switch (off by default), separate from `BLESSED_ASYNC`/
`BLESSED_ASYNC_PASSES` (which remain the traced gi/vol-rt passes' own
switches). Setting it also requests the async queue and *forces*
`BLESSED_ASYNC_QUEUE=graphics` (a second queue in the graphics family, not
the dedicated compute-only family) -- see the why below. Files:

- `src/d3d11/blessed_vol_async.h/.cpp` (new): app-thread hook. Opens the
  async-queue recording window at the generate dispatch (shader-hash match,
  `BLESSED_VOL_ASYNC_GEN_CS`, default `ab674eb1`); closes it at the very
  next draw call, on *every* draw entry point (`Draw`, `DrawIndexed`,
  `DrawInstanced`, `DrawIndexedInstanced`) -- unconditionally, before
  anything else in each of those functions runs, so a draw can never slip
  into the async command buffer. Waits (`blessedAsyncSync`) specifically at
  the draw whose pixel shader matches `BLESSED_VOL_ASYNC_WAIT_PS` (default
  `c480e36e`, pass 138).
- `src/dxvk/blessed/blessed_vol_async.cpp` (new) + two new `DxvkContext`
  methods (`blessedVolAsyncBegin/End`, declared in `dxvk_context.h`): a
  same-family-only wrapper over the existing `blessedAsyncBegin/End`
  (`src/dxvk/blessed/blessed_async.cpp`). Refuses (logs once, stays on
  graphics) if the picked async queue turns out to be a different family.
- `src/dxvk/blessed/blessed_async.h/.cpp`: added `BlessedAsync::
  VanillaVolRequested()`; `IsRequested()`/`WantsGraphicsFamily()` now also
  true when `BLESSED_VOL_ASYNC=1`.
- `src/d3d11/d3d11_context.h`: `friend class BlessedVolAsync;` (needs
  `EmitCs`, same reasoning as `BlessedVolumetrics`/`BlessedAo`/etc already
  there).
- `src/d3d11/d3d11_context.cpp`: the hook wired into `Dispatch()` (after the
  half-rate/skip-volumetrics early-returns, so a skipped frame never opens
  the window for a dispatch that will not run) and into all four draw
  entry points (right after each one's zero-count early return).

Does not need to know whether `BLESSED_VOL_COLLAPSE` is on: the window
just spans however many dispatches happen (1 collapsed, 90 raw, or up to
90 during collapse's own learning phase) -- all equally safe to move,
since nothing but dispatches ever happen in that stretch (see the dump
evidence above). Composes with vol-collapse with no changes to
`blessed_vol_collapse.cpp` at all.

## why the graphics-family queue, not the dedicated compute family

The audit's framing (and the existing `BLESSED_ASYNC`/gi+rt-vol design,
`docs/research/async-compute.md`) uses a genuinely separate compute-only
queue family, which needs every image it touches to either be pass-owned
and tagged `blessedConcurrent` (dxvk's existing opt-in: concurrent sharing
across families, set at image creation, currently only used on our own
scratch images) or to never be touched at all (copy in, copy out).

Vanilla's own generate/chain dispatches touch the game's own persistent
textures directly through the normal D3D11 resource-binding path (`ctx->
dispatch()`, not a hand-rolled compute pass like the traced gi/vol kicks):
the two froxel volumes, and -- as an SRV, but still requiring a queue-family
ownership transfer per the Vulkan spec if the family actually differs -- the
4096x4096 sun cascade atlas, a resource read by many other passes every
frame. Retagging any of these `blessedConcurrent` needs a new hook in
`d3d11_texture.cpp`'s texture creation (shape/format matching, since the
generate dispatch's own inputs are the only place that would tell us which
resource needs it, but the flag has to be set before the image is ever used);
doing that blind, with no game or GPU access to verify sync, was judged too
large a correctness risk for this seat under "correctness first". A
same-family second queue needs **no** ownership transfer at all (Vulkan:
an exclusive-mode resource may be used from any queue in the same family
without a transfer) -- vanilla's dispatches then run completely unmodified,
same bits, at the cost of losing the guarantee that the two queues actually
execute concurrently on this hardware (nvidia may time-slice two
graphics-family queues instead of overlapping them -- already a documented
open question for the existing gi/vol-rt async work's own `a` in its a/b).
`BLESSED_ASYNC_TIMING=1`'s `kick_ms`/`stall_ms` already covers this kick for
free (it counts every `blessedAsyncBegin/End` pair, not just the gi/vol-rt
ones), so the in-game a/b answers this empirically.

## not done (deliberately, for the lead to pick up)

- **the blur pair** (`cf1e1a21`/`dbb99d2e`): left on graphics. Their real
  producer-consumer window is this-frame's-pass-138 to next-frame's-
  pass-138 (see above) -- a different, much larger-margin problem, and
  moving them would need a *second* independent kick/wait pair (their own
  `m_blessedVolAsyncActive`-style bookkeeping spanning a frame boundary),
  which risks interacting badly with this window's draw-boundary-close
  invariant if built carelessly. Worth a follow-up seat once this one is
  proven safe in game.
- **the dedicated compute-only family**: would need the `blessedConcurrent`
  tagging work above (or real queue-ownership-transfer barriers) before it
  is safe for vanilla's own textures. Flagged, not attempted here.

## what the lead must check in game

- `BLESSED_VOL_ASYNC=1 BLESSED_GPU_PASSES=1 BLESSED_ASYNC_TIMING=1`, alone
  first (no `BLESSED_ASYNC`, no gi/vol-rt passes), against a plain
  `BLESSED_VOL_COLLAPSE=1` baseline: image must stay bit-identical (nothing
  here changes any dispatch's shader, inputs, or outputs, only which queue
  records it -- vol-collapse's own `BLESSED_VOL_COLLAPSE_VERIFY=1` should
  still read clean 8/8). Watch `blessed: async: ... kicks=N ... hidden~X ms`
  in the log / `async.jsonl`: `kicks` should be ~1/frame, `stall_ms` should
  be well under `kick_ms` (the whole point).
- `tools/vvl` sync validation on at least one capture with this flag on:
  confirm no `SYNC-HAZARD` around the two async-queue submissions, and that
  the second graphics-family queue was actually created (log line
  `blessed: async: compute queue family N index M`).
- then the combined case: `BLESSED_VOL_ASYNC=1` together with `BLESSED_ASYNC=1
  BLESSED_ASYNC_PASSES=gi,vol` (the existing traced passes) -- since both
  now force the same graphics-family queue and share the single
  `blessedAsyncSync()`/pending-kick bookkeeping in `DxvkContext`, a wait at
  pass 138 could also end up waiting on an outstanding gi/vol-rt kick if one
  is still in flight. Not a correctness bug (waiting for more than strictly
  necessary is safe), but worth an in-game timing check to see if it costs
  anything real.

## round two, step 1: the mirror readiness gap (audit finding 5), fixed

Confirmed the race the audit named, from the code, not just plausibility:
`blessedFlushCbMirror()` (`blessed_cb_mirror.cpp`) only runs once, inside
`flushCommandList()`, and attaches whatever it flushes to whatever chunk
is "current" at that moment. `blessedAsyncBegin()`/`blessedVolAsyncBegin()`
(`blessed_async.cpp`, `blessed_vol_async.cpp`) commit the producer chunk
via `splitCommands()`/`m_cmd->blessedAsyncBegin()`'s own `next()` *before*
any of the window's dispatches record; `DxvkCommandList::submit()`
(`dxvk_cmdlist.cpp:283-297`) submits a kick strictly *before* processing
the chunk whose index equals the kick's own `beforeChunk` -- and that is
exactly the chunk any CB rename recorded *during* the window (the
generate/chain dispatches' own constants) lands in, since only
`ExecBuffer` gets swapped to the async pool, not `SdmaBuffer`. So: a
rename from earlier in the frame stayed pending past the kick (nothing
flushed it before `flushCommandList`'s once-per-list call), and a rename
*during* the window had no earlier point to flush into at all. Either
way the async dispatches could read a stale mirror copy of a constant
buffer they themselves just renamed.

Fixed both halves (`src/dxvk/blessed/blessed_vol_async.cpp`,
`src/dxvk/blessed/blessed_cb_ring.cpp`): `blessedVolAsyncBegin()` now
flushes the mirror before splitting (attaches pending copies to the
still-current producer chunk, whose own per-chunk transfer-to-graphics
wait then covers the kick); `blessedRenameBuffers()` keeps a rename on
the host block instead of the mirror while `m_blessedVolAsyncActive` is
set, the same fallback `blessedCpuRead` already uses. Verified by
reading the exact submit-order code path above, not by running the game
(cannot). `BLESSED_VOL_ASYNC_VERIFY=1` (new, `blessed_vol_async_verify.*`)
gives the lead a bitwise, in-game way to actually catch a regression of
this if the reasoning above is wrong somewhere: it shadow-replays the
whole window (generate, then whatever chain shape is running) through
ordinary synchronous dispatches onto scratch, then compares against the
real (possibly async) result at the wait point, every
`BLESSED_VOL_ASYNC_VERIFY_PERIOD`-th window (default 60), logging to
`shader-verify.jsonl` as `vol-async-generate` and `vol-async-final`.
Not yet run in game -- see "what the lead must check", below.

## round two, step 2: the blur pair's consumers, settled from the dump

Both audits partially right, about different readers of the same
resource. `python -c` over `bench/runs/whiterun-dump-1/probe/frame-1920
.jsonl`, every record whose `ps_srv`/`cs_srv` references
`0x28013c22b00` (dispatch 12328 `dbb99d2e`'s uav0 output):

```
(12323, 'draw_indexed', 'ps_srv', 1, 'fs.c480e36e...')   # pass 138, slot 1
(12377, 'draw_indexed', 'ps_srv', 0, 'fs.61eac670...')   # a later pass, slot 0
```

`i=12323 < 12328 < 12377`, all in the same frame's dump. So: pass 138
(`c480e36e`) reads this resource at slot 1 *before* this frame's blur
dispatch runs -- that is last frame's blur output, its history input,
exactly as `NOTES` already said. Draw 12377 (`fs.61eac670`, a different
pass, *not* pass 138) reads it at slot 0 right *after* this frame's blur,
same frame -- a genuine same-frame consumer, as astra said. Fable's "next
frame's pass 138" and astra's "draw 12377, same frame" are both real,
for two different readers of the same pointer across time: the
resource is double-duty (this frame's fresh blur output for 12377, next
frame's history input for pass 138).

Settled: astra's constraint holds. **If the blur pair is ever moved to
the async queue, the window can only stay open up to draw 12377** (49
events after the blur, not "nearly a whole frame" -- draw 12377 is
close behind). Fable's 0.08-0.10 ms estimate for the blur pair assumed
a whole-frame window and should be treated as an upper bound, not the
expected value, if that work is picked up. This seat left the blur pair
alone, per the brief; this only sharpens the note for whoever picks it
up next.

## round two, step 3: the dedicated compute-family path (=2) -- designed, not built

Chose queue-ownership-transfer barriers over `VK_SHARING_MODE_CONCURRENT`
(`blessedConcurrent`), and did not build either. Why the choice, and why
not shipped:

**Why ownership transfers, not `blessedConcurrent`:** concurrent sharing
has to be decided at image *creation*, and these are vanilla's own
persistent textures, created through the ordinary D3D11 resource path
long before any of this fork's code knows the async window will ever
touch them. Tagging them concurrent would mean a new hook in
`d3d11_texture.cpp`'s texture creation matching on shape/format (4096x4096,
320x192x90 R16F, etc.) to *guess* which resources need it -- fragile, and
wrong for any other texture that happens to share a shape. Ownership
transfers instead operate on already-created exclusive-sharing-mode
resources at the point of use (a release barrier on the queue handing
off, an acquire on the queue picking up), so nothing about resource
creation has to change or guess ahead of time.

**The design, full scope:** generate on the dedicated family needs a
release (graphics, in the producer chunk, right before the kick) and
acquire (compute family, at the start of the kick's own command buffer)
for three resources: the two froxel volumes and the 4096x4096 sun
cascade atlas (generate's own `cs_srv` slot 0). After the chain's last
dispatch, the reverse: release (compute family, end of the kick) and
acquire (graphics, in the chunk after `blessedAsyncSync()`'s wait,
before pass 138's draw) for the two froxel volumes (the atlas is
read-only here, no write-back needed, but its *read* still needs the
initial acquire before generate touches it).

**A smaller variant considered:** since only *generate* reads the atlas
(the chain's 90 dispatches, or the collapsed one, only ping-pong the two
froxel volumes, no atlas dependency -- `blessed_vol_collapse.cpp`'s own
`Step`/`Classify` model confirms this), a narrower `=2` could leave
generate on the graphics-family second queue (today's `=1`, already
judged safe) and move only the *chain* to the dedicated family, with
ownership transfers on the two froxel volumes alone -- never touching
the atlas's ownership at all. This bounds the resources needing transfer
tracking to two, with a known, small, enumerated set of other consumers
(pass 138's draw, and the lens-flare draw `69c92902` per
`blessed_vol_collapse.h`'s own comment) rather than the atlas's "many
other passes every frame."

**Why not built, either way:** a wrong wait or a missed transfer in this
class of bug is corruption or a device loss (fable's audit item a's own
risk line), and this seat cannot launch the game or attach a gpu to
check either variant landed correctly -- the exact situation the brief's
"say so plainly rather than shipping something unverified" anticipates.
The full-scope version's blast radius (the cascade atlas, read by many
unrelated passes across the whole frame) would need ownership-state
tracking well beyond this window, a much bigger change than "wrap the
vol-async window" and squarely out of a single medium-effort seat's
verifiable scope. The narrower, chain-only variant is smaller and more
tractable, and is the one worth a follow-up seat's time -- but even its
two-resource version still needs a real in-game (or at minimum, gpu-
attached synthetic replay) check before it can be trusted, which this
seat does not have. Flagged rather than attempted; this matches the
prior seat's own conclusion in the "not done" section above, now with a
concrete design for whoever picks it up.

## round two: validation and synthetic timing -- not done, said plainly

The brief's steps 4-5 (a synthetic replay of the real sequence under
`tools/vvl` with `validate_sync = true` for `=1` and `=2`, a negative
control, and timed sync-vs-`=1`-vs-`=2` numbers) were not built. `=2`
does not exist (see above), so there is nothing to validate or time on
that side. For `=1`: no vol-async-shaped synthetic harness exists yet
(`blessed-tests/layertest` is `perlayer`'s, a different feature); building
one -- matching shader hash lookup via `BLESSED_VOL_ASYNC_GEN_CS`/
`_WAIT_PS`, two persistent 3d textures, a generate/chain/consumer shape
close enough to be a fair proxy, wired through `tools/vvl` and a
negative control (dropping the wait) -- is itself a real build task, and
budget for this seat went to the correctness fix (step 1, the brief's
own "both audits' top lever") and the settled blur question (step 2)
instead. The readiness fix above is checked by reading the exact
submit-order code path (cited line by line), not by a running proof;
that is real evidence, but it is not the bitwise/vvl-clean proof the
brief asked for, and it should not be reported as one.

**What the lead should actually run, in order:**
1. `BLESSED_VOL_ASYNC=1 BLESSED_GPU_PASSES=1 BLESSED_ASYNC_TIMING=1` vs
   plain `BLESSED_VOL_COLLAPSE=1`, whiterun, image bit-identical
   (`BLESSED_VOL_COLLAPSE_VERIFY=1` clean 8/8), `kicks` ~1/frame,
   `stall_ms` well under `kick_ms`.
2. The same run with `BLESSED_VOL_ASYNC_VERIFY=1` added (default period
   60): watch `shader-verify.jsonl` for `vol-async-generate` and
   `vol-async-final` entries, zero mismatches, and check the dxvk log for
   the one-time "coverage broken" warning (should never fire; if it
   does, the shape assumption in this file is wrong for that scene and
   the fix above needs re-checking against whatever it actually saw).
3. `tools/vvl` (`validate_sync = true`) over a capture with `BLESSED_VOL_
   ASYNC=1`: zero `SYNC-HAZARD` around the async-queue submissions --
   this is the check that would have caught the mirror race directly if
   the fix were wrong, and it has not been run yet.
4. Only after 1-3 are clean: whiterun and traversal fps, `BLESSED_VOL_
   ASYNC=1` vs `BLESSED_VOL_COLLAPSE=1`-only, 3 repeats each, reading
   whole-frame `fps`/`gaps.frame_ms`, not `kick_ms - stall_ms`.

## round three (opus seat, 2026-09-26): the harness, what it found, =2, validation, timing

### read this first

**Every in-game number taken with `BLESSED_VOL_ASYNC=1` or `BLESSED_ASYNC=1` before `15b4ed5f` is invalid.** `DxvkCommandList::next()` (upstream dxvk 3.1.1, `b1a1c99a`) replaced the ended exec buffer with a fresh one before pushing the chunk, so the chunk carried the fresh, empty buffer: the recorded commands were never submitted, and the fresh buffer was submitted twice. Upstream only splits for sparse binds, so it is latent there; `blessedAsyncBegin` splits every frame. For vol-async that dropped everything recorded since the last flush before the generate dispatch, cascades included, every frame. vvl on the harness: `VUID-vkQueueSubmit2-commandBuffer-03875`, `UNASSIGNED-DrawState-CommandBufferSingleSubmitViolation`, timestamps never written, the cascade atlas left `UNDEFINED`. Fixed in `next()`: push first, then replace.

### the harness: `blessed-tests/volasynctest` (`deffc8f7`)

`volasynctest.cpp` replays the shape of whiterun-dump-1 frame 1920, passes 7-138:

- four cascade draws into a 4096x4096 d16 atlas (R16_TYPELESS, sampled as R16_UNORM, like the dump's generate srv 0)
- generate: srv t0-t3 = atlas, 512x512, 4096 lut, 32^3 noise; uav u0/u1 = two 320x192x90 R16_FLOAT volumes; 10x6x90 groups
- a 90-step ping-pong chain: srv t0 = one volume, uav u0 = the other, 10x6x1 groups, a WRITE_DISCARD map before each step (the dump's pattern)
- a default-usage cbuffer written with UpdateSubresource inside the window (vol-collapse's params buffer does this in game)
- a depth clear still inside the window (the dump's `clear_dsv`, i=6251), then raster that touches neither volume
- the consumer: ps srv t2 = the final volume (generate's uav 0, as pass 138)

Generate's constants are mapped at the top of the frame, so their cb-mirror copy is pending when the window opens (the readiness fix's case); the chain's are mapped inside the window (the host-block fallback's case). `run.sh` writes the game's own cbuffer lines into `dxvk.conf` (`d3d11.cachedDynamicResources = c`, `blessedCbRing`, `blessedCbMirror`, `blessedThreadedFrontEnd`) and points `BLESSED_VOL_ASYNC_GEN_CS`/`_WAIT_PS` at the test's own shaders.

Every input depends only on the frame index, so two per-frame hashes must match across sync, =1 and =2: `chk` (the final volume, read back at the end of the frame) and `cchk` (a draw right after the consumer that folds every slice of the final volume into a 320x192 target: what the graphics queue sees right after the wait).

`run.sh <outdir> <sync|1|2> <vvl|novvl> <verify|noverify> <seconds> <maxFrames> <steps> <rasterDraws> <rasterIters> <checkEvery> [uploadMB]`. Env: `BUILD`, `TIMING=1`, `FE=0`, `PERIOD`, `DRYRUN=1`, `DESC=sets`. Two of these matter for reading results:

- `DESC=sets` forces dxvk's descriptor-set binding model. With the descriptor heap (this machine's default, and the game's), vvl's sync validation cannot see what a shader reads or writes through a descriptor; it sees copies, clears, attachments, barriers and ownership only.
- `uploadMB` adds a large upload late in the frame. dxvk puts it on the transfer queue ahead of the cb-mirror copies recorded at the flush. Without it the transfer queue is idle, a late mirror copy still beats the kick, and a missing readiness flush is invisible (see the controls).

### what else the harness found, all fixed (`15b4ed5f`)

1. **the verify mode could not catch the readiness race, and in game it never compared at all.** Its shadow dispatches were recorded inside the window, right after the real ones, so they ran on the same async queue and read the same mirrored constants: a stale mirror would agree with itself. It abandoned any window whose closing draw was not the wait draw; in game the window closes at the depth prepass (i=6252), 6000 events before pass 138, so every check was abandoned. And it remapped uavs but not srvs (the chain reads one volume as an srv). Rebuilt, see below.
2. **a transfer inside the window ran after the kick that reads it.** dxvk places buffer updates, uploads and image inits out of order, in the init or sdma buffer of the current graphics chunk; inside the window that chunk is submitted after the kick. Fixed in `prepareOutOfOrderTransfer`: while any kick records, every transfer stays in order, in the kick. The harness cannot exercise this one: a default-usage cbuffer is host-visible in dxvk, so its UpdateSubresource is a rename plus a cpu copy, no transfer (vol-collapse's params buffer takes the same path). Kept as a guard; its control is not caught.
3. **deferred clears could be flushed into the kick.** A dispatch ends the render pass through `prepareShaderReadableImages`, which flushes deferred clears into the current exec buffer; at the window's first dispatch that is the kick (for =2's chain, a compute-family buffer, which cannot record a clear). Fixed: `blessedVolAsyncBegin` ends the pass on graphics first. From code, not from a failing run.
4. **a flush inside the window sent the list out with the kick half-recorded** (dfe5fc91: the exec slot still held the async buffer). My first fix waited for the kick inside the flush, but that wait sits in an empty chunk that is never submitted. Now: the flush ends the kick, and the next command list waits for it first thing. A flush in the window happens on a cpu readback that must wait (vol-collapse's own `BLESSED_VOL_COLLAPSE_VERIFY` diff at the chain's end does one) or on dxvk's implicit flushes.

The log now has one line at windows 30, 300 and every 3000: `blessed: vol-async: N windows, P opened with cb mirror copies pending (flushed before the kick), H cb renames inside a window kept on the host block, C chains on the compute family, F windows ended early by a flush`. In the harness: 30 windows, 30 pending, 2639 host renames, 29 compute chains (=2), 0 ended early.

### the verify mode, rebuilt (`BLESSED_VOL_ASYNC_VERIFY=1`)

On a checked window each dispatch is recorded as it runs: shader, srvs, uavs, samplers, and the **host bytes** of each dynamic cbuffer (the replay-side map pointer, never the ring's vram mirror). When the window closes, whatever draw closes it, the recorded dispatches are replayed in order **on the graphics queue** onto a scratch pair: any srv or uav that referenced one of the two volumes points at its scratch twin, every dynamic cbuffer is a default-usage copy of the recorded bytes. At the wait draw the real pair is copied on graphics (after the wait); the bitwise comparison and its blocking readback run one window later, at the next window's close, so the checked frame keeps its normal submission timing (period at least 2). vol-collapse ignores the replay's dispatches, and the replayed generate does not reopen the window. Log: `BlessedShaderVerify: vol-async-final texture check n/m: 2 target(s), bit-identical` (or `k pixel(s) differ`), also in `shader-verify.jsonl`.

Two things tried and dropped, said plainly:

- **a separate generate check.** A snapshot copy of the real volumes taken inside the kick right after generate disagreed with both the replay and a graphics-only run (`BLESSED_VOL_ASYNC_DRYRUN=1`) in 13-18 of 43 windows, in patterns that look like the copy saw generate unfinished or the chain already started; two back-to-back snapshot copies agreed with each other; a full memory barrier around the copy and unbinding the uavs first changed nothing; vvl (descriptor sets) was clean. The real final volumes agreed in every window and every consumer-time hash matched sync, so this is the snapshot's problem, not the async path's, but it is **not root-caused**. The replay's generate reads its own copy of every input, so a wrong generate input on the real side shows in the final check anyway; there is no separate generate check.
- **comparing at the wait draw itself.** The blocking readback at the checked frame's own wait drained the queues mid-frame; with the late upload, a missing readiness flush corrupted 40 of 40 frames by hash while the in-frame verify saw 28 of 28 identical. With the comparison deferred one window, the same control reads 28 of 28 differ.

`BLESSED_VOL_ASYNC_DRYRUN=1`: the window, the hooks and the verify mode run, but nothing leaves the graphics queue. A verify mismatch there is the verify mode's own.

### =2: the chain on a compute-family queue

**The atlas claim, verified from the dump.** All 90 chain dispatches (`cs.1c4ebb62`) bind srv slot 0 and uav slot 0 only, and every binding is one of the two volumes (`0x2805d96e960`, `0x2805d96f7a0`). The atlas (`0x28013c20270`) is read by generate (i=6066) and by one draw, i=8111 (`fs.b070feb5`). The volumes are touched by generate, the chain, pass 138 (i=12323) and the lens flare (i=12332, `fs.69c92902`), both after the wait. In the window (i=6066-6251) there are only dispatches, maps and the one `clear_dsv`. vol-collapse's replacement dispatch binds only the same pair, as uavs.

**Design.** Generate stays where =1 puts it: kick A, on the second graphics-family queue. At the first chain dispatch, the app-thread hook checks that every image the dispatch binds is one of generate's two uav images; buffers need nothing, since =2 adds the compute family to the device's sharing list and every buffer (cbuffers, ring blocks, descriptor heaps) is created concurrent. Then `DxvkContext::blessedVolAsyncSwitch`:

1. flushes kick A's pending barriers into kick A
2. records a queue family **release** of both volumes, graphics to compute, into kick A, and ends it
3. begins kick B on the compute-only family (`BLESSED_VOL_ASYNC=2` creates that queue: `blessedPickVolComputeQueue`, family 2 on this machine) and records the matching **acquire** at its top

The chain records into kick B. At window close, kick B gets its own pending barriers, then a **release**, compute to graphics. At the wait (`blessedAsyncSync`), the chunk that waits records the graphics **acquire** into its InitBarriers, so it runs right after the semaphore wait and before any out-of-order transfer in that chunk. Kick A signals the async timeline; kick B waits for that value and signals the next; graphics waits for the latest, so it covers both.

dxvk's own barriers inside kick B name every stage an image may ever be used in, graphics stages included, which a compute queue rejects. `DxvkCommandList::blessedComputeBarrier` maps graphics shader stages to the compute shader stage and drops fixed-function graphics stages and the accesses only they perform; the graphics work they refer to never runs on that queue, and the semaphores order it. If a chain dispatch binds any other image, =2 logs once and runs as =1 from there (mid-chain: kick B ends, is waited for and handed back at once). If a window's wait never comes, the next window waits and takes the volumes back before generate writes them.

Ownership transfers rather than `VK_SHARING_MODE_CONCURRENT`: the volumes are the game's own textures, created long before anything knows the window will touch them, and a creation-time hook would have to guess them by shape.

### validation (RTX, driver as installed; `r/` under the seat scratch)

Final code (`15b4ed5f` plus the verify deferral and acquire placement in the next commit). Hashes: 12 frames (every 5th of 60) for the plain runs, 40 of 40 frames for the late-upload runs (`uploadMB=32`), compared against sync. vvl: 30 frames with `validate_sync = true`, errors / `SYNC-HAZARD` count. verify: `PERIOD=2`, 90 frames.

| row | final-volume hash `chk` | consumer-time hash `cchk` | vvl, heap | vvl, descriptor sets | verify `vol-async-final` |
|---|---|---|---|---|---|
| sync | reference | reference | 0 / 0 | 0 / 0 | n/a |
| =1 | 12/12 match | 12/12 match | 0 / 0 | 0 / 0 | 43/43 identical |
| =2 | 12/12 match | 12/12 match | 0 / 0 | 0 / 0 | 43/43 identical |
| =1, late upload | 40/40 match | 40/40 match | | 0 / 0 | 28/28 identical |
| =2, late upload | 40/40 match | 40/40 match | | | 28/28 identical |
| nc1: =1, readiness flush removed | 12/12 match | 12/12 match | 0 / 0 | 0 / 0 | 43/43 identical |
| nc1, late upload | **40/40 differ** | **40/40 differ** | | 0 / 0 | **28/28 differ** |
| nc3: =1, in-order transfer guard removed | 12/12 match | 12/12 match | 0 / 0 | 0 / 0 | 43/43 identical |
| nc2: =2, compute-side acquire dropped | 12/12 match | 12/12 match | 0 / 0 | 0 / 0 | 43/43 identical |
| nc4: =2, generate-kick release dropped | 12/12 match | 12/12 match | **10 errors** | **10 errors** | 43/43 identical |

nc4's errors are `VUID-vkQueueSubmit2-commandBuffer-03879`: "acquires ownership of VkImage ... for destination queue family 2, but no matching release operation was queued for execution from source queue family 0". The nc rows were built from the same tree with one line removed (`build.ctl`, never committed). Every row ran strictly alone on the gpu, after `quiet.sh`.

The controls, said plainly:

- **nc1 (readiness flush removed) is caught, but only with the late upload.** The readiness path is live in every window (30 of 30 opened with pending mirror copies), but with an idle transfer queue the late copy still lands before the kick reads it, so nothing sees a difference. With a 32 MB upload late in the frame, the copy lands after the kick: 40 of 40 frames differ by both hashes. vvl never flags it, in either binding model: its cross-queue sync validation does not report this read-then-write.
- **nc3 (in-order transfer guard removed) is not caught**: the harness has no GPU transfer inside the window (see fix 2 above). The harness does not exercise that path.
- **nc2 (=2, compute-side acquire dropped) is not caught** by anything: vvl 1.4.357 does not check that an image is acquired before use, and the hardware ignores ownership in practice (hashes match). The =2 ownership transfers are checked by vvl for pairing only, and by reading the code. **nc4 (=2, the generate kick's release dropped) is caught by vvl** in both binding models (`03879`, above), so vvl does check that =2's transfers pair up on these two images.

### whole-frame timing on the synthetic (`BLESSED_ASYNC_TIMING=1` on the async rows)

Frame period median of each 20 s run, gpu timestamps, 1920x1080, the chain at 90 steps. Kick and stall are `BLESSED_ASYNC_TIMING`'s per-kick averages (for =2, kick runs from kick A's start to kick B's end).

| synthetic | sync | =1 | =2 | =1 kick / stall | =2 kick / stall |
|---|---|---|---|---|---|
| 8 raster draws x 64 iters, 3 runs | 9.02 ms (9.02, 8.96, 9.09) | 9.23 ms (9.34, 8.94, 9.41) | **8.93 ms** (8.91, 8.93, 8.95) | 4.60 / 3.92 ms | 4.40 / 3.84 ms |
| 64 raster draws x 4 iters, 2 runs | 7.06 ms (7.11, 7.00) | 7.22 ms (7.06, 7.39) | **6.92 ms** (7.01, 6.83) | 4.46 / 4.02 ms | 4.20 / 3.67 ms |
| no raster, 2 runs | 4.69 ms (4.76, 4.61) | 4.84 ms (4.92, 4.75) | **4.34 ms** (4.35, 4.34) | 4.47 / 4.77 ms | 4.07 / 4.27 ms |

What it says: in this synthetic the window's own work is heavy (4.7 ms on graphics alone) and so is the raster; both saturate the gpu, so there is little idle time for a second queue to fill. =1 is no faster than sync and noisier (per row, one run matched sync and the others were 0.2-0.4 ms slower). =2 is 0.09-0.14 ms faster with raster, and 0.35 ms (7%) faster with no raster at all: the same window runs faster as a compute-queue kick than on graphics. The stall (3.8 ms) is most of the kick (4.4 ms), so `kick - stall` would claim ~0.5 ms hidden per frame while the whole frame gained 0.09; read whole-frame numbers only. The game's window is much lighter than this one, and its raster stretch has its own bubbles, so the in-game answer can differ in either direction.

### what the lead should run in game

Base: the c52 conf lines (`d3d11.blessedCbRing = True`, `d3d11.blessedThreadedFrontEnd = True`, `d3d11.blessedCbMirror = True`) plus `BLESSED_VOL_COLLAPSE=1`, whiterun save.

1. **=1 correctness**: `BLESSED_VOL_ASYNC=1 BLESSED_VOL_ASYNC_VERIFY=1 BLESSED_VOL_ASYNC_VERIFY_PERIOD=30`. Expect every `vol-async-final` line `bit-identical`, and the `blessed: vol-async:` line with windows ~= pending, 0 ended early by a flush (if that count is not 0, dxvk's implicit flushes are cutting the window: correct, but the overlap is lost for that frame).
2. **=2 correctness**: the same with `BLESSED_VOL_ASYNC=2`. Also expect `blessed: vol-async: chain queue family N index 0 (compute only)` once, and compute chains ~= windows. No `runs as =1` warning.
3. **vvl**: `BLESSED_VOL_ASYNC=2`, `tools/vvl` with `validate_sync = true`, zero `SYNC-HAZARD` and zero errors. Once more with `dxvk.enableDescriptorHeap = False` and `dxvk.enableDescriptorBuffer = False` in the conf: that is the only way vvl sees the shaders' accesses (it changes the binding model, so not for timing).
4. **timing**, only after 1-3 are clean: whiterun and traversal, 3 repeats each, interleaved: base; base + `BLESSED_VOL_ASYNC=1 BLESSED_ASYNC_TIMING=1`; base + `BLESSED_VOL_ASYNC=2 BLESSED_ASYNC_TIMING=1`. Read whole-frame `fps` and `gaps.frame_ms`. No verify, no `BLESSED_VOL_COLLAPSE_VERIFY` on timing rows (both read back inside or right after the window).

Do not combine `BLESSED_VOL_COLLAPSE_VERIFY=1` with a timing row: its diff at the chain's end is a blocking readback inside the window, which now ends the kick early (correct, but it measures the flush path).

### not sure / open

- The generate-snapshot anomaly above is not root-caused. Everything that reads the real result (final verify, per-frame hashes at frame end and right after the wait) agrees with sync in every run, and vvl is clean, so I judge it a property of copying from inside the kick, not a production hazard. The game records no copy inside the window.
- The =2 ownership transfers: vvl checks that they pair up (nc4 caught), but nothing on this machine detects an image used on the compute queue without its acquire (nc2 not caught; the hardware ignores ownership in practice). Beyond pairing, the evidence is reading the code against the spec. Another vendor's driver could behave differently where this one forgives.
- vvl's sync validation reported nothing for the readiness race (nc1) even with descriptor sets, where the hashes showed it corrupting every frame. vvl clean is necessary here, not sufficient.
- Fix 3 (deferred clears) and the flush-in-window path (fix 4) are from code reading; the harness never hit either (0 windows ended early).
- `BLESSED_VOL_HALFRATE=1` (in the c52 rows) skips generate on alternate frames; the window then opens every other frame. Not exercised in the harness.
