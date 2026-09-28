# vol-period seat notes (2026-09-27)

task: generalize the vanilla-volumetrics half-rate cadence
(`blessed_vanilla_halfrate.cpp`) from a hardcoded period of 2 to
`BLESSED_VOL_PERIOD=<n>` (default 2 = today's behaviour exactly), for
round-seven's item pricing a period of 3 at +0.06-0.08 ms over strict, on
top of half-rate's existing 0.18-0.20 ms save.

## the existing cadence (period=2, unchanged behaviour)

`lastSkipped` / `runsSinceShift` only ever allowed one skip in a row
(`AllowsSkip() = !lastSkipped && runsSinceShift < 8`), and every 8 runs
that followed a skip, forced a second run back-to-back -- that's the
"shift": it walks the skip/run parity by one frame relative to the
engine's assumed 8-step jitter cycle, so a fixed period-2 sample doesn't
get stuck always landing on the same 4 of 8 jitter phases (gcd(2,8)=2).

## the generalization

`Cadence(period)` replaces the bool with `skipStreak` (skips taken since
the last run, capped at `period - 1`) and keeps `lapsSinceShift`, counting
only the runs that used up the full allowance ("on schedule"). Same
8-lap shift trigger, independent of `period` -- it's tied to the engine's
jitter cycle length, not to how many frames we skip between runs.

```
AllowsSkip() = skipStreak < period - 1 && lapsSinceShift < 8
Record(skip):
  skip  -> skipStreak++
  run   -> onSchedule = (skipStreak == period - 1)
           skipStreak = 0
           lapsSinceShift = onSchedule ? lapsSinceShift + 1 : 0
```

Verified in `scratchpad/vol-period/sim_equiv.py`: with `period=2` this
produces the byte-identical skip/run sequence as the old
`lastSkipped`/`runsSinceShift` pair over 2000 simulated frames (0
mismatches) -- the default is an exact behaviour match, not just
"close enough."

## jitter coverage proof (`scratchpad/vol-period/sim_cadence.py`)

Worst case: no motion/target forcing at all (pure cadence, the case with
the *least* opportunity to resample a missed phase), frame's jitter phase
= `frame_index mod 8`, 400 simulated frames.

| period | max consecutive skips (limit) | run-count per jitter phase (0..7) | missing phases |
|---|---|---|---|
| 2 | 1 (limit 1) | 27,26,27,26,27,26,26,27 | none |
| 3 | 2 (limit 2) | 18,18,18,18,18,18,18,18 | none |
| 4 | 3 (limit 3) | 14,13,13,13,14,14,14,14 | none |

every period tested visits all 8 jitter phases repeatedly and evenly (no
phase starved), and the skip streak never exceeds `period - 1`, i.e. the
cadence never silently degrades into a longer skip run than asked.

first 40 decisions at `period=3` (`f`=frame, `phase`=`f mod 8`), showing
the shift landing at f=21/22 (two runs back to back -- the walk):

```
f= 0 phase=0 run     f=10 phase=2 skip    f=20 phase=4 skip    f=30 phase=6 skip
f= 1 phase=1 skip    f=11 phase=3 skip    f=21 phase=5 run     f=31 phase=7 run
f= 2 phase=2 skip    f=12 phase=4 run     f=22 phase=6 run     f=32 phase=0 skip
f= 3 phase=3 run     f=13 phase=5 skip    f=23 phase=7 skip    f=33 phase=1 skip
f= 4 phase=4 skip    f=14 phase=6 skip    f=24 phase=0 skip    f=34 phase=2 run
f= 5 phase=5 skip    f=15 phase=7 run     f=25 phase=1 run     f=35 phase=3 skip
f= 6 phase=6 run     f=16 phase=0 skip    f=26 phase=2 skip    f=36 phase=4 skip
f= 7 phase=7 skip    f=17 phase=1 skip    f=27 phase=3 skip    f=37 phase=5 run
f= 8 phase=0 skip    f=18 phase=2 run     f=28 phase=4 run     f=38 phase=6 skip
f= 9 phase=1 run     f=19 phase=3 skip    f=29 phase=5 skip    f=39 phase=7 skip
```

before the shift (f<21) runs land on phases 0,3,6,1,4,7,2 -- walking by
+3 mod 8 each run (period 3's natural stride); the shift at 21/22 inserts
an extra run with no skip gap, which is a +1 instead of the usual +3,
permanently offsetting every run afterward by 2 phases relative to the
unshifted schedule. Over enough shifts the run phase visits all 8
residues instead of only the `8/gcd(3,8) = 8` it would already visit
unshifted (period 3 is coprime with 8, so it was never actually stuck --
period 2 and period 4 are the cases that needed the shift, and both check
out clean in the table above).

## the switch

- `BLESSED_VOL_PERIOD=<n>` (unset or invalid -> 2). `n=1` disables the
  skip outright (never skips -- `period - 1 == 0`, so `AllowsSkip()` is
  always false).
- logged once at startup on the existing `vol-halfrate:` line
  (`Logger::info`, `VolConfig::Get()`), and every ~5s cycle in
  `vol_halfrate.jsonl`'s `"period"` field.
- costs nothing extra when left at the default: same struct size, same
  branch shape, one more `uint32_t` compare per decision either way.

## the log line

`vol_halfrate.jsonl` (unchanged cadence: every 120 presents / `~2-5s`
depending on framerate) already carried `runs` (generated), `skips`
(reused), and `forced_motion` per window -- this seat only added
`"period"` so a run can be told apart from another at a glance. full
window fields: `t, period, frames, runs, skips, screen_skips,
skip_share, forced_cadence, forced_target, forced_motion, forced_camera,
skipped_dispatches, skipped_draws, cam_ok, cam_bad, mean_deg, max_deg,
mean_move, iter_mask`.

## build

`build-wt.cmd E:\blessed_skyrim\wt\vol-period` -- noprobe mingw, clean,
see commit for result.

## files touched

- `src/d3d11/blessed_vanilla_halfrate.cpp` -- `VolPeriod()`, generalized
  `Cadence`, `cadence{VolPeriod()}`, `period` in both log points.
- `src/d3d11/blessed_vanilla_halfrate.h` -- doc comment only.

## opus review (2026-09-27)

- period 2 checked against the old `lastSkipped`/`runsSinceShift` code by hand, not only the sim: identical, including the first decision and the forced-early reset.
- jitter: the cycle length 8 is hardcoded (`kShiftLaps`), backed by in-game `iter_mask` 255. A superperiod is `8p+1` frames with 9 runs, and `gcd(p, 8p+1) = 1`, so every phase is reached for any engine cycle length, not only 8. Period 4 reaches only 2 phases per 33 frames, so full coverage takes about 132 frames.
- fix: level 2's screen-off gate compared against the last generate run's camera. With period 3+ a skip frame can run pass 138 after that run, so the held blurred result could be up to `maxdeg` (1 degree) off while the gate read 0.05. It now compares against `lastScreenCam`, the camera of the last frame pass 138 ran. With period 2 that is always the last run, so the default is unchanged.
- fix: the `Record` comment said a forced-early run leaves the lap count alone; the code (like the old code) restarts it.
- added `run_iter_mask` to `vol_halfrate.jsonl`: the jitter phases that got a real run in the window.
