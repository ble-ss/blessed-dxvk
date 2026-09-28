#pragma once

#include "../util/config/config.h"

#include "../dxgi/dxgi_options.h"

#include "../dxvk/dxvk_device.h"

#include "d3d11_include.h"

namespace dxvk {

  struct D3D11Options {
    D3D11Options(const Config& config);

    /// Force thread-group shared memory accesses to be volatile
    ///
    /// Workaround for compute shaders that read and
    /// write from the same shared memory location
    /// without explicit synchronization.
    bool forceComputeLdsBarriers = false;

    /// Force UAV synchronization insided compute shaders
    ///
    /// Workaround for compute shaders that access overlapping
    /// memory regions within a UAV without proper workgroup
    /// synchroniation. Will have a negative performance impact.
    bool forceComputeUavBarriers = false;

    /// Use relaxed memory barriers
    ///
    /// May improve performance in some games,
    /// but might also cause rendering issues.
    bool relaxedBarriers = false;

    /// Ignore graphics barriers
    ///
    /// May improve performance in some games,
    /// but might also cause rendering issues.
    bool relaxedGraphicsBarriers = false;

    /// Maximum tessellation factor.
    ///
    /// Limits tessellation factors in tessellation
    /// control shaders. Values from 8 to 64 are
    /// supported, other values will be ignored.
    int32_t maxTessFactor = 0;

    /// Anisotropic filter override
    ///
    /// Enforces anisotropic filtering with the
    /// given anisotropy value for all samplers.
    int32_t samplerAnisotropy = -1;

    /// Mipmap LOD bias
    ///
    /// Enforces the given LOD bias for all samplers.
    float samplerLodBias = 0.0f;

    /// Clamps negative LOD bias
    bool clampNegativeLodBias = false;

    /// Override maximum frame latency if the app specifies
    /// a higher value. May help with frame timing issues.
    int32_t maxFrameLatency = 0;

    /// Defer surface creation until first present call. This
    /// fixes issues with games that create multiple swap chains
    /// for a single window that may interfere with each other.
    bool deferSurfaceCreation = false;

    /// Enables sample rate shading by interpolating fragment shader
    /// inputs at the sample location rather than pixel center,
    /// unless otherwise specified by the application.
    bool forceSampleRateShading = false;

    /// Forces the sample count of all textures to be 1, and
    /// performs the required shader and resolve fixups.
    bool disableMsaa = false;

    /// Dynamic resources with the given bind flags will be allocated
    /// in cached system memory. Enabled automatically when recording
    /// an api trace.
    uint32_t cachedDynamicResources = 0;

    /// blessed: cb-ring -- Map(WRITE_DISCARD) on small dynamic constant
    /// buffers in cached memory hands out chunks of a persistently mapped
    /// ring and batches the renames into one cs command. Off by default.
    bool blessedCbRing = false;

    /// blessed: perf-halfrate -- the ring's blocks in device-local,
    /// host-visible memory (resizable bar) instead of cached host memory.
    /// Needs blessedCbRing. Off by default, and a loss on skyrim
    /// (bis-rebar, -30% fps): every locked instruction after write-combined
    /// stores waits ~200 ns for them to drain over pcie, and skyrim runs one
    /// between every cbuffer it writes (scratchpad wcbench, 2026-09-24).
    bool blessedCbRingDeviceLocal = false;

    /// blessed: cb-mirror -- a device-local mirror of each cached ring
    /// block. Cbuffer bindings point at the mirror, kept current by a
    /// transfer-queue copy of the block's written range every submission,
    /// instead of the cpu writing vram directly (see blessedCbRingDeviceLocal's
    /// -30%). Needs blessedCbRing. Off by default; see docs/tuning-report.md.
    bool blessedCbMirror = false;

    /// blessed: traverse-passes -- opts DYNAMIC buffers bound only as a
    /// vertex and/or index buffer (never also a constant buffer) out of
    /// cachedDynamicResources's cached-system-memory override, so they keep
    /// dxvk's own default for a dynamic buffer with bind flags: device-local,
    /// host-visible (resizable bar). Unlike blessedCbRingDeviceLocal's -30%
    /// (many small locked writes into write-combined memory), the write
    /// pattern here is a bulk streaming copy, which write-combined memory is
    /// built for. Off by default; a lead a/b, not a proven win -- see
    /// blessed-notes/traverse-passes-notes.md.
    bool blessedVbRebar = false;

    /// blessed: threaded-fe -- the app gets a recording facade as its
    /// immediate context, and a front end thread replays into the real
    /// one. Off by default: the one kill switch.
    bool blessedThreadedFrontEnd = false;

    /// blessed: threaded-fe -- stage 0: same facade and ring, but the
    /// recording thread replays itself at every publish. Only read when
    /// blessedThreadedFrontEnd is set.
    bool blessedThreadedFrontEndLoopback = false;

    /// blessed: threaded-fe-2 -- stage 2: Present is recorded and runs on
    /// the front end; off keeps stage 1 (Present drains and runs on the
    /// game thread). Only read when blessedThreadedFrontEnd is set.
    bool blessedThreadedPresent = true;

    /// blessed: threaded-fe-2 -- where the front end thread runs: empty
    /// (the scheduler decides), "auto" (ideal processor on a core other
    /// than the recording thread's), "auto-mask" (hard affinity to every
    /// core but that one), "<n>" (ideal logical processor n), or
    /// "mask:<hex>" (hard affinity mask).
    std::string blessedFrontEndCpu;

    /// blessed: threaded-fe-2 -- stage 3: redundant binds are not recorded,
    /// the per-draw sequence packs into draw packets, and publishes come
    /// every 128 calls instead of after every draw. Only read when
    /// blessedThreadedFrontEnd is set.
    bool blessedFrontEndCompact = true;

    /// Always lock immediate context on every API call. May be
    /// useful for debugging purposes or when applications have
    /// race conditions.
    bool enableContextLock = false;

    /// Whether to expose the driver command list feature. Enabled by
    /// default and generally beneficial, but some games may assume that
    /// this is not supported when running on an AMD GPU.
    bool exposeDriverCommandLists = true;

    /// Ensure that for the same D3D commands the output VK commands
    /// don't change between runs. Useful for comparative benchmarking,
    /// can negatively affect performance.
    bool reproducibleCommandStream = false;

    /// Whether to force a staging buffer for mapped images.
    /// Some games are broken and ignore row pitch.
    bool disableDirectImageMapping = false;

    /// Shader dump path
    std::string shaderDumpPath;
  };
  
}
