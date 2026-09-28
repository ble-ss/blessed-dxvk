d3d11_res = wrc_generator.process('version.rc')

dxgi_common_src = [
  '../dxgi/dxgi_format.cpp',
]

d3d10_src = [
  '../d3d10/d3d10_blend.cpp',
  '../d3d10/d3d10_buffer.cpp',
  '../d3d10/d3d10_depth_stencil.cpp',
  '../d3d10/d3d10_device.cpp',
  '../d3d10/d3d10_input_layout.cpp',
  '../d3d10/d3d10_multithread.cpp',
  '../d3d10/d3d10_query.cpp',
  '../d3d10/d3d10_rasterizer.cpp',
  '../d3d10/d3d10_sampler.cpp',
  '../d3d10/d3d10_texture.cpp',
  '../d3d10/d3d10_util.cpp',
  '../d3d10/d3d10_view_dsv.cpp',
  '../d3d10/d3d10_view_rtv.cpp',
  '../d3d10/d3d10_view_srv.cpp',
]

d3d11_src = [
  'd3d11_annotation.cpp',
  'd3d11_blend.cpp',
  'blessed_ao.cpp', # blessed: rtao
  'blessed_autoinstance.cpp', # blessed: auto-instancing census
  'blessed_cascades.cpp',
  'blessed_cascade_cache.cpp', # blessed: cascade-cache
  'blessed_cascade_bounds.cpp', # blessed: cascade-cache, partial restore
  'blessed_cb_ring.cpp', # blessed: cb-ring
  'blessed_dump.cpp',
  'blessed_gi.cpp',
  'blessed_flush.cpp', # blessed: gpu-gaps
  'blessed_flush_pass_end.cpp', # blessed: flush-pass-end, BLESSED_FLUSH_AT_PASS_END
  'blessed_gpu_passes.cpp', # blessed: gpu-pass-timing
  'blessed_midframe_split.cpp', # blessed: between-passes, BLESSED_MIDFRAME_SPLIT
  'blessed_hook.cpp',
  'blessed_point_shadow.cpp', # blessed: point-lights
  'blessed_look.cpp', # blessed: look-post
  'blessed_halfrate.cpp', # blessed: half-rate far field
  'blessed_reflect_halfrate.cpp', # blessed: refl-harden
  'blessed_scene_capture.cpp',
  'blessed_shadow.cpp',
  'blessed_shader_replace.cpp', # blessed: shader-replace
  'blessed_dispatch_rewrite.cpp', # blessed: shader-replace, dispatch-size override
  'blessed_vol_collapse.cpp', # blessed: vol-collapse
  'blessed_vol_async.cpp', # blessed: vanilla-vol-async
  'blessed_vol_async_verify.cpp', # blessed: vanilla-vol-async, BLESSED_VOL_ASYNC_VERIFY
  'blessed_threaded_context.cpp', # blessed: threaded-fe
  'blessed_threaded_record.cpp', # blessed: threaded-fe
  'blessed_threaded_present.cpp', # blessed: threaded-fe-2
  'blessed_threaded_placement.cpp', # blessed: threaded-fe-2
  'blessed_threaded_forensics.cpp', # blessed: fe-crash-2
  'blessed_threaded_fe_diag.cpp', # blessed: fe-getters
  'blessed_vanilla_halfrate.cpp', # blessed: perf-halfrate
  'blessed_crash_log.cpp', # blessed: crash-log -- std::terminate / abort leave a log line
  'blessed_skip_replaced.cpp', # blessed: gpu track step 2 -- skip replaced vanilla work
  'blessed_volumetrics.cpp', # blessed: volumetrics
  'blessed_vrs.cpp', # blessed: vrs
  'blessed_zero_copy.cpp', # blessed: zero-copy-present
  'd3d11_buffer.cpp',
  'd3d11_class_linkage.cpp',
  'd3d11_cmdlist.cpp',
  'd3d11_context.cpp',
  'd3d11_context_def.cpp',
  'd3d11_context_ext.cpp',
  'd3d11_context_imm.cpp',
  'd3d11_cuda.cpp',
  'd3d11_depth_stencil.cpp',
  'd3d11_device.cpp',
  'd3d11_enums.cpp',
  'd3d11_features.cpp',
  'd3d11_fence.cpp',
  'd3d11_gdi.cpp',
  'd3d11_initializer.cpp',
  'd3d11_input_layout.cpp',
  'd3d11_interop.cpp',
  'd3d11_main.cpp',
  'd3d11_on_12.cpp',
  'd3d11_options.cpp',
  'd3d11_query.cpp',
  'd3d11_rasterizer.cpp',
  'd3d11_resource.cpp',
  'd3d11_sampler.cpp',
  'd3d11_shader.cpp',
  'd3d11_state.cpp',
  'd3d11_state_object.cpp',
  'd3d11_swapchain.cpp',
  'd3d11_texture.cpp',
  'd3d11_util.cpp',
  'd3d11_video.cpp',
  'd3d11_view_dsv.cpp',
  'd3d11_view_rtv.cpp',
  'd3d11_view_srv.cpp',
  'd3d11_view_uav.cpp',
]

d3d11_shaders = files([
  'shaders/d3d11_video_blit_frag.frag',
  'shaders/d3d11_video_blit_vert.vert',
  'shaders/d3d11_composition_frag.frag',
  'shaders/d3d11_composition_vert.vert',
])

d3d11_ld_args      = []
d3d11_link_depends = []

if platform == 'windows'
  d3d11_dxgi_deps = [ lib_gdi32, lib_dxgi ]
else
  d3d11_ld_args      += [ '-Wl,--version-script', join_paths(meson.current_source_dir(), 'd3d11.sym') ]
  d3d11_link_depends += files('d3d11.sym')
  d3d11_dxgi_deps = [ dxgi_dep ]
endif

d3d11_dll = shared_library(dxvk_name_prefix+'d3d11', dxgi_common_src + d3d11_src + d3d10_src,
    glsl_generator.process(d3d11_shaders), d3d11_res,
  dependencies        : d3d11_dxgi_deps + [ dxvk_dep, dxbc_spirv_dep ],
  include_directories : dxvk_include_path,
  install             : true,
  vs_module_defs      : 'd3d11'+def_spec_ext,
  link_args           : d3d11_ld_args,
  link_depends        : [ d3d11_link_depends ],
  kwargs              : dxvk_so_version,
)

d3d11_dep = declare_dependency(
  link_with           : [ d3d11_dll ],
  include_directories : [ dxvk_include_path ],
)

if platform != 'windows'
  pkg.generate(d3d11_dll,
    filebase: dxvk_pkg_prefix + 'd3d11',
    subdirs:  'dxvk',
  )
endif
