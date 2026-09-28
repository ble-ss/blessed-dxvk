dxvk_shaders = files([
  'shaders/dxvk_cursor_vert.vert',
  'shaders/dxvk_cursor_frag.frag',

  'shaders/dxvk_dummy_frag.frag',

  'shaders/dxvk_mipgen.comp',

  'shaders/dxvk_present_frag.frag',
  'shaders/dxvk_present_frag_blit.frag',
  'shaders/dxvk_present_frag_ms.frag',
  'shaders/dxvk_present_frag_ms_blit.frag',
  'shaders/dxvk_present_vert.vert',

  'hud/shaders/hud_chunk_frag_background.frag',
  'hud/shaders/hud_chunk_frag_visualize.frag',
  'hud/shaders/hud_chunk_vert_background.vert',
  'hud/shaders/hud_chunk_vert_visualize.vert',

  'hud/shaders/hud_frame_time_eval.comp',

  'hud/shaders/hud_graph_frag.frag',
  'hud/shaders/hud_graph_vert.vert',

  'hud/shaders/hud_text_frag.frag',
  'hud/shaders/hud_text_vert.vert',
])

# blessed: compiled with glsl_generator_blessed (no -gVS) -- see meson.build
dxvk_shaders_blessed = files([
  'shaders/blessed_rt_selftest.comp',
  'shaders/blessed_sun_shadow.comp',
  'shaders/blessed_soft_shadow_trace.comp',
  'shaders/blessed_soft_shadow_filter.comp',
  'shaders/blessed_skin.comp',
  'shaders/blessed_gi_trace.comp',
  'shaders/blessed_gi_albedo.comp',
  'shaders/blessed_gi_bounds.comp', # blessed: gi-bounds
  'shaders/blessed_point_shadow.comp', # blessed: point-lights
  'shaders/blessed_point_soft_trace.comp', # blessed: point-lights
  'shaders/blessed_point_soft_filter.comp', # blessed: point-lights
  'shaders/blessed_ao_trace.comp',   # blessed: rtao
  'shaders/blessed_ao_blur.comp',    # blessed: rtao
  'shaders/blessed_ao_resolve.comp', # blessed: rtao
  'shaders/blessed_vol_trace.comp',    # blessed: volumetrics
  'shaders/blessed_vol_upsample.comp', # blessed: volumetrics
  'shaders/blessed_vol_dump.comp',     # blessed: volumetrics
  'shaders/blessed_look_prep.comp', # blessed: look-post
  'shaders/blessed_look_composite.comp', # blessed: look-post
  'shaders/blessed_cascade_min_vert.vert', # blessed: cascade-cache
  'shaders/blessed_cascade_min_frag.frag', # blessed: cascade-cache
  'shaders/blessed_cascade_verify.comp',   # blessed: cascade-cache
  'shaders/blessed_cascade_bounds.comp',   # blessed: cascade-cache
  'shaders/blessed_vrs_rate.comp', # blessed: vrs
  'shaders/blessed_vrs_tint.comp', # blessed: vrs
  'shaders/blessed_halfrate_capture.comp', # blessed: half-rate far field
  'shaders/blessed_halfrate_fill_vert.vert', # blessed: half-rate far field
  'shaders/blessed_halfrate_fill_frag.frag', # blessed: half-rate far field
])

dxvk_src = [
  'dxvk_access.cpp',
  'dxvk_adapter.cpp',
  'dxvk_allocator.cpp',
  'dxvk_barrier.cpp',
  'dxvk_buffer.cpp',
  'dxvk_cmdlist.cpp',

  'blessed/blessed_ao.cpp', # blessed: rtao
  'blessed/blessed_async.cpp', # blessed: async-compute
  'blessed/blessed_vol_async.cpp', # blessed: vanilla-vol-async
  'blessed/blessed_cascade_cache.cpp', # blessed: cascade-cache
  'blessed/blessed_cascade_bounds.cpp', # blessed: cascade-cache
  'blessed/blessed_cb_ring.cpp', # blessed: cb-ring
  'blessed/blessed_cb_mirror.cpp', # blessed: cb-mirror
  'blessed/blessed_gi.cpp',
  'blessed/blessed_gi_bounds.cpp', # blessed: gi-bounds
  'blessed/blessed_gpu_gaps.cpp', # blessed: gpu-gaps
  'blessed/blessed_present.cpp', # blessed: present-idle
  'blessed/blessed_present_bridge.cpp', # blessed: present-idle
  'blessed/blessed_fse.cpp', # blessed: present-fse-appcontrolled
  'blessed/blessed_point_shadow.cpp', # blessed: point-lights
  'blessed/blessed_look.cpp', # blessed: look-post
  'blessed/blessed_halfrate.cpp', # blessed: half-rate far field
  'blessed/blessed_layer_tracking.cpp', # blessed: perlayer
  'blessed/blessed_rt.cpp',
  'blessed/blessed_scene.cpp',
  'blessed/blessed_shadow.cpp',
  'blessed/blessed_skin_debug.cpp',
  'blessed/blessed_soft_shadow.cpp',
  'blessed/blessed_volumetrics.cpp', # blessed: volumetrics
  'blessed/blessed_vrs.cpp', # blessed: vrs

  'dxvk_compute.cpp',
  'dxvk_constant_state.cpp',
  'dxvk_context.cpp',
  'dxvk_cs.cpp',
  'dxvk_descriptor_heap.cpp',
  'dxvk_descriptor_info.cpp',
  'dxvk_descriptor_pool.cpp',
  'dxvk_descriptor_worker.cpp',
  'dxvk_device.cpp',
  'dxvk_device_filter.cpp',
  'dxvk_device_info.cpp',
  'dxvk_fence.cpp',
  'dxvk_format.cpp',
  'dxvk_framebuffer.cpp',
  'dxvk_gpu_event.cpp',
  'dxvk_gpu_query.cpp',
  'dxvk_graphics.cpp',
  'dxvk_hang.cpp',
  'dxvk_image.cpp',
  'dxvk_implicit_resolve.cpp',
  'dxvk_instance.cpp',
  'dxvk_latency_builtin.cpp',
  'dxvk_latency_reflex.cpp',
  'dxvk_memory.cpp',
  'dxvk_meta_blit.cpp',
  'dxvk_meta_clear.cpp',
  'dxvk_meta_copy.cpp',
  'dxvk_meta_mipgen.cpp',
  'dxvk_meta_resolve.cpp',
  'dxvk_options.cpp',
  'dxvk_pipelayout.cpp',
  'dxvk_pipemanager.cpp',
  'dxvk_platform_exts.cpp',
  'dxvk_presenter.cpp',
  'dxvk_queue.cpp',
  'dxvk_sampler.cpp',
  'dxvk_shader.cpp',
  'dxvk_shader_builtin.cpp',
  'dxvk_shader_cache.cpp',
  'dxvk_shader_io.cpp',
  'dxvk_shader_ir.cpp',
  'dxvk_shader_key.cpp',
  'dxvk_shader_spirv.cpp',
  'dxvk_signal.cpp',
  'dxvk_sparse.cpp',
  'dxvk_staging.cpp',
  'dxvk_stats.cpp',
  'dxvk_swapchain_blitter.cpp',
  'dxvk_unbound.cpp',
  'dxvk_util.cpp',

  'hud/dxvk_hud.cpp',
  'hud/dxvk_hud_font.cpp',
  'hud/dxvk_hud_item.cpp',
  'hud/dxvk_hud_renderer.cpp',
]

if platform == 'windows'
  dxvk_src += [
    'dxvk_openvr.cpp',
    'dxvk_openxr.cpp',
  ]
endif

dxvk_extra_deps = [ dependency('threads') ]
if platform == 'linux'
  dxvk_extra_deps += [ cpp.find_library('dl', required: false) ]
endif

dxvk_lib = static_library('dxvk', dxvk_src,
  glsl_generator.process(dxvk_shaders),
  glsl_generator_blessed.process(dxvk_shaders_blessed),
  dxvk_version,
  link_with           : [ util_lib, spirv_lib, wsi_lib ],
  dependencies        : [ dxbc_spirv_dep, vkcommon_dep ] + dxvk_extra_deps,
  include_directories : [ dxvk_include_path ],
)

dxvk_dep = declare_dependency(
  link_with           : [ dxvk_lib ],
  include_directories : [ dxvk_include_path ],
)
