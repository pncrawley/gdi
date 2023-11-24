//gdi interface
const std = @import("std");
pub const backend = @import("gdi_backend.zig");

pub const ResourceHandle = backend.ResourceHandle;
pub const RenderStateID = ResourceHandle;

pub const GdiError = error {
    CreateResourceFailed,
};

//useful default resources
pub var black_texture = ResourceHandle.empty;
pub var white_texture = ResourceHandle.empty;
pub var error_texture = ResourceHandle.empty;

pub const dynamic_bind_space_range_begin: u32 = 500;
pub const dynamic_bind_space_range_end: u32 = 549;

pub const Format = enum(i32) {
    invalid,
    swap_chain,
    r32g32b32a32_float,
    r32g32b32a32_uint,
    r16g16b16a16_float,
    r16g16b16a16_uint,
    r8g8b8a8_unorm,
    r8g8b8a8_unorm_srgb,
    b8g8r8a8_unorm,
    b8g8r8a8_unorm_srgb,
    r32_float,
    r32_uint,
    r16_uint,
    d32_float,
    d32_float_s8x24_uint,
    bc1_unorm,
    bc1_unorm_srgb,
    bc2_unorm,
    bc2_unorm_srgb,
    bc3_unorm,
    bc3_unorm_srgb,
    bc4_unorm,
    bc4_snorm,
    bc5_unorm,
    bc5_snorm,
    bc6h_uf16,
    bc6h_sf16,
    bc7_unorm,
    bc7_unorm_srgb,
    r9g9b9e5_sharedexp,
    r32g32_float,
    r32g32b32a32_sint,
    r16_float,
    r8_unorm,
    r16g16_float,
    r8g8_unorm,

    const default_depth = Format.d32_float;
    const default_depth_stencil = Format.d32_float_s8x24_uint;
};

pub const FormatInfo = backend.FormatInfo;
pub const formatQueryInfo = backend.formatQueryInfo;

pub const ResourceKind = backend.ResourceKind;

pub const ResourceTransition = enum(u32) {
    shader_visible,
    uav,
    writable,
    default,
    indirect_args,
};

pub const ShaderParameterLifetime = enum(u3) {
    persistent,
    dynamic,
    draw_constants,
    static_sampler,
};

pub const CullMode = enum(u2) {
    none,
    front,
    back,
};

pub const Comparison = enum(u3) {
    always,
    equal,
    not_equal,
    greater,
    greater_equal,
    less,
    less_equal,
    never,
};

pub const Blend = enum(u5) {
    zero,
    one,
    src_color,
    inv_src_color,
    src_alpha,
    inv_src_alpha,
    dst_alpha,
    inv_dst_alpha,
    dst_color,
    inv_dst_color,
};

pub const BlendOp = enum(u3) {
    add,
    subtract,
    reverse_subtract,
    minimum,
    maximum,
};

pub const PrimitiveType = enum(u3) {
    tri_list = 0,
    tri_strip,
    line_list,
    line_strip,
    point_list,
};

pub const RenderStateDesc = packed struct {
    const BlendState = packed struct {
        is_blending_enabled: bool = false,
        src_blend: Blend = Blend.zero,
        dst_blend: Blend = Blend.zero,
        blend_op: BlendOp = BlendOp.add,
        src_blend_alpha: Blend = Blend.zero,
        dst_blend_alpha: Blend = Blend.zero,
        blend_op_alpha: BlendOp = BlendOp.add,
        disable_write_red: bool = false,
        disable_write_green: bool = false,
        disable_write_blue: bool = false,
        disable_write_alpha: bool = false,
    };
    const RasterState = packed struct {
        is_wireframe_enabled: bool = false,
        is_depth_clip_enabled: bool = true,
        cull_mode: CullMode = CullMode.none,
    };
    const DepthState = packed struct {
        is_depth_write_enabled: bool = false,
        depth_test: Comparison = Comparison.always,
    };
    const StencilState = packed struct {
        is_enabled : bool = false,
    };

    blend: BlendState = .{},
    raster: RasterState = .{},
    depth: DepthState = .{},
    stencil:StencilState = .{},
    prim_type: PrimitiveType = PrimitiveType.tri_list,
};

pub const ClearValue = union {
    pub const zero = ClearValue{ .color = .{0, 0, 0, 0} };

    color: [4]f32,
    depth_stencil: packed struct {
        depth: f32,
        stencil: u8,
    },
};

pub const RenderTargetDesc = struct {
    format: Format,
    size_x: u32,
    size_y: u32,
    array_count: u32,
    allow_unordered_access: bool,
    clear_value: ClearValue,
};

pub const ShaderParameterDesc = struct {
    value_kind: ResourceKind,
    lifetime: ShaderParameterLifetime = .persistent,
    array_count: u32 = 1,
    is_unordered_access_view: bool = false,
    is_non_srgb: bool = false, //when the format of the texture or render target is already non-srgb this does nothing; when it's srgb, this forces it to non-srgb for the parameter
    binding_name: []const u8,
};

pub const ConstantBufferDesc = struct {
    size: u32,
};

pub const ShaderKind = enum {
    invalid,
    ps,
    vs,
    cs,
    ms,
    as,
};

pub const ShaderByteCode = struct {
    kind: ShaderKind,
    len: usize,
    code: [*]u8,

    //reflection info
    output_render_target_count: u8,
    descriptor_bind_set: ResourceHandle,
};

pub const ShaderModel = enum {
    sm_5_1,
    sm_6_5,
};

pub const ShaderRef = struct {
    shader_name: []const u8,
    cs: ?*ShaderByteCode = null,
    vs: ?*ShaderByteCode = null,
    ps: ?*ShaderByteCode = null,
    ms: ?*ShaderByteCode = null,
    as: ?*ShaderByteCode = null,
};

pub fn ShaderKey(comptime PermutationEnum: type) type {
    return struct {
        const Self = @This();
        pub const Permutations = PermutationEnum;
        pub const PermutationsSet = std.EnumSet(PermutationEnum);
        pub const Kind = enum { draw, mesh, mesh_amp, compute };

        shader_name: []const u8,
        permutations: PermutationsSet,
        render_state_id: ?RenderStateID,
        kind: Kind,

        pub fn initDraw(shader_name: []const u8, render_state_id: RenderStateID, init_perms: std.enums.EnumFieldStruct(PermutationEnum, bool, false)) Self {
            return Self{
                .shader_name = shader_name,
                .permutations = std.EnumSet(PermutationEnum).init(init_perms),
                .render_state_id = render_state_id,
                .kind = .draw,
            };
        }

        pub fn initCompute(shader_name: []const u8, init_perms: std.enums.EnumFieldStruct(PermutationEnum, bool, false)) Self {
            return Self{
                .shader_name = shader_name,
                .permutations = std.EnumSet(PermutationEnum).init(init_perms),
                .render_state_id = null,
                .kind = .compute,
            };
        }

        pub fn initMesh(shader_name: []const u8, render_state_id: RenderStateID, init_perms: std.enums.EnumFieldStruct(PermutationEnum, bool, false)) Self {
            return Self{
                .shader_name = shader_name,
                .permutations = std.EnumSet(PermutationEnum).init(init_perms),
                .render_state_id = render_state_id,
                .kind = .mesh,
            };
        }

        pub fn initMeshAmplify(shader_name: []const u8, render_state_id: RenderStateID, init_perms: std.enums.EnumFieldStruct(PermutationEnum, bool, false)) Self {
            return Self{
                .shader_name = shader_name,
                .permutations = std.EnumSet(PermutationEnum).init(init_perms),
                .render_state_id = render_state_id,
                .kind = .mesh_amp,
            };
        }

        pub fn MapContext(comptime ignore_render_state_id: bool) type {
            return struct {
                pub fn hash(self: @This(), k: Self) u64 {
                    _ = self;
                    var h = std.hash.Wyhash.init(42);
                    h.update(k.shader_name);
                    if (PermutationsSet.len <= @bitSizeOf(usize)) {
                        h.update(std.mem.asBytes(&k.permutations.bits.mask));
                    } else {
                        h.update(std.mem.sliceAsBytes(k.permutations.bits.masks[0..k.permutations.bits.num_masks]));
                    }
                    if (!ignore_render_state_id) {
                        if (k.render_state_id) |id| {
                            h.update(std.mem.asBytes(&id));
                        }
                    }
                    return h.final();
                }

                pub fn eql(self: @This(), a: Self, b: Self) bool {
                    _ = self;
                    if (!ignore_render_state_id) {
                        if (a.render_state_id) |aid| {
                            if (null == b.render_state_id) {
                                return false;
                            }
                            const bid = b.render_state_id.?;
                            if (aid.gen != bid.gen) {
                                return false;
                            }
                            if (aid.idx != bid.idx) {
                                return false;
                            }
                        } else {
                            if (null != b.render_state_id) {
                                return false;
                            }
                        }
                    }
                    if (!std.hash_map.eqlString(a.shader_name, b.shader_name)) {
                        return false;
                    }
                    if (PermutationsSet.len <= @bitSizeOf(usize)) {
                        return a.permutations.bits.mask == b.permutations.bits.mask;
                    } else {
                        return std.mem.eql(usize, a.permutations.bits.masks, b.permutations.bits.masks);
                    }
                }
            };
        }
    };
}

pub fn ShaderStateMap(comptime PermutationEnum: type) type {
    return struct {
        const Self = @This();
        pub const Key = ShaderKey(PermutationEnum);
        pub const ShaderMap = std.HashMapUnmanaged(Key, ShaderSet, Key.MapContext(true), std.hash_map.default_max_load_percentage);
        pub const StateMap = std.HashMapUnmanaged(Key, ShaderPipelineState, Key.MapContext(false), std.hash_map.default_max_load_percentage);
        pub const ShaderSet = struct {
            cs: ?*ShaderByteCode = null,
            vs: ?*ShaderByteCode = null,
            ps: ?*ShaderByteCode = null,
            ms: ?*ShaderByteCode = null,
            as: ?*ShaderByteCode = null,
        };
        pub const ShaderPipelineState = ResourceHandle;

        allocator: std.mem.Allocator,
        compiler: backend.ShaderCompiler,
        shader_map: ShaderMap,
        state_map: StateMap,

        pub fn init(allocator: std.mem.Allocator, auto_load_dir_base: ?[]const u8, compiled_files_dir: ?[]const u8) !Self {
            return Self{
                .allocator = allocator,
                .compiler = try backend.ShaderCompiler.init(
                    if (auto_load_dir_base) |dir| dir else "./shaders/",
                    if (compiled_files_dir) |dir| dir else ".runtime/Shaders/"
                ),
                .shader_map = .{},
                .state_map = .{},
            };
        }

        pub fn getOrCreateShaderSet(self: *Self, key: Key) !*ShaderSet {
            std.debug.assert(key.shader_name.len > 0);

            //see if it already exists
            if (self.shader_map.getEntry(key)) |item| {
                return item.value_ptr;
            }

            //doesn't exist so we need to create one

            //build a permutation defines list
            const fields = std.meta.fields(Key.Permutations);
            var defines = try std.ArrayList(backend.ShaderCompiler.Define).initCapacity(self.allocator, fields.len);
            defer defines.deinit();
            inline for (fields) |field| {
                const value = if (key.permutations.contains(@enumFromInt(field.value))) "1" else "0";
                defines.appendAssumeCapacity(.{ .name = field.name, .value = value });
            }

            var entry = ShaderSet{};
            switch (key.kind) {
                .compute => {
                    entry.cs = try self.compiler.compile(self.allocator, key.shader_name, ShaderKind.cs, defines.items, .sm_5_1);
                },
                .draw => {
                    entry.vs = try self.compiler.compile(self.allocator, key.shader_name, ShaderKind.vs, defines.items, .sm_5_1);
                    entry.ps = try self.compiler.compile(self.allocator, key.shader_name, ShaderKind.ps, defines.items, .sm_5_1);
                },
                .mesh => {
                    entry.ms = try self.compiler.compile(self.allocator, key.shader_name, ShaderKind.ms, defines.items, .sm_6_5);
                    entry.ps = try self.compiler.compile(self.allocator, key.shader_name, ShaderKind.ps, defines.items, .sm_6_5);
                },
                .mesh_amp => {
                    entry.as = try self.compiler.compile(self.allocator, key.shader_name, ShaderKind.as, defines.items, .sm_6_5);
                    entry.ms = try self.compiler.compile(self.allocator, key.shader_name, ShaderKind.ms, defines.items, .sm_6_5);
                    entry.ps = try self.compiler.compile(self.allocator, key.shader_name, ShaderKind.ps, defines.items, .sm_6_5);
                },
            }

            self.shader_map.putNoClobber(self.allocator, key, entry) catch unreachable;
            return self.shader_map.getEntry(key).?.value_ptr;
        }

        pub fn getOrCreateShaderState(self: *Self, key: Key) !ShaderPipelineState {
            std.debug.assert(key.shader_name.len > 0);

            //see if it already exists
            if (self.state_map.getEntry(key)) |item| {
                return item.value_ptr.*;
            }

            //doesn't exist so we need to create one
            //first create or get the shader set
            const shader_set = try self.getOrCreateShaderSet(key);

            //now create the state
            const shader_ref = ShaderRef{
                .shader_name = key.shader_name,
                .as = shader_set.as,
                .ms = shader_set.ms,
                .cs = shader_set.cs,
                .vs = shader_set.vs,
                .ps = shader_set.ps,
            };

            var state = try createResource(
                "Shader State", //key.shader_name,
                PipelineStateDesc{
                    .shader_ref = shader_ref,
                    .render_state_id = key.render_state_id,
                }
            );

            self.state_map.putNoClobber(self.allocator, key, state) catch unreachable;
            return self.state_map.getEntry(key).?.value_ptr.*;
        }
    };
}

pub const PipelineStateDesc = struct {
    shader_ref: ShaderRef,
    render_state_id: ?RenderStateID,
};

pub const BufferDesc = struct {
    size: u32,
    format: Format,
    allow_unordered_access: bool = false,
    is_used_for_readback: bool = false,
    allow_index_buffer: bool = false,
};

pub const SamplerDesc = struct {
    //TODO: samplers
    stuff_goes_here: u32 = 0,
};

pub const TextureDesc = struct {
    pub const max_mip_count = 15;
    pub const MipDesc = struct {
        size_x: u32,
        size_y: u32,
        size_z: u32,
    };
    format: Format,
    array_count: u32,
    mip_count: u32,
    allow_unordered_access: bool = false,
    is_cube_map: bool = false,
    mip_descs: [max_mip_count] MipDesc, //in c this is an unsized array; instead here we just set it to the max size
};

pub const TextureMipDesc = struct {
    size: u32,
    offset: u32,
    level: u32,
    size_x: u32,
    size_y: u32,
    size_z: u32,
    stride_x: u32,
    rows_y: u32,
    format: Format,
    backend_format: u32,
};

pub const RenderPass = struct {
    pub const cmd_kind = CmdKind.begin_render_pass;
    pub const AccessDesc = packed struct {
        pub const Kind = enum(u4) {
            no_access,
            discard,
            keep,
            clear,
            resolve,
        };
        begin: Kind = .no_access,
        end: Kind = .keep,
    };

    render_targets: [backend.max_render_target_count]?ResourceHandle = .{null, null, null, null, null, null, null, null},
    depth_stencil: ?ResourceHandle = null,
    depth_stencil_slice: u32 = 0,
    render_target_access: [backend.max_render_target_count] AccessDesc = .{.{}, .{}, .{}, .{}, .{}, .{}, .{}, .{}},
    depth_access: AccessDesc = .{},
    stencil_access: AccessDesc = .{},
    is_depth_read_only: bool = false,
    debug_name: []const u8 = "",
};

pub const IndirectCmd = enum(u8) { draw, draw_indexed, dispatch_mesh };

pub const DrawArgs = extern struct { //the layout is very important and must not change
    vertex_count_per_instance: u32,
    instance_count: u32,
    start_vertex_loc: u32,
    start_instance_loc: u32,
};

pub const DrawIndexedArgs = extern struct { //the layout is very important and must not change
    index_count_per_instance: u32,
    instance_count: u32,
    start_index_loc: u32,
    base_vertex_loc: i32,
    start_instance_loc: u32,
};

pub const DispatchMeshArgs = extern struct { //the layout is very important and must not change
    thread_group_count_x: u32,
    thread_group_count_y: u32,
    thread_group_count_z: u32,
};

pub const DrawShaderConstants = extern union { //depends upon aliasing the fields
    pub const count = 4;
    f: [count] f32,
    i: [count] i32,
    u: [count] u32,
};

pub const DispatchThreadGroupCounts = [3]u32;

pub const GpuProfileID = u32;

//createResource() is thread safe for some resource kinds (TODO: make thread safe for all of them)
pub fn createResource(comptime debug_name: [:0]const u8, desc: anytype) !ResourceHandle {
    return backend.createResource(debug_name, desc);
}

pub fn destroyResource(resource: anytype) void {
    backend.destroyResource(resource) catch unreachable;
}

pub const CmdKind = enum(u6) {
    invalid,
    end_of_list,
    jump,
    begin_render_pass,
    end_render_pass,
    set_pipeline_state,
    draw,
    draw_indexed,
    execute_indirect,
    dispatch_compute,
    clear_buffer,
    copy_buffer,
    transition,
    read_buffer_async,
    begin_profile_query,
    end_profile_query,
    custom_cmd,
};

pub const CustomCmdFunc = fn(ctx: *anyopaque, cl: backend.PlatformCommandList) void;

pub const CmdBuf = struct {
    const Self = @This();
    const CmdList = backend.CmdList;

    cl: CmdList,
    tcb: ?*backend.TranslatedCmdBuf.Node,

    pub fn init() CmdBuf {
        return CmdBuf{
            .cl = backend.CmdList{},
            .tcb = null,
        };
    }
    pub fn deinit(self: *Self) void {
        backend.destroyCmdBuf(self);
    }

    pub fn beginRenderPass(self: *Self, rp: RenderPass) void {
        self.cl.writeCmd(rp);
    }
    pub fn endRenderPass(self: *Self) void {
        self.cl.beginCmd(CmdKind.end_render_pass, 0);
        self.cl.endCmd();
    }

    pub fn setPipelineState(self: *Self, state: ResourceHandle) void {
        self.cl.writeCmd(CmdList.CmdSetPipelineState{ .pipeline_state = state });
    }

    // graphics
    pub fn draw(self: *Self, params: DrawArgs, dsc: DrawShaderConstants) void {
        self.cl.writeCmd(CmdList.CmdDraw{ .params = params, .dsc = dsc });
    }
    pub fn drawIndexed(self: *Self, idx_buf: ResourceHandle, params: DrawIndexedArgs, dsc: DrawShaderConstants) void {
        self.cl.writeCmd(CmdList.CmdDrawIndexed{ .params = params, .dsc = dsc, .idx_buf = idx_buf });
    }

    // indirect commands
    pub fn executeIndirect(self: *Self, kind: IndirectCmd, arg_buf: ResourceHandle, arg_offset: usize, count_buf: ?ResourceHandle, dsc: DrawShaderConstants) void {
        self.cl.writeCmd(CmdList.CmdExecuteIndirect{
            .kind = kind,
            .arg_buf = arg_buf,
            .arg_offset = arg_offset,
            .count_buf = if (count_buf) |buf| buf else ResourceHandle.empty,
            .dsc = dsc,
        });
    }
    pub fn executeIndirectSimple(self: *Self, kind: IndirectCmd, arg_buf: ResourceHandle) void {
        self.cl.writeCmd(CmdList.CmdExecuteIndirect{
            .kind = kind,
            .arg_buf = arg_buf,
            .arg_offset = 0,
            .count_buf = ResourceHandle.empty,
            .dsc = std.mem.zeroes(DrawShaderConstants),
        });
    }

    // compute
    pub fn dispatch(self: *Self, tgc: DispatchThreadGroupCounts) void {
        self.cl.writeCmd(CmdList.CmdDispatch{ .counts = tgc });
    }
    pub fn dispatchWithState(self: *Self, ds: ResourceHandle, tgc: DispatchThreadGroupCounts) void {
        self.setPipelineState(ds);
        self.dispatch(tgc);
    }

    pub fn transition(self: *Self, resource: ResourceHandle, kind: ResourceTransition) void {
        self.cl.writeCmd(CmdList.CmdTransition{ .resource = resource, .kind = kind, });
    }

    pub fn readBufferAsync(self: *Self, resource: ResourceHandle, signal_read_finished: ?*std.atomic.Atomic(bool), output: []u8) void {
        std.debug.assert((signal_read_finished == null) or (signal_read_finished.?.load(.Unordered) == false));
        self.cl.writeCmd(CmdList.CmdReadBufferAsync{ .resource = resource, .signal_read_finished = signal_read_finished, .output_ptr = output.ptr, .output_len = output.len });
    }

    pub fn clearBuffer(self: *Self, buf: ResourceHandle, values: [4]u32) void {
        self.cl.writeCmd(CmdList.CmdClearBuffer{ .buf = buf, .values = values, });
    }

    pub fn copyBuffer(self: *Self, src: ResourceHandle, dst: ResourceHandle) void {
        self.cl.writeCmd(CmdList.CmdCopyBuffer{ .src = src, .dst = dst, });
    }

    pub fn translateAndSubmit(self: *Self) !void {
        if (self.cl.cur != null) { //check we have commands
            try backend.submitPendingGpuCopies();
            std.debug.assert(self.tcb == null); //check we are not already translated
            try backend.translateCmdBuf(self);
            backend.submitCmdBuf(self);
            try backend.releaseCmdBuf(self);
        }
    }

    pub fn beginGpuProfile(self: *Self, comptime src: std.builtin.SourceLocation, color: anytype, comptime name_str: ?[*:0]const u8) GpuProfileID {
        const id = backend.beginGpuProfileZone(src, @intFromEnum(color), name_str);
        self.cl.writeCmd(CmdList.CmdBeginProfileQuery{ .id = id, .name = if (name_str) |n| n else "(unknown)" });
        return id;
    }
    pub fn endGpuProfile(self: *Self, id: GpuProfileID) void {
        backend.endGpuProfileZone(id);
        self.cl.writeCmd(CmdList.CmdEndProfileQuery{ .id = id, });
    }

    pub fn customCmd(self: *Self, func: *const CustomCmdFunc, ctx: *anyopaque) void {
        self.cl.writeCmd(CmdList.CmdCustom{ .func = func, .ctx = ctx });
    }
};

pub fn init(allocator: std.mem.Allocator, dx: *backend.PlatformModule) !void {
    try backend.init(allocator, dx);
    black_texture = try createTextureSimpleRGBA("Black Texture", 1, 1, &[_]u8{ 0, 0, 0, 0 });
    white_texture = try createTextureSimpleRGBA("White Texture", 1, 1, &[_]u8{ 255, 255, 255, 255 });
    error_texture = try createTextureSimpleRGBA("Error Texture", 1, 1, &[_]u8{ 255, 0, 255, 128 });
}

pub fn nextFrame() !void {
    try backend.nextFrame();
}

pub fn setShaderParamValue(shader_param: ResourceHandle, resource: ResourceHandle) void {
    backend.setShaderParamValue(shader_param, resource);
}

pub fn setShaderParamArrayValue(shader_param: ResourceHandle, resource: ResourceHandle, idx: usize) void {
    backend.setShaderParamArrayValue(shader_param, resource, idx);
}

pub const ResizeMode = enum {
    exact,
    equal_or_greater_than,
};

pub fn resizeRenderTargetIfNecessary(rt: ResourceHandle, size_x: u32, size_y: u32, mode: ResizeMode) !bool {
    return backend.resizeRenderTargetIfNecessary(rt, size_x, size_y, mode);
}

pub fn getRenderTargetViewPtr(rt: ResourceHandle) u64 {
    return backend.render_targets.lookup(rt).shader_view.ptr;
}

pub fn getPlatformResource(r: ResourceHandle) backend.PlatformResource {
    return backend.getPlatformResource(r);
}

//beginUpdateTextureMipLevel() is thread safe
pub fn beginUpdateTextureMipLevel(tex: ResourceHandle, mip_level: usize, array_idx: usize) !backend.GpuUploadRegion {
    return backend.beginUpdateTextureMipLevel(tex, mip_level, array_idx);
}

//endUpdateTextureMipLevel() is thread safe
pub fn endUpdateTextureMipLevel(tex: ResourceHandle, region: *backend.GpuUploadRegion) !void {
    _ = tex;
    return region.queue(null, null);
}

pub fn updateTextureMipLevel(tex: ResourceHandle, mip_level: usize, array_idx: usize, mip_data: []const u8) !void {
    var region = try backend.beginUpdateTextureMipLevel(tex, mip_level, array_idx);
    std.debug.assert(region.copy_info.texture.mip_desc.size >= mip_data.len);
    std.mem.copy(u8, region.mem, mip_data);
    try region.queue(null, null);
}

pub fn updateBuffer(buf: ResourceHandle, comptime T: type, offset_in_bytes: usize, data: []const T) void {
    backend.updateBuffer(buf, T, offset_in_bytes, data) catch unreachable;
}

pub fn updateConstantBuffer(buf: ResourceHandle, comptime T: type, data: T) void {
    backend.updateConstantBuffer(buf, T, data) catch unreachable;
}

pub fn zeroEntireBuffer(buf: ResourceHandle) !void {
    return backend.zeroEntireBuffer(buf);
}

pub fn createTextureSimpleRGBA(comptime debug_name: [:0]const u8, size_x: u32, size_y: u32, data: []const u8) !ResourceHandle {
    std.debug.assert((size_x*size_y*4) == data.len);
    var desc = TextureDesc{
        .format = .r8g8b8a8_unorm,
        .array_count = 1,
        .mip_count = 1,
        .mip_descs = undefined,
    };
    desc.mip_descs[0] = .{ .size_x = size_x, .size_y = size_y, .size_z = 1 };
    const hdl = try createResource(debug_name, desc);
    try updateTextureMipLevel(hdl, 0, 0, data);
    return hdl;
}

pub fn flush() !void {
    try backend.submitPendingGpuCopies();
}

pub fn supportsMeshShaders() bool {
    return backend.features.has_mesh_shaders;
}

//test "gdi interface test" {
//}
