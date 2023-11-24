const std = @import("std");
const gdi = @import("gdi.zig");
const D3D = @import("D3D");
pub const d3d = D3D.c;
pub const PlatformModule = D3D;
pub const PlatformResource = D3D.Resource;
pub const PlatformDevice = D3D.Device;
pub const PlatformCommandList = D3D.GraphicsCommandList;
const PlatformCommandQueue = D3D.CommandQueue;
const PlatformResourceBarrierBatcher = D3D.ResourceBarrierBatcher;
const PlatformCommandAllocator = D3D.CommandAllocator;
const PlatformPipelineState = D3D.PipelineState;
const PlatformGpuDescriptorHeap = D3D.GpuDescriptorHeap;
const PlatformCpuDescriptorHandle = D3D.CpuDescriptorHandle;
const PlatformGpuDescriptorHandle = D3D.GpuDescriptorHandle;
const ShaderByteCode = gdi.ShaderByteCode;
const ShaderKind = gdi.ShaderKind;
const FrameNumber = u64;
const DependencyFile = @import("DependencyFile.zig");
const AMDMemAlloc = @cImport({ @cInclude("cD3D12MemAlloc.h"); });
const platformSetResourceName = D3D.setResourceName;

const log = std.log.scoped(.gdi);

const Error = error {
    InvalidShaderKind,
    ProcessExecutionFailed,
    BadShaderParameterDescriptorRangeKind,
    InvalidDescriptorBindSetWithOverlappingRanges,
    ShaderHasOverlappingDescriptors,
    BadlyFormedOutputFromShaderCompile,
};

pub const max_render_target_count = 8;

const option_use_async_copy_queue = true;

// ************************************************************************************************************************
// module
// ************************************************************************************************************************

pub const Frame = struct {
    frame_num: u64,
    deletion_queue: std.ArrayListUnmanaged(D3D.AnyComReleasable),
    handle_deletion_queue: std.ArrayListUnmanaged(ResourceHandle),
    staging_descriptor_heap_deletion_queue: std.ArrayListUnmanaged(PlatformCpuDescriptorHandle),
    rt_descriptor_heap_deletion_queue: std.ArrayListUnmanaged(PlatformCpuDescriptorHandle),
    ds_descriptor_heap_deletion_queue: std.ArrayListUnmanaged(PlatformCpuDescriptorHandle),
    command_allocator: PlatformCommandAllocator,
    direct_upload_heap: GpuUploadHeap,
    async_upload_heap: GpuUploadHeap,

    pub fn create() !*Frame {
        var f = try allocator.create(Frame);
        f.frame_num = 1;
        f.deletion_queue = .{};
        f.handle_deletion_queue = .{};
        f.staging_descriptor_heap_deletion_queue = .{};
        f.rt_descriptor_heap_deletion_queue = .{};
        f.ds_descriptor_heap_deletion_queue = .{};
        f.command_allocator = try dx.device.createCommandAllocator(.direct);
        f.direct_upload_heap = try GpuUploadHeap.init(.direct_queue);
        f.async_upload_heap = try GpuUploadHeap.init(.async_queue);
        try f.command_allocator.reset();
        return f;
    }

    pub fn nextFrame(self: *Frame) !void {
        try self.command_allocator.reset();
        self.async_upload_heap.onGpuCompletionAssumesLocked();
        self.direct_upload_heap.onGpuCompletionAssumesLocked();

        //process deletion queue
        for (self.deletion_queue.items) |*item| {
            item.release();
        }
        self.deletion_queue.clearRetainingCapacity();

        //release all the handles so they can be reused now
        while (self.handle_deletion_queue.popOrNull()) |hdl| {
            switch (hdl.kind) {
                .none => unreachable,
                .render_target => render_targets.destroy(hdl),
                .pipeline_state => pipeline_states.destroy(hdl),
                .shader_parameter => shader_parameters.destroy(hdl),
                .constant_buffer => constant_buffers.destroy(hdl),
                .buffer => buffers.destroy(hdl),
                .texture => textures.destroy(hdl),
                .sampler => samplers.destroy(hdl),
                .render_state => render_states.destroy(hdl),
                .root_signature => root_signatures.destroy(hdl),
                .render_target_format_group => render_target_format_groups.destroy(hdl),
                .descriptor_bind_set => descriptor_bind_sets.destroy(hdl),
            }
        }

        //free all the pending staging descriptors now
        while (self.staging_descriptor_heap_deletion_queue.popOrNull()) |hdl| {
            dx.staging_descriptor_heap.free(hdl);
        }
        while (self.rt_descriptor_heap_deletion_queue.popOrNull()) |hdl| {
            dx.render_target_descriptor_heap.free(hdl);
        }
        while (self.ds_descriptor_heap_deletion_queue.popOrNull()) |hdl| {
            dx.depth_stencil_descriptor_heap.free(hdl);
        }
    }
};


pub var dx: *D3D = undefined;
var allocator: std.mem.Allocator = undefined;

var copy_command_queue: PlatformCommandQueue = undefined;
var copy_command_queue_fence: D3D.Fence = undefined;
var copy_frame_num: u64 = 0;

var gpu_descriptor_heap: PlatformGpuDescriptorHeap = undefined;

//var background_copy_command_queue: PlatformCommandQueue = undefined;

var samplers: ResourceTable(Sampler) = undefined;
var buffers: ResourceTable(Buffer) = undefined;
var constant_buffers: ResourceTable(ConstantBuffer) = undefined;
var textures: ResourceTable(Texture) = undefined;
pub var render_targets: ResourceTable(RenderTarget) = undefined;
var shader_parameters: ResourceTable(ShaderParameter) = undefined;
var shader_parameter_map: std.StringHashMapUnmanaged(ResourceHandle) = undefined;
var pipeline_states: ResourceTable(PipelineState) = undefined;
var render_target_format_groups: ResourceTable(RenderTargetFormatGroup) = undefined;
var render_states: ResourceTable(RenderState) = undefined;
var root_signatures: ResourceTable(RootSignature) = undefined;
var descriptor_bind_sets: ResourceTable(DescriptorBindSet) = undefined;

pub var frames: std.ArrayList(*Frame) = undefined;
var frame_num: u64 = 1;
var frame_fence: D3D.Fence = undefined;

var indirect_draw_sig: D3D.CommandSignature = undefined;
var indirect_draw_indexed_sig: D3D.CommandSignature = undefined;
var indirect_dispatch_mesh_sig: D3D.CommandSignature = undefined;

const TranslatedCmdBufPool = ThreadSafePool(TranslatedCmdBuf, 32);
var translated_cmd_buf_pool: TranslatedCmdBufPool = undefined;
const CmdBlockPool = ThreadSafePool(CmdBlock, 100);
var cmd_block_pool: CmdBlockPool = undefined;

var gpu_mem_allocator: *AMDMemAlloc.OpaqueAllocator = undefined;

pub var features: D3D.SupportedFeatures = undefined;

const ReadBackRequest = struct {
    resource: ResourceHandle, 
    signal: ?*std.atomic.Atomic(bool),
    output: []u8,
    ready_frame_num: u64,
};
var read_back_requests: std.ArrayList(ReadBackRequest) = undefined;

pub fn init(in_allocator: std.mem.Allocator, d3d_ptr: *D3D) !void {
    dx = d3d_ptr;
    allocator = in_allocator;
    features = dx.device.checkFeatureSupport();

    gpu_mem_allocator = AMDMemAlloc.CreateAllocator(@ptrCast(dx.device.device), @ptrCast(dx.dxgi_adapter)).?;

    gpu_descriptor_heap = try PlatformGpuDescriptorHeap.init(dx, d3d.D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 10000);
    samplers = try ResourceTable(Sampler).init(allocator, 32, ResourceKind.sampler);
    buffers = try ResourceTable(Buffer).init(allocator, 1000, ResourceKind.buffer);
    constant_buffers = try ResourceTable(ConstantBuffer).init(allocator, 100, ResourceKind.constant_buffer);
    textures = try ResourceTable(Texture).init(allocator, 1000, ResourceKind.texture);
    render_targets = try ResourceTable(RenderTarget).init(allocator, 32, ResourceKind.render_target);
    shader_parameters = try ResourceTable(ShaderParameter).init(allocator, 200, ResourceKind.shader_parameter);
    pipeline_states = try ResourceTable(PipelineState).init(allocator, 100, ResourceKind.pipeline_state);
    render_target_format_groups = try ResourceTable(RenderTargetFormatGroup).init(allocator, 100, ResourceKind.render_target_format_group);
    render_states = try ResourceTable(RenderState).init(allocator, 100, ResourceKind.render_state);
    root_signatures = try ResourceTable(RootSignature).init(allocator, 32, ResourceKind.root_signature);
    descriptor_bind_sets = try ResourceTable(DescriptorBindSet).init(allocator, 32, ResourceKind.descriptor_bind_set);

    translated_cmd_buf_pool = TranslatedCmdBufPool.init(allocator);
    cmd_block_pool = CmdBlockPool.init(allocator);

    read_back_requests = std.ArrayList(ReadBackRequest).init(allocator);

    if (option_use_async_copy_queue) {
        copy_command_queue = try dx.device.createCommandQueue(.copy, .high);
        copy_command_queue_fence = try dx.device.createFence();
    }

    frame_fence = try dx.device.createFence();
    try frame_fence.signalOnCpu(0);

    //background_copy_command_queue = try dx.device.createCommandQueue(.copy, .normal);

    shader_parameter_map = std.StringHashMapUnmanaged(ResourceHandle){};

    //_ = try buildEmptyRootSignature();

    //create the frame objects used to store per-frame data
    const frame_delay = 3;
    frames = try std.ArrayList(*Frame).initCapacity(allocator, frame_delay);
    var i: usize = 0;
    while (i < frame_delay) : (i += 1) {
        frames.appendAssumeCapacity(try Frame.create());
    }

    _ = try createShaderParameter(
        "draw constants",
        .{
            .value_kind = gdi.ResourceKind.constant_buffer,
            .lifetime = gdi.ShaderParameterLifetime.draw_constants,
            .array_count = 1,
            .is_unordered_access_view = false,
            .binding_name = "DrawConstants",
        }
    );

    const indirect_draw_sig_desc_arg = d3d.D3D12_INDIRECT_ARGUMENT_DESC{
        .Type = d3d.D3D12_INDIRECT_ARGUMENT_TYPE_DRAW,
        .unnamed_0 = undefined,
    };
    const indirect_draw_sig_desc = d3d.D3D12_COMMAND_SIGNATURE_DESC{
        .NumArgumentDescs = 1,
        .pArgumentDescs = &indirect_draw_sig_desc_arg,
        .NodeMask = 0,
        .ByteStride = @sizeOf(gdi.DrawArgs),
    };
    indirect_draw_sig = try dx.device.createCommandSignature(indirect_draw_sig_desc);

    const indirect_draw_indexed_sig_desc_arg = d3d.D3D12_INDIRECT_ARGUMENT_DESC{
        .Type = d3d.D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED,
        .unnamed_0 = undefined,
    };
    const indirect_draw_indexed_sig_desc = d3d.D3D12_COMMAND_SIGNATURE_DESC{
        .NumArgumentDescs = 1,
        .pArgumentDescs = &indirect_draw_indexed_sig_desc_arg,
        .NodeMask = 0,
        .ByteStride = @sizeOf(gdi.DrawIndexedArgs),
    };
    indirect_draw_indexed_sig = try dx.device.createCommandSignature(indirect_draw_indexed_sig_desc);

    const indirect_dispatch_mesh_sig_desc_arg = d3d.D3D12_INDIRECT_ARGUMENT_DESC{
        .Type = D3D.D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH,
        .unnamed_0 = undefined,
    };
    const indirect_dispatch_mesh_sig_desc = d3d.D3D12_COMMAND_SIGNATURE_DESC{
        .NumArgumentDescs = 1,
        .pArgumentDescs = &indirect_dispatch_mesh_sig_desc_arg,
        .NodeMask = 0,
        .ByteStride = @sizeOf(gdi.DispatchMeshArgs),
    };
    indirect_dispatch_mesh_sig = try dx.device.createCommandSignature(indirect_dispatch_mesh_sig_desc);
}

fn currentFrame() *Frame {
    return frames.items[0];
}

pub fn nextFrame() !void {
    try submitPendingGpuCopies();

    try dx.command_queue.signal(frame_fence, frame_num);
    dx.tracyNewFrame();

    //advance to the next frame
    frame_num += 1;
    try frames.insert(0, frames.pop());
    try currentFrame().nextFrame();

    const gpu_frame_num = frame_fence.getCompletedValue();
    //std.debug.print("cpu_frame_num {}    gpu_frame_num {}\n", .{frame_num, gpu_frame_num});

    //now process any readback requests
    while (read_back_requests.items.len > 0) {
        const request = read_back_requests.items[0];
        if (gpu_frame_num >= request.ready_frame_num) {
            const buf = lookupResource(request.resource, Buffer).gpu_resource.resource;
            var mem = try buf.mapEntireBuffer();
            std.mem.copy(u8, request.output, mem[0..request.output.len]);
            buf.unmapEntireBuffer();
            if (request.signal) |sig| {
                sig.store(true, .Release);
            }
            _ = read_back_requests.orderedRemove(0);
        } else {
            break;
        }
    }
}

pub fn submitPendingGpuCopies() !void {
    copy_frame_num += 1;
    try currentFrame().async_upload_heap.submitPendingCopies(copy_command_queue);
    try copy_command_queue.signal(copy_command_queue_fence, copy_frame_num);

    try currentFrame().direct_upload_heap.submitPendingCopies(dx.command_queue);
}

fn scheduleFrameResourceForDeletion(resource: anytype) !void {
    try currentFrame().deletion_queue.append(allocator, D3D.comReleasable(resource));
}

fn scheduleStagingDescriptorHandleForDeletion(hdl: *PlatformCpuDescriptorHandle) !void {
    if (hdl.ptr != 0) {
        try currentFrame().staging_descriptor_heap_deletion_queue.append(allocator, hdl.*);
        hdl.ptr = 0;
    }
}

fn lookupResource(hdl: ResourceHandle, comptime T: type) *T {
    switch (T) {
        Sampler => {
            std.debug.assert(hdl.kind == .sampler);
            return samplers.lookup(hdl);
        },
        Buffer => {
            std.debug.assert(hdl.kind == .buffer);
            return buffers.lookup(hdl);
        },
        ConstantBuffer => {
            std.debug.assert(hdl.kind == .constant_buffer);
            return constant_buffers.lookup(hdl);
        },
        Texture => {
            std.debug.assert(hdl.kind == .texture);
            return textures.lookup(hdl);
        },
        RenderTarget => {
            std.debug.assert(hdl.kind == .render_target);
            return render_targets.lookup(hdl);
        },
        PipelineState => {
            std.debug.assert(hdl.kind == .pipeline_state);
            return pipeline_states.lookup(hdl);
        },
        else => @compileError("lookupResource does not exist for resource type " ++ @typeName(T)),
    }
}

fn lookupOptionalResource(hdl: ResourceHandle, comptime T: type) ?*T {
    if (hdl.isEmptyResource()) {
        return null;
    } else {
        return lookupResource(hdl, T);
    }
}

pub fn createResource(comptime debug_name: [:0]const u8, desc: anytype) !ResourceHandle {
    return switch (@TypeOf(desc)) {
        gdi.SamplerDesc         => createSampler(debug_name, desc),
        gdi.BufferDesc          => createBuffer(debug_name, desc),
        gdi.TextureDesc         => createTexture(debug_name, desc),
        gdi.ConstantBufferDesc  => createConstantBuffer(debug_name, desc),
        gdi.RenderTargetDesc    => createRenderTarget(debug_name, desc),
        gdi.ShaderParameterDesc => createShaderParameter(debug_name, desc),
        gdi.PipelineStateDesc   => createPipelineState(debug_name, desc),
        gdi.RenderStateDesc     => createRenderState(debug_name, desc),
        else => @compileError("createResource does not exist for resource desc type " ++ @typeName(@TypeOf(desc))),
    };
}

pub fn destroyResource(hdl: ResourceHandle) !void {
    std.debug.assert(hdl.gen != 0);

    switch (hdl.kind) {
        else => {
            log.err("Error: destroyResource does not exist for resource type '{s}' Memory Leak!\n", .{ @tagName(hdl.kind) });
            unreachable;
        },
        .buffer           => try destroyBuffer(buffers.lookup(hdl)),
        .constant_buffer  => try destroyConstantBuffer(constant_buffers.lookup(hdl)),
        .pipeline_state   => try destroyPipelineState(pipeline_states.lookup(hdl)),
        .render_target    => try destroyRenderTarget(render_targets.lookup(hdl)),
        .sampler          => try destroySampler(samplers.lookup(hdl)),
        .texture          => try destroyTexture(textures.lookup(hdl)),
        .render_state     => log.warn("trying to destroy RenderStates is pointless and does nothing", .{}),
        .shader_parameter => log.warn("trying to destroy ShaderParameters is pointless and does nothing", .{}),
    }

    try currentFrame().handle_deletion_queue.append(allocator, hdl);
}

// ************************************************************************************************************************
// cmd bufs impl
// ************************************************************************************************************************

const CmdKind = gdi.CmdKind;

const Cmd = struct {
    kind: CmdKind,
    size: usize,
};

//10 bit size version
//  inline fn unpackCmd(byte0: u8, byte1: u8) Cmd {
//      return .{
//          .kind = @enumFromInt(CmdKind, (byte0 & 0b00111111)),
//          .size = (@intCast(usize, byte1) << 2) + ( @intCast(usize, byte0 & 0b11000000) << 10),
//      };
//  }
//  inline fn packCmd(kind: CmdKind, size: u10) [2]u8 {
//      const data_size_div4 = size>>2;
//      const kind_int = @intFromEnum(kind);
//      std.debug.assert(kind_int < 64);
//      return .{
//          @intCast(u8, kind_int) | @intCast(u8, (data_size_div4 & 0b1100000000)>>8),
//          @intCast(u8, (data_size_div4 & 0b0011111111)),
//      };
//  }
//8 bit size version
inline fn unpackCmd(byte0: u8, byte1: u8) Cmd {
    return .{
        .kind = @enumFromInt(byte0),
        .size = (@as(usize, @intCast(byte1)) << 2),
    };
}
inline fn packCmd(kind: CmdKind, size: u10) [2]u8 {
    return .{
        @intCast(@intFromEnum(kind)),
        @intCast(size>>2),
    };
}

pub const CmdList = struct {
    pub const CmdJump = packed struct {
        pub const cmd_kind = CmdKind.jump;
        ptr: usize,
    };
    pub const CmdSetPipelineState = packed struct {
        pub const cmd_kind = CmdKind.set_pipeline_state;
        pipeline_state: ResourceHandle,
    };
    pub const CmdDraw = struct {
        pub const cmd_kind = CmdKind.draw;
        params: gdi.DrawArgs,
        dsc: gdi.DrawShaderConstants,
    };
    pub const CmdDrawIndexed = struct {
        pub const cmd_kind = CmdKind.draw_indexed;
        params: gdi.DrawIndexedArgs,
        dsc: gdi.DrawShaderConstants,
        idx_buf: ResourceHandle,
    };
    pub const CmdExecuteIndirect = struct {
        pub const cmd_kind = CmdKind.execute_indirect;
        dsc: gdi.DrawShaderConstants,
        arg_buf: ResourceHandle,
        arg_offset: usize,
        count_buf: ResourceHandle,
        kind: gdi.IndirectCmd,
    };
    pub const CmdDispatch = struct {
        pub const cmd_kind = CmdKind.dispatch_compute;
        counts: gdi.DispatchThreadGroupCounts,
    };
    pub const CmdClearBuffer = extern struct {
        pub const cmd_kind = CmdKind.clear_buffer;
        buf: ResourceHandle,
        values: [4]u32,
    };
    pub const CmdCopyBuffer = packed struct {
        pub const cmd_kind = CmdKind.copy_buffer;
        src: ResourceHandle,
        dst: ResourceHandle,
    };
    pub const CmdTransition = packed struct {
        pub const cmd_kind = CmdKind.transition;
        resource: ResourceHandle,
        kind: gdi.ResourceTransition,
    };
    pub const CmdReadBufferAsync = packed struct {
        pub const cmd_kind = CmdKind.read_buffer_async;
        resource: ResourceHandle,
        signal_read_finished: ?*std.atomic.Atomic(bool),
        output_ptr: [*]u8,
        output_len: usize,
    };
    pub const CmdBeginProfileQuery = packed struct {
        pub const cmd_kind = CmdKind.begin_profile_query;
        id: gdi.GpuProfileID,
        name: [*:0]const u8,
    };
    pub const CmdEndProfileQuery = packed struct {
        pub const cmd_kind = CmdKind.end_profile_query;
        id: gdi.GpuProfileID,
    };
    pub const CmdCustom = struct {
        pub const cmd_kind = CmdKind.custom_cmd;
        func: *const gdi.CustomCmdFunc,
        ctx: *anyopaque,
    };

    const Self = @This();

    cur: ?[*] u8 = null,
    end: usize = 0,
    block_node: ?*CmdBlock.Node = null,

    pub fn writeCmd(self: *Self, cmd: anytype) void {
        self.beginCmd(@field(@TypeOf(cmd), "cmd_kind"), @sizeOf(@TypeOf(cmd)));
        self.write(cmd);
        self.endCmd();
    }

    pub fn beginCmd(self: *Self, kind: CmdKind, size: u8) void {
        std.debug.assert((size%4) == 0);

        if ((self.end - @intFromPtr(self.cur)) < (size + CmdBlock.MinSizeBeforeJump)) {
            //need a new block
            var block_node = cmd_block_pool.acquire() catch unreachable;
            var block = &block_node.data;
            block.memory[0] = @intFromEnum(CmdKind.end_of_list);
            block.memory[1] = 0;
            if (self.block_node != null) {
                block.prev_block_node = self.block_node;

                //insert a jump cmd to the new block
                std.debug.assert((self.end - @intFromPtr(self.cur)) >= CmdBlock.JumpCmdSize);
                self.cur.?[0] = @intFromEnum(CmdKind.jump);
                self.cur.?[1] = 8>>2; //8 == ptr size
                //note this assumes endian
                //this is the same as:  ((unsigned char**)(Cur+2))[0] = &block->Memory[0];
                const mem_ptr = @intFromPtr(&block.memory[0]);
                const mem = std.mem.asBytes(&mem_ptr);
                self.cur.?[2] = mem[0];
                self.cur.?[3] = mem[1];
                self.cur.?[4] = mem[2];
                self.cur.?[5] = mem[3];
                self.cur.?[6] = mem[4];
                self.cur.?[7] = mem[5];
                self.cur.?[8] = mem[6];
                self.cur.?[9] = mem[7];
            } else {
                std.debug.assert(block.prev_block_node == null);
            }

            self.block_node = block_node;
            self.cur = @as(?[*] u8, @ptrCast(&block.memory[0]));
            self.end = @intFromPtr(&block.memory[0]) + CmdBlock.MemoryBlockSize;
        }

        const bytes = packCmd(kind, size);
        self.cur.?[0] = bytes[0];
        self.cur.?[1] = bytes[1];
        self.cur.? += 2;
    }

    pub inline fn write(self: *Self, data: anytype) void {
        const DataType = @TypeOf(data);
        for (std.mem.asBytes(&data), 0..) |d, i| {
            self.cur.?[i] = d;
        }
        self.cur.? += @sizeOf(DataType);
    }

    pub inline fn endCmd(self: *Self) void {
        std.debug.assert(@intFromPtr(self.cur) < ((@intFromPtr(&self.block_node.?.data.memory[0]) + CmdBlock.MemoryBlockSize) - CmdBlock.JumpCmdSize));
        std.debug.assert(@intFromPtr(self.cur) < (self.end - CmdBlock.JumpCmdSize));
        self.cur.?[0] = @intFromEnum(CmdKind.end_of_list);
        self.cur.?[1] = 0;
    }

    pub fn findFirstBlock(self: *Self) *CmdBlock {
        var node = self.block_node.?;
        while (node.data.prev_block_node != null) {
            node = node.data.prev_block_node.?;
        }
        return &node.data;
    }
};

const CmdBlock = struct {
    const MemoryBlockSize: usize = 4*1024 - 8*2;
    const JumpCmdSize: usize = 1 + 8;
    const MinSizeBeforeJump: usize = 2 * JumpCmdSize;
    pub const Node = CmdBlockPool.Node;

    memory: [MemoryBlockSize] u8,
    prev_block_node: ?*Node,

    pub fn init(self: *CmdBlock) !void {
        self.prev_block_node = null;
    }

    pub fn reinit(self: *CmdBlock) void {
        self.prev_block_node = null;
    }

    pub fn deinit(self: *CmdBlock) void {
        _ = self;
    }
};

pub const TranslatedCmdBuf = struct {
    pub const Node = TranslatedCmdBufPool.Node;

    cl: PlatformCommandList,

    pub fn init(self: *TranslatedCmdBuf) !void {
        self.cl = try dx.device.createCommandList(.direct, currentFrame().command_allocator, null);
        try self.cl.close();
    }

    pub fn reinit(self: *TranslatedCmdBuf) void {
        _ = self;
    }

    pub fn deinit(self: *TranslatedCmdBuf) void {
        self.cl.deinit();
    }
};

inline fn setDrawConstants(cl: *PlatformCommandList, root_sig: *const RootSignature, constants: []const u32) void {
    _ = root_sig;
    cl.setGraphicsRoot32BitConstants(0, constants, 0);
}

fn setupDescriptorTable(cl: *PlatformCommandList, root_sig: *RootSignature, comptime is_compute: bool) void {
    if (root_sig.descriptor_root_idx >= 0) {
        const bind_set = root_sig.descriptor_bind_set;
        bind_set.updateGpuDescriptorsIfDirty();
        if (is_compute) {
            cl.setComputeRootDescriptorTable(@intCast(root_sig.descriptor_root_idx), bind_set.desc_range_gpu);
        } else {
            cl.setGraphicsRootDescriptorTable(@intCast(root_sig.descriptor_root_idx), bind_set.desc_range_gpu);
        }
    }
}

pub fn translateCmdBuf(cb: *gdi.CmdBuf) !void {
    std.debug.assert(null != cb.cl.block_node); //check that the list isn't empty (maybe this should just skip instead and set a flag?)
    std.debug.assert(null == cb.tcb); //check that it is not already translated

    var cmd_block = cb.cl.findFirstBlock();

    //acquire a new command list
    var tcb = try translated_cmd_buf_pool.acquire();
    cb.tcb = tcb;
    var cl = &tcb.data.cl;

    //prepare for recording
    var frame = currentFrame();
    try cl.reset(frame.command_allocator, null);
    try cl.setDescriptorHeaps(&gpu_descriptor_heap, null);

    var cur_root_sig_draw: ?*RootSignature = null;
    var cur_root_sig_compute: ?*RootSignature = null;
    var rt_format_group_idx: ?usize = null;

    var cur:[*]const u8 = @ptrCast(&cmd_block.memory[0]);
    while (cur[0] != @intFromEnum(CmdKind.end_of_list)) {
        //unpack cmd and size
        const cmd = unpackCmd(cur[0], cur[1]);
        const size = cmd.size;
        cur += 2;

        //std.debug.print("[translate cmd]: cmd({}) size({})\n", .{cmd, size});
        switch (cmd.kind) {
            else => { std.debug.assert(false); }, //invalid cmd
            .jump => {
                cur = @ptrFromInt(readCmd(&cur, size, CmdList.CmdJump).ptr);
            },
            .begin_render_pass => {
                const pass = readCmd(&cur, size, gdi.RenderPass);
                //std.debug.print("begin_render_pass: {s}\n", .{pass.debug_name});
                var ds_view: ?PlatformCpuDescriptorHandle = null;
                var rt_views: [8]PlatformCpuDescriptorHandle = undefined;
                var rt_group = RenderTargetFormatGroup{
                    .ds = d3d.DXGI_FORMAT_UNKNOWN,
                    .rt = undefined,
                    .rt_count = 0,
                };
                var size_x: u32 = 0;
                var size_y: u32 = 0;

                //transition all render targets to the state we need and store their views
                var batcher = PlatformResourceBarrierBatcher(10){};
                var rt_idx: usize = 0;
                while (rt_idx < max_render_target_count) : (rt_idx += 1) {
                    if (pass.render_targets[rt_idx]) |rt_hdl| {
                        var rt = render_targets.lookup(rt_hdl);
                        rt.gpu_resource.transitionIfNecessary(&batcher, cl, d3d.D3D12_RESOURCE_STATE_RENDER_TARGET);
                        rt_views[rt_idx] = rt.views[0]; //TODO: render target array slice
                        rt_group.rt[rt_idx] = rt.format;
                        rt_group.rt_count += 1;
                        if ((size_x == 0) or (size_y == 0)) {
                            size_x = rt.desc.size_x;
                            size_y = rt.desc.size_y;
                        } else {
                            std.debug.assert(size_x == rt.desc.size_x);
                            std.debug.assert(size_y == rt.desc.size_y);
                        }
                    } else {
                        rt_views[rt_idx] = .{ .ptr = 0, };
                        rt_group.rt[rt_idx] = d3d.DXGI_FORMAT_UNKNOWN;
                    }
                }
                if (pass.depth_stencil) |ds_hdl| {
                    var rt = render_targets.lookup(ds_hdl);
                    rt_group.ds = rt.format;
                    if (pass.is_depth_read_only) {
                        rt.gpu_resource.transitionIfNecessary(&batcher, cl, d3d.D3D12_RESOURCE_STATE_DEPTH_READ);
                        ds_view = rt.read_only_views[pass.depth_stencil_slice];
                    } else {
                        rt.gpu_resource.transitionIfNecessary(&batcher, cl, d3d.D3D12_RESOURCE_STATE_DEPTH_WRITE);
                        ds_view = rt.views[pass.depth_stencil_slice];
                    }
                    if ((size_x == 0) or (size_y == 0)) {
                        size_x = rt.desc.size_x;
                        size_y = rt.desc.size_y;
                    } else {
                        std.debug.assert(size_x == rt.desc.size_x);
                        std.debug.assert(size_y == rt.desc.size_y);
                    }
                }
                batcher.flush(cl.*);

                std.debug.assert( (size_x != 0) and (size_y != 0) );

                //clear depth if requested
                if (pass.depth_access.begin == .clear) {
                    const ds = render_targets.lookup(pass.depth_stencil.?);
                    if (pass.stencil_access.begin == .clear) {
                        cl.clearDepthStencilView(ds.views[pass.depth_stencil_slice], ds.desc.clear_value.depth_stencil.depth, ds.desc.clear_value.depth_stencil.stencil);
                    } else {
                        cl.clearDepthStencilView(ds.views[pass.depth_stencil_slice], ds.desc.clear_value.depth_stencil.depth, null);
                    }
                } else if (pass.stencil_access.begin == .clear) {
                    const ds = render_targets.lookup(pass.depth_stencil.?);
                    cl.clearDepthStencilView(ds.views[pass.depth_stencil_slice], null, ds.desc.clear_value.depth_stencil.stencil);
                }

                //clear render targets if requested
                rt_idx = 0;
                while (rt_idx < max_render_target_count) : (rt_idx += 1) {
                    if (pass.render_targets[rt_idx]) |rt_hdl| {
                        var rt = render_targets.lookup(rt_hdl);
                        const access = pass.render_target_access[rt_idx];
                        if (access.begin == .clear) {
                            //TODO: render target array slice
                            cl.clearRenderTargetView(rt.views[0], rt.desc.clear_value.color);
                        }
                        std.debug.assert(access.end != .clear); //invalid state
                    }
                }

                if (rt_group.rt_count > 0) {
                    cl.setRenderTargetViews(rt_views[0..rt_group.rt_count], ds_view);
                } else {
                    cl.setRenderTargetViewsDepthOnly(ds_view.?);
                }
                cl.setViewportAndScissor(0, 0, size_x, size_y, 0.0, 1.0);

                std.debug.assert(rt_format_group_idx == null);
                rt_format_group_idx = findOrCreateRenderTargetFormatGroup(rt_group);
            },
            .end_render_pass => {
                std.debug.assert(size == 0);
                rt_format_group_idx = null;
                //cl.endRenderPass();
            },
            .draw => {
                const d = readCmd(&cur, size, CmdList.CmdDraw);
                const params = d.params;
                const constants = d.dsc;
                setDrawConstants(cl, cur_root_sig_draw.?, constants.u[0..constants.u.len]);
                cl.drawInstanced(params.vertex_count_per_instance, params.instance_count, params.start_vertex_loc, params.start_instance_loc);
            },
            .draw_indexed => {
                const d = readCmd(&cur, size, CmdList.CmdDrawIndexed);
                const buf = lookupResource(d.idx_buf, Buffer);
                const params = d.params;
                const constants = d.dsc;
                cl.setIndexBufferView(&buf.idx_buf_view);
                setDrawConstants(cl, cur_root_sig_draw.?, constants.u[0..constants.u.len]);
                cl.drawIndexedInstanced(params.index_count_per_instance, params.instance_count, params.start_index_loc, params.base_vertex_loc, params.start_instance_loc);
            },
            .execute_indirect => {
                const d = readCmd(&cur, size, CmdList.CmdExecuteIndirect);
                const arg_buf = lookupResource(d.arg_buf, Buffer);
                const count_buf = lookupOptionalResource(d.count_buf, Buffer);
                const constants = d.dsc;
                setDrawConstants(cl, cur_root_sig_draw.?, constants.u[0..constants.u.len]);

                var count_resource: ?PlatformResource = null;
                var max_cmd_count: u32 = 1;
                var batcher = PlatformResourceBarrierBatcher(2){};
                arg_buf.gpu_resource.transitionIfNecessary(&batcher, cl, d3d.D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
                if (count_buf) |buf| {
                    count_resource = buf.gpu_resource.resource;
                    buf.gpu_resource.transitionIfNecessary(&batcher, cl, d3d.D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);

                    //compute the max command count
                    const stride: usize = switch (d.kind) {
                        .draw => @sizeOf(gdi.DrawArgs),
                        .draw_indexed => @sizeOf(gdi.DrawIndexedArgs),
                        .dispatch_mesh => @sizeOf(gdi.DispatchMeshArgs),
                    };
                    max_cmd_count = arg_buf.desc.size / @as(u32, @intCast(stride));
                }
                batcher.flush(cl.*);

                const sig = switch (d.kind) {
                    .draw => indirect_draw_sig,
                    .draw_indexed => indirect_draw_indexed_sig,
                    .dispatch_mesh => indirect_dispatch_mesh_sig,
                };
                cl.executeIndirect(sig, max_cmd_count, arg_buf.gpu_resource.resource, d.arg_offset, count_resource, 0);
            },
            .set_pipeline_state => {
                const d = readCmd(&cur, size, CmdList.CmdSetPipelineState);
                const state = lookupResource(d.pipeline_state, PipelineState);
                const pso = try state.getOrCreatePipelineState(rt_format_group_idx);
                cl.setPipelineState(pso);
                switch (state.pipeline_state_desc) {
                    .compute =>
                        if (cur_root_sig_compute != state.root_sig) {
                            cur_root_sig_compute = state.root_sig;
                            cl.setComputeRootSignature(state.root_sig.rs);
                            setupDescriptorTable(cl, state.root_sig, true);
                        },
                    .graphics => {
                        if (cur_root_sig_draw != state.root_sig) {
                            cur_root_sig_draw = state.root_sig;
                            cl.setGraphicsRootSignature(state.root_sig.rs);
                            setupDescriptorTable(cl, state.root_sig, false);
                        }
                        cl.setPrimTopology(state.prim_topology);
                    },
                    .mesh => {
                        //PNC: i have no idea if I need to do either of these or which I would need to do
                        if (cur_root_sig_draw != state.root_sig) {
                            cur_root_sig_draw = state.root_sig;
                            cl.setGraphicsRootSignature(state.root_sig.rs);
                            setupDescriptorTable(cl, state.root_sig, false);
                        }
                        if (cur_root_sig_compute != state.root_sig) {
                            cur_root_sig_compute = state.root_sig;
                            cl.setComputeRootSignature(state.root_sig.rs);
                            setupDescriptorTable(cl, state.root_sig, true);
                        }
                        cl.setPrimTopology(state.prim_topology);
                    },
                }
            },
            .dispatch_compute => {
                const d = readCmd(&cur, size, CmdList.CmdDispatch);
                cl.dispatch(d.counts[0], d.counts[1], d.counts[2]);
            },
            .clear_buffer => {
                const d = readCmd(&cur, size, CmdList.CmdClearBuffer);
                var buf = lookupResource(d.buf, Buffer);

                //transition buffer if necessary
                var batcher = PlatformResourceBarrierBatcher(1){};
                buf.gpu_resource.transitionIfNecessary(&batcher, cl, d3d.D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                batcher.flush(cl.*);

                //build a gpu visible descriptor
                const desc_range_idx = gpu_descriptor_heap.allocIdxRange(1);
                const gpu_uav = gpu_descriptor_heap.idxToGpuHandle(desc_range_idx);
                const cpu_uav = gpu_descriptor_heap.idxToCpuAccessHandle(desc_range_idx);
                dx.device.copyDescriptorSimple(cpu_uav, buf.uav, d3d.D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

                //now do the actual clear
                cl.clearUnorderedAccessViewUint(gpu_uav, buf.uav, buf.gpu_resource.resource, d.values);
            },
            .copy_buffer => {
                const d = readCmd(&cur, size, CmdList.CmdCopyBuffer);
                var src = lookupResource(d.src, Buffer);
                var dst = lookupResource(d.dst, Buffer);
                std.debug.assert(src.desc.size == dst.desc.size); //sizes must match
                var batcher = PlatformResourceBarrierBatcher(2){};
                src.gpu_resource.transitionIfNecessary(&batcher, cl, d3d.D3D12_RESOURCE_STATE_COPY_SOURCE);
                dst.gpu_resource.transitionIfNecessary(&batcher, cl, d3d.D3D12_RESOURCE_STATE_COPY_DEST);
                batcher.flush(cl.*);
                cl.copyResource(dst.gpu_resource.resource, src.gpu_resource.resource);
            },
            .transition => {
                var batcher = PlatformResourceBarrierBatcher(2){};
                const d = readCmd(&cur, size, CmdList.CmdTransition);
                switch (d.resource.kind) {
                    else => {
                        log.err("CmdTransition has invalid resource kind {}\n", .{ d.resource.kind });
                        std.debug.assert(false); //invalid resource kind
                    },
                    .render_target => {
                        const rt = lookupResource(d.resource, RenderTarget);
                        switch (d.kind) {
                            .shader_visible => { rt.gpu_resource.transitionIfNecessary(&batcher, cl, d3d.D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE); },
                            .uav            => { rt.gpu_resource.transitionIfNecessary(&batcher, cl, d3d.D3D12_RESOURCE_STATE_UNORDERED_ACCESS); },
                            .writable       => {
                                const state = if (formatQueryInfo(rt.desc.format).is_depth_format) d3d.D3D12_RESOURCE_STATE_DEPTH_WRITE else d3d.D3D12_RESOURCE_STATE_RENDER_TARGET;
                                rt.gpu_resource.transitionIfNecessary(&batcher, cl, @intCast(state));
                            },
                            .default, .indirect_args => unreachable, //invalid kinds
                        }
                    },
                    .buffer => {
                        const buf = lookupResource(d.resource, Buffer);
                        switch (d.kind) {
                            .shader_visible, .writable => unreachable, //invalid kinds
                            .uav => {
                                buf.gpu_resource.transitionIfNecessary(&batcher, cl, d3d.D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                                batcher.unorderedAccess(cl.*, buf.gpu_resource.resource);
                            },
                            .default => {
                                const states = d3d.D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER|d3d.D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE|d3d.D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                                buf.gpu_resource.transitionIfNecessary(&batcher, cl, states);
                            },
                            .indirect_args => {
                                buf.gpu_resource.transitionIfNecessary(&batcher, cl, d3d.D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
                                batcher.unorderedAccess(cl.*, buf.gpu_resource.resource);
                            },
                        }
                    },
                    .texture => {
                        const tex = lookupResource(d.resource, Texture);
                        switch (d.kind) {
                            else => unreachable, //invalid kinds
                            .default, .shader_visible => {
                                const states = d3d.D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
                                tex.gpu_resource.transitionIfNecessary(&batcher, cl, states);
                            },
                            .uav => {
                                tex.gpu_resource.transitionIfNecessary(&batcher, cl, d3d.D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                                batcher.unorderedAccess(cl.*, tex.gpu_resource.resource);
                            },
                        }
                    },
                }
                batcher.flush(cl.*);
            },
            .read_buffer_async => {
                const d = readCmd(&cur, size, CmdList.CmdReadBufferAsync);
                std.debug.assert(d.resource.kind == .buffer);
                try read_back_requests.append(.{
                    .resource = d.resource,
                    .signal = d.signal_read_finished,
                    .output = d.output_ptr[0..d.output_len],
                    .ready_frame_num = (frame_num + 1),
                });
            },
            .begin_profile_query => {
                const d = readCmd(&cur, size, CmdList.CmdBeginProfileQuery);
                cl.beginEvent(@truncate(d.id), d.name);
                D3D.beginGpuProfileQuery(dx, cl, d.id);
            },
            .end_profile_query => {
                const d = readCmd(&cur, size, CmdList.CmdEndProfileQuery);
                D3D.endGpuProfileQuery(dx, cl, d.id);
                cl.endEvent();
            },
            .custom_cmd => {
                const d = readCmd(&cur, size, CmdList.CmdCustom);
                d.func(d.ctx, cl.*);

                //these may be dirty now
                cur_root_sig_compute = null;
                cur_root_sig_draw = null;
                try cl.setDescriptorHeaps(&gpu_descriptor_heap, null);
            },
        }
    }

    //end recording
    std.debug.assert(cur == cb.cl.cur);
    try cl.close();
    try recycleCmdBlocks(cb);
}

inline fn readCmd(cur: *[*]const u8, size: usize, comptime T: type) T {
    std.debug.assert(@sizeOf(T) == size);
    var data: T = undefined;
    @memcpy(@as([*]u8, @ptrCast(&data))[0..@sizeOf(T)], cur.*);
    //std.debug.print("readCmd({}, {}) => {}\n", .{@intFromPtr(cur.*), @sizeOf(T), data});
    cur.* += @sizeOf(T);
    return data;
}

pub fn submitCmdBuf(cb: *gdi.CmdBuf) void {
    if (option_use_async_copy_queue) {
        dx.command_queue.wait(copy_command_queue_fence, copy_frame_num) catch unreachable;
    }
    dx.command_queue.executeCommandList(cb.tcb.?.data.cl);
}

pub fn releaseCmdBuf(cb: *gdi.CmdBuf) !void {
    try recycleCmdBlocks(cb);
    if (cb.tcb) |tcb| {
        try translated_cmd_buf_pool.release(tcb);
        cb.tcb = null;
    }
}

fn recycleCmdBlocks(cb: *gdi.CmdBuf) !void {
    var node = cb.cl.block_node;
    while (node) |n| {
        const next = n.data.prev_block_node;
        n.data.prev_block_node = null;
        @memset(@as([*]u8, @ptrCast(&n.data.memory[0]))[0..n.data.memory.len], 0xde); //for debugging reasons only
        try cmd_block_pool.release(n);
        node = next;
    }
    cb.cl.block_node = null;
}

// ************************************************************************************************************************
// resources
// ************************************************************************************************************************

const GpuResource = struct {
    resource: PlatformResource,
    states: d3d.D3D12_RESOURCE_STATES,
    pinned_memory: ?[*]u8,
    size_in_bytes: u64,

    pub fn deinit(self: *GpuResource) !void {
        if (null != self.pinned_memory) {
            self.resource.unmapEntireBuffer();
            self.pinned_memory = null;
        }
        try scheduleFrameResourceForDeletion(self.resource);
        self.size_in_bytes = 0;
        self.resource = undefined;
        self.states = undefined;
    }

    pub fn transitionIfNecessary(self: *GpuResource, batcher: anytype, cl: *PlatformCommandList, states: d3d.D3D12_RESOURCE_STATES) void {
        if (batcher.transitionIfNecessary(cl.*, self.resource, self.states, states)) {
            self.states = states;
        }
    }
};

pub const ResourceKind = enum(u4) {
    none,
    buffer,
    constant_buffer,
    descriptor_bind_set,
    pipeline_state,
    render_state,
    render_target,
    render_target_format_group,
    root_signature,
    sampler,
    shader_parameter,
    texture,
};

pub const ResourceGeneration = u8;

pub const ResourceHandle = packed struct {
    pub const empty = ResourceHandle{};

    kind: ResourceKind = .none,
    gen: ResourceGeneration = 0,
    idx: u20 = 0,

    pub fn eql(a: ResourceHandle, b: ResourceHandle) bool {
        return (a.kind == b.kind) and (a.gen == b.gen) and (a.idx == b.idx);
    }

    pub fn isEmptyResource(a: ResourceHandle) bool {
        return a.eql(ResourceHandle.empty);
    }
};

fn ResourceTable(comptime T: type) type {
    return struct {
        const Self = @This();

        is_free: std.DynamicBitSetUnmanaged,
        generations: []ResourceGeneration, //this is just for debugging and could go away in optimized builds
        items: []T,
        kind: ResourceKind,
        mutex: std.Thread.Mutex,

        pub fn init(a: std.mem.Allocator, capacity: usize, kind: ResourceKind) !Self {
            var arr0 = try std.DynamicBitSetUnmanaged.initFull(a, capacity);
            errdefer arr0.deinit(a);
            var arr1 = try a.alloc(ResourceGeneration, capacity);
            errdefer a.free(arr1);
            return Self{
                .is_free = arr0,
                .generations = arr1,
                .items = try a.alloc(T, capacity),
                .kind = kind,
                .mutex = .{},
            };
        }

        pub fn deinit(self: *Self, a: std.mem.Allocator) !void {
            self.is_free.deinit(a);
            a.free(self.generations);
            a.free(self.items);
        }

        pub fn lookup(self: *Self, hdl: ResourceHandle) *T {
            self.validateHandle(hdl);
            return &self.items[hdl.idx];
        }

        pub fn create(self: *Self) ResourceHandle {
            self.mutex.lock();
            defer self.mutex.unlock();
            const idx = self.is_free.toggleFirstSet().?;
            const ov = @addWithOverflow(self.generations[idx], 1);
            if (0 != ov[1]) {
                self.generations[idx] = 1;
            } else {
                self.generations[idx] = ov[0];
            }
            return ResourceHandle{
                .kind = self.kind,
                .gen = self.generations[idx],
                .idx = @intCast(idx),
            };
        }

        pub fn destroy(self: *Self, hdl: ResourceHandle) void {
            self.mutex.lock();
            defer self.mutex.unlock();
            self.validateHandle(hdl);
            self.is_free.set(hdl.idx);
        }

        pub inline fn validateHandle(self: *const Self, hdl: ResourceHandle) void {
            std.debug.assert(hdl.gen != 0);
            std.debug.assert(hdl.kind == self.kind);
            std.debug.assert(!self.is_free.isSet(hdl.idx));
            std.debug.assert(hdl.gen == self.generations[hdl.idx]);
        }
    };
}

// ************************************************************************************************************************
// samplers
// ************************************************************************************************************************

pub const Sampler = struct {
    desc: gdi.SamplerDesc,
    debug_name: [:0]u8,
};

pub fn createSampler(comptime debug_name: [:0]const u8, desc: gdi.SamplerDesc) !ResourceHandle {
    var hdl = samplers.create();
    var sam = samplers.lookup(hdl);
    sam.desc = desc;
    sam.debug_name = try allocator.dupeZ(u8, debug_name);
    return hdl;
}

pub fn destroySampler(self: *Sampler) !void {
    allocator.free(self.debug_name);
    self.* = undefined;
}

// ************************************************************************************************************************
// buffer
// ************************************************************************************************************************

pub const Buffer = struct {
    desc: gdi.BufferDesc,
    gpu_resource: GpuResource,
    view: PlatformCpuDescriptorHandle,
    uav: PlatformCpuDescriptorHandle,
    idx_buf_view: d3d.D3D12_INDEX_BUFFER_VIEW,
    debug_name: [:0]u8,
};

fn createBuffer(comptime debug_name: [:0]const u8, desc: gdi.BufferDesc) !ResourceHandle {
    var hdl = buffers.create();
    var buf = buffers.lookup(hdl);
    buf.desc = desc;
    buf.debug_name = try allocator.dupeZ(u8, debug_name);
    std.debug.assert(!(desc.is_used_for_readback and desc.allow_unordered_access)); //these flags cannot be used together

    //TODO: use amd's allocator
    //TODO: use amd's allocator
    //TODO: use amd's allocator
    buf.gpu_resource = .{
        .resource = try dx.device.createCommittedResource(
            debug_name,
            &d3d.D3D12_HEAP_PROPERTIES{
                .Type = (if (desc.is_used_for_readback) d3d.D3D12_HEAP_TYPE_READBACK else d3d.D3D12_HEAP_TYPE_DEFAULT),
                .CPUPageProperty = d3d.D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                .MemoryPoolPreference = d3d.D3D12_MEMORY_POOL_UNKNOWN,
                .CreationNodeMask = 1,
                .VisibleNodeMask = 1,
            },
            &d3d.D3D12_RESOURCE_DESC{
                .Dimension = d3d.D3D12_RESOURCE_DIMENSION_BUFFER,
                .Alignment = d3d.D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT,
                .Width = @intCast(desc.size),
                .Height = 1,
                .DepthOrArraySize = 1,
                .MipLevels = 1,
                .Format = d3d.DXGI_FORMAT_UNKNOWN,
                .SampleDesc = .{ .Count = 1, .Quality = 0, },
                .Layout = d3d.D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                .Flags = (if (desc.allow_unordered_access) d3d.D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS else d3d.D3D12_RESOURCE_FLAG_NONE),
            },
            d3d.D3D12_RESOURCE_STATE_COMMON,
        ),
        .states = d3d.D3D12_RESOURCE_STATE_COMMON,
        .pinned_memory = null,
        .size_in_bytes = @intCast(desc.size),
    };

    if (desc.is_used_for_readback) {
        //reading from readback resources is a bad idea
        buf.view = dx.null_descriptor_srv;
    } else {
        buf.view = dx.staging_descriptor_heap.alloc();
        dx.device.createShaderResourceView(
            buf.gpu_resource.resource,
            d3d.D3D12_SHADER_RESOURCE_VIEW_DESC{
                .Format = formatQueryInfo(desc.format).format,
                .ViewDimension = d3d.D3D12_SRV_DIMENSION_BUFFER,
                .Shader4ComponentMapping = d3d.D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                .unnamed_0 = .{
                    .Buffer = .{
                        .NumElements = (desc.size / formatQueryInfo(desc.format).size_in_bytes),
                        .StructureByteStride = 0,
                        .FirstElement = 0,
                        .Flags = d3d.D3D12_BUFFER_SRV_FLAG_NONE,
                    },
                },
            },
            buf.view
        );
    }

    if (desc.allow_unordered_access) {
        buf.uav = dx.staging_descriptor_heap.alloc();
        dx.device.createUnorderedAccessView(
            buf.gpu_resource.resource,
            null, //counter resource
            d3d.D3D12_UNORDERED_ACCESS_VIEW_DESC{
                .Format = d3d.DXGI_FORMAT_R32_TYPELESS,
                .ViewDimension = d3d.D3D12_UAV_DIMENSION_BUFFER,
                .unnamed_0 = .{
                    .Buffer = .{
                        .FirstElement = 0,
                        .NumElements = (desc.size / 4),
                        .StructureByteStride = 0,
                        .CounterOffsetInBytes = 0,
                        .Flags = d3d.D3D12_BUFFER_UAV_FLAG_RAW,
                    },
                },
            },
            buf.uav
        );
    } else {
        buf.uav = dx.null_descriptor_uav;
    }

    if (desc.allow_index_buffer) {
        std.debug.assert((desc.format == .r32_uint) or (desc.format == .r16_uint));
        buf.idx_buf_view = .{
            .BufferLocation = buf.gpu_resource.resource.getGpuVirtualAddress(),
            .SizeInBytes = desc.size,
            .Format = formatQueryInfo(desc.format).format,
        };
    }

    return hdl;
}

fn destroyBuffer(self: *Buffer) !void {
    try self.gpu_resource.deinit();
    dx.staging_descriptor_heap.free(self.view);
    dx.staging_descriptor_heap.free(self.uav);
    allocator.free(self.debug_name);
    self.* = undefined;
}

// ************************************************************************************************************************
// constant buffers
// ************************************************************************************************************************

pub const ConstantBuffer = struct {
    desc: gdi.ConstantBufferDesc,
    gpu_resource: GpuResource,
    //heap_ptr: ?*anyopaque,
    //resource_offset: u64,
    view: PlatformCpuDescriptorHandle,
    debug_name: [:0]const u8,
};

pub fn createConstantBuffer(comptime debug_name: [:0]const u8, desc: gdi.ConstantBufferDesc) !ResourceHandle {
    const size = std.mem.alignForward(u32, desc.size, 256); //constant buffers must be 256 byte aligned

    var hdl = constant_buffers.create();
    var cb = constant_buffers.lookup(hdl);
    cb.desc.size = size;

    //TODO: use amd's allocator
    //TODO: use amd's allocator
    //TODO: use amd's allocator
    cb.gpu_resource = .{
        .resource = try dx.device.createCommittedResource(
            "ConstantBuffer.<unknown>",
            &d3d.D3D12_HEAP_PROPERTIES{
                .Type = d3d.D3D12_HEAP_TYPE_DEFAULT,
                .CPUPageProperty = d3d.D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                .MemoryPoolPreference = d3d.D3D12_MEMORY_POOL_UNKNOWN,
                .CreationNodeMask = 1,
                .VisibleNodeMask = 1,
            },
            &d3d.D3D12_RESOURCE_DESC{
                .Dimension = d3d.D3D12_RESOURCE_DIMENSION_BUFFER,
                .Alignment = d3d.D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT,
                .Width = @intCast(size),
                .Height = 1,
                .DepthOrArraySize = 1,
                .MipLevels = 1,
                .Format = d3d.DXGI_FORMAT_UNKNOWN,
                .SampleDesc = .{ .Count = 1, .Quality = 0, },
                .Layout = d3d.D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                .Flags = d3d.D3D12_RESOURCE_FLAG_NONE,
            },
            d3d.D3D12_RESOURCE_STATE_COPY_DEST,
        ),
        .states = d3d.D3D12_RESOURCE_STATE_COPY_DEST,
        .pinned_memory = null,
        .size_in_bytes = @intCast(size),
    };

    cb.view = dx.staging_descriptor_heap.alloc();
    dx.device.createConstantBufferView(
        d3d.D3D12_CONSTANT_BUFFER_VIEW_DESC{
            .SizeInBytes = size,
            .BufferLocation = cb.gpu_resource.resource.getGpuVirtualAddress(),
        },
        cb.view
    );

    cb.debug_name = try allocator.dupeZ(u8, debug_name);
    return hdl;
}

pub fn destroyConstantBuffer(cb: *ConstantBuffer) !void {
    try cb.gpu_resource.deinit();
    allocator.free(cb.debug_name);
    cb.desc = undefined;
    cb.view = undefined;
}

// ************************************************************************************************************************
// textures
// ************************************************************************************************************************

pub const Texture = struct {
    pub const max_mip_count = gdi.TextureDesc.max_mip_count;
    view: PlatformCpuDescriptorHandle,
    array_view: PlatformCpuDescriptorHandle,
    uav_mip_views: [max_mip_count]PlatformCpuDescriptorHandle,
    format: gdi.Format,
    array_count: u32,
    mip_count: u32,
    gpu_resource: GpuResource,
    mip_descs: [max_mip_count]gdi.TextureMipDesc,
    debug_name: [:0]u8,
};

pub fn createTexture(comptime debug_name: [:0]const u8, desc: gdi.TextureDesc) !ResourceHandle {
    std.debug.assert(desc.mip_count > 0);
    std.debug.assert(desc.mip_descs[0].size_x > 0);
    std.debug.assert(desc.mip_descs[0].size_y > 0);
    var hdl = textures.create();
    var tex = textures.lookup(hdl);

    tex.format = desc.format;
    tex.array_count = desc.array_count;
    tex.mip_count = desc.mip_count;

    if (desc.is_cube_map) {
        //should this be 6 instead??
        std.debug.assert(1 == desc.mip_descs[0].size_z);
        std.debug.assert(1 == desc.array_count); //could be supported as cube map arrays
    }

    const is_3d = (desc.mip_descs[0].size_z > 1);
    const dxgi_format = formatQueryInfo(desc.format).format;
    const d3d_desc = d3d.D3D12_RESOURCE_DESC{
        .Alignment = 0,
        .Dimension = if (is_3d) d3d.D3D12_RESOURCE_DIMENSION_TEXTURE3D else d3d.D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        .Width = @intCast(desc.mip_descs[0].size_x),
        .Height = @intCast(desc.mip_descs[0].size_y),
        .DepthOrArraySize = 
            if (desc.is_cube_map) 6
            else if (is_3d) @intCast(desc.mip_descs[0].size_z)
            else @intCast(tex.array_count),
        .MipLevels = @intCast(tex.mip_count),
        .Format = dxgi_format,
        .SampleDesc = .{ .Count = 1, .Quality = 0 },
        .Layout = d3d.D3D12_TEXTURE_LAYOUT_UNKNOWN,
        .Flags = if (desc.allow_unordered_access) d3d.D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS else d3d.D3D12_RESOURCE_FLAG_NONE,
    };
    const heap_info = d3d.D3D12_HEAP_PROPERTIES{
        .Type = d3d.D3D12_HEAP_TYPE_DEFAULT,
        .CPUPageProperty = d3d.D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        .MemoryPoolPreference = d3d.D3D12_MEMORY_POOL_UNKNOWN,
        .CreationNodeMask = 1,
        .VisibleNodeMask = 1,
    };
    //TODO: use amd's allocator
    //TODO: use amd's allocator
    //TODO: use amd's allocator
    tex.gpu_resource.resource = try dx.device.createCommittedResource("Texture.<unknown>", &heap_info, &d3d_desc, d3d.D3D12_RESOURCE_STATE_COPY_DEST);
    tex.gpu_resource.states = d3d.D3D12_RESOURCE_STATE_COPY_DEST;
    tex.gpu_resource.pinned_memory = null;
    tex.gpu_resource.size_in_bytes = 0; //TODO: size

    var layouts: [Texture.max_mip_count]d3d.D3D12_PLACED_SUBRESOURCE_FOOTPRINT = undefined;
    var num_rows: [Texture.max_mip_count]u32 = undefined;
    var row_size_in_bytes: [Texture.max_mip_count]u64 = undefined;

    dx.device.device.*.lpVtbl.*.GetCopyableFootprints.?(
        dx.device.device,
        &d3d_desc,
        0, //first subresource
        @intCast(tex.mip_count), //num subresources
        0, //base offset
        &layouts[0],
        &num_rows[0],
        &row_size_in_bytes[0],
        &tex.gpu_resource.size_in_bytes
    );
    tex.gpu_resource.size_in_bytes *= @intCast(tex.array_count);

    var i: usize = 0;
    while (i < tex.mip_count) : (i += 1) {
        const layout = layouts[i];
        std.debug.assert(layout.Footprint.Format == dxgi_format);
        tex.mip_descs[i] = .{
            .size = @intCast(layout.Footprint.RowPitch * num_rows[i] * layout.Footprint.Depth),
            .offset = @intCast(layout.Offset),
            .level = @intCast(i),
            .size_x = layout.Footprint.Width,
            .size_y = layout.Footprint.Height,
            .size_z = layout.Footprint.Depth,
            .stride_x = layout.Footprint.RowPitch,
            .rows_y = num_rows[i],
            .format = tex.format,
            .backend_format = dxgi_format,
        };
    }

    //cube/2d/3d view
    tex.view = dx.staging_descriptor_heap.alloc();
    if (desc.is_cube_map) {
        dx.device.createShaderResourceView(
            tex.gpu_resource.resource,
            d3d.D3D12_SHADER_RESOURCE_VIEW_DESC{
                .Format = formatQueryInfo(desc.format).format,
                .ViewDimension = d3d.D3D12_SRV_DIMENSION_TEXTURECUBE,
                .Shader4ComponentMapping = d3d.D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                .unnamed_0 = .{
                    .TextureCube = .{
                        .MostDetailedMip = 0,
                        .MipLevels = tex.mip_count,
                        .ResourceMinLODClamp = 0.0,
                    },
                },
            },
            tex.view
        );
    } else if (is_3d) {
        dx.device.createShaderResourceView(
            tex.gpu_resource.resource,
            d3d.D3D12_SHADER_RESOURCE_VIEW_DESC{
                .Format = formatQueryInfo(desc.format).format,
                .ViewDimension = d3d.D3D12_SRV_DIMENSION_TEXTURE3D,
                .Shader4ComponentMapping = d3d.D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                .unnamed_0 = .{
                    .Texture3D = .{
                        .MostDetailedMip = 0,
                        .MipLevels = tex.mip_count,
                        .ResourceMinLODClamp = 0.0,
                    },
                },
            },
            tex.view
        );
    } else {
        dx.device.createShaderResourceView(
            tex.gpu_resource.resource,
            d3d.D3D12_SHADER_RESOURCE_VIEW_DESC{
                .Format = formatQueryInfo(desc.format).format,
                .ViewDimension = d3d.D3D12_SRV_DIMENSION_TEXTURE2D,
                .Shader4ComponentMapping = d3d.D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                .unnamed_0 = .{
                    .Texture2D = .{
                        .MostDetailedMip = 0,
                        .MipLevels = tex.mip_count,
                        .PlaneSlice = 0,
                        .ResourceMinLODClamp = 0.0,
                    },
                },
            },
            tex.view
        );
    }
    //2d/3d array view
    if (desc.is_cube_map or is_3d) {
        tex.array_view = .{ .ptr=0 };
    } else {
        tex.array_view = dx.staging_descriptor_heap.alloc();
        dx.device.createShaderResourceView(
            tex.gpu_resource.resource,
            d3d.D3D12_SHADER_RESOURCE_VIEW_DESC{
                .Format = formatQueryInfo(desc.format).format,
                .ViewDimension = d3d.D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
                .Shader4ComponentMapping = d3d.D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                .unnamed_0 = .{
                    .Texture2DArray = .{
                        .MostDetailedMip = 0,
                        .MipLevels = tex.mip_count,
                        .FirstArraySlice = 0,
                        .ArraySize = tex.array_count,
                        .PlaneSlice = 0,
                        .ResourceMinLODClamp = 0.0,
                    },
                },
            },
            tex.array_view
        );
    }

    if (desc.allow_unordered_access) {
        std.debug.assert(formatQueryInfo(desc.format).non_srgb_format == null);
        std.debug.assert((!desc.is_cube_map) and (!is_3d)); //these aren't handled
        for (0..tex.mip_count) |mip| {
            tex.uav_mip_views[mip] = dx.staging_descriptor_heap.alloc();
            dx.device.createUnorderedAccessView(
                tex.gpu_resource.resource,
                null, //counter resource
                d3d.D3D12_UNORDERED_ACCESS_VIEW_DESC{
                    .Format = formatQueryInfo(desc.format).format,
                    .ViewDimension = d3d.D3D12_UAV_DIMENSION_TEXTURE2D,
                    .unnamed_0 = .{
                        .Texture2D = .{
                            .MipSlice = @intCast(mip),
                            .PlaneSlice = 0,
                        },
                    },
                },
                tex.uav_mip_views[mip]
            );
        }
    }

    tex.debug_name = try allocator.dupeZ(u8, debug_name);
    return hdl;
}

fn destroyTexture(self: *Texture) !void {
    try scheduleStagingDescriptorHandleForDeletion(&self.view);
    if (self.array_view.ptr != 0) {
        try scheduleStagingDescriptorHandleForDeletion(&self.array_view);
    }
    try self.gpu_resource.deinit();
    allocator.free(self.debug_name);
    self.* = undefined;
}

// ************************************************************************************************************************
// render target format groups
// to handle draw state permutations per render target format set; we store a global list of all possible render targets set
// any time a new group is set we create the necessary psos and group tracking objects internally
// this means that pso creation is deferred until the first setPipelineState
// ************************************************************************************************************************

const RenderTargetFormatGroup = struct {
    ds: d3d.DXGI_FORMAT,
    rt: [8]d3d.DXGI_FORMAT,
    rt_count: usize,

    pub const MapContext = struct {
        pub fn hash(self: @This(), k: RenderTargetFormatGroup) u64 {
            _ = self;
            var h = std.hash.Wyhash.init(42);
            h.update(std.mem.asBytes(&k.ds));
            h.update(std.mem.asBytes(&k.rt_count));
            var i: usize = 0;
            while (i < k.rt_count) : (i += 1) {
                h.update(std.mem.asBytes(&k.rt[i]));
            }
            return h.final();
        }

        pub fn eql(self: @This(), a: RenderTargetFormatGroup, b: RenderTargetFormatGroup) bool {
            _ = self;
            if ((a.rt_count != b.rt_count) or (a.ds != b.ds)) {
                return false;
            }
            var i: usize = 0;
            while (i < a.rt_count) : (i += 1) {
                if (a.rt[i] != b.rt[i]) {
                    return false;
                }
            }
            return true;
        }
    };
};

var render_target_format_group_map: std.HashMapUnmanaged(RenderTargetFormatGroup, ResourceHandle, RenderTargetFormatGroup.MapContext, std.hash_map.default_max_load_percentage) = .{};

fn findOrCreateRenderTargetFormatGroup(group: RenderTargetFormatGroup) usize {
    if (render_target_format_group_map.get(group)) |hdl| {
        return @intCast(hdl.idx);
    }

    var hdl = render_target_format_groups.create();
    var grp = render_target_format_groups.lookup(hdl);
    grp.* = group;

    render_target_format_group_map.putNoClobber(allocator, group, hdl) catch unreachable;
    return @intCast(hdl.idx);
}

// ************************************************************************************************************************
// draw state
// ************************************************************************************************************************

pub const PipelineState = struct {
    pipeline_state_desc: union (enum) {
        graphics: d3d.D3D12_GRAPHICS_PIPELINE_STATE_DESC,
        compute: void,
        mesh: D3D.PIPELINE_STATE_MESH_STREAM,
    },
    desc: gdi.PipelineStateDesc,
    pso_list: std.ArrayListUnmanaged(?PlatformPipelineState),
    root_sig: *RootSignature,
    prim_topology: d3d.D3D12_PRIMITIVE_TOPOLOGY,
    debug_name: [:0]const u8,

    pub fn getOrCreatePipelineState(self: *PipelineState, rt_format_idx: ?usize) !PlatformPipelineState {
        return switch (self.pipeline_state_desc) {
            .compute => try self.getOrCreateComputePipelineState(),
            .graphics => try self.getOrCreateGraphicsPipelineState(rt_format_idx.?),
            .mesh => try self.getOrCreateMeshPipelineState(rt_format_idx.?),
        };
    }

    fn getOrCreateComputePipelineState(self: *PipelineState) !PlatformPipelineState {
        if (self.pso_list.items.len < 1) {
            const d3d_desc = d3d.D3D12_COMPUTE_PIPELINE_STATE_DESC{
                .pRootSignature = self.root_sig.rs.root_signature,
                .CS = .{
                    .pShaderBytecode = self.desc.shader_ref.cs.?.code,
                    .BytecodeLength = self.desc.shader_ref.cs.?.len,
                },
                .Flags = d3d.D3D12_PIPELINE_STATE_FLAG_NONE,
                .NodeMask = 0,
                .CachedPSO = .{
                    .pCachedBlob = null,
                    .CachedBlobSizeInBytes = 0,
                },
            };
            var pso = try dx.device.createComputePipelineState(&d3d_desc);
            pso.setName("PipelineState.pso_list.item[0] (compute)");
            try self.pso_list.append(allocator, pso);
        }

        std.debug.assert(self.pso_list.items.len == 1);
        return self.pso_list.items[0].?;
    }

    fn getOrCreateMeshPipelineState(self: *PipelineState, rt_format_idx: usize) !PlatformPipelineState {
        if (self.pso_list.items.len > rt_format_idx) {
            if (self.pso_list.items[rt_format_idx]) |pso| {
                return pso;
            }
        }

        //resize the array up (making sure new elements are null
        while (self.pso_list.items.len <= rt_format_idx) {
            try self.pso_list.append(allocator, null);
        }

        var group = &render_target_format_groups.items[rt_format_idx];
        self.checkPixelShaderMatchesRenderTargetGroup(self.desc.shader_ref.ps.?, group);

        //setup the render target states
        self.pipeline_state_desc.mesh.RTVFormats.NumRenderTargets = @intCast(group.rt_count);
        var i: usize = 0;
        while (i < 8) : (i+=1) {
            self.pipeline_state_desc.mesh.RTVFormats.RTFormats[i] = group.rt[i];
        }
        self.pipeline_state_desc.mesh.DSVFormat = group.ds;

        //now create the pipeline
        const stream_desc = D3D.D3D12_PIPELINE_STATE_STREAM_DESC{
            .SizeInBytes = @sizeOf(D3D.PIPELINE_STATE_MESH_STREAM),
            .pPipelineStateSubobjectStream = &self.pipeline_state_desc.mesh,
        };
        const pso = try dx.device.createPipelineState(&stream_desc);
        pso.setName("PipelineState.pso_list.item[N] (mesh)");
        self.pso_list.items[rt_format_idx] = pso;
        return pso;
    }

    fn getOrCreateGraphicsPipelineState(self: *PipelineState, rt_format_idx: usize) !PlatformPipelineState {
        if (self.pso_list.items.len > rt_format_idx) {
            if (self.pso_list.items[rt_format_idx]) |pso| {
                return pso;
            }
        }

        //resize the array up (making sure new elements are null
        while (self.pso_list.items.len <= rt_format_idx) {
            try self.pso_list.append(allocator, null);
        }

        //setup the render target states
        var group = &render_target_format_groups.items[rt_format_idx];
        self.checkPixelShaderMatchesRenderTargetGroup(self.desc.shader_ref.ps.?, group);
    
        self.pipeline_state_desc.graphics.NumRenderTargets = @intCast(group.rt_count);
        var i: usize = 0;
        while (i < 8) : (i+=1) {
            self.pipeline_state_desc.graphics.RTVFormats[i] = group.rt[i];
        }
        self.pipeline_state_desc.graphics.DSVFormat = group.ds;

        //now create the pipeline
        const pso = try dx.device.createGraphicsPipelineState(&self.pipeline_state_desc.graphics);
        pso.setName("PipelineState.pso_list.item[N] (graphics)");
        self.pso_list.items[rt_format_idx] = pso;
        return pso;
    }

    fn checkPixelShaderMatchesRenderTargetGroup(self: PipelineState, ps: *const gdi.ShaderByteCode, group: *const RenderTargetFormatGroup) void {
        //check if shaders are compatible with the render target formats
        if (ps.output_render_target_count > group.rt_count) {
            log.warn("Output Render Target Mismatch: Shader '{s}' Shader RT Count [{}] != Bound RT Counts [{}]\n", .{ self.desc.shader_ref.shader_name, ps.output_render_target_count, group.rt_count });
        }
    }
};

fn createPipelineState(comptime debug_name: [:0]const u8, desc: gdi.PipelineStateDesc) !ResourceHandle {
    var hdl = pipeline_states.create();
    var s = pipeline_states.lookup(hdl);
    s.desc = desc;
    s.pso_list = .{};
    s.debug_name = try allocator.dupeZ(u8, debug_name);

    const descriptor_bind_set = 
        if (null != desc.shader_ref.cs) desc.shader_ref.cs.?.descriptor_bind_set
        else desc.shader_ref.ps.?.descriptor_bind_set;
    s.root_sig = root_signatures.lookup(try getOrBuildRootSignature(descriptor_bind_set));

    if (null != desc.shader_ref.vs) {
        std.debug.assert(desc.shader_ref.as == null);
        std.debug.assert(desc.shader_ref.ms == null);
        std.debug.assert(desc.shader_ref.cs == null);
        std.debug.assert(desc.shader_ref.ps != null);
        std.debug.assert(ResourceHandle.eql(descriptor_bind_set, desc.shader_ref.vs.?.descriptor_bind_set)); //the descriptor bind set must match
        s.pipeline_state_desc = .{ .graphics = undefined };
        buildGraphicsPipelineState(s, &s.pipeline_state_desc.graphics, render_states.lookup(desc.render_state_id.?).desc);
    } else if (null != desc.shader_ref.ms) {
        std.debug.assert(desc.shader_ref.ps != null);
        std.debug.assert(desc.shader_ref.cs == null);
        std.debug.assert(ResourceHandle.eql(descriptor_bind_set, desc.shader_ref.ms.?.descriptor_bind_set)); //the descriptor bind set must match
        s.pipeline_state_desc = .{ .mesh = undefined };
        buildMeshPipelineState(s, &s.pipeline_state_desc.mesh, render_states.lookup(desc.render_state_id.?).desc);
    } else {
        std.debug.assert(desc.shader_ref.as == null);
        std.debug.assert(desc.shader_ref.ps == null);
        std.debug.assert(desc.shader_ref.cs != null);
        s.pipeline_state_desc = .{ .compute = undefined };
    }

    return hdl;
}

fn buildBlendState(state: gdi.RenderStateDesc) d3d.D3D12_BLEND_DESC {
    var bs = std.mem.zeroes(d3d.D3D12_BLEND_DESC);
    var rt: usize = 0;
    while (rt < 8) : (rt += 1) {
        var write_mask: u32 = 0;
        if (state.blend.disable_write_red or state.blend.disable_write_green or state.blend.disable_write_blue or state.blend.disable_write_alpha) {
            if (state.blend.disable_write_red) {
                write_mask = write_mask | d3d.D3D12_COLOR_WRITE_ENABLE_RED;
            }
            if (state.blend.disable_write_green) {
                write_mask = write_mask | d3d.D3D12_COLOR_WRITE_ENABLE_GREEN;
            }
            if (state.blend.disable_write_blue) {
                write_mask = write_mask | d3d.D3D12_COLOR_WRITE_ENABLE_BLUE;
            }
            if (state.blend.disable_write_alpha) {
                write_mask = write_mask | d3d.D3D12_COLOR_WRITE_ENABLE_ALPHA;
            }
        } else {
            write_mask = d3d.D3D12_COLOR_WRITE_ENABLE_ALL;
        }

        bs.RenderTarget[rt] = .{
            .BlendEnable    = if (state.blend.is_blending_enabled) d3d.TRUE else d3d.FALSE,
            .SrcBlend       = blendToD3D12(state.blend.src_blend),
            .DestBlend      = blendToD3D12(state.blend.dst_blend),
            .BlendOp        = blendOpToD3D12(state.blend.blend_op),
            .SrcBlendAlpha  = blendToD3D12(state.blend.src_blend_alpha),
            .DestBlendAlpha = blendToD3D12(state.blend.dst_blend_alpha),
            .BlendOpAlpha   = blendOpToD3D12(state.blend.blend_op_alpha),
            .RenderTargetWriteMask = @intCast(write_mask),
            .LogicOpEnable = d3d.FALSE,
            .LogicOp = d3d.D3D12_LOGIC_OP_NOOP,
        };
    }
    return bs;
}

fn buildDepthStencilState(state: gdi.RenderStateDesc) d3d.D3D12_DEPTH_STENCIL_DESC {
    return .{
        .DepthEnable = if (state.depth.is_depth_write_enabled or ((state.depth.depth_test != gdi.Comparison.always) and (state.depth.depth_test != gdi.Comparison.never))) d3d.TRUE else d3d.FALSE,
        .DepthWriteMask = (if (state.depth.is_depth_write_enabled) d3d.D3D12_DEPTH_WRITE_MASK_ALL else d3d.D3D12_DEPTH_WRITE_MASK_ZERO),
        .DepthFunc = comparisonToD3D12(state.depth.depth_test),
        .StencilEnable = d3d.FALSE,
        .FrontFace = .{
            .StencilFailOp = d3d.D3D12_STENCIL_OP_KEEP,
            .StencilDepthFailOp = d3d.D3D12_STENCIL_OP_KEEP,
            .StencilPassOp = d3d.D3D12_STENCIL_OP_KEEP,
            .StencilFunc = d3d.D3D12_COMPARISON_FUNC_ALWAYS,
        },
        .BackFace = .{
            .StencilFailOp = d3d.D3D12_STENCIL_OP_KEEP,
            .StencilDepthFailOp = d3d.D3D12_STENCIL_OP_KEEP,
            .StencilPassOp = d3d.D3D12_STENCIL_OP_KEEP,
            .StencilFunc = d3d.D3D12_COMPARISON_FUNC_ALWAYS,
        },
        .StencilReadMask = 0xff,
        .StencilWriteMask = 0xff,
    };
}

fn buildRasterizerState(state: gdi.RenderStateDesc) d3d.D3D12_RASTERIZER_DESC {
    var rs = std.mem.zeroes(d3d.D3D12_RASTERIZER_DESC);
    if (state.raster.is_wireframe_enabled) {
        rs.FillMode = d3d.D3D12_FILL_MODE_WIREFRAME;
        rs.AntialiasedLineEnable = d3d.TRUE;
    } else {
        rs.FillMode = d3d.D3D12_FILL_MODE_SOLID;
    }
    rs.CullMode = switch (state.raster.cull_mode) {
        gdi.CullMode.none  => d3d.D3D12_CULL_MODE_NONE,
        gdi.CullMode.front => d3d.D3D12_CULL_MODE_FRONT,
        gdi.CullMode.back  => d3d.D3D12_CULL_MODE_BACK,
    };
    rs.FrontCounterClockwise = d3d.TRUE;
    rs.DepthClipEnable = if (state.raster.is_depth_clip_enabled) d3d.TRUE else d3d.FALSE;

    return rs;
}

fn buildPrimTopo(state: gdi.RenderStateDesc, out_type: *d3d.D3D12_PRIMITIVE_TOPOLOGY_TYPE, out_topo: *d3d.D3D_PRIMITIVE_TOPOLOGY) void {
    switch (state.prim_type) {
        gdi.PrimitiveType.tri_list => {
            out_type.* = d3d.D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            out_topo.* = d3d.D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        },
        gdi.PrimitiveType.tri_strip => {
            out_type.* = d3d.D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            out_topo.* = d3d.D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;
        },
        gdi.PrimitiveType.line_list => {
            out_type.* = d3d.D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
            out_topo.* = d3d.D3D_PRIMITIVE_TOPOLOGY_LINELIST;
        },
        gdi.PrimitiveType.line_strip => {
            out_type.* = d3d.D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
            out_topo.* = d3d.D3D_PRIMITIVE_TOPOLOGY_LINESTRIP;
        },
        gdi.PrimitiveType.point_list => {
            out_type.* = d3d.D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT;
            out_topo.* = d3d.D3D_PRIMITIVE_TOPOLOGY_POINTLIST;
        },
    }
}

fn buildMeshPipelineState(s: *PipelineState, d: *D3D.PIPELINE_STATE_MESH_STREAM, state: gdi.RenderStateDesc) void {
    d.* = .{
        .pRootSignature = s.root_sig.rs.root_signature,
        .PS = .{
            .pShaderBytecode = s.desc.shader_ref.ps.?.code,
            .BytecodeLength = s.desc.shader_ref.ps.?.len,
        },
        .AS = if (s.desc.shader_ref.as) |as| .{
                .pShaderBytecode = as.code,
                .BytecodeLength = as.len,
            } else .{
                .pShaderBytecode = null,
                .BytecodeLength = 0,
            },
        .MS = .{
            .pShaderBytecode = s.desc.shader_ref.ms.?.code,
            .BytecodeLength = s.desc.shader_ref.ms.?.len,
        },
        .BlendState = buildBlendState(state),
        .DepthStencilState = buildDepthStencilState(state),
        .RasterizerState = buildRasterizerState(state),
        .PrimitiveTopologyType = undefined,
        .SampleDesc = .{
            .Count = 1,
            .Quality = 0,
        },
        .SampleMask = d3d.UINT_MAX,
        //these are filled in only later once the render targets are known (in getOrCreateMeshPipelineState())
        .DSVFormat = undefined,
        .RTVFormats = .{
            .RTFormats = undefined,
            .NumRenderTargets = 0,
        },
    };

    buildPrimTopo(state, &d.PrimitiveTopologyType, &s.prim_topology);
}

fn buildGraphicsPipelineState(s: *PipelineState, d: *d3d.D3D12_GRAPHICS_PIPELINE_STATE_DESC, state: gdi.RenderStateDesc) void {
    d.* = std.mem.zeroes(d3d.D3D12_GRAPHICS_PIPELINE_STATE_DESC);
    d.pRootSignature = s.root_sig.rs.root_signature;
    d.VS = .{
        .pShaderBytecode = s.desc.shader_ref.vs.?.code,
        .BytecodeLength = s.desc.shader_ref.vs.?.len,
    };
    d.PS = .{
        .pShaderBytecode = s.desc.shader_ref.ps.?.code,
        .BytecodeLength = s.desc.shader_ref.ps.?.len,
    };
    d.SampleMask = d3d.UINT_MAX;
    d.SampleDesc = .{ .Count = 1, .Quality = 0, };
    d.BlendState = buildBlendState(state);
    d.RasterizerState = buildRasterizerState(state);
    d.DepthStencilState = buildDepthStencilState(state);

    buildPrimTopo(state, &d.PrimitiveTopologyType, &s.prim_topology);
}

fn destroyPipelineState(self: *PipelineState) !void {
    for (self.pso_list.items) |opt_pso| {
        if (opt_pso) |pso| {
            try scheduleFrameResourceForDeletion(pso);
            unreachable;
        }
    }
    self.pso_list.deinit(allocator);
    allocator.free(self.debug_name);
    self.* = undefined;
}

fn comparisonToD3D12(comp: gdi.Comparison) d3d.D3D12_COMPARISON_FUNC {
    return switch (comp) {
        gdi.Comparison.always        => d3d.D3D12_COMPARISON_FUNC_ALWAYS,
        gdi.Comparison.equal         => d3d.D3D12_COMPARISON_FUNC_EQUAL,
        gdi.Comparison.not_equal     => d3d.D3D12_COMPARISON_FUNC_NOT_EQUAL,
        gdi.Comparison.greater       => d3d.D3D12_COMPARISON_FUNC_GREATER,
        gdi.Comparison.greater_equal => d3d.D3D12_COMPARISON_FUNC_GREATER_EQUAL,
        gdi.Comparison.less          => d3d.D3D12_COMPARISON_FUNC_LESS,
        gdi.Comparison.less_equal    => d3d.D3D12_COMPARISON_FUNC_LESS_EQUAL,
        gdi.Comparison.never         => d3d.D3D12_COMPARISON_FUNC_NEVER,
    };
}

fn blendToD3D12(blend: gdi.Blend) d3d.D3D12_BLEND {
    return switch (blend) {
        gdi.Blend.zero          => d3d.D3D12_BLEND_ZERO,
        gdi.Blend.one           => d3d.D3D12_BLEND_ONE,
        gdi.Blend.src_color     => d3d.D3D12_BLEND_SRC_COLOR,
        gdi.Blend.inv_src_color => d3d.D3D12_BLEND_INV_SRC_COLOR,
        gdi.Blend.src_alpha     => d3d.D3D12_BLEND_SRC_ALPHA,
        gdi.Blend.inv_src_alpha => d3d.D3D12_BLEND_INV_SRC_ALPHA,
        gdi.Blend.dst_alpha     => d3d.D3D12_BLEND_DEST_ALPHA,
        gdi.Blend.inv_dst_alpha => d3d.D3D12_BLEND_INV_DEST_ALPHA,
        gdi.Blend.dst_color     => d3d.D3D12_BLEND_DEST_COLOR,
        gdi.Blend.inv_dst_color => d3d.D3D12_BLEND_INV_DEST_COLOR,
    };
}

fn blendOpToD3D12(op: gdi.BlendOp) d3d.D3D12_BLEND_OP {
    return switch (op) {
        gdi.BlendOp.add              => d3d.D3D12_BLEND_OP_ADD,
        gdi.BlendOp.subtract         => d3d.D3D12_BLEND_OP_SUBTRACT,
        gdi.BlendOp.reverse_subtract => d3d.D3D12_BLEND_OP_REV_SUBTRACT,
        gdi.BlendOp.minimum          => d3d.D3D12_BLEND_OP_MIN,
        gdi.BlendOp.maximum          => d3d.D3D12_BLEND_OP_MAX,
    };
}

// ************************************************************************************************************************
// render state
// ************************************************************************************************************************

pub const RenderState = struct {
    desc: gdi.RenderStateDesc,
    debug_name: [:0]const u8,
};

fn createRenderState(comptime debug_name: [:0]const u8, desc: gdi.RenderStateDesc) !ResourceHandle {
    var hdl = render_states.create();
    var rs = render_states.lookup(hdl);
    rs.desc = desc;
    rs.debug_name = debug_name;
    return hdl;
}

// ************************************************************************************************************************
// descriptor binding info
// ************************************************************************************************************************

const DescriptorKind = enum { cbv, srv, uav, sam };

const DescriptorBindInfo = struct {
    name: [:0]const u8,
    kind: DescriptorKind,
    value_kind: gdi.ResourceKind,
    space: u32,
    start: u32,
    count: u32,
    heap_offset: u32,

    pub fn eql(a: DescriptorBindInfo, b: DescriptorBindInfo) bool {
        if ( (a.kind != b.kind)
            or (a.value_kind != b.value_kind)
            or (a.space != b.space)
            or (a.start != b.start)
            or (a.count != b.count)
        ) {
            return false;
        }
        return std.mem.eql(u8, a.name, b.name);
    }

    pub fn doesOverlap(a: DescriptorBindInfo, b: DescriptorBindInfo) bool {
        if (a.space != b.space) {
            return false;
        }
        const a0 = a.start;
        const a1 = a.start+a.count-1;
        const b0 = b.start;
        const b1 = b.start+b.count-1;
        const overlaps = (a0 <= b1) and (b0 <= a1);
        return overlaps;
    }
};

const DescriptorRange = struct {
    kind: DescriptorKind,
    space: u32,
    start: u32,
    count: u32,
    heap_offset: u32,

    pub fn appendBindingInfo(self: *DescriptorRange, bind_info: DescriptorBindInfo) void {
        std.debug.assert(bind_info.kind == self.kind);
        std.debug.assert(bind_info.space == self.space);
        const end = self.start+self.count;
        const end_info = bind_info.start+bind_info.count;
        const new_end = @max(end, end_info);
        self.start = @min(self.start, bind_info.start);
        self.count = new_end - self.start;
    }

    pub fn lessThan(context: anytype, a: DescriptorRange, b: DescriptorRange) bool {
        _ = context;
        const a_kind = @intFromEnum(a.kind);
        const b_kind = @intFromEnum(b.kind);
        if (a_kind != b_kind) {
            return a_kind < b_kind;
        }
        if (a.space != b.space) {
            return a.space < b.space;
        }
        if (a.count != b.count) {
            return a.count < b.count;
        }
        return a.start < b.start;
    }

    pub fn doesOverlap(a: DescriptorRange, b: DescriptorRange) bool {
        if ( (a.kind != b.kind) or (a.space != b.space) ) {
            return false;
        }
        const a0 = a.start;
        const a1 = a.start+a.count;
        const b0 = b.start;
        const b1 = b.start+b.count;
        const overlaps = (a0 <= b1) and (b0 <= a1);
        return overlaps;
    }
};

const DescriptorBindSet = struct {
    ranges: std.ArrayListUnmanaged(DescriptorRange),
    infos: std.ArrayListUnmanaged(DescriptorBindInfo),
    params: std.ArrayListUnmanaged(*ShaderParameter),
    is_mergable: bool,
    is_dirty: bool,
    descriptor_count: u32,
    desc_range_gpu: PlatformGpuDescriptorHandle = .{ .ptr=0 },
    desc_range_cpu: PlatformCpuDescriptorHandle = .{ .ptr=0 },

    pub fn init() DescriptorBindSet {
        return .{
            .ranges = .{},
            .infos = .{},
            .params = .{},
            .descriptor_count = 0,
            .is_mergable = true,
            .is_dirty = true,
        };
    }

    pub fn deinit(self: *DescriptorBindSet) void {
        self.ranges.deinit(allocator);
        for (self.infos.items) |info| {
            allocator.free(info.name);
        }
        self.infos.deinit(allocator);
    }

    pub fn markDirty(self: *DescriptorBindSet) void {
        self.is_dirty = true;
    }

    fn findRangeForInfo(self: *DescriptorBindSet, info: DescriptorBindInfo) ?*DescriptorRange {
        for (self.ranges.items) |*range| {
            if ( (range.kind == info.kind) and (range.space == info.space) ) {
                return range;
            }
        }
        return null;
    }

    pub fn canMergeBindInfo(self: *const DescriptorBindSet, info: DescriptorBindInfo) bool {
        if (!self.is_mergable) {
            return false;
        }

        for (self.infos.items) |i| {
            if (i.eql(info)) {
                return true;
            } else if (i.doesOverlap(info)) {
                log.warn("canMergeBindInfo failed due to overlap {} with {}\n", .{info, i});
                return false;
            }
        }
        return true;
    }

    pub fn mergeBindInfo(self: *DescriptorBindSet, info: DescriptorBindInfo) bool {
        if (!self.is_mergable) {
            return false;
        }

        //check if the info exists already
        for (self.infos.items) |i| {
            if (i.eql(info)) {
                return true;
            } else if (i.doesOverlap(info)) {
                //overlaps another binding info, so we cant merge
                return false;
            }
        }

        //didn't find it and it's mergable so lets add a new one
        var copy = info;
        copy.name = allocator.dupeZ(u8, info.name) catch unreachable; //make a copy of the name
        self.infos.append(allocator, copy) catch unreachable;

        //find a matching descriptor range
        //or add a new one
        if (self.findRangeForInfo(copy)) |range| {
            range.appendBindingInfo(copy);
        } else {
            self.ranges.append(allocator, DescriptorRange{
                .kind  = copy.kind,
                .space = copy.space,
                .start = copy.start,
                .count = copy.count,
                .heap_offset = 0xffffffff,
            }) catch unreachable;
        }
        self.markDirty();
        return true;
    }

    pub fn merge(self: *DescriptorBindSet, other: DescriptorBindSet) bool {
        if (!self.is_mergable) {
            return false;
        }

        for (other.infos.items) |info| {
            if (!self.canMergeBindInfo(info)) {
                return false;
            }
        }
        for (other.infos.items) |info| {
            const result = self.mergeBindInfo(info);
            std.debug.assert(result);
        }
        return true;
    }

    pub fn generateDirtyParametersAndHeapOffsets(self: *DescriptorBindSet) !void {
        if (self.params.items.len != self.infos.items.len) {
            //if a parameter could disappear from a bind set then we should remove ourself from it's bind_sets map
            //at the time I'm writing this they only add and never remove

            try self.params.resize(allocator, self.infos.items.len);
            for (self.infos.items, 0..) |info, idx| {
                const lifetime = 
                    if ((gdi.dynamic_bind_space_range_begin <= info.space) and (info.space <= gdi.dynamic_bind_space_range_end))
                        gdi.ShaderParameterLifetime.dynamic
                    else
                        gdi.ShaderParameterLifetime.persistent;
                const param = shader_parameters.lookup(try createShaderParameter(
                    "ShaderParameter.Unknown",
                    gdi.ShaderParameterDesc{
                        .value_kind = info.value_kind,
                        .lifetime = lifetime,
                        .array_count = info.count,
                        .is_unordered_access_view = (info.kind == .uav),
                        .binding_name = info.name,
                    }
                ));
                self.params.items[idx] = param;
                try param.bind_sets.put(allocator, self, {});
            }
        }

        //update ranges heap_offsets
        var start_offset: u32 = 0;
        for (self.ranges.items) |*range| {
            range.heap_offset = start_offset;
            start_offset += range.count;
        }
        self.descriptor_count = start_offset;

        //update infos heap_offsets
        for (self.infos.items) |*info| {
            const range = self.findRangeForInfo(info.*).?;
            std.debug.assert(info.start >= range.start);
            info.heap_offset = range.heap_offset + info.start - range.start;
        }
    }

    pub fn updateGpuDescriptorsIfDirty(bind_set: *DescriptorBindSet) void {
        if (!bind_set.is_dirty) {
            return;
        }
        bind_set.is_dirty = false;

        if (bind_set.desc_range_gpu.ptr != 0) {
            //TODO
            //TODO
            //TODO
            //TODO
            //TODO
            //TODO
            log.warn("updateDescriptorSetBindIfDirty() is leaking the old descriptors as free() isnt yet implemented\n", .{});
        }

        var heap = gpu_descriptor_heap;
        var desc_range_idx = heap.allocIdxRange(bind_set.descriptor_count);
        bind_set.desc_range_gpu = heap.idxToGpuHandle(desc_range_idx);
        bind_set.desc_range_cpu = heap.idxToCpuAccessHandle(desc_range_idx);

        //for each parameter, copy it's value into the shader visible heap
        for (bind_set.params.items, 0..) |param, idx| {
            const bind_info = bind_set.infos.items[idx];
            std.debug.assert(param.desc.array_count == bind_info.count);
            std.debug.assert(param.desc.array_count == param.values.len);

            if (param.desc.lifetime == .dynamic) {
                std.debug.assert(false);
                //store a list to look up later per draw or per set?

                //fall though and update these for default values?, but handle them differently for destinations below?

                //instead should i just have a custom root sig for each shader that i want to be more dynamic?
                continue;
            }

            if (param.desc.lifetime == .static_sampler) {
                //TODO: handle static samplers here?
            }

            for (param.values, 0..) |v, v_idx| {
                var src: PlatformCpuDescriptorHandle = .{ .ptr = 0 };

                switch (param.desc.value_kind) {
                    .constant_buffer => {
                        if (param.descriptor_range_kind == .uav) {
                            std.debug.assert(false); //TODO
                        } else {
                            if (v) |value| {
                                src = value.constant_buffer.view;
                                std.debug.assert(src.ptr != 0);
                            } else {
                                src = dx.null_descriptor_cbv;
                            }
                        }
                    },
                    .buffer => {
                        if (param.descriptor_range_kind == .uav) {
                            if (v) |value| {
                                if (param.desc.array_count > 1) {
                                    if (param.descriptor_handles[v_idx].ptr == 0) {
                                        param.descriptor_handles[v_idx] = dx.staging_descriptor_heap.alloc();
                                    }
                                    src = param.descriptor_handles[v_idx];
                                    //note... this assumes we are dividing the buffer into regions based upon the array size
                                    const element_stride: u32 = @intCast((value.buffer.desc.size / 4) / param.values.len);
                                    dx.device.createUnorderedAccessView(
                                        value.buffer.gpu_resource.resource,
                                        null, //counter resource
                                        d3d.D3D12_UNORDERED_ACCESS_VIEW_DESC{
                                            .Format = d3d.DXGI_FORMAT_R32_TYPELESS,
                                            .ViewDimension = d3d.D3D12_UAV_DIMENSION_BUFFER,
                                            .unnamed_0 = .{
                                                .Buffer = .{
                                                    .FirstElement = element_stride*v_idx,
                                                    .NumElements = element_stride,
                                                    .StructureByteStride = 0,
                                                    .CounterOffsetInBytes = 0,
                                                    .Flags = d3d.D3D12_BUFFER_UAV_FLAG_RAW,
                                                },
                                            },
                                        },
                                        src
                                    );
                                } else {
                                    std.debug.assert(value.buffer.desc.allow_unordered_access);
                                    src = value.buffer.uav;
                                    std.debug.assert(src.ptr != 0);
                                }
                            } else {
                                src = dx.null_descriptor_uav;
                            }
                        } else {
                            if (v) |value| {
                                src = value.buffer.view;
                                std.debug.assert(src.ptr != 0);
                            } else {
                                src = dx.null_descriptor_uav;
                            }
                        }
                    },
                    .texture => {
                        if (param.descriptor_range_kind == .uav) {
                            if (param.desc.array_count > 1) {
                                //PNC: is kinda a hack; there must be a better way to do this
                                //PNC: there must be a better way to handle which mip slice here; also support it for the non-array case
                                const tex = v.?.texture;
                                if (v_idx < tex.mip_count) {
                                    src = tex.uav_mip_views[v_idx];
                                } else {
                                    src = dx.null_descriptor_uav;
                                }
                            } else {
                                src = v.?.texture.uav_mip_views[0];
                            }
                        } else {
                            if (v) |value| {
                                src = value.texture.view;
                                std.debug.assert(src.ptr != 0);
                            } else {
                                //null is valid, but in reality we never want it for non-arrays, so do this check here to catch problems faster
                                if (param.desc.array_count == 1) {
                                    log.warn("shader param '{s}' is null!\n", .{param.desc.binding_name});
                                }
                                src = dx.null_descriptor_srv;
                            }
                        }
                    },
                    .render_target => {
                        if (param.descriptor_range_kind == .uav) {
                            src = v.?.render_target.unordered_access_view;
                            std.debug.assert(src.ptr != 0);
                        } else {
                            if (param.desc.is_non_srgb == true) {
                                src = v.?.render_target.shader_view_non_srgb;
                            } else {
                                src = v.?.render_target.shader_view;
                            }
                            std.debug.assert(src.ptr != 0);
                        }
                    },
                    .sampler => {
                        std.debug.assert(false); //TODO: samplers
                    },
                    else => {
                        unreachable; //invalid value kind
                    },
                }
                std.debug.assert(src.ptr != 0);

                const dst = PlatformCpuDescriptorHandle{ .ptr = bind_set.desc_range_cpu.ptr + ((bind_info.heap_offset + v_idx) * heap.heap.descriptor_size) };
                dx.device.copyDescriptorSimple(dst, src, d3d.D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            }
        }
    }
};

fn findOrCreateDescriptorBindSet(bind_set: DescriptorBindSet) !ResourceHandle {
    if (bind_set.is_mergable) {
        //attempt to merge this list with any of our existing ones
        for (descriptor_bind_sets.items, 0..) |*item, idx| {
            if (!descriptor_bind_sets.is_free.isSet(idx)) {
                if (item.merge(bind_set)) {
                    return ResourceHandle{
                        .kind = descriptor_bind_sets.kind,
                        .gen = descriptor_bind_sets.generations[idx],
                        .idx = @intCast(idx),
                    };
                }
            }
        }
    }

    //create a new one
    var hdl = descriptor_bind_sets.create();
    var bs = descriptor_bind_sets.lookup(hdl);
    bs.* = DescriptorBindSet.init();
    bs.is_mergable = bind_set.is_mergable;
    for (bind_set.infos.items) |info| {
        if (!bs.mergeBindInfo(info)) {
            return Error.InvalidDescriptorBindSetWithOverlappingRanges;
        }
    }
    return hdl;
}

// ************************************************************************************************************************
// shader parameters
// ************************************************************************************************************************

pub const ShaderParameter = struct {
    pub const Value = union {
        buffer: *Buffer,
        constant_buffer: *ConstantBuffer,
        texture: *Texture,
        render_target: *RenderTarget,
        sampler: *Sampler,
    };
    desc: gdi.ShaderParameterDesc,
    descriptor_range_kind: DescriptorKind,
    bind_sets: std.AutoArrayHashMapUnmanaged(*DescriptorBindSet, void),
    values: []?Value,
    descriptor_handles: []PlatformCpuDescriptorHandle,
    debug_name: [:0]const u8,
};

fn createShaderParameter(comptime debug_name: [:0]const u8, desc: gdi.ShaderParameterDesc) !ResourceHandle {
    std.debug.assert(desc.array_count >= 1);

    const binding_name = std.mem.sliceTo(desc.binding_name, 0);

    //search for an existing one
    if (shader_parameter_map.get(binding_name)) |hdl| {
        //std.debug.print("   ...already existed (hdl {},{},{})\n", .{ hdl.kind, hdl.gen, hdl.idx });
        //verify it matches
        std.debug.assert(hdl.kind == .shader_parameter);
        const param = shader_parameters.lookup(hdl);
        std.debug.assert( (param.desc.value_kind == desc.value_kind) or ((param.desc.value_kind == .render_target) and (desc.value_kind == .texture)) );
        std.debug.assert(std.mem.eql(u8, param.desc.binding_name, binding_name));
        std.debug.assert(param.desc.lifetime == desc.lifetime);
        std.debug.assert(param.desc.array_count == desc.array_count);
        std.debug.assert(param.desc.is_unordered_access_view == desc.is_unordered_access_view);
        return hdl;
    }
    //std.debug.print("real-create: createShaderParameter({s})\n", .{binding_name});

    //now create a new one
    var hdl = shader_parameters.create();
    var param = shader_parameters.lookup(hdl);
    param.* = .{
        .desc = desc,
        .descriptor_range_kind = undefined,
        .bind_sets = .{},
        .values = try allocator.alloc(?ShaderParameter.Value, desc.array_count),
        .descriptor_handles = try allocator.alloc(PlatformCpuDescriptorHandle, desc.array_count),
        .debug_name = try allocator.dupeZ(u8, debug_name),
    };
    param.desc.binding_name = try allocator.dupeZ(u8, desc.binding_name); //make a copy of the binding_name
    for (param.values) |*value| {
        value.* = null;
    }
    for (param.descriptor_handles) |*h| {
        h.* = .{ .ptr = 0 };
    }

    if (param.desc.is_unordered_access_view) {
        param.descriptor_range_kind = DescriptorKind.uav;
        std.debug.assert( (desc.value_kind == gdi.ResourceKind.render_target) or (desc.value_kind == gdi.ResourceKind.buffer) or (desc.value_kind == gdi.ResourceKind.texture) );
    } else {
        param.descriptor_range_kind = switch (desc.value_kind) {
            else => { return Error.BadShaderParameterDescriptorRangeKind; }, //bad value kind
            .constant_buffer => DescriptorKind.cbv,
            .render_target, .buffer, .texture => DescriptorKind.srv,
            .sampler => DescriptorKind.sam,
        };
    }

    try shader_parameter_map.putNoClobber(allocator, param.desc.binding_name, hdl);
    return hdl;
}

pub fn setShaderParamValue(shader_param: ResourceHandle, resource: ResourceHandle) void {
    std.debug.assert(shader_parameters.lookup(shader_param).values.len == 1);
    setShaderParamArrayValue(shader_param, resource, 0);
}

pub fn setShaderParamArrayValue(shader_param: ResourceHandle, resource: ResourceHandle, idx: usize) void {
    var param = shader_parameters.lookup(shader_param);
    switch (resource.kind) {
        .none => {
            if (param.values[idx] != null) {
                param.values[idx] = null;
                markShaderParamDirty(param);
            }
        },
        .sampler => {
            const v = samplers.lookup(resource);
            if ((param.values[idx] == null) or (param.values[idx].?.sampler != v)) {
                param.values[idx] = ShaderParameter.Value{ .sampler = v };
                markShaderParamDirty(param);
            }
        },
        .buffer => {
            const v = buffers.lookup(resource);
            if ((param.values[idx] == null) or (param.values[idx].?.buffer != v)) {
                param.values[idx] = ShaderParameter.Value{ .buffer = v };
                markShaderParamDirty(param);
            }
        },
        .constant_buffer => {
            const v = constant_buffers.lookup(resource);
            if ((param.values[idx] == null) or (param.values[idx].?.constant_buffer != v)) {
                param.values[idx] = ShaderParameter.Value{ .constant_buffer = v };
                markShaderParamDirty(param);
            }
        },
        .texture => {
            const v = textures.lookup(resource);
            if ((param.values[idx] == null) or (param.values[idx].?.texture != v)) {
                param.values[idx] = ShaderParameter.Value{ .texture = v };
                markShaderParamDirty(param);
            }
        },
        .render_target => {
            const v = render_targets.lookup(resource); 
            if ((param.values[idx] == null) or (param.values[idx].?.render_target != v) or (v.last_recreate_frame >= frame_num)) {
                param.values[idx] = ShaderParameter.Value{ .render_target = v };
                markShaderParamDirty(param);
            }
        },
        else => unreachable,
    }
}

fn markShaderParamDirty(param: *ShaderParameter) void {
    //mark bind sets as dirty
    for (param.bind_sets.keys()) |bind_set| {
        bind_set.markDirty();
    }
}

// ************************************************************************************************************************
// root signatures
// ************************************************************************************************************************

const RootSignature = struct {
    rs: D3D.RootSignature,
    descriptor_bind_set_hdl: ResourceHandle,
    descriptor_bind_set: *DescriptorBindSet,
    descriptor_root_idx: i32,
};

fn buildEmptyRootSignature() !ResourceHandle {
    const desc = d3d.D3D12_ROOT_SIGNATURE_DESC{
        .NumParameters = 0,
        .pParameters = null,
        .NumStaticSamplers = 0,
        .pStaticSamplers = null,
        .Flags = d3d.D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS | d3d.D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS | d3d.D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS,

    };
    var signature = try dx.serializeRootSignature(desc);
    defer D3D.releaseComPtr(signature);
    var root_sig = try dx.device.createRootSignature(signature);

    var bind_set = DescriptorBindSet.init();
    bind_set.is_mergable = false;

    var hdl = root_signatures.create();
    var sig: *RootSignature = root_signatures.lookup(hdl);
    sig.rs = root_sig;
    sig.descriptor_bind_set_hdl = try findOrCreateDescriptorBindSet(bind_set);
    sig.descriptor_bind_set = descriptor_bind_sets.lookup(sig.descriptor_bind_set_hdl);
    sig.descriptor_root_idx = -1;
    return hdl;
}

//maybe remove these and get it from the shader?
const ShaderSpace = struct {
    const draw_constants = 440;
    const static_sampler = 880;
};

fn getOrBuildRootSignature(descriptor_bind_set: ResourceHandle) !ResourceHandle {
    //return it if it already exists; note that descriptor bind sets and root signatures match 1-to-1
    if (root_signatures.items.len >= descriptor_bind_set.idx) {
        if (!root_signatures.is_free.isSet(descriptor_bind_set.idx)) {
            //an idea that we allow changing currently in use root signatures... for now we don't do this and mark them not mergable as soon as they are used once
            //const set = descriptor_bind_sets.lookup(descriptor_bind_set);
            //if (set.has_changed_layout) {
                //TODO: memory leak, destroy the root signature internal data first...
                //doesnt work as i need to invalidate the draw/dispatch states
                //try updateRootSignature(&root_signatures.items[descriptor_bind_set.idx], descriptor_bind_set, false);
            //}
            return ResourceHandle{
                .kind = ResourceKind.root_signature,
                .gen = root_signatures.generations[descriptor_bind_set.idx],
                .idx = descriptor_bind_set.idx,
            };
        }
    }

    var hdl = root_signatures.create();
    std.debug.assert(hdl.idx == descriptor_bind_set.idx);
    var sig: *RootSignature = root_signatures.lookup(hdl);
    log.info("building a new root signature! [count: {}]\n", .{ hdl.idx+1 });
    if (hdl.idx > 1) {
        log.warn("there are known bugs with this and resources copying wrong, TODO FIX ME\n", .{});
        log.warn("there are known bugs with this and resources copying wrong, TODO FIX ME\n", .{});
        log.warn("there are known bugs with this and resources copying wrong, TODO FIX ME\n", .{});
    }
    try updateRootSignature(sig, descriptor_bind_set, true);
    return hdl;
}

fn updateRootSignature(sig: *RootSignature, descriptor_bind_set: ResourceHandle, is_first_time: bool) !void {
    if (!is_first_time) {
        //TODO: free all the info that changes
        //TODO: free all the info that changes
        //TODO: free all the info that changes
        //TODO: free all the info that changes
    }

    var root_params = try std.ArrayList(d3d.D3D12_ROOT_PARAMETER).initCapacity(allocator, 2);
    defer root_params.deinit();
    var ranges = try std.ArrayList(d3d.D3D12_DESCRIPTOR_RANGE).initCapacity(allocator, 100); //we store pointers to these, so we can't reallocate it
    defer ranges.deinit();

    sig.descriptor_bind_set_hdl = descriptor_bind_set;
    sig.descriptor_bind_set = descriptor_bind_sets.lookup(descriptor_bind_set);
    try sig.descriptor_bind_set.generateDirtyParametersAndHeapOffsets();

    //i would like to invalidate this, but i cant do that without also invalidating all the draw/dispatch states associated with this root signature
    sig.descriptor_bind_set.is_mergable = false;
    sig.descriptor_bind_set.markDirty();

    //add draw constants param
    try root_params.append(.{
        .ParameterType = d3d.D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
        .ShaderVisibility = d3d.D3D12_SHADER_VISIBILITY_ALL,
        .unnamed_0 = .{
            .Constants = .{
                .ShaderRegister = 0,
                .RegisterSpace = ShaderSpace.draw_constants,
                .Num32BitValues = gdi.DrawShaderConstants.count,
            },
        },
    });

    for (sig.descriptor_bind_set.ranges.items) |range| {
        var r = ranges.addOneAssumeCapacity(); //we store pointers to these, so we can't reallocate it
        r.* = .{
            .RangeType = switch (range.kind) {
                .cbv => d3d.D3D12_DESCRIPTOR_RANGE_TYPE_CBV,
                .srv => d3d.D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                .uav => d3d.D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
                .sam => d3d.D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
            },
            .NumDescriptors = range.count,
            .BaseShaderRegister = range.start,
            .RegisterSpace = range.space,
            .OffsetInDescriptorsFromTableStart = range.heap_offset,
        };
    }

    if (sig.descriptor_bind_set.ranges.items.len > 0) {
        sig.descriptor_root_idx = @intCast(root_params.items.len);
        var p = try root_params.addOne();
        p.* = .{
            .ParameterType = d3d.D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            .ShaderVisibility = d3d.D3D12_SHADER_VISIBILITY_ALL,
            .unnamed_0 = .{
                .DescriptorTable = .{
                    .NumDescriptorRanges = @intCast(ranges.items.len),
                    .pDescriptorRanges = &ranges.items[0],
                },
            },
        };
    } else {
        sig.descriptor_root_idx = -1;
    }

    const static_samplers = [_]d3d.D3D12_STATIC_SAMPLER_DESC{
        .{
            .Filter = d3d.D3D12_FILTER_MIN_MAG_MIP_LINEAR,
            .AddressU = d3d.D3D12_TEXTURE_ADDRESS_MODE_WRAP,
            .AddressV = d3d.D3D12_TEXTURE_ADDRESS_MODE_WRAP,
            .AddressW = d3d.D3D12_TEXTURE_ADDRESS_MODE_WRAP,
            .MipLODBias = 0.0,
            .MaxAnisotropy = 0,
            .ComparisonFunc = d3d.D3D12_COMPARISON_FUNC_ALWAYS,
            .BorderColor = d3d.D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK,
            .MinLOD = 0.0,
            .MaxLOD = d3d.D3D12_FLOAT32_MAX,
            .ShaderRegister = 0,
            .RegisterSpace = ShaderSpace.static_sampler,
            .ShaderVisibility = d3d.D3D12_SHADER_VISIBILITY_ALL,
        }, .{
            .Filter = d3d.D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT,
            .AddressU = d3d.D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            .AddressV = d3d.D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            .AddressW = d3d.D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            .MipLODBias = 0.0,
            .MaxAnisotropy = 0,
            .ComparisonFunc = d3d.D3D12_COMPARISON_FUNC_ALWAYS,
            .BorderColor = d3d.D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK,
            .MinLOD = 0.0,
            .MaxLOD = d3d.D3D12_FLOAT32_MAX,
            .ShaderRegister = 1,
            .RegisterSpace = ShaderSpace.static_sampler,
            .ShaderVisibility = d3d.D3D12_SHADER_VISIBILITY_ALL,
        }, .{
            .Filter = d3d.D3D12_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT,
            .AddressU = d3d.D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            .AddressV = d3d.D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            .AddressW = d3d.D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            .MipLODBias = 0.0,
            .MaxAnisotropy = 0,
            .ComparisonFunc = d3d.D3D12_COMPARISON_FUNC_LESS_EQUAL,
            .BorderColor = d3d.D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE,
            .MinLOD = 0.0,
            .MaxLOD = 0.0,
            .ShaderRegister = 2,
            .RegisterSpace = ShaderSpace.static_sampler,
            .ShaderVisibility = d3d.D3D12_SHADER_VISIBILITY_PIXEL,
        }, .{
            .Filter = d3d.D3D12_FILTER_MIN_MAG_MIP_POINT,
            .AddressU = d3d.D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            .AddressV = d3d.D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            .AddressW = d3d.D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            .MipLODBias = 0.0,
            .MaxAnisotropy = 0,
            .ComparisonFunc = d3d.D3D12_COMPARISON_FUNC_ALWAYS,
            .BorderColor = d3d.D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK,
            .MinLOD = 0.0,
            .MaxLOD = d3d.D3D12_FLOAT32_MAX,
            .ShaderRegister = 3,
            .RegisterSpace = ShaderSpace.static_sampler,
            .ShaderVisibility = d3d.D3D12_SHADER_VISIBILITY_ALL,
        },
    };

    const desc = d3d.D3D12_ROOT_SIGNATURE_DESC{
        .NumParameters = @intCast(root_params.items.len),
        .pParameters = root_params.items.ptr,
        .NumStaticSamplers = @intCast(static_samplers.len),
        .pStaticSamplers = &static_samplers[0],
        .Flags = d3d.D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS | d3d.D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS | d3d.D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS,
    };

    var signature = try dx.serializeRootSignature(desc);
    defer D3D.releaseComPtr(signature);
    sig.rs = try dx.device.createRootSignature(signature);
}

// ************************************************************************************************************************
// render targets
// ************************************************************************************************************************

pub const RenderTarget = struct {
    shader_view: PlatformCpuDescriptorHandle,
    shader_view_non_srgb: PlatformCpuDescriptorHandle,
    views: []PlatformCpuDescriptorHandle,
    read_only_views: []PlatformCpuDescriptorHandle,
    unordered_access_view: PlatformCpuDescriptorHandle,
    desc: gdi.RenderTargetDesc,
    format: d3d.DXGI_FORMAT,
    gpu_resource: GpuResource,
    read_back_gpu_resource: GpuResource,
    last_recreate_frame: u64,
    debug_name: [:0]const u8,
};

fn createRenderTarget(comptime debug_name: [:0]const u8, desc: gdi.RenderTargetDesc) !ResourceHandle {
    std.debug.assert(desc.format != gdi.Format.swap_chain);
    std.debug.assert(!formatQueryInfo(desc.format).is_block_compressed);

    var hdl = render_targets.create();
    var rt: *RenderTarget = render_targets.lookup(hdl);
    rt.shader_view = .{ .ptr = 0 };
    rt.shader_view_non_srgb = .{ .ptr = 0 };
    rt.views = try allocator.alloc(PlatformCpuDescriptorHandle, desc.array_count);
    rt.read_only_views = try allocator.alloc(PlatformCpuDescriptorHandle, desc.array_count);
    for (0..desc.array_count) |i| {
        rt.views[i] = .{ .ptr = 0 };
        rt.read_only_views[i] = .{ .ptr = 0 };
    }
    rt.unordered_access_view = .{ .ptr = 0 };
    rt.desc = desc;
    rt.format = formatQueryInfo(rt.desc.format).format;
    rt.debug_name = try allocator.dupeZ(u8, debug_name);
    try recreateRenderTargetResourceAndViews(rt, true);

    return hdl;
}

fn destroyRenderTarget(self: *RenderTarget) !void {
    for (0..self.desc.array_count) |i| {
        if (formatQueryInfo(self.desc.format).is_depth_format) {
            try currentFrame().ds_descriptor_heap_deletion_queue.append(allocator, self.views[i]);
            try currentFrame().ds_descriptor_heap_deletion_queue.append(allocator, self.read_only_views[i]);
        } else {
            try currentFrame().rt_descriptor_heap_deletion_queue.append(allocator, self.views[i]);
            try currentFrame().rt_descriptor_heap_deletion_queue.append(allocator, self.read_only_views[i]);
        }
    }
    if (self.shader_view.ptr != self.shader_view_non_srgb.ptr) {
        try scheduleStagingDescriptorHandleForDeletion(&self.shader_view_non_srgb);
    }
    try scheduleStagingDescriptorHandleForDeletion(&self.shader_view);
    try scheduleStagingDescriptorHandleForDeletion(&self.unordered_access_view);
    try self.gpu_resource.deinit();
    try self.read_back_gpu_resource.deinit();
    allocator.free(self.debug_name);
    self.* = undefined;
}

pub fn resizeRenderTargetIfNecessary(hdl: ResourceHandle, size_x: u32, size_y: u32, mode: gdi.ResizeMode) !bool {
    if ( (size_x < 1) or (size_y < 1) ) {
        return false;
    }

    var rt = render_targets.lookup(hdl);
    switch (mode) {
        .exact => if ( (rt.desc.size_x == size_x) and (rt.desc.size_y == size_y) ) {
            //already the correct size
            return false;
        },
        .equal_or_greater_than => if ( (rt.desc.size_x >= size_x) and (rt.desc.size_y >= size_y) ) {
            //already the correct size
            return false;
        },
    }

    log.debug("resize 'Unknown RenderTarget' from {},{} to {},{}\n", .{rt.desc.size_x, rt.desc.size_y, size_x, size_y});

    try dx.gpuFlush();

    rt.desc.size_x = size_x;
    rt.desc.size_y = size_y;
    try recreateRenderTargetResourceAndViews(rt, false);
    return true;
}

fn recreateRenderTargetResourceAndViews(rt: *RenderTarget, is_first_time: bool) !void {
    rt.last_recreate_frame = frame_num;
    const format_info = formatQueryInfo(rt.desc.format);

    //the size may have changed; so always update it
    rt.gpu_resource.size_in_bytes = rt.desc.size_x * rt.desc.size_y * format_info.size_in_bytes;

    if (is_first_time) {
        rt.gpu_resource.pinned_memory = null;
    } else {
        try scheduleFrameResourceForDeletion(rt.gpu_resource.resource);
    }

    //create the resource
    {
        std.debug.assert(rt.desc.array_count > 0);
        var d3d_desc = d3d.D3D12_RESOURCE_DESC{
            .Dimension = d3d.D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            .Alignment = d3d.D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT,
            .Width = rt.desc.size_x,
            .Height = rt.desc.size_y,
            .DepthOrArraySize = @intCast(rt.desc.array_count),
            .MipLevels = 1,
            .Format = format_info.format,
            .SampleDesc = .{ .Count = 1, .Quality = 0, },
            .Layout = d3d.D3D12_TEXTURE_LAYOUT_UNKNOWN,
            .Flags = d3d.D3D12_RESOURCE_FLAG_NONE,
        };
        if (format_info.is_depth_format) {
            d3d_desc.Alignment = 0;
            d3d_desc.Flags |= d3d.D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
            rt.gpu_resource.states = d3d.D3D12_RESOURCE_STATE_DEPTH_WRITE;
        } else {
            d3d_desc.Flags |= d3d.D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
            rt.gpu_resource.states = d3d.D3D12_RESOURCE_STATE_RENDER_TARGET;
        }
        if (rt.desc.allow_unordered_access) {
            std.debug.assert(d3d_desc.SampleDesc.Count == 1); //UAV doesn't support MSAA
            d3d_desc.Flags |= d3d.D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        }

        const clear_value = d3d.D3D12_CLEAR_VALUE{
            .Format = format_info.format,
            .unnamed_0 =
                if (format_info.is_depth_format)
                    .{ .DepthStencil = .{ .Depth = rt.desc.clear_value.depth_stencil.depth, .Stencil = rt.desc.clear_value.depth_stencil.stencil, } }
                else
                    .{ .Color = rt.desc.clear_value.color },
        };
        const heap_info = d3d.D3D12_HEAP_PROPERTIES{
            .Type = d3d.D3D12_HEAP_TYPE_DEFAULT,
            .CPUPageProperty = d3d.D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            .MemoryPoolPreference = d3d.D3D12_MEMORY_POOL_UNKNOWN,
            .CreationNodeMask = 1,
            .VisibleNodeMask = 1,
        };

        rt.gpu_resource.resource = try dx.device.createCommittedResourceWithClearValue(&heap_info, &d3d_desc, &clear_value, rt.gpu_resource.states);
        //rt.gpu_resource.resource.setName(rt.debug_name);
        //TODO: setName
    }

    //create the views
    if (!is_first_time) {
        if (rt.shader_view.ptr != rt.shader_view_non_srgb.ptr) {
            try scheduleStagingDescriptorHandleForDeletion(&rt.shader_view_non_srgb);
        }
        try scheduleStagingDescriptorHandleForDeletion(&rt.shader_view);
    }
    rt.shader_view = dx.staging_descriptor_heap.alloc();
    rt.shader_view_non_srgb = rt.shader_view;
    if (format_info.is_depth_format) {
        if (!is_first_time) {
            for (0..rt.desc.array_count) |i| {
                dx.depth_stencil_descriptor_heap.free(rt.views[i]);
                dx.depth_stencil_descriptor_heap.free(rt.read_only_views[i]);
            }
        }
        for (0..rt.desc.array_count) |i| {
            rt.views[i] = dx.depth_stencil_descriptor_heap.alloc();
            rt.read_only_views[i] = dx.depth_stencil_descriptor_heap.alloc();
        }

        if (rt.desc.array_count > 1) {
            for (0..rt.desc.array_count) |i| {
                var ds_desc = d3d.D3D12_DEPTH_STENCIL_VIEW_DESC{
                    .Format = format_info.format,
                    .ViewDimension = d3d.D3D12_DSV_DIMENSION_TEXTURE2DARRAY,
                    .unnamed_0 = .{
                        .Texture2DArray = .{ .MipSlice = 0, .FirstArraySlice=@intCast(i), .ArraySize=1, },
                    },
                    .Flags = d3d.D3D12_DSV_FLAG_NONE,
                };
                dx.device.createDepthStencilView(rt.gpu_resource.resource, ds_desc, rt.views[i]);
                ds_desc.Flags = d3d.D3D12_DSV_FLAG_READ_ONLY_DEPTH;
                dx.device.createDepthStencilView(rt.gpu_resource.resource, ds_desc, rt.read_only_views[i]);
            }
        } else {
            var ds_desc = d3d.D3D12_DEPTH_STENCIL_VIEW_DESC{
                .Format = format_info.format,
                .ViewDimension = d3d.D3D12_DSV_DIMENSION_TEXTURE2D,
                .unnamed_0 = .{
                    .Texture2D = .{ .MipSlice = 0, },
                },
                .Flags = d3d.D3D12_DSV_FLAG_NONE,
            };
            dx.device.createDepthStencilView(rt.gpu_resource.resource, ds_desc, rt.views[0]);
            ds_desc.Flags = d3d.D3D12_DSV_FLAG_READ_ONLY_DEPTH;
            dx.device.createDepthStencilView(rt.gpu_resource.resource, ds_desc, rt.read_only_views[0]);
        }

        if ( (format_info.format == d3d.DXGI_FORMAT_D32_FLOAT) or (format_info.format == d3d.DXGI_FORMAT_D32_FLOAT_S8X24_UINT) ) {
            if (rt.desc.array_count > 1) {
                const sv_desc = d3d.D3D12_SHADER_RESOURCE_VIEW_DESC{
                    .ViewDimension = d3d.D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
                    .Format = d3d.DXGI_FORMAT_R32_FLOAT,
                    .Shader4ComponentMapping = d3d.D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    .unnamed_0 = .{
                        .Texture2DArray = .{
                            .MostDetailedMip = 0,
                            .MipLevels = 1,
                            .FirstArraySlice = 0,
                            .ArraySize = rt.desc.array_count,
                            .PlaneSlice = 0,
                            .ResourceMinLODClamp = 0.0,
                        },
                    },
                };
                dx.device.createShaderResourceView(rt.gpu_resource.resource, sv_desc, rt.shader_view);
            } else {
                const sv_desc = d3d.D3D12_SHADER_RESOURCE_VIEW_DESC{
                    .ViewDimension = d3d.D3D12_SRV_DIMENSION_TEXTURE2D,
                    .Format = d3d.DXGI_FORMAT_R32_FLOAT,
                    .Shader4ComponentMapping = d3d.D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    .unnamed_0 = .{
                        .Texture2D = .{
                            .MostDetailedMip = 0,
                            .MipLevels = 1,
                            .PlaneSlice = 0,
                            .ResourceMinLODClamp = 0.0,
                        },
                    },
                };
                dx.device.createShaderResourceView(rt.gpu_resource.resource, sv_desc, rt.shader_view);
            }
        } else {
            dx.device.copyDescriptorSimple(rt.shader_view, dx.null_descriptor_srv, d3d.D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        }
    } else {
        for (0..rt.desc.array_count) |i| {
            if (!is_first_time) {
                dx.render_target_descriptor_heap.free(rt.views[i]);
            }
            rt.views[i] = dx.render_target_descriptor_heap.alloc();
            const rtv_desc = d3d.D3D12_RENDER_TARGET_VIEW_DESC{
                .Format = format_info.format,
                .ViewDimension = d3d.D3D12_RTV_DIMENSION_TEXTURE2D,
                .unnamed_0 = .{
                    .Texture2D = .{
                        .MipSlice = 0,
                        .PlaneSlice = @intCast(i),
                    },
                },
            };
            dx.device.createRenderTargetView(rt.gpu_resource.resource, &rtv_desc, rt.views[i]);

            //read only view has no meaning with non-depth targets
            rt.read_only_views[i].ptr = 0;
        }

        var sv_desc: d3d.D3D12_SHADER_RESOURCE_VIEW_DESC = undefined;
        if (rt.desc.array_count > 1) {
            sv_desc = d3d.D3D12_SHADER_RESOURCE_VIEW_DESC{
                .ViewDimension = d3d.D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
                .Format = format_info.format,
                .Shader4ComponentMapping = d3d.D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                .unnamed_0 = .{
                    .Texture2DArray = .{
                        .MostDetailedMip = 0,
                        .MipLevels = 1,
                        .FirstArraySlice = 0,
                        .ArraySize = rt.desc.array_count,
                        .PlaneSlice = 0,
                        .ResourceMinLODClamp = 0.0,
                    },
                },
            };
        } else {
            sv_desc = d3d.D3D12_SHADER_RESOURCE_VIEW_DESC{
                .ViewDimension = d3d.D3D12_SRV_DIMENSION_TEXTURE2D,
                .Format = format_info.format,
                .Shader4ComponentMapping = d3d.D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                .unnamed_0 = .{
                    .Texture2D = .{
                        .MostDetailedMip = 0,
                        .MipLevels = 1,
                        .PlaneSlice = 0,
                        .ResourceMinLODClamp = 0.0,
                    },
                },
            };
        }
        dx.device.createShaderResourceView(rt.gpu_resource.resource, sv_desc, rt.shader_view);
        if (format_info.non_srgb_format) |non_srgb_format| {
            rt.shader_view_non_srgb = dx.staging_descriptor_heap.alloc();
            sv_desc.Format = non_srgb_format;
            dx.device.createShaderResourceView(rt.gpu_resource.resource, sv_desc, rt.shader_view_non_srgb);
        }
    }

    if (rt.desc.allow_unordered_access) {
        std.debug.assert(format_info.non_srgb_format == null);
        if (!is_first_time) {
            try scheduleStagingDescriptorHandleForDeletion(&rt.unordered_access_view);
        }
        rt.unordered_access_view = dx.staging_descriptor_heap.alloc();
        if (rt.desc.array_count > 1) {
            dx.device.createUnorderedAccessView(
                rt.gpu_resource.resource,
                null, //counter resource
                d3d.D3D12_UNORDERED_ACCESS_VIEW_DESC{
                    .Format = format_info.format,
                    .ViewDimension = d3d.D3D12_UAV_DIMENSION_TEXTURE2DARRAY,
                    .unnamed_0 = .{
                        .Texture2DArray = .{
                            .MipSlice = 0,
                            .FirstArraySlice = 0,
                            .ArraySize = rt.desc.array_count,
                            .PlaneSlice = 0,
                        },
                    },
                },
                rt.unordered_access_view
            );
        } else {
            dx.device.createUnorderedAccessView(
                rt.gpu_resource.resource,
                null, //counter resource
                d3d.D3D12_UNORDERED_ACCESS_VIEW_DESC{
                    .Format = format_info.format,
                    .ViewDimension = d3d.D3D12_UAV_DIMENSION_TEXTURE2D,
                    .unnamed_0 = .{
                        .Texture2D = .{
                            .MipSlice = 0,
                            .PlaneSlice = 0,
                        },
                    },
                },
                rt.unordered_access_view
            );
        }
    }
}

// ************************************************************************************************************************
// gpu copies
// ************************************************************************************************************************

const GpuDeletionItem = struct {
    frame_num: u64,
    resource: PlatformResource,
};
const GpuDeletionList = std.ArrayListUnmanaged(GpuDeletionItem);

const GpuCopyList = struct {
    const Self = @This();
    pub const BufferCopy = struct {
        src: PlatformResource,
        src_off: u64,
        dst: *GpuResource,
        dst_off: u64,
        size: u64,
        delete_src_after_copy: bool,
    };
    pub const TextureCopy = struct {
        src: PlatformResource,
        src_off: u64,
        dst: *GpuResource,
        subresource_idx: usize,
        mip_desc: gdi.TextureMipDesc,
        delete_src_after_copy: bool,
    };

    pending_buffers: std.ArrayListUnmanaged(BufferCopy) = .{},
    pending_textures: std.ArrayListUnmanaged(TextureCopy) = .{},
    mutex: std.Thread.Mutex = .{},

    pub fn deinit(self: *Self) void {
        self.pending_buffers.deinit(allocator);
        self.pending_textures.deinit(allocator);
    }

    pub fn queueBufferCopy(self: *Self, copy: BufferCopy) !void {
        self.mutex.lock(); defer self.mutex.unlock();
        return self.pending_buffers.append(allocator, copy);
    }

    pub fn queueTextureCopy(self: *Self, copy: TextureCopy) !void {
        self.mutex.lock(); defer self.mutex.unlock();
        return self.pending_textures.append(allocator, copy);
    }

    pub fn hasPendingCopies(self: *Self) bool {
        return (self.pending_buffers.items.len > 0) or (self.pending_textures.items.len > 0);
    }

    pub fn submit(self: *Self, cl: PlatformCommandList, deletion_list: *GpuDeletionList, deletion_frame_num: u64, transition_resources_back_immediately: bool) void {
        self.mutex.lock(); defer self.mutex.unlock();
        std.debug.assert(self.hasPendingCopies());

        //transition all resources to copy dest state
        var batcher = PlatformResourceBarrierBatcher(32){};
        for (self.pending_buffers.items) |copy| {
            if (batcher.transitionIfNecessary(cl, copy.dst.resource, copy.dst.states, d3d.D3D12_RESOURCE_STATE_COPY_DEST)) {
                copy.dst.states = d3d.D3D12_RESOURCE_STATE_COPY_DEST;
            }
        }
        for (self.pending_textures.items) |copy| {
            if (batcher.transitionIfNecessary(cl, copy.dst.resource, copy.dst.states, d3d.D3D12_RESOURCE_STATE_COPY_DEST)) {
                copy.dst.states = d3d.D3D12_RESOURCE_STATE_COPY_DEST;
            }
        }
        batcher.flush(cl);

        //do buffer copies
        for (self.pending_buffers.items) |copy| {
            //std.debug.print("copyBufferRegion(dst={}, dst_off={}, src_off={}, size={})\n", .{@intFromPtr(copy.dst.resource.resource), copy.dst_off, copy.src_off, copy.size});
            std.debug.assert(0 == (copy.dst_off % 4));
            std.debug.assert(0 == (copy.src_off % 4));
            std.debug.assert(0 == (copy.size % 4));
            cl.copyBufferRegion(copy.dst.resource, copy.dst_off, copy.src, copy.src_off, copy.size);
            if (copy.delete_src_after_copy) {
                copy.src.unmapEntireBuffer();
                deletion_list.append(allocator, .{ .resource = copy.src, .frame_num = deletion_frame_num }) catch unreachable;
            }
        }

        //do texture copies
        for (self.pending_textures.items) |copy| {
            var dst = d3d.D3D12_TEXTURE_COPY_LOCATION{
                .pResource = copy.dst.resource.resource,
                .Type = d3d.D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                .unnamed_0 = .{
                    .SubresourceIndex = @intCast(copy.subresource_idx),
                },
            };
            var src = d3d.D3D12_TEXTURE_COPY_LOCATION{
                .pResource = copy.src.resource,
                .Type = d3d.D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                .unnamed_0 = .{
                    .PlacedFootprint = .{
                        .Offset = copy.src_off,
                        .Footprint = .{
                            .Format = copy.mip_desc.backend_format,
                            .Width = copy.mip_desc.size_x,
                            .Height = copy.mip_desc.size_y,
                            .Depth = copy.mip_desc.size_z,
                            .RowPitch = copy.mip_desc.stride_x,
                        },
                    },
                },
            };
            cl.copyTextureSubresource(dst, src);
            if (copy.delete_src_after_copy) {
                copy.src.unmapEntireBuffer();
                deletion_list.append(allocator, .{ .resource = copy.src, .frame_num = deletion_frame_num }) catch unreachable;
            }
        }

        //transition all resources back to a usable state
        if (transition_resources_back_immediately) {
            for (self.pending_buffers.items) |copy| {
                const new_state = (d3d.D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER | d3d.D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | d3d.D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                if (batcher.transitionIfNecessary(cl, copy.dst.resource, copy.dst.states, new_state)) {
                    copy.dst.states = new_state;
                }
            }
            for (self.pending_textures.items) |copy| {
                if (batcher.transitionIfNecessary(cl, copy.dst.resource, copy.dst.states, d3d.D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)) {
                    copy.dst.states = d3d.D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
                }
            }
        }
        batcher.flush(cl);

        self.pending_buffers.resize(allocator, 0) catch unreachable;
        self.pending_textures.resize(allocator, 0) catch unreachable;
    }
};

// ************************************************************************************************************************
// gpu upload heaps
// ************************************************************************************************************************

pub const GpuCopyCompleteFunctionPtr = *const fn(user_data: ?*anyopaque) void;

pub const GpuUploadRegion = struct {
    const Self = @This();
    pub const CopyInfo = union (enum) {
        buffer: GpuCopyList.BufferCopy,
        texture: GpuCopyList.TextureCopy,
    };

    mem: []u8,
    owning_heap: *GpuUploadHeap,
    copy_info: CopyInfo,

    pub fn queue(self: *Self, on_copy_complete: ?GpuCopyCompleteFunctionPtr, on_copy_complete_user_data: ?*anyopaque) !void {
        std.debug.assert( (&currentFrame().direct_upload_heap == self.owning_heap) or (&currentFrame().async_upload_heap == self.owning_heap) ); //check we are doing the queue on the same frame we allocated
        switch (self.copy_info) {
            .buffer  => {
                try self.owning_heap.copy_list.queueBufferCopy(self.copy_info.buffer);
            },
            .texture => {
                try self.owning_heap.copy_list.queueTextureCopy(self.copy_info.texture);
            },
        }
        if (on_copy_complete) |func| {
            self.owning_heap.mutex.lock();
            defer self.owning_heap.mutex.unlock();
            try self.owning_heap.completion_callback_list.append(allocator, .{ .func = func, .user_data = on_copy_complete_user_data });
        }
    }
};

pub const GpuCopyQueueKind = enum {
    direct_queue,
    async_queue,
};

pub const GpuUploadHeap = struct {
    const Self = @This();
    const GpuCopyCompleteCallback = struct {
        func: GpuCopyCompleteFunctionPtr,
        user_data: ?*anyopaque,
    };
    const small_upload_allocation_size = 8000;
    const small_upload_buffer_size = 8*1024*1024;

    copy_list: GpuCopyList,
    completion_callback_list: std.ArrayListUnmanaged(GpuCopyCompleteCallback),
    small_upload_buffer: PlatformResource,
    small_upload_offset: u64,
    small_upload_mem: []u8,
    copy_command_list: PlatformCommandList,
    command_allocator: PlatformCommandAllocator,
    deletion_list: GpuDeletionList,
    cpu_frame_num: u64,
    copy_fence: D3D.Fence,
    mutex: std.Thread.Mutex,
    queue_kind: GpuCopyQueueKind,

    pub fn init(queue_kind: GpuCopyQueueKind) !Self {
        std.debug.assert(0 == (small_upload_buffer_size % d3d.D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT));
        const d3d_desc = d3d.D3D12_RESOURCE_DESC{
            .Dimension = d3d.D3D12_RESOURCE_DIMENSION_BUFFER,
            .Alignment = d3d.D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT,
            .Width = small_upload_buffer_size,
            .Height = 1,
            .DepthOrArraySize = 1,
            .MipLevels = 1,
            .Format = d3d.DXGI_FORMAT_UNKNOWN,
            .SampleDesc = .{ .Count = 1, .Quality = 0 },
            .Layout = d3d.D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            .Flags = d3d.D3D12_RESOURCE_FLAG_NONE,
        };
        const heap_info = d3d.D3D12_HEAP_PROPERTIES{
            .Type = d3d.D3D12_HEAP_TYPE_UPLOAD,
            .CPUPageProperty = d3d.D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            .MemoryPoolPreference = d3d.D3D12_MEMORY_POOL_UNKNOWN,
            .CreationNodeMask = 1,
            .VisibleNodeMask = 1,
        };
        const list_kind = if (queue_kind == .async_queue) D3D.CommandListKind.copy else D3D.CommandListKind.direct;

        var resource = try dx.device.createCommittedResource("GpuUploadHeap.small_upload_buffer", &heap_info, &d3d_desc, d3d.D3D12_RESOURCE_STATE_GENERIC_READ);
        errdefer resource.deinit();

        //crap crashes... compiler bug it looks like
        //_ = heap_info;
        //var allocation = try createGpuAllocation(.upload, false, 0, d3d_desc, d3d.D3D12_RESOURCE_STATE_GENERIC_READ, null);
        //errdefer destroyGpuAllocation(allocation);
        //const resource = gpuAllocationGetResource(allocation);

        var ca = try dx.device.createCommandAllocator(list_kind);
        errdefer ca.deinit();

        var cl = try dx.device.createCommandList(list_kind, ca, null);
        errdefer cl.deinit();
        try cl.close();

        var copy_fence = try dx.device.createFence();
        try copy_fence.signalOnCpu(0);

        return Self{
            .copy_list = .{},
            .completion_callback_list = .{},
            .small_upload_buffer = resource,
            .small_upload_offset = 0,
            .small_upload_mem = (try resource.mapEntireBuffer())[0..small_upload_buffer_size],
            .copy_command_list = cl,
            .command_allocator = ca,
            .deletion_list = try GpuDeletionList.initCapacity(allocator, 50),
            .cpu_frame_num = 1,
            .copy_fence = copy_fence,
            .mutex = .{},
            .queue_kind = queue_kind,
        };
    }

    pub fn deinit(self: *Self) void {
        self.copy_list.deinit(allocator);
        self.completion_callback_list.deinit(allocator);
        scheduleFrameResourceForDeletion(self.small_upload_buffer);
    }

    pub fn submitPendingCopies(self: *Self, queue: PlatformCommandQueue) !void {
        self.mutex.lock(); defer self.mutex.unlock();
        if (self.copy_list.hasPendingCopies()) {
            try self.copy_command_list.reset(self.command_allocator, null);
            self.copy_list.submit(self.copy_command_list, &self.deletion_list, self.cpu_frame_num, self.queue_kind == .direct_queue);
            try self.copy_command_list.close();
            queue.executeCommandList(self.copy_command_list);
            self.cpu_frame_num += 1;
            try queue.signal(self.copy_fence, self.cpu_frame_num);
        }

        //delete any finished copy sources
        //note: this should really be with the queue as if that changes across these calls then you will have a crash
        if (self.deletion_list.items.len > 0) {
            //std.debug.print("try delete gpu copies {d}\n", .{self.deletion_list.items.len});
            const gpu_frame_num = self.copy_fence.getCompletedValue();
            while (self.deletion_list.items.len > 0) {
                const item = self.deletion_list.items[0];
                if (item.frame_num < gpu_frame_num) {
                    //safe to delete now
                    _ = D3D.comReleasable(item.resource);
                    _ = self.deletion_list.orderedRemove(0);
                } else {
                    //not ready to delete yet
                    break;
                }
            }
        }
    }

    pub fn allocBufferCopy(self: *GpuUploadHeap, dst: *GpuResource, dst_off: u64, size: usize) !GpuUploadRegion {
        var region = GpuUploadRegion{
            .mem = undefined,
            .owning_heap = self,
            .copy_info = .{
                .buffer = .{
                    .src = undefined,
                    .src_off = undefined,
                    .dst = dst,
                    .dst_off = dst_off,
                    .size = size,
                    .delete_src_after_copy = false,
                },
            },
        };
        region.copy_info.buffer.delete_src_after_copy = try self.allocUpload(size, &region.copy_info.buffer.src, &region.copy_info.buffer.src_off, &region.mem);
        return region;
    }

    pub fn allocTextureCopy(self: *GpuUploadHeap, tex: *Texture, mip_idx: usize, array_idx: usize) !GpuUploadRegion {
        const mip_desc = tex.mip_descs[mip_idx];
        var region = GpuUploadRegion{
            .mem = undefined,
            .owning_heap = self,
            .copy_info = .{
                .texture = .{
                    .src = undefined,
                    .src_off = undefined,
                    .dst = &tex.gpu_resource,
                    .subresource_idx = mip_idx + (array_idx * tex.mip_count),
                    .mip_desc = mip_desc,
                    .delete_src_after_copy = false,
                },
            },
        };
        region.copy_info.texture.delete_src_after_copy = try self.allocUpload(mip_desc.size, &region.copy_info.texture.src, &region.copy_info.texture.src_off, &region.mem);
        return region;
    }

    pub fn onGpuCompletionAssumesLocked(self: *GpuUploadHeap) void {
        self.small_upload_offset = 0;
        for (self.completion_callback_list.items) |callback| {
            callback.func(callback.user_data);
        }
        self.completion_callback_list.resize(allocator, 0) catch unreachable;
    }

    fn allocUpload(self: *GpuUploadHeap, size: usize, out_resource: *PlatformResource, out_offset: *u64, out_mem: *[]u8) !bool {
        if (false) {
        //if (size <= small_upload_allocation_size) {
            self.mutex.lock(); defer self.mutex.unlock();
            //small upload path
            const copy_alignment = 512; //alignment according to d3d docs (same as D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT)
            const aligned_size = std.mem.alignForward(u64, size, copy_alignment);
            std.mem.assert(0 == (self.small_upload_offset % copy_alignment));
            const new_offset = self.small_upload_offset + aligned_size;
            if (new_offset < small_upload_buffer_size) {
                //we fit in the buffer, so return it
                out_resource.* = self.small_upload_buffer;
                out_offset.* = self.small_upload_offset;
                out_mem.* = self.small_upload_mem[self.small_upload_offset..new_offset];
                self.small_upload_offset = new_offset;
                return false;
            }
        }

        //large upload path
        const aligned_size = std.mem.alignForward(u64, size, d3d.D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT);
        const d3d_desc = d3d.D3D12_RESOURCE_DESC{
            .Dimension = d3d.D3D12_RESOURCE_DIMENSION_BUFFER,
            .Alignment = d3d.D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT,
            .Width = aligned_size,
            .Height = 1,
            .DepthOrArraySize = 1,
            .MipLevels = 1,
            .Format = d3d.DXGI_FORMAT_UNKNOWN,
            .SampleDesc = .{ .Count = 1, .Quality = 0 },
            .Layout = d3d.D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            .Flags = d3d.D3D12_RESOURCE_FLAG_NONE,
        };
        const heap_info = d3d.D3D12_HEAP_PROPERTIES{
            .Type = d3d.D3D12_HEAP_TYPE_UPLOAD,
            .CPUPageProperty = d3d.D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            .MemoryPoolPreference = d3d.D3D12_MEMORY_POOL_UNKNOWN,
            .CreationNodeMask = 1,
            .VisibleNodeMask = 1,
        };
        out_offset.* = 0;
        out_resource.* = try dx.device.createCommittedResource("GpuUploadHeap.<large upload buffer>", &heap_info, &d3d_desc, d3d.D3D12_RESOURCE_STATE_GENERIC_READ);
        out_mem.* = (try out_resource.mapEntireBuffer())[0..aligned_size];
        return true;
    }
};

// ************************************************************************************************************************
// buffer and texture updates
// ************************************************************************************************************************

pub fn beginUpdateTextureMipLevel(hdl: ResourceHandle, mip_level: usize, array_idx: usize) !GpuUploadRegion {
    var tex = textures.lookup(hdl);
    if (option_use_async_copy_queue) {
        return try currentFrame().async_upload_heap.allocTextureCopy(tex, mip_level, array_idx);
    } else {
        return try currentFrame().direct_upload_heap.allocTextureCopy(tex, mip_level, array_idx);
    }
}

pub fn updateConstantBuffer(hdl: ResourceHandle, comptime T: type, data: T) !void {
    var cb = constant_buffers.lookup(hdl);
    var heap = &currentFrame().direct_upload_heap;
    var region = try heap.allocBufferCopy(&cb.gpu_resource, 0, @sizeOf(T));
    std.mem.copy(u8, region.mem, std.mem.asBytes(&data));
    try region.queue(null, null);
}

pub fn updateBuffer(hdl: ResourceHandle, comptime T: type, offset_in_bytes: usize, data: []const T) !void {
    const len_in_bytes = data.len * @sizeOf(T);
    var buf = buffers.lookup(hdl);
    var heap = &currentFrame().direct_upload_heap;
    std.debug.assert(buf.desc.size >= (offset_in_bytes + len_in_bytes));
    var region = try heap.allocBufferCopy(&buf.gpu_resource, offset_in_bytes, len_in_bytes);
    @memcpy(region.mem.ptr[0..len_in_bytes], @as([*]const u8, @ptrCast(data.ptr)));
    try region.queue(null, null);
}

pub fn zeroEntireBuffer(hdl: ResourceHandle) !void {
    var buf = buffers.lookup(hdl);
    var heap = &currentFrame().direct_upload_heap;
    var region = try heap.allocBufferCopy(&buf.gpu_resource, 0, buf.desc.size);
    std.debug.assert(buf.desc.size <= region.mem.len);
    @memset(region.mem.ptr[0..buf.desc.size], 0);
    try region.queue(null, null);
}

// ************************************************************************************************************************
// shader compiling
// ************************************************************************************************************************

pub const ShaderCompiler = struct {
    const dependency_version = 1;
    const Self = @This();

    pub const Define = struct {
        name: []const u8,
        value: []const u8,
    };

    search_dir: []const u8,
    output_dir: []const u8,
    fxc_exe: []const u8,
    dxc_exe: []const u8,

    pub fn init(search_dir: []const u8, output_dir: []const u8) !Self {
        std.debug.assert(std.mem.endsWith(u8, search_dir, "/"));
        std.debug.assert(std.mem.endsWith(u8, output_dir, "/"));
        return Self{
            .search_dir = search_dir,
            .output_dir = output_dir,
            .fxc_exe = "C:/Program Files (x86)/Windows Kits/10/bin/10.0.22621.0/x64/fxc.exe", //TODO: fix hardcoded path
            .dxc_exe = "C:/Program Files (x86)/Windows Kits/10/bin/10.0.22621.0/x64/dxc.exe", //TODO: fix hardcoded path
        };
    }

    fn buildCompileArgList(self: *ShaderCompiler, arena_allocator: std.mem.Allocator, kind: ShaderKind, defines: []const Define, entry_point: []const u8, compiled_path: []const u8, source_path: []const u8, comptime shader_model: gdi.ShaderModel) !std.ArrayList([]const u8) {
        const arg_prefix = switch (shader_model) {
            .sm_5_1 => "/",
            .sm_6_5 => "-",
        };
        const profile_hlsl = switch (kind) {
            .invalid => { return Error.InvalidShaderKind; },
            .ps => switch(shader_model) { .sm_5_1 =>    "ps_5_1", .sm_6_5 => "ps_6_5" },
            .vs => switch(shader_model) { .sm_5_1 =>    "vs_5_1", .sm_6_5 => "vs_6_5" },
            .cs => switch(shader_model) { .sm_5_1 =>    "cs_5_1", .sm_6_5 => "cs_6_5" },
            .ms => switch(shader_model) { .sm_5_1 => unreachable, .sm_6_5 => "ms_6_5" },
            .as => switch(shader_model) { .sm_5_1 => unreachable, .sm_6_5 => "as_6_5" },
        };

        var args = try std.ArrayList([]const u8).initCapacity(arena_allocator, defines.len*2 + 20);
        const use_dxc = arg_prefix[0] == '-';
        if (use_dxc) {
            std.debug.assert(shader_model == .sm_6_5);
            try args.append(self.dxc_exe);
        } else {
            std.debug.assert(shader_model == .sm_5_1);
            try args.append(self.fxc_exe);
        }
        try args.append(arg_prefix ++ "nologo");
        try args.append(arg_prefix ++ "Vi");
        if (!use_dxc) {
            try args.append(arg_prefix ++ "enable_unbounded_descriptor_tables");
        }

        try args.append(arg_prefix ++ "D");
        switch (shader_model) {
            .sm_5_1 => try args.append("ShaderModel=51"),
            .sm_6_5 => try args.append("ShaderModel=65"),
        }
        try args.append(arg_prefix ++ "D");
        try args.append("D3D12");
        try args.append(arg_prefix ++ "D");
        try args.append(if (kind == ShaderKind.cs) "CS=1" else "CS=0");
        try args.append(arg_prefix ++ "D");
        try args.append(if (kind == ShaderKind.vs) "VS=1" else "VS=0");
        try args.append(arg_prefix ++ "D");
        try args.append(if (kind == ShaderKind.ps) "PS=1" else "PS=0");
        try args.append(arg_prefix ++ "D");
        try args.append(if (kind == ShaderKind.ms) "MS=1" else "MS=0");
        try args.append(arg_prefix ++ "D");
        try args.append(if (kind == ShaderKind.as) "AS=1" else "AS=0");

        for (defines) |define| {
            try args.append(arg_prefix ++ "D");
            try args.append(try std.fmt.allocPrint(arena_allocator, "{s}={s}", .{define.name, define.value}));
        }

        try args.append(arg_prefix ++ "Zpr");
        try args.append(arg_prefix ++ "Zi");
        try args.append(arg_prefix ++ "E");
        try args.append(entry_point);
        try args.append(arg_prefix ++ "T");
        try args.append(profile_hlsl);
        try args.append(arg_prefix ++ "Fo");
        try args.append(compiled_path);
        try args.append(source_path);

        return args;
    }

    pub fn compile(self: *ShaderCompiler, in_allocator: std.mem.Allocator, name: []const u8, kind: ShaderKind, defines: []const Define, comptime shader_model: gdi.ShaderModel) !*ShaderByteCode {
        //our return result; allocate this first in case it reduces fragmentation (due to everything else being freed)
        var bc = try in_allocator.create(ShaderByteCode);
        errdefer in_allocator.destroy(bc);

        //we use an arena allocator rather than trying to free anything during this function
        var arena_allocator = std.heap.ArenaAllocator.init(in_allocator);
        defer arena_allocator.deinit();
        var a = arena_allocator.allocator();

        const entry_point_posfix = switch (kind) {
            .invalid => { return Error.InvalidShaderKind; },
            .ps => "PixelMain",
            .vs => "VertexMain",
            .cs => "ComputeMain",
            .ms => "MeshMain",
            .as => "AmplifyMain",
        };
        const kind_str = switch (kind) {
            .invalid => { return Error.InvalidShaderKind; },
            .ps => ".ps",
            .vs => ".vs",
            .cs => ".cs",
            .ms => ".ms",
            .as => ".as",
        };

        var hash_name_buffer: [20]u8 = undefined;
        var defines_hash_name: []u8 = undefined; {
            const buf = hash_name_buffer[0..];
            var h = std.hash.Wyhash.init(42);
            for (defines) |d| {
                h.update(d.name);
                h.update(d.value);
            }
            defines_hash_name = std.fmt.bufPrintIntToSlice(buf, h.final(), 16, .lower, .{});
        }

        const entry_point = try join(a, &.{name, entry_point_posfix});
        const source_path = try join(a, &.{self.search_dir, name, ".hlsl"});
        const dep_path = try join(a, &.{self.output_dir, name, ".", defines_hash_name, kind_str, ".depends"});
        const compiled_path = try join(a, &.{self.output_dir, name, ".", defines_hash_name, kind_str, ".bytecode"});

        const force_rebuild = false;
        if (force_rebuild or !DependencyFile.isFileUpToDate(a, dep_path, dependency_version)) {
            var args = try self.buildCompileArgList(a, kind, defines, entry_point, compiled_path, source_path, shader_model);

            //send to the shader compiler
            var result = try self.exec(a, args.items);
            defer a.free(result.stdout);
            defer a.free(result.stderr);

            //generate dependency file
            var df = try DependencyFile.initEmpty(a, compiled_path, dependency_version);
            defer df.deinit();
            try df.addDependency(source_path);
            try parseDependencyList(a, result.stdout, &df);
            try df.write(dep_path);
        }

        //load the output file
        log.debug("read compiled file '{s}'\n", .{compiled_path});
        var bytecode_raw = try std.fs.cwd().readFileAlloc(in_allocator, compiled_path, 32*1024*1024);
        errdefer in_allocator.free(bytecode_raw);

        //setup our shader
        bc.kind = kind;
        bc.len = bytecode_raw.len;
        bc.code = bytecode_raw.ptr;
        bc.output_render_target_count = 0;
        bc.descriptor_bind_set = ResourceHandle.empty; //will be filled in by reflectShader()

        //reflect the file
        try reflectShader(bc, shader_model);
        std.debug.assert(!bc.descriptor_bind_set.isEmptyResource());

        return bc;
    }

    pub fn reflectShader(bc: *ShaderByteCode, shader_model: gdi.ShaderModel) !void {
        var bind_set = DescriptorBindSet.init();
        defer bind_set.deinit();
        bc.output_render_target_count = 0;

        var reflect: *d3d.ID3D12ShaderReflection = undefined;
        if (shader_model == .sm_6_5) {
            //var utils_opaque: ?*anyopaque = null;
            //try D3D.verify(dx.dxcCreateInstance.?(&D3D.CLSID_DxcUtils, &D3D.IID_DxcUtils, &utils_opaque));

            //std.debug.assert(false); //TODO: have to reflect using dxcompiler.dll

            //hack... just find the first descriptor set and hope it works
            bc.output_render_target_count = 0;
            for (descriptor_bind_sets.items, 0..) |*item, idx| {
                _ = item;
                if (!descriptor_bind_sets.is_free.isSet(idx)) {
                    bc.descriptor_bind_set = ResourceHandle{
                        .kind = descriptor_bind_sets.kind,
                        .gen = descriptor_bind_sets.generations[idx],
                        .idx = @intCast(idx),
                    };
                }
            }
            return;

        } else {
            var reflect_opaque: ?*anyopaque = null;
            try D3D.verify(dx.d3dReflect(bc.code, bc.len, &D3D.IID_ID3D12ShaderReflection, &reflect_opaque));
            reflect = D3D.d3dPtrCast(d3d.ID3D12ShaderReflection, reflect_opaque);
        }

        var d3d_desc: d3d.D3D12_SHADER_DESC = undefined;
        try D3D.verify(reflect.*.lpVtbl.*.GetDesc.?(reflect, &d3d_desc));

        var out_param_idx: c_uint = 0;
        while (out_param_idx < d3d_desc.OutputParameters) : (out_param_idx += 1) {
            var param_desc: d3d.D3D12_SIGNATURE_PARAMETER_DESC = undefined;
            if (d3d.S_OK == reflect.*.lpVtbl.*.GetOutputParameterDesc.?(reflect, out_param_idx, &param_desc)) {
//              if (gdiLogShaders) {
//                cosVerbosePrint("      output param %d : '%s' idx:%d\n", p, paramDesc.SemanticName, paramDesc.SemanticIndex);
//              }
                if (std.mem.eql(u8, std.mem.sliceTo(param_desc.SemanticName, 0), "SV_Target")) {
                    bc.output_render_target_count = @intCast(@max(bc.output_render_target_count, param_desc.SemanticIndex+1));
                }
            }
        }
//      if (gdiLogShaders) {
//        cosVerbosePrint("    output render target count : %d\n", outputRenderTargetCount);
//        cosVerbosePrint("    bound resources: %d\n", d3dDesc.BoundResources);
//      }

        var b: c_uint = 0;
        while (b < d3d_desc.BoundResources) : (b += 1) {
            var bind_desc: d3d.D3D12_SHADER_INPUT_BIND_DESC = undefined;
            if (d3d.S_OK != reflect.*.lpVtbl.*.GetResourceBindingDesc.?(reflect, b, &bind_desc)) {
                continue;
            }

            //skip these two kind for now
            if ( (bind_desc.Space == ShaderSpace.draw_constants) or (bind_desc.Space == ShaderSpace.static_sampler) ) {
                continue;
            }

//          if (gdiLogShaders) {
//            cosVerbosePrint("  '%s': type:%d bindPoint:%d bindCount:%d space:%d flags:%d\n", d.Name, d.Type, d.BindPoint, d.BindCount, d.Space, d.uFlags);
//          }
//          //  cosInfoPrint("  '%s': type:%d bindPoint:%d bindCount:%d space:%d flags:%d\n", d.Name, d.Type, d.BindPoint, d.BindCount, d.Space, d.uFlags);

            var bind_info = DescriptorBindInfo{
                .name = try allocator.dupeZ(u8, bind_desc.Name[0..std.mem.indexOfScalar(u8, bind_desc.Name[0..1000000], 0).?]),
                .space = bind_desc.Space,
                .start = bind_desc.BindPoint,
                .count = bind_desc.BindCount,
                .kind = undefined,
                .value_kind = undefined,
                .heap_offset = 0xffffffff,
            };
            errdefer allocator.free(bind_info.name);

            switch (bind_desc.Type) {
                d3d.D3D_SIT_SAMPLER => {
                    bind_info.value_kind = ResourceKind.sampler;
                    bind_info.kind = DescriptorKind.sam;
                },
                d3d.D3D_SIT_TBUFFER, d3d.D3D_SIT_TEXTURE => {
                    if (bind_desc.Dimension == d3d.D3D12_RESOURCE_DIMENSION_BUFFER) {
                        bind_info.value_kind = ResourceKind.buffer;
                    } else {
                        bind_info.value_kind = ResourceKind.texture;
                    }
                    bind_info.kind = DescriptorKind.srv;
                },
                d3d.D3D_SIT_CBUFFER => {
                    bind_info.value_kind = ResourceKind.constant_buffer;
                    bind_info.kind = DescriptorKind.cbv;
                },
                d3d.D3D_SIT_UAV_APPEND_STRUCTURED,
                d3d.D3D_SIT_UAV_CONSUME_STRUCTURED,
                d3d.D3D_SIT_UAV_RWBYTEADDRESS,
                d3d.D3D_SIT_UAV_RWSTRUCTURED,
                d3d.D3D_SIT_UAV_RWSTRUCTURED_WITH_COUNTER,
                d3d.D3D_SIT_UAV_RWTYPED => {
                    if (bind_desc.Dimension == d3d.D3D12_RESOURCE_DIMENSION_BUFFER) {
                        bind_info.value_kind = ResourceKind.buffer;
                    } else {
                        bind_info.value_kind = ResourceKind.texture;
                    }
                    bind_info.kind = DescriptorKind.uav;
                },
                d3d.D3D_SIT_BYTEADDRESS, d3d.D3D_SIT_STRUCTURED => {
                    std.debug.assert(bind_desc.Dimension == d3d.D3D12_RESOURCE_DIMENSION_BUFFER);
                    bind_info.value_kind = ResourceKind.buffer;
                    bind_info.kind = DescriptorKind.srv;
                },
                else => unreachable, //invalid type
            }

            if (!bind_set.mergeBindInfo(bind_info)) {
                return Error.ShaderHasOverlappingDescriptors;
            }
        }

        bc.descriptor_bind_set = try findOrCreateDescriptorBindSet(bind_set);
    }

    fn parseDependencyList(a: std.mem.Allocator, stdout: []const u8, df: *DependencyFile) !void {
        //parse out "Resolved to [" <path> "]"
        const needle: []const u8 = "Resolved to [";
        var idx: usize = 0;
        while (std.mem.indexOfPos(u8, stdout, idx, needle)) |found| {
            const start = found+needle.len;
            const end = std.mem.indexOfPos(u8, stdout, start, "]") orelse return Error.BadlyFormedOutputFromShaderCompile;
            const path = stdout[start..end];
            log.info("    => shader dependency '{s}'\n", .{path});
            try df.addDependency(try a.dupe(u8, path));
            idx = found + (needle.len);
        }
    }

    fn exec(self: *Self, in_allocator: std.mem.Allocator, argv: []const []const u8) !std.ChildProcess.ExecResult {
        _ = self;
        const should_log = true;
        if (should_log) {
            log.info("executing: {s}\n", .{argv});
        }
        var result = try std.ChildProcess.exec(.{
            .allocator = in_allocator,
            .argv = argv
        });
        errdefer in_allocator.free(result.stdout);
        errdefer in_allocator.free(result.stderr);
        switch (result.term) {
            .Exited => if (result.term.Exited != 0) {
                if (should_log) {
                    log.info("result term: {}\n", .{result.term});
                    log.info("result stdout: {s}\n", .{result.stdout});
                    log.info("result stderr: {s}\n", .{result.stderr});
                }
                return Error.ProcessExecutionFailed;
            },
            else => { return Error.ProcessExecutionFailed; },
        }
        return result;
    }

    fn joinPath(in_allocator: std.mem.Allocator, paths: []const []const u8) ![]u8 {
        return std.fs.path.join(in_allocator, paths);
    }

    fn join(in_allocator: std.mem.Allocator, strs: []const []const u8) ![]u8 {
        return std.mem.join(in_allocator, "", strs);
    }
};

// ************************************************************************************************************************
// info about formats
// ************************************************************************************************************************

const FormatInfo = struct {
    format: d3d.DXGI_FORMAT,
    size_in_bytes: u8,
    is_block_compressed: bool,
    is_depth_format: bool,
    is_render_target_supported: bool,
    is_blendable: bool,
    non_srgb_format: ?d3d.DXGI_FORMAT,
};

pub fn formatQueryInfo(format: gdi.Format) FormatInfo {
    return switch (format) {
        .invalid              => .{ .format=d3d.DXGI_FORMAT_R8G8B8A8_UNORM      , .size_in_bytes= 4, .is_block_compressed=false, .is_depth_format=false, .is_render_target_supported=false, .is_blendable=false , .non_srgb_format=null },
        .swap_chain           => .{ .format=d3d.DXGI_FORMAT_B8G8R8A8_UNORM      , .size_in_bytes= 4, .is_block_compressed=false, .is_depth_format=false, .is_render_target_supported=true , .is_blendable=true  , .non_srgb_format=null },
        .r32g32b32a32_float   => .{ .format=d3d.DXGI_FORMAT_R32G32B32A32_FLOAT  , .size_in_bytes=32, .is_block_compressed=false, .is_depth_format=false, .is_render_target_supported=true , .is_blendable=true  , .non_srgb_format=null },
        .r32g32b32a32_uint    => .{ .format=d3d.DXGI_FORMAT_R32G32B32A32_UINT   , .size_in_bytes=32, .is_block_compressed=false, .is_depth_format=false, .is_render_target_supported=true , .is_blendable=false , .non_srgb_format=null },
        .r32g32b32a32_sint    => .{ .format=d3d.DXGI_FORMAT_R32G32B32A32_SINT   , .size_in_bytes=32, .is_block_compressed=false, .is_depth_format=false, .is_render_target_supported=true , .is_blendable=false , .non_srgb_format=null },
        .r16g16b16a16_float   => .{ .format=d3d.DXGI_FORMAT_R16G16B16A16_FLOAT  , .size_in_bytes= 8, .is_block_compressed=false, .is_depth_format=false, .is_render_target_supported=true , .is_blendable=true  , .non_srgb_format=null },
        .r16g16b16a16_uint    => .{ .format=d3d.DXGI_FORMAT_R16G16B16A16_UINT   , .size_in_bytes= 8, .is_block_compressed=false, .is_depth_format=false, .is_render_target_supported=true , .is_blendable=false , .non_srgb_format=null },
        .r8g8b8a8_unorm       => .{ .format=d3d.DXGI_FORMAT_R8G8B8A8_UNORM      , .size_in_bytes= 4, .is_block_compressed=false, .is_depth_format=false, .is_render_target_supported=true , .is_blendable=true  , .non_srgb_format=null },
        .r8g8b8a8_unorm_srgb  => .{ .format=d3d.DXGI_FORMAT_R8G8B8A8_UNORM_SRGB , .size_in_bytes= 4, .is_block_compressed=false, .is_depth_format=false, .is_render_target_supported=true , .is_blendable=true  , .non_srgb_format=d3d.DXGI_FORMAT_R8G8B8A8_UNORM },
        .b8g8r8a8_unorm       => .{ .format=d3d.DXGI_FORMAT_B8G8R8A8_UNORM      , .size_in_bytes= 4, .is_block_compressed=false, .is_depth_format=false, .is_render_target_supported=true , .is_blendable=true  , .non_srgb_format=null },
        .b8g8r8a8_unorm_srgb  => .{ .format=d3d.DXGI_FORMAT_B8G8R8A8_UNORM_SRGB , .size_in_bytes= 4, .is_block_compressed=false, .is_depth_format=false, .is_render_target_supported=true , .is_blendable=true  , .non_srgb_format=d3d.DXGI_FORMAT_B8G8R8A8_UNORM },
        .r32_float            => .{ .format=d3d.DXGI_FORMAT_R32_FLOAT           , .size_in_bytes= 4, .is_block_compressed=false, .is_depth_format=false, .is_render_target_supported=true , .is_blendable=true  , .non_srgb_format=null },
        .r32_uint             => .{ .format=d3d.DXGI_FORMAT_R32_UINT            , .size_in_bytes= 4, .is_block_compressed=false, .is_depth_format=false, .is_render_target_supported=true , .is_blendable=false , .non_srgb_format=null },
        .r16_uint             => .{ .format=d3d.DXGI_FORMAT_R16_UINT            , .size_in_bytes= 2, .is_block_compressed=false, .is_depth_format=false, .is_render_target_supported=true , .is_blendable=false , .non_srgb_format=null },
        .r16_float            => .{ .format=d3d.DXGI_FORMAT_R16_FLOAT           , .size_in_bytes= 2, .is_block_compressed=false, .is_depth_format=false, .is_render_target_supported=true , .is_blendable=false , .non_srgb_format=null },
        .d32_float            => .{ .format=d3d.DXGI_FORMAT_D32_FLOAT           , .size_in_bytes= 4, .is_block_compressed=false, .is_depth_format=true , .is_render_target_supported=true , .is_blendable=false , .non_srgb_format=null },
        .d32_float_s8x24_uint => .{ .format=d3d.DXGI_FORMAT_D32_FLOAT_S8X24_UINT, .size_in_bytes= 8, .is_block_compressed=false, .is_depth_format=true , .is_render_target_supported=true , .is_blendable=false , .non_srgb_format=null },
        .bc1_unorm            => .{ .format=d3d.DXGI_FORMAT_BC1_UNORM           , .size_in_bytes= 8, .is_block_compressed=true , .is_depth_format=false, .is_render_target_supported=false, .is_blendable=false , .non_srgb_format=null },
        .bc1_unorm_srgb       => .{ .format=d3d.DXGI_FORMAT_BC1_UNORM_SRGB      , .size_in_bytes= 8, .is_block_compressed=true , .is_depth_format=false, .is_render_target_supported=false, .is_blendable=false , .non_srgb_format=d3d.DXGI_FORMAT_BC1_UNORM },
        .bc2_unorm            => .{ .format=d3d.DXGI_FORMAT_BC2_UNORM           , .size_in_bytes=16, .is_block_compressed=true , .is_depth_format=false, .is_render_target_supported=false, .is_blendable=false , .non_srgb_format=null },
        .bc2_unorm_srgb       => .{ .format=d3d.DXGI_FORMAT_BC2_UNORM_SRGB      , .size_in_bytes=16, .is_block_compressed=true , .is_depth_format=false, .is_render_target_supported=false, .is_blendable=false , .non_srgb_format=d3d.DXGI_FORMAT_BC2_UNORM },
        .bc3_unorm            => .{ .format=d3d.DXGI_FORMAT_BC3_UNORM           , .size_in_bytes=16, .is_block_compressed=true , .is_depth_format=false, .is_render_target_supported=false, .is_blendable=false , .non_srgb_format=null },
        .bc3_unorm_srgb       => .{ .format=d3d.DXGI_FORMAT_BC3_UNORM_SRGB      , .size_in_bytes=16, .is_block_compressed=true , .is_depth_format=false, .is_render_target_supported=false, .is_blendable=false , .non_srgb_format=d3d.DXGI_FORMAT_BC3_UNORM },
        .bc4_unorm            => .{ .format=d3d.DXGI_FORMAT_BC4_UNORM           , .size_in_bytes= 8, .is_block_compressed=true , .is_depth_format=false, .is_render_target_supported=false, .is_blendable=false , .non_srgb_format=null },
        .bc4_snorm            => .{ .format=d3d.DXGI_FORMAT_BC4_SNORM           , .size_in_bytes= 8, .is_block_compressed=true , .is_depth_format=false, .is_render_target_supported=false, .is_blendable=false , .non_srgb_format=null },
        .bc5_unorm            => .{ .format=d3d.DXGI_FORMAT_BC5_UNORM           , .size_in_bytes=16, .is_block_compressed=true , .is_depth_format=false, .is_render_target_supported=false, .is_blendable=false , .non_srgb_format=null },
        .bc5_snorm            => .{ .format=d3d.DXGI_FORMAT_BC5_SNORM           , .size_in_bytes=16, .is_block_compressed=true , .is_depth_format=false, .is_render_target_supported=false, .is_blendable=false , .non_srgb_format=null },
        .bc6h_uf16            => .{ .format=d3d.DXGI_FORMAT_BC6H_UF16           , .size_in_bytes= 0, .is_block_compressed=true , .is_depth_format=false, .is_render_target_supported=false, .is_blendable=false , .non_srgb_format=null },
        .bc6h_sf16            => .{ .format=d3d.DXGI_FORMAT_BC6H_SF16           , .size_in_bytes= 0, .is_block_compressed=true , .is_depth_format=false, .is_render_target_supported=false, .is_blendable=false , .non_srgb_format=null },
        .bc7_unorm            => .{ .format=d3d.DXGI_FORMAT_BC7_UNORM           , .size_in_bytes= 0, .is_block_compressed=true , .is_depth_format=false, .is_render_target_supported=false, .is_blendable=false , .non_srgb_format=null },
        .bc7_unorm_srgb       => .{ .format=d3d.DXGI_FORMAT_BC7_UNORM_SRGB      , .size_in_bytes= 0, .is_block_compressed=true , .is_depth_format=false, .is_render_target_supported=false, .is_blendable=false , .non_srgb_format=d3d.DXGI_FORMAT_BC7_UNORM },
        .r9g9b9e5_sharedexp   => .{ .format=d3d.DXGI_FORMAT_R9G9B9E5_SHAREDEXP  , .size_in_bytes= 4, .is_block_compressed=false, .is_depth_format=false, .is_render_target_supported=true , .is_blendable=false , .non_srgb_format=null },
        .r32g32_float         => .{ .format=d3d.DXGI_FORMAT_R32G32_FLOAT        , .size_in_bytes= 8, .is_block_compressed=false, .is_depth_format=false, .is_render_target_supported=true , .is_blendable=false , .non_srgb_format=null },
        .r8_unorm             => .{ .format=d3d.DXGI_FORMAT_R8_UNORM            , .size_in_bytes= 1, .is_block_compressed=false, .is_depth_format=false, .is_render_target_supported=true , .is_blendable=false , .non_srgb_format=null },
        .r16g16_float         => .{ .format=d3d.DXGI_FORMAT_R16G16_FLOAT        , .size_in_bytes= 4, .is_block_compressed=false, .is_depth_format=false, .is_render_target_supported=true , .is_blendable=false , .non_srgb_format=null },
        .r8g8_unorm           => .{ .format=d3d.DXGI_FORMAT_R8G8_UNORM          , .size_in_bytes= 2, .is_block_compressed=false, .is_depth_format=false, .is_render_target_supported=true , .is_blendable=false , .non_srgb_format=null },
    };
}

// ************************************************************************************************************************
// thread safe pool
// ************************************************************************************************************************

pub fn ThreadSafePool(comptime T: type, comptime max_capacity: usize) type {
    return struct {
        const aprox_max_capacity = max_capacity;
        const Self = @This();
        const FreeList = std.atomic.Stack(T);
        pub const Node = FreeList.Node;

        allocator: std.mem.Allocator,
        free_list: FreeList,
        aprox_capacity: std.atomic.Atomic(usize),

        pub fn init(in_allocator: std.mem.Allocator) Self {
            return .{
                .allocator = in_allocator,
                .free_list = FreeList.init(),
                .aprox_capacity = std.atomic.Atomic(usize).init(0),
            };
        }

        pub fn acquire(self: *Self) !*Node {
            //either grab from the free list or create a new one
            if (self.free_list.pop()) |node| {
                _ = self.aprox_capacity.fetchSub(1, std.atomic.Ordering.AcqRel);
                return node;
            } else {
                var node = try self.allocator.create(Node);
                try node.data.init();
                return node;
            }
        }

        pub fn release(self: *Self, node: *Node) !void {
            //if we have too many stored, then release this one; otherwise put into the free list; this isn't exact and may be off by one, but that doesn't really matter
            if (self.aprox_capacity.fetchAdd(1, std.atomic.Ordering.AcqRel) > aprox_max_capacity) {
                _ = self.aprox_capacity.fetchSub(1, std.atomic.Ordering.AcqRel);
                node.data.deinit();
                self.allocator.destroy(node);
            } else {
                //we could use typeinfo to find out of this function exists or not at comptime ... maybe?
                node.data.reinit();
                self.free_list.push(node);
            }
        }
    };
}

// ************************************************************************************************************************
// D3D12MemAlloc helper
// ************************************************************************************************************************

fn createGpuAllocation(
    heap_type: enum { default, upload, readback },
    force_committed: bool,
    extra_heap_flags: AMDMemAlloc.D3D12_HEAP_FLAGS,
    resource_desc: d3d.D3D12_RESOURCE_DESC,
    init_state: d3d.D3D12_RESOURCE_STATES,
    clear_value: ?*const d3d.D3D12_CLEAR_VALUE,
) !*AMDMemAlloc.GpuAllocation {
    const desc = AMDMemAlloc.ALLOCATION_DESC{
        .Flags = if (force_committed) AMDMemAlloc.ALLOCATION_FLAG_COMMITTED else AMDMemAlloc.ALLOCATION_FLAG_NONE,
        .HeapType = switch (heap_type) {
            .default  => AMDMemAlloc.D3D12_HEAP_TYPE_DEFAULT,
            .upload   => AMDMemAlloc.D3D12_HEAP_TYPE_UPLOAD,
            .readback => AMDMemAlloc.D3D12_HEAP_TYPE_READBACK,
        },
        .ExtraHeapFlags = extra_heap_flags,
        .CustomPool = null,
        .pPrivateData = null,
    };
    var opaque_resource: ?*anyopaque = null;
    var result: ?*AMDMemAlloc.GpuAllocation = null;
    try D3D.verify(AMDMemAlloc.CreateResource(
        gpu_mem_allocator,
        &desc,
        @ptrCast(&resource_desc),
        @bitCast(init_state),
        @ptrCast(clear_value),
        &result,
        @ptrCast(&D3D.IID_ID3D12Resource),
        &opaque_resource
    ));
    //the ref count should be two at this point; so release this reference now
    const resource = D3D.d3dPtrCast(d3d.ID3D12Resource, opaque_resource);
    const count = D3D.vtbl(resource).Release.?(resource);
    std.debug.assert(count == 1);
    return result.?;
}

fn destroyGpuAllocation(a: *AMDMemAlloc.GpuAllocation) void {
    _ = AMDMemAlloc.ReleaseAllocation(a);
    //std.debug.assert(val == 0);
}

fn gpuAllocationGetResource(a: *AMDMemAlloc.GpuAllocation) PlatformResource {
    return PlatformResource{ .resource = D3D.d3dPtrCast(d3d.ID3D12Resource, a.resource) };
}

pub inline fn beginGpuProfileZone(comptime src: std.builtin.SourceLocation, color: u32, name_str: ?[*:0]const u8) u32 {
    return D3D.beginGpuProfileZone(dx, src, color, name_str);
}

pub fn endGpuProfileZone(id: gdi.GpuProfileID) void {
    D3D.endGpuProfileZone(dx, id);
}

pub fn getPlatformResource(hdl: ResourceHandle) PlatformResource {
    switch (hdl.kind) {
        .buffer => {
            const buf = buffers.lookup(hdl);
            return buf.gpu_resource.resource;
        },
        .render_target => {
            const rt = render_targets.lookup(hdl);
            return rt.gpu_resource.resource;
        },
        else => std.debug.assert(false), //not implemented yet
    }
    unreachable;
}
