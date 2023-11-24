const std = @import("std");
pub const d3d = @cImport({
    //@cDefine("TRACY_ENABLE", "");
    @cInclude("dxgi1_4.h");
    @cInclude("d3d12.h");
    @cInclude("d3dcompiler.h");
    @cInclude("d3d12sdklayers.h");
    @cInclude("../d3d/c_pix.h");
    @cInclude("tracy/TracyC.h");
    @cInclude("../d3d/TracyD3D12.h");
});
pub const c = d3d;
pub const Format = c.DXGI_FORMAT;

pub const IID_ID3D12Device           = d3d.GUID{ .Data1 = 0x189819f1, .Data2 = 0x1db6, .Data3 = 0x4b57, .Data4 = .{ 0xbe, 0x54, 0x18, 0x21, 0x33, 0x9b, 0x85, 0xf7 }, };
pub const IID_ID3D12Device2          = d3d.GUID{ .Data1 = 0x30baa41e, .Data2 = 0xb15b, .Data3 = 0x475c, .Data4 = .{ 0xa0, 0xbb, 0x1a, 0xf5, 0xc5, 0xb6, 0x43, 0x28 }, };
pub const IID_ID3D12RootSignature    = d3d.GUID{ .Data1 = 0xc54a6b66, .Data2 = 0x72df, .Data3 = 0x4ee8, .Data4 = .{ 0x8b, 0xe5, 0xa9, 0x46, 0xa1, 0x42, 0x92, 0x14 }, };
pub const IID_IDXGIFactory1          = d3d.GUID{ .Data1 = 0x770aae78, .Data2 = 0xf26f, .Data3 = 0x4dba, .Data4 = .{ 0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87 }, };
pub const IID_IDXGIFactory4          = d3d.GUID{ .Data1 = 0x1bc6ea02, .Data2 = 0xef36, .Data3 = 0x464f, .Data4 = .{ 0xbf, 0x0c, 0x21, 0xca, 0x39, 0xe5, 0x16, 0x8a }, };
pub const IID_ID3D12PipelineState    = d3d.GUID{ .Data1 = 0x765a30f3, .Data2 = 0xf624, .Data3 = 0x4c6f, .Data4 = .{ 0xa8, 0x28, 0xac, 0xe9, 0x48, 0x62, 0x24, 0x45 }, };
pub const IID_ID3D12Resource         = d3d.GUID{ .Data1 = 0x696442be, .Data2 = 0xa72e, .Data3 = 0x4059, .Data4 = .{ 0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad }, };
pub const IID_ID3D12DescriptorHeap   = d3d.GUID{ .Data1 = 0x8efb471d, .Data2 = 0x616c, .Data3 = 0x4f49, .Data4 = .{ 0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51 }, };
pub const IID_IGraphicsCommandList   = d3d.GUID{ .Data1 = 0x5b160d0f, .Data2 = 0xac1b, .Data3 = 0x4185, .Data4 = .{ 0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55 }, };
pub const IID_ICommandAllocator      = d3d.GUID{ .Data1 = 0x6102dee4, .Data2 = 0xaf59, .Data3 = 0x4b09, .Data4 = .{ 0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24 }, };
pub const IID_ICommandSignature      = d3d.GUID{ .Data1 = 0xc36a797c, .Data2 = 0xec80, .Data3 = 0x4f0a, .Data4 = .{ 0x89, 0x85, 0xa7, 0xb2, 0x47, 0x50, 0x82, 0xd1 }, };
pub const IID_ICommandQueue          = d3d.GUID{ .Data1 = 0x0ec870a6, .Data2 = 0x5d7e, .Data3 = 0x4c22, .Data4 = .{ 0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed }, };
pub const IID_ISwapChain3            = d3d.GUID{ .Data1 = 0x94d99bdb, .Data2 = 0xf1f8, .Data3 = 0x4ab0, .Data4 = .{ 0xb2, 0x36, 0x7d, 0xa0, 0x17, 0x0e, 0xda, 0xb1 }, };
pub const IID_ID3D12Fence            = d3d.GUID{ .Data1 = 0x0a753dcf, .Data2 = 0xc4d8, .Data3 = 0x4b91, .Data4 = .{ 0xad, 0xf6, 0xbe, 0x5a, 0x60, 0xd9, 0x5a, 0x76 }, };
pub const IID_ID3D12Debug1           = d3d.GUID{ .Data1 = 0xaffaa4ca, .Data2 = 0x63fe, .Data3 = 0x4d8e, .Data4 = .{ 0xb8, 0xad, 0x15, 0x90, 0x00, 0xaf, 0x43, 0x04 }, };
pub const IID_ID3D12ShaderReflection = d3d.GUID{ .Data1 = 0x5a58797d, .Data2 = 0xa72c, .Data3 = 0x478d, .Data4 = .{ 0x8b, 0xa2, 0xef, 0xc6, 0xb0, 0xef, 0xe8, 0x8e }, };
pub const WKPDID_D3DDebugObjectName  = d3d.GUID{ .Data1 = 0x429b8c22, .Data2 = 0x9188, .Data3 = 0x4b0c, .Data4 = .{ 0x87, 0x42, 0xac, 0xb0, 0xbf, 0x85, 0xc2, 0x00 }, };
pub const CLSID_DxcUtils             = d3d.GUID{ .Data1 = 0x6245d6af, .Data2 = 0x66e0, .Data3 = 0x48fd, .Data4 = .{ 0x80, 0xb4, 0x4d, 0x27, 0x17, 0x96, 0x74, 0x8c }, };
pub const IID_DxcUtils               = d3d.GUID{ .Data1 = 0x4605C4CB, .Data2 = 0x2019, .Data3 = 0x492A, .Data4 = .{ 0xAD, 0xA4, 0x65, 0xF2, 0x0B, 0xB7, 0xD6, 0x7F }, };

const Error = error{
    GenericD3DFailure,
    LookupRootSignatureFunctionFailed,
    LookupGetDebugInterfaceFunctionFailed,
    LookupD3DCompileFunctionFailed,
    LookupD3DReflectFunctionFailed,
    LookupDXCreateInstanceFunctionFailed,
    LookupCreateDXGIFactoryFunctionFailed,
    CreateDXGIFactoryFailed,
    CouldNotFindValidAdapter,
    LookupCreateDeviceFunctionFailed,
    CreateDeviceFailed,
    CreateCommitedResourceFailed,
    CreateDescriptorHeapFailed,
    CpuDescriptorHeapCreationShouldBeValid,
    CreatePipelineStateObjectFailed,
    CreateCommandAllocatorFailed,
    CreateCommandSignatureFailed,
    CreateCommandListFailed,
    CreateRootSignatureFailed,
    ResetCommandAllocatorFailed,
    ShaderVisibleDescriptorHeapsCanOnlyBeCbvSrvUavOrSampler,
    PresentD3DFailure,
};
const D3D = @This();

const DxcCreateInstanceFuncPtr = ?*const fn(rclsid: [*c]const d3d.GUID, riid: [*c]const d3d.GUID, ppv: [*c]?*anyopaque) d3d.HRESULT;
const D3DCompileFuncPtr = *const @TypeOf(d3d.D3DCompile);
const D3DReflectFuncPtr = *const @TypeOf(d3d.D3DReflect);
const SerializeRootSigFuncPtr = *const @TypeOf(c.D3D12SerializeRootSignature);

pub const log = std.log.scoped(.d3d);

allocator: std.mem.Allocator,
d3d_lib: std.DynLib,
dxgi_lib: std.DynLib,
compiler_lib: std.DynLib,
dxcompiler_lib: std.DynLib,
factory: *d3d.IDXGIFactory4,
device: Device,
dxgi_adapter: *c.IDXGIAdapter,
command_queue: CommandQueue,
render_target_descriptor_heap: DescriptorHeap,
depth_stencil_descriptor_heap: DescriptorHeap,
staging_descriptor_heap: DescriptorHeap,
null_descriptor_cbv: CpuDescriptorHandle,
null_descriptor_srv: CpuDescriptorHandle,
null_descriptor_uav: CpuDescriptorHandle,
null_descriptor_sam: CpuDescriptorHandle,
d3dCompile: D3DCompileFuncPtr,
d3dReflect: D3DReflectFuncPtr,
dxcCreateInstance: DxcCreateInstanceFuncPtr,
flush_fence: Fence,
flush_fence_value: u64,
serialize_root_signature_func: SerializeRootSigFuncPtr,
tracy_queue_ctx: ?*anyopaque = undefined,

pub fn verify(r: c.HRESULT) !void {
    switch (r) {
        c.S_OK => { return; },
        else => {
            log.err("D3D.verify() failed! with result {x}\n", .{@as(u32, @bitCast(r))});
            return Error.GenericD3DFailure;
        },
    }
}

pub fn verifyUsingError(r: c.HRESULT, e: Error) !void {
    if (r != c.S_OK) {
        log.err("D3D.verifyUsingError() failed! with result {x}\n", .{@as(u32, @bitCast(r))});
        return e;
    }
}

pub fn d3dPtrCast(comptime T: anytype, opaque_ptr: ?*anyopaque) *T {
    return @alignCast(@ptrCast(opaque_ptr.?));
}

pub inline fn vtbl(com: anytype) @TypeOf(com.*.lpVtbl.*) {
    return com.*.lpVtbl.*;
}

pub fn releaseComPtr(com: anytype) void {
    _ = vtbl(com).Release.?(com);
}

pub fn setResourceName(com: anytype, comptime name: [:0]const u8) void {
    _ = vtbl(com).SetName.?(com, std.unicode.utf8ToUtf16LeStringLiteral(name));
}

pub const AnyComReleasable = struct {
    pub const ReleaseFuncPtr = *const fn (*anyopaque) callconv(.C) d3d.ULONG;
    ptr: *anyopaque,
    release_func: ReleaseFuncPtr,

    pub fn release(self: *AnyComReleasable) void {
        _ = self.release_func(self.ptr);
    }
};

pub fn comReleasable(com: anytype) AnyComReleasable {
    return switch (@TypeOf(com)) {
        Resource => .{
            .ptr = com.resource,
            .release_func = @ptrCast(vtbl(com.resource).Release.?),
        },
        PipelineState => .{
            .ptr = com.pso,
            .release_func = @ptrCast(vtbl(com.pso).Release.?),
        },
        else => @compileError("comReleasable: unknown resource type " ++ @typeName(@TypeOf(com))),
    };
}

fn vendorIdToName(id: c.UINT) []const u8 {
    return switch (id) {
        4098  => "AMD",
        4318  => "Nvidia",
        32902 => "Intel",
        5140  => "Microsoft",
        else  => "Unknown",
    };
}

pub fn init(allocator: std.mem.Allocator) !D3D {
    var dx: D3D = undefined;
    dx.allocator = allocator;

    pixInitModule();
    pixSetEnableHUD(true);

    dx.d3d_lib = try std.DynLib.open("d3d12.dll");
    errdefer dx.d3d_lib.close();
    dx.dxgi_lib = try std.DynLib.open("dxgi.dll");
    errdefer dx.dxgi_lib.close();
    dx.compiler_lib = try std.DynLib.open(c.D3DCOMPILER_DLL_A);
    errdefer dx.compiler_lib.close();
    dx.dxcompiler_lib = try std.DynLib.open("dxcompiler.dll");
    errdefer dx.dxcompiler_lib.close();
    dx.d3dCompile = dx.compiler_lib.lookup(D3DCompileFuncPtr, "D3DCompile") orelse return Error.LookupD3DCompileFunctionFailed;
    dx.d3dReflect = dx.compiler_lib.lookup(D3DReflectFuncPtr, "D3DReflect") orelse return Error.LookupD3DReflectFunctionFailed;
    dx.dxcCreateInstance = dx.dxcompiler_lib.lookup(DxcCreateInstanceFuncPtr, "DxcCreateInstance") orelse return Error.LookupDXCreateInstanceFunctionFailed;

    if (true) {
        var lib = try std.DynLib.open("d3d12SDKLayers.dll");
        const get_debug_face_func = dx.d3d_lib.lookup(*const @TypeOf(d3d.D3D12GetDebugInterface), "D3D12GetDebugInterface") orelse return Error.LookupGetDebugInterfaceFunctionFailed;
        var debug_controller_opaque: ?*anyopaque = null;
        try verify(get_debug_face_func(&IID_ID3D12Debug1, &debug_controller_opaque));
        if (debug_controller_opaque) |ptr| {
            var debug = d3dPtrCast(d3d.ID3D12Debug1, ptr);
            vtbl(debug).EnableDebugLayer.?(debug);
            _ = releaseComPtr(debug);
            log.info("debug layer is enabled!\n", .{});
        }
        _ = lib;
        //lib.close();
    }

    const create_dxgi_factory_func = dx.dxgi_lib.lookup(*const @TypeOf(d3d.CreateDXGIFactory1), "CreateDXGIFactory1") orelse return Error.LookupCreateDXGIFactoryFunctionFailed;
    var factory: ?*anyopaque = null;
    switch (create_dxgi_factory_func(&IID_IDXGIFactory4, &factory)) {
        d3d.S_OK => { dx.factory = d3dPtrCast(d3d.IDXGIFactory4, factory); },
        else => { return Error.CreateDXGIFactoryFailed; },
    }

    const vendor_amd = 4098;
    const vendor_nvidia = 4318;
    const vendor_intel = 32902;
    const vendor_microsoft = 5140;

    var selected_adapter: ?*c.IDXGIAdapter = null;
    var selected_adapter_priority: isize = -1;
    var selected_adapter_desc: c.DXGI_ADAPTER_DESC = undefined;
    var adapter_idx: c.UINT = 0;
    var adapter: ?*c.IDXGIAdapter = null;
    while (vtbl(dx.factory).EnumAdapters.?(dx.factory, adapter_idx, &adapter) == c.S_OK) : (adapter_idx += 1) {
        var desc: c.DXGI_ADAPTER_DESC = undefined;
        if (c.S_OK == vtbl(adapter.?).GetDesc.?(adapter, &desc)) {
            //const max_mem_size = desc.DedicatedVideoMemory + desc.DedicatedSystemMemory + desc.SharedSystemMemory;
//                    if (gdiLogAdapters) {
//                        cosVerbosePrint(" dxgi adapter %d\n", a);
//                        cosVerbosePrint(" ==>desc(%S)\n", desc1.Description);
//                        cosVerbosePrint(" ==>vendorID(%d) deviceID(%d) subSysID(%d) rev(%d) flags(%d)\n", desc1.VendorId, desc1.DeviceId, desc1.SubSysId, desc1.Revision, desc1.Flags);
//                        cosVerbosePrint(" ==>DedVidMem(%lldmb) DedSysMem(%lldmb) SharedSysMem(%lldmb)\n", desc1.DedicatedVideoMemory>>20, desc1.DedicatedSystemMemory>>20, desc1.SharedSystemMemory>>20);
//                    }
            const priority: isize = switch (desc.VendorId) {
                vendor_amd, vendor_nvidia => 10,
                vendor_intel => 3,
                vendor_microsoft => 1,
                else => 0,
            };
            if ( (selected_adapter == null) or (selected_adapter_priority < priority) ) {
                selected_adapter = adapter;
                selected_adapter_priority = priority;
                selected_adapter_desc = desc;
            }
        }
    }

    if (selected_adapter == null) {
        return Error.CouldNotFindValidAdapter;
    }

    const gb = 1024*1024*1024;
    log.info("Selected Adapter [Idx {}] [VendorID: {}] [VendorName: {s}] [DeviceID: {}]\n", .{ adapter_idx, selected_adapter_desc.VendorId, vendorIdToName(selected_adapter_desc.VendorId), selected_adapter_desc.DeviceId });
    log.info("  [VRAM: {d:.3}GB] [SysMem: {d:.3}GB] [SharedMem: {d:.3}GB]\n", .{ @as(f64, @floatFromInt(selected_adapter_desc.DedicatedVideoMemory)) / gb, @as(f64, @floatFromInt(selected_adapter_desc.DedicatedSystemMemory)) / gb, @as(f64, @floatFromInt(selected_adapter_desc.SharedSystemMemory)) / gb });

    const create_func = dx.d3d_lib.lookup(*const @TypeOf(d3d.D3D12CreateDevice), "D3D12CreateDevice") orelse return Error.LookupCreateDeviceFunctionFailed;
    var opaque_device: ?*anyopaque = null;
    try verifyUsingError(create_func(@as(*d3d.IUnknown, @ptrCast(selected_adapter.?)), c.D3D_FEATURE_LEVEL_12_1, &IID_ID3D12Device2, &opaque_device), Error.CreateDeviceFailed);
    dx.device = Device.init(d3dPtrCast(d3d.ID3D12Device, opaque_device));
    dx.dxgi_adapter = selected_adapter.?;

    dx.command_queue = try dx.device.createCommandQueue(.direct, .high);

    dx.render_target_descriptor_heap = try DescriptorHeap.init(&dx, c.D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 100);
    dx.depth_stencil_descriptor_heap = try DescriptorHeap.init(&dx, c.D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 100);
    dx.staging_descriptor_heap = try DescriptorHeap.init(&dx, c.D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 10000);

    dx.flush_fence = try dx.device.createFence();
    dx.flush_fence_value = 100;

    dx.serialize_root_signature_func = dx.d3d_lib.lookup(SerializeRootSigFuncPtr, "D3D12SerializeRootSignature") orelse return Error.LookupRootSignatureFunctionFailed;

    //TODO???
    //  r = device->QueryInterface(__uuidof(ID3D12Device5), (void**)&gdi.Device);
    //  if ((S_OK != r) || (!gdi.Device)) {
    //      assert(!"dx12 not supported (device6)");
    //  }

    //null cbv
    {
        dx.null_descriptor_cbv = dx.staging_descriptor_heap.alloc();
        const desc = c.D3D12_CONSTANT_BUFFER_VIEW_DESC{
            .BufferLocation = 0,
            .SizeInBytes = 0,
        };
        vtbl(dx.device.device).CreateConstantBufferView.?(dx.device.device, &desc, dx.null_descriptor_cbv);
    }

    //null srv
    {
        dx.null_descriptor_srv = dx.staging_descriptor_heap.alloc();
        const desc = c.D3D12_SHADER_RESOURCE_VIEW_DESC{
            .Format = c.DXGI_FORMAT_R32_FLOAT,
            .ViewDimension = c.D3D12_SRV_DIMENSION_BUFFER,
            .Shader4ComponentMapping = c.D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            .unnamed_0 = .{
                .Buffer = .{
                    .FirstElement = 0,
                    .NumElements = 0,
                    .StructureByteStride = 0,
                    .Flags = c.D3D12_BUFFER_SRV_FLAG_NONE,
                },
            },
        };
        vtbl(dx.device.device).CreateShaderResourceView.?(dx.device.device, null, &desc, dx.null_descriptor_srv);
    }

    //null uav
    {
        dx.null_descriptor_uav = dx.staging_descriptor_heap.alloc();
        const desc = c.D3D12_UNORDERED_ACCESS_VIEW_DESC{
            .Format = c.DXGI_FORMAT_R32_TYPELESS,
            .ViewDimension = c.D3D12_UAV_DIMENSION_BUFFER,
            .unnamed_0 = .{
                .Buffer = .{
                    .FirstElement = 0,
                    .NumElements = 0,
                    .StructureByteStride = 0,
                    .CounterOffsetInBytes = 0,
                    .Flags = c.D3D12_BUFFER_UAV_FLAG_RAW,
                },
            },
        };
        vtbl(dx.device.device).CreateUnorderedAccessView.?(dx.device.device, null, null, &desc, dx.null_descriptor_uav);
    }

    //null sampler
//  {
//      dx.null_descriptor_sam = dx.samplers.alloc();
//      const desc = c.D3D12_SAMPLER_DESC{
//          .Filter = c.D3D12_FILTER_MIN_MAG_MIP_LINEAR,
//          .AddressU = c.D3D12_TEXTURE_ADDRESS_MODE_WRAP,
//          .AddressV = c.D3D12_TEXTURE_ADDRESS_MODE_WRAP,
//          .AddressW = c.D3D12_TEXTURE_ADDRESS_MODE_WRAP,
//          .MipLODBias = 0.0,
//          .MaxAnisotropy = 16,
//          .ComparisonFunc = c.D3D12_COMPARISON_FUNC_ALWAYS,
//          .BorderColor = [_]c.FLOAT{ 0.0, 0.0, 0.0, 0.0 },
//          .MinLOD = 0.0,
//          .MaxLOD = 1000.0,
//      };
//      vtbl(dx.device.device).CreateSampler.?(dx.device.device, &desc, dx.null_descriptor_sam);
//  }

    pixSetEnableHUD(false);

    if (tracy_enabled) {
        dx.tracy_queue_ctx = c.tracyCreateD3D12Context(dx.device.device, dx.command_queue.cq, "Direct Queue", "Direct Queue".len).?;
    }

    return dx;
}

pub fn deinit(self: *D3D) void {
    self.d3d_lib.close();
    self.dxgi_lib.close();
    self.dxcompiler_lib.close();
}

pub fn serializeRootSignature(self: *D3D, desc: c.D3D12_ROOT_SIGNATURE_DESC) !*c.ID3DBlob {
    var blob: ?*c.ID3DBlob = null;
    try verify(self.serialize_root_signature_func(&desc, c.D3D_ROOT_SIGNATURE_VERSION_1, &blob, null));
    return blob.?;
}

pub fn shaderBlobToByteCode(shader_blob: ?*c.ID3DBlob) c.D3D12_SHADER_BYTECODE {
    if (shader_blob) |blob| {
        return .{
            .pShaderBytecode = vtbl(blob).GetBufferPointer.?(blob),
            .BytecodeLength = vtbl(blob).GetBufferSize.?(blob),
        };
    } else {
        return .{
            .pShaderBytecode = null,
            .BytecodeLength = 0,
        };
    }
}

pub const CommandListKind = enum { direct, compute, copy };

pub fn commandListKindToD3D(kind: CommandListKind) d3d.D3D12_COMMAND_LIST_TYPE {
    return switch (kind) {
        .direct => c.D3D12_COMMAND_LIST_TYPE_DIRECT,
        .compute => c.D3D12_COMMAND_LIST_TYPE_COMPUTE,
        .copy => c.D3D12_COMMAND_LIST_TYPE_COPY,
    };
}

pub const Device = struct {
    device: *c.ID3D12Device,

    pub fn init(device: *c.ID3D12Device) Device {
        return Device{ .device = device };
    }

    pub fn createFence(self: *Device) !Fence {
        var opaque_fence: ?*anyopaque = null;
        try verify(vtbl(self.device).CreateFence.?(self.device, 0, c.D3D12_FENCE_FLAG_NONE, &IID_ID3D12Fence, &opaque_fence));
        const event_flags = (std.os.windows.CREATE_EVENT_INITIAL_SET | std.os.windows.CREATE_EVENT_MANUAL_RESET);
        return Fence{
            .fence = d3dPtrCast(c.ID3D12Fence, opaque_fence.?),
            .event = try std.os.windows.CreateEventEx(null, "Fence Event", event_flags, std.os.windows.EVENT_ALL_ACCESS),
        };
    }

    pub fn createCommandSignature(self: *Device, desc: c.D3D12_COMMAND_SIGNATURE_DESC) !CommandSignature {
        var opaque_ptr: ?*anyopaque = null;
        try verifyUsingError(vtbl(self.device).CreateCommandSignature.?(self.device, &desc, null, &IID_ICommandSignature, &opaque_ptr), Error.CreateCommandSignatureFailed);
        return CommandSignature{
            .sig = d3dPtrCast(c.ID3D12CommandSignature, opaque_ptr),
        };
    }

    pub fn createCommandQueue(self: *Device, kind: CommandListKind, priority: enum { normal, high }) !CommandQueue {
        const desc = c.D3D12_COMMAND_QUEUE_DESC{
            .Type = commandListKindToD3D(kind),
            .Priority = if (priority == .high) c.D3D12_COMMAND_QUEUE_PRIORITY_HIGH else c.D3D12_COMMAND_QUEUE_PRIORITY_NORMAL,
            .Flags = c.D3D12_COMMAND_QUEUE_FLAG_NONE,
            .NodeMask = 0,
        };
        var opaque_ptr: ?*anyopaque = null;
        try verify(vtbl(self.device).CreateCommandQueue.?(self.device, &desc, &IID_ICommandQueue, &opaque_ptr));
        return CommandQueue{
            .cq = d3dPtrCast(c.ID3D12CommandQueue, opaque_ptr),
        };
    }
    
    pub fn createCommandAllocator(self: *Device, kind: CommandListKind) !CommandAllocator {
        const d3d_kind = commandListKindToD3D(kind);
        var opaque_ptr: ?*anyopaque = null;
        try verifyUsingError(vtbl(self.device).CreateCommandAllocator.?(self.device, d3d_kind, &IID_ICommandAllocator, &opaque_ptr), Error.CreateCommandAllocatorFailed);
        try verifyUsingError(vtbl(self.device).CreateCommandAllocator.?(self.device, d3d_kind, &IID_ICommandAllocator, &opaque_ptr), Error.CreateCommandAllocatorFailed);
        return CommandAllocator{
            .ca = d3dPtrCast(c.ID3D12CommandAllocator, opaque_ptr),
        };
    }

    pub fn createCommandList(self: *Device, kind: CommandListKind, command_allocator: CommandAllocator, pso: ?PipelineState) !GraphicsCommandList {
        var opaque_ptr: ?*anyopaque = null;
        try verifyUsingError(vtbl(self.device).CreateCommandList.?(self.device, 0, commandListKindToD3D(kind), command_allocator.ca, (if (pso != null) pso.?.pso else null), &IID_IGraphicsCommandList, &opaque_ptr), Error.CreateCommandListFailed);
        return GraphicsCommandList{
            .cl = d3dPtrCast(c.ID3D12GraphicsCommandList, opaque_ptr),
        };
    }

    pub fn createCommittedResource(self: *Device, comptime name: [:0]const u8, heap_info: *const c.D3D12_HEAP_PROPERTIES, desc: *const c.D3D12_RESOURCE_DESC, states: c.D3D12_RESOURCE_STATES) !Resource {
        const flags = 0x1000; //c.D3D12_HEAP_FLAG_CREATE_NOT_ZEROED
        var opaque_resource: ?*anyopaque = null;
        try verifyUsingError(vtbl(self.device).CreateCommittedResource.?(self.device, heap_info, flags, desc, states, null, &IID_ID3D12Resource, &opaque_resource), Error.CreateCommitedResourceFailed);
        const r = d3dPtrCast(c.ID3D12Resource, opaque_resource);
        setResourceName(r, name);
        return Resource{
            .resource = r,
        };
    }

    pub fn createCommittedResourceWithClearValue(self: *Device, heap_info: *const c.D3D12_HEAP_PROPERTIES, desc: *const c.D3D12_RESOURCE_DESC, clear_value: *const c.D3D12_CLEAR_VALUE, states: c.D3D12_RESOURCE_STATES) !Resource {
        const flags = 0x1000; //c.D3D12_HEAP_FLAG_CREATE_NOT_ZEROED
        var opaque_resource: ?*anyopaque = null;
        try verifyUsingError(vtbl(self.device).CreateCommittedResource.?(self.device, heap_info, flags, desc, states, clear_value, &IID_ID3D12Resource, &opaque_resource), Error.CreateCommitedResourceFailed);
        return Resource{
            .resource = d3dPtrCast(c.ID3D12Resource, opaque_resource),
        };
    }

    pub fn createRootSignature(self: *Device, blob: ?*c.ID3DBlob) !RootSignature {
        var opaque_ptr: ?*anyopaque = null;
        const size = vtbl(blob.?).GetBufferSize.?(blob);
        const blob_stuff = vtbl(blob.?).GetBufferPointer.?(blob);
        //std.debug.print("createRootSignature(blob.size = {}, blob.stuff = '{s}')\n", .{size, @ptrCast([*]u8, blob_stuff)[0..size]});
        try verifyUsingError(vtbl(self.device).CreateRootSignature.?(self.device, 0, blob_stuff, size, &D3D.IID_ID3D12RootSignature, &opaque_ptr), Error.CreateRootSignatureFailed);
        return RootSignature{
            .root_signature = d3dPtrCast(c.ID3D12RootSignature, opaque_ptr),
        };
    }

    pub fn createPipelineState(self: *Device, desc: *const D3D12_PIPELINE_STATE_STREAM_DESC) !PipelineState {
        var opaque_ptr: ?*anyopaque = null;
        const vtbl2: *ID3D12Device2Vtbl = @ptrCast(self.device.*.lpVtbl);
        try verifyUsingError(vtbl2.CreatePipelineState.?(self.device, desc, &IID_ID3D12PipelineState, &opaque_ptr), Error.CreatePipelineStateObjectFailed);
        return PipelineState{
            .pso = d3dPtrCast(c.ID3D12PipelineState, opaque_ptr),
        };
    }

    pub fn createGraphicsPipelineState(self: *Device, desc: *const c.D3D12_GRAPHICS_PIPELINE_STATE_DESC) !PipelineState {
        var opaque_ptr: ?*anyopaque = null;
        try verifyUsingError(vtbl(self.device).CreateGraphicsPipelineState.?(self.device, desc, &IID_ID3D12PipelineState, &opaque_ptr), Error.CreatePipelineStateObjectFailed);
        return PipelineState{
            .pso = d3dPtrCast(c.ID3D12PipelineState, opaque_ptr),
        };
    }

    pub fn createComputePipelineState(self: *Device, desc: *const c.D3D12_COMPUTE_PIPELINE_STATE_DESC) !PipelineState {
        var opaque_ptr: ?*anyopaque = null;
        try verifyUsingError(vtbl(self.device).CreateComputePipelineState.?(self.device, desc, &IID_ID3D12PipelineState, &opaque_ptr), Error.CreatePipelineStateObjectFailed);
        return PipelineState{
            .pso = d3dPtrCast(c.ID3D12PipelineState, opaque_ptr),
        };
    }

    pub fn createRenderTargetView(self: *Device, resource: Resource, desc: ?*const d3d.D3D12_RENDER_TARGET_VIEW_DESC, view: CpuDescriptorHandle) void {
        vtbl(self.device).CreateRenderTargetView.?(self.device, resource.resource, desc, view);
    }

    pub fn createDepthStencilView(self: *Device, resource: Resource, desc: d3d.D3D12_DEPTH_STENCIL_VIEW_DESC, view: CpuDescriptorHandle) void {
        vtbl(self.device).CreateDepthStencilView.?(self.device, resource.resource, &desc, view);
    }

    pub fn createConstantBufferView(self: *Device, desc: c.D3D12_CONSTANT_BUFFER_VIEW_DESC, view: CpuDescriptorHandle) void {
        vtbl(self.device).CreateConstantBufferView.?(self.device, &desc, view);
    }

    pub fn createShaderResourceView(self: *Device, resource: Resource, desc: c.D3D12_SHADER_RESOURCE_VIEW_DESC, view: CpuDescriptorHandle) void {
        vtbl(self.device).CreateShaderResourceView.?(self.device, resource.resource, &desc, view);
    }

    pub fn createUnorderedAccessView(self: *Device, resource: Resource, counter: ?Resource, desc: c.D3D12_UNORDERED_ACCESS_VIEW_DESC, view: CpuDescriptorHandle) void {
        vtbl(self.device).CreateUnorderedAccessView.?(self.device, resource.resource, (if (counter) |ctr| ctr.resource else null), &desc, view);
    }

    pub fn copyDescriptorRangeSimple(self: *Device, len: usize, dst: CpuDescriptorHandle, src: CpuDescriptorHandle, desc_type: c.D3D12_DESCRIPTOR_HEAP_TYPE) void {
        vtbl(self.device).CopyDescriptorsSimple.?(self.device, len, dst, src, desc_type);
    }

    pub fn copyDescriptorSimple(self: *Device, dst: CpuDescriptorHandle, src: CpuDescriptorHandle, desc_type: c.D3D12_DESCRIPTOR_HEAP_TYPE) void {
        vtbl(self.device).CopyDescriptorsSimple.?(self.device, 1, dst, src, desc_type);
    }

    pub fn checkFeatureSupport(self: *Device) SupportedFeatures {
        var features = SupportedFeatures{};

        //crap crashes
        //var levels: c.D3D12_FEATURE_DATA_FEATURE_LEVELS = undefined;
        //if (c.S_OK == vtbl(self.device).CheckFeatureSupport.?(self.device, c.D3D12_FEATURE_FEATURE_LEVELS, &levels, @sizeOf(c.D3D12_FEATURE_DATA_FEATURE_LEVELS))) {
        //    features.max_feature_level = levels.MaxSupportedFeatureLevel;
        //}

        var shader_model: c.D3D12_FEATURE_DATA_SHADER_MODEL = undefined;
        if (c.S_OK == vtbl(self.device).CheckFeatureSupport.?(self.device, c.D3D12_FEATURE_SHADER_MODEL, &shader_model, @sizeOf(c.D3D12_FEATURE_DATA_SHADER_MODEL))) {
            features.highest_shader_model = shader_model.HighestShaderModel;
        } else {
            //std.debug.print("CheckFeatureSupport(shader model) failed!\n", .{});
            features.highest_shader_model = 0x51;
        }

        var options: c.D3D12_FEATURE_DATA_D3D12_OPTIONS = undefined;
        if (c.S_OK == vtbl(self.device).CheckFeatureSupport.?(self.device, c.D3D12_FEATURE_D3D12_OPTIONS, &options, @sizeOf(c.D3D12_FEATURE_DATA_D3D12_OPTIONS))) {
            if (0 != (options.MinPrecisionSupport & c.D3D12_SHADER_MIN_PRECISION_SUPPORT_10_BIT)) {
                features.has_10bit_support = true;
            }
            if (0 != (options.MinPrecisionSupport & c.D3D12_SHADER_MIN_PRECISION_SUPPORT_16_BIT)) {
                features.has_16bit_support = true;
            }
            features.resource_binding_tier = @intCast(options.ResourceBindingTier);
            features.resource_heap_tier = @intCast(options.ResourceHeapTier);
        }

        var options1: c.D3D12_FEATURE_DATA_D3D12_OPTIONS1 = undefined;
        if (c.S_OK == vtbl(self.device).CheckFeatureSupport.?(self.device, c.D3D12_FEATURE_D3D12_OPTIONS1, &options1, @sizeOf(c.D3D12_FEATURE_DATA_D3D12_OPTIONS1))) {
            if (options1.WaveOps == c.TRUE) {
                features.has_wave_ops = true;
            }
            if (options1.Int64ShaderOps == c.TRUE) {
                features.has_int64_shader_ops = true;
            }
            features.wave_lane_count_min = @intCast(options1.WaveLaneCountMin);
            features.wave_lane_count_max = @intCast(options1.WaveLaneCountMax);
            features.total_lane_count = @intCast(options1.TotalLaneCount);
        }

        var options4: c.D3D12_FEATURE_DATA_D3D12_OPTIONS4 = undefined;
        if (c.S_OK == vtbl(self.device).CheckFeatureSupport.?(self.device, c.D3D12_FEATURE_D3D12_OPTIONS4, &options4, @sizeOf(c.D3D12_FEATURE_DATA_D3D12_OPTIONS4))) {
            if (options4.Native16BitShaderOpsSupported == c.TRUE) {
                features.has_native_16bit_shader_ops = true;
            }
        }

        var options5: c.D3D12_FEATURE_DATA_D3D12_OPTIONS5 = undefined;
        if (c.S_OK == vtbl(self.device).CheckFeatureSupport.?(self.device, c.D3D12_FEATURE_D3D12_OPTIONS5, &options5, @sizeOf(c.D3D12_FEATURE_DATA_D3D12_OPTIONS5))) {
            if (options5.RaytracingTier >= c.D3D12_RAYTRACING_TIER_1_0) {
                features.has_raytracing = true;
            }
        }

        var options7: D3D12_FEATURE_DATA_D3D12_OPTIONS7 = undefined;
        if (c.S_OK == vtbl(self.device).CheckFeatureSupport.?(self.device, c.D3D12_FEATURE_D3D12_OPTIONS7, &options7, @sizeOf(D3D12_FEATURE_DATA_D3D12_OPTIONS7))) {
            if (options7.MeshShaderTier >= D3D12_MESH_SHADER_TIER_1) {
                features.has_mesh_shaders = true;
            }
            if (options7.SamplerFeedbackTier >= D3D12_SAMPLER_FEEDBACK_TIER_0_9) {
                features.has_sampler_feedback = true;
            }
        }

        var options9: D3D12_FEATURE_DATA_D3D12_OPTIONS9 = undefined;
        if (c.S_OK == vtbl(self.device).CheckFeatureSupport.?(self.device, D3D12_FEATURE_D3D12_OPTIONS9, &options9, @sizeOf(D3D12_FEATURE_DATA_D3D12_OPTIONS9))) {
            if (options9.MeshShaderPipelineStatsSupported == c.TRUE) {
                features.has_mesh_shader_stats = true;
            }
            if (options9.AtomicInt64OnTypedResourceSupported == c.TRUE) {
                features.has_atomic_int64_typed_resources = true;
            }
            if (options9.AtomicInt64OnGroupSharedSupported == c.TRUE) {
                features.has_atomic_int64_group_shared = true;
            }
            if (options9.DerivativesInMeshAndAmplificationShadersSupported == c.TRUE) {
                features.has_derivatives_in_mesh_shaders = true;
            }
        }

        var options11: D3D12_FEATURE_DATA_D3D12_OPTIONS11 = undefined;
        if (c.S_OK == vtbl(self.device).CheckFeatureSupport.?(self.device, D3D12_FEATURE_D3D12_OPTIONS11, &options11, @sizeOf(D3D12_FEATURE_DATA_D3D12_OPTIONS11))) {
            if (options11.AtomicInt64OnDescriptorHeapResourceSupported == c.TRUE) {
                features.has_atomic_int64_on_descheap_resource = true;
            }
        }

        var options12: D3D12_FEATURE_DATA_D3D12_OPTIONS12 = undefined;
        if (c.S_OK == vtbl(self.device).CheckFeatureSupport.?(self.device, D3D12_FEATURE_D3D12_OPTIONS12, &options12, @sizeOf(D3D12_FEATURE_DATA_D3D12_OPTIONS12))) {
            if (options12.EnhancedBarriersSupported == c.TRUE) {
                features.has_enhanced_barriers = true;
            }
        }

        return features;
    }
};

pub const SupportedFeatures = struct {
    max_feature_level: u32 = 0,
    highest_shader_model: u32 = 0,
    has_10bit_support: bool = false,
    has_16bit_support: bool = false,
    resource_binding_tier: u8 = 0,
    resource_heap_tier: u8 = 0,
    has_wave_ops: bool = false,
    has_int64_shader_ops: bool = false,
    wave_lane_count_min: u32 = 0,
    wave_lane_count_max: u32 = 0,
    total_lane_count: u32 = 0,
    has_native_16bit_shader_ops: bool = false,
    has_raytracing: bool = false,
    has_mesh_shaders: bool = false,
    has_sampler_feedback: bool = false,
    has_mesh_shader_stats: bool = false,
    has_atomic_int64_typed_resources: bool = false,
    has_atomic_int64_group_shared: bool = false,
    has_derivatives_in_mesh_shaders: bool = false,
    has_atomic_int64_on_descheap_resource: bool = false,
    has_enhanced_barriers: bool = false,
};

pub const ResourceStates = c.D3D12_RESOURCE_STATES;

pub const Resource = struct {
    resource: *c.ID3D12Resource,

    pub fn deinit(self: *Resource) void {
        releaseComPtr(self.resource);
    }

    pub fn setName(self: Resource, comptime name: [:0]const u8) void {
        setResourceName(self.resource, name);
    }

    pub fn mapEntireBuffer(self: Resource) ![*]u8 {
        var opaque_pinned_memory: ?*anyopaque = null;
        try D3D.verify(vtbl(self.resource).Map.?(self.resource, 0, null, &opaque_pinned_memory));
        return @ptrCast(opaque_pinned_memory.?);
    }

    pub fn unmapEntireBuffer(self: Resource) void {
        vtbl(self.resource).Unmap.?(self.resource, 0, null);
    }

    pub fn getGpuVirtualAddress(self: Resource) c.D3D12_GPU_VIRTUAL_ADDRESS {
        return vtbl(self.resource).GetGPUVirtualAddress.?(self.resource);
    }
};

pub const PipelineState = struct {
    pso: *c.ID3D12PipelineState,

    pub fn deinit(self: *PipelineState) void {
        releaseComPtr(self.pso);
    }

    pub fn setName(self: *const PipelineState, comptime name: [:0]const u8) void {
        setResourceName(self.pso, name);
    }
};

pub extern "kernel32" fn ResetEvent(event: std.os.windows.HANDLE) callconv(std.os.windows.WINAPI) std.os.windows.BOOL;

pub const Fence = struct {
    fence: *c.ID3D12Fence,
    event: std.os.windows.HANDLE,

    pub fn deinit(self: *Fence) void {
        releaseComPtr(self.fence);
        _ = std.os.windows.CloseHandle(self.event);
    }

    pub fn signalOnCpu(self: *Fence, value: u64) !void {
        try verify(vtbl(self.fence).Signal.?(self.fence, value));
    }

    pub fn setEventOnCompletion(self: *Fence, value: u64) !void {
        _ = ResetEvent(self.event);
        try verify(vtbl(self.fence).SetEventOnCompletion.?(self.fence, value, self.event));
    }

    pub fn setEventOnCompletionAndWait(self: *Fence, value: u64) !u64 {
        try self.setEventOnCompletion(value);
        _ = std.os.windows.kernel32.WaitForSingleObject(self.event, std.os.windows.INFINITE);
        return self.getCompletedValue();
    }

    pub fn isEventComplete(self: *Fence) !bool {
        switch (std.os.windows.kernel32.WaitForSingleObject(self.event, 0)) {
            std.os.windows.WAIT_ABANDONED => return std.os.windows.WaitForSingleObjectError.WaitAbandoned,
            std.os.windows.WAIT_OBJECT_0 => return true,
            std.os.windows.WAIT_TIMEOUT => return false,
            std.os.windows.WAIT_FAILED => return std.os.windows.WaitForSingleObjectError.Unexpected,
            else => return std.os.windows.WaitForSingleObjectError.Unexpected,
        }
    }

    pub fn getCompletedValue(self: *Fence) u64 {
        return vtbl(self.fence).GetCompletedValue.?(self.fence);
    }
};

pub const CommandQueue = struct {
    cq: *c.ID3D12CommandQueue,

    pub fn deinit(self: *CommandQueue) void {
        releaseComPtr(self.cq);
    }

    pub fn executeCommandLists(self: CommandQueue, lists: []GraphicsCommandList) void {
        var array = std.BoundedArray([*c]c.ID3D12CommandList, 16).init(0);
        for (lists) |*list| {
            if (list.cl) |cl| {
                try array.append(cl);
            }
        }
        vtbl(self.cq).ExecuteCommandLists.?(self.cq, lists.len, lists.ptr);
    }

    pub fn executeCommandList(self: CommandQueue, list: GraphicsCommandList) void {
        var clc: [*c]c.ID3D12CommandList = @ptrCast(list.cl);
        vtbl(self.cq).ExecuteCommandLists.?(self.cq, 1, &clc);
    }

    pub fn signal(self: CommandQueue, fence: Fence, value: u64) !void {
        try verify(vtbl(self.cq).Signal.?(self.cq, fence.fence, value));
    }

    pub fn wait(self: CommandQueue, fence: Fence, value: u64) !void {
        try verify(vtbl(self.cq).Wait.?(self.cq, fence.fence, value));
    }

    pub inline fn setMarker(self: CommandQueue, category: u8, name: [*:0]const u8) void {
        pixSetMarkerCQInternal(self.cq, category, name);
    }

    pub inline fn beginEvent(self: CommandQueue, category: u8, name: [*:0]const u8) void {
        pixBeginEventCQInternal(self.cq, category, name);
    }

    pub inline fn endEvent(self: CommandQueue) void {
        pixEndEventCQInternal(self.cq);
    }
};

pub const CommandSignature = struct {
    sig: *c.ID3D12CommandSignature,

    pub fn deinit(self: *CommandSignature) void {
        releaseComPtr(self.sig);
    }
};

pub const CommandAllocator = struct {
    ca: *c.ID3D12CommandAllocator,

    pub fn deinit(self: *CommandAllocator) void {
        releaseComPtr(self.ca);
    }

    pub fn reset(self: *CommandAllocator) !void {
        try verifyUsingError(vtbl(self.ca).Reset.?(self.ca), Error.ResetCommandAllocatorFailed);
    }
};

pub const GraphicsCommandList = struct {
    cl: *c.ID3D12GraphicsCommandList,

    pub fn deinit(self: *GraphicsCommandList) void {
        releaseComPtr(self.cl);
    }

    pub fn reset(self: GraphicsCommandList, ca: CommandAllocator, pso: ?PipelineState) !void {
        try verify(vtbl(self.cl).Reset.?(self.cl, ca.ca, if (pso) |p| p.pso else null));
    }

    pub fn close(self: GraphicsCommandList) !void {
        try verify(vtbl(self.cl).Close.?(self.cl));
    }

    pub fn resourceBarriers(self: GraphicsCommandList, barriers: []c.D3D12_RESOURCE_BARRIER) void {
        vtbl(self.cl).ResourceBarrier.?(self.cl, @intCast(barriers.len), barriers.ptr);
    }
    pub fn resourceBarrier(self: GraphicsCommandList, barrier: c.D3D12_RESOURCE_BARRIER) void {
        vtbl(self.cl).ResourceBarrier.?(self.cl, 1, &barrier);
    }

    //usually you have more than one and should use a ResourceBarrierBatcher
    pub fn transitionResourceIfNecessary(self: GraphicsCommandList, resource: anytype, new_states: c.D3D12_RESOURCE_STATES) void {
        return switch (@TypeOf(resource)) {
            *RenderTarget => {
                if (resource.states != new_states) {
                    self.resourceBarrier(c.D3D12_RESOURCE_BARRIER{
                        .Type = c.D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                        .Flags = c.D3D12_RESOURCE_BARRIER_FLAG_NONE,
                        .unnamed_0 = .{
                            .Transition = .{
                                .pResource = resource.resource.resource,
                                .Subresource = 0,
                                .StateBefore = resource.states,
                                .StateAfter = new_states,
                            },
                        },
                    });
                    resource.states = new_states;
                }
            },
            *DepthStencil => {
                if (resource.states != new_states) {
                    self.resourceBarrier(c.D3D12_RESOURCE_BARRIER{
                        .Type = c.D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                        .Flags = c.D3D12_RESOURCE_BARRIER_FLAG_NONE,
                        .unnamed_0 = .{
                            .Transition = .{
                                .pResource = resource.resource.resource,
                                .Subresource = 0,
                                .StateBefore = resource.states,
                                .StateAfter = new_states,
                            },
                        },
                    });
                    resource.states = new_states;
                }
            },
            else => @compileError("transitionResourceIfNecessary: unknown resource type " ++ @typeName(@TypeOf(resource))),
        };
    }

    pub fn setDescriptorHeaps(self: GraphicsCommandList, heap_cbv_srv_uav: *const GpuDescriptorHeap, heap_sampler: ?*const GpuDescriptorHeap) !void {
        if (heap_sampler != null) {
            const array = [2]*c.ID3D12DescriptorHeap{
                heap_cbv_srv_uav.heap.heap,
                heap_sampler.?.heap.heap,
            };
            vtbl(self.cl).SetDescriptorHeaps.?(self.cl, 2, &array[0]);
        } else {
            vtbl(self.cl).SetDescriptorHeaps.?(self.cl, 1, &(heap_cbv_srv_uav.heap.heap));
        }
    }

    pub fn setViewports(self: GraphicsCommandList, vp: []c.D3D12_VIEWPORT) void {
        vtbl(self.cl).RSSetViewports.?(self.cl, vp.len, vp.ptr);
    }

    pub fn setViewport(self: GraphicsCommandList, vp: c.D3D12_VIEWPORT) void {
        vtbl(self.cl).RSSetViewports.?(self.cl, 1, &vp);
    }

    pub fn clearRenderTarget(self: GraphicsCommandList, rt: RenderTarget, color: [4]f32) void {
        std.debug.assert(rt.view.ptr != 0);
        vtbl(self.cl).ClearRenderTargetView.?(self.cl, rt.view, &color[0], 0, null);
    }

    pub fn clearRenderTargetView(self: GraphicsCommandList, view: CpuDescriptorHandle, color: [4]f32) void {
        std.debug.assert(view.ptr != 0);
        vtbl(self.cl).ClearRenderTargetView.?(self.cl, view, &color[0], 0, null);
    }

    pub fn clearDepthStencilView(self: GraphicsCommandList, view: CpuDescriptorHandle, depth: ?f32, stencil: ?u8) void {
        std.debug.assert(view.ptr != 0);
        const d = if (depth) |dval| dval else 0;
        const s = if (stencil) |sval| sval else 0;
        var flags: d3d.UINT = 0;
        if (depth != null) {
            flags |= d3d.D3D12_CLEAR_FLAG_DEPTH;
        }
        if (stencil != null) {
            flags |= d3d.D3D12_CLEAR_FLAG_STENCIL;
        }
        vtbl(self.cl).ClearDepthStencilView.?(self.cl, view, flags, d, s, 0, null);
    }

    pub fn setViewportAndScissor(self: GraphicsCommandList, x: u32, y: u32, size_x: u32, size_y: u32, min_depth: f32, max_depth: f32) void {
        const viewport = d3d.D3D12_VIEWPORT{
            .TopLeftX = @floatFromInt(x),
            .TopLeftY = @floatFromInt(y),
            .Width    = @floatFromInt(size_x),
            .Height   = @floatFromInt(size_y),
            .MinDepth = min_depth,
            .MaxDepth = max_depth,
        };
        const scissor = d3d.RECT{
            .left   = @intCast(x),
            .top    = @intCast(y),
            .right  = @intCast(x+size_x),
            .bottom = @intCast(y+size_y),
        };
        vtbl(self.cl).RSSetViewports.?(self.cl, 1, &viewport);
        vtbl(self.cl).RSSetScissorRects.?(self.cl, 1, &scissor);
    }

    pub fn setRenderTargets(self: GraphicsCommandList, rts: []const RenderTarget, ds: ?DepthStencil) void {
        var rt_views: [16]CpuDescriptorHandle = undefined;
        std.debug.assert(rts.len <= rt_views.len);
        var i: usize = 0;
        while (i < rts.len) : (i += 1) {
            std.debug.assert(0 != (rts[i].states & c.D3D12_RESOURCE_STATE_RENDER_TARGET));
            std.debug.assert( (rts[i].desc.size_x > 0) and (rts[i].desc.size_y > 0) );
            std.debug.assert(rts[i].view.ptr != 0);
            rt_views[i] = rts[i].view;
        }
        var ds_view: ?*const CpuDescriptorHandle = null;
        if (ds) |ptr| {
            std.debug.assert(0 != (ptr.states & (c.D3D12_RESOURCE_STATE_DEPTH_WRITE | c.D3D12_RESOURCE_STATE_DEPTH_READ)));
            std.debug.assert( (ptr.desc.size_x > 0) and (ptr.desc.size_y > 0) );
            std.debug.assert(ptr.view.ptr != 0);
            ds_view = &ptr.view;
        }
        vtbl(self.cl).OMSetRenderTargets.?(self.cl, @intCast(rts.len), &rt_views[0], c.FALSE, ds_view);
    }

    pub inline fn setRenderTarget(self: GraphicsCommandList, rt: RenderTarget, ds: ?DepthStencil) void {
        const array = [1]RenderTarget{ rt };
        self.setRenderTargets(array[0..1], ds);
    }

    pub fn unsetRenderTargets(self: GraphicsCommandList) void {
        vtbl(self.cl).OMSetRenderTargets.?(self.cl, 0, null, c.FALSE, null);
    }

    pub fn setRenderTargetViews(self: GraphicsCommandList, rts: []const CpuDescriptorHandle, ds: ?CpuDescriptorHandle) void {
        vtbl(self.cl).OMSetRenderTargets.?(self.cl, @intCast(rts.len), &rts[0], c.FALSE, (if (ds) |x| &x else null));
    }

    pub fn setRenderTargetViewsDepthOnly(self: GraphicsCommandList, ds: CpuDescriptorHandle) void {
        vtbl(self.cl).OMSetRenderTargets.?(self.cl, 0, null, c.FALSE, &ds);
    }

    pub fn setPrimTopology(self: GraphicsCommandList, prim_topology: d3d.D3D12_PRIMITIVE_TOPOLOGY) void {
        vtbl(self.cl).IASetPrimitiveTopology.?(self.cl, prim_topology);
    }

    pub fn setIndexBuffer(self: GraphicsCommandList, buf: IndexBuffer) void {
        vtbl(self.cl).IASetIndexBuffer.?(self.cl, buf.view);
    }

    pub fn setIndexBufferView(self: GraphicsCommandList, view: *const d3d.D3D12_INDEX_BUFFER_VIEW) void {
        vtbl(self.cl).IASetIndexBuffer.?(self.cl, view);
    }

    pub fn setVertexBuffers(self: GraphicsCommandList, start_slot: u32, bufs: []const VertexBuffer) void {
        std.debug.assert(bufs.len < 16);
        var views: [16]c.D3D12_VERTEX_BUFFER_VIEW = undefined;
        var i: usize = 0;
        while (i < bufs.len) : (i += 1) {
            views[i] = bufs[i].view;
        }
        vtbl(self.cl).IASetVertexBuffers.?(self.cl, start_slot, @intCast(bufs.len), views.ptr);
    }

    pub fn setVertexBufferViews(self: GraphicsCommandList, start_slot: u32, views: []*const c.D3D12_VERTEX_BUFFER_VIEW) void {
        vtbl(self.cl).IASetVertexBuffers.?(self.cl, start_slot, @intCast(views.len), views.ptr);
    }
    pub fn setVertexBufferView(self: GraphicsCommandList, start_slot: u32, view: *const c.D3D12_VERTEX_BUFFER_VIEW) void {
        vtbl(self.cl).IASetVertexBuffers.?(self.cl, start_slot, 1, view);
    }

    pub fn setBlendFactor(self: GraphicsCommandList, factor: [4]f32) void {
        vtbl(self.cl).OMSetBlendFactor.?(self.cl, &factor[0]);
    }

    pub fn setPrimitiveTopology(self: GraphicsCommandList, topology: c.D3D12_PRIMITIVE_TOPOLOGY) void {
        vtbl(self.cl).IASetPrimitiveTopology.?(self.cl, topology);
    }

    pub fn setPipelineState(self: GraphicsCommandList, pso: PipelineState) void {
        vtbl(self.cl).SetPipelineState.?(self.cl, pso.pso);
    }

    pub fn setGraphicsRootSignature(self: GraphicsCommandList, root_sig: RootSignature) void {
        vtbl(self.cl).SetGraphicsRootSignature.?(self.cl, root_sig.root_signature);
    }

    pub fn setGraphicsRootDescriptorTable(self: GraphicsCommandList, root_parameter_index: u32, gpu_handle: c.D3D12_GPU_DESCRIPTOR_HANDLE) void {
        vtbl(self.cl).SetGraphicsRootDescriptorTable.?(self.cl, root_parameter_index, gpu_handle);
    }

    pub fn setGraphicsRoot32BitConstant(self: GraphicsCommandList, root_parameter_index: u32, src_data: u32, dest_offset_in_32bit_values: u32) void {
        vtbl(self.cl).SetGraphicsRoot32BitConstants.?(self.cl, root_parameter_index, src_data, dest_offset_in_32bit_values);
    }

    pub fn setGraphicsRoot32BitConstants(self: GraphicsCommandList, root_parameter_index: u32, src_data: []const u32, dest_offset_in_32bit_values: u32) void {
        vtbl(self.cl).SetGraphicsRoot32BitConstants.?(self.cl, root_parameter_index, @intCast(src_data.len), src_data.ptr, dest_offset_in_32bit_values);
    }

    pub fn setGraphicsRootShaderResourceView(self: GraphicsCommandList, root_parameter_index: u32, buffer_loc: c.D3D12_GPU_VIRTUAL_ADDRESS) void {
        vtbl(self.cl).SetGraphicsRootShaderResourceView.?(self.cl, root_parameter_index, buffer_loc);
    }

    pub fn setGraphicsRootUnorderedAccessView(self: GraphicsCommandList, root_parameter_index: u32, buffer_loc: c.D3D12_GPU_VIRTUAL_ADDRESS) void {
        vtbl(self.cl).SetGraphicsRootUnorderedAccessView.?(self.cl, root_parameter_index, buffer_loc);
    }

    pub fn setGraphicsRootConstantBufferView(self: GraphicsCommandList, root_parameter_index: u32, buffer_loc: c.D3D12_GPU_VIRTUAL_ADDRESS) void {
        vtbl(self.cl).SetGraphicsRootConstantBufferView.?(self.cl, root_parameter_index, buffer_loc);
    }

    pub fn setComputeRootSignature(self: GraphicsCommandList, root_sig: RootSignature) void {
        vtbl(self.cl).SetComputeRootSignature.?(self.cl, root_sig.root_signature);
    }

    pub fn setComputeRootDescriptorTable(self: GraphicsCommandList, root_parameter_index: u32, gpu_handle: c.D3D12_GPU_DESCRIPTOR_HANDLE) void {
        vtbl(self.cl).SetComputeRootDescriptorTable.?(self.cl, root_parameter_index, gpu_handle);
    }

    pub fn setComputeRoot32BitConstant(self: GraphicsCommandList, root_parameter_index: u32, src_data: u32, dest_offset_in_32bit_values: u32) void {
        vtbl(self.cl).SetComputeRoot32BitConstants.?(self.cl, root_parameter_index, src_data, dest_offset_in_32bit_values);
    }

    pub fn setComputeRoot32BitConstants(self: GraphicsCommandList, root_parameter_index: u32, src_data: []const u32, dest_offset_in_32bit_values: u32) void {
        vtbl(self.cl).SetComputeRoot32BitConstants.?(self.cl, root_parameter_index, @intCast(src_data.len), src_data.ptr, dest_offset_in_32bit_values);
    }

    pub fn setComputeRootShaderResourceView(self: GraphicsCommandList, root_parameter_index: u32, buffer_loc: c.D3D12_GPU_VIRTUAL_ADDRESS) void {
        vtbl(self.cl).SetComputeRootShaderResourceView.?(self.cl, root_parameter_index, buffer_loc);
    }

    pub fn setComputeRootUnorderedAccessView(self: GraphicsCommandList, root_parameter_index: u32, buffer_loc: c.D3D12_GPU_VIRTUAL_ADDRESS) void {
        vtbl(self.cl).SetComputeRootUnorderedAccessView.?(self.cl, root_parameter_index, buffer_loc);
    }

    pub fn setComputeRootConstantBufferView(self: GraphicsCommandList, root_parameter_index: u32, buffer_loc: c.D3D12_GPU_VIRTUAL_ADDRESS) void {
        vtbl(self.cl).SetComputeRootConstantBufferView.?(self.cl, root_parameter_index, buffer_loc);
    }

    pub fn setScissorRects(self: GraphicsCommandList, rects: []const c.D3D12_RECT) void {
        vtbl(self.cl).RSSetScissorRects.?(self.cl, @intCast(rects.len), rects.ptr);
    }

    pub fn setScissorRect(self: GraphicsCommandList, rect: c.D3D12_RECT) void {
        vtbl(self.cl).RSSetScissorRects.?(self.cl, 1, &rect);
    }

    pub fn drawInstanced(self: GraphicsCommandList, vertex_count_per_instance: u32, instance_count: u32, start_vertex_location: u32, start_instance_location: u32) void {
        vtbl(self.cl).DrawInstanced.?(self.cl, vertex_count_per_instance, instance_count, start_vertex_location, start_instance_location);
    }

    pub fn drawIndexedInstanced(self: GraphicsCommandList, index_count_per_instance: u32, instance_count: u32, start_index_location: u32, base_vertex_location: i32, start_instance_location: u32) void {
        vtbl(self.cl).DrawIndexedInstanced.?(self.cl, index_count_per_instance, instance_count, start_index_location, base_vertex_location, start_instance_location);
    }

    pub fn dispatch(self: GraphicsCommandList, count_x: u32, count_y: u32, count_z: u32) void {
        vtbl(self.cl).Dispatch.?(self.cl, count_x, count_y, count_z);
    }

    pub fn executeIndirect(self: GraphicsCommandList, cmd_sig: CommandSignature, max_cmd_count: u32, argument_buffer: Resource, argument_buffer_offset: u64, count_buffer: ?Resource, count_buffer_offset: u64) void {
        vtbl(self.cl).ExecuteIndirect.?(self.cl, cmd_sig.sig, max_cmd_count, argument_buffer.resource, argument_buffer_offset, if (count_buffer) |buf| buf.resource else null, count_buffer_offset);
    }

    pub fn copyResource(self: GraphicsCommandList, dst: Resource, src: Resource) void {
        vtbl(self.cl).CopyResource.?(self.cl, dst.resource, src.resource);
    }

    pub fn copyBufferRegion(self: GraphicsCommandList, dst: Resource, dst_off: u64, src: Resource, src_off: u64, size: u64) void {
        vtbl(self.cl).CopyBufferRegion.?(self.cl, dst.resource, dst_off, src.resource, src_off, size);
    }

    pub fn copyTextureRegion(self: GraphicsCommandList, dst: d3d.D3D12_TEXTURE_COPY_LOCATION, x: u32, y: u32, z: u32, src: d3d.D3D12_TEXTURE_COPY_LOCATION, src_box: d3d.D3D12_BOX) void {
        std.debug.assert(0 == (src.unnamed_0.PlacedFootprint.Offset % d3d.D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT));
        vtbl(self.cl).CopyTextureRegion.?(self.cl, &dst, x, y, z, &src, &src_box);
    }

    pub fn copyTextureSubresource(self: GraphicsCommandList, dst: d3d.D3D12_TEXTURE_COPY_LOCATION, src: d3d.D3D12_TEXTURE_COPY_LOCATION) void {
        std.debug.assert(0 == (src.unnamed_0.PlacedFootprint.Offset % d3d.D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT));
        vtbl(self.cl).CopyTextureRegion.?(self.cl, &dst, 0, 0, 0, &src, null);
    }

    pub inline fn setMarker(self: GraphicsCommandList, category: u8, name: [*:0]const u8) void {
        pixSetMarkerCLInternal(self.cl, category, name);
    }

    pub inline fn beginEvent(self: GraphicsCommandList, category: u8, name: [*:0]const u8) void {
        pixBeginEventCLInternal(self.cl, category, name);
    }

    pub inline fn endEvent(self: GraphicsCommandList) void {
        pixEndEventCLInternal(self.cl);
    }

    pub fn clearUnorderedAccessViewUint(self: GraphicsCommandList, gpu_view: GpuDescriptorHandle, cpu_view: CpuDescriptorHandle, resource: Resource, values: [4]u32) void {
        vtbl(self.cl).ClearUnorderedAccessViewUint.?(self.cl, gpu_view, cpu_view, resource.resource, &values[0], 0, null);
    }
};

pub const RootSignature = struct {
    root_signature: *c.ID3D12RootSignature,

    pub fn deinit(self: *RootSignature) void {
        releaseComPtr(self.root_signature);
    }
};

// ////////////////////////////////////////////////////////////////////////////////
//below are helpers that are not directly wrappers of the d3d12 api
// ////////////////////////////////////////////////////////////////////////////////

pub const RenderTarget = struct {
    const Desc = struct {
        format: Format,
        size_x: u32,
        size_y: u32,
        allow_unordered_access : bool,
    };

    resource: Resource,
    states: ResourceStates,
    view: CpuDescriptorHandle,
    desc: Desc,
};

pub const DepthStencil = struct {
    const Desc = struct {
        format: Format,
        size_x: u32,
        size_y: u32,
    };

    resource: Resource,
    states: ResourceStates,
    view: CpuDescriptorHandle,
    desc: Desc,
};

pub const IndexBuffer = struct {
    resource: Resource,
    view: c.D3D12_INDEX_BUFFER_VIEW,
};

pub const VertexBuffer = struct {
    resource: Resource,
    view: c.D3D12_VERTEX_BUFFER_VIEW,
};

pub const CpuDescriptorHandle = c.D3D12_CPU_DESCRIPTOR_HANDLE;
pub const GpuDescriptorHandle = c.D3D12_GPU_DESCRIPTOR_HANDLE;

pub const AnyDescriptorHeap = struct {
    heap: *c.ID3D12DescriptorHeap,
    descriptor_type: c.D3D12_DESCRIPTOR_HEAP_TYPE,
    is_shader_visible: bool,
    cpu_handle: CpuDescriptorHandle,
    len: u32,
    descriptor_size: u32,

    fn init(dx: *D3D, descriptor_type: c.D3D12_DESCRIPTOR_HEAP_TYPE, is_shader_visible: bool, len: u32) !AnyDescriptorHeap {
        if (is_shader_visible) {
            switch(descriptor_type) {
                c.D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, c.D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER => {},
                else => { return Error.ShaderVisibleDescriptorHeapsCanOnlyBeCbvSrvUavOrSampler; },
            }
        }

        const desc = c.D3D12_DESCRIPTOR_HEAP_DESC{
            .Type = descriptor_type,
            .NumDescriptors = len,
            .Flags = if (is_shader_visible) c.D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE else 0,
            .NodeMask = 0,
        };
        var opaque_heap: ?*anyopaque = null;
        try verifyUsingError(vtbl(dx.device.device).CreateDescriptorHeap.?(dx.device.device, &desc, &IID_ID3D12DescriptorHeap, &opaque_heap), Error.CreateDescriptorHeapFailed);
        var heap = d3dPtrCast(c.ID3D12DescriptorHeap, opaque_heap);

        var cpu: CpuDescriptorHandle = undefined;
        _ = vtbl(heap).GetCPUDescriptorHandleForHeapStart.?(heap, &cpu);
        std.debug.assert(cpu.ptr != 0);

        return AnyDescriptorHeap{
            .heap = heap,
            .descriptor_type = descriptor_type,
            .is_shader_visible = is_shader_visible,
            .cpu_handle = cpu,
            .len = len,
            .descriptor_size = vtbl(dx.device.device).GetDescriptorHandleIncrementSize.?(dx.device.device, descriptor_type),
        };
    }

    pub fn idxToCpuHandle(self: *const AnyDescriptorHeap, idx: u64) CpuDescriptorHandle {
        std.debug.assert(self.cpu_handle.ptr != 0);
        return .{ .ptr = self.cpu_handle.ptr + (idx * self.descriptor_size) };
    }

    pub fn cpuHandleToIdx(self: *const AnyDescriptorHeap, hdl: CpuDescriptorHandle) u64 {
        return (hdl.ptr - self.cpu_handle.ptr) / self.descriptor_size;
    }
};

pub const GpuDescriptorHeap = struct {
    const FreeList = std.ArrayList(struct { idx:u32, len: u32 });
    heap: AnyDescriptorHeap,
    gpu_handle: GpuDescriptorHandle,
    free_list: FreeList,

    pub fn init(dx: *D3D, descriptor_type: c.D3D12_DESCRIPTOR_HEAP_TYPE, len: u32) !GpuDescriptorHeap {
        if ( (descriptor_type != c.D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) and (descriptor_type != c.D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER) ) {
            return Error.ShaderVisibleDescriptorHeapsCanOnlyBeCbvSrvUavOrSampler;
        }

        var h = GpuDescriptorHeap{
            .heap = try AnyDescriptorHeap.init(dx, descriptor_type, true, len),
            .gpu_handle = .{ .ptr = 0 },
            .free_list = FreeList.init(dx.allocator),
        };
        try h.free_list.append(.{ .idx=0, .len=len });

        //get gpu handle
        _ = vtbl(h.heap.heap).GetGPUDescriptorHandleForHeapStart.?(h.heap.heap, &h.gpu_handle);
        std.debug.assert(h.gpu_handle.ptr != 0);

        return h;
    }

    pub fn allocIdxRange(self: *GpuDescriptorHeap, count: u32) u32 {
        for (self.free_list.items, 0..) |*slot, idx| {
            const loc = slot.idx;
            if (slot.len > count) {
                //std.debug.print("allocIdxRange({}) -> {}\n", .{count, loc});
                //std.debug.print("     shrink case\n", .{});
                //self.dumpFreeList();
                //it fits in this block so shrink it; then return
                slot.idx += count;
                slot.len -= count;
                return loc;
            } else if (slot.len == count) {
                //std.debug.print("allocIdxRange({}) -> {}\n", .{count, loc});
                //std.debug.print("     remove case\n", .{});
                //self.dumpFreeList();
                //entire block so remove it and return the entire thing
                _ = self.free_list.orderedRemove(idx);
                return loc;
            }
        }
        std.debug.assert(false); //we ran out of space
        return 0xffffffff;
    }

    pub fn freeIdxRange(self: *GpuDescriptorHeap, idx: u32, count: u32) !void {
        //std.debug.print("freeIdxRange({}, {})\n", .{idx, count});
        //self.dumpFreeList();
        //find where to insert or merge
        for (self.free_list.items, 0..) |*slot, loc| {
            if ((slot.idx+slot.len) == idx) {
                //std.debug.print("     merge case #1\n", .{});
                //merge case #1
                //it goes on the end of this block so just add it
                slot.len += count;
                return;
            } else if (slot.idx == (idx+count)) {
                //std.debug.print("     merge case #2\n", .{});
                //merge case #2
                //it goes on the beginning of this block so just add it
                slot.idx = idx;
                slot.len += count;
                return;
            } else if (slot.idx > (idx+count)) {
                //std.debug.print("     insert case\n", .{});
                //insert case (in the middle)
                //it goes right here
                try self.free_list.insert(loc, .{ .idx=idx, .len=count });
                return;
            }
        }
        //it goes on the end if it didnt fit anywhere else
        //std.debug.print("     append case\n", .{});
        try self.free_list.append(.{ .idx=idx, .len=count });
    }

    fn dumpFreeList(self: *GpuDescriptorHeap) void {
        for (self.free_list.items, 0..) |slot, idx| {
            log.info("  slot {} -> idx:{} len:{}\n", .{idx, slot.idx, slot.len});
        }
    }

    pub fn freeAll(self: *GpuDescriptorHeap) !void {
        self.alloc_idx = 0;
    }

    pub fn idxToGpuHandle(self: *const GpuDescriptorHeap, idx: u32) GpuDescriptorHandle {
        std.debug.assert((idx >= 0) and (idx < self.heap.len));
        return .{ .ptr = self.gpu_handle.ptr + (idx * self.heap.descriptor_size) };
    }

    pub fn idxToCpuAccessHandle(self: *const GpuDescriptorHeap, idx: u32) CpuDescriptorHandle {
        std.debug.assert((idx >= 0) and (idx < self.heap.len));
        std.debug.assert(self.heap.cpu_handle.ptr != 0);
        return .{ .ptr = self.heap.cpu_handle.ptr + (idx * self.heap.descriptor_size) };
    }

    pub fn gpuHandleToCpuAccessHandle(self: *const GpuDescriptorHeap, cpu: GpuDescriptorHandle) CpuDescriptorHandle {
        std.debug.assert(cpu.ptr >= self.heap.cpu_handle);
        return .{ .ptr = (self.gpu_handle.ptr + (cpu.ptr - self.heap.cpu_handle)) };
    }
};

pub const DescriptorHeap = struct{
    heap: AnyDescriptorHeap,
    is_free: std.DynamicBitSet,
    mutex: std.Thread.Mutex,

    pub fn init(dx: *D3D, descriptor_type: c.D3D12_DESCRIPTOR_HEAP_TYPE, len: u32) !DescriptorHeap {
        return DescriptorHeap{
            .heap = try AnyDescriptorHeap.init(dx, descriptor_type, false, len),
            .is_free = try std.DynamicBitSet.initFull(dx.allocator, len),
            .mutex = .{},
        };
    }

    pub fn allocIdx(self: *DescriptorHeap) u64 {
        self.mutex.lock();
        const found = self.is_free.toggleFirstSet();
        self.mutex.unlock();
        if (found) |idx| {
            return idx;
        } else {
            unreachable; //ran out of space ... maybe handle this error?
        }
    }

    pub fn alloc(self: *DescriptorHeap) CpuDescriptorHandle {
        return self.heap.idxToCpuHandle(self.allocIdx());
    }

    pub fn freeIdx(self: *DescriptorHeap, idx: u64) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        std.debug.assert(!self.is_free.isSet(idx));
        self.is_free.set(idx);
    }

    pub fn free(self: *DescriptorHeap, hdl: CpuDescriptorHandle) void {
        self.freeIdx(self.heap.cpuHandleToIdx(hdl));
    }
};

pub fn ResourceBarrierBatcher(comptime capacity: usize) type {
    return struct {
        const Self = @This();
        barriers: std.BoundedArray(c.D3D12_RESOURCE_BARRIER, capacity) = .{ .buffer=undefined, .len=0 },

        pub fn transitionIfNecessary(self: *Self, cl: GraphicsCommandList, resource: Resource, current_states: c.D3D12_RESOURCE_STATES, required_states: c.D3D12_RESOURCE_STATES) bool {
            if (required_states != (current_states & required_states)) {
                self.flushIfFull(cl);
                var x = self.barriers.addOne() catch unreachable;
                x.* = .{
                    .Type = c.D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                    .Flags = c.D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    .unnamed_0 = .{
                        .Transition = .{
                            .pResource   = resource.resource,
                            .Subresource = c.D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                            .StateBefore = current_states,
                            .StateAfter  = required_states,
                        },
                    },
                };
                return true;
            } else {
                return false;
            }
        }

        pub fn unorderedAccess(self: *Self, cl: GraphicsCommandList, resource: Resource) void {
            self.flushIfFull(cl);
            var x = self.barriers.addOne() catch unreachable;
            x.* = .{
                .Type = c.D3D12_RESOURCE_BARRIER_TYPE_UAV,
                .Flags = c.D3D12_RESOURCE_BARRIER_FLAG_NONE,
                .unnamed_0 = .{
                    .UAV = .{ .pResource = resource.resource, },
                },
            };
        }

        pub fn alias(self: *Self, cl: GraphicsCommandList, resource_before: ?Resource, resource_after: Resource) void {
            self.flushIfFull(cl);
            var x = self.barriers.addOne() catch unreachable;
            x.* = .{
                .Type = c.D3D12_RESOURCE_BARRIER_TYPE_ALIASING,
                .Flags = c.D3D12_RESOURCE_BARRIER_FLAG_NONE,
                .unnamed_0 = .{
                    .Aliasing = .{
                        .pResourceBefore = resource_before.resource,
                        .pResourceAfter  = resource_after.resource,
                    },
                },
            };
        }

        pub fn flush(self: *Self, cl: GraphicsCommandList) void {
            if (self.barriers.len > 0) {
                cl.resourceBarriers(self.barriers.slice());
                self.barriers.resize(0) catch unreachable;
            }
        }

        pub fn flushIfFull(self: *Self, cl: GraphicsCommandList) void {
            if (self.barriers.len == self.barriers.buffer.len) {
                self.flush(cl);
            }
        }
    };
}

pub const BasicTexture = struct {
    pub const Desc = struct {
        format: Format,
        size_x: u32,
        size_y: u32,
        array_count: u32 = 1,
        stride_x: u32,
    };

    desc: Desc,
    resource: Resource,
    resource_stride_x: u32,
    states: c.D3D12_RESOURCE_STATES,
    size_in_bytes: usize,
    array_slice_size_in_bytes: u64,
    texture_view: CpuDescriptorHandle,
    texture_array_view: CpuDescriptorHandle,
    staging_descriptor_heap: *DescriptorHeap,

    //create no mips 2d textures only; designed to be used for UI textures
    //the user will need to free the upload_resource at the correct time (after it isn't in use anymore)
    pub fn init(dx: *D3D, desc: Desc, image_data: []const u8, upload_resource_output: *?Resource) !BasicTexture {
        std.debug.assert(desc.size_x > 0);
        std.debug.assert(desc.size_y > 0);
        std.debug.assert(desc.stride_x > 0);
        std.debug.assert(desc.array_count > 0);

        //create our texture resource
        var resource: ?Resource = null;
        var size_in_bytes: u64 = undefined;
        var array_slice_size_in_bytes: u64 = undefined;
        var row_pitch: usize = 0;
        {
            const d3d_desc = c.D3D12_RESOURCE_DESC{
                .Dimension = c.D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                .Alignment = 0,
                .Width = @intCast(desc.size_x),
                .Height = desc.size_y,
                .DepthOrArraySize = @intCast(desc.array_count),
                .MipLevels = 1,
                .Format = desc.format,
                .SampleDesc = .{ .Count = 1, .Quality = 0 },
                .Layout = c.D3D12_TEXTURE_LAYOUT_UNKNOWN,
                .Flags = c.D3D12_RESOURCE_FLAG_NONE,
            };
            const heap_info = c.D3D12_HEAP_PROPERTIES{
                .Type = c.D3D12_HEAP_TYPE_DEFAULT,
                .CPUPageProperty = c.D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                .MemoryPoolPreference = c.D3D12_MEMORY_POOL_UNKNOWN,
                .CreationNodeMask = 1,
                .VisibleNodeMask = 1,
            };

            resource = try dx.device.createCommittedResource("BasicTexture.<unknown>", &heap_info, &d3d_desc, c.D3D12_RESOURCE_STATE_COPY_DEST);
            std.debug.assert(resource != null);

            //get the resource stride
            var layout: c.D3D12_PLACED_SUBRESOURCE_FOOTPRINT = undefined;
            var num_rows: c.UINT = undefined;
            var row_size_in_bytes: c.UINT64 = undefined;
            vtbl(dx.device.device).GetCopyableFootprints.?(
                dx.device.device,
                &d3d_desc,
                0, //FirstSubresource
                1, //NumSubresources
                0, //BaseOffset
                &layout, //pLayouts
                &num_rows, //pNumRows
                &row_size_in_bytes, //pRowSizeInBytes
                &array_slice_size_in_bytes //pTotalBytes
            );
            std.debug.assert(resource != null);
            std.debug.assert(layout.Footprint.Width == desc.size_x);
            std.debug.assert(layout.Footprint.Height == desc.size_y);
            std.debug.assert(layout.Footprint.Format == desc.format);
            std.debug.assert(size_in_bytes > 0);
            row_pitch = layout.Footprint.RowPitch;
            array_slice_size_in_bytes = std.mem.alignForward(u64, array_slice_size_in_bytes, c.D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);
            size_in_bytes = array_slice_size_in_bytes * desc.array_count;
        }
        std.debug.assert(resource != null);
        //std.debug.print("size_x {}  size_y {}  row_pitch {}  array_slice_size {}\n", .{ desc.size_x, desc.size_y, row_pitch, array_slice_size_in_bytes });

        //create the srvs
        var tex_srv_handle = dx.staging_descriptor_heap.alloc();
        var tex_array_srv_handle = dx.staging_descriptor_heap.alloc();
        {
            const srv_desc = c.D3D12_SHADER_RESOURCE_VIEW_DESC{
                .Format = desc.format,
                .ViewDimension = c.D3D12_SRV_DIMENSION_TEXTURE2D,
                .Shader4ComponentMapping = c.D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                .unnamed_0 = .{
                    .Texture2D = .{
                        .MostDetailedMip = 0,
                        .MipLevels = 1,
                        .PlaneSlice = 0,
                        .ResourceMinLODClamp = 0.0,
                    },
                },
            };
            vtbl(dx.device.device).CreateShaderResourceView.?(dx.device.device, resource.?.resource, &srv_desc, tex_srv_handle);

            const srv_array_desc = c.D3D12_SHADER_RESOURCE_VIEW_DESC{
                .Format = desc.format,
                .ViewDimension = c.D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
                .Shader4ComponentMapping = c.D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                .unnamed_0 = .{
                    .Texture2DArray = .{
                        .MostDetailedMip = 0,
                        .MipLevels = 1,
                        .FirstArraySlice = 0,
                        .ArraySize = desc.array_count,
                        .PlaneSlice = 0,
                        .ResourceMinLODClamp = 0.0,
                    },
                },
            };
            vtbl(dx.device.device).CreateShaderResourceView.?(dx.device.device, resource.?.resource, &srv_array_desc, tex_array_srv_handle);
        }

        //create and copy the image data to an upload resource
        {
            const alignment = @max(c.D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT, c.D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT);
            const upload_resource_desc = c.D3D12_RESOURCE_DESC{
                .Dimension = c.D3D12_RESOURCE_DIMENSION_BUFFER,
                .Alignment = alignment,
                .Width = std.mem.alignForward(u64, size_in_bytes, alignment),
                .Height = 1,
                .DepthOrArraySize = 1,
                .MipLevels = 1,
                .Format = c.DXGI_FORMAT_UNKNOWN,
                .SampleDesc = .{ .Count = 1, .Quality = 0, },
                .Layout = c.D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                .Flags = c.D3D12_RESOURCE_FLAG_NONE,
            };
            const upload_heap_info = c.D3D12_HEAP_PROPERTIES{
                .Type = c.D3D12_HEAP_TYPE_UPLOAD,
                .CPUPageProperty = c.D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                .MemoryPoolPreference = c.D3D12_MEMORY_POOL_UNKNOWN,
                .CreationNodeMask = 1,
                .VisibleNodeMask = 1,
            };
            var upload_resource = try dx.device.createCommittedResource("BasicTexture.upload_resource", &upload_heap_info, &upload_resource_desc, c.D3D12_RESOURCE_STATE_GENERIC_READ);
            upload_resource_output.* = upload_resource;

            pixBeginEvent(200, "copyTextureData");
            var upload_mem = (try upload_resource.mapEntireBuffer())[0..size_in_bytes];
            if (desc.stride_x == row_pitch) {
                var sz = desc.size_y*desc.stride_x*desc.array_count;
                std.debug.assert(sz <= size_in_bytes);
                std.mem.copy(u8, upload_mem[0..sz], image_data[0..sz]);
            } else {
                var array_idx: usize = 0;
                var src_off: usize = 0;
                while (array_idx < desc.array_count) : (array_idx += 1) {
                    var y: u32 = 0;
                    //var src_off: usize = (desc.stride_x * desc.size_y) * array_idx;
                    var dst_off: usize = array_slice_size_in_bytes * array_idx;
                    while (y < desc.size_y) : (y += 1) {
                        const next_src_off = src_off + desc.stride_x;
                        const next_dst_off = dst_off + row_pitch;
                        std.mem.copy(u8, upload_mem[dst_off..(dst_off + desc.stride_x)], image_data[src_off..next_src_off]);
                        @memset(@as([*]u8, @ptrCast(&upload_mem[dst_off + desc.stride_x]))[0..(row_pitch - desc.stride_x)], 0);
                        src_off = next_src_off;
                        dst_off = next_dst_off;
                    }
                }
            }
            upload_resource.unmapEntireBuffer();
            pixEndEvent();
        }

        return BasicTexture{
            .desc = desc,
            .resource = resource.?,
            .resource_stride_x = @intCast(row_pitch),
            .states = c.D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            .size_in_bytes = size_in_bytes,
            .array_slice_size_in_bytes = array_slice_size_in_bytes,
            .texture_view = tex_srv_handle,
            .texture_array_view = tex_array_srv_handle,
            .staging_descriptor_heap = &dx.staging_descriptor_heap,
        };
    }

    pub fn deinit(self: *BasicTexture) void {
        self.resource.deinit();
        self.staging_descriptor_heap.free(self.texture_view);
        self.staging_descriptor_heap.free(self.texture_array_view);
    }

    pub fn upload(self: *BasicTexture, upload_resource: Resource, cl: GraphicsCommandList) void {
        //copy from the upload texture to the real texture; one array slice at a time
        var array_idx: usize = 0;
        while (array_idx < self.desc.array_count) : (array_idx += 1) {
            cl.copyTextureSubresource(
                c.D3D12_TEXTURE_COPY_LOCATION{
                    .pResource = self.resource.resource,
                    .Type = c.D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    .unnamed_0 = .{
                        .SubresourceIndex = @intCast(array_idx),
                    },
                },
                c.D3D12_TEXTURE_COPY_LOCATION{
                    .pResource = upload_resource.resource,
                    .Type = c.D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    .unnamed_0 = .{
                        .PlacedFootprint = .{
                            .Offset = array_idx * self.array_slice_size_in_bytes,
                            .Footprint = .{
                                .Format = self.desc.format,
                                .Width = self.desc.size_x,
                                .Height = self.desc.size_y,
                                .Depth = 1,
                                .RowPitch = self.resource_stride_x,
                            },
                        },
                    },
                }
            );
        }

        //now make the texture ready for use
        cl.resourceBarrier(c.D3D12_RESOURCE_BARRIER{
            .Type = c.D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            .Flags = c.D3D12_RESOURCE_BARRIER_FLAG_NONE,
            .unnamed_0 = .{
                .Transition = .{
                    .pResource   = self.resource.resource,
                    .Subresource = c.D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    .StateBefore = c.D3D12_RESOURCE_STATE_COPY_DEST,
                    .StateAfter  = c.D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                },
            },
        });
    }
};

pub const SwapChain = struct {
    const back_buffer_count = 2;
    const BackBuffer = struct {
        resource: Resource,
        states: ResourceStates,
        view: CpuDescriptorHandle,
        fence_value: u64,
    };

    rt: RenderTarget,
    os_window_handle: std.os.windows.HWND,
    back_buffer_idx: u32,
    back_buffers: [back_buffer_count]SwapChain.BackBuffer,
    swap_fence: Fence,
    swap_chain: *c.IDXGISwapChain3,
    swap_num: u64,
    gpu_swap_num: u64,
    is_occluded: bool,

    pub fn init(dx: *D3D, size_x: u32, size_y: u32, os_window_handle: std.os.windows.HWND) !SwapChain {
        const format = c.DXGI_FORMAT_B8G8R8A8_UNORM;
        var sc = SwapChain{
            .rt = .{
                .resource = undefined,
                .states = c.D3D12_RESOURCE_STATE_PRESENT,
                .view = undefined,
                .desc = .{
                    .format = format,
                    .size_x = size_x,
                    .size_y = size_y,
                    .allow_unordered_access = false,
                },
            },
            .os_window_handle = os_window_handle,
            .back_buffer_idx = undefined,
            .back_buffers = undefined,
            .swap_fence = undefined,
            .swap_chain = undefined,
            .swap_num = 1,
            .gpu_swap_num = 1,
            .is_occluded = false,
        };

        const desc = c.DXGI_SWAP_CHAIN_DESC1{
            .Width = size_x,
            .Height = size_y,
            .Format = format,
            .Stereo = c.FALSE,
            .SampleDesc = .{ .Count = 1, .Quality = 0 },
            .BufferUsage = c.DXGI_USAGE_RENDER_TARGET_OUTPUT | c.DXGI_USAGE_BACK_BUFFER,
            .BufferCount = back_buffer_count,
            .Scaling = c.DXGI_SCALING_NONE,
            .SwapEffect = c.DXGI_SWAP_EFFECT_FLIP_DISCARD,
            .AlphaMode = c.DXGI_ALPHA_MODE_IGNORE,
            .Flags = 0,
        };

        {
            var sc1: ?*c.IDXGISwapChain1 = null;
            try verify(@as(*const fn ([*c]c.IDXGIFactory4, [*c]c.IUnknown, std.os.windows.HWND, [*c]const c.DXGI_SWAP_CHAIN_DESC1, [*c]const c.DXGI_SWAP_CHAIN_FULLSCREEN_DESC, [*c]c.IDXGIOutput, [*c][*c]c.IDXGISwapChain1) callconv(.C) c.HRESULT, @ptrCast(vtbl(dx.factory).CreateSwapChainForHwnd.?))(
                dx.factory,
                @as(*c.IUnknown, @ptrCast(dx.command_queue.cq)),
                os_window_handle,
                &desc,
                null,
                null,
                &sc1
            ));
            std.debug.assert(sc1 != null);
            var op: ?*anyopaque = null;
            try verify(vtbl(sc1.?).QueryInterface.?(sc1.?, &IID_ISwapChain3, &op));
            sc.swap_chain = d3dPtrCast(c.IDXGISwapChain3, op.?);
        }

        //this is for alt-enter handling
        try verify(@as(*const fn ([*c]c.IDXGIFactory4, std.os.windows.HWND, c.UINT) callconv(.C) c.HRESULT, @ptrCast(vtbl(dx.factory).MakeWindowAssociation.?))(dx.factory, os_window_handle, 0));

        //get back buffer info
        var idx: u32 = 0;
        while (idx < back_buffer_count) : (idx += 1) {
            var opaque_buf: ?*anyopaque = null;
            try verify(vtbl(sc.swap_chain).GetBuffer.?(sc.swap_chain, idx, &IID_ID3D12Resource, &opaque_buf));
            sc.back_buffers[idx] = SwapChain.BackBuffer{
                .resource = .{ .resource = d3dPtrCast(c.ID3D12Resource, opaque_buf.?) },
                .states = c.D3D12_RESOURCE_STATE_PRESENT,
                .view = dx.render_target_descriptor_heap.alloc(),
                .fence_value = 0,
            };
            dx.device.createRenderTargetView(sc.back_buffers[idx].resource, null, sc.back_buffers[idx].view);
        }

        //create fence
        sc.swap_fence = try dx.device.createFence();

        sc.setRenderTargetToMatchCurrentBackBuffer();
        return sc;
    }

    fn setRenderTargetToMatchCurrentBackBuffer(self: *SwapChain) void {
        self.back_buffer_idx = vtbl(self.swap_chain).GetCurrentBackBufferIndex.?(self.swap_chain);
        self.rt.resource = self.back_buffers[self.back_buffer_idx].resource;
        self.rt.states = self.back_buffers[self.back_buffer_idx].states;
        self.rt.view = self.back_buffers[self.back_buffer_idx].view;
    }

    pub fn resizeIfNecessary(self: *SwapChain, dx: *D3D, size_x: u32, size_y: u32) !void {
        if ( (size_x < 1) or (size_y < 1) ) {
            return;
        }
        if ( (self.rt.desc.size_x == size_x) and (self.rt.desc.size_y == size_y) ) {
            return;
        }

        //wait for the swap chain to be completely finished by the GPU
        try dx.gpuFlush();

        //release back buffers
        var idx: usize = 0;
        while (idx < back_buffer_count) : (idx += 1) {
            self.back_buffers[idx].resource.deinit();
        }

        //do the resize
        try verify(vtbl(self.swap_chain).ResizeBuffers.?(self.swap_chain, back_buffer_count, size_x, size_y, self.rt.desc.format, 0));

        //reacquire the back buffers
        idx = 0;
        while (idx < back_buffer_count) : (idx += 1) {
            var opaque_buf: ?*anyopaque = null;
            try verify(vtbl(self.swap_chain).GetBuffer.?(self.swap_chain, @intCast(idx), &IID_ID3D12Resource, &opaque_buf));
            self.back_buffers[idx].resource = .{ .resource = d3dPtrCast(c.ID3D12Resource, opaque_buf.?) };
            self.back_buffers[idx].states = c.D3D12_RESOURCE_STATE_PRESENT;
            //?? self.back_buffers[idx].fence_value = 0;
            vtbl(dx.device.device).CreateRenderTargetView.?(dx.device.device, self.back_buffers[idx].resource.resource, null, self.back_buffers[idx].view);
        }

        self.rt.desc.size_x = size_x;
        self.rt.desc.size_y = size_y;
        self.setRenderTargetToMatchCurrentBackBuffer();
    }

    pub fn present(self: *SwapChain, dx: *D3D) !void {
        std.debug.assert(self.rt.states == c.D3D12_RESOURCE_STATE_PRESENT);
        self.back_buffers[self.back_buffer_idx].states = self.rt.states;
        self.back_buffers[self.back_buffer_idx].fence_value = self.swap_num;
        const r = vtbl(self.swap_chain).Present.?(self.swap_chain, 1, 0);
        switch (r) {
            c.S_OK => { self.is_occluded = false; },
            c.DXGI_STATUS_OCCLUDED => { self.is_occluded = true; },
            else => {
                log.err("SwapChain.present() failed! with result {x}\n", .{@as(u32, @bitCast(r))});
                return Error.PresentD3DFailure;
            },
        }
        try dx.command_queue.signal(self.swap_fence, self.swap_num);
        self.swap_num += 1;
        self.setRenderTargetToMatchCurrentBackBuffer();

        //handle waiting for the previous buffered frame
        const buffered_fence = self.back_buffers[self.back_buffer_idx].fence_value;
        const completed_fence = self.swap_fence.getCompletedValue();
        if (completed_fence < buffered_fence) {
            self.gpu_swap_num = try self.swap_fence.setEventOnCompletionAndWait(buffered_fence);
            std.debug.assert(self.gpu_swap_num >= buffered_fence);
        } else {
            self.gpu_swap_num = completed_fence;
        }

        self.back_buffers[self.back_buffer_idx].fence_value = self.swap_num;
    }
};

pub fn gpuFlush(self: *D3D) !void {
    self.flush_fence_value += 1;
    try self.command_queue.signal(self.flush_fence, self.flush_fence_value);
    const completed_value = try self.flush_fence.setEventOnCompletionAndWait(self.flush_fence_value);
    std.debug.assert(completed_value == self.flush_fence_value);
}

// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// PIX helpers
// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const pixInitModule = switch (c.IS_PIX_ENABLED) {
    0 => pixInitModuleVoid,
    else => c.CPIXInitModule,
};
inline fn pixInitModuleVoid() void {}

const pixSetEnableHUDInternal = switch (c.IS_PIX_ENABLED) {
    0 => pixSetEnableHUDVoid,
    else => c.CPIXSetEnableHUD,
};
inline fn pixSetEnableHUDVoid(x: c.BOOL) void { _ = x; }

pub inline fn pixSetEnableHUD(x: bool) void {
    pixSetEnableHUDInternal(if (x) c.TRUE else c.FALSE);
}

const pixReportCounterInternal = switch (c.IS_PIX_ENABLED) {
    0 => pixReportCounterVoid,
    else => c.CPIXReportCounter,
};
inline fn pixReportCounterVoid(name: [*c]const u16, value: f32) void {
    _ = name;
    _ = value;
}

pub inline fn pixReportCounter(name: [:0]const u16, value: f32) void {
    pixReportCounterInternal(name.ptr, value);
}

const pixBeginEventCLInternal = switch (c.IS_PIX_ENABLED) {
    0 => pixBeginEventCLVoid,
    else => c.CPIXBeginEventCL,
};
inline fn pixBeginEventCLVoid(cl: *c.ID3D12GraphicsCommandList, category: u8, name: [*c]const u8) void {
    _ = cl;
    _ = category;
    _ = name;
}

const pixEndEventCLInternal = switch (c.IS_PIX_ENABLED) {
    0 => pixEndEventCLVoid,
    else => c.CPIXEndEventCL,
};
inline fn pixEndEventCLVoid(cl: *c.ID3D12GraphicsCommandList) void {
    _ = cl;
}

const pixBeginEventCQInternal = switch (c.IS_PIX_ENABLED) {
    0 => pixBeginEventCQVoid,
    else => c.CPIXBeginEventCQ,
};
inline fn pixBeginEventCQVoid(cq: *c.ID3D12CommandQueue, category: u8, name: [*c]const u8) void {
    _ = cq;
    _ = category;
    _ = name;
}

const pixEndEventCQInternal = switch (c.IS_PIX_ENABLED) {
    0 => pixEndEventCQVoid,
    else => c.CPIXEndEventCQ,
};
inline fn pixEndEventCQVoid(cq: *c.ID3D12CommandQueue) void {
    _ = cq;
}

const pixBeginEventInternal = switch (c.IS_PIX_ENABLED) {
    0 => pixBeginEventVoid,
    else => c.CPIXBeginEvent,
};
inline fn pixBeginEventVoid(category: u8, name: [*:0]const u8) void {
    _ = category;
    _ = name;
}

pub inline fn pixBeginEvent(category: u8, name: [*:0]const u8) void {
    pixBeginEventInternal(category, name);
}

const pixEndEventInternal = switch (c.IS_PIX_ENABLED) {
    0 => pixEndEventVoid,
    else => c.CPIXEndEvent,
};
inline fn pixEndEventVoid() void {}

pub inline fn pixEndEvent() void {
    pixEndEventInternal();
}

const pixSetMarkerCLInternal = switch (c.IS_PIX_ENABLED) {
    0 => pixSetMarkerCLVoid,
    else => c.CPIXSetMarkerCL,
};
inline fn pixSetMarkerCLVoid(cl: *c.ID3D12GraphicsCommandList, category: u8, name: [*c]const u8) void {
    _ = cl;
    _ = category;
    _ = name;
}

const pixSetMarkerCQInternal = switch (c.IS_PIX_ENABLED) {
    0 => pixSetMarkerCQVoid,
    else => c.CPIXSetMarkerCQ,
};
inline fn pixSetMarkerCQVoid(cq: *c.ID3D12CommandQueue, category: u8, name: [*c]const u8) void {
    _ = cq;
    _ = category;
    _ = name;
}

const pixSetMarkerInternal = switch (c.IS_PIX_ENABLED) {
    0 => pixSetMarkerVoid,
    else => c.CPIXSetMarker,
};
inline fn pixSetMarkerVoid(category: u8, name: [*c]const u8) void {
    _ = category;
    _ = name;
}

pub inline fn pixSetMarker(category: u8, name: [:0]const u8) void {
    pixSetMarkerInternal(category, name.ptr);
}

pub const D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH = 10;

pub const D3D12_PIPELINE_STATE_STREAM_DESC = extern struct {
    SizeInBytes: c.SIZE_T,
    pPipelineStateSubobjectStream: *anyopaque,
};

pub const D3D12_PIPELINE_STATE_SUBOBJECT_TYPE = enum(c.UINT) {
    ROOT_SIGNATURE = 0,
    VS,
    PS,
    DS,
    HS,
    GS,
    CS,
    STREAM_OUTPUT,
    BLEND,
    SAMPLE_MASK,
    RASTERIZER,
    DEPTH_STENCIL,
    INPUT_LAYOUT,
    IB_STRIP_CUT_VALUE,
    PRIMITIVE_TOPOLOGY,
    RENDER_TARGET_FORMATS,
    DEPTH_STENCIL_FORMAT,
    SAMPLE_DESC,
    NODE_MASK,
    CACHED_PSO,
    FLAGS,
    DEPTH_STENCIL1,
    VIEW_INSTANCING,
    AS = 24,
    MS = 25,
    MAX_VALID
};

pub const D3D12_RT_FORMAT_ARRAY = extern struct {
    RTFormats: [8]Format,
    NumRenderTargets: c.UINT,
};

pub const PIPELINE_STATE_MESH_STREAM = extern struct {
    pRootSignature_Type: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE align(8) = .ROOT_SIGNATURE,
    pRootSignature: ?*c.ID3D12RootSignature,
    PS_Type: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE align(8) = .PS,
    PS: c.D3D12_SHADER_BYTECODE,
    AS_Type: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE align(8) = .AS,
    AS: c.D3D12_SHADER_BYTECODE,
    MS_Type: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE align(8) = .MS,
    MS: c.D3D12_SHADER_BYTECODE,
    BlendState_Type: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE align(8) = .BLEND,
    BlendState: c.D3D12_BLEND_DESC,
    DepthStencilState_Type: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE align(8) = .DEPTH_STENCIL,
    DepthStencilState: c.D3D12_DEPTH_STENCIL_DESC,
    DSVFormat_Type: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE align(8) = .DEPTH_STENCIL_FORMAT,
    DSVFormat: Format,
    RasterizerState_Type: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE align(8) = .RASTERIZER,
    RasterizerState: c.D3D12_RASTERIZER_DESC,
    RTVFormats_Type: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE align(8) = .RENDER_TARGET_FORMATS,
    RTVFormats: D3D12_RT_FORMAT_ARRAY,
    SampleDesc_Type: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE align(8) = .SAMPLE_DESC,
    SampleDesc: c.DXGI_SAMPLE_DESC,
    SampleMask_Type: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE align(8) = .SAMPLE_MASK,
    SampleMask: c.UINT,
    PrimitiveTopologyType_Type: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE align(8) = .PRIMITIVE_TOPOLOGY,
    PrimitiveTopologyType: c.D3D12_PRIMITIVE_TOPOLOGY_TYPE,
};

const ID3D12Device2Vtbl = extern struct {
    device_vtbl: c.ID3D12DeviceVtbl,
    device1_vtbl: extern struct {
        //these are incomplete, fill in if you need them
        CreatePipelineLibrary: ?*const fn(self: *anyopaque) callconv(std.os.windows.WINAPI) c.HRESULT,
        SetEventOnMultipleFenceCompletion: ?*const fn(self: *anyopaque) callconv(std.os.windows.WINAPI) c.HRESULT,
        SetResidencyPriority: ?*const fn(self: *anyopaque) callconv(std.os.windows.WINAPI) c.HRESULT,
    },

    //device2
    CreatePipelineState: ?*const fn(self: *anyopaque, desc: *const D3D12_PIPELINE_STATE_STREAM_DESC, *const c.GUID, *?*anyopaque) callconv(std.os.windows.WINAPI) c.HRESULT,
};

pub const D3D12_MESH_SHADER_TIER_NOT_SUPPORTED = 0;
pub const D3D12_MESH_SHADER_TIER_1 = 10;
pub const D3D12_SAMPLER_FEEDBACK_TIER_NOT_SUPPORTED = 0;
pub const D3D12_SAMPLER_FEEDBACK_TIER_0_9 = 90;
pub const D3D12_SAMPLER_FEEDBACK_TIER_1_0 = 100;

pub const D3D12_FEATURE_DATA_D3D12_OPTIONS7 = extern struct {
    MeshShaderTier: d3d.LONG,
    SamplerFeedbackTier: d3d.LONG,
};

pub const D3D12_FEATURE_DATA_D3D12_OPTIONS9 = extern struct {
    MeshShaderPipelineStatsSupported: c.BOOL,
    MeshShaderSupportsFullRangeRenderTargetArrayIndex: c.BOOL,
    AtomicInt64OnTypedResourceSupported: c.BOOL,
    AtomicInt64OnGroupSharedSupported: c.BOOL,
    DerivativesInMeshAndAmplificationShadersSupported: c.BOOL,
    WaveMMATier: c.LONG,
};

pub const D3D12_FEATURE_DATA_D3D12_OPTIONS10 = extern struct {
    VariableRateShadingSumCombinerSupported: c.BOOL,
    MeshShaderPerPrimitiveShadingRateSupported: c.BOOL,
};

pub const D3D12_FEATURE_DATA_D3D12_OPTIONS11 = extern struct {
    AtomicInt64OnDescriptorHeapResourceSupported: c.BOOL,
};

pub const D3D12_FEATURE_DATA_D3D12_OPTIONS12 = extern struct {
    MSPrimitivesPipelineStatisticIncludesCulledPrimitives: c.LONG,
    EnhancedBarriersSupported: c.BOOL,
    RelaxedFormatCastingSupported: c.BOOL,
};

pub const D3D12_FEATURE_D3D12_OPTIONS9 = 37;
pub const D3D12_FEATURE_D3D12_OPTIONS10 = 39;
pub const D3D12_FEATURE_D3D12_OPTIONS11 = 40;
pub const D3D12_FEATURE_D3D12_OPTIONS12 = 41;
pub const D3D12_FEATURE_D3D12_OPTIONS13 = 42;

// ************************************************************************************************************************
// tracy
// ************************************************************************************************************************

const tracy_enabled = @hasDecl(c, "TRACY_ENABLE");
const tracy_callstack_depth = if (@hasDecl(c, "TRACY_HAS_CALLSTACK") and @hasDecl(c, "TRACY_CALLSTACK")) c.TRACY_CALLSTACK else 0;

inline fn tracyGetCtx(dx: *const D3D) *c.TracyD3D12QueueCtx {
    return @ptrCast(dx.tracy_queue_ctx.?);
}

pub fn tracyNewFrame(dx: *const D3D) void {
    if (tracy_enabled) {
        c.tracyContextNewFrame(tracyGetCtx(dx));
    }
}

pub inline fn beginGpuProfileZone(dx: *const D3D, comptime src: std.builtin.SourceLocation, color: u32, name_str: ?[*:0]const u8) u32 {
    if (tracy_enabled) {
        //this helps make every location a unique address in memory; since every src is a different location
        const static = struct {
            var loc: c.___tracy_source_location_data = undefined;
        };
        static.loc = .{
            .name = name_str,
            .function = src.fn_name.ptr,
            .file = src.file.ptr,
            .line = src.line,
            .color = color,
        };

        return c.tracyD3D12ZoneBegin(tracyGetCtx(dx), &static.loc, @max(1, tracy_callstack_depth));
    } else {
        return 0;
    }
}

pub inline fn endGpuProfileZone(dx: *const D3D, zone_id: u32) void {
    if (tracy_enabled) {
        c.tracyD3D12ZoneEnd(tracyGetCtx(dx), zone_id);
    }
}

pub inline fn beginGpuProfileQuery(dx: *const D3D, cl: *GraphicsCommandList, id: u32) void {
    if (tracy_enabled) {
        c.tracyD3D12QueryBegin(tracyGetCtx(dx), cl.cl, id);
    }
}

pub inline fn endGpuProfileQuery(dx: *const D3D, cl: *GraphicsCommandList, id: u32) void {
    if (tracy_enabled) {
        c.tracyD3D12QueryEnd(tracyGetCtx(dx), cl.cl, id);
    }
}
