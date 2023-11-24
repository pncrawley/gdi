const std = @import("std");

pub fn addModule(b: *std.build.Builder) *std.build.Module {
    if (b.modules.get("D3D")) |module| {
        return module;
    } else {
        b.installBinFile("shared/d3d/WinPixEventRuntime/WinPixEventRuntime.dll", "WinPixEventRuntime.dll");
        b.installBinFile("C:\\Program Files (x86)\\Windows Kits\\10\\Redist\\D3D\\x64\\dxcompiler.dll", "dxcompiler.dll");
        b.installBinFile("C:\\Program Files (x86)\\Windows Kits\\10\\Redist\\D3D\\x64\\d3dcompiler_47.dll", "d3dcompiler_47.dll");
        b.installBinFile("C:\\Program Files (x86)\\Windows Kits\\10\\Redist\\D3D\\x64\\d3dcsx_47.dll", "d3dcsx_47.dll");
        b.installBinFile("C:\\Program Files (x86)\\Windows Kits\\10\\Redist\\D3D\\x64\\dxc.exe", "dxc.exe");
        b.installBinFile("C:\\Program Files (x86)\\Windows Kits\\10\\Redist\\D3D\\x64\\dxil.dll", "dxil.dll");

        return b.addModule("D3D", .{
            .source_file = .{ .path = "shared/d3d/D3D.zig" },
            .dependencies = &.{},
        });
    }
}

pub fn addCpp(exe: *std.build.LibExeObjStep, options: struct { enable_tracy: bool = true, enable_pix: bool = false }) void {
    //AMD's d3d12 memory allocator and a c interface
    if (true) {
        exe.addIncludePath("shared/d3d");
        exe.addCSourceFiles(&.{"shared/d3d/D3D12MemAlloc.cpp", "shared/d3d/cD3D12MemAlloc.cpp"}, &.{});
    }

    const is_test_exe = (exe.kind == .@"test");

    //add tracy
    if (!is_test_exe and options.enable_tracy) {
        exe.defineCMacro("TRACY_ENABLE", "");
        exe.addIncludePath("shared/stdlib/stdlib/tracy/public");
        exe.addCSourceFiles(
            &.{ "shared/d3d/TracyD3D12.cpp" },
            &.{
                "-DTRACY_ENABLE",
                "-DTRACY_FIBERS",
            }
        );
    }

    const use_pix = (exe.optimize != .ReleaseFast) and (exe.optimize != .ReleaseSmall) and !is_test_exe and options.enable_pix;
    if (use_pix) {
        exe.defineCMacro("USE_PIX", "");
        exe.addCSourceFiles(&.{"shared/d3d/c_pix.cpp"}, &.{});
        exe.addLibraryPath("shared/d3d/WinPixEventRuntime");
        exe.linkSystemLibraryName("WinPixEventRuntime");
    }

    const source_file_dir = std.fs.path.dirname(@src().file) orelse ".";
    exe.addIncludePath(source_file_dir);
}
