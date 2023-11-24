const std = @import("std");

pub fn addModule(b: *std.build.Builder) *std.build.Module {
    return b.addModule("gdi", .{
        .source_file = .{ .path = relativePath(b, "gdi.zig") },
        .dependencies = &.{
            .{ .name = "D3D", .module = b.modules.get("D3D").? },
        },
    });
}

fn relativePath(b: *std.build.Builder, path: []const u8) []const u8 {
    return b.pathJoin(&.{ std.fs.path.dirname(@src().file) orelse ".", path });
}
