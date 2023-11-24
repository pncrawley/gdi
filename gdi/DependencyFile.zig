const std = @import("std");
const Self = @This();

allocator: std.mem.Allocator,
version: u16,
output: []u8,
depends: std.ArrayList([]u8),

pub fn initEmpty(allocator: std.mem.Allocator, output: []const u8, version: u16) !Self {
    return Self{
        .allocator = allocator,
        .version = version,
        .output = try allocator.dupe(u8, output),
        .depends = std.ArrayList([]u8).init(allocator),
    };
}

pub fn initFromFile(allocator: std.mem.Allocator, path: []const u8) !Self {
    //read whole file
    var whole_file = try std.fs.cwd().readFileAlloc(allocator, path, 128*1024*1024);
    defer allocator.free(whole_file);

    var d = Self{
        .allocator = allocator,
        .version = 0,
        .output = undefined,
        .depends = std.ArrayList([]u8).init(allocator),
    };

    //split by newlines; first is the version; second is the output; everything else is the depends
    var idx: usize = 0;
    var it = std.mem.split(u8, whole_file, "\n");
    while (it.next()) |line| {
        std.debug.assert(null == std.mem.indexOf(u8, line, "\n"));
        switch (idx) {
            0 => { d.version = try std.fmt.parseUnsigned(u16, line, 10); },
            1 => { d.output = try allocator.dupe(u8, line); },
            else => { try d.depends.append(try allocator.dupe(u8, line)); },
        }
        idx += 1;
    }

    std.debug.assert(d.depends.items.len > 0);
    return d;
}

pub fn deinit(self: *Self) void {
    for (self.depends.items) |dep| {
        self.allocator.free(dep);
    }
    self.depends.deinit();
    self.allocator.free(self.output);
}

//takes ownership of 'path' memory
pub fn addDependency(self: *Self, path: []u8) !void {
    std.debug.assert(null == std.mem.indexOf(u8, path, "\n"));
    replaceSeperatorsInline(path);
    try self.depends.append(path);
    //std.debug.print("addDependency({s})\n", .{path});
}

pub fn addDependencyDirectory(self: *Self, search_dir: []const u8) !void {
    var dir = try std.fs.openDirAbsolute(search_dir, .{ .iterate = true });
    var walker = try dir.walk(self.allocator);
    defer walker.deinit();

    while (try walker.next()) |entry| {
        if (entry.kind == std.fs.Dir.Entry.Kind.File) {
            var path = try std.fs.path.joinZ(self.allocator, &.{search_dir, entry.path});
            try self.addDependency(path);
        }
    }
}

pub fn isUpToDate(self: *Self, version: u16) bool {
    const output_ts = fileTimestamp(self.output);
    if (self.version != version) {
        std.debug.print("   isUpToDate() version mismatch {} vs expected {}\n", .{self.version, version});
        return false;
    }
    for (self.depends.items) |d| {
        const ts = fileTimestamp(d);
        //std.debug.print("   isUpToDate() check timestamp isOutOfDate:{} => '{s}'{} < {}'{s}'\n", .{ts > output_ts, d, ts, output_ts, self.output});
        if (ts > output_ts) {
            std.debug.print("   isUpToDate() check timestamp isOutOfDate:{} => '{s}'{} < {}'{s}'\n", .{ts > output_ts, d, ts, output_ts, self.output});
            return false;
        }
    }
    return true;
}

pub fn write(self: *Self, path: []const u8) !void {
    //delete existing file if necessary, and ignore the error that it doesn't exist
    std.fs.cwd().deleteFile(path) catch {};

    if (self.depends.items.len < 1) {
        return;
    }

    //create our output file
    std.debug.assert(self.depends.items.len > 0);
    var file = try std.fs.cwd().createFile(path, std.fs.File.CreateFlags{ .exclusive = true });
    defer file.close();

    { //write version number
        var buffer: [10]u8 = undefined;
        const buf = buffer[0..];
        _ = try file.write(std.fmt.bufPrintIntToSlice(buf, self.version, 10, .lower, std.fmt.FormatOptions{}));
        _ = try file.write("\n");
    }

    //write output file
    std.debug.assert(null == std.mem.indexOf(u8, self.output, "\n"));
    _ = try file.write(self.output);

    //write each dependency file
    for (self.depends.items) |d| {
        std.debug.assert(null == std.mem.indexOf(u8, d, "\n"));
        _ = try file.write("\n");
        _ = try file.write(d);
    }
}

fn fileTimestamp(path: []const u8) i128 {
    const stat = std.fs.cwd().statFile(path) catch {
        return std.math.maxInt(i128);
    };
    return stat.mtime;
}

pub fn isFileUpToDate(allocator: std.mem.Allocator, path: []const u8, version: u16) bool {
    var d = Self.initFromFile(allocator, path) catch {
        return false;
    };
    defer d.deinit();
    return d.isUpToDate(version);
}

pub fn replaceSeperatorsInline(path: []u8) void {
    for (path) |*ch| {
        if (ch.* == '\\') {
            ch.* = '/';
        }
    }
}
