using Radar.Core.Inference;

namespace Radar.Cli;

/// <summary>CLI 侧的小工具：仓库根定位与设备名解析。</summary>
internal static class CliEnv
{
    /// <summary>
    /// 从程序目录逐级上溯，找到含 models/radar_vision.onnx 的目录作为仓库根
    /// （与 Radar.Backend 的定位方式一致，保证从 bin 目录直接运行也能找到模型与数据）。
    /// </summary>
    public static string FindRepoRoot()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
            if (File.Exists(Path.Combine(dir.FullName, "models", "radar_vision.onnx")))
                return dir.FullName;
        return Directory.GetCurrentDirectory();
    }

    /// <summary>设备名 -> 执行后端。未识别的一律回落 CPU。</summary>
    public static ExecutionDevice ParseDevice(string device) => device.ToLowerInvariant() switch
    {
        "cuda" => ExecutionDevice.Cuda,
        "directml" or "dml" => ExecutionDevice.DirectMl,
        _ => ExecutionDevice.Cpu,
    };
}
