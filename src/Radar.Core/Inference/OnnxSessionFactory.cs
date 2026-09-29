using Microsoft.ML.OnnxRuntime;

namespace Radar.Core.Inference;

public enum ExecutionDevice
{
    Cpu,
    DirectMl,
    Cuda,
}

/// <summary>
/// ONNX 会话工厂：按设备枚举创建 SessionOptions（CPU / CUDA / DirectML）。
/// CUDA EP 依赖本机 CUDA 12 + cuDNN 9 运行库（Gpu 包只带 onnxruntime_providers_cuda.dll，
/// 不带 cudart/cublas/cudnn），创建会话前 EnsureCudaRuntime 自动把本机安装的
/// CUDA/cuDNN bin 目录挂进进程 DLL 搜索路径（PATH 前插），探测失败抛异常由调用方回退 CPU。
/// </summary>
public static class OnnxSessionFactory
{
    private static bool _cudaRuntimePrepared;

    /// <summary>创建 session；目标 EP 不可用时静默回退 CPU（返回 Used 供调用方校验）。</summary>
    public static (InferenceSession Session, ExecutionDevice Used) Create(string modelPath, ExecutionDevice device)
    {
        if (device == ExecutionDevice.Cuda)
        {
            try
            {
                EnsureCudaRuntime();
                var so = CpuOptions();
                so.AppendExecutionProvider_CUDA(0);
                return (NewSession(modelPath, so), ExecutionDevice.Cuda);
            }
            catch (Exception)
            {
                // 回退 CPU（native 库缺失 / 驱动不匹配 / cuDNN 版本不符）
            }
        }

        if (device == ExecutionDevice.DirectMl)
        {
            try
            {
                var so = CpuOptions();
                so.AppendExecutionProvider_DML(0);
                return (NewSession(modelPath, so), ExecutionDevice.DirectMl);
            }
            catch (Exception)
            {
                // 回退 CPU（无兼容 GPU / DML 设备不可用）
            }
        }

        return (NewSession(modelPath, CpuOptions()), ExecutionDevice.Cpu);
    }

    /// <summary>
    /// 探测 CUDA EP 是否真实可用（建一次会话并校验实际 EP，用后即毁）。
    /// Create 会静默回退，因此探活必须比对 Used 与请求一致，否则 GPU 不可用会被误报。
    /// </summary>
    public static bool IsCudaAvailable(string modelPath)
    {
        try
        {
            var (sess, used) = Create(modelPath, ExecutionDevice.Cuda);
            sess.Dispose();
            return used == ExecutionDevice.Cuda;
        }
        catch
        {
            return false;
        }
    }

    // ---- CUDA/cuDNN 运行库路径准备 ----

    /// <summary>按优先级探测的 cuDNN bin 目录（相对 NVIDIA\CUDNN\vX.Y 下）。</summary>
    private static readonly string[] CudnnSubDirs = ["bin\\12.9", "bin\\12.8", "bin\\11.8", "bin"];

    /// <summary>进程级只做一次：把 CUDA bin + cuDNN bin 前插到本进程 PATH。</summary>
    private static void EnsureCudaRuntime()
    {
        if (_cudaRuntimePrepared) return;
        lock (typeof(OnnxSessionFactory))
        {
            if (_cudaRuntimePrepared) return;

            var dirs = new List<string>();

            // CUDA toolkit bin（cudart64_12.dll / cublas64_12.dll）：CUDA_PATH 优先，其次常见版本
            string? cudaPath = Environment.GetEnvironmentVariable("CUDA_PATH");
            if (cudaPath is not null && File.Exists(Path.Combine(cudaPath, "bin", "cudart64_12.dll")))
                dirs.Add(Path.Combine(cudaPath, "bin"));
            string toolkitRoot = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles),
                "NVIDIA GPU Computing Toolkit", "CUDA");
            foreach (var ver in Directory.Exists(toolkitRoot)
                         ? Directory.GetDirectories(toolkitRoot, "v12.*").OrderDescending().ToArray()
                         : [])
            {
                string bin = Path.Combine(ver, "bin");
                if (File.Exists(Path.Combine(bin, "cudart64_12.dll"))) dirs.Add(bin);
            }

            // cuDNN bin（cudnn64_9.dll + cudnn_* 子库）
            string cudnnRoot = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles),
                "NVIDIA", "CUDNN");
            foreach (var ver in Directory.Exists(cudnnRoot)
                         ? Directory.GetDirectories(cudnnRoot, "v9.*").OrderDescending().ToArray()
                         : [])
            {
                foreach (var sub in CudnnSubDirs)
                {
                    string bin = Path.Combine(ver, sub);
                    if (File.Exists(Path.Combine(bin, "cudnn64_9.dll"))) { dirs.Add(bin); break; }
                }
            }

            if (dirs.Count > 0)
            {
                string path = Environment.GetEnvironmentVariable("PATH") ?? "";
                string add = string.Join(';', dirs.Where(d => !path.Contains(d, StringComparison.OrdinalIgnoreCase)));
                if (add.Length > 0)
                    Environment.SetEnvironmentVariable("PATH", add + ";" + path);
            }

            _cudaRuntimePrepared = true;
        }
    }

    private static SessionOptions CpuOptions()
    {
        var so = new SessionOptions();
        so.IntraOpNumThreads = 0;      // ORT 默认线程数
        so.InterOpNumThreads = 1;
        so.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        return so;
    }

    private static InferenceSession NewSession(string path, SessionOptions so) => new(path, so);
}
