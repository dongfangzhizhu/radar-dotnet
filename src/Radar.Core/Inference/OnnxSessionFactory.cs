using Microsoft.ML.OnnxRuntime;

namespace Radar.Core.Inference;

public enum ExecutionDevice
{
    Cpu,
    DirectMl,
    Cuda,
}

/// <summary>
/// ONNX 会话工厂：按设备枚举创建 SessionOptions（CPU / DirectML / CUDA），
/// auto 模式依次探活回退。EP 选择见 PLAN.md §3.2。
/// </summary>
public static class OnnxSessionFactory
{
    /// <summary>创建 session；device=Auto 时按 CUDA→DirectML→CPU 顺序探活。</summary>
    public static (InferenceSession Session, ExecutionDevice Used) Create(string modelPath, ExecutionDevice device)
    {
        if (device == ExecutionDevice.Cpu)
            return (NewSession(modelPath, CpuOptions()), ExecutionDevice.Cpu);

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
                // 回退 CPU（探活失败：无兼容 GPU / DML 设备不可用）
            }
        }

        if (device == ExecutionDevice.Cuda)
        {
            try
            {
                var so = CpuOptions();
                so.AppendExecutionProvider_CUDA(0);
                return (NewSession(modelPath, so), ExecutionDevice.Cuda);
            }
            catch (Exception)
            {
                // 回退 CPU
            }
        }

        return (NewSession(modelPath, CpuOptions()), ExecutionDevice.Cpu);
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
