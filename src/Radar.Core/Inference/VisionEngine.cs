using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Radar.Core.Inference;

/// <summary>
/// 视觉主干单窗推理结果（radar_vision.onnx 的 8 个输出）。
/// 尺寸契约见 export/vec_vision_branch.py 头注释：
///   SegProbs (37,96,128,192) / PredMask (96,256,384) uint8
///   ResX1 (1152,256) ResX2 (9216,256) ResX3 (73728,256)
///   Flags1 (36,1152) Flags2 (36,9216) Flags3 (36,73728)（bool→byte 0/1）
/// </summary>
public sealed record WindowOutputs(
    float[] SegProbs,
    byte[] PredMask,
    float[] ResX1, float[] ResX2, float[] ResX3,
    byte[] Flags1, byte[] Flags2, byte[] Flags3)
{
    public const int SegClasses = 37;
    public const int L1 = 96 * 128 / 1 * 192 / 1; // 1152（96,128,192 展平）
    public const int OrganCount = 36;
}

/// <summary>
/// radar_vision.onnx 推理封装：固定输入 (1,1,96,256,384)，输出 8 张量。
/// </summary>
public sealed class VisionEngine : IDisposable
{
    public const int InputD = 96, InputH = 256, InputW = 384;

    private readonly InferenceSession _session;
    private readonly string _inputName;

    public ExecutionDevice Device { get; }

    public VisionEngine(string modelPath, ExecutionDevice device = ExecutionDevice.Cpu)
    {
        (_session, Device) = OnnxSessionFactory.Create(modelPath, device);
        _inputName = _session.InputMetadata.Keys.First();
    }

    private static readonly string[] OutputNames =
    [
        "seg_probs", "pred_mask", "res_x1", "res_x2", "res_x3", "flags1", "flags2", "flags3",
    ];

    /// <summary>单窗前向。window 长度 = 1*1*96*256*384，(D,H,W) 行主序。</summary>
    public WindowOutputs Run(float[] window)
    {
        var shape = new[] { 1, 1, InputD, InputH, InputW };
        var tensor = new DenseTensor<float>(window, shape);
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_inputName, tensor) };

        using var results = _session.Run(inputs, OutputNames);
        var byName = results.ToDictionary(r => r.Name, r => r);

        var segProbs = TensorToFloat(byName["seg_probs"], 37 * 96 * 128 * 192);
        var predMask = TensorToByte(byName["pred_mask"], InputD * InputH * InputW);
        var resX1 = TensorToFloat(byName["res_x1"], 1152 * 256);
        var resX2 = TensorToFloat(byName["res_x2"], 9216 * 256);
        var resX3 = TensorToFloat(byName["res_x3"], 73728 * 256);
        var flags1 = TensorToByte(byName["flags1"], 36 * 1152);
        var flags2 = TensorToByte(byName["flags2"], 36 * 9216);
        var flags3 = TensorToByte(byName["flags3"], 36 * 73728);

        return new WindowOutputs(segProbs, predMask, resX1, resX2, resX3, flags1, flags2, flags3);
    }

    private static float[] TensorToFloat(NamedOnnxValue v, int expected)
    {
        var t = (DenseTensor<float>)v.Value;
        if (t.Length != expected)
            throw new InvalidDataException($"输出尺寸不符: {t.Length} != {expected}");
        return t.Buffer.ToArray();
    }

    private static byte[] TensorToByte(NamedOnnxValue v, int expected)
    {
        // ONNX bool/uint8 输出统一读 byte
        if (v.Value is DenseTensor<byte> bu8)
        {
            if (bu8.Length != expected)
                throw new InvalidDataException($"输出尺寸不符: {bu8.Length} != {expected}");
            return bu8.Buffer.ToArray();
        }
        var b = (DenseTensor<bool>)v.Value;
        if (b.Length != expected)
            throw new InvalidDataException($"输出尺寸不符: {b.Length} != {expected}");
        var arr = new byte[expected];
        var buf = b.Buffer;
        for (int i = 0; i < expected; i++)
            arr[i] = buf.Span[i] ? (byte)1 : (byte)0;
        return arr;
    }

    public void Dispose() => _session.Dispose();
}
