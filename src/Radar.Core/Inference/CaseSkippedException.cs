namespace Radar.Core.Inference;

/// <summary>
/// 病例被<b>跳过</b>（不是失败）：对齐上游 <c>inference_demo.py</c> 的保护逻辑 ——
/// 预处理后任一空间维超过 1000 时 <c>continue</c>，不进入推理。
/// 批处理时应当捕获本异常并跳过该病例，而不是中断整批。
/// </summary>
public sealed class CaseSkippedException(string message) : Exception(message);
