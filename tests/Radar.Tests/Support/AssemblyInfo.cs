// 禁用 xunit 测试并行：端到端 / DICOM / Plus 等重测试各自创建 ONNX 会话
// （单会话 BFC arena >1GB），并行跑会触发 BFCArena 分配失败（1.2GB OOM，实测）。
[assembly: Xunit.CollectionBehavior(DisableTestParallelization = true)]
