using System.Text;
using Radar.Cli;

Console.OutputEncoding = Encoding.UTF8;

if (args.Length == 0 || args[0] is "-h" or "--help" or "help")
{
    PrintHelp();
    return 0;
}

try
{
    return args[0] switch
    {
        "infer" => InferCommand.Run(Options.Parse(args, 1)),
        "eval" => EvalCommand.Run(Options.Parse(args, 1)),
        _ => Fail($"未知子命令: {args[0]}（可用: infer / eval / help）"),
    };
}
catch (Exception ex)
{
    Console.Error.WriteLine($"错误: {ex.Message}");
    return 1;
}

static int Fail(string message)
{
    Console.Error.WriteLine(message);
    return 2;
}

static void PrintHelp()
{
    Console.WriteLine("""
        radar —— RADAR 推理命令行（对齐上游 damo-radar/RADAR_inference 的用法）

        用法:
          radar infer --img-dir <目录> [选项]
          radar eval  --results-csv <宽表CSV> --labels <标签JSON> [选项]

        infer 选项:
          --img-dir <目录>     必填。待推理病例目录（.nii / .nii.gz / .dcm / .zip）
          --save-dir <目录>    结果目录（默认 <仓库根>/results）
          --save-tag <标签>    输出文件名后缀（默认 demo）
                               产出 <save-dir>/RADAR_infer_results_<标签>.csv
          --device <设备>      cpu | directml | cuda（默认 cpu）
          --devices <列表>     多设备分片，逗号分隔，如 cpu,cuda（与 --workers 配合）
          --workers <N>        并行度（默认 1）。每个 worker 独占一个推理会话
          --model <模型>       radar | radar_plus（默认 radar）
          --variant <变体>     radar_plus 用: pretrain | merlin_ft（默认 merlin_ft）
          --save-mask          额外输出分割掩码透叠图 <save-dir>/masks/<病例>.png
          --limit <N>          只处理前 N 个病例（调试用）

        eval 选项:
          --results-csv <路径> 必填。infer 产出的宽表 CSV
          --labels <路径>      必填。MERLIN 标签 JSON {疾病键: {病例ID: 标签}}
          --sag-counts <路径>  可选。胆囊分割体素数 JSON {病例ID: 体素数}，
                               用于复刻上游对「胆囊_术后胆囊缺失」的体素数判定口径

        说明:
          · 空间任一维超过 1000 的病例按上游约定跳过（不计入结果表，也不视为失败）。
          · 多设备/多 worker 的结果按输入文件顺序聚合，等价上游多卡 all_gather 的有序汇总。
        """);
}
