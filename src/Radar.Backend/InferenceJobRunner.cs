using System.Collections.Concurrent;
using System.Threading.Channels;
using Radar.Core.Inference;
using Radar.Core.Nifti;
using Radar.Core.Preprocessing;
using Radar.Core.Support;

namespace Radar.Backend;

/// <summary>任务状态机：Queued → Running → Completed / Failed。</summary>
public enum JobStatus { Queued, Running, Completed, Failed }

/// <summary>单条推理任务的记录（线程安全读：字段只在完成时一次性写入）。</summary>
public class JobRecord
{
    public required string Id { get; init; }
    public required string FileName { get; init; }
    public required string Device { get; init; }
    public JobStatus Status { get; set; } = JobStatus.Queued;
    public DateTime CreatedAt { get; init; } = DateTime.Now;
    public DateTime? StartedAt { get; set; }
    public DateTime? FinishedAt { get; set; }
    public long ElapsedMs { get; set; }
    public int WindowCount { get; set; }
    public Dictionary<string, double>? Probabilities { get; set; }
    public string? Error { get; set; }
    /// <summary>内部：上传/样例 NIfTI 的磁盘路径。</summary>
    internal string? FilePath { get; set; }
}

/// <summary>
/// 推理任务队列：单 worker（Channel）串行执行，符合 GPU 会话独占约束；
/// 模型会话按任务创建（设备不同）、用后即毁；文本特征/轻头权重进程级缓存。
/// </summary>
public sealed class InferenceJobRunner : BackgroundService
{
    private readonly Channel<JobRecord> _queue = Channel.CreateUnbounded<JobRecord>();
    private readonly ConcurrentDictionary<string, JobRecord> _jobs = new();
    private readonly string _modelsDir;
    private readonly string _uploadsDir;
    private readonly ILogger<InferenceJobRunner> _logger;

    // 设备探活结果缓存（true/false；null=未探测）
    private readonly ConcurrentDictionary<string, bool> _deviceProbe = new();

    public InferenceJobRunner(string repoRoot, ILogger<InferenceJobRunner> logger)
    {
        _modelsDir = Path.Combine(repoRoot, "models");
        _uploadsDir = Path.Combine(repoRoot, "uploads");
        Directory.CreateDirectory(_uploadsDir);
        _logger = logger;
    }

    public IReadOnlyDictionary<string, JobRecord> Jobs => _jobs;

    // ---- 公共依赖（进程级单例，线程安全：纯只读数学） ----
    private OrganHead? _head;
    private OrganHead Head => _head ??= new OrganHead(
        OrganHeadWeights.Load(Path.Combine(_modelsDir, "organ_head.npz")),
        TextFeatures.Load(
            Path.Combine(_modelsDir, "text_feats.bin"),
            Path.Combine(_modelsDir, "text_feats.json")));

    /// <summary>146 个打分条目（来自 text_feats.json 的索引）。</summary>
    public IReadOnlyList<string> TestItems => Head.Text.Items;

    /// <summary>test_items 涉及的器官集合（「器官_病灶」取前缀去重）。</summary>
    public List<string> TestOrgans =>
        TestItems.Select(i => i.Split('_')[0]).Distinct().ToList();

    public string ModelsDir => _modelsDir;

    // ---- 设备探活 ----

    /// <summary>
    /// 探测某 EP 是否可用。注意 OnnxSessionFactory.Create 失败会静默回退 CPU，
    /// 因此必须校验返回的实际 EP 与请求一致，否则 DML/CUDA 不可用时会被误报为可用。
    /// </summary>
    public bool ProbeDevice(string device)
    {
        var want = ParseDevice(device);
        return _deviceProbe.GetOrAdd(device, d =>
        {
            try
            {
                var (sess, used) = OnnxSessionFactory.Create(
                    Path.Combine(_modelsDir, "radar_vision.onnx"), want);
                sess.Dispose();
                if (used != want)
                {
                    _logger.LogWarning("设备 {Device} 不可用（会话回退为 {Used}）", d, used);
                    return false;
                }
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogWarning("设备 {Device} 探活失败: {Msg}", d, ex.Message);
                return false;
            }
        });
    }

    public static ExecutionDevice ParseDevice(string device) => device.ToLowerInvariant() switch
    {
        "cuda" => ExecutionDevice.Cuda,
        "directml" => ExecutionDevice.DirectMl,
        _ => ExecutionDevice.Cpu,
    };

    // ---- 入队 ----

    public async Task<JobRecord> EnqueueAsync(string filePath, string fileName, string device)
    {
        var job = new JobRecord
        {
            Id = Guid.NewGuid().ToString("N")[..12],
            FileName = fileName,
            Device = device.ToLowerInvariant(),
            FilePath = filePath,
        };
        _jobs[job.Id] = job;
        await _queue.Writer.WriteAsync(job);
        return job;
    }

    // ---- worker ----

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        await foreach (var job in _queue.Reader.ReadAllAsync(stoppingToken))
        {
            job.Status = JobStatus.Running;
            job.StartedAt = DateTime.Now;
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                var (probs, windows) = RunInference(job.FilePath!, job.Device);
                job.Probabilities = probs;
                job.WindowCount = windows;
                job.Status = JobStatus.Completed;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "任务 {Id} 推理失败", job.Id);
                job.Status = JobStatus.Failed;
                job.Error = ex.Message;
            }
            finally
            {
                sw.Stop();
                job.ElapsedMs = sw.ElapsedMilliseconds;
                job.FinishedAt = DateTime.Now;
            }
        }
    }

    private (Dictionary<string, double> Probabilities, int Windows) RunInference(string niftiPath, string device)
    {
        // 1) NIfTI 读取 + 预处理（与 M2 同管线）
        var img = NiftiReader.Read(niftiPath);
        var prep = Preprocessor.Run(img, Path.GetFileName(niftiPath));

        // 2) 会话（按任务设备）+ 引擎
        var engine = new VisionEngine(Path.Combine(_modelsDir, "radar_vision.onnx"),
            ParseDevice(device));
        try
        {
            using var radar = new RadarInferenceEngine(engine, Head,
                RadarLabels.Organs, TestItems.ToList());
            var result = radar.Run(prep.Data, prep.D, prep.H, prep.W, TestOrgans);
            return (result.Probabilities.ToDictionary(p => p.Key, p => p.Value), result.WindowCount);
        }
        finally
        {
            engine.Dispose();
        }
    }
}
