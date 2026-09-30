using System.Text.Json.Serialization;
using Radar.Backend;
using Radar.Core.Postprocessing;

// 仓库根定位：从 BaseDirectory 逐级上溯，找到含 models/radar_vision.onnx 的目录
string repoRoot = AppContext.BaseDirectory;
for (var dir = new DirectoryInfo(repoRoot); dir is not null; dir = dir.Parent)
{
    if (File.Exists(Path.Combine(dir.FullName, "models", "radar_vision.onnx")))
    {
        repoRoot = dir.FullName;
        break;
    }
}

var builder = WebApplication.CreateBuilder(args);
builder.Logging.ClearProviders().AddConsole();
builder.Services.ConfigureHttpJsonOptions(o =>
{
    // 前端（wwwroot/index.html）按 camelCase 字段名读取，这里显式声明契约，
    // 不依赖 Minimal API 的隐式默认（默认即 camelCase，但改了配置/版本容易漂）。
    o.SerializerOptions.PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.CamelCase;
    o.SerializerOptions.Converters.Add(new JsonStringEnumConverter()); // enum 序列化为字符串（前端依赖）
});
builder.Services.AddCors(o => o.AddDefaultPolicy(p => p.AllowAnyOrigin().AllowAnyHeader().AllowAnyMethod()));
builder.Services.AddSingleton<InferenceJobRunner>(sp =>
    new InferenceJobRunner(repoRoot, sp.GetRequiredService<ILogger<InferenceJobRunner>>()));
builder.Services.AddHostedService(sp => sp.GetRequiredService<InferenceJobRunner>());

var app = builder.Build();
app.UseCors();
app.UseDefaultFiles();
app.UseStaticFiles();

string testdataDir = Path.Combine(repoRoot, "testdata");

// ---- 设备列表（CPU 恒可用，DirectML/CUDA 探活） ----
app.MapGet("/api/devices", (InferenceJobRunner runner) =>
{
    var list = new[] { "cpu", "directml", "cuda" }
        .Select(d => new { name = d, available = d == "cpu" || runner.ProbeDevice(d) })
        .ToList();
    return Results.Json(list);
});

// ---- 样例列表（testdata 下 NIfTI / DICOM） ----
app.MapGet("/api/samples", () =>
{
    if (!Directory.Exists(testdataDir)) return Results.Json(Array.Empty<object>());
    var samples = new[] { "*.nii.gz", "*.nii", "*.zip", "*.dcm" }
        .SelectMany(p => Directory.GetFiles(testdataDir, p))
        .Select(p => new { name = Path.GetFileName(p) })
        .ToList();
    return Results.Json(samples);
});

// ---- 用样例建任务 ----
app.MapPost("/api/jobs/sample", async (SampleJobRequest req, InferenceJobRunner runner) =>
{
    string path = Path.Combine(testdataDir, req.Sample ?? "");
    if (string.IsNullOrEmpty(req.Sample) || !File.Exists(path))
        return Results.BadRequest(new { error = $"样例不存在: {req.Sample}" });
    var job = await runner.EnqueueAsync(path, req.Sample, req.Device ?? "cpu",
        req.Model ?? "radar", req.Variant ?? "merlin_ft");
    return Results.Ok(new { jobId = job.Id });
});

// ---- 上传建任务 ----
app.MapPost("/api/jobs/upload", async (IFormFile file, string? device, string? model,
    string? variant, InferenceJobRunner runner) =>
{
    if (file.Length == 0) return Results.BadRequest(new { error = "空文件" });
    bool ok = file.FileName.EndsWith(".nii") || file.FileName.EndsWith(".nii.gz")
        || file.FileName.EndsWith(".dcm") || file.FileName.EndsWith(".zip");
    if (!ok)
        return Results.BadRequest(new { error = "仅支持 .nii / .nii.gz / .dcm（多帧）/ .zip（DICOM 序列）" });

    string uploads = Path.Combine(repoRoot, "uploads");
    Directory.CreateDirectory(uploads);
    string path = Path.Combine(uploads, $"{Guid.NewGuid():N}_{file.FileName}");
    await using var fs = File.Create(path);
    await file.CopyToAsync(fs);

    var job = await runner.EnqueueAsync(path, file.FileName, device ?? "cpu",
        model ?? "radar", variant ?? "merlin_ft");
    return Results.Ok(new { jobId = job.Id });
}).DisableAntiforgery();

// ---- 任务列表 / 详情 ----
app.MapGet("/api/jobs", (InferenceJobRunner runner) =>
    Results.Json(runner.Jobs.Values
        .OrderByDescending(j => j.CreatedAt)
        .Select(JobView)
        .ToList()));

app.MapGet("/api/jobs/{id}", (string id, InferenceJobRunner runner) =>
    runner.Jobs.TryGetValue(id, out var j) ? Results.Json(JobView(j)) : Results.NotFound());

// ---- 结果 CSV 下载 ----
// 默认：长表（item,probability）；format=official：与上游 inference_demo.py 对齐的宽表
app.MapGet("/api/jobs/{id}/csv", (string id, string? format, InferenceJobRunner runner) =>
{
    if (!runner.Jobs.TryGetValue(id, out var j) || j.Probabilities is null)
        return Results.NotFound();

    if (string.Equals(format, "official", StringComparison.OrdinalIgnoreCase))
    {
        // 列序：radar 路径固定用 146 项（含未打分条目的空列）；radar_plus 用其 9 项
        var items = j.Model == "radar_plus"
            ? j.Probabilities.Keys.ToList()
            : runner.TestItems.ToList();
        var official = ResultCsv.Build(items, [new ResultCsv.Row(j.FileName, j.Probabilities)]);
        return Results.File(official, "text/csv",
            fileDownloadName: $"RADAR_infer_results_{j.Id}.csv");
    }

    var lines = new List<string> { "item,probability" };
    lines.AddRange(j.Probabilities.OrderBy(p => p.Key).Select(p => $"{p.Key},{p.Value:0.000000}"));
    var bytes = System.Text.Encoding.UTF8.GetBytes(string.Join("\n", lines));
    return Results.File(bytes, "text/csv", fileDownloadName: $"radar_result_{id}.csv");
});

// ---- 结果 JSON 下载（中文不转义） ----
app.MapGet("/api/jobs/{id}/json", (string id, InferenceJobRunner runner) =>
{
    if (!runner.Jobs.TryGetValue(id, out var j) || j.Probabilities is null)
        return Results.NotFound();

    var payload = new
    {
        j.Id, j.FileName, j.Device, j.Model, j.Variant, j.Status,
        j.ElapsedMs, j.WindowCount, j.Probabilities,
    };
    var opts = new System.Text.Json.JsonSerializerOptions
    {
        WriteIndented = true,
        Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    };
    var json = System.Text.Json.JsonSerializer.SerializeToUtf8Bytes(payload, opts);
    return Results.File(json, "application/json", fileDownloadName: $"radar_result_{id}.json");
});

// ---- 分割掩码透叠图（仅 radar 路径产出） ----
app.MapGet("/api/jobs/{id}/mask.png", (string id, InferenceJobRunner runner) =>
{
    if (!runner.Jobs.TryGetValue(id, out var j)
        || string.IsNullOrEmpty(j.MaskPath) || !File.Exists(j.MaskPath))
        return Results.NotFound();
    return Results.File(j.MaskPath, "image/png");
});

// ---- 批量提交（目录级批处理） ----
app.MapPost("/api/jobs/batch", async (BatchJobRequest req, InferenceJobRunner runner) =>
{
    if (req.Samples is null || req.Samples.Length == 0)
        return Results.BadRequest(new { error = "samples 为空" });

    var cases = new List<(string, string)>();
    foreach (var s in req.Samples)
    {
        string path = Path.Combine(testdataDir, s);
        if (!File.Exists(path))
            return Results.BadRequest(new { error = $"样例不存在: {s}" });
        cases.Add((path, s));
    }
    var jobs = await runner.EnqueueBatchAsync(cases, req.Device ?? "cpu",
        req.Model ?? "radar", req.Variant ?? "merlin_ft");
    return Results.Ok(new { jobIds = jobs.Select(j => j.Id).ToArray(), count = jobs.Count });
});

static object JobView(JobRecord j) => new
{
    j.Id, j.FileName, j.Device, j.Model, j.Variant, j.Status, j.CreatedAt, j.StartedAt,
    j.FinishedAt, j.ElapsedMs, j.WindowCount, j.Probabilities, j.Error,
    HasMask = !string.IsNullOrEmpty(j.MaskPath),
};

app.Run();

record SampleJobRequest(string? Sample, string? Device, string? Model, string? Variant);
record BatchJobRequest(string[]? Samples, string? Device, string? Model, string? Variant);
