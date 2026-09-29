using System.Text.Json.Serialization;
using Radar.Backend;

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
    o.SerializerOptions.Converters.Add(new JsonStringEnumConverter())); // enum 序列化为字符串（前端依赖）
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

// ---- 样例列表（testdata 下 NIfTI 文件） ----
app.MapGet("/api/samples", () =>
{
    if (!Directory.Exists(testdataDir)) return Results.Json(Array.Empty<object>());
    var samples = Directory.GetFiles(testdataDir, "*.nii.gz")
        .Concat(Directory.GetFiles(testdataDir, "*.nii"))
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
    var job = await runner.EnqueueAsync(path, req.Sample, req.Device ?? "cpu");
    return Results.Ok(new { jobId = job.Id });
});

// ---- 上传建任务 ----
app.MapPost("/api/jobs/upload", async (IFormFile file, string? device, InferenceJobRunner runner) =>
{
    if (file.Length == 0) return Results.BadRequest(new { error = "空文件" });
    if (!file.FileName.EndsWith(".nii") && !file.FileName.EndsWith(".nii.gz"))
        return Results.BadRequest(new { error = "仅支持 .nii / .nii.gz" });

    string uploads = Path.Combine(repoRoot, "uploads");
    Directory.CreateDirectory(uploads);
    string path = Path.Combine(uploads, $"{Guid.NewGuid():N}_{file.FileName}");
    await using var fs = File.Create(path);
    await file.CopyToAsync(fs);

    var job = await runner.EnqueueAsync(path, file.FileName, device ?? "cpu");
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

// ---- 结果 CSV 下载（条目,阳性概率） ----
app.MapGet("/api/jobs/{id}/csv", (string id, InferenceJobRunner runner) =>
{
    if (!runner.Jobs.TryGetValue(id, out var j) || j.Probabilities is null)
        return Results.NotFound();
    var lines = new List<string> { "item,probability" };
    lines.AddRange(j.Probabilities.OrderBy(p => p.Key).Select(p => $"{p.Key},{p.Value:0.000000}"));
    var bytes = System.Text.Encoding.UTF8.GetBytes(string.Join("\n", lines));
    return Results.File(bytes, "text/csv", fileDownloadName: $"radar_result_{id}.csv");
});

static object JobView(JobRecord j) => new
{
    j.Id, j.FileName, j.Device, j.Status, j.CreatedAt, j.StartedAt,
    j.FinishedAt, j.ElapsedMs, j.WindowCount, j.Probabilities, j.Error,
};

app.Run();

record SampleJobRequest(string? Sample, string? Device);
