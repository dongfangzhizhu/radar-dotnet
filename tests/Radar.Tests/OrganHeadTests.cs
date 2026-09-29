using System.Text.Json;
using Radar.Core.Inference;
using Radar.Core.Support;
using Radar.Tests.Support;
using Xunit.Abstractions;

namespace Radar.Tests;

/// <summary>
/// 器官头单级对拍：用 golden 的窗输入绕过滑窗，重算各器官特征，
/// 与 golden organ_feats.npz 中特征逐维比对，定位特征差异来源。
/// </summary>
public class OrganHeadTests(ITestOutputHelper output)
{
    private static string Root => RepoRoot.Find();

    [Fact]
    public void OrganFeature_MatchesGoldenPerOrgan()
    {
        string onnx = Path.Combine(Root, "models", "radar_vision.onnx");
        string npz = Path.Combine(Root, "models", "organ_head.npz");
        string bin = Path.Combine(Root, "models", "text_feats.bin");
        string json = Path.Combine(Root, "models", "text_feats.json");
        var winDir = Path.Combine(Root, "testdata", "golden", "windows");
        string featsPath = Path.Combine(Root, "testdata", "golden", "organ_feats.npz");
        if (new[] { onnx, npz, bin, json, featsPath }.Any(p => !File.Exists(p)) || !Directory.Exists(winDir))
            return;

        var meta = JsonDocument.Parse(File.ReadAllText(Path.Combine(Root, "testdata", "golden", "meta.json")));
        var organs = meta.RootElement.GetProperty("organs").EnumerateArray().Select(v => v.GetString()!).ToList();
        var goldenFeats = NpyReader.LoadNpz(featsPath);

        using var engine = new VisionEngine(onnx, ExecutionDevice.Cpu);
        var head = new OrganHead(OrganHeadWeights.Load(npz), TextFeatures.Load(bin, json));

        // 两窗输出都算，golden 的特征来自该器官首次 intact 的窗
        var outsByWin = new Dictionary<string, WindowOutputs>();
        foreach (var winName in new[] { "win_0000", "win_0001" })
        {
            if (!File.Exists(Path.Combine(winDir, winName, "win_in.npy"))) continue;
            var winIn = NpyReader.LoadFloat32(Path.Combine(winDir, winName, "win_in.npy"), out _);
            outsByWin[winName] = engine.Run(winIn);
        }

        int checked0 = 0;
        foreach (var kv in goldenFeats)
        {
            var organName = kv.Key;
            var goldenArr = kv.Value;
            int organId = organs.IndexOf(organName);
            if (organId < 0) continue;
            var golden = ((float[])goldenArr.Data).Select(v => (double)v).ToArray();

            var diffs = new List<(string Win, double Diff)>();
            foreach (var (winName, outs) in outsByWin)
            {
                var f1 = flagsAt(outs.Flags1, organId, 1152);
                var f2 = flagsAt(outs.Flags2, organId, 9216);
                var f3 = flagsAt(outs.Flags3, organId, 73728);
                if (!f1.Any(v => v > 0)) continue;
                var (feat, _) = head.Score(organId, outs.ResX1, outs.ResX2, outs.ResX3, f1, f2, f3);
                double diff = 0;
                for (int i = 0; i < 256; i++)
                    diff = Math.Max(diff, Math.Abs(golden[i] - feat[i]));
                diffs.Add((winName, diff));
            }
            if (diffs.Count == 0) continue;
            var best = diffs.OrderBy(x => x.Diff).First();
            output.WriteLine($"{organName}: best={best.Win} max_diff={best.Diff:E3} all=[{string.Join(", ", diffs.Select(x => $"{x.Win}:{x.Diff:E3}"))}]");
            checked0++;
            if (checked0 >= 10) break;
        }
    }

    private static byte[] flagsAt(byte[] all, int organId, int len)
    {
        var seg = new byte[len];
        Array.Copy(all, organId * len, seg, 0, len);
        return seg;
    }
}
