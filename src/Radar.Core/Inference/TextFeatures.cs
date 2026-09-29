using System.Text.Json;

namespace Radar.Core.Inference;

/// <summary>
/// 预计算文本特征（text_feats.bin + text_feats.json，由 export/export_text_feats.py 生成）。
/// 146 项「器官_病灶」，每项 (2,256)：第 0 行负向 prompt、第 1 行正向 prompt；
/// 打分取 softmax 后 index 1（阳性概率）。
/// </summary>
public sealed class TextFeatures
{
    private readonly float[] _data;   // count * rows * dim 连续
    private readonly Dictionary<string, int> _index;
    public IReadOnlyList<string> Items { get; }
    public int Count => Items.Count;
    public int Dim { get; }
    public int RowsPerItem { get; }

    private TextFeatures(float[] data, List<string> items, int dim, int rows)
    {
        _data = data;
        Items = items;
        Dim = dim;
        RowsPerItem = rows;
        _index = items.Select((name, i) => (name, i)).ToDictionary(t => t.name, t => t.i);
    }

    public static TextFeatures Load(string binPath, string jsonPath)
    {
        var meta = JsonDocument.Parse(File.ReadAllText(jsonPath));
        int dim = meta.RootElement.GetProperty("dim").GetInt32();
        int rows = meta.RootElement.GetProperty("rows_per_item").GetInt32();
        var items = meta.RootElement.GetProperty("items").EnumerateArray()
            .Select(i => i.GetProperty("name").GetString()!)
            .ToList();
        var data = File.ReadAllBytes(binPath);
        var floats = new float[data.Length / 4];
        Buffer.BlockCopy(data, 0, floats, 0, data.Length);
        return new TextFeatures(floats, items, dim, rows);
    }

    /// <summary>取第 i 项的全部特征（rows*dim 连续）。</summary>
    public float[] GetItem(int index)
    {
        var seg = new float[RowsPerItem * Dim];
        Buffer.BlockCopy(_data, index * RowsPerItem * Dim * 4, seg, 0, seg.Length * 4);
        return seg;
    }

    public int IndexOf(string name) => _index.TryGetValue(name, out var i) ? i : -1;
}

/// <summary>
/// 器官轻头权重（organ_head.npz，由 export/export_weights.py 生成）：
/// attention.in_proj (768,256) 行序 Q[0:256] K[256:512] V[512:768]、
/// out_proj (256,256)、vision_projs.{0..35} (256,256)、query_tokens (36,256)、temp 标量。
/// </summary>
public sealed class OrganHeadWeights
{
    public float[] InProjW { get; }   // (768,256)
    public float[] InProjB { get; }   // (768,)
    public float[] OutProjW { get; }  // (256,256)
    public float[] OutProjB { get; }  // (256,)
    public float[] QueryTokens { get; } // (36,256)
    public float[] ProjW { get; }     // 36 * (256,256) 连续
    public float[] ProjB { get; }     // 36 * (256,)
    public float Temp { get; }
    public int OrganCount { get; } = 36;

    private OrganHeadWeights(float[] w, float[] b, float[] ow, float[] ob, float[] qt, float[] pw, float[] pb, float temp)
    {
        InProjW = w; InProjB = b; OutProjW = ow; OutProjB = ob;
        QueryTokens = qt; ProjW = pw; ProjB = pb; Temp = temp;
    }

    public static OrganHeadWeights Load(string npzPath)
    {
        var z = Radar.Core.Support.NpyReader.LoadNpz(npzPath);
        var w = (float[])z["attention.in_proj_weight"].Data;
        var b = (float[])z["attention.in_proj_bias"].Data;
        var ow = (float[])z["attention.out_proj.weight"].Data;
        var ob = (float[])z["attention.out_proj.bias"].Data;
        var qt = (float[])z["query_tokens"].Data;
        var pw = new float[36 * 256 * 256];
        var pb = new float[36 * 256];
        for (int i = 0; i < 36; i++)
        {
            var wi = (float[])z[$"vision_projs.{i}.weight"].Data;
            var bi = (float[])z[$"vision_projs.{i}.bias"].Data;
            Array.Copy(wi, 0, pw, i * 256 * 256, wi.Length);
            Array.Copy(bi, 0, pb, i * 256, bi.Length);
        }
        var temp = ((float[])z["temp"].Data)[0];
        return new OrganHeadWeights(w, b, ow, ob, qt, pw, pb, temp);
    }
}
