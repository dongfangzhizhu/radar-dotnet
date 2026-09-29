using System.Text.Json;
using Radar.Core.Support;

namespace Radar.Core.Inference;

/// <summary>
/// RADAR+ whole-head 权重（plus_head*.npz，由 export/export_plus.py 生成）：
/// attention_whole.in_proj (768,256) 行序 Q[0:256] K[256:512] V[512:768]、
/// out_proj (256,256)、vision_projs_whole (256,256)、query_tokens_whole (1,256)、temp 标量。
/// </summary>
public sealed class PlusHeadWeights
{
    public float[] InProjW { get; }   // (768,256)
    public float[] InProjB { get; }   // (768,)
    public float[] OutProjW { get; }  // (256,256)
    public float[] OutProjB { get; }  // (256,)
    public float[] ProjW { get; }     // (256,256)
    public float[] ProjB { get; }     // (256,)
    public float[] Query { get; }     // (256)（去掉了 (1,256) 的 batch 维）
    public float Temp { get; }

    private PlusHeadWeights(float[] ipw, float[] ipb, float[] ow, float[] ob,
        float[] pw, float[] pb, float[] query, float temp)
    {
        InProjW = ipw; InProjB = ipb; OutProjW = ow; OutProjB = ob;
        ProjW = pw; ProjB = pb; Query = query; Temp = temp;
    }

    public static PlusHeadWeights Load(string npzPath)
    {
        var z = NpyReader.LoadNpz(npzPath);
        var w = (float[])z["attention_whole.in_proj_weight"].Data;
        var b = (float[])z["attention_whole.in_proj_bias"].Data;
        var ow = (float[])z["attention_whole.out_proj.weight"].Data;
        var ob = (float[])z["attention_whole.out_proj.bias"].Data;
        var pw = (float[])z["vision_projs_whole.weight"].Data;
        var pb = (float[])z["vision_projs_whole.bias"].Data;
        var qt = (float[])z["query_tokens_whole"].Data;
        var temp = ((float[])z["temp"].Data)[0];
        if (w.Length != 768 * 256 || qt.Length != 256)
            throw new InvalidDataException($"plus_head npz 形状不符: in_proj={w.Length} query={qt.Length}");
        return new PlusHeadWeights(w, b, ow, ob, pw, pb, qt, temp);
    }
}

/// <summary>
/// RADAR+ 文本特征（plus_text_feats*.bin/.json，由 export/export_plus.py 生成）。
/// 9 疾病 × (num_positive + num_negative) 条 prompt 特征（已 L2 归一，256 维连续）。
/// json 记录每病的 num_positive / start / count，用于 pos/neg 分组均值。
/// </summary>
public sealed class PlusTextFeatures
{
    public sealed record Disease(string Name, int NumPositive, int Start, int Count);

    private readonly float[] _data;   // count * 256 连续
    public IReadOnlyList<Disease> Diseases { get; }
    public int Dim { get; }
    public int TotalPrompts => _data.Length / Dim;

    private PlusTextFeatures(float[] data, List<Disease> diseases, int dim)
    {
        _data = data;
        Diseases = diseases;
        Dim = dim;
    }

    public static PlusTextFeatures Load(string binPath, string jsonPath)
    {
        var meta = JsonDocument.Parse(File.ReadAllText(jsonPath));
        int dim = meta.RootElement.GetProperty("dim").GetInt32();
        var diseases = new List<Disease>();
        foreach (var d in meta.RootElement.GetProperty("diseases").EnumerateArray())
        {
            diseases.Add(new Disease(
                d.GetProperty("name").GetString()!,
                d.GetProperty("num_positive").GetInt32(),
                d.GetProperty("start").GetInt32(),
                d.GetProperty("count").GetInt32()));
        }
        var data = File.ReadAllBytes(binPath);
        var floats = new float[data.Length / 4];
        Buffer.BlockCopy(data, 0, floats, 0, data.Length);
        return new PlusTextFeatures(floats, diseases, dim);
    }

    /// <summary>取第 i 条 prompt 特征的偏移（float 数）。</summary>
    internal int Offset(int promptIndex) => promptIndex * Dim;

    internal float[] Data => _data;
}

/// <summary>
/// RADAR+ whole-image 打分头：query(1 token) 对全部 84096 token（x1|x2|x3 拼接）
/// 4 头 attention → vision_projs_whole → L2 norm → 与 9 疾病多 prompt 文本特征打分。
/// 复刻 infer_merlin_whole.py L670-693。
///
/// 数学优化（与朴素实现等价的线性恒等式）：
///   score_j = q_h·(W_k x_j + b_k)[off:off+64] = (W_kᵀ q_h)·x_j + q_h·b_k[off:off+64]
///   out_h   = Σ_j attn_j (W_v x_j + b_v)[off:off+64] = W_v[off:off+64,:]·(Σ_j attn_j x_j) + ...
/// 把对 84096 token 的 512 维线性展开（~11G 乘加）降为两次 token 遍历（~0.4G）。
/// </summary>
public sealed class PlusHead
{
    private const int Dim = 256;
    private const int Heads = 4;
    private const int HeadDim = 64;
    public const int TokenCount = 1152 + 9216 + 73728;  // 84096

    private readonly PlusHeadWeights _w;
    private readonly PlusTextFeatures _text;

    public PlusHead(PlusHeadWeights weights, PlusTextFeatures text)
    {
        _w = weights;
        _text = text;
    }

    public PlusTextFeatures Text => _text;

    public sealed record Result(float[] Feature, Dictionary<string, float> Probabilities);

    public Result Score(float[] x1, float[] x2, float[] x3)
    {
        if (x1.Length != 1152 * Dim || x2.Length != 9216 * Dim || x3.Length != 73728 * Dim)
            throw new InvalidDataException(
                $"embeds 尺寸不符: {x1.Length},{x2.Length},{x3.Length}");

        // ---- Q：只算各 head 用到的 64 维 ----
        // q = W_q·query + b_q；q_h[d] = Σ_k W_q[off+d,k]·query[k] + b_q[off+d]
        var qAll = new float[Dim];
        for (int o = 0; o < Dim; o++)
        {
            float acc = _w.InProjB[o];
            int row = o * Dim;
            for (int k = 0; k < Dim; k++) acc += _w.InProjW[row + k] * _w.Query[k];
            qAll[o] = acc;
        }

        var scores = new float[TokenCount];
        var attn = new float[TokenCount];
        var attended = new float[Dim];   // 各 head 输出拼接

        for (int h = 0; h < Heads; h++)
        {
            int off = h * HeadDim;

            // ---- scores_j = (W_kᵀ q_h)·x_j + q_h·b_k[off:off+64] ----
            var u = new float[Dim];               // W_kᵀ q_h
            for (int d = 0; d < HeadDim; d++)
            {
                float qhd = qAll[off + d];
                if (qhd == 0f) continue;
                long row = (Dim + off + d) * Dim; // K 段行号 256..511
                for (int k = 0; k < Dim; k++) u[k] += _w.InProjW[row + k] * qhd;
            }
            float bK = 0f;
            for (int d = 0; d < HeadDim; d++) bK += qAll[off + d] * _w.InProjB[Dim + off + d];

            float maxA = float.NegativeInfinity;
            AccumScores(x1, 1152, u, bK, scores, 0, ref maxA);
            AccumScores(x2, 9216, u, bK, scores, 1152, ref maxA);
            AccumScores(x3, 73728, u, bK, scores, 1152 + 9216, ref maxA);

            // ---- softmax（数值稳定）----
            float sum = 0f;
            for (int j = 0; j < TokenCount; j++) { attn[j] = MathF.Exp(scores[j] - maxA); sum += attn[j]; }
            float inv = 1f / sum;

            // ---- s = Σ_j attn_j·x_j，再 out_h = W_v[off:off+64,:]·s + b_v[off:off+64] ----
            var s = new float[Dim];
            WeightedSum(x1, 1152, attn, inv, s, 0);
            WeightedSum(x2, 9216, attn, inv, s, 1152);
            WeightedSum(x3, 73728, attn, inv, s, 1152 + 9216);
            for (int d = 0; d < HeadDim; d++)
            {
                float acc = _w.InProjB[2 * Dim + off + d];
                long row = (2 * Dim + off + d) * Dim;
                for (int k = 0; k < Dim; k++) acc += _w.InProjW[row + k] * s[k];
                attended[off + d] = acc;
            }
        }

        // ---- out_proj → vision_proj → L2 norm ----
        var afterOutProj = OrganHead.Linear(attended, 1, _w.OutProjW, _w.OutProjB);
        var feat = OrganHead.Linear(afterOutProj, 1, _w.ProjW, _w.ProjB);
        float n2 = 0f;
        foreach (float v in feat) n2 += v * v;
        float norm = MathF.Sqrt(n2);
        if (norm > 0)
            for (int i = 0; i < feat.Length; i++) feat[i] /= norm;

        // ---- 9 疾病打分：pos/neg 分组均值 → softmax([neg,pos]) ----
        var probs = new Dictionary<string, float>(_text.Diseases.Count);
        foreach (var dis in _text.Diseases)
        {
            float posSum = 0f, negSum = 0f;
            var data = _text.Data;
            for (int i = 0; i < dis.Count; i++)
            {
                float sim = 0f;
                long off2 = _text.Offset(dis.Start + i);
                for (int k = 0; k < Dim; k++) sim += feat[k] * data[off2 + k];
                sim /= _w.Temp;
                if (i < dis.NumPositive) posSum += sim;
                else negSum += sim;
            }
            float posMean = dis.NumPositive > 0 ? posSum / dis.NumPositive : 0f;
            int negCount = dis.Count - dis.NumPositive;
            float negMean = negCount > 0 ? negSum / negCount : 0f;
            float m = MathF.Max(posMean, negMean);
            float ePos = MathF.Exp(posMean - m), eNeg = MathF.Exp(negMean - m);
            probs[dis.Name] = ePos / (ePos + eNeg);
        }
        return new Result(feat, probs);
    }

    private static void AccumScores(float[] x, int tokens, float[] u, float bK,
        float[] scores, int idxStart, ref float maxA)
    {
        float scale = 1f / MathF.Sqrt(HeadDim);   // 与 nn.MultiheadAttention 一致：q·k / √head_dim
        for (int j = 0; j < tokens; j++)
        {
            float s2 = bK;
            long b = (long)j * Dim;
            for (int k = 0; k < Dim; k++) s2 += u[k] * x[b + k];
            s2 *= scale;
            scores[idxStart + j] = s2;
            if (s2 > maxA) maxA = s2;
        }
    }

    private static void WeightedSum(float[] x, int tokens, float[] attn, float inv, float[] s, int idxStart)
    {
        for (int j = 0; j < tokens; j++)
        {
            float a = attn[idxStart + j] * inv;
            long b = (long)j * Dim;
            for (int k = 0; k < Dim; k++) s[k] += a * x[b + k];
        }
    }
}
