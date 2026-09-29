using System.Numerics;

namespace Radar.Core.Inference;

/// <summary>
/// 器官轻头：单 query 4 头 attention + per-organ 投影 + L2 归一化 + 文本相似度打分。
/// 与 export/golden_dump.py 的 mha() 逐式对应（fp32）。
/// </summary>
public sealed class OrganHead
{
    private const int Dim = 256;
    private const int Heads = 4;
    private const int HeadDim = 64;

    private readonly OrganHeadWeights _w;
    private readonly TextFeatures _text;

    public OrganHead(OrganHeadWeights weights, TextFeatures text)
    {
        _w = weights;
        _text = text;
    }

    public TextFeatures Text => _text;

    /// <summary>
    /// 对一个器官打分：query (1,256) 单 token，对 key tokens (K,256) 做 attention
    /// → out_proj → vision_proj → L2 norm → 与全部 146 项 text_feat 点积 / temp → softmax。
    /// 返回 (L2 归一化特征, {itemIndex: 阳性概率})。
    /// </summary>
    public (float[] Feature, Dictionary<int, float> Scores) Score(int organId, float[] resX1, float[] resX2, float[] resX3,
        ReadOnlySpan<byte> flags1, ReadOnlySpan<byte> flags2, ReadOnlySpan<byte> flags3)
    {
        // query = query_tokens[organId]
        var query = new float[Dim];
        Array.Copy(_w.QueryTokens, organId * Dim, query, 0, Dim);

        // key/value = cat(选中 token)（golden_dump 顺序：x1, x2, x3）
        var k1 = Gather(resX1, flags1, 1152);
        var k2 = Gather(resX2, flags2, 9216);
        var k3 = Gather(resX3, flags3, 73728);
        int k = k1.Length / Dim + k2.Length / Dim + k3.Length / Dim;
        var key = new float[k * Dim];
        Array.Copy(k1, 0, key, 0, k1.Length);
        Array.Copy(k2, 0, key, k1.Length, k2.Length);
        Array.Copy(k3, 0, key, k1.Length + k2.Length, k3.Length);

        var attended = Mha(query, key, k);          // (1,256)
        var projW = new float[Dim * Dim];
        var projB = new float[Dim];
        Array.Copy(_w.ProjW, organId * Dim * Dim, projW, 0, projW.Length);
        Array.Copy(_w.ProjB, organId * Dim, projB, 0, projB.Length);
        var projected = Linear(attended, 1, projW, projB);   // (1,256)
        L2Normalize(projected);

        // 与全部文本特征打分：feat·row / temp → softmax(2)
        var scores = new Dictionary<int, float>(_text.Count);
        for (int i = 0; i < _text.Count; i++)
        {
            var seg = _text.GetItem(i); // (2,256)
            float neg = Dot(projected, seg, 0);
            float pos = Dot(projected, seg, Dim);
            float l0 = neg / _w.Temp, l1 = pos / _w.Temp;
            float m = MathF.Max(l0, l1);
            float e0 = MathF.Exp(l0 - m), e1 = MathF.Exp(l1 - m);
            scores[i] = e1 / (e0 + e1);
        }
        return (projected, scores);
    }

    /// <summary>单 query 多头 attention（4 头，fp32），逐式对应 torch nn.MultiheadAttention。</summary>
    private float[] Mha(float[] query, float[] key, int kTokens)
    {
        // in_proj: Q=W[0:256], K=W[256:512], V=W[512:768]
        var q = Linear(query, 1, new ArraySegment<float>(_w.InProjW, 0, Dim * Dim).Array!,
            new ArraySegment<float>(_w.InProjB, 0, Dim).Array!);
        var kv = Linear(key, kTokens,
            _w.InProjW.AsSpan(Dim * Dim, 2 * Dim * Dim).ToArray(),
            _w.InProjB.AsSpan(Dim, 2 * Dim).ToArray(),
            outDim: 2 * Dim); // K(256) + V(256) 合并输出
        // kv (kTokens, 512)：行内列拼接 [K_j(256) | V_j(256)]——torch F.linear 输出布局
        var kMat = new float[kTokens * Dim];
        var vMat = new float[kTokens * Dim];
        for (int j = 0; j < kTokens; j++)
        {
            Array.Copy(kv, j * 2 * Dim, kMat, j * Dim, Dim);
            Array.Copy(kv, j * 2 * Dim + Dim, vMat, j * Dim, Dim);
        }

        var outp = new float[Dim];
        for (int h = 0; h < Heads; h++)
        {
            int off = h * HeadDim;
            // attn(j) = softmax_j(q·k_j / sqrt(64))
            var attn = new float[kTokens];
            float maxA = float.NegativeInfinity;
            for (int j = 0; j < kTokens; j++)
            {
                float s = 0f;
                for (int d = 0; d < HeadDim; d++)
                    s += q[off + d] * kMat[j * Dim + off + d];
                s /= MathF.Sqrt(HeadDim);
                attn[j] = s;
                if (s > maxA) maxA = s;
            }
            float sum = 0f;
            for (int j = 0; j < kTokens; j++) { attn[j] = MathF.Exp(attn[j] - maxA); sum += attn[j]; }
            for (int j = 0; j < kTokens; j++) attn[j] /= sum;

            for (int d = 0; d < HeadDim; d++)
            {
                float acc = 0f;
                for (int j = 0; j < kTokens; j++)
                    acc += attn[j] * vMat[j * Dim + off + d];
                outp[off + d] = acc;
            }
        }
        // out_proj
        return Linear(outp, 1, _w.OutProjW, _w.OutProjB);
    }

    /// <summary>y = x @ Wᵀ + b（torch F.linear 语义），x (rows, inDim)，W (outDim, inDim)。</summary>
    internal static float[] Linear(float[] x, int rows, float[] weight, float[] bias, int inDim = Dim, int outDim = Dim)
    {
        var y = new float[rows * outDim];
        for (int r = 0; r < rows; r++)
        {
            int xb = r * inDim, yb = r * outDim;
            for (int o = 0; o < outDim; o++)
            {
                float acc = bias[o];
                int wb = o * inDim;
                for (int d = 0; d < inDim; d++)
                    acc += x[xb + d] * weight[wb + d];
                y[yb + o] = acc;
            }
        }
        return y;
    }

    private static float[] Gather(float[] tokens, ReadOnlySpan<byte> flags, int tokenCount)
    {
        int sel = 0;
        for (int i = 0; i < tokenCount; i++) sel += flags[i];
        var result = new float[sel * Dim];
        int w = 0;
        for (int i = 0; i < tokenCount; i++)
        {
            if (flags[i] == 0) continue;
            Array.Copy(tokens, i * Dim, result, w * Dim, Dim);
            w++;
        }
        return result;
    }

    private static void L2Normalize(float[] v)
    {
        float sum = 0f;
        foreach (float x in v) sum += x * x;
        float norm = MathF.Sqrt(sum);
        if (norm > 0)
            for (int i = 0; i < v.Length; i++) v[i] /= norm;
    }

    private static float Dot(float[] a, float[] b, int bOffset)
    {
        float s = 0f;
        for (int i = 0; i < a.Length; i++) s += a[i] * b[bOffset + i];
        return s;
    }
}
