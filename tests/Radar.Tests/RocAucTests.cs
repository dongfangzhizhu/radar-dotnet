using Radar.Core.Support;
using Xunit.Abstractions;

namespace Radar.Tests;

/// <summary>
/// AUC 评估（实测差距 #7 的核心算子）。语义必须与 <c>sklearn.metrics.roc_auc_score</c> 一致。
/// <para>
/// 预言机说明：被测实现走「排序 + 平均秩」（Mann-Whitney U 的秩形式，O(n log n)），
/// 测试端用<b>朴素两两配对</b>（O(n²)，平局计 0.5）作为独立算法交叉验证 ——
/// 两条完全不同的路径给出同一结果，才算真的对齐 sklearn。
/// </para>
/// </summary>
public class RocAucTests(ITestOutputHelper output)
{
    /// <summary>朴素 AUC：直接数所有 (正, 负) 配对的一致/平局，不依赖任何排序或秩。</summary>
    private static double NaiveAuc(IReadOnlyList<double> scores, IReadOnlyList<int> labels)
    {
        double sum = 0;
        long pair = 0;
        for (int i = 0; i < scores.Count; i++)
        {
            if (labels[i] == 0) continue;
            for (int j = 0; j < scores.Count; j++)
            {
                if (labels[j] != 0) continue;
                pair++;
                if (scores[i] > scores[j]) sum += 1;
                else if (scores[i] == scores[j]) sum += 0.5;
            }
        }
        return pair == 0 ? double.NaN : sum / pair;
    }

    private void CrossCheck(IReadOnlyList<double> scores, IReadOnlyList<int> labels)
    {
        double fast = RocAuc.Compute(scores, labels);
        double slow = NaiveAuc(scores, labels);
        output.WriteLine($"rank 法={fast:F12}  朴素配对={slow:F12}");
        Assert.Equal(slow, fast, 12);
    }

    [Fact]
    public void PerfectSeparation_IsOne()
    {
        CrossCheck([0.1, 0.2, 0.3, 0.4], [0, 0, 1, 1]);
        Assert.Equal(1.0, RocAuc.Compute([0.1, 0.2, 0.3, 0.4], [0, 0, 1, 1]));
    }

    [Fact]
    public void FullyInverted_IsZero()
    {
        CrossCheck([0.4, 0.3, 0.2, 0.1], [0, 0, 1, 1]);
        Assert.Equal(0.0, RocAuc.Compute([0.4, 0.3, 0.2, 0.1], [0, 0, 1, 1]));
    }

    [Fact]
    public void HandComputed_MixedOrder()
    {
        // sklearn: roc_auc_score([0,0,1,1], [0.1,0.4,0.35,0.8]) = 0.75
        var s = new double[] { 0.1, 0.4, 0.35, 0.8 };
        var l = new[] { 0, 0, 1, 1 };
        CrossCheck(s, l);
        Assert.Equal(0.75, RocAuc.Compute(s, l), 12);
    }

    [Fact]
    public void AllTied_IsHalf()
    {
        var s = new double[] { 0.5, 0.5, 0.5, 0.5 };
        var l = new[] { 0, 0, 1, 1 };
        CrossCheck(s, l);
        Assert.Equal(0.5, RocAuc.Compute(s, l), 12);   // 平均秩：全平局 -> 0.5
    }

    [Fact]
    public void PartialTies_AverageRankIsUsed()
    {
        // 2 正 3 负，两个平局值横跨正负类，专门考平均秩
        var s = new double[] { 0.2, 0.7, 0.7, 0.9, 0.4 };
        var l = new[] { 0, 0, 1, 1, 0 };
        CrossCheck(s, l);
    }

    [Fact]
    public void Labels_AnyNonZeroCountsAsPositive()
    {
        // MERLIN 标签里 1/2/3 都可能当正类（如分级），实现必须只判 != 0
        var s = new double[] { 0.1, 0.8, 0.6, 0.3 };
        double a = RocAuc.Compute(s, [0, 1, 2, 0]);
        double b = RocAuc.Compute(s, [0, 1, 1, 0]);
        Assert.Equal(b, a, 12);
        CrossCheck(s, [0, 1, 2, 0]);
    }

    [Fact]
    public void SingleClass_IsNaN_LikeSklearn()
    {
        Assert.True(double.IsNaN(RocAuc.Compute([0.1, 0.2, 0.3], [1, 1, 1])));
        Assert.True(double.IsNaN(RocAuc.Compute([0.1, 0.2, 0.3], [0, 0, 0])));
        Assert.True(double.IsNaN(RocAuc.Compute([], [])));
    }

    [Fact]
    public void SinglePositiveAmongNegatives()
    {
        // 1 正 3 负，正中分值最高 -> 1.0；居中 -> 取决于位次
        Assert.Equal(1.0, RocAuc.Compute([0.1, 0.2, 0.3, 0.9], [0, 0, 0, 1]));
        CrossCheck([0.1, 0.9, 0.3, 0.5], [0, 0, 0, 1]);
    }

    [Fact]
    public void LengthMismatch_Throws()
        => Assert.Throws<ArgumentException>(() => RocAuc.Compute([0.1, 0.2], [0]));

    [Fact]
    public void LargeRandom_MatchesBruteForce()
    {
        var rnd = new Random(42);
        for (int trial = 0; trial < 5; trial++)
        {
            int n = 60;
            var s = new double[n];
            var l = new int[n];
            for (int i = 0; i < n; i++) { s[i] = Math.Round(rnd.NextDouble(), 1); l[i] = rnd.Next(2); }
            // 保证两类都存在
            l[0] = 1; l[1] = 0;
            CrossCheck(s, l);
        }
    }

    [Fact]
    public void OrderOfInputDoesNotMatter()
    {
        var s = new double[] { 0.2, 0.7, 0.7, 0.9, 0.4 };
        var l = new[] { 0, 0, 1, 1, 0 };
        double a = RocAuc.Compute(s, l);

        var idx = Enumerable.Range(0, s.Length).OrderByDescending(i => i).ToArray();
        double b = RocAuc.Compute(idx.Select(i => s[i]).ToArray(), idx.Select(i => l[i]).ToArray());
        Assert.Equal(a, b, 12);
    }
}
