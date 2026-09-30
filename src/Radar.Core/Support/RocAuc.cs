namespace Radar.Core.Support;

/// <summary>
/// 二分类 ROC AUC，语义与 <c>sklearn.metrics.roc_auc_score</c> 一致
/// （即 Mann-Whitney U 统计量：<c>AUC = [Σrank(正类) − n₊(n₊+1)/2] / (n₊·n₋)</c>，
/// 相同分值使用平均秩）。
/// </summary>
public static class RocAuc
{
    /// <summary>
    /// 计算 AUC。<paramref name="labels"/> 非 0 视为正类；
    /// 只有单一类别（全正或全负）时无定义，按 sklearn 的行为返回 <see cref="double.NaN"/>。
    /// </summary>
    /// <remarks>调用方需自行过滤无效样本（上游对 label == -1 与缺失标签是跳过，对 NaN 分值置 0）。</remarks>
    public static double Compute(IReadOnlyList<double> scores, IReadOnlyList<int> labels)
    {
        if (scores.Count != labels.Count)
            throw new ArgumentException(
                $"scores 与 labels 长度不一致: {scores.Count} vs {labels.Count}");

        int n = scores.Count;
        long nPos = 0;
        for (int i = 0; i < n; i++)
            if (labels[i] != 0) nPos++;
        long nNeg = n - nPos;
        if (nPos == 0 || nNeg == 0) return double.NaN;

        // 按分值升序稳定排序
        var order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Array.Sort(order, (a, b) =>
        {
            int c = scores[a].CompareTo(scores[b]);
            return c != 0 ? c : a.CompareTo(b);
        });

        // 平均秩（1-based）
        var rank = new double[n];
        int i2 = 0;
        while (i2 < n)
        {
            int j = i2;
            while (j + 1 < n && scores[order[j + 1]] == scores[order[i2]]) j++;
            double avg = ((i2 + 1) + (j + 1)) / 2.0;
            for (int k = i2; k <= j; k++) rank[order[k]] = avg;
            i2 = j + 1;
        }

        double sumPos = 0;
        for (int i3 = 0; i3 < n; i3++)
            if (labels[i3] != 0) sumPos += rank[i3];

        return (sumPos - nPos * (nPos + 1) / 2.0) / ((double)nPos * nNeg);
    }
}
