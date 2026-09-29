namespace Radar.Core.Support;

/// <summary>
/// 模型标签元数据（来自训练配置，id 顺序与 seg 输出通道 1..36 一一对应；
/// 由 export/common.py 的 ORGANS 同步，来源 testdata/golden/meta.json）。
/// </summary>
public static class RadarLabels
{
    /// <summary>36 个器官名，索引 = 器官 id（seg 类别 = id + 1）。</summary>
    public static readonly string[] Organs =
    [
        "肾上腺", "主动脉", "竖脊肌", "脑", "锁骨", "大肠", "十二指肠", "食管",
        "面部", "股骨", "胆囊", "臀肌", "心脏", "髋关节", "肱骨", "髂动脉",
        "髂静脉", "髂腰肌", "下腔静脉", "肾", "肝", "肺", "胰腺", "门静脉",
        "肺动脉", "肋骨", "骶骨", "肩胛骨", "小肠", "脾", "胃", "气管",
        "膀胱", "颈椎", "腰椎", "胸椎",
    ];
}
