namespace Radar.Core.Support;

/// <summary>
/// 上游 <c>calc_metrics_merlin_testset.py</c> 的 <c>map_radar_merlin</c>：
/// RAD-CT 条目名 → MERLIN 标签键，用于在下游标签 JSON 中取真值。
/// </summary>
/// <remarks>
/// 注意：其中若干键（如 <c>大肠_粘膜下水肿</c>、<c>心脏_主动脉瓣钙化</c>、<c>肝_肝大</c>、
/// <c>肾_肾积水</c>、<c>胆囊_术后胆囊缺失</c>）<b>不在</b> RAD-CT 的 146 项里 ——
/// 上游 MERLIN 评估用的是另一套 finding 名表（对应 MERLIN 微调权重），
/// 与 <c>inference_demo.py</c> 的 146 项并非同一集合。评估时应对缺失项显式报告而非静默跳过。
/// </remarks>
public static class MerlinDiseases
{
    /// <summary>RAD-CT 条目名 -> MERLIN 标签键。</summary>
    public static readonly IReadOnlyDictionary<string, string> Map =
        new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["主动脉_主动脉瘤"] = "abdominal_aortic_aneurysm",
            ["主动脉_粥样硬化"] = "atherosclerosis",
            ["大肠_粘膜下水肿"] = "submucosal_edema",
            ["大肠_阑尾炎"] = "appendicitis",
            ["小肠_梗阻"] = "bowel_obstruction",
            ["心脏_主动脉瓣钙化"] = "aortic_valve_calcification",
            ["心脏_心影（脏）增大"] = "cardiomegaly",
            ["肝_肝内胆管扩张"] = "biliary_ductal_dilation",
            ["肝_肝大"] = "hepatomegaly",
            ["肝_脂肪肝"] = "hepatic_steatosis",
            ["肺_胸腔积液"] = "pleural_effusion",
            ["肺_膨胀不全"] = "atelectasis",
            ["肾_低密度影"] = "renal_hypodensities",
            ["肾_囊肿"] = "renal_cyst",
            ["肾_肾积水"] = "hydronephrosis",
            ["胆囊_结石"] = "gallstones",
            ["胰腺_萎缩"] = "pancreatic_atrophy",
            ["脾_脾大"] = "splenomegaly",
            ["腰椎_骨折"] = "fracture",
            ["食管_裂孔疝"] = "hiatal_hernia",
            ["胆囊_术后胆囊缺失"] = "surgically_absent_gallbladder",
        };

    /// <summary>上游对 <c>胆囊_术后胆囊缺失</c> 的评估口径：用分割体素数 &lt; 1000 判定，而非概率。</summary>
    public const string GallbladderAbsentItem = "胆囊_术后胆囊缺失";

    /// <summary>上游阈值。</summary>
    public const int GallbladderAbsentThreshold = 1000;
}
