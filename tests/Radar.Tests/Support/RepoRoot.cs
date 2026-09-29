namespace Radar.Tests.Support;

/// <summary>定位仓库根（含 PLAN.md 的目录）。</summary>
public static class RepoRoot
{
    public static string Find()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null && !File.Exists(Path.Combine(dir.FullName, "PLAN.md")))
            dir = dir.Parent;
        if (dir == null)
            throw new InvalidOperationException("未找到仓库根（PLAN.md）");
        return dir.FullName;
    }
}
