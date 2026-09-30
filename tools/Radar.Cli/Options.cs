namespace Radar.Cli;

/// <summary>极简命令行解析：<c>--key value</c> 与 <c>--flag</c>（无值即视为 "true"）。</summary>
internal sealed class Options
{
    private readonly Dictionary<string, string> _map = new(StringComparer.OrdinalIgnoreCase);

    public static Options Parse(string[] argv, int start)
    {
        var o = new Options();
        for (int i = start; i < argv.Length; i++)
        {
            string a = argv[i];
            if (!a.StartsWith("--", StringComparison.Ordinal))
                throw new ArgumentException($"无法识别的参数: {a}（选项须以 -- 开头）");

            string key = a[2..];
            int eq = key.IndexOf('=');
            if (eq >= 0)
            {
                o._map[key[..eq]] = key[(eq + 1)..];
                continue;
            }
            if (i + 1 < argv.Length && !argv[i + 1].StartsWith("--", StringComparison.Ordinal))
                o._map[key] = argv[++i];
            else
                o._map[key] = "true";
        }
        return o;
    }

    public string? Get(string key) => _map.GetValueOrDefault(key);

    public string GetOr(string key, string fallback) => _map.TryGetValue(key, out var v) ? v : fallback;

    public int GetInt(string key, int fallback)
        => _map.TryGetValue(key, out var v) && int.TryParse(v, out var n) ? n : fallback;

    public bool Flag(string key) => _map.TryGetValue(key, out var v) && v != "false";

    public string[] GetList(string key)
    {
        if (!_map.TryGetValue(key, out var v)) return [];
        return v.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
    }
}
