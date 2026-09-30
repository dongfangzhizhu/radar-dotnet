#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""前端渲染回归：把后端「实际服务出来」的 <script> 抽出来，在 Node vm 里配 DOM shim 跑真实 payload。

为什么不用浏览器/静态检查：
  - 编译器不检查裸 JS，模板字符串里的字段名拼错只有在运行时才暴露（上一轮 undefined bug 就是这么漏的）；
  - 必须校验「被服务的那份脚本」，而不是磁盘上的源文件 —— 否则静态资源没更新会假绿。

流程：启动后端 -> 探活(/api/samples 200) -> GET / 抓 script -> GET 真实 /api/jobs
      -> Node vm 断言 -> 拆服务。全过程在单次调用内完成（后台进程不跨工具调用存活）。
"""
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ = os.path.join(ROOT, "src", "Radar.Backend")
EXE = os.path.join(PROJ, "bin", "Release", "net10.0", "Radar.Backend.exe")
NODE = r"C:\Users\Administrator\.workbuddy\binaries\node\versions\22.22.2-3\node.exe"
PORT = 5399
BASE = f"http://127.0.0.1:{PORT}"


def get(path, timeout=10):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return r.status, r.read().decode("utf-8", "replace")


def port_free(p):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", p)) != 0


def synth_jobs():
    """按 Program.cs JobView 的真实字段构造样本（含 Completed+掩码 / Completed 无掩码 / Skipped）。"""
    probs = {"食管_裂孔疝": 0.106583, "肝_肝囊肿": 0.7331, "肾_肾结石": 0.4999999}
    return [
        {"id": "J001", "fileName": "merlin_a.nii.gz", "device": "cpu", "model": "radar",
         "variant": None, "status": "Completed", "createdAt": "2026-01-01T00:00:00Z",
         "startedAt": None, "finishedAt": None, "elapsedMs": 107600, "windowCount": 3,
         "probabilities": probs, "error": None, "hasMask": True},
        {"id": "J002", "fileName": "merlin_b.nii.gz", "device": "cpu", "model": "radar_plus",
         "variant": "merlin_ft", "status": "Completed", "createdAt": "2026-01-01T00:01:00Z",
         "startedAt": None, "finishedAt": None, "elapsedMs": 30000, "windowCount": 5,
         "probabilities": {"coronary_calcification": 0.42, "ascites": 0.08}, "error": None,
         "hasMask": False},
        {"id": "J003", "fileName": "huge_case.nii.gz", "device": "cpu", "model": "radar",
         "variant": None, "status": "Skipped", "createdAt": "2026-01-01T00:02:00Z",
         "startedAt": None, "finishedAt": None, "elapsedMs": None, "windowCount": None,
         "probabilities": None, "error": "空间尺寸超限", "hasMask": False},
    ]


HARNESS = r"""
const fs = require('fs'), vm = require('vm');
const P = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));

const els = {};
const ALL_IDS = ['device','model','variant','sample','submitErr','file','barFill',
                 'jobRows','resultPanel','resJob','probRows','kpiCount','kpiPos','kpiTime',
                 'csvLink','csvLongLink','jsonLink','maskLink','maskImg'];
const rejections = [];
process.on('unhandledRejection', e => rejections.push(String(e && e.stack || e)));
function mkEl(id) {
  return { id, innerHTML: '', textContent: '', value: '', href: '', src: '', alt: '',
           style: {}, selectedOptions: [], files: [], dataset: {},
           removeAttribute(){ delete this.src; }, setAttribute(){}, addEventListener(){} };
}
const document = {
  getElementById(id){ if(!els[id]) els[id] = mkEl(id); return els[id]; },
  querySelector(){ return null; }, createElement(t){ return mkEl('_'+t); }
};
const fetchCalls = [];
function fetchMock(url, opts){
  fetchCalls.push({ url, opts });
  const respond = obj => Promise.resolve({ ok: true, status: 200, json: async () => obj });
  if (url === '/api/devices') return respond(P.devices);
  if (url === '/api/samples') return respond(P.samples);
  if (url === '/api/jobs')    return respond(P.jobsNow);
  return respond({});
}
const sandbox = {
  document, fetch: fetchMock, console,
  setInterval(){ return 0; }, clearInterval(){}, setTimeout,
  Date, JSON, Object, Array, Math, Promise, Number, String,
};
vm.createContext(sandbox);

const fails = [], notes = [];
function ok(cond, msg){ (cond ? notes : fails).push((cond ? 'PASS  ' : 'FAIL  ') + msg); }

(async () => {
  ALL_IDS.forEach(id => document.getElementById(id));
  vm.runInContext(P.script, sandbox);
  for (let i = 0; i < 6; i++) await new Promise(r => setImmediate(r));
  if (rejections.length) { console.log('DIAG 未处理拒绝:\n' + rejections.join('\n')); }
  console.log('DIAG 已创建元素: ' + Object.keys(els).join(','));

  const rows = els['jobRows'] ? els['jobRows'].innerHTML : '';
  ok(rows.length > 0, '任务行已渲染');
  ok(/cpu/.test(els['device'].innerHTML) && /directml/.test(els['device'].innerHTML),
     'init() 已用 /api/devices 填充设备下拉');
  ok(/AC423ccbe\.nii\.gz/.test(els['sample'].innerHTML), 'init() 已用 /api/samples 填充样例下拉');
  ok(!els['sample'].innerHTML.includes('undefined'), '设备/样例下拉无 undefined');
  ok(!rows.includes('undefined'), '任务行无 undefined');
  ok(!rows.includes('NaN'), '任务行无 NaN');
  ok(rows.includes('官方CSV') && rows.includes('长表') && rows.includes('JSON'), '任务行含三种下载链接');
  ok(rows.includes('掩码'), 'hasMask=true 的行有掩码链接');
  ok(rows.includes('跳过'), 'Skipped 状态中文为「跳过」');
  ok(/<span class="tag Skipped">跳过<\/span>/.test(rows), 'Skipped 标签 class 正确');
  ok(rows.includes('超限跳过'), 'Skipped 行末列为提示文案');
  // J002 hasMask=false，其行内不应出现 /mask.png
  const j002 = rows.split('<tr>').find(s => s.includes('merlin_b.nii.gz')) || '';
  ok(!j002.includes('mask.png'), 'hasMask=false 的行无掩码链接');
  const j001 = rows.split('<tr>').find(s => s.includes('merlin_a.nii.gz')) || '';
  ok(j001.includes('/api/jobs/J001/json'), 'J001 JSON 链接路径正确');
  ok(j001.includes('/api/jobs/J001/csv?format=official'), 'J001 官方CSV 链接路径正确');
  ok(j001.includes('RADAR</td>'), 'radar 行模型列显示 RADAR');
  ok(rows.includes('RADAR+ · merlin_ft'), 'radar_plus 行显示 RADAR+ · 变体');

  // 结果面板：含掩码任务
  sandbox.renderResult(P.jobsNow[0]);
  const p0 = els['probRows'].innerHTML;
  ok(!p0.includes('undefined') && !p0.includes('NaN'), '结果表无 undefined/NaN');
  ok(p0.split('<tr>').length - 1 === 3, '结果表行数 = 概率条目数 (3)');
  ok(p0.indexOf('肝_肝囊肿') < p0.indexOf('食管_裂孔疝'), '结果按概率降序');
  ok(els['csvLink'].href === '/api/jobs/J001/csv?format=official', '官方CSV href 正确');
  ok(els['csvLongLink'].href === '/api/jobs/J001/csv', '长表 href 正确');
  ok(els['jsonLink'].href === '/api/jobs/J001/json', 'JSON href 正确');
  ok(els['maskLink'].style.display === '', 'hasMask=true 时掩码链接可见');
  ok(String(els['maskImg'].style.display) === '' && /mask\.png/.test(els['maskImg'].src), '掩码图已加载');
  ok(els['kpiCount'].textContent === 3, 'KPI 条目数正确');
  ok(els['kpiPos'].textContent === 1, 'KPI 阳性数正确 (>=0.5)');
  ok(els['kpiTime'].textContent === '107.6 s', 'KPI 耗时格式正确');

  // 结果面板：无掩码任务，必须把上一次的掩码清掉
  sandbox.renderResult(P.jobsNow[1]);
  ok(els['maskLink'].style.display === 'none', 'hasMask=false 时隐藏掩码链接');
  ok(els['maskImg'].style.display === 'none', 'hasMask=false 时隐藏掩码图');
  ok(els['maskImg'].src === undefined || els['maskImg'].src === '', 'hasMask=false 时清空掩码 src');
  ok(els['kpiCount'].textContent === 2, 'radar_plus 结果条目数正确 (2)');

  // 批量提交：多选走 /api/jobs/batch
  fetchCalls.length = 0;
  els['sample'].selectedOptions = [{value:'merlin_a.nii.gz'},{value:'merlin_b.nii.gz'}];
  els['device'].value = 'cpu'; els['model'].value = 'radar'; els['variant'].value = 'merlin_ft';
  const origFetch = sandbox.fetch;
  await sandbox.runSample();
  for (let i = 0; i < 4; i++) await new Promise(r => setImmediate(r));
  const batchCall = fetchCalls.find(c => c.url === '/api/jobs/batch');
  ok(!!batchCall, '多选提交命中 /api/jobs/batch');
  if (batchCall) {
    const body = JSON.parse(batchCall.opts.body);
    ok(Array.isArray(body.samples) && body.samples.length === 2, 'batch body.samples 长度=2');
    ok(body.model === 'radar' && body.device === 'cpu', 'batch body 字段正确');
  }

  // 单选仍走单样例端点
  fetchCalls.length = 0;
  els['sample'].selectedOptions = [{value:'merlin_a.nii.gz'}];
  await sandbox.runSample();
  for (let i = 0; i < 4; i++) await new Promise(r => setImmediate(r));
  ok(fetchCalls.some(c => c.url === '/api/jobs/sample'), '单选命中 /api/jobs/sample');
  ok(!fetchCalls.some(c => c.url === '/api/jobs/batch'), '单选不触发 batch');

  // 空选择必须有提示，且不发请求
  fetchCalls.length = 0;
  els['sample'].selectedOptions = [];
  await sandbox.runSample();
  ok(els['submitErr'].textContent.length > 0, '空选择给出提示');
  ok(fetchCalls.length === 0, '空选择不发请求');

  console.log(notes.join('\n'));
  if (fails.length) { console.log(fails.join('\n')); console.log(`\n== 失败 ${fails.length} 项 ==`); process.exit(1); }
  console.log(`\n== 全部通过 ${notes.length} 项 ==`);
})().catch(e => { console.error('HARNESS ERROR:', e); process.exit(2); });
"""


def main():
    if not os.path.exists(EXE):
        print("缺少后端产物，请先 dotnet build -c Release"); return 2
    if not port_free(PORT):
        print(f"端口 {PORT} 被占用"); return 2

    env = dict(os.environ)
    proc = subprocess.Popen(
        [EXE, "--urls", BASE, "--contentRoot", PROJ, "--webroot", os.path.join(PROJ, "wwwroot")],
        cwd=PROJ, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        ready, t0 = False, time.time()
        while time.time() - t0 < 60:
            if proc.poll() is not None:
                print(f"后端提前退出 rc={proc.returncode}"); return 2
            try:
                st, _ = get("/api/samples", timeout=2)
                if st == 200: ready = True; break
            except Exception:
                time.sleep(0.5)
        if not ready:
            print("后端 60s 内未就绪"); return 2
        print(f"后端就绪 ({(time.time()-t0):.1f}s)")

        _, html = get("/")
        m = re.search(r"<script>([\s\S]*?)</script>", html)
        if not m:
            print("未从服务端 HTML 中提取到 <script>"); return 2
        script = m.group(1)
        print(f"抓到服务端脚本 {len(script)} 字符")

        _, dev = get("/api/devices")
        _, smp = get("/api/samples")
        _, jobs = get("/api/jobs")
        payload = {"script": script, "devices": json.loads(dev), "samples": json.loads(smp),
                   "jobsNow": synth_jobs(), "realJobs": json.loads(jobs)}
        print(f"真实 /api/jobs 条目数 = {len(payload['realJobs'])}（用合成 payload 补足状态分支）")

        tmp = os.path.join(ROOT, ".tmp_verify_frontend")
        os.makedirs(tmp, exist_ok=True)
        pj = os.path.join(tmp, "payload.json")
        hj = os.path.join(tmp, "harness.js")
        with open(pj, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        with open(hj, "w", encoding="utf-8") as f:
            f.write(HARNESS)

        r = subprocess.run([NODE, hj, pj], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        print("--- NODE ---")
        print(r.stdout)
        if r.stderr.strip():
            print("--- STDERR ---"); print(r.stderr)
        return r.returncode
    finally:
        proc.terminate()
        try: proc.wait(timeout=10)
        except Exception: proc.kill()


if __name__ == "__main__":
    sys.exit(main())
