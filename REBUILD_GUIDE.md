# TTS → ASR → WER：新手跨机器重建指南

> 读者：第一次接触这套代码、希望在另一台 Linux 机器上独立重建和运行的人。
>
> 本指南分成「照着搭建」「理解后自己敲代码」「完整源码附录」三部分。源码附录包含这条流水线需要的客户端代码和四个评分工具，不依赖旧机器上的源码路径。单独带走这份 Markdown 即可恢复这些文本文件；**模型服务、模型权重、真实参考 WAV、数据集和依赖环境不包含在 Markdown 中**。
>
> 本文保留当前代码的行为和已知限制，不偷偷修正业务逻辑。演示配置使用低并发、单服务端点；不照搬旧机器的私网 IP 和参考音路径。

## 目录

1. [先认识整条流水线](#1-先认识整条流水线)
2. [迁移前必须准备的东西](#2-迁移前必须准备的东西)
3. [在新机器上恢复目录与源码](#3-在新机器上恢复目录与源码)
4. [创建 Python 环境](#4-创建-python-环境)
5. [准备输入与参考音](#5-准备输入与参考音)
6. [填写 TTS 与 ASR 配置](#6-填写-tts-与-asr-配置)
7. [检查环境与服务](#7-检查环境与服务)
8. [先分阶段运行再跑循环](#8-先分阶段运行再跑循环)
9. [输出文件与断点恢复](#9-输出文件与断点恢复)
10. [自己敲代码的实现路线](#10-自己敲代码的实现路线)
11. [参数速查与常见问题](#11-参数速查与常见问题)
12. [验收与迁移清单](#12-验收与迁移清单)
13. 完整源码附录：见文末各文件代码块。

## 1. 先认识整条流水线

```text
input.json：原始对话，每个 turn 的 user 是待合成文字
    ↓ scripts/tts_pipeline.py
TTS：调用远程服务，以参考音色生成 WAV
    → turn.tts_audio_path
    → 参考音信息 tts_ref_audio / tts_ref_text
    ↓ scripts/asr_pipeline.py
ASR：读取 WAV，以 Base64 发送到远程识别服务
    → turn.tts_audio_asr
    ↓ scripts/wer_filter_mp.py
WER：只比较 user 和 tts_audio_asr 两段文字，不读取 WAV
    → turn.tts_wer / turn.tts_wer_source / turn.wer_pass
    ↓ run.sh
有失败 turn → 下一轮只重做未通过的 turn
全部通过或达到 max_iter → 停止
最后筛选整段 sample 所有非空 user turn 都通过的数据
```

三个概念：

- **sample**：一整段对话，由唯一的 `id` 标识。
- **turn**：其中一轮，放在 `turns` 列表中。只合成 `user`，不合成 `assistant`。
- **参考音**：用来克隆音色的 WAV，参考文本必须是它实际说的话，不是待合成原文。

这些 Python 是**客户端**：它们不会在本地自动加载 Qwen 模型。TTS / ASR 推理在已部署的服务端完成；WER 在本地 CPU 计算。迁移客户端并不一定需要本地 GPU。

## 2. 迁移前必须准备的东西

### 2.1 两种迁移情况

| 情况 | 新机器要准备什么 |
|---|---|
| 新机器仍能访问已有 TTS / ASR 服务 | 客户端代码、Python 环境、实际可访问的服务地址、参考音和输入数据 |
| 新机器要连模型一起重新部署 | 上述所有内容，外加模型权重、兼容的 GPU/驱动、服务端代码和服务端独立环境 |

本指南完整覆盖客户端重建。**仅有本文不能保证在完全空白机器上部署相同模型服务**，因为原项目没有在这些客户端文件中锁定服务端环境版本。

已有部署脚本使用了 `qwen-asr-serve` 和 `vllm-omni serve`，并包含旧环境的 Conda、CUDA、模型绝对路径。若也迁移服务端，应另行备份并改写 `server_code/`、部署 YAML、模型权重和环境版本；不要直接照搬旧 CUDA 路径。

### 2.2 服务接口必须兼容

TTS 客户端要求：

- 地址形如 `http://HOST:9700/v1`。
- POST `/v1/audio/speech`。
- 接受 `model`、`input`、`task_type: Base`、`language`、`response_format: wav`、`ref_audio`、`ref_text`。
- `ref_audio` 是带 `data:audio/wav;base64,` 前缀的参考音编码。
- 返回**可解码的 WAV 二进制内容**，而不是音频 URL 或 JSON。
- 使用支持参考音克隆的 Base 服务；仅名字相同并不保证接口相同。

ASR 客户端要求：

- 地址形如 `http://HOST:9800/v1`。
- POST `/v1/chat/completions`。
- 支持消息内容中的 `type: input_audio`，其中 `input_audio` 有 `data` 和 `format`。
- 返回 JSON，识别结果在 `choices[0].message.content`。
- content 可以是纯正文，也可以是 `language Chinese<asr_text>正文`。

当前客户端**没有添加 API Key / Bearer 认证头**。如果新服务需要认证、HTTPS 特殊证书或不同音频格式，应先适配请求代码，不能只换 URL 就假定可用。

只请求你有权限访问的服务。真实输入文字和音频会被发送到所配置服务，先确认数据使用许可。

### 2.3 必须实际迁移的非代码材料

- 一段有使用授权的真实参考 WAV，及其准确转写。
- 输入 JSON 数据集。
- 若继续用原音色池：音色池 JSON **和里面列出的全部音频文件**。
- 若继续以前的任务：输出 JSON、WAV、manifest、未完成 ASR 的 JSONL，一起迁移并处理绝对路径变化。
- 旧机器的依赖版本记录，特别是提供 `tn` 的归一化库。

**建议新机器第一次验证从一个新输出目录开始，不恢复旧任务。**这样最容易排除历史路径和缓存干扰。

## 3. 在新机器上恢复目录与源码

### 3.1 目标目录

先在新机器终端创建一个学习目录，例如：

```bash
mkdir -p "$HOME/asr_tts_wer/code"
cd "$HOME/asr_tts_wer/code"
```

将本指南保存为这个目录中的 `REBUILD_GUIDE.md`，并在 VSCode 打开 `code` 目录。后续相对路径命令均在这个目录执行。目录最终是：

```text
asr_tts_wer/
└── code/
    ├── REBUILD_GUIDE.md
    ├── run.sh
    ├── scripts/
    │   ├── tts_pipeline.py
    │   ├── asr_pipeline.py
    │   ├── wer_filter_mp.py
    │   └── wer_filter.py              # 可选串行参考版本，run.sh 不调用
    ├── tools/
    │   ├── zh/
    │   │   ├── compute_wer_jsonl.py
    │   │   └── compute_syllable_wer_jsonl.py
    │   ├── en/
    │   │   └── compute_wer_jsonl.py
    │   └── other/
    │       └── compute_wer_jiwer.py
    ├── config_data/
    │   ├── config_tts.yaml            # 新机器演示配置，必须改服务地址
    │   └── config_asr.yaml
    ├── data/
    │   ├── input_demo.json
    │   └── refs.json                  # 你稍后创建
    ├── refs/
    │   └── reference.wav              # 你实际复制的参考音
    └── run_output/                    # 首次运行时生成
```

`tools/` 这四个文件不能漏。主 WER 脚本会导入它们；不是只复制三个 pipeline Python 就足够了。没有 `__init__.py` 也可通过 Python 的 namespace package 导入；不要把这些工具放成一个扁平目录。

### 3.2 两种恢复方式

**方法 A：手动敲写。**按附录的每个文件标题创建文件，把代码块中的内容输入进去。不要复制 Markdown 标题或围栏，不要带编辑器显示的行号。缩进使用四个空格，特别注意 `run.sh` 内 Python here-document 的结束标记 `PY` 必须顶格。

**方法 B：从本指南一次提取源码。**这是避免小白手敲遗漏的备用方式。先进入上述 `code` 目录，在终端执行下面命令。脚本只提取文末显式标记的文件，不会执行这些源码。为了避免覆盖，它要求所有目标文件尚不存在；若已存在，改用一个全新的目录。

```bash
python3 - <<'PY'
from pathlib import Path
import hashlib
import re

root = Path.cwd().resolve()
guide = root / "REBUILD_GUIDE.md"
text = guide.read_text(encoding="utf-8")
pattern = re.compile(
    r"<!-- REBUILD_FILE: ([^\n]+) SHA256: ([0-9a-f]{64}) -->\n"
    r"```[^\n]*\n(.*?)\n```\n<!-- /REBUILD_FILE -->",
    re.S,
)
records = pattern.findall(text)
if not records:
    raise SystemExit("没有找到完整源码附录，请确认复制的是整份指南。")
outputs = []
seen = set()
for name, expected, body in records:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise SystemExit(f"不允许的文件路径：{name}")
    target = (root / relative).resolve()
    if not target.is_relative_to(root) or target in seen:
        raise SystemExit(f"路径越界或重复：{name}")
    if target.exists():
        raise SystemExit(f"目标已存在，不覆盖：{name}；请改用全新目录。")
    content = (body + "\n").encode("utf-8")
    actual = hashlib.sha256(content).hexdigest()
    if actual != expected:
        raise SystemExit(f"源码校验失败：{name}；请重新复制指南。")
    seen.add(target)
    outputs.append((target, content))
for target, content in outputs:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    print("已恢复：", target.relative_to(root))
print(f"完成：{len(outputs)} 个文件。接下来安装依赖、准备参考音并修改配置。")
PY
```

源码附录带 SHA256，校验通过只说明源码没有在复制中变化，**不代表依赖、服务或评分逻辑已经验收**。

## 4. 创建 Python 环境

建议 Linux + Bash + Python 3.10 或更新版本；更优先沿用旧机器已经验证过的 Python 和依赖版本。类型注解用了 `list[str]`、`str | None`，ASR 文件没有延迟解释类型的 future import，因此不要直接用老版本 Python。

在 `code` 目录执行：

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install requests PyYAML soundfile scipy pypinyin jiwer
```

这些包的作用：

| 导入名 | 安装包 | 用途 |
|---|---|---|
| requests | requests | HTTP 请求服务 |
| yaml | PyYAML | 读 YAML 配置 |
| soundfile | soundfile | TTS WAV 解码、保存 |
| scipy | scipy | 重采样 |
| pypinyin | pypinyin | 中文转拼音 |
| jiwer | jiwer | 词级错误率 |
| tn | 见下一段 | 中英文文本归一化 |

`argparse`、`json`、`os`、`threading`、`multiprocessing` 等是 Python 标准库，不用逐个 pip 安装。

### 4.1 `tn` 归一化依赖：安装名不等于导入名

源码要求这两个导入：

```python
from tn.chinese.normalizer import Normalizer
from tn.english.normalizer import Normalizer
```

**不要直接猜测执行 `pip install tn`。**常见提供这套接口的实现是 WeTextProcessing，但仅凭导入语句不能锁定原环境的发行包、版本或私有修改。最稳妥的方法是在旧机器的实际运行环境里记录提供 `tn` 的安装包及来源：

```bash
python3 - <<'PY'
import importlib.metadata as metadata
import importlib.util

spec = importlib.util.find_spec("tn")
print("tn 模块位置：", spec.origin if spec else "没有找到")
providers = metadata.packages_distributions().get("tn", [])
print("发行包映射：", providers or "元数据无法确定，请检查旧环境安装记录")
for name in providers:
    dist = metadata.distribution(name)
    print("包与版本：", name, dist.version)
    print("安装来源：", dist.read_text("direct_url.json") or "没有 direct_url.json")
PY
```

在新机器安装已记录的包版本；如果来自本地源码或私有仓库，必须一并迁移那份源码及其依赖，不要照抄失效的旧绝对路径。

如果无法访问旧环境，可把下面命令作为**候选安装方案，而不是原环境精确复现保证**：

```bash
python3 -m pip install WeTextProcessing
```

是否适配当前 Python/系统，以安装结果和实际导入、初始化为准。该类文本归一化实现可能依赖有限状态转换相关二进制库（例如 Pynini/OpenFst）以及规则/缓存资源；首次初始化可能编译或加载资源，需要时间和可写目录。只安装 pip 包或只看 `find_spec` 成功都不足以证明可评分。

用下面命令实际初始化并检查：

```bash
python3 - <<'PY'
from tn.chinese.normalizer import Normalizer as ZhNormalizer
from tn.english.normalizer import Normalizer as EnNormalizer

zh = ZhNormalizer()
en = EnNormalizer()
print("中文归一化示例：", zh.normalize("今天是2026年。"))
print("英文归一化示例：", en.normalize("I have 2 apples."))
PY
```

上面的示例输出可能随库版本和规则变化；关键是正常初始化、返回可用文本，而不是必须返回本文规定的某个字符串。离线迁移还需要检查库是否缺少规则数据、模型资源或已有编译缓存。后续真实 WER 验证应确认数字等归一化口径符合你的任务。

本指南没有在新环境执行该候选安装或真实初始化，因此不声明依赖版本兼容或评分结果已验证。

每次新开终端都要重新激活 `.venv`。检查当前运行环境：

```bash
which python3
python3 -m pip --version
```

不要只执行裸 `pip` 安装再用另一个 Python 运行。`run.sh` 调用的是 `python3`，应确认它属于当前激活环境。

遇到 `venv` 创建失败，需按系统说明安装相应的 Python venv 支持。遇到 libsndfile 动态库缺失，需按系统和管理员许可安装对应系统库，不能只改 Python 代码解决。

联网受限机器：先在**兼容的新机器操作系统/CPU/Python版本**环境准备 wheel 和其他资源，再迁移安装；有 C++ 扩展的包不一定能直接跨平台复制。旧 `.venv` 整个复制到不同路径通常不可靠，优先重建环境。

## 5. 准备输入与参考音

### 5.1 输入样例

附录已提供 `data/input_demo.json`：只有一个 sample、两轮中文短句。顶层必须是列表，sample 的 `id` 唯一且非空，`turns` 是列表，`user` 是字符串或空值。

```json
[
  {
    "id": "demo_001",
    "turns": [
      {"user": "你好，欢迎使用语音合成。"},
      {"user": "请帮我查一下北京的天气。"}
    ]
  }
]
```

建议 ID 使用字母、数字、下划线等安全字符；它会进入 WAV 文件名。不要重复 ID，不要在 ID 中放路径分隔符。

新任务的原始输入不要预填 `wer_pass: true`，否则会跳过处理。

### 5.2 真实参考音

1. 在 `code` 下创建 `refs` 目录。
2. 将一段有授权的真实 WAV 复制为 `refs/reference.wav`。
3. 核对它实际说的每句话。
4. 用下面代码创建 `data/refs.json`；先修改 `ref_text` 为真实转写。

```bash
mkdir -p refs data
python3 - <<'PY'
from pathlib import Path
import json
import soundfile as sf

wav = Path("refs/reference.wav").resolve()
ref_text = "请在执行前将这一句改成参考音频实际说的内容。"
if ref_text.startswith("请在执行前"):
    raise SystemExit("请先编辑 ref_text，再执行。")
if not wav.is_file():
    raise SystemExit(f"没有找到真实参考音：{wav}")
info = sf.info(str(wav))
if info.frames <= 0:
    raise SystemExit("参考音没有有效采样数据。")
dst = Path("data/refs.json")
if dst.exists():
    raise SystemExit("data/refs.json 已存在，请先查看，不自动覆盖。")
dst.write_text(
    json.dumps([{"audio": str(wav), "text": ref_text}], ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(info)
print("已创建", dst)
PY
```

参考音与转写是一对；不要用待合成句子冒充参考音转写。不要用静音 WAV 来验证音色克隆质量。

音色池 `audio` 使用新机器真实绝对路径。当前代码不会将音色池中的音频相对路径自动按音色池文件所在目录解析。

## 6. 填写 TTS 与 ASR 配置

附录演示配置用 `127.0.0.1`，只适用于服务就在新机器本地的情况。若服务在其他机器，把地址改成你有权限访问的实际地址，保持 `/v1`，**不要额外在 base URL 后加 `/audio/speech` 或 `/chat/completions`，Python 会自己拼接**。

### 6.1 config_data/config_tts.yaml

```yaml
base_urls:
  - "http://127.0.0.1:9700/v1"
model_name: "qwen3-tts-1.7b"
max_workers: 1
target_sr: 16000
request_timeout: 60
max_retry: 2
per_url_exclusive: true
check_audio_length: true
min_sec_per_char: 0.001
max_sec_per_char: 2.0
min_audio_sec: 0.5
language: "Chinese"
output_wav_subdir: "wavs"
manifest_filename: "manifest.jsonl"
ref_list_path: "../data/refs.json"
```

`ref_list_path` 按 YAML 所在目录解析，`config_data/../data/refs.json` 正好指向准备好的音色池。

原中文 Livis 配置是 16 个地址、64 个线程、每地址排他。它同时占用地址的任务最多为 16，而不是 64。先单端点、单线程成功，再逐步增加并发。演示配置的超时与重试也有意与旧配置不同，避免首次验证长时间堆积请求。

### 6.2 config_data/config_asr.yaml

```yaml
servers:
  - "http://127.0.0.1:9800/v1"
model: "qwen3-asr-1.7b"
request:
  temperature: 0.0
  max_tokens: 2048
  timeout: 60
threads: 1
json_fields:
  audio_path: tts_audio_path
  asr_result: tts_audio_asr
```

模型名必须与 served model name 一致。只写名字不能加载模型。客户端没有自动重试 ASR 单次请求。

TTS `language: Chinese` 控制生成语言；WER `--lang zh_syllable` 控制评分方法，两个配置不会自动联动。

## 7. 检查环境与服务

### 7.1 不请求服务的基础检查

在 `code` 目录、激活环境后执行：

```bash
bash -n run.sh
python3 -m compileall -q scripts tools
python3 -m json.tool data/input_demo.json > /dev/null
python3 -m json.tool data/refs.json > /dev/null
python3 -m pip check
python3 - <<'PY'
import requests, yaml, soundfile, scipy, pypinyin, jiwer
from pathlib import Path
from tn.chinese.normalizer import Normalizer as ZhNormalizer
from tn.english.normalizer import Normalizer as EnNormalizer
print("基础依赖可以导入；仍需实际初始化归一化器和核对服务。")
for filename in ["config_data/config_tts.yaml", "config_data/config_asr.yaml"]:
    raw = yaml.safe_load(Path(filename).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit(f"配置不是字典：{filename}")
    print("YAML可解析：", filename)
PY
```

`compileall` 只证明语法能编译，不证明第三方库能导入，也不证明模型服务已经运行。首次初始化归一化器可能需要较长时间和可写缓存目录，见依赖说明。

### 7.2 可选的服务地址检查

在确认有权限请求后，将 URL 替换为实际地址再执行：

```bash
curl --fail --connect-timeout 5 --max-time 10 http://127.0.0.1:9700/v1/models
curl --fail --connect-timeout 5 --max-time 10 http://127.0.0.1:9800/v1/models
```

如果服务支持 `/models`，它能帮助检查连通性和模型名。**该接口成功不等于音频端点可用；有些服务没有该接口，失败也不单独证明音频端点失效。**最终要以下面的真实小样本验证为准。

不要在未知服务上直接启动大批量任务。

## 8. 先分阶段运行再跑循环

### 8.1 首次验证只做一轮、低并发

每次独立验证使用一个新的输出目录。在同一终端执行：

```bash
OUT="run_output/manual_$(date +%Y%m%d_%H%M%S)_${RANDOM}"
mkdir -p "$OUT"
```

**第一步：TTS。**会把原文和参考音发送到配置的服务。

```bash
python3 scripts/tts_pipeline.py \
  --config config_data/config_tts.yaml \
  --input data/input_demo.json \
  --output "$OUT/demo_tts.json"
```

检查日志中本次成功/失败数，打开 JSON 检查路径，实际试听 `wavs/`。仅出现输出 JSON 并不保证所有 turn 合成成功。

**第二步：ASR。**会把生成的音频发送到识别服务。

```bash
python3 scripts/asr_pipeline.py \
  --config config_data/config_asr.yaml \
  --input "$OUT/demo_tts.json" \
  --output "$OUT/demo_tts_asr.json" \
  --threads 1
```

打开输出，确认每个有效 turn 的 `tts_audio_asr` 有合理正文；`null` 表示没有识别文本。日志 `OK` 不保证文字正确。

**第三步：WER。**这一阶段不向 TTS/ASR 服务发送数据，但归一化库初始化可能涉及本地资源。

```bash
python3 scripts/wer_filter_mp.py \
  --input "$OUT/demo_tts_asr.json" \
  --output "$OUT/demo_tts_asr_wer.json" \
  --threshold 0.1 \
  --lang zh_syllable \
  --workers 1 \
  --record-all-scores
```

打开最终 JSON，检查 `tts_wer`、`tts_wer_source`、`wer_pass`、所有方法分数是否合理。尤其检查有没有所有方法为无穷大、最终却为 0 的异常情况，详见已知问题。

### 8.2 端到端循环

手动三个阶段都通过验证后，再用新的目录跑 shell：

```bash
LOOP_OUT="run_output/loop_$(date +%Y%m%d_%H%M%S)_${RANDOM}"
WER_WORKERS=1 bash run.sh \
  --input data/input_demo.json \
  --out "$LOOP_OUT" \
  --max-iter 2 \
  --wer-threshold 0.1 \
  --lang zh_syllable \
  --tts-config config_data/config_tts.yaml \
  --asr-config config_data/config_asr.yaml
```

这里 `WER_WORKERS=1` 是环境变量赋值，只作用于这次命令。`run.sh` 没有 `--workers` 选项；直接运行 WER Python 才用 `--workers`。

- `--max-iter` 必须是正整数，先用 1 或 2。
- 帮助注释写默认 5，但实际 `MAX_ITER` 默认是 2；显式传入避免歧义。
- 原 `run.sh` 默认 WER worker 数为 24，首次验证显式降低。
- `--threshold 0.1` 的含义是严格 **WER < 10%**。恰好 10% 不通过。
- 大批量前先确认服务吞吐，再调整 TTS、ASR、WER 三种不同并发参数。

## 9. 输出文件与断点恢复

### 9.1 循环输出

输入名为 `input_demo.json` 时：

```text
<out>/
├── wavs/
├── manifest.jsonl
├── iter_1_input_demo_tts.json
├── iter_1_input_demo_tts_asr.json
├── iter_1_input_demo_tts_asr_wer.json
├── iter_2_input_demo_tts.json             # 若进行了第二轮
├── iter_2_input_demo_tts_asr.json
├── iter_2_input_demo_tts_asr_wer.json
└── input_demo_final_all_pass.json
```

最新一轮 WER JSON 保留全部样本；`final_all_pass.json` 只保留整段合格 sample。空 `user` turn 不影响最后判定，没有 turn 的 sample 排除。不要把“最新结果”和“全部通过的结果”混为一谈。

### 9.2 阶段恢复规则

| 部分 | 恢复行为 | 限制 |
|---|---|---|
| run.sh | 找到可解析的本轮 WER 就跳过整轮；ASR 存在就只补 WER；TTS 存在就补 ASR + WER | 不校验输入、配置、阈值是否变化，合法 JSON 不等于该阶段完整成功 |
| TTS | manifest 恢复参考音和最新路径；已 `wer_pass=true` 跳过 | `load_done()` 未用于排除任务；中断后可能重做已合成 turn |
| ASR | 已输出 JSON 优先；恢复 `<output>.jsonl` 中文本；有文本或已通过则跳过 | 恢复靠 sample/turn 列表位置，不验证对应音频；不要重排数据 |
| WER | 已 `wer_pass=true` 直接透传 | 没有 JSONL 断点恢复，最终普通写入一次 |

换模型、音色、阈值、输入数据做**新实验**，推荐新输出目录，不要删除旧目录来“清缓存”。需要继续旧任务时，保持输入结构和参数不变。

### 9.3 跨机器续跑的路径问题

JSON 里的 `tts_audio_path` 和参考音 `audio` 常是绝对路径。换机器后，需要复制实际音频并建立新路径映射；单独复制 JSON 不够。

修改路径前先备份；只修改已确认对应的路径前缀，不要盲目全文件字符串替换。manifest 还包含参考音绝对路径，ASR JSONL 又按位置恢复，应一并核对。

TTS 在新音频成功且 manifest 写入后会删除输入 turn 指向的旧 WAV。不要把共享、不可删除的珍贵音频路径当作待重合成旧文件。

## 10. 自己敲代码的实现路线

这一节是重写时的“设计图”；完整实现见附录。不要尝试仅靠伪代码恢复同等评分行为，底层归一化和 token 切分规则也必须一致。

### 10.1 先固定数据契约

```text
sample.id：唯一标识
sample.turns：每轮列表
turn.user：参考原文
turn.tts_audio_path：生成的音频路径
turn.tts_audio_asr：识别文字
turn.tts_wer：错误率小数，可大于 1
turn.tts_wer_source：产生最小值的方法名
turn.wer_pass：真正的布尔值 true / false
sample或turn.tts_ref_audio / tts_ref_text：成对参考音信息
```

Python `True` 对应 JSON `true`，Python `None` 对应 JSON `null`。字符串 `"true"` 不是布尔值 `true`。

### 10.2 第一份：tts_pipeline.py

按下面顺序理解和实现：

1. **Config**：用 dataclass 保存配置；`from_yaml()` 解析 YAML，优先读外部音色池；两个 property 拼接 WAV 目录和 manifest 路径。
2. **TTSClient**：缓存参考音 Base64；锁避免重复编码；排他队列管理空闲 URL。
3. **clone()**：构造参考音克隆请求 → 获取 WAV → 重采样 → 检查时长 → 保存 PCM_16；仅特定异常与校验失败进入重试；finally 归还 URL。
4. **ManifestStore**：追加一行一个 JSON；锁保护并发写；flush + fsync；读取时最新 turn 记录覆盖旧记录。
5. **build_jobs()**：遍历 turn；空文本跳过、已通过跳过，其余组成 `(sample_id, turn_idx, text, filename)`。这里 turn 编号从 1 开始。
6. **write_augmented_json()**：未通过 turn 删除旧 ASR/WER/pass 三个字段；写最新路径；同样参考音放 sample 层，不一致则放 turn 层；最终写 JSON。
7. **run_pipeline()**：选参考音 → 检查多轮音色一致性 → 收集旧 WAV → 线程池调用 work → 写 manifest → 成功后删旧 WAV → 汇总并回写。
8. **main()**：读 CLI，设置输入/输出路径，调用流水线。

参考音优先级：完整 turn 音频+文本 → 完整 sample 音频+文本 → 历史/按 sample 随机兜底。音频和文本不得跨来源拼装。音色不一致只警告，不强制修正。

时长公式：`duration = 采样帧数 / 目标采样率`；`spc = duration / len(text.strip())`。字符数包括标点和内部空白，不是英文词数。长度检查不等于内容正确。

### 10.3 第二份：asr_pipeline.py

1. **Config / load_config()**：读服务地址、模型、请求设置、字段名。
2. **encode_audio()**：读取 WAV 等文件，编码成不带 data URI 前缀的 Base64，扩展名推断 format；缓存锁只保护字典，读取编码在锁外。
3. **build_messages()**：可选 system context，加一条包含 input_audio 的 user 消息；这里 user 角色不是输入 JSON 的 user 原文。
4. **parse_asr()**：用正则识别 `language ...<asr_text>...`；否则整个返回作为正文。
5. **transcribe()**：随机选 URL → 请求 chat/completions → 提取 content → 解析；普通异常返回 `text=None` 与 error，不立即重试。
6. **save_json_atomic()**：先写 `.tmp`，再 `os.replace` 替换最终文件。
7. **run_json_batch()**：已有输出优先 → 覆盖 JSONL 恢复文本 → 收集 pending → 线程池请求 → 主线程写 data 和 JSONL → 最后完整 JSON 保存并删除 JSONL。
8. **main()**：激活全局 CONFIG，CLI 线程覆盖 YAML 线程，执行批量。

ASR JSONL 的 `(ri, ti)` 从 0 开始，按位置恢复，不按 sample ID。最终 JSON 主要只保存 text，不会保留完整请求诊断信息。缓存只在本次进程中存在。

### 10.4 第三份：wer_filter_mp.py

1. **导入四个评分工具**：保持 tools 目录层级，用别名区分同名 `cer_one`。
2. **LANG_LIBS**：语言选择的是方法列表；当前 zh 和 zh_syllable 相同。
3. **_collect_jobs()**：历史 true 跳过；无原文按规则 true；缺路径或识别文本 false；其余生成 `(job_id, ref, hyp)` 与 locations。
4. **_worker_init()**：每个子进程保存自己的方法列表和 quiet 设置；模块导入实际在文件顶层，不在 initializer 中。
5. **_worker_compute()**：多方法计算百分比 → 单方法异常用 infinity → 取最小值和来源 → 返回所有分数。原行为包含“所有方法失败仍回退 0”的缺陷。
6. **main()**：阈值小数乘 100 → 开进程池 → as_completed 收结果 → 根据 job_id 找 turn → 除 100 写分数 → 严格小于判断 → 最后保存一次。

基本公式：`(替换 + 删除 + 插入) / 参考单位数`；不是语义相似度。字符级偏文字一致，拼音级偏发音一致，英文主要词级。选最小值是一种宽松策略，不等于标准统一口径 WER。

### 10.5 第四份：run.sh

1. `set -euo pipefail`：正常命令失败时尽量退出；Python 捕获并吞掉单条错误不一定导致 shell 失败。
2. 从脚本路径计算 CODE_DIR，解析 CLI 和默认环境变量。
3. 每轮拼三个结果路径，根据可解析的已有结果选择补跑阶段。
4. 上一轮 WER 输出作为下一轮 TTS 输入。
5. 内嵌 Python 统计非空 user 且未通过的 turn；为零提前结束。
6. 最后内嵌 Python 筛选整段全通过 sample。

Shell 参数 `-t` 是 WER 阈值；ASR Python 参数 `-t` 是线程数，含义不要混淆。`WER_WORKERS` 是 shell 环境变量；`--workers` 是 WER Python 参数。

### 10.6 底层四个评分工具

工具里的 `cer_one(ref, hyp)` 统一返回含 `wer` 的字典，`wer` 使用百分比数值。它们包含各自的文本归一化、拆分、大小写/标点处理、对齐与统计。

有些文件还提供独立命令行处理逻辑，但这条流水线只 import 并调用 `cer_one`。重建时建议直接保留原工具文件，而不是另写一个“字符串编辑距离”就认为结果等价。

### 10.7 常见 Python 语法速记

| 写法 | 含义 |
|---|---|
| `def f(...):` | 定义函数 |
| `class Config:` | 定义一类对象 |
| `@dataclass` | 自动生成常用初始化等代码 |
| `@property` | 方法可以像属性访问 |
| `raw["model"]` | 必须存在的字段，不存在报错 |
| `raw.get("x", default)` | 缺字段时返回默认值 |
| `with open(...)` | 结束时自动关闭文件 |
| `continue` | 跳过本轮循环，处理下一个 |
| `return` | 返回并结束当前函数 |
| `try / except / finally` | 尝试执行 / 处理异常 / 做清理 |
| `enumerate(list)` | 同时取下标和元素 |
| `pool.submit(...)` | 提交异步任务，拿到 Future 回执 |
| `as_completed(...)` | 谁先完成先取谁 |
| `future.result()` | 取结果，任务异常可能在这里再次抛出 |
| `is True` | 必须是布尔值 True，不是字符串或其他真值 |
| `float("inf")` | 正无穷，常用于初始化最小值搜索 |
| `global CONFIG` | 函数修改模块级 CONFIG 变量 |

## 11. 参数速查与常见问题

### 11.1 参数区别

| 参数 | 控制什么 | 不代表什么 |
|---|---|---|
| TTS max_workers | 本地 TTS 工作线程 | GPU 数量；排他开启时仍受地址数限制 |
| ASR threads | 本地 ASR 工作线程 | 固定均分端点；服务端模型实例数 |
| WER workers | 本地 CPU 工作进程 | GPU 数量 |
| target_sr | 输出采样率 | 音量、语速、音频时长 |
| request timeout | 连接/读取等待超时 | 音频最长秒数；严格总任务倒计时 |
| max_tokens | ASR 输出文本预算 | 汉字数、音频字节数 |
| temperature | ASR 输出随机程度 | 必然识别正确 |
| max_retry | TTS 特定失败额外重试次数 | 所有异常都重试；ASR 自动重试 |
| min/max_sec_per_char | 输出后粗略时长比例检查 | 设置语速；保证没有错字 |
| threshold | 小数错误率阈值 | 百分比整数，例如 10 不该用来表示 10% |

### 11.2 已知问题：原样保留，运行前必须知道

1. **WER 所有方法失败会回退到 0%**，正阈值下误标通过。单方法结果缺失 `wer` 也默认 0。正式质量验收前建议另行修正为失败/不可评估并记录异常；本文附录仍保留原源码。
2. **错误率可超过 1**，例如大量插入；不能强制当成 0～1 概率。多方法混用且取最小值应在结果说明中明确。
3. **记录所有分数时，失败库可能写出 Infinity**。Python json 默认能写入非严格 JSON 数值，但其他系统可能拒绝。检查下游格式要求，建议用 null + 错误字段表示不可评估。
4. **历史 wer_pass=true 不重评估**。调严阈值后，输入中的旧 true 仍透传；新实验应从原始数据启动，或在备份上有意识清除旧状态。
5. **TTS 重合成失败可能仍输出历史路径**，而旧 ASR 字段被清空。以日志本次成功数和文件实际存在性核验，不只看 JSON 有路径。
6. **TTS 成功会删旧 WAV**，早期迭代 JSON 中的旧路径可能失效。需要完整历史音频时应先备份。
7. **TTS / WER 最终普通写入**，不像 ASR 那样原子替换；被中断可能留下残缺 JSON。
8. **WER chunk-size 当前只控制进度打印**，没有按批投递；注释不要当作实际执行逻辑。
9. **WER quiet 没有包住顶层 import**，不能保证隐藏导入阶段的输出，只主要隐藏计算函数普通 print。
10. **TTS 缺 id 会打印后返回**，不一定是非零退出；用户看到“命令结束”不能当作成功。服务地址空列表等也没有完整配置校验。
11. **run.sh 的 max_iter 未强校验**，0 或负数会导致最终结果为空路径，不要这样传。
12. **可选串行 wer_filter.py 有别名错误**：`_import_cer_funcs()` 的 en 列表引用未定义 `zh_cer_one`，构造 groups 时就可能报 NameError。它不是可直接替换的已验证备用，主流程使用 mp 版本。

### 11.3 常见报错定位

| 现象 | 优先检查 |
|---|---|
| No module named yaml | 当前 python3 的环境是否安装 PyYAML |
| No module named tn | 安装提供 tn 的准确归一化依赖，不是随便找同名包 |
| No module named zh/en/other | tools 文件层级是否完整，是否被同名模块冲突 |
| libsndfile / pynini 相关错误 | 平台、二进制包、底层库与 Python 版本是否兼容 |
| Connection refused | 服务是否启动，IP/端口是否正确 |
| Connection timed out | 路由、防火墙、服务负载和超时设置 |
| 401 / 403 | 新服务需要认证，当前请求代码未实现认证 |
| 404 | 接口路径是否重复拼接，服务实现是否兼容 |
| TTS 返回 server error JSON | 服务错误详情、模型/task_type/参考音是否正确 |
| ASR choices / message / content 相关错误 | 返回结构是否符合约定，是否被网关返回其他 JSON |
| 音频文件找不到 | 旧绝对路径是否仍存在，音色池音频是否一并迁移 |
| 大量 ASR null | 请求失败日志，音频合法性，服务端限制 |
| 结果全通过但明显不合理 | 全库失败回退 0、旧 true 跳过、宽松最小值策略 |
| 改配置但结果不变 | 旧输出导致阶段跳过，换新输出目录验证 |

## 12. 验收与迁移清单

### 12.1 新机器验收

- [ ] 完整恢复 run.sh、三个主 Python、四个工具和配置。
- [ ] Python 版本兼容；当前 python3 和 pip 指向同一环境。
- [ ] pip check 无依赖冲突；tn 中英文归一化器能实际初始化。
- [ ] 服务地址、模型名、接口格式正确，有权限使用并发送数据。
- [ ] 参考音真实存在且能解码；参考文本准确；样例 ID 唯一。
- [ ] 单线程 TTS 成功，两个 WAV 可试听，内容与音色合理。
- [ ] 单线程 ASR 有非空识别文本，未发生响应截断等问题。
- [ ] WER 单 worker 完成，最小值与所有分数合理，未出现全库失败假通过。
- [ ] run.sh 新输出目录完成 1～2 轮，final_all_pass 格式正确。
- [ ] 达到阈值是否合理已人工抽听；不把程序正常退出当作数据全合格。
- [ ] 才开始逐步增加并发和输入规模。

### 12.2 留给下次迁移的材料

验证成功后执行：

```bash
python3 --version > python-version.txt
python3 -m pip freeze > requirements-verified.txt
```

这两条会写文件；若已有同名记录，先查看再决定是否覆盖。额外记录操作系统/CPU、归一化器资源或缓存、模型服务版本/部署方式、served model name。`pip freeze` 可能含旧本地 editable 路径，迁移前核对，不是所有条目都能在新机器直接安装。

把本指南、验证后的配置、环境记录和有权限迁移的数据一起保存。不要仅迁移输出 JSON；不要把访问密钥、私有认证信息放进公开文档。

### 12.3 本指南随附源码的验证记录

编写指南时已完成以下本地检查，均在临时目录或离线模拟中进行。检查环境的 Python 为 3.12.11；这不是所有依赖和模型服务已验证支持的版本承诺：

- 运行本文提取器，恢复 12 个附录文件；重复执行会拒绝覆盖已有文件。
- 12 个文件 SHA256 校验通过；9 个原源码文件与当前项目一致，仅对原本无末尾换行的文件补一个换行。
- 恢复后的全部 Python 可编译；YAML / JSON 可解析；全部演示 Bash 命令及其内嵌 Python 通过语法检查。
- 用假 HTTP 响应验证 TTS 配置读取、任务筛选、WAV 保存；用假 ASR 响应验证音频编码、返回解析、JSONL 恢复和最终原子保存。
- 用假评分函数验证 WER 分类、最小值规则，并复现“所有方法失败仍返回 0”的问题；没有运行真实底层归一化评分库。
- 用预制两轮 WER JSON 验证 run.sh 的阶段跳过、提前结束和最终 sample 筛选。

**没有安装新依赖、没有连接真实模型服务、没有发送真实音频、没有验证新机器的归一化库初始化或真实 WER 分数。**静音 WAV 只用于测试本地文件处理，并不用于证明音色克隆质量。以上不是新机器端到端验收的替代。

---

# 完整源码附录

下面带 `REBUILD_FILE` 标记的代码块可以通过第 3 节脚本提取。主代码和工具是当前项目源码快照；配置和输入是本文设计的可修改演示模板。可选串行 WER 文件仅供阅读比较，不被主流程使用。

**已有缺陷保持原样，不因收录完整源码就视为已修复。不要将这些客户端源码当作模型部署源码。**

## 附录 1：run.sh

<!-- REBUILD_FILE: run.sh SHA256: 125c213060ed686e8d92076e44b0a976e509c902e35a0e59235e8184fb4667c5 -->
```bash
#!/usr/bin/env bash
# TTS -> ASR -> WER 循环流水线
#
# 同一输出目录里、单一 JSON 逐轮演进。每个 turn 用 wer_pass 字段标记状态：
#   iter_N 里，wer_pass=false 的 turn 会在 iter_(N+1) 里被重新合成（旧音频删除，
#   tts_audio_path 更新为新音频，tts_audio_asr 清空重跑 ASR）。
#
# 每轮在 <out> 目录下产出三份 JSON（<stem> = 输入文件名去扩展）：
#   iter_N_<stem>_tts.json          -> TTS 后
#   iter_N_<stem>_tts_asr.json      -> ASR 后
#   iter_N_<stem>_tts_asr_wer.json  -> WER 后（带 wer_pass / tts_wer）
# 全部 turn 通过 或 达到 max_iter 后停止；最终结果即最后一轮的 iter_N_*_tts_asr_wer.json。
#
# 用法：
#   ./run.sh [选项]
#
# 选项（也可用同名环境变量覆盖默认值）：
#   -i, --input PATH         初始输入 JSON（必填）
#   -o, --out DIR            输出目录（默认 <code>/run_output）
#   -n, --max-iter N         最大循环次数（默认 5）
#   -t, --wer-threshold F    WER 阈值，分数形式，如 0.1 = 10%（默认 0.1）
#   -l, --lang MODE          WER 计算：zh / zh_syllable / en（默认 zh_syllable）
#       --tts-config PATH    TTS 模板配置（默认 <code>/config_data/config_tts.yaml）
#       --asr-config PATH    ASR 模板配置（默认 <code>/config_data/config_asr.yaml）
#   -h, --help               显示本帮助并退出
#
# 示例：
#   ./run.sh --input a.json --out ./out --max-iter 3 --wer-threshold 0.1 --lang en

set -euo pipefail

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$CODE_DIR/scripts"

MAX_ITER="${MAX_ITER:-2}"
WER_THRESHOLD="${WER_THRESHOLD:-0.1}"
WER_WORKERS="${WER_WORKERS:-24}"
LANG_MODE="${LANG_MODE:-zh_syllable}"
TTS_CONFIG="${TTS_CONFIG:-$CODE_DIR/config_data/config_tts.yaml}"
ASR_CONFIG="${ASR_CONFIG:-$CODE_DIR/config_data/config_asr.yaml}"
BASE_OUT="${BASE_OUT:-$CODE_DIR/run_output}"
INPUT_JSON="${INPUT_JSON:-}"

print_help() {
  sed -n '2,29p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

# 检查 JSON 文件是否完整可解析
json_is_valid() {
  python3 -c "import json,sys; json.load(open(sys.argv[1])); print('ok')" "$1" 2>/dev/null | grep -q ok
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--input)          INPUT_JSON="$2"; shift 2 ;;
    -o|--out)            BASE_OUT="$2"; shift 2 ;;
    -n|--max-iter)       MAX_ITER="$2"; shift 2 ;;
    -t|--wer-threshold)  WER_THRESHOLD="$2"; shift 2 ;;
    -l|--lang)           LANG_MODE="$2"; shift 2 ;;
    --tts-config)        TTS_CONFIG="$2"; shift 2 ;;
    --asr-config)        ASR_CONFIG="$2"; shift 2 ;;
    -h|--help)           print_help; exit 0 ;;
    --input=*)           INPUT_JSON="${1#*=}"; shift ;;
    --out=*)             BASE_OUT="${1#*=}"; shift ;;
    --max-iter=*)        MAX_ITER="${1#*=}"; shift ;;
    --wer-threshold=*)   WER_THRESHOLD="${1#*=}"; shift ;;
    --lang=*)            LANG_MODE="${1#*=}"; shift ;;
    --tts-config=*)      TTS_CONFIG="${1#*=}"; shift ;;
    --asr-config=*)      ASR_CONFIG="${1#*=}"; shift ;;
    *) echo "未知参数: $1" >&2; print_help; exit 1 ;;
  esac
done

if [[ -z "$INPUT_JSON" ]]; then
  echo "错误：必须通过 --input / -i 指定初始输入 JSON。" >&2
  print_help
  exit 1
fi

mkdir -p "$BASE_OUT"
STEM="$(basename "$INPUT_JSON")"; STEM="${STEM%.*}"

echo "======================================================================"
echo "配置："
echo "  MAX_ITER      = $MAX_ITER"
echo "  WER_THRESHOLD = $WER_THRESHOLD"
echo "  LANG_MODE     = $LANG_MODE"
echo "  INPUT_JSON    = $INPUT_JSON"
echo "  BASE_OUT      = $BASE_OUT"
echo "  STEM          = $STEM"
echo "  TTS_CONFIG    = $TTS_CONFIG"
echo "  ASR_CONFIG    = $ASR_CONFIG"
echo "======================================================================"

current_input="$INPUT_JSON"
final_wer_json=""

for (( i=1; i<=MAX_ITER; i++ )); do
  TTS_OUT="$BASE_OUT/iter_${i}_${STEM}_tts.json"
  ASR_OUT="$BASE_OUT/iter_${i}_${STEM}_tts_asr.json"
  WER_OUT="$BASE_OUT/iter_${i}_${STEM}_tts_asr_wer.json"

  echo ""
  echo "########## 第 $i / $MAX_ITER 轮 ##########"
  echo "本轮输入: $current_input"

  if [[ -s "$WER_OUT" ]] && json_is_valid "$WER_OUT"; then
    # WER 存在 → 本轮已完整跑过，跳过
    echo "检测到已有 WER 结果，跳过本轮：$WER_OUT"
    final_wer_json="$WER_OUT"
    current_input="$WER_OUT"
    echo "本轮跳过（已完成）。"
  elif [[ -s "$ASR_OUT" ]] && json_is_valid "$ASR_OUT"; then
    # ASR 已完成但 WER 没跑完 → 只跑 WER
    echo "检测到已有 ASR 结果，跳过 TTS/ASR，直接执行 WER..."
    echo "---- [iter $i] WER (阈值=$WER_THRESHOLD, 语言=$LANG_MODE, workers=$WER_WORKERS) -> $WER_OUT ----"
    python3 "$SCRIPTS_DIR/wer_filter_mp.py" \
      --input "$ASR_OUT" \
      --output "$WER_OUT" \
      --threshold "$WER_THRESHOLD" \
      --lang "$LANG_MODE" \
      --workers "$WER_WORKERS"
    final_wer_json="$WER_OUT"
    current_input="$WER_OUT"
    echo "本轮部分跳过（WER 已补跑完成）。"
  elif [[ -s "$TTS_OUT" ]] && json_is_valid "$TTS_OUT"; then
    # TTS 已完成但 ASR 没跑完 → 从 ASR 开始
    echo "检测到已有 TTS 结果，跳过 TTS，直接从 ASR 开始..."
    echo "---- [iter $i] ASR -> $ASR_OUT ----"
    python3 "$SCRIPTS_DIR/asr_pipeline.py" \
      --config "$ASR_CONFIG" \
      --input "$TTS_OUT" \
      --output "$ASR_OUT"

    echo "---- [iter $i] WER (阈值=$WER_THRESHOLD, 语言=$LANG_MODE, workers=$WER_WORKERS) -> $WER_OUT ----"
    python3 "$SCRIPTS_DIR/wer_filter_mp.py" \
      --input "$ASR_OUT" \
      --output "$WER_OUT" \
      --threshold "$WER_THRESHOLD" \
      --lang "$LANG_MODE" \
      --workers "$WER_WORKERS"
    final_wer_json="$WER_OUT"
    current_input="$WER_OUT"
    echo "本轮部分跳过（ASR+WER 已补跑完成）。"
  else
    # 本轮尚未开始，从头跑
    echo "---- [iter $i] TTS -> $TTS_OUT ----"
    python3 "$SCRIPTS_DIR/tts_pipeline.py" \
      --config "$TTS_CONFIG" \
      --input "$current_input" \
      --output "$TTS_OUT"

    if [[ ! -s "$TTS_OUT" ]]; then
      echo "错误：TTS 未产出 $TTS_OUT" >&2
      exit 1
    fi

    echo "---- [iter $i] ASR -> $ASR_OUT ----"
    python3 "$SCRIPTS_DIR/asr_pipeline.py" \
      --config "$ASR_CONFIG" \
      --input "$TTS_OUT" \
      --output "$ASR_OUT"

    echo "---- [iter $i] WER (阈值=$WER_THRESHOLD, 语言=$LANG_MODE, workers=$WER_WORKERS) -> $WER_OUT ----"
    python3 "$SCRIPTS_DIR/wer_filter_mp.py" \
      --input "$ASR_OUT" \
      --output "$WER_OUT" \
      --threshold "$WER_THRESHOLD" \
      --lang "$LANG_MODE" \
      --workers "$WER_WORKERS"

    final_wer_json="$WER_OUT"
    current_input="$WER_OUT"
  fi

  remaining="$(python3 - "$WER_OUT" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
n = 0
for s in data:
    for t in s.get("turns", []):
        if not (t.get("user") or "").strip():
            continue
        if t.get("wer_pass") is not True:
            n += 1
print(n)
PY
)"
  echo "剩余待重试 turn 数: $remaining"

  if [[ "$remaining" == "0" ]]; then
    echo "全部通过，提前结束循环（用了 $i 轮）。"
    break
  fi
done

# 结尾再生成一份"整段 sample 全部 turn 都通过 WER"的过滤结果
FINAL_ALL_PASS="$BASE_OUT/${STEM}_final_all_pass.json"
python3 - "$final_wer_json" "$FINAL_ALL_PASS" <<'PY'
import json, sys
src, dst = sys.argv[1:3]
with open(src, "r", encoding="utf-8") as f:
    data = json.load(f)
out = []
for s in data:
    turns = s.get("turns", [])
    if not turns:
        continue
    all_pass = True
    for t in turns:
        if not (t.get("user") or "").strip():
            continue  # 无 user 文本的 turn 视作不影响判定
        if t.get("wer_pass") is not True:
            all_pass = False
            break
    if all_pass:
        out.append(s)
with open(dst, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"整段全通过样本: {len(out)}/{len(data)} -> {dst}")
PY

echo ""
echo "======================================================================"
echo "完成。"
echo "  最新一轮 WER 结果: $final_wer_json"
echo "  整段全通过数据:    $FINAL_ALL_PASS"
echo "======================================================================"
```
<!-- /REBUILD_FILE -->

## 附录 2：scripts/tts_pipeline.py

<!-- REBUILD_FILE: scripts/tts_pipeline.py SHA256: 45ce8b919c79e3da3fe9e2fcbbfc93b38d7ba4700a72a1375abf0267782ff9ab -->
```python
"""TTS 批量合成流水线。

从 JSON 数据集读取多轮对话，调用远端 TTS 服务进行音色克隆合成，
支持断点续跑与并发，完成后把合成音频路径写回原始 JSON。
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import queue
import random
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from math import gcd
from typing import Any

import requests
import soundfile as sf
import yaml
from scipy.signal import resample_poly

logger = logging.getLogger("tts_pipeline")

JobKey = tuple[Any, int]
Job = tuple[Any, int, str, str]


@dataclass
class Config:
    base_urls: list[str]
    model_name: str
    max_workers: int
    target_sr: int
    request_timeout: int
    max_retry: int
    output_wav_subdir: str
    manifest_filename: str
    ref_list: list[dict]
    language: str
    per_url_exclusive: bool
    check_audio_length: bool
    min_sec_per_char: float
    max_sec_per_char: float
    min_audio_sec: float
    # 以下三项由 CLI 注入（--input / --output），yaml 不再配置
    json_path: str = ""
    output_dir: str = ""
    output_json_path: str = ""

    @property
    def output_wav_dir(self) -> str:
        return os.path.join(self.output_dir, self.output_wav_subdir)

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.output_dir, self.manifest_filename)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        cfg_dir = os.path.dirname(os.path.abspath(path))
        # 参考音色池：优先从外部 JSON 文件加载，否则用内联 ref_list
        ref_list_path = raw.get("ref_list_path")
        if ref_list_path:
            if not os.path.isabs(ref_list_path):
                ref_list_path = os.path.join(cfg_dir, ref_list_path)
            with open(ref_list_path, "r", encoding="utf-8") as f:
                ref_list = json.load(f)
        else:
            ref_list = raw.get("ref_list") or []
        if not ref_list:
            raise ValueError("ref_list_path 或 ref_list 至少需要提供一个非空的参考音色池")
        return cls(
            base_urls=list(raw["base_urls"]),
            model_name=raw["model_name"],
            max_workers=int(raw["max_workers"]),
            target_sr=int(raw["target_sr"]),
            request_timeout=int(raw["request_timeout"]),
            max_retry=int(raw["max_retry"]),
            output_wav_subdir=raw.get("output_wav_subdir", "wavs"),
            manifest_filename=raw.get("manifest_filename", "manifest.jsonl"),
            ref_list=list(ref_list),
            language=str(raw.get("language", "Auto")),
            per_url_exclusive=bool(raw.get("per_url_exclusive", True)),
            check_audio_length=bool(raw.get("check_audio_length", False)),
            min_sec_per_char=float(raw.get("min_sec_per_char", 0.03)),
            max_sec_per_char=float(raw.get("max_sec_per_char", 0.30)),
            min_audio_sec=float(raw.get("min_audio_sec", 0.5)),
        )


class TTSClient:
    """封装单次音色克隆合成请求，含重试与重采样。"""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        # 参考音频编码池：懒加载，命中则直接复用，未命中则编码后写回
        self._ref_audio_b64: dict[str, str] = {}
        self._ref_lock = threading.Lock()
        # 每个 base_url 同一时刻只处理一个请求：用队列作为互斥令牌池，
        # get() 阻塞到有 URL 空闲，请求结束后 put() 归还
        self._url_queue: queue.Queue[str] | None = None
        if cfg.per_url_exclusive:
            self._url_queue = queue.Queue()
            for u in cfg.base_urls:
                self._url_queue.put(u)

    @staticmethod
    def _encode_ref_audio(path: str) -> str:
        with open(path, "rb") as f:
            return "data:audio/wav;base64," + base64.b64encode(f.read()).decode()

    def _get_ref_b64(self, path: str) -> str:
        b64 = self._ref_audio_b64.get(path)
        if b64 is not None:
            return b64
        with self._ref_lock:
            b64 = self._ref_audio_b64.get(path)  # double-check，避免并发重复编码
            if b64 is None:
                b64 = self._encode_ref_audio(path)
                self._ref_audio_b64[path] = b64
            return b64

    def clone(self, text: str, out_path: str, ref_audio: str, ref_text: str) -> None:
        payload = {
            "model": self.cfg.model_name,
            "input": text,
            # "voice": "default",
            "task_type": "Base",
            "language": self.cfg.language,
            "response_format": "wav",
            "ref_audio": self._get_ref_b64(ref_audio),
            "ref_text": ref_text,
        }
        tried: set[str] = set()
        last_exc: Exception | None = None
        text_len = len(text.strip())
        length_mismatch: tuple[float, float] | None = None
        too_short: float | None = None
        for attempt in range(self.cfg.max_retry + 1):
            if self._url_queue is not None:
                base_url = self._url_queue.get()
            else:
                candidates = [u for u in self.cfg.base_urls if u not in tried]
                if not candidates:
                    candidates = self.cfg.base_urls
                    tried.clear()
                base_url = random.choice(candidates)
                tried.add(base_url)
            try:
                resp = requests.post(
                    f"{base_url}/audio/speech",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=self.cfg.request_timeout,
                )
                resp.raise_for_status()
                if resp.headers.get("content-type", "").startswith("application/json"):
                    raise RuntimeError(f"server error: {resp.text}")
                audio, sr = sf.read(io.BytesIO(resp.content))
                if sr != self.cfg.target_sr:
                    g = gcd(sr, self.cfg.target_sr)
                    audio = resample_poly(audio, self.cfg.target_sr // g, sr // g, axis=0)
                duration = len(audio) / self.cfg.target_sr
                if duration < self.cfg.min_audio_sec:
                    too_short = duration
                    logger.warning(
                        "音频时长过短(第%d次) dur=%.2fs 期望>=%.2fs，重合成：%s",
                        attempt + 1, duration, self.cfg.min_audio_sec, text,
                    )
                    continue
                if self.cfg.check_audio_length and text_len > 0:
                    spc = duration / text_len
                    if not (self.cfg.min_sec_per_char <= spc <= self.cfg.max_sec_per_char):
                        length_mismatch = (duration, spc)
                        logger.warning(
                            "音频时长与文本长度不匹配(第%d次) dur=%.2fs text_len=%d spc=%.3f 期望[%.3f,%.3f]，重合成：%s",
                            attempt + 1, duration, text_len, spc,
                            self.cfg.min_sec_per_char, self.cfg.max_sec_per_char, text,
                        )
                        continue
                sf.write(out_path, audio, self.cfg.target_sr, subtype="PCM_16")
                return
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_exc = e
                logger.warning(
                    "请求 %s 失败(第%d次)，换 URL 重试: %s", base_url, attempt + 1, e,
                )
                continue
            finally:
                if self._url_queue is not None:
                    self._url_queue.put(base_url)
        if last_exc is not None:
            raise last_exc
        if too_short is not None and length_mismatch is None:
            raise RuntimeError(
                f"音频时长过短多次失败：dur={too_short:.2f}s 期望>={self.cfg.min_audio_sec}s"
            )
        assert length_mismatch is not None
        dur, spc = length_mismatch
        raise RuntimeError(
            f"音频长度校验多次失败：dur={dur:.2f}s text_len={text_len} spc={spc:.3f} "
            f"期望[{self.cfg.min_sec_per_char},{self.cfg.max_sec_per_char}]"
        )


class ManifestStore:
    """线程安全的 manifest 写入 + 历史记录读取。"""

    def __init__(self, path: str) -> None:
        self.path = path
        self._fp = None
        self._lock = threading.Lock()

    def open(self) -> None:
        self._fp = open(self.path, "a", encoding="utf-8")

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def append(self, rec: dict) -> None:
        assert self._fp is not None, "ManifestStore not opened"
        with self._lock:
            self._fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._fp.flush()
            os.fsync(self._fp.fileno())

    def load_done(self) -> set[JobKey]:
        done: set[JobKey] = set()
        for rec in self._iter_records():
            done.add((rec["id"], rec["turn"]))
        return done

    def load_path_map(self, wav_dir: str) -> dict[JobKey, str]:
        # 后写覆盖先写：同一 (sid, turn) 的最新 manifest 记录才是当前生效的音频
        path_map: dict[JobKey, str] = {}
        for rec in self._iter_records():
            path_map[(rec["id"], rec["turn"])] = os.path.join(wav_dir, rec["file"])
        return path_map

    def load_ref_map(self) -> dict[JobKey, dict]:
        ref_map: dict[JobKey, dict] = {}
        for rec in self._iter_records():
            ref_map[(rec["id"], rec["turn"])] = {
                "audio": rec.get("ref_audio"),
                "text": rec.get("ref_text"),
            }
        return ref_map

    def load_sample_refs(self) -> dict[Any, dict]:
        refs: dict[Any, dict] = {}
        for rec in self._iter_records():
            sid = rec["id"]
            if sid in refs:
                continue
            ref_audio = rec.get("ref_audio")
            ref_text = rec.get("ref_text")
            if ref_audio is None:
                continue
            refs[sid] = {"audio": ref_audio, "text": ref_text}
        return refs

    def _iter_records(self):
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except (json.JSONDecodeError, KeyError):
                    continue


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def fmt_eta(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def build_jobs(data: list[dict], ts: str) -> tuple[list[Job], int, int]:
    """构造合成任务清单。

    跳过：wer_pass == True 的 turn（上轮已通过）。
    生成：其余含非空 user 文本的 turn（wer_pass 未设或为 False）。
    返回：jobs, n_skip_passed（已通过跳过数）, n_skip_empty（无 user 文本跳过数）。
    """
    jobs: list[Job] = []
    n_skip_passed = 0
    n_skip_empty = 0
    for item in data:
        sample_id = item.get("id")
        for turn_idx, turn in enumerate(item.get("turns", []), start=1):
            text = (turn.get("user") or "").strip()
            if not text:
                n_skip_empty += 1
                continue
            if turn.get("wer_pass") is True:
                n_skip_passed += 1
                continue
            fname = f"{ts}_{sample_id}_turn{turn_idx}_{uuid.uuid4().hex[:8]}.wav"
            jobs.append((sample_id, turn_idx, text, fname))
    return jobs, n_skip_passed, n_skip_empty


def write_augmented_json(
    data: list[dict],
    path_map: dict[JobKey, str],
    ref_map: dict[JobKey, dict],
    out_json_path: str,
) -> int:
    n = 0
    for item in data:
        sample_id = item.get("id")
        sample_ref: dict | None = None
        conflict = False
        for turn_idx, turn in enumerate(item.get("turns", []), start=1):
            key = (sample_id, turn_idx)
            # 未通过或首轮尚未评估的 turn：清掉上一轮遗留的 ASR/WER 字段，
            # 避免下游 ASR / WER 因看到旧值而跳过。
            if turn.get("wer_pass") is not True:
                turn.pop("tts_audio_asr", None)
                turn.pop("tts_wer", None)
                turn.pop("wer_pass", None)
            p = path_map.get(key)
            if p:
                turn["tts_audio_path"] = os.path.abspath(p)
                # 兼容旧格式：先清掉可能残留在 turn 里的 ref 字段
                turn.pop("tts_ref_audio", None)
                turn.pop("tts_ref_text", None)
                ref = ref_map.get(key)
                if ref is not None:
                    if sample_ref is None:
                        sample_ref = ref
                    elif (ref.get("audio") != sample_ref.get("audio")
                          or ref.get("text") != sample_ref.get("text")):
                        conflict = True
                n += 1
        if sample_ref is not None and not conflict:
            if sample_ref.get("audio") is not None:
                item["tts_ref_audio"] = sample_ref["audio"]
            if sample_ref.get("text") is not None:
                item["tts_ref_text"] = sample_ref["text"]
        elif conflict:
            # 同一 sample 内 turn 之间 ref 不一致，只能退回 turn 层
            item.pop("tts_ref_audio", None)
            item.pop("tts_ref_text", None)
            for turn_idx, turn in enumerate(item.get("turns", []), start=1):
                key = (sample_id, turn_idx)
                if not path_map.get(key):
                    continue
                ref = ref_map.get(key)
                if ref is None:
                    continue
                if ref.get("audio") is not None:
                    turn["tts_ref_audio"] = ref["audio"]
                if ref.get("text") is not None:
                    turn["tts_ref_text"] = ref["text"]
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return n


def run_pipeline(cfg: Config) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.output_wav_dir, exist_ok=True)

    with open(cfg.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    manifest = ManifestStore(cfg.manifest_path)

    sample_refs = manifest.load_sample_refs()
    for item in data:
        sid = item.get("id")
        if sid is None:
            print('id not in data')
            return
        if sid not in sample_refs:
            sample_refs[sid] = random.choice(cfg.ref_list)

    # 每个 turn 的参考音优先级：
    #   1) 输入 JSON 里 turn 自带的 tts_ref_audio + tts_ref_text（兼容旧格式）
    #   2) 输入 JSON 里 sample 外层的 tts_ref_audio + tts_ref_text（新格式）
    #   3) 上一步 sample_refs（来自 manifest 或按 sample 随机的 fallback）
    turn_refs: dict[JobKey, dict] = {}
    n_from_input = 0
    for item in data:
        sid = item.get("id")
        if sid is None:
            print('id not in data')
            return
        sample_ra = item.get("tts_ref_audio")
        sample_rt = item.get("tts_ref_text")
        # 参考音是 {audio, text} 不可分的整体（text 必须是 audio 的转写），
        # 因此按“整体成对”选取，绝不跨来源拼装 audio 与 text。
        for turn_idx, turn in enumerate(item.get("turns", []), start=1):
            turn_ra = turn.get("tts_ref_audio")
            turn_rt = turn.get("tts_ref_text")
            if turn_ra and turn_rt:
                # turn 级：audio+text 都非空才采用
                turn_refs[(sid, turn_idx)] = {"audio": turn_ra, "text": turn_rt}
                n_from_input += 1
            elif sample_ra and sample_rt:
                # sample 级：audio+text 都非空才采用
                turn_refs[(sid, turn_idx)] = {"audio": sample_ra, "text": sample_rt}
                n_from_input += 1
            else:
                # 兜底：manifest 恢复或按 sample 随机（本身即配对）
                turn_refs[(sid, turn_idx)] = sample_refs[sid]
                if turn_ra or turn_rt or sample_ra or sample_rt:
                    logger.warning(
                        "id=%s turn=%d 参考音字段不成对（audio/text 缺一），已回退到 sample 级随机音色。",
                        sid, turn_idx,
                    )
    if n_from_input:
        logger.info("发现 %d 个 turn 使用输入 JSON 中指定的参考音（tts_ref_audio/tts_ref_text）。", n_from_input)

    # 一致性校验：同一 sample（多轮对话）内所有 turn 理应共用同一音色。
    # 若检测到某 sample 落到多个不同音色，说明输入 JSON 在 turn 级指定了
    # 不一致的 tts_ref_audio，多轮音频将出现音色跳变，打 warning 提示。
    n_multi_voice = 0
    for item in data:
        sid = item.get("id")
        voices = {
            turn_refs[(sid, turn_idx)]["audio"]
            for turn_idx, _ in enumerate(item.get("turns", []), start=1)
            if (sid, turn_idx) in turn_refs
        }
        if len(voices) > 1:
            n_multi_voice += 1
            logger.warning(
                "id=%s 的多轮对话使用了 %d 种不同音色（audio=%s），多轮音色将不一致。",
                sid, len(voices), sorted(voices),
            )
    if n_multi_voice:
        logger.warning("共 %d 个 sample 存在多轮音色不一致。", n_multi_voice)

    ts = time.strftime("%Y%m%d%H%M%S")
    jobs, n_skip_passed, n_skip_empty = build_jobs(data, ts)

    # 收集失败 turn 的旧 wav 路径，成功合成前先清掉旧文件，避免残留
    stale_wavs: dict[JobKey, str] = {}
    for item in data:
        sid = item.get("id")
        if sid is None:
            print('id not in data')
            return
        for turn_idx, turn in enumerate(item.get("turns", []), start=1):
            if turn.get("wer_pass") is True:
                continue
            old_path = turn.get("tts_audio_path")
            if old_path:
                stale_wavs[(sid, turn_idx)] = old_path

    if not jobs:
        logger.info("无待合成任务（跳过已通过 %d，空文本 %d），直接输出 JSON。",
                    n_skip_passed, n_skip_empty)
        path_map = manifest.load_path_map(cfg.output_wav_dir)
        ref_map = manifest.load_ref_map()
        n_aug = write_augmented_json(data, path_map, ref_map, cfg.output_json_path)
        logger.info("已写回 %d 条音频路径 -> %s", n_aug, cfg.output_json_path)
        return

    logger.info("待合成 %d 条（跳过已通过 %d，空文本 %d），并发 %d，开始...",
                len(jobs), n_skip_passed, n_skip_empty, cfg.max_workers)

    client = TTSClient(cfg)
    manifest.open()
    ok = fail = 0
    t_start = time.time()

    def work(job: Job) -> str:
        sample_id, turn_idx, text, fname = job
        out_path = os.path.join(cfg.output_wav_dir, fname)
        ref = turn_refs[(sample_id, turn_idx)]
        client.clone(text, out_path, ref["audio"], ref["text"])
        manifest.append({
            "id": sample_id,
            "turn": turn_idx,
            "file": fname,
            "text": text,
            "ref_audio": ref["audio"],
            "ref_text": ref["text"],
            "ts": ts,
        })
        # 新音频写入成功且 manifest 已记录后，再删除旧 wav
        old = stale_wavs.get((sample_id, turn_idx))
        if old and os.path.isfile(old) and os.path.abspath(old) != os.path.abspath(out_path):
            try:
                os.remove(old)
            except OSError as e:
                logger.warning("删除旧 wav 失败 %s: %s", old, e)
        return fname

    try:
        with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
            futures = {executor.submit(work, job): job for job in jobs}
            n = len(jobs)
            for fut in as_completed(futures):
                job = futures[fut]
                fname: str | None = None
                err: Exception | None = None
                try:
                    fname = fut.result()
                    ok += 1
                except Exception as e:
                    fail += 1
                    err = e
                done_cnt = ok + fail
                elapsed = time.time() - t_start
                rate = done_cnt / elapsed if elapsed > 0 else 0.0
                eta = (n - done_cnt) / rate if rate > 0 else 0.0
                progress = done_cnt * 100 // n
                if fname is not None:
                    logger.info(
                        "[%d/%d %d%%] OK=%d FAIL=%d %.2f req/s ETA=%s 已保存 %s",
                        done_cnt, n, progress, ok, fail, rate, fmt_eta(eta), fname,
                    )
                else:
                    logger.error(
                        "[%d/%d %d%%] OK=%d FAIL=%d %.2f req/s ETA=%s 失败 id=%s turn=%s: %s",
                        done_cnt, n, progress, ok, fail, rate, fmt_eta(eta),
                        job[0], job[1], err,
                    )
    finally:
        manifest.close()

    t_used = round(time.time() - t_start, 2)
    total = len(jobs) + n_skip_passed + n_skip_empty
    logger.info(
        "完成：成功 %d，跳过已通过 %d，跳过空文本 %d，失败 %d，共 %d，耗时 %s 秒",
        ok, n_skip_passed, n_skip_empty, fail, total, t_used,
    )

    path_map = manifest.load_path_map(cfg.output_wav_dir)
    ref_map = manifest.load_ref_map()
    n_aug = write_augmented_json(data, path_map, ref_map, cfg.output_json_path)
    logger.info("已写回 %d 条音频路径 -> %s", n_aug, cfg.output_json_path)
    logger.info("输出目录：%s", cfg.output_dir)
    logger.info("清单文件：%s", cfg.manifest_path)
    logger.info("音频路径：%s", cfg.output_wav_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TTS 批量合成流水线")
    default_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    parser.add_argument("--config", "-c", default=default_cfg, help="YAML 配置文件路径")
    parser.add_argument("--input", "-i", required=True,
                        help="输入 JSON 路径（必填）。")
    parser.add_argument("--output", "-o", required=True,
                        help="最终写回 JSON 的完整路径（必填）；wavs / manifest 会挂在其所在目录下。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    cfg = Config.from_yaml(args.config)
    cfg.json_path = args.input
    cfg.output_json_path = args.output
    cfg.output_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
```
<!-- /REBUILD_FILE -->

## 附录 3：scripts/asr_pipeline.py

<!-- REBUILD_FILE: scripts/asr_pipeline.py SHA256: 4a255bc75080a354d01a8297053201b7a8b6c39795083a3bd2d1504e50fce6e2 -->
```python
#!/usr/bin/env python3
"""Multi-threaded client for the Qwen3-ASR vLLM server.

Runtime settings (servers, model, request params, JSON field names, default
thread count) live in config.yaml next to this script. Every request randomly
picks a server from the configured list, spreading load across instances.

Usage:
    # Read from input.json, write results to output.json (both required)
    python asr_pipeline.py --input input.json --output output.json

    # Override thread count / use an alternate config file
    python asr_pipeline.py --input input.json --output output.json --threads 16 --config other.yaml
"""

import argparse
import base64
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "config.yaml"


@dataclass
class Config:
    """Runtime configuration loaded from a YAML file."""

    servers: list[str]
    model: str
    temperature: float
    max_tokens: int
    timeout: int
    default_threads: int
    audio_field: str = "tts_audio_path"
    asr_field: str = "tts_audio_asr"

    def random_base_url(self) -> str:
        return random.choice(self.servers)


def load_config(path: str | Path) -> Config:
    """Load and validate configuration from a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    req = raw.get("request", {})
    fields = raw.get("json_fields", {})

    return Config(
        servers=list(raw["servers"]),
        model=raw["model"],
        temperature=float(req.get("temperature", 0.0)),
        max_tokens=int(req.get("max_tokens", 2048)),
        timeout=int(req.get("timeout", 120)),
        default_threads=int(raw.get("threads", 8)),
        audio_field=fields.get("audio_path", "tts_audio_path"),
        asr_field=fields.get("asr_result", "tts_audio_asr"),
    )


# Global config; populated by main() before any worker runs.
CONFIG: Config = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
_encode_cache: dict[str, tuple[str, str]] = {}
_encode_lock = threading.Lock()


def encode_audio(path: str) -> tuple[str, str]:
    """Encode audio file to base64; result is cached to avoid re-reading the same file.

    Only the cache dict access is under the lock — the actual file read and
    base64 encoding run outside so I/O and CPU work stay parallel across threads.
    """
    with _encode_lock:
        cached = _encode_cache.get(path)
    if cached is not None:
        return cached
    fmt = path.rsplit(".", 1)[-1].lower()
    with open(path, "rb") as f:
        result = (base64.b64encode(f.read()).decode(), fmt)
    with _encode_lock:
        # Another thread may have raced and inserted it — keep whichever won.
        return _encode_cache.setdefault(path, result)


def parse_asr(raw: str) -> tuple[str | None, str]:
    m = re.match(r"\s*language\s+(?P<lang>[^<]+?)\s*<asr_text>(?P<text>.*)", raw, re.S)
    if m:
        return m.group("lang").strip(), m.group("text").strip()
    return None, raw.strip()


def build_messages(b64: str, fmt: str, context: str = "", language: str | None = None):
    system = context
    if language:
        system = (system + f"\nDetected language: {language}").strip()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "input_audio", "input_audio": {"data": b64, "format": fmt}}
            ],
        }
    )
    return messages


# ---------------------------------------------------------------------------
# Single-request transcription (random URL each call)
# ---------------------------------------------------------------------------
def transcribe(audio_path: str, context: str = "", language: str | None = None) -> dict:
    import requests

    base_url = CONFIG.random_base_url()
    t0 = time.time()
    try:
        b64, fmt = encode_audio(audio_path)
        payload = {
            "model": CONFIG.model,
            "messages": build_messages(b64, fmt, context, language),
            "temperature": CONFIG.temperature,
            "max_tokens": CONFIG.max_tokens,
        }
        r = requests.post(f"{base_url}/chat/completions", json=payload, timeout=CONFIG.timeout)
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"]
        lang, text = parse_asr(raw)
        return {
            "audio": audio_path,
            "base_url": base_url,
            "language": lang,
            "text": text,
            "raw": raw,
            "elapsed": round(time.time() - t0, 3),
            "error": None,
        }
    except Exception as exc:
        return {
            "audio": audio_path,
            "base_url": base_url,
            "language": None,
            "text": None,
            "raw": None,
            "elapsed": round(time.time() - t0, 3),
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# JSON processing (with real-time save + resume)
# ---------------------------------------------------------------------------
def save_json_atomic(data, path: str) -> None:
    """Write JSON to a temp file then atomically replace the target."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def run_json_batch(input_path: str, output_path: str, num_threads: int,
                   context: str = "") -> None:
    """Transcribe every turn's audio-path field in a JSON file.

    Results are appended in real time to a ``<output_path>.jsonl`` sidecar
    (one line per completed turn). No full-JSON writes happen mid-run.
    After all pending tasks finish, the sidecar is merged back into the main
    JSON exactly once and then removed. If the run is interrupted, the sidecar
    acts as the resume source next time.
    """
    jsonl_path = output_path + ".jsonl"
    audio_field = CONFIG.audio_field
    asr_field = CONFIG.asr_field

    source = output_path if os.path.exists(output_path) else input_path
    with open(source, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Overlay any incremental results left behind by a previous interrupted run.
    jsonl_recovered = 0
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # partial trailing line from a crash — ignore
                ri = rec.get("ri")
                ti = rec.get("ti")
                text = rec.get("text")
                if ri is None or ti is None or text is None:
                    continue
                data[ri]["turns"][ti][asr_field] = text
                jsonl_recovered += 1

    pending: list[tuple[int, int, str]] = []
    done = 0
    for ri, record in enumerate(data):
        for ti, turn in enumerate(record.get("turns", [])):
            audio = turn.get(audio_field)
            if not audio:
                continue
            # 上一轮已通过 WER：直接跳过，无需重跑 ASR
            if turn.get("wer_pass") is True:
                done += 1
                continue
            if turn.get(asr_field):
                done += 1
                continue
            pending.append((ri, ti, audio))

    total = len(pending)
    print(f"Input    : {input_path}", flush=True)
    print(f"Output   : {output_path}", flush=True)
    print(f"Sidecar  : {jsonl_path}", flush=True)
    print(f"Resume   : {done} done ({jsonl_recovered} recovered from sidecar), "
          f"{total} pending", flush=True)
    print("-" * 72, flush=True)

    if total == 0:
        if jsonl_recovered > 0 or source != output_path:
            save_json_atomic(data, output_path)
        if os.path.exists(jsonl_path):
            os.remove(jsonl_path)
        print("Nothing to do.", flush=True)
        return

    t_start = time.time()
    counter = 0

    # Only the main thread writes to the sidecar (inside the as_completed loop),
    # so no lock is needed. Line-buffered so each result is flushed as it lands.
    jsonl_f = open(jsonl_path, "a", encoding="utf-8", buffering=1)
    try:
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            future_to_loc = {
                pool.submit(transcribe, audio, context): (ri, ti)
                for ri, ti, audio in pending
            }
            for future in as_completed(future_to_loc):
                ri, ti = future_to_loc[future]
                result = future.result()
                text = result["text"]
                data[ri]["turns"][ti][asr_field] = text
                jsonl_f.write(json.dumps(
                    {"ri": ri, "ti": ti, "text": text, "error": result["error"]},
                    ensure_ascii=False,
                ) + "\n")
                counter += 1
                status = "OK " if result["error"] is None else "ERR"
                elapsed = time.time() - t_start
                rate = counter / elapsed if elapsed > 0 else 0.0
                eta_s = (total - counter) / rate if rate > 0 else 0
                eta = str(timedelta(seconds=int(eta_s)))
                print(
                    f"[{counter:4d}/{total}] [{status}] [{result['elapsed']:6.2f}s] "
                    f"{rate:5.1f}r/s eta {eta}  "
                    f"{result['base_url']}  "
                    f"{os.path.basename(result['audio'])}"
                    + (f"  -> {result['text']}" if result["text"]
                       else f"  ERROR: {result['error']}"),
                    flush=True,
                )
    finally:
        jsonl_f.close()

    # Final merge: one full-file write, then drop the sidecar.
    print("-" * 72, flush=True)
    print("Merging sidecar into final JSON...", flush=True)
    t0 = time.time()
    save_json_atomic(data, output_path)
    os.remove(jsonl_path)
    print(f"  merge saved in {time.time() - t0:.2f}s", flush=True)
    print(f"Done. Wall: {time.time() - t_start:.2f}s  Saved to {output_path}",
          flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    global CONFIG

    # Force line-buffered stdout even when piped through `tee`.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Multi-threaded Qwen3-ASR client")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH),
                        help=f"Path to the YAML config file (default: {DEFAULT_CONFIG_PATH})")
    parser.add_argument("--threads", "-t", type=int, default=None,
                        help="Number of worker threads (default: from config)")
    parser.add_argument("--context", "-c", default="",
                        help="Optional hotword / system context sent to the model")
    parser.add_argument("--input", "-i", required=True,
                        help="Input JSON file to transcribe (audio-path fields), with "
                             "real-time save & resume.")
    parser.add_argument("--output", "-o", required=True,
                        help="Output JSON file. If it exists, it is the resume source.")
    args = parser.parse_args()

    CONFIG = load_config(args.config)
    threads = args.threads if args.threads is not None else CONFIG.default_threads

    print(f"Config   : {args.config}", flush=True)
    print(f"Servers  : {CONFIG.servers}", flush=True)
    print(f"Threads  : {threads}", flush=True)

    run_json_batch(args.input, args.output, threads, args.context)


if __name__ == "__main__":
    main()
```
<!-- /REBUILD_FILE -->

## 附录 4：scripts/wer_filter_mp.py

<!-- REBUILD_FILE: scripts/wer_filter_mp.py SHA256: 2e04a8f3ea8daf101237c7e17316165e1e908e23c7e1c5dda80ed48b553fbd72 -->
```python
"""多进程版 WER 打标：按 turn 并行计算 WER，语义与 wer_filter.py 完全一致。

设计要点：
  - ProcessPoolExecutor + initializer：每个 worker 只 import 一次 cer_one，
    避开 tn.chinese.normalizer / pypinyin 反复初始化的开销。
  - 使用多个 WER 库计算，取最小值作为最终 WER（更宽容，减少误判 fail）。
  - 主进程只做 JSON 遍历和结果回写，全部 CPU 密集的 WER 计算下放给 worker。
  - worker 内可选 --quiet：把 cer_one 里 print 出来的噪声吞掉，只保留返回值。

用法与 wer_filter.py 相同，多一个 --workers（默认取 CPU 核数-1，最少 1）。
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, Callable, Dict, List, Tuple

TOOLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools")
sys.path.insert(0, os.path.abspath(TOOLS_DIR))
# WER computation libraries — all imported at module level,
# available to each worker via multiprocessing pickle.
from zh.compute_wer_jsonl import cer_one as zh_cer_one
from en.compute_wer_jsonl import cer_one as en_cer_one
from zh.compute_syllable_wer_jsonl import cer_one as syllabel_cer_one
from other.compute_wer_jiwer import cer_one as jiwer_cer_one

# 根据 --lang 选择要运行的库列表（模块级常量，worker 可访问）
LANG_LIBS: Dict[str, List[Tuple[str, Callable]]] = {
    "zh":          [("zh_char", zh_cer_one), ("zh_syllable", syllabel_cer_one), ("jiwer", jiwer_cer_one)],
    "zh_syllable": [("zh_char", zh_cer_one), ("zh_syllable", syllabel_cer_one), ("jiwer", jiwer_cer_one)],
    "en":          [("en_word", en_cer_one), ("zh_char", zh_cer_one), ("jiwer", jiwer_cer_one)],
}

# worker 进程内的全局缓存，由 initializer 填充。
_CER_FUNCS: List[Tuple[str, Callable]] = []
_QUIET = False


def _worker_init(lang: str, quiet: bool) -> None:
    """每个子进程启动时调用一次，把使用的库列表缓存到全局。"""
    global _CER_FUNCS, _QUIET
    funcs = LANG_LIBS.get(lang)
    if funcs is None:
        raise ValueError(f"unknown lang: {lang}")
    if quiet:
        # 静默 import 阶段的噪声（有的库在 import 时也会打印）
        with redirect_stdout(StringIO()):
            _CER_FUNCS = funcs
    else:
        _CER_FUNCS = funcs
    _QUIET = quiet


def _worker_compute(job: Tuple[int, str, str]) -> Tuple[int, float, str, Dict[str, float]]:
    """job = (job_id, ref, hyp) -> (job_id, min_wer_pct, source_name, all_scores)。"""
    job_id, ref, hyp = job
    assert _CER_FUNCS is not None, "worker not initialized"
    ref_safe = ref or ""
    hyp_safe = hyp or ""

    scores: Dict[str, float] = {}
    best_wer = float("inf")
    best_name = "none"

    for name, func in _CER_FUNCS:
        try:
            if _QUIET:
                buf = StringIO()
                with redirect_stdout(buf):
                    result = func(ref_safe, hyp_safe)
            else:
                result = func(ref_safe, hyp_safe)
            wer_val = float(result.get("wer", 0.0))
        except Exception:
            wer_val = float("inf")   # 计算失败，跳过该库
        scores[name] = wer_val
        if wer_val < best_wer:
            best_wer = wer_val
            best_name = name

    if best_wer == float("inf"):      # 所有库都失败
        best_wer = 0.0
        best_name = _CER_FUNCS[0][0]

    return job_id, best_wer, best_name, scores


def _collect_jobs(
    data: List[Dict[str, Any]],
    ref_field: str,
    audio_field: str,
    asr_field: str,
) -> Tuple[List[Tuple[int, str, str]], List[Tuple[int, int]], Dict[str, int]]:
    """遍历 JSON，把所有真正需要跑 WER 的 turn 收集成 job 列表。

    返回：
      jobs        —— List[(job_id, ref, hyp)]
      locations   —— job_id -> (sample_idx, turn_idx) 的映射（用 list 表达）
      counters    —— 直接可判定的 skip / fail 计数（noref / no-audio / prev-pass）
    """
    jobs: List[Tuple[int, str, str]] = []
    locations: List[Tuple[int, int]] = []
    counters = {"skip_pass": 0, "skip_noref": 0, "direct_fail": 0}

    for si, sample in enumerate(data):
        for ti, turn in enumerate(sample.get("turns", [])):
            if turn.get("wer_pass") is True:
                counters["skip_pass"] += 1
                continue
            ref = (turn.get(ref_field) or "").strip()
            hyp = (turn.get(asr_field) or "").strip()
            audio = turn.get(audio_field)
            if not ref:
                turn["wer_pass"] = True
                counters["skip_noref"] += 1
                continue
            if not audio or hyp == "":
                turn["wer_pass"] = False
                counters["direct_fail"] += 1
                continue
            job_id = len(jobs)
            jobs.append((job_id, ref, hyp))
            locations.append((si, ti))

    return jobs, locations, counters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="TTS+ASR 之后的 JSON")
    ap.add_argument("--output", required=True, help="打上 wer_pass 的 JSON 输出路径")
    ap.add_argument("--threshold", type=float, default=0.1,
                    help="WER 阈值（分数，如 0.1 表示 10%%）；WER < 阈值算 pass")
    ap.add_argument("--lang", default="zh_syllable",
                    choices=["zh", "zh_syllable", "en"],
                    help="WER 计算方式")
    ap.add_argument("--ref-field", default="user")
    ap.add_argument("--audio-field", default="tts_audio_path")
    ap.add_argument("--asr-field", default="tts_audio_asr")
    ap.add_argument("--wer-field", default="tts_wer",
                    help="写回到 turn 里的 WER 分数字段名（分数，非百分比）")
    ap.add_argument("--record-all-scores", action="store_true", default=False,
                    help="记录所有 WER 库的分数到 tts_wer_scores 字段")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="并行进程数，默认 CPU 核数-1")
    ap.add_argument("--chunk-size", type=int, default=32,
                    help="imap 每批投递给 worker 的任务数，默认 32")
    ap.add_argument("--quiet", dest="quiet", action="store_true", default=True,
                    help="静默 cer_one 内部的 print（默认开启）")
    ap.add_argument("--verbose", dest="quiet", action="store_false",
                    help="关闭静默，保留 cer_one 的原始 print")
    ap.add_argument("--mp-context", default="spawn",
                    choices=["spawn", "fork", "forkserver"],
                    help="multiprocessing 启动方式，默认 spawn（更稳）")
    args = ap.parse_args()

    threshold_pct = args.threshold * 100.0

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    jobs, locations, counters = _collect_jobs(
        data, args.ref_field, args.audio_field, args.asr_field,
    )

    n_total = len(jobs) + counters["direct_fail"]
    n_pass = 0
    n_fail = counters["direct_fail"]

    if jobs:
        workers = max(1, min(args.workers, len(jobs)))
        ctx = mp.get_context(args.mp_context)
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(args.lang, args.quiet),
        ) as pool:
            # 用 map 保证按 job_id 顺序返回也可以，但 imap_unordered 更好用；
            # 我们已经通过 job_id 记录了位置，返回顺序无所谓。
            futures = [
                pool.submit(_worker_compute, job)
                for job in jobs
            ]
            done = 0
            total = len(futures)
            for fut in as_completed(futures):
                job_id, wer_pct, source_name, all_scores = fut.result()
                si, ti = locations[job_id]
                turn = data[si]["turns"][ti]
                turn[args.wer_field] = wer_pct / 100.0
                turn["tts_wer_source"] = source_name
                if args.record_all_scores:
                    turn["tts_wer_scores"] = {k: v / 100.0 for k, v in all_scores.items()}
                if wer_pct < threshold_pct:
                    turn["wer_pass"] = True
                    n_pass += 1
                else:
                    turn["wer_pass"] = False
                    n_fail += 1
                done += 1
                if done % max(1, args.chunk_size) == 0 or done == total:
                    print(f"[wer_filter_mp] progress {done}/{total}", file=sys.stderr)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(
        f"[wer_filter_mp] threshold={args.threshold} ({threshold_pct:.1f}%) "
        f"lang={args.lang} workers={args.workers} "
        f"total={n_total} pass={n_pass} fail={n_fail} "
        f"skip(prev-pass)={counters['skip_pass']} "
        f"skip(no-ref)={counters['skip_noref']} "
        f"direct-fail={counters['direct_fail']}"
        f" (多库取最小WER) -> {args.output}"
    )


if __name__ == "__main__":
    main()
```
<!-- /REBUILD_FILE -->

## 附录 5：scripts/wer_filter.py

<!-- REBUILD_FILE: scripts/wer_filter.py SHA256: 13e82a9d1bfbef38e19e822148604d364eab3442bebfa744db8dd2952640844a -->
```python
"""按 WER 阈值给每个 turn 打上 wer_pass 标记。

使用多个 WER 库计算，取最小值作为最终 WER（更宽容，减少误判 fail）。

输入：TTS + ASR 之后的 JSON（每个 turn 含 user / tts_audio_path / tts_audio_asr）。
输出：与输入结构一致的 JSON，只是每个 turn 新增/更新字段：
  - tts_wer:          0-1 之间的分数（所有库中最小值）
  - tts_wer_source:   产生最小值的库名
  - tts_wer_scores:   所有库的详细分数（仅 --record-all-scores 时写入）
  - wer_pass:         true / false

跳过规则：
  - 已 wer_pass == true 的 turn 直接透传（不会覆盖为 false）
  - 无 user 参考文本的 turn 视作 pass（避免死循环）
  - 缺 tts_audio_path 或 tts_audio_asr 的 turn 视作 false（会在下轮 TTS 重合成）

不会删除任何音频、不会清空任何字段：下一轮 TTS 会负责清理失败音频与旧 ASR 结果。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, Callable, Dict, List, Tuple

TOOLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools")


def _import_cer_funcs(lang: str) -> List[Tuple[str, Callable]]:
    """根据 lang 返回 [(name, cer_one_function), ...] 列表。"""
    sys.path.insert(0, os.path.abspath(TOOLS_DIR))
    from zh.compute_wer_jsonl import cer_one as zh_fn
    from en.compute_wer_jsonl import cer_one as en_fn
    from zh.compute_syllable_wer_jsonl import cer_one as syllable_fn
    from other.compute_wer_jiwer import cer_one as jiwer_fn

    groups: Dict[str, List[Tuple[str, Callable]]] = {
        "zh":         [("zh_char", zh_fn), ("zh_syllable", syllable_fn), ("jiwer", jiwer_fn)],
        "zh_syllable": [("zh_char", zh_fn), ("zh_syllable", syllable_fn), ("jiwer", jiwer_fn)],
        "en":         [("en_word", en_fn), ("zh_char", zh_cer_one), ("jiwer", jiwer_fn)],
    }
    funcs = groups.get(lang)
    if funcs is None:
        raise ValueError(f"unknown lang: {lang}")
    return funcs


def compute_wer_multi(
    cer_funcs: List[Tuple[str, Callable]],
    ref: str, hyp: str,
    quiet: bool = True,
) -> Tuple[float, str, Dict[str, float]]:
    """用所有 WER 库计算，返回 (最小WER百分比, 来源库名, 所有分数dict)。"""
    ref_safe = ref or ""
    hyp_safe = hyp or ""
    scores: Dict[str, float] = {}
    best_wer = float("inf")
    best_name = "none"

    for name, func in cer_funcs:
        try:
            if quiet:
                buf = StringIO()
                with redirect_stdout(buf):
                    result = func(ref_safe, hyp_safe)
            else:
                result = func(ref_safe, hyp_safe)
            wer_val = float(result.get("wer", 0.0))
        except Exception:
            wer_val = float("inf")   # 计算失败，跳过
        scores[name] = wer_val
        if wer_val < best_wer:
            best_wer = wer_val
            best_name = name

    if best_wer == float("inf"):          # 所有库都失败
        best_wer = 0.0
        best_name = cer_funcs[0][0]

    return best_wer, best_name, scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="TTS+ASR 之后的 JSON")
    ap.add_argument("--output", required=True, help="打上 wer_pass 的 JSON 输出路径")
    ap.add_argument("--threshold", type=float, default=0.1,
                    help="WER 阈值（分数，如 0.1 表示 10%%）；WER < 阈值算 pass")
    ap.add_argument("--lang", default="zh_syllable",
                    choices=["zh", "zh_syllable", "en"],
                    help="WER 计算方式")
    ap.add_argument("--ref-field", default="user")
    ap.add_argument("--audio-field", default="tts_audio_path")
    ap.add_argument("--asr-field", default="tts_audio_asr")
    ap.add_argument("--wer-field", default="tts_wer",
                    help="写回到 turn 里的 WER 分数字段名（分数，非百分比）")
    ap.add_argument("--quiet", dest="quiet", action="store_true", default=True,
                    help="静默 cer_one 内部的 print（默认开启）")
    ap.add_argument("--verbose", dest="quiet", action="store_false",
                    help="关闭静默，保留 cer_one 的原始 print")
    ap.add_argument("--record-all-scores", action="store_true", default=False,
                    help="记录所有 WER 库的分数到 tts_wer_scores 字段")
    args = ap.parse_args()

    cer_funcs = _import_cer_funcs(args.lang)
    threshold_pct = args.threshold * 100.0

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    n_total = n_pass = n_fail = n_skip_pass = n_skip_noref = 0

    for sample in data:
        for turn in sample.get("turns", []):
            # 上轮已通过：透传
            if turn.get("wer_pass") is True:
                n_skip_pass += 1
                continue
            ref = (turn.get(args.ref_field) or "").strip()
            hyp = (turn.get(args.asr_field) or "").strip()
            audio = turn.get(args.audio_field)
            if not ref:
                # 无参考文本无法评估：标记为 pass，避免死循环
                turn["wer_pass"] = True
                n_skip_noref += 1
                continue
            if not audio or hyp == "":
                # 缺 TTS 音频或缺 ASR 结果 -> fail，下一轮重跑
                turn["wer_pass"] = False
                n_fail += 1
                n_total += 1
                continue
            wer_pct, source_name, all_scores = compute_wer_multi(
                cer_funcs, ref, hyp, quiet=args.quiet,
            )
            n_total += 1
            turn[args.wer_field] = wer_pct / 100.0
            turn["tts_wer_source"] = source_name
            if args.record_all_scores:
                turn["tts_wer_scores"] = {k: v / 100.0 for k, v in all_scores.items()}
            if wer_pct < threshold_pct:
                turn["wer_pass"] = True
                n_pass += 1
            else:
                turn["wer_pass"] = False
                n_fail += 1

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(
        f"[wer_filter] threshold={args.threshold} ({threshold_pct:.1f}%) lang={args.lang} "
        f"total={n_total} pass={n_pass} fail={n_fail} "
        f"skip(prev-pass)={n_skip_pass} skip(no-ref)={n_skip_noref}"
        f" (多库取最小WER) -> {args.output}"
    )


if __name__ == "__main__":
    main()
```
<!-- /REBUILD_FILE -->

## 附录 6：tools/zh/compute_wer_jsonl.py

<!-- REBUILD_FILE: tools/zh/compute_wer_jsonl.py SHA256: 8166ef7ff5056011578316175ecaa4eac1e02075e8bd734ec404179f34a36b30 -->
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import re, sys, unicodedata, json
import codecs
from tn.chinese.normalizer import Normalizer

normalizer = Normalizer()

remove_tag = True
spacelist= [' ', '\t', '\r', '\n']
#puncts = ['!', ',', '?',
#          '、', '。', '！', '，', '；', '？',
#          '：', '「', '」', '︰',  '『', '』', '《', '》']
puncts="""!"#$%&()*+,-.·/:;<=>?@，。?★、…【】《》？“”‘！[\\]^_`{|}~"""

def rm_punct(in_str):
    for p in puncts:
       in_str = in_str.replace(p, '')
    return in_str

def characterize(string) :
  string = rm_punct(string)
  res = []
  i = 0
  while i < len(string):
    char = string[i]
    if char in puncts:
      i += 1
      continue
    cat1 = unicodedata.category(char)
    #https://unicodebook.readthedocs.io/unicode.html#unicode-categories
    if cat1 == 'Zs' or cat1 == 'Cn' or char in spacelist: # space or not assigned
       i += 1
       continue
    if cat1 == 'Lo': # letter-other
       res.append(char)
       i += 1
    else:
       # some input looks like: <unk><noise>, we want to separate it to two words.
       sep = ' '
       if char == '<': sep = '>'
       j = i+1
       while j < len(string):
         c = string[j]
         if ord(c) >= 128 or (c in spacelist) or (c==sep):
           break
         j += 1
       if j < len(string) and string[j] == '>':
         j += 1
       res.append(string[i:j])
       i = j
  return res

def stripoff_tags(x):
  if not x: return ''
  chars = []
  i = 0; T=len(x)
  while i < T:
    if x[i] == '<':
      while i < T and x[i] != '>':
         i += 1
      i += 1
    else:
      chars.append(x[i])
      i += 1
  return ''.join(chars)


def normalize(sentence, ignore_words, cs, split=None):
    """ sentence, ignore_words are both in unicode
    """
    new_sentence = []
    # print(sentence)
    for token in sentence:
        x = token
        if not cs:
           x = x.upper()
        if x in ignore_words:
           continue
        if remove_tag:
          x = stripoff_tags(x)
        if not x:
          continue
        if split and x in split:
          new_sentence += split[x]
        else:
          new_sentence.append(x)
    return new_sentence


def replace_line(sentence, cs, replace=None):
    splits = sentence.replace("\t", " ").split(" ", 1)
    fid = splits[0]
    sentence = splits[1]
    # print(sentence)
    for key, value in replace.items():
        if not cs:
            key = key.upper()
            value = value.upper()
            sentence = sentence.upper()
        sentence = sentence.replace(key, value)
    return "{} {}".format(fid, sentence)

class Calculator :
  def __init__(self) :
    self.data = {}
    self.space = []
    self.cost = {}
    self.cost['cor'] = 0
    self.cost['sub'] = 1
    self.cost['del'] = 1
    self.cost['ins'] = 1
  def calculate(self, lab, rec) :
    # Initialization
    lab.insert(0, '')
    rec.insert(0, '')
    while len(self.space) < len(lab) :
      self.space.append([])
    for row in self.space :
      for element in row :
        element['dist'] = 0
        element['error'] = 'non'
      while len(row) < len(rec) :
        row.append({'dist' : 0, 'error' : 'non'})
    for i in range(len(lab)) :
      self.space[i][0]['dist'] = i
      self.space[i][0]['error'] = 'del'
    for j in range(len(rec)) :
      self.space[0][j]['dist'] = j
      self.space[0][j]['error'] = 'ins'
    self.space[0][0]['error'] = 'non'
    for token in lab :
      if token not in self.data and len(token) > 0 :
        self.data[token] = {'all' : 0, 'cor' : 0, 'sub' : 0, 'ins' : 0, 'del' : 0}
    for token in rec :
      if token not in self.data and len(token) > 0 :
        self.data[token] = {'all' : 0, 'cor' : 0, 'sub' : 0, 'ins' : 0, 'del' : 0}
    # Computing edit distance
    for i, lab_token in enumerate(lab) :
      for j, rec_token in enumerate(rec) :
        if i == 0 or j == 0 :
          continue
        min_dist = sys.maxsize
        min_error = 'none'
        dist = self.space[i-1][j]['dist'] + self.cost['del']
        error = 'del'
        if dist < min_dist :
          min_dist = dist
          min_error = error
        dist = self.space[i][j-1]['dist'] + self.cost['ins']
        error = 'ins'
        if dist < min_dist :
          min_dist = dist
          min_error = error
        if lab_token == rec_token :
          dist = self.space[i-1][j-1]['dist'] + self.cost['cor']
          error = 'cor'
        else :
          dist = self.space[i-1][j-1]['dist'] + self.cost['sub']
          error = 'sub'
        if dist < min_dist :
          min_dist = dist
          min_error = error
        self.space[i][j]['dist'] = min_dist
        self.space[i][j]['error'] = min_error
    # Tracing back
    result = {'lab':[], 'rec':[], 'all':0, 'cor':0, 'sub':0, 'ins':0, 'del':0}
    i = len(lab) - 1
    j = len(rec) - 1
    while True :
      if self.space[i][j]['error'] == 'cor' : # correct
        if len(lab[i]) > 0 :
          self.data[lab[i]]['all'] = self.data[lab[i]]['all'] + 1
          self.data[lab[i]]['cor'] = self.data[lab[i]]['cor'] + 1
          result['all'] = result['all'] + 1
          result['cor'] = result['cor'] + 1
        result['lab'].insert(0, lab[i])
        result['rec'].insert(0, rec[j])
        i = i - 1
        j = j - 1
      elif self.space[i][j]['error'] == 'sub' : # substitution
        if len(lab[i]) > 0 :
          self.data[lab[i]]['all'] = self.data[lab[i]]['all'] + 1
          self.data[lab[i]]['sub'] = self.data[lab[i]]['sub'] + 1
          result['all'] = result['all'] + 1
          result['sub'] = result['sub'] + 1
        result['lab'].insert(0, lab[i])
        result['rec'].insert(0, rec[j])
        i = i - 1
        j = j - 1
      elif self.space[i][j]['error'] == 'del' : # deletion
        if len(lab[i]) > 0 :
          self.data[lab[i]]['all'] = self.data[lab[i]]['all'] + 1
          self.data[lab[i]]['del'] = self.data[lab[i]]['del'] + 1
          result['all'] = result['all'] + 1
          result['del'] = result['del'] + 1
        result['lab'].insert(0, lab[i])
        result['rec'].insert(0, "")
        i = i - 1
      elif self.space[i][j]['error'] == 'ins' : # insertion
        if len(rec[j]) > 0 :
          self.data[rec[j]]['ins'] = self.data[rec[j]]['ins'] + 1
          result['ins'] = result['ins'] + 1
        result['lab'].insert(0, "")
        result['rec'].insert(0, rec[j])
        j = j - 1
      elif self.space[i][j]['error'] == 'non' : # starting point
        break
      else : # shouldn't reach here
        print('this should not happen , i = {i} , j = {j} , error = {error}'.format(i = i, j = j, error = self.space[i][j]['error']))
    return result
  def overall(self) :
    result = {'all':0, 'cor':0, 'sub':0, 'ins':0, 'del':0}
    for token in self.data :
      result['all'] = result['all'] + self.data[token]['all']
      result['cor'] = result['cor'] + self.data[token]['cor']
      result['sub'] = result['sub'] + self.data[token]['sub']
      result['ins'] = result['ins'] + self.data[token]['ins']
      result['del'] = result['del'] + self.data[token]['del']
    return result
  def cluster(self, data) :
    result = {'all':0, 'cor':0, 'sub':0, 'ins':0, 'del':0}
    for token in data :
      if token in self.data :
        result['all'] = result['all'] + self.data[token]['all']
        result['cor'] = result['cor'] + self.data[token]['cor']
        result['sub'] = result['sub'] + self.data[token]['sub']
        result['ins'] = result['ins'] + self.data[token]['ins']
        result['del'] = result['del'] + self.data[token]['del']
    return result
  def keys(self) :
      return list(self.data.keys())

def width(string):
  return sum(1 + (unicodedata.east_asian_width(c) in "AFW") for c in string)

def default_cluster(word) :
  # unicode_names = [ unicodedata.name(char) for char in word ]
  unicode_names = []
  for char in word:
    try:
      tmp_char = unicodedata.name(char)
      unicode_names.append(tmp_char)
    except Exception as e:
      print('char error:', e, f'===({char})===')
      continue
  for i in reversed(range(len(unicode_names))) :
    if unicode_names[i].startswith('DIGIT') :  # 1
      unicode_names[i] = 'Number'  # 'DIGIT'
    elif (unicode_names[i].startswith('CJK UNIFIED IDEOGRAPH') or
          unicode_names[i].startswith('CJK COMPATIBILITY IDEOGRAPH')) :
      # 明 / 郎
      unicode_names[i] = 'Mandarin'  # 'CJK IDEOGRAPH'
    elif (unicode_names[i].startswith('LATIN CAPITAL LETTER') or
          unicode_names[i].startswith('LATIN SMALL LETTER')) :
      # A / a
      unicode_names[i] = 'English'  # 'LATIN LETTER'
    elif unicode_names[i].startswith('HIRAGANA LETTER') :  # は こ め
      unicode_names[i] = 'Japanese'  # 'GANA LETTER'
    elif (unicode_names[i].startswith('AMPERSAND') or
          unicode_names[i].startswith('APOSTROPHE') or
          unicode_names[i].startswith('COMMERCIAL AT') or
          unicode_names[i].startswith('DEGREE CELSIUS') or
          unicode_names[i].startswith('EQUALS SIGN') or
          unicode_names[i].startswith('FULL STOP') or
          unicode_names[i].startswith('HYPHEN-MINUS') or
          unicode_names[i].startswith('LOW LINE') or
          unicode_names[i].startswith('NUMBER SIGN') or
          unicode_names[i].startswith('PLUS SIGN') or
          unicode_names[i].startswith('SEMICOLON')) :
      # & / ' / @ / ℃ / = / . / - / _ / # / + / ;
      del unicode_names[i]
    else :
      return 'Other'
  if len(unicode_names) == 0 :
      return 'Other'
  if len(unicode_names) == 1 :
      return unicode_names[0]
  for i in range(len(unicode_names)-1) :
    if unicode_names[i] != unicode_names[i+1] :
      return 'Other'
  return unicode_names[0]

def cer_one(ref_text, hyp_text, ignore_words=set(), case_sensitive=False, split=None):
  # 相当于 --char=1 --v=1 
  calculator = Calculator()
  ref_text = normalizer.normalize(ref_text)
  hyp_text = normalizer.normalize(hyp_text)
  # tochar
  ref_array = characterize(ref_text)
  hyp_array = characterize(hyp_text)

  ref = normalize(ref_array, ignore_words, case_sensitive, split)
  hyp = normalize(hyp_array, ignore_words, case_sensitive, split)

  # 计算cer结果
  result = calculator.calculate(ref, hyp)

  if result['all'] != 0 :
    wer = float(result['ins'] + result['sub'] + result['del']) * 100.0 / result['all']
  else :
    wer = 0.0
    ser = 0.0
  print('Overall -> %4.2f %%' % wer, end = ' ')
  print('N=%d C=%d S=%d D=%d I=%d' %
        (result['all'], result['cor'], result['sub'], result['del'], result['ins']))
  result['wer'] = wer

  return result

def usage() :
  print("compute-wer.py : compute word error rate (WER) and align recognition results and references.")
  print("         usage : python compute-wer.py [--cs={0,1}] [--cluster=foo] [--ig=ignore_file] [--char={0,1}] [--v={0,1}] [--padding-symbol={space,underline}] test.ref test.hyp > test.wer")


if __name__ == '__main__':
  if len(sys.argv) == 1 :
    usage()
    sys.exit(0)
  calculator = Calculator()
  cluster_file = ''
  ignore_words = set()
  tochar = False
  verbose= 1
  padding_symbol= ' '
  case_sensitive = False
  max_words_per_line = sys.maxsize
  split = None
  replace = None
  while len(sys.argv) > 4:
     a = '--maxw='
     if sys.argv[1].startswith(a):
        b = sys.argv[1][len(a):]
        del sys.argv[1]
        max_words_per_line = int(b)
        continue
     a = '--rt='
     if sys.argv[1].startswith(a):
        b = sys.argv[1][len(a):].lower()
        del sys.argv[1]
        remove_tag = (b == 'true') or (b != '0')
        continue
     a = '--cs='
     if sys.argv[1].startswith(a):
        b = sys.argv[1][len(a):].lower()
        del sys.argv[1]
        case_sensitive = (b == 'true') or (b != '0')
        continue
     a = '--cluster='
     if sys.argv[1].startswith(a):
       cluster_file = sys.argv[1][len(a):]
       del sys.argv[1]
       continue
     a = '--splitfile='
     if sys.argv[1].startswith(a):
       split_file = sys.argv[1][len(a):]
       del sys.argv[1]
       split = dict()
       with codecs.open(split_file, 'r', 'utf-8') as fh:
         for line in fh:  # line in unicode
           words = line.strip().split()
           if len(words) >= 2:
             split[words[0]] = words[1:]
       continue
     a = '--ig='
     if sys.argv[1].startswith(a):
       ignore_file = sys.argv[1][len(a):]
       del sys.argv[1]
       with codecs.open(ignore_file, 'r', 'utf-8') as fh:
         for line in fh:  # line in unicode
           line = line.strip()
           if len(line) > 0:
             ignore_words.add(line)
       continue
     a = '--char='
     if sys.argv[1].startswith(a):
        b = sys.argv[1][len(a):].lower()
        del sys.argv[1]
        tochar = (b == 'true') or (b != '0')
        continue
     a = '--v='
     if sys.argv[1].startswith(a):
        b = sys.argv[1][len(a):].lower()
        del sys.argv[1]
        verbose=0
        try:
          verbose=int(b)
        except:
           if b == 'true' or b != '0':
              verbose = 1
        continue
     a = '--padding-symbol='
     if sys.argv[1].startswith(a):
        b = sys.argv[1][len(a):].lower()
        del sys.argv[1]
        if b == 'space':
          padding_symbol= ' '
        elif b == 'underline':
          padding_symbol= '_'
        continue
     a = '--replace='
     if sys.argv[1].startswith(a):
       replace_file = sys.argv[1][len(a):]
       del sys.argv[1]
       replace = dict()
       with codecs.open(replace_file, 'r', 'utf-8') as fh:
         for line in fh:  # line in unicode
           words = line.strip().split("->")
           if len(words) == 2:
             replace[words[0]] = words[1]
       continue
     if True or sys.argv[1].startswith('-'):
        #ignore invalid switch
        del sys.argv[1]
        continue

  if not case_sensitive:
     ig=set([w.upper() for w in ignore_words])
     ignore_words = ig

  default_clusters = {}
  default_words = {}

  ref_file = sys.argv[1]
  hyp_file = sys.argv[2]
  out_file = sys.argv[3]
  rec_set = {}
  utter_count = 0
  utter_err_count = 0
  if split and not case_sensitive:
     newsplit = dict()
     for w in split:
        words = split[w]
        for i in range(len(words)):
           words[i] = words[i].upper()
        newsplit[w.upper()] = words
     split = newsplit

  with codecs.open(hyp_file, 'r', 'utf-8') as fh:
     for line in fh:
        # print(line)
        try:
          temp = line.split()
          temp = json.loads(line.strip())
          if len(temp) < 2:
            continue
          # wav_id = temp[0]
          # text = ' '.join(temp[1:])
          wav_id = temp['uttid']
          if 'text' in temp:
            text = temp['text']
          elif 'value' in temp:
            text = temp['value']
          else:
            print('not text in ', temp)
            continue
          text = normalizer.normalize(text)
          if replace:
            text = replace_line(text, case_sensitive, replace)
          if tochar:
            # print(line)
            array = characterize(text)
          else:
            array = text.strip().split()

          rec_set[wav_id] = normalize(array, ignore_words, case_sensitive, split)
          # line = wav_id + ' ' + text
        except Exception as e:
          print('error:', e) 
          continue
        # ==============================
        # if replace:
        #     line = replace_line(line, case_sensitive, replace)
        # if tochar:
        #     # print(line)
        #     array = characterize(line)
        # else:
        #     array = line.strip().split()
        # # print(array)
        # if len(array)==0: continue
        # fid = array[0]
        # rec_set[fid] = normalize(array[1:], ignore_words, case_sensitive, split)
        # print(rec_set)
        # ==============================

  res_list = []
  # compute error rate on the interaction of reference file and hyp file
  for line in open(ref_file, 'r', encoding='utf-8') :
    temp = json.loads(line.strip())
    if len(temp) < 2:
      continue
    # wav_id = temp[0]
    # text = ' '.join(temp[1:])
    wav_id = temp['uttid']
    fid = wav_id
    if 'text' in temp:
      text = temp['text']
    elif 'value' in temp:
      text = temp['value']
    else:
      print('not text in ', temp)
      continue
    text = normalizer.normalize(text)
    if replace:
        text = replace_line(text, case_sensitive, replace)
    if tochar:
          array = characterize(text)
    else:
          array = text.rstrip('\n').split()
    if len(array)==0: continue
    if wav_id not in rec_set:
       continue
    lab = normalize(array, ignore_words, case_sensitive, split)
    rec = rec_set[wav_id]
    # ============================
    # line = wav_id + ' ' + text
    # if replace:
    #     line = replace_line(line, case_sensitive, replace)
    # if tochar:
    #       array = characterize(line)
    # else:
    #       array = line.rstrip('\n').split()
    # if len(array)==0: continue
    # fid = array[0]
    # if fid not in rec_set:
    #    continue
    # lab = normalize(array[1:], ignore_words, case_sensitive, split)
    # rec = rec_set[fid]
    # ============================
    # print(lab, rec) # 按字划分的列表
    # 文本正则化
    # lab = normalizer.normalize(lab)
    # rec = normalizer.normalize(rec)
    if verbose:
      print('\nutt: %s' % fid)

    for word in rec + lab :
      if word not in default_words :
         default_cluster_name = default_cluster(word)
         if default_cluster_name not in default_clusters :
           default_clusters[default_cluster_name] = {}
         if word not in default_clusters[default_cluster_name] :
           default_clusters[default_cluster_name][word] = 1
         default_words[word] = default_cluster_name

    result = calculator.calculate(lab, rec)
    utter_count += 1
    error_count = 0
    if result['all']:
      error_count = result['ins'] + result['sub'] + result['del']
      if error_count:
        utter_err_count += 1
    if verbose:
      if result['all'] != 0 :
        wer = float(result['ins'] + result['sub'] + result['del']) * 100.0 / result['all']
      else :
        wer = 0.0
      res_list.append({'uttid': fid, 'wer': round(wer, 2), 'wer_info': result})
      # res_list.append({'uttid': fid, 'wer': round(wer, 2)})
      print('WER: %4.2f %%' % wer, end = ' ')
      print('N=%d C=%d S=%d D=%d I=%d' %
          (result['all'], result['cor'], result['sub'], result['del'], result['ins']))
      space = {}
      space['lab'] = []
      space['rec'] = []
      for idx in range(len(result['lab'])) :
        len_lab = width(result['lab'][idx])
        len_rec = width(result['rec'][idx])
        length = max(len_lab, len_rec)
        space['lab'].append(length-len_lab)
        space['rec'].append(length-len_rec)
      upper_lab = len(result['lab'])
      upper_rec = len(result['rec'])
      lab1, rec1 = 0, 0
      while lab1 < upper_lab or rec1 < upper_rec:
         if verbose > 1:
             print('lab(%s):' % fid.encode('utf-8'), end = ' ')
         else:
             print('lab:', end = ' ')
         lab2 = min(upper_lab, lab1 + max_words_per_line)
         for idx in range(lab1, lab2):
           token = result['lab'][idx]
           print('{token}'.format(token = token), end = '')
           for n in range(space['lab'][idx]) :
             print(padding_symbol, end = '')
           print(' ',end='')
         print()
         if verbose > 1:
            print('rec(%s):' % fid.encode('utf-8'), end = ' ')
         else:
            print('rec:', end = ' ')
         rec2 = min(upper_rec, rec1 + max_words_per_line)
         for idx in range(rec1, rec2):
           token = result['rec'][idx]
           print('{token}'.format(token = token), end = '')
           for n in range(space['rec'][idx]) :
             print(padding_symbol, end = '')
           print(' ',end='')
         print('\n', end='\n')
         lab1 = lab2
         rec1 = rec2

  with open(out_file, 'w', encoding='u8') as f_out:
    for item in res_list:
      json_data = json.dumps(item, ensure_ascii=False)
      # 写入文件
      f_out.write(json_data + '\n')
      f_out.flush()
  if verbose:
    print('===========================================================================')
    print()

  result = calculator.overall()
  if result['all'] != 0 :
    wer = float(result['ins'] + result['sub'] + result['del']) * 100.0 / result['all']
    ser = utter_err_count * 100.0 / utter_count
  else :
    wer = 0.0
    ser = 0.0
  print('Overall -> %4.2f %%' % wer, end = ' ')
  print('N=%d C=%d S=%d D=%d I=%d' %
        (result['all'], result['cor'], result['sub'], result['del'], result['ins']))
  print('SER -> %4.2f %%' % ser, end = ' ')
  print('utt=%d err=%d' % (utter_count, utter_err_count))
  if not verbose:
     print()

  if verbose:   
   for cluster_id in default_clusters :
     result = calculator.cluster([ k for k in default_clusters[cluster_id] ])
     if result['all'] != 0 :
        wer = float(result['ins'] + result['sub'] + result['del']) * 100.0 / result['all']
     else :
        wer = 0.0
     print('%s -> %4.2f %%' % (cluster_id, wer), end = ' ')
     print('N=%d C=%d S=%d D=%d I=%d' %
          (result['all'], result['cor'], result['sub'], result['del'], result['ins']))
   if len(cluster_file) > 0 : # compute separated WERs for word clusters
     cluster_id = ''
     cluster = []
     for line in open(cluster_file, 'r', encoding='utf-8') :
       for token in line.decode('utf-8').rstrip('\n').split() :
        # end of cluster reached, like </Keyword>
        if token[0:2] == '</' and token[len(token)-1] == '>' and \
           token.lstrip('</').rstrip('>') == cluster_id :
          result = calculator.cluster(cluster)
          if result['all'] != 0 :
            wer = float(result['ins'] + result['sub'] + result['del']) * 100.0 / result['all']
          else :
            wer = 0.0
          print('%s -> %4.2f %%' % (cluster_id, wer), end = ' ')
          print('N=%d C=%d S=%d D=%d I=%d' %
                (result['all'], result['cor'], result['sub'], result['del'], result['ins']))
          cluster_id = ''
          cluster = []
        # begin of cluster reached, like <Keyword>
        elif token[0] == '<' and token[len(token)-1] == '>' and \
             cluster_id == '' :
          cluster_id = token.lstrip('<').rstrip('>')
          cluster = []
        # general terms, like WEATHER / CAR / ...
        else :
          cluster.append(token)
   print()
   print('===========================================================================')
```
<!-- /REBUILD_FILE -->

## 附录 7：tools/zh/compute_syllable_wer_jsonl.py

<!-- REBUILD_FILE: tools/zh/compute_syllable_wer_jsonl.py SHA256: 8328d82faebae0c682aa76780aa82a4d7f63c73b3c06eafb593d90b2dc1b2ad4 -->
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import re, sys, unicodedata, json
import codecs
from tn.chinese.normalizer import Normalizer
from pypinyin import pinyin, lazy_pinyin, Style

normalizer = Normalizer()

remove_tag = True
spacelist= [' ', '\t', '\r', '\n']
#puncts = ['!', ',', '?',
#          '、', '。', '！', '，', '；', '？',
#          '：', '「', '」', '︰',  '『', '』', '《', '》']
puncts="""!"#$%&()*+,-.·/:;<=>?@，。?★、…【】《》？“”‘！[\\]^_`{|}~"""

def rm_punct(in_str):
    for p in puncts:
       in_str = in_str.replace(p, '')
    return in_str

def characterize(string) :
  string = rm_punct(string)
  res = []
  i = 0
  while i < len(string):
    char = string[i]
    if char in puncts:
      i += 1
      continue
    cat1 = unicodedata.category(char)
    #https://unicodebook.readthedocs.io/unicode.html#unicode-categories
    if cat1 == 'Zs' or cat1 == 'Cn' or char in spacelist: # space or not assigned
       i += 1
       continue
    if cat1 == 'Lo': # letter-other
       res.append(char)
       i += 1
    else:
       # some input looks like: <unk><noise>, we want to separate it to two words.
       sep = ' '
       if char == '<': sep = '>'
       j = i+1
       while j < len(string):
         c = string[j]
         if ord(c) >= 128 or (c in spacelist) or (c==sep):
           break
         j += 1
       if j < len(string) and string[j] == '>':
         j += 1
       res.append(string[i:j])
       i = j
  return res

def stripoff_tags(x):
  if not x: return ''
  chars = []
  i = 0; T=len(x)
  while i < T:
    if x[i] == '<':
      while i < T and x[i] != '>':
         i += 1
      i += 1
    else:
      chars.append(x[i])
      i += 1
  return ''.join(chars)


def normalize(sentence, ignore_words, cs, split=None):
    """ sentence, ignore_words are both in unicode
    """
    new_sentence = []
    # print(sentence)
    for token in sentence:
        x = token
        if not cs:
           x = x.upper()
        if x in ignore_words:
           continue
        if remove_tag:
          x = stripoff_tags(x)
        if not x:
          continue
        if split and x in split:
          new_sentence += split[x]
        else:
          new_sentence.append(x)
    return new_sentence


def replace_line(sentence, cs, replace=None):
    splits = sentence.replace("\t", " ").split(" ", 1)
    fid = splits[0]
    sentence = splits[1]
    # print(sentence)
    for key, value in replace.items():
        if not cs:
            key = key.upper()
            value = value.upper()
            sentence = sentence.upper()
        sentence = sentence.replace(key, value)
    return "{} {}".format(fid, sentence)

class Calculator :
  def __init__(self) :
    self.data = {}
    self.space = []
    self.cost = {}
    self.cost['cor'] = 0
    self.cost['sub'] = 1
    self.cost['del'] = 1
    self.cost['ins'] = 1
  def calculate(self, lab, rec) :
    # Initialization
    lab.insert(0, '')
    rec.insert(0, '')
    while len(self.space) < len(lab) :
      self.space.append([])
    for row in self.space :
      for element in row :
        element['dist'] = 0
        element['error'] = 'non'
      while len(row) < len(rec) :
        row.append({'dist' : 0, 'error' : 'non'})
    for i in range(len(lab)) :
      self.space[i][0]['dist'] = i
      self.space[i][0]['error'] = 'del'
    for j in range(len(rec)) :
      self.space[0][j]['dist'] = j
      self.space[0][j]['error'] = 'ins'
    self.space[0][0]['error'] = 'non'
    for token in lab :
      if token not in self.data and len(token) > 0 :
        self.data[token] = {'all' : 0, 'cor' : 0, 'sub' : 0, 'ins' : 0, 'del' : 0}
    for token in rec :
      if token not in self.data and len(token) > 0 :
        self.data[token] = {'all' : 0, 'cor' : 0, 'sub' : 0, 'ins' : 0, 'del' : 0}
    # Computing edit distance
    for i, lab_token in enumerate(lab) :
      for j, rec_token in enumerate(rec) :
        if i == 0 or j == 0 :
          continue
        min_dist = sys.maxsize
        min_error = 'none'
        dist = self.space[i-1][j]['dist'] + self.cost['del']
        error = 'del'
        if dist < min_dist :
          min_dist = dist
          min_error = error
        dist = self.space[i][j-1]['dist'] + self.cost['ins']
        error = 'ins'
        if dist < min_dist :
          min_dist = dist
          min_error = error
        if lab_token == rec_token :
          dist = self.space[i-1][j-1]['dist'] + self.cost['cor']
          error = 'cor'
        else :
          dist = self.space[i-1][j-1]['dist'] + self.cost['sub']
          error = 'sub'
        if dist < min_dist :
          min_dist = dist
          min_error = error
        self.space[i][j]['dist'] = min_dist
        self.space[i][j]['error'] = min_error
    # Tracing back
    result = {'lab':[], 'rec':[], 'all':0, 'cor':0, 'sub':0, 'ins':0, 'del':0}
    i = len(lab) - 1
    j = len(rec) - 1
    while True :
      if self.space[i][j]['error'] == 'cor' : # correct
        if len(lab[i]) > 0 :
          self.data[lab[i]]['all'] = self.data[lab[i]]['all'] + 1
          self.data[lab[i]]['cor'] = self.data[lab[i]]['cor'] + 1
          result['all'] = result['all'] + 1
          result['cor'] = result['cor'] + 1
        result['lab'].insert(0, lab[i])
        result['rec'].insert(0, rec[j])
        i = i - 1
        j = j - 1
      elif self.space[i][j]['error'] == 'sub' : # substitution
        if len(lab[i]) > 0 :
          self.data[lab[i]]['all'] = self.data[lab[i]]['all'] + 1
          self.data[lab[i]]['sub'] = self.data[lab[i]]['sub'] + 1
          result['all'] = result['all'] + 1
          result['sub'] = result['sub'] + 1
        result['lab'].insert(0, lab[i])
        result['rec'].insert(0, rec[j])
        i = i - 1
        j = j - 1
      elif self.space[i][j]['error'] == 'del' : # deletion
        if len(lab[i]) > 0 :
          self.data[lab[i]]['all'] = self.data[lab[i]]['all'] + 1
          self.data[lab[i]]['del'] = self.data[lab[i]]['del'] + 1
          result['all'] = result['all'] + 1
          result['del'] = result['del'] + 1
        result['lab'].insert(0, lab[i])
        result['rec'].insert(0, "")
        i = i - 1
      elif self.space[i][j]['error'] == 'ins' : # insertion
        if len(rec[j]) > 0 :
          self.data[rec[j]]['ins'] = self.data[rec[j]]['ins'] + 1
          result['ins'] = result['ins'] + 1
        result['lab'].insert(0, "")
        result['rec'].insert(0, rec[j])
        j = j - 1
      elif self.space[i][j]['error'] == 'non' : # starting point
        break
      else : # shouldn't reach here
        print('this should not happen , i = {i} , j = {j} , error = {error}'.format(i = i, j = j, error = self.space[i][j]['error']))
    return result
  def overall(self) :
    result = {'all':0, 'cor':0, 'sub':0, 'ins':0, 'del':0}
    for token in self.data :
      result['all'] = result['all'] + self.data[token]['all']
      result['cor'] = result['cor'] + self.data[token]['cor']
      result['sub'] = result['sub'] + self.data[token]['sub']
      result['ins'] = result['ins'] + self.data[token]['ins']
      result['del'] = result['del'] + self.data[token]['del']
    return result
  def cluster(self, data) :
    result = {'all':0, 'cor':0, 'sub':0, 'ins':0, 'del':0}
    for token in data :
      if token in self.data :
        result['all'] = result['all'] + self.data[token]['all']
        result['cor'] = result['cor'] + self.data[token]['cor']
        result['sub'] = result['sub'] + self.data[token]['sub']
        result['ins'] = result['ins'] + self.data[token]['ins']
        result['del'] = result['del'] + self.data[token]['del']
    return result
  def keys(self) :
      return list(self.data.keys())

def width(string):
  return sum(1 + (unicodedata.east_asian_width(c) in "AFW") for c in string)

def default_cluster(word) :
  # unicode_names = [ unicodedata.name(char) for char in word ]
  unicode_names = []
  for char in word:
    try:
      tmp_char = unicodedata.name(char)
      unicode_names.append(tmp_char)
    except Exception as e:
      print('char error:', e, f'===({char})===')
      continue
  for i in reversed(range(len(unicode_names))) :
    if unicode_names[i].startswith('DIGIT') :  # 1
      unicode_names[i] = 'Number'  # 'DIGIT'
    elif (unicode_names[i].startswith('CJK UNIFIED IDEOGRAPH') or
          unicode_names[i].startswith('CJK COMPATIBILITY IDEOGRAPH')) :
      # 明 / 郎
      unicode_names[i] = 'Mandarin'  # 'CJK IDEOGRAPH'
    elif (unicode_names[i].startswith('LATIN CAPITAL LETTER') or
          unicode_names[i].startswith('LATIN SMALL LETTER')) :
      # A / a
      unicode_names[i] = 'English'  # 'LATIN LETTER'
    elif unicode_names[i].startswith('HIRAGANA LETTER') :  # は こ め
      unicode_names[i] = 'Japanese'  # 'GANA LETTER'
    elif (unicode_names[i].startswith('AMPERSAND') or
          unicode_names[i].startswith('APOSTROPHE') or
          unicode_names[i].startswith('COMMERCIAL AT') or
          unicode_names[i].startswith('DEGREE CELSIUS') or
          unicode_names[i].startswith('EQUALS SIGN') or
          unicode_names[i].startswith('FULL STOP') or
          unicode_names[i].startswith('HYPHEN-MINUS') or
          unicode_names[i].startswith('LOW LINE') or
          unicode_names[i].startswith('NUMBER SIGN') or
          unicode_names[i].startswith('PLUS SIGN') or
          unicode_names[i].startswith('SEMICOLON')) :
      # & / ' / @ / ℃ / = / . / - / _ / # / + / ;
      del unicode_names[i]
    else :
      return 'Other'
  if len(unicode_names) == 0 :
      return 'Other'
  if len(unicode_names) == 1 :
      return unicode_names[0]
  for i in range(len(unicode_names)-1) :
    if unicode_names[i] != unicode_names[i+1] :
      return 'Other'
  return unicode_names[0]

def cer_one(ref_text, hyp_text, ignore_words=set(), case_sensitive=False, split=None):
  # 相当于 --char=1 --v=1 
  calculator = Calculator()
  ref_text = normalizer.normalize(ref_text)
  hyp_text = normalizer.normalize(hyp_text)
  # tochar
  ref_array = characterize(ref_text)
  hyp_array = characterize(hyp_text)

  ref = normalize(ref_array, ignore_words, case_sensitive, split)
  hyp = normalize(hyp_array, ignore_words, case_sensitive, split)

  # 汉字转拼音
  lab_out = pinyin(ref, style=Style.TONE)
  ref = [item for sublist in lab_out for item in sublist]
  rec_out = pinyin(hyp, style=Style.TONE)
  hyp = [item for sublist in rec_out for item in sublist]

  # 计算cer结果
  result = calculator.calculate(ref, hyp)

  if result['all'] != 0 :
    wer = float(result['ins'] + result['sub'] + result['del']) * 100.0 / result['all']
  else :
    wer = 0.0
    ser = 0.0
  print('Overall -> %4.2f %%' % wer, end = ' ')
  print('N=%d C=%d S=%d D=%d I=%d' %
        (result['all'], result['cor'], result['sub'], result['del'], result['ins']))
  result['wer'] = wer

  return result

def usage() :
  print("compute-wer.py : compute word error rate (WER) and align recognition results and references.")
  print("         usage : python compute-wer.py [--cs={0,1}] [--cluster=foo] [--ig=ignore_file] [--char={0,1}] [--v={0,1}] [--padding-symbol={space,underline}] test.ref test.hyp > test.wer")

if __name__ == '__main__':
  if len(sys.argv) == 1 :
    usage()
    sys.exit(0)
  calculator = Calculator()
  cluster_file = ''
  ignore_words = set()
  tochar = False
  verbose= 1
  padding_symbol= ' '
  case_sensitive = False
  max_words_per_line = sys.maxsize
  split = None
  replace = None
  while len(sys.argv) > 4:
     a = '--maxw='
     if sys.argv[1].startswith(a):
        b = sys.argv[1][len(a):]
        del sys.argv[1]
        max_words_per_line = int(b)
        continue
     a = '--rt='
     if sys.argv[1].startswith(a):
        b = sys.argv[1][len(a):].lower()
        del sys.argv[1]
        remove_tag = (b == 'true') or (b != '0')
        continue
     a = '--cs='
     if sys.argv[1].startswith(a):
        b = sys.argv[1][len(a):].lower()
        del sys.argv[1]
        case_sensitive = (b == 'true') or (b != '0')
        continue
     a = '--cluster='
     if sys.argv[1].startswith(a):
       cluster_file = sys.argv[1][len(a):]
       del sys.argv[1]
       continue
     a = '--splitfile='
     if sys.argv[1].startswith(a):
       split_file = sys.argv[1][len(a):]
       del sys.argv[1]
       split = dict()
       with codecs.open(split_file, 'r', 'utf-8') as fh:
         for line in fh:  # line in unicode
           words = line.strip().split()
           if len(words) >= 2:
             split[words[0]] = words[1:]
       continue
     a = '--ig='
     if sys.argv[1].startswith(a):
       ignore_file = sys.argv[1][len(a):]
       del sys.argv[1]
       with codecs.open(ignore_file, 'r', 'utf-8') as fh:
         for line in fh:  # line in unicode
           line = line.strip()
           if len(line) > 0:
             ignore_words.add(line)
       continue
     a = '--char='
     if sys.argv[1].startswith(a):
        b = sys.argv[1][len(a):].lower()
        del sys.argv[1]
        tochar = (b == 'true') or (b != '0')
        continue
     a = '--v='
     if sys.argv[1].startswith(a):
        b = sys.argv[1][len(a):].lower()
        del sys.argv[1]
        verbose=0
        try:
          verbose=int(b)
        except:
           if b == 'true' or b != '0':
              verbose = 1
        continue
     a = '--padding-symbol='
     if sys.argv[1].startswith(a):
        b = sys.argv[1][len(a):].lower()
        del sys.argv[1]
        if b == 'space':
          padding_symbol= ' '
        elif b == 'underline':
          padding_symbol= '_'
        continue
     a = '--replace='
     if sys.argv[1].startswith(a):
       replace_file = sys.argv[1][len(a):]
       del sys.argv[1]
       replace = dict()
       with codecs.open(replace_file, 'r', 'utf-8') as fh:
         for line in fh:  # line in unicode
           words = line.strip().split("->")
           if len(words) == 2:
             replace[words[0]] = words[1]
       continue
     if True or sys.argv[1].startswith('-'):
        #ignore invalid switch
        del sys.argv[1]
        continue

  if not case_sensitive:
     ig=set([w.upper() for w in ignore_words])
     ignore_words = ig

  default_clusters = {}
  default_words = {}

  ref_file = sys.argv[1]
  hyp_file = sys.argv[2]
  out_file = sys.argv[3]
  rec_set = {}
  utter_count = 0
  utter_err_count = 0
  if split and not case_sensitive:
     newsplit = dict()
     for w in split:
        words = split[w]
        for i in range(len(words)):
           words[i] = words[i].upper()
        newsplit[w.upper()] = words
     split = newsplit

  with codecs.open(hyp_file, 'r', 'utf-8') as fh:
     for line in fh:
        # print(line)
        try:
          temp = line.split()
          temp = json.loads(line.strip())
          if len(temp) < 2:
            continue
          # wav_id = temp[0]
          # text = ' '.join(temp[1:])
          wav_id = temp['uttid']
          if 'text' in temp:
            text = temp['text']
          elif 'value' in temp:
            text = temp['value']
          else:
            print('not text in ', temp)
            continue
          text = normalizer.normalize(text)
          if replace:
            text = replace_line(text, case_sensitive, replace)
          if tochar:
            # print(line)
            array = characterize(text)
          else:
            array = text.strip().split()

          rec_set[wav_id] = normalize(array, ignore_words, case_sensitive, split)
          # line = wav_id + ' ' + text
        except Exception as e:
          print('error:', e) 
          continue
        # ==============================
        # if replace:
        #     line = replace_line(line, case_sensitive, replace)
        # if tochar:
        #     # print(line)
        #     array = characterize(line)
        # else:
        #     array = line.strip().split()
        # # print(array)
        # if len(array)==0: continue
        # fid = array[0]
        # rec_set[fid] = normalize(array[1:], ignore_words, case_sensitive, split)
        # print(rec_set)
        # ==============================

  # compute error rate on the interaction of reference file and hyp file
  res_list = []
  for line in open(ref_file, 'r', encoding='utf-8') :
    temp = json.loads(line.strip())
    if len(temp) < 2:
      continue
    wav_id = temp['uttid']
    fid = wav_id
    if 'text' in temp:
      text = temp['text']
    elif 'value' in temp:
      text = temp['value']
    else:
      print('not text in ', temp)
      continue
    text = normalizer.normalize(text)
    if replace:
        text = replace_line(text, case_sensitive, replace)
    if tochar:
          array = characterize(text)
    else:
          array = text.rstrip('\n').split()
    if len(array)==0: continue
    if wav_id not in rec_set:
       continue
    lab = normalize(array, ignore_words, case_sensitive, split)
    rec = rec_set[wav_id]
    # ============================
    # line = wav_id + ' ' + text
    # if replace:
    #     line = replace_line(line, case_sensitive, replace)
    # if tochar:
    #       array = characterize(line)
    # else:
    #       array = line.rstrip('\n').split()
    # if len(array)==0: continue
    # fid = array[0]
    # if fid not in rec_set:
    #    continue
    # lab = normalize(array[1:], ignore_words, case_sensitive, split)
    # rec = rec_set[fid]
    # =============================
    # print(lab, rec) # 按字划分的列表
    # 文本正则化
    # lab = normalizer.normalize(lab)
    # rec = normalizer.normalize(rec)
    if verbose:
      print('\nutt: %s' % fid)

    for word in rec + lab :
      if word not in default_words :
         default_cluster_name = default_cluster(word)
         if default_cluster_name not in default_clusters :
           default_clusters[default_cluster_name] = {}
         if word not in default_clusters[default_cluster_name] :
           default_clusters[default_cluster_name][word] = 1
         default_words[word] = default_cluster_name

    # 汉字转拼音
    lab_out = pinyin(lab, style=Style.TONE)
    lab = [item for sublist in lab_out for item in sublist]
    rec_out = pinyin(rec, style=Style.TONE)
    rec = [item for sublist in rec_out for item in sublist]
    # 计算cer结果
    result = calculator.calculate(lab, rec)
    utter_count += 1
    error_count = 0
    if result['all']:
      error_count = result['ins'] + result['sub'] + result['del']
      if error_count:
        utter_err_count += 1
    if verbose:
      if result['all'] != 0 :
        wer = float(result['ins'] + result['sub'] + result['del']) * 100.0 / result['all']
      else :
        wer = 0.0
      res_list.append({'uttid': fid, 'wer': round(wer, 2), 'wer_info': result})
      print('WER: %4.2f %%' % wer, end = ' ')
      print('N=%d C=%d S=%d D=%d I=%d' %
          (result['all'], result['cor'], result['sub'], result['del'], result['ins']))
      space = {}
      space['lab'] = []
      space['rec'] = []
      for idx in range(len(result['lab'])) :
        len_lab = width(result['lab'][idx])
        len_rec = width(result['rec'][idx])
        length = max(len_lab, len_rec)
        space['lab'].append(length-len_lab)
        space['rec'].append(length-len_rec)
      upper_lab = len(result['lab'])
      upper_rec = len(result['rec'])
      lab1, rec1 = 0, 0
      while lab1 < upper_lab or rec1 < upper_rec:
         if verbose > 1:
             print('lab(%s):' % fid.encode('utf-8'), end = ' ')
         else:
             print('lab:', end = ' ')
         lab2 = min(upper_lab, lab1 + max_words_per_line)
         for idx in range(lab1, lab2):
           token = result['lab'][idx]
           print('{token}'.format(token = token), end = '')
           for n in range(space['lab'][idx]) :
             print(padding_symbol, end = '')
           print(' ',end='')
         print()
         if verbose > 1:
            print('rec(%s):' % fid.encode('utf-8'), end = ' ')
         else:
            print('rec:', end = ' ')
         rec2 = min(upper_rec, rec1 + max_words_per_line)
         for idx in range(rec1, rec2):
           token = result['rec'][idx]
           print('{token}'.format(token = token), end = '')
           for n in range(space['rec'][idx]) :
             print(padding_symbol, end = '')
           print(' ',end='')
         print('\n', end='\n')
         lab1 = lab2
         rec1 = rec2
  with open(out_file, 'w', encoding='u8') as f_out:
    for item in res_list:
      json_data = json.dumps(item, ensure_ascii=False)
      # 写入文件
      f_out.write(json_data + '\n')
      f_out.flush()

  if verbose:
    print('===========================================================================')
    print()

  result = calculator.overall()
  if result['all'] != 0 :
    wer = float(result['ins'] + result['sub'] + result['del']) * 100.0 / result['all']
    ser = utter_err_count * 100.0 / utter_count
  else :
    wer = 0.0
    ser = 0.0
  print('Overall -> %4.2f %%' % wer, end = ' ')
  print('N=%d C=%d S=%d D=%d I=%d' %
        (result['all'], result['cor'], result['sub'], result['del'], result['ins']))
  print('SER -> %4.2f %%' % ser, end = ' ')
  print('utt=%d err=%d' % (utter_count, utter_err_count))
  if not verbose:
     print()

  if verbose:
   for cluster_id in default_clusters :
     result = calculator.cluster([ k for k in default_clusters[cluster_id] ])
     if result['all'] != 0 :
        wer = float(result['ins'] + result['sub'] + result['del']) * 100.0 / result['all']
     else :
        wer = 0.0
     print('%s -> %4.2f %%' % (cluster_id, wer), end = ' ')
     print('N=%d C=%d S=%d D=%d I=%d' %
          (result['all'], result['cor'], result['sub'], result['del'], result['ins']))
   if len(cluster_file) > 0 : # compute separated WERs for word clusters
     cluster_id = ''
     cluster = []
     for line in open(cluster_file, 'r', encoding='utf-8') :
       for token in line.decode('utf-8').rstrip('\n').split() :
        # end of cluster reached, like </Keyword>
        if token[0:2] == '</' and token[len(token)-1] == '>' and \
           token.lstrip('</').rstrip('>') == cluster_id :
          result = calculator.cluster(cluster)
          if result['all'] != 0 :
            wer = float(result['ins'] + result['sub'] + result['del']) * 100.0 / result['all']
          else :
            wer = 0.0
          print('%s -> %4.2f %%' % (cluster_id, wer), end = ' ')
          print('N=%d C=%d S=%d D=%d I=%d' %
                (result['all'], result['cor'], result['sub'], result['del'], result['ins']))
          cluster_id = ''
          cluster = []
        # begin of cluster reached, like <Keyword>
        elif token[0] == '<' and token[len(token)-1] == '>' and \
             cluster_id == '' :
          cluster_id = token.lstrip('<').rstrip('>')
          cluster = []
        # general terms, like WEATHER / CAR / ...
        else :
          cluster.append(token)
   print()
   print('===========================================================================')
```
<!-- /REBUILD_FILE -->

## 附录 8：tools/en/compute_wer_jsonl.py

<!-- REBUILD_FILE: tools/en/compute_wer_jsonl.py SHA256: 1451410bd99e440f2cb7948e066b149d903b45f339fb91c3ea9b5e2aa143d2c1 -->
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import re, sys, unicodedata, json
import codecs
from tn.english.normalizer import Normalizer

normalizer = Normalizer()

remove_tag = True
spacelist= [' ', '\t', '\r', '\n']
#puncts = ['!', ',', '?',
#          '、', '。', '！', '，', '；', '？',
#          '：', '「', '」', '︰',  '『', '』', '《', '》']
puncts="""!"#$%&()*+,-.·/:;<=>?@，。?★、…【】《》？“”‘！[\\]^_`{|}~’…'"""

def rm_punct(in_str):
    for p in puncts:
       in_str = in_str.replace(p, ' ')
    return in_str

def characterize(string) :
  string = rm_punct(string)
  res = []
  i = 0
  while i < len(string):
    char = string[i]
    if char in puncts:
      i += 1
      continue
    cat1 = unicodedata.category(char)
    #https://unicodebook.readthedocs.io/unicode.html#unicode-categories
    if cat1 == 'Zs' or cat1 == 'Cn' or char in spacelist: # space or not assigned
       i += 1
       continue
    if cat1 == 'Lo': # letter-other
       res.append(char)
       i += 1
    else:
       # some input looks like: <unk><noise>, we want to separate it to two words.
       sep = ' '
       if char == '<': sep = '>'
       j = i+1
       while j < len(string):
         c = string[j]
         if ord(c) >= 128 or (c in spacelist) or (c==sep):
           break
         j += 1
       if j < len(string) and string[j] == '>':
         j += 1
       res.append(string[i:j])
       i = j
  return res

def stripoff_tags(x):
  if not x: return ''
  chars = []
  i = 0; T=len(x)
  while i < T:
    if x[i] == '<':
      while i < T and x[i] != '>':
         i += 1
      i += 1
    else:
      chars.append(x[i])
      i += 1
  return ''.join(chars)


def normalize(sentence, ignore_words, cs, split=None):
    """ sentence, ignore_words are both in unicode
    """
    new_sentence = []
    # print(sentence)
    for token in sentence:
        x = token
        if not cs:
           x = x.upper()
        if x in ignore_words:
           continue
        if remove_tag:
          x = stripoff_tags(x)
        if not x:
          continue
        if split and x in split:
          new_sentence += split[x]
        else:
          new_sentence.append(x)
    return new_sentence


def replace_line(sentence, cs, replace=None):
    splits = sentence.replace("\t", " ").split(" ", 1)
    fid = splits[0]
    sentence = splits[1]
    # print(sentence)
    for key, value in replace.items():
        if not cs:
            key = key.upper()
            value = value.upper()
            sentence = sentence.upper()
        sentence = sentence.replace(key, value)
    return "{} {}".format(fid, sentence)

class Calculator :
  def __init__(self) :
    self.data = {}
    self.space = []
    self.cost = {}
    self.cost['cor'] = 0
    self.cost['sub'] = 1
    self.cost['del'] = 1
    self.cost['ins'] = 1
  def calculate(self, lab, rec) :
    # Initialization
    lab.insert(0, '')
    rec.insert(0, '')
    while len(self.space) < len(lab) :
      self.space.append([])
    for row in self.space :
      for element in row :
        element['dist'] = 0
        element['error'] = 'non'
      while len(row) < len(rec) :
        row.append({'dist' : 0, 'error' : 'non'})
    for i in range(len(lab)) :
      self.space[i][0]['dist'] = i
      self.space[i][0]['error'] = 'del'
    for j in range(len(rec)) :
      self.space[0][j]['dist'] = j
      self.space[0][j]['error'] = 'ins'
    self.space[0][0]['error'] = 'non'
    for token in lab :
      if token not in self.data and len(token) > 0 :
        self.data[token] = {'all' : 0, 'cor' : 0, 'sub' : 0, 'ins' : 0, 'del' : 0}
    for token in rec :
      if token not in self.data and len(token) > 0 :
        self.data[token] = {'all' : 0, 'cor' : 0, 'sub' : 0, 'ins' : 0, 'del' : 0}
    # Computing edit distance
    for i, lab_token in enumerate(lab) :
      for j, rec_token in enumerate(rec) :
        if i == 0 or j == 0 :
          continue
        min_dist = sys.maxsize
        min_error = 'none'
        dist = self.space[i-1][j]['dist'] + self.cost['del']
        error = 'del'
        if dist < min_dist :
          min_dist = dist
          min_error = error
        dist = self.space[i][j-1]['dist'] + self.cost['ins']
        error = 'ins'
        if dist < min_dist :
          min_dist = dist
          min_error = error
        if lab_token == rec_token :
          dist = self.space[i-1][j-1]['dist'] + self.cost['cor']
          error = 'cor'
        else :
          dist = self.space[i-1][j-1]['dist'] + self.cost['sub']
          error = 'sub'
        if dist < min_dist :
          min_dist = dist
          min_error = error
        self.space[i][j]['dist'] = min_dist
        self.space[i][j]['error'] = min_error
    # Tracing back
    result = {'lab':[], 'rec':[], 'all':0, 'cor':0, 'sub':0, 'ins':0, 'del':0}
    i = len(lab) - 1
    j = len(rec) - 1
    while True :
      if self.space[i][j]['error'] == 'cor' : # correct
        if len(lab[i]) > 0 :
          self.data[lab[i]]['all'] = self.data[lab[i]]['all'] + 1
          self.data[lab[i]]['cor'] = self.data[lab[i]]['cor'] + 1
          result['all'] = result['all'] + 1
          result['cor'] = result['cor'] + 1
        result['lab'].insert(0, lab[i])
        result['rec'].insert(0, rec[j])
        i = i - 1
        j = j - 1
      elif self.space[i][j]['error'] == 'sub' : # substitution
        if len(lab[i]) > 0 :
          self.data[lab[i]]['all'] = self.data[lab[i]]['all'] + 1
          self.data[lab[i]]['sub'] = self.data[lab[i]]['sub'] + 1
          result['all'] = result['all'] + 1
          result['sub'] = result['sub'] + 1
        result['lab'].insert(0, lab[i])
        result['rec'].insert(0, rec[j])
        i = i - 1
        j = j - 1
      elif self.space[i][j]['error'] == 'del' : # deletion
        if len(lab[i]) > 0 :
          self.data[lab[i]]['all'] = self.data[lab[i]]['all'] + 1
          self.data[lab[i]]['del'] = self.data[lab[i]]['del'] + 1
          result['all'] = result['all'] + 1
          result['del'] = result['del'] + 1
        result['lab'].insert(0, lab[i])
        result['rec'].insert(0, "")
        i = i - 1
      elif self.space[i][j]['error'] == 'ins' : # insertion
        if len(rec[j]) > 0 :
          self.data[rec[j]]['ins'] = self.data[rec[j]]['ins'] + 1
          result['ins'] = result['ins'] + 1
        result['lab'].insert(0, "")
        result['rec'].insert(0, rec[j])
        j = j - 1
      elif self.space[i][j]['error'] == 'non' : # starting point
        break
      else : # shouldn't reach here
        print('this should not happen , i = {i} , j = {j} , error = {error}'.format(i = i, j = j, error = self.space[i][j]['error']))
    return result
  def overall(self) :
    result = {'all':0, 'cor':0, 'sub':0, 'ins':0, 'del':0}
    for token in self.data :
      result['all'] = result['all'] + self.data[token]['all']
      result['cor'] = result['cor'] + self.data[token]['cor']
      result['sub'] = result['sub'] + self.data[token]['sub']
      result['ins'] = result['ins'] + self.data[token]['ins']
      result['del'] = result['del'] + self.data[token]['del']
    return result
  def cluster(self, data) :
    result = {'all':0, 'cor':0, 'sub':0, 'ins':0, 'del':0}
    for token in data :
      if token in self.data :
        result['all'] = result['all'] + self.data[token]['all']
        result['cor'] = result['cor'] + self.data[token]['cor']
        result['sub'] = result['sub'] + self.data[token]['sub']
        result['ins'] = result['ins'] + self.data[token]['ins']
        result['del'] = result['del'] + self.data[token]['del']
    return result
  def keys(self) :
      return list(self.data.keys())

def width(string):
  return sum(1 + (unicodedata.east_asian_width(c) in "AFW") for c in string)

def default_cluster(word) :
  # unicode_names = [ unicodedata.name(char) for char in word ]
  unicode_names = []
  for char in word:
    try:
      tmp_char = unicodedata.name(char)
      unicode_names.append(tmp_char)
    except Exception as e:
      print('char error:', e, f'===({char})===')
      continue
  for i in reversed(range(len(unicode_names))) :
    if unicode_names[i].startswith('DIGIT') :  # 1
      unicode_names[i] = 'Number'  # 'DIGIT'
    elif (unicode_names[i].startswith('CJK UNIFIED IDEOGRAPH') or
          unicode_names[i].startswith('CJK COMPATIBILITY IDEOGRAPH')) :
      # 明 / 郎
      unicode_names[i] = 'Mandarin'  # 'CJK IDEOGRAPH'
    elif (unicode_names[i].startswith('LATIN CAPITAL LETTER') or
          unicode_names[i].startswith('LATIN SMALL LETTER')) :
      # A / a
      unicode_names[i] = 'English'  # 'LATIN LETTER'
    elif unicode_names[i].startswith('HIRAGANA LETTER') :  # は こ め
      unicode_names[i] = 'Japanese'  # 'GANA LETTER'
    elif (unicode_names[i].startswith('AMPERSAND') or
          unicode_names[i].startswith('APOSTROPHE') or
          unicode_names[i].startswith('COMMERCIAL AT') or
          unicode_names[i].startswith('DEGREE CELSIUS') or
          unicode_names[i].startswith('EQUALS SIGN') or
          unicode_names[i].startswith('FULL STOP') or
          unicode_names[i].startswith('HYPHEN-MINUS') or
          unicode_names[i].startswith('LOW LINE') or
          unicode_names[i].startswith('NUMBER SIGN') or
          unicode_names[i].startswith('PLUS SIGN') or
          unicode_names[i].startswith('SEMICOLON')) :
      # & / ' / @ / ℃ / = / . / - / _ / # / + / ;
      del unicode_names[i]
    else :
      return 'Other'
  if len(unicode_names) == 0 :
      return 'Other'
  if len(unicode_names) == 1 :
      return unicode_names[0]
  for i in range(len(unicode_names)-1) :
    if unicode_names[i] != unicode_names[i+1] :
      return 'Other'
  return unicode_names[0]

def cer_one(ref_text, hyp_text, ignore_words=set(), case_sensitive=False, split=None):
  # 相当于 --char=1 --v=1 
  calculator = Calculator()
  ref_text = normalizer.normalize(ref_text)
  hyp_text = normalizer.normalize(hyp_text)
  # tochar
  ref_array = characterize(ref_text)
  hyp_array = characterize(hyp_text)

  ref = normalize(ref_array, ignore_words, case_sensitive, split)
  hyp = normalize(hyp_array, ignore_words, case_sensitive, split)

  # 计算cer结果
  result = calculator.calculate(ref, hyp)

  if result['all'] != 0 :
    wer = float(result['ins'] + result['sub'] + result['del']) * 100.0 / result['all']
  else :
    wer = 0.0
    ser = 0.0
  print('Overall -> %4.2f %%' % wer, end = ' ')
  print('N=%d C=%d S=%d D=%d I=%d' %
        (result['all'], result['cor'], result['sub'], result['del'], result['ins']))
  result['wer'] = wer

  return result

def usage() :
  print("compute-wer.py : compute word error rate (WER) and align recognition results and references.")
  print("         usage : python compute-wer.py [--cs={0,1}] [--cluster=foo] [--ig=ignore_file] [--char={0,1}] [--v={0,1}] [--padding-symbol={space,underline}] test.ref test.hyp > test.wer")

if __name__ == '__main__':
  if len(sys.argv) == 1 :
    usage()
    sys.exit(0)
  calculator = Calculator()
  cluster_file = ''
  ignore_words = set()
  tochar = False
  verbose= 1
  padding_symbol= ' '
  case_sensitive = False
  max_words_per_line = sys.maxsize
  split = None
  replace = None
  while len(sys.argv) > 4:
     a = '--maxw='
     if sys.argv[1].startswith(a):
        b = sys.argv[1][len(a):]
        del sys.argv[1]
        max_words_per_line = int(b)
        continue
     a = '--rt='
     if sys.argv[1].startswith(a):
        b = sys.argv[1][len(a):].lower()
        del sys.argv[1]
        remove_tag = (b == 'true') or (b != '0')
        continue
     a = '--cs='
     if sys.argv[1].startswith(a):
        b = sys.argv[1][len(a):].lower()
        del sys.argv[1]
        case_sensitive = (b == 'true') or (b != '0')
        continue
     a = '--cluster='
     if sys.argv[1].startswith(a):
       cluster_file = sys.argv[1][len(a):]
       del sys.argv[1]
       continue
     a = '--splitfile='
     if sys.argv[1].startswith(a):
       split_file = sys.argv[1][len(a):]
       del sys.argv[1]
       split = dict()
       with codecs.open(split_file, 'r', 'utf-8') as fh:
         for line in fh:  # line in unicode
           words = line.strip().split()
           if len(words) >= 2:
             split[words[0]] = words[1:]
       continue
     a = '--ig='
     if sys.argv[1].startswith(a):
       ignore_file = sys.argv[1][len(a):]
       del sys.argv[1]
       with codecs.open(ignore_file, 'r', 'utf-8') as fh:
         for line in fh:  # line in unicode
           line = line.strip()
           if len(line) > 0:
             ignore_words.add(line)
       continue
     a = '--char='
     if sys.argv[1].startswith(a):
        b = sys.argv[1][len(a):].lower()
        del sys.argv[1]
        tochar = (b == 'true') or (b != '0')
        continue
     a = '--v='
     if sys.argv[1].startswith(a):
        b = sys.argv[1][len(a):].lower()
        del sys.argv[1]
        verbose=0
        try:
          verbose=int(b)
        except:
           if b == 'true' or b != '0':
              verbose = 1
        continue
     a = '--padding-symbol='
     if sys.argv[1].startswith(a):
        b = sys.argv[1][len(a):].lower()
        del sys.argv[1]
        if b == 'space':
          padding_symbol= ' '
        elif b == 'underline':
          padding_symbol= '_'
        continue
     a = '--replace='
     if sys.argv[1].startswith(a):
       replace_file = sys.argv[1][len(a):]
       del sys.argv[1]
       replace = dict()
       with codecs.open(replace_file, 'r', 'utf-8') as fh:
         for line in fh:  # line in unicode
           words = line.strip().split("->")
           if len(words) == 2:
             replace[words[0]] = words[1]
       continue
     if True or sys.argv[1].startswith('-'):
        #ignore invalid switch
        del sys.argv[1]
        continue

  if not case_sensitive:
     ig=set([w.upper() for w in ignore_words])
     ignore_words = ig

  default_clusters = {}
  default_words = {}

  ref_file = sys.argv[1]
  hyp_file = sys.argv[2]
  out_file = sys.argv[3]
  rec_set = {}
  utter_count = 0
  utter_err_count = 0
  if split and not case_sensitive:
     newsplit = dict()
     for w in split:
        words = split[w]
        for i in range(len(words)):
           words[i] = words[i].upper()
        newsplit[w.upper()] = words
     split = newsplit

  with codecs.open(hyp_file, 'r', 'utf-8') as fh:
     for line in fh:
        # print(line)
        try:
          temp = line.split()
          temp = json.loads(line.strip())
          if len(temp) < 2:
            continue
          # wav_id = temp[0]
          # text = ' '.join(temp[1:])
          wav_id = temp['uttid']
          if 'text' in temp:
            text = temp['text']
          elif 'value' in temp:
            text = temp['value']
          else:
            print('not text in ', temp)
            continue
          text = normalizer.normalize(text)
          line = wav_id + ' ' + text
        except Exception as e:
          print('error:', e) 
          continue
        if replace:
            line = replace_line(line, case_sensitive, replace)
        if tochar:
            # print(line)
            array = characterize(line)
        else:
            array = line.strip().split()
        # print(array)
        if len(array)==0: continue
        fid = array[0]
        rec_set[fid] = normalize(array[1:], ignore_words, case_sensitive, split)
        # print(rec_set)

  # compute error rate on the interaction of reference file and hyp file
  res_list = []
  for line in open(ref_file, 'r', encoding='utf-8') :
    # temp = line.split()
    # wav_id = temp[0]
    # text = ' '.join(temp[1:])
    temp = json.loads(line.strip())
    if len(temp) < 2:
      continue
    # wav_id = temp[0]
    # text = ' '.join(temp[1:])
    wav_id = temp['uttid']
    fid = wav_id
    if 'text' in temp:
      text = temp['text']
    elif 'value' in temp:
      text = temp['value']
    else:
      print('not text in ', temp)
      continue
    text = normalizer.normalize(text)
    line = wav_id + ' ' + text
    if replace:
        line = replace_line(line, case_sensitive, replace)
    if tochar:
          array = characterize(line)
    else:
          array = line.rstrip('\n').split()
    if len(array)==0: continue
    fid = array[0]
    if fid not in rec_set:
       continue
    lab = normalize(array[1:], ignore_words, case_sensitive, split)
    rec = rec_set[fid]
    # print(lab, rec) # 按字划分的列表
    # 文本正则化
    # lab = normalizer.normalize(lab)
    # rec = normalizer.normalize(rec)
    if verbose:
      print('\nutt: %s' % fid)

    for word in rec + lab :
      if word not in default_words :
         default_cluster_name = default_cluster(word)
         if default_cluster_name not in default_clusters :
           default_clusters[default_cluster_name] = {}
         if word not in default_clusters[default_cluster_name] :
           default_clusters[default_cluster_name][word] = 1
         default_words[word] = default_cluster_name

    result = calculator.calculate(lab, rec)
    utter_count += 1
    error_count = 0
    if result['all']:
      error_count = result['ins'] + result['sub'] + result['del']
      if error_count:
        utter_err_count += 1
    if verbose:
      if result['all'] != 0 :
        wer = float(result['ins'] + result['sub'] + result['del']) * 100.0 / result['all']
      else :
        wer = 0.0
      res_list.append({'uttid': fid, 'wer': round(wer, 2), 'wer_info': result})
      print('WER: %4.2f %%' % wer, end = ' ')
      print('N=%d C=%d S=%d D=%d I=%d' %
          (result['all'], result['cor'], result['sub'], result['del'], result['ins']))
      space = {}
      space['lab'] = []
      space['rec'] = []
      for idx in range(len(result['lab'])) :
        len_lab = width(result['lab'][idx])
        len_rec = width(result['rec'][idx])
        length = max(len_lab, len_rec)
        space['lab'].append(length-len_lab)
        space['rec'].append(length-len_rec)
      upper_lab = len(result['lab'])
      upper_rec = len(result['rec'])
      lab1, rec1 = 0, 0
      while lab1 < upper_lab or rec1 < upper_rec:
         if verbose > 1:
             print('lab(%s):' % fid.encode('utf-8'), end = ' ')
         else:
             print('lab:', end = ' ')
         lab2 = min(upper_lab, lab1 + max_words_per_line)
         for idx in range(lab1, lab2):
           token = result['lab'][idx]
           print('{token}'.format(token = token), end = '')
           for n in range(space['lab'][idx]) :
             print(padding_symbol, end = '')
           print(' ',end='')
         print()
         if verbose > 1:
            print('rec(%s):' % fid.encode('utf-8'), end = ' ')
         else:
            print('rec:', end = ' ')
         rec2 = min(upper_rec, rec1 + max_words_per_line)
         for idx in range(rec1, rec2):
           token = result['rec'][idx]
           print('{token}'.format(token = token), end = '')
           for n in range(space['rec'][idx]) :
             print(padding_symbol, end = '')
           print(' ',end='')
         print('\n', end='\n')
         lab1 = lab2
         rec1 = rec2

  with open(out_file, 'w', encoding='u8') as f_out:
    for item in res_list:
      json_data = json.dumps(item, ensure_ascii=False)
      # 写入文件
      f_out.write(json_data + '\n')
      f_out.flush()

  if verbose:
    print('===========================================================================')
    print()

  result = calculator.overall()
  if result['all'] != 0 :
    wer = float(result['ins'] + result['sub'] + result['del']) * 100.0 / result['all']
    ser = utter_err_count * 100.0 / utter_count
  else :
    wer = 0.0
    ser = 0.0
  print('Overall -> %4.2f %%' % wer, end = ' ')
  print('N=%d C=%d S=%d D=%d I=%d' %
        (result['all'], result['cor'], result['sub'], result['del'], result['ins']))
  print('SER -> %4.2f %%' % ser, end = ' ')
  print('utt=%d err=%d' % (utter_count, utter_err_count))
  if not verbose:
     print()

  if verbose:
   for cluster_id in default_clusters :
     result = calculator.cluster([ k for k in default_clusters[cluster_id] ])
     if result['all'] != 0 :
        wer = float(result['ins'] + result['sub'] + result['del']) * 100.0 / result['all']
     else :
        wer = 0.0
     print('%s -> %4.2f %%' % (cluster_id, wer), end = ' ')
     print('N=%d C=%d S=%d D=%d I=%d' %
          (result['all'], result['cor'], result['sub'], result['del'], result['ins']))
   if len(cluster_file) > 0 : # compute separated WERs for word clusters
     cluster_id = ''
     cluster = []
     for line in open(cluster_file, 'r', encoding='utf-8') :
       for token in line.decode('utf-8').rstrip('\n').split() :
        # end of cluster reached, like </Keyword>
        if token[0:2] == '</' and token[len(token)-1] == '>' and \
           token.lstrip('</').rstrip('>') == cluster_id :
          result = calculator.cluster(cluster)
          if result['all'] != 0 :
            wer = float(result['ins'] + result['sub'] + result['del']) * 100.0 / result['all']
          else :
            wer = 0.0
          print('%s -> %4.2f %%' % (cluster_id, wer), end = ' ')
          print('N=%d C=%d S=%d D=%d I=%d' %
                (result['all'], result['cor'], result['sub'], result['del'], result['ins']))
          cluster_id = ''
          cluster = []
        # begin of cluster reached, like <Keyword>
        elif token[0] == '<' and token[len(token)-1] == '>' and \
             cluster_id == '' :
          cluster_id = token.lstrip('<').rstrip('>')
          cluster = []
        # general terms, like WEATHER / CAR / ...
        else :
          cluster.append(token)
   print()
   print('===========================================================================')
```
<!-- /REBUILD_FILE -->

## 附录 9：tools/other/compute_wer_jiwer.py

<!-- REBUILD_FILE: tools/other/compute_wer_jiwer.py SHA256: fceab7d1dff5d0f70381016dfba3c62bef8c13f587722e9d23db9e01055060ef -->
```python

from jiwer import wer, process_words, transforms

# 自定义缩写映射表（可无限扩充）
abbrev_map = {
    # 一、主语 + be动词缩写 (I'm / you're / he's / she's / it's / we're / they're)
    "i'm": "i am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "we're": "we are",
    "they're": "they are",
    "that's": "that is",
    "who's": "who is",
    "what's": "what is",
    "where's": "where is",
    "when's": "when is",
    "why's": "why is",
    "how's": "how is",

    # 二、主语 + will / would
    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "it'll": "it will",
    "we'll": "we will",
    "they'll": "they will",
    "i'd": "i would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "we'd": "we would",
    "they'd": "they would",

    # 三、have / has / had 缩写
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "he's": "he has",
    "she's": "she has",
    "it's": "it has",
    "who's": "who has",
    "i'd've": "i would have",
    "you'd've": "you would have",
    "he'd've": "he would have",
    "she'd've": "she would have",
    "we'd've": "we would have",
    "they'd've": "they would have",

    # 四、否定缩写 not (核心最多，ASR高频)
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "won't": "will not",
    "wouldn't": "would not",
    "shan't": "shall not",
    "shouldn't": "should not",
    "can't": "cannot",
    "cannot": "can not",
    "couldn't": "could not",
    "mustn't": "must not",
    "mightn't": "might not",
    "needn't": "need not",
    "daren't": "dare not",

    # 五、名词/疑问词 + is / have
    "there's": "there is",
    "there're": "there are",
    "there'll": "there will",
    "there'd": "there would",
    "here's": "here is",

    # 六、口语特有缩写、美式日常简写（TTS/直播高频）
    "ain't": "am not",
    "o'clock": "of the clock",
    "ma'am": "madam",
    "sir's": "sir is",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'll": "you all will"
}

# 英文文本归一化：小写+移除标点
trans = transforms.Compose([
    transforms.ToLowerCase(),
    transforms.SubstituteWords(abbrev_map),
    transforms.RemovePunctuation()
])

def cer_one(ref_text, hyp_text):
    ref = trans(ref_text)
    hyp = trans(hyp_text)
    # 详细错误拆解
    stats = process_words(ref, hyp)
    res = {
        'lab': stats.references,
        'rec': stats.hypotheses,
        'all': len(stats.references),
        'cor': stats.hits,
        'sub': stats.substitutions,
        'ins': stats.insertions,
        'del': stats.deletions,
        'wer': 100*stats.wer
    }
    
    return res
    
```
<!-- /REBUILD_FILE -->

## 附录 10：config_data/config_tts.yaml

<!-- REBUILD_FILE: config_data/config_tts.yaml SHA256: 0e4cc43fbf4a02f5a09e6003ac6bc0dd32296ac286142565680b9e81a7df112a -->
```yaml
# 新机器演示配置：确认实际服务地址后再运行
base_urls:
  - "http://127.0.0.1:9700/v1"
model_name: "qwen3-tts-1.7b"
max_workers: 1
target_sr: 16000
request_timeout: 60
max_retry: 2
per_url_exclusive: true
check_audio_length: true
min_sec_per_char: 0.001
max_sec_per_char: 2.0
min_audio_sec: 0.5
language: "Chinese"
output_wav_subdir: "wavs"
manifest_filename: "manifest.jsonl"
ref_list_path: "../data/refs.json"
```
<!-- /REBUILD_FILE -->

## 附录 11：config_data/config_asr.yaml

<!-- REBUILD_FILE: config_data/config_asr.yaml SHA256: d249c1bb44808ab0fa55f24aa52edf21b004f91c09de0c8320d2b3a86fd87fb9 -->
```yaml
# 新机器演示配置：确认实际服务地址后再运行
servers:
  - "http://127.0.0.1:9800/v1"
model: "qwen3-asr-1.7b"
request:
  temperature: 0.0
  max_tokens: 2048
  timeout: 60
threads: 1
json_fields:
  audio_path: tts_audio_path
  asr_result: tts_audio_asr
```
<!-- /REBUILD_FILE -->

## 附录 12：data/input_demo.json

<!-- REBUILD_FILE: data/input_demo.json SHA256: 84f757397ac5c3a280175c6409377990fc7b2c0a41e2035bae023a0302f4a95d -->
```json
[
  {
    "id": "demo_001",
    "turns": [
      {"user": "你好，欢迎使用语音合成。"},
      {"user": "请帮我查一下北京的天气。"}
    ]
  }
]
```
<!-- /REBUILD_FILE -->

