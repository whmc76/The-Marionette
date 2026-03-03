# The Marionette V2

LLM 驱动的用户评论批量生成工具，专为汽车品牌社交媒体运营设计。上传营销 brief 文档，自动解析评论分类与人设规则，通过大语言模型批量生成符合品牌调性的真实感评论，导出为 CSV。

---

## 功能特性

### 文档解析
- 支持上传 `.docx` 格式的评论运营 brief
- **三解析器自动竞选**：按置信度得分自动选择最佳解析器
  - `ClassifiedParser`：正向 / 反击 / 引导分类型 brief（如岚图梦想家）
  - `ScriptLibParser`：话术库 + 舆情引导型 brief（如启境尾翼）
  - `FallbackParser`：未知结构兜底，将全文作为背景上下文
- 两阶段 LLM 提取：基础信息（产品名、规则、禁用词）+ 评论分类（主题、人设、示例）
- 解析过程**流式显示**，支持 `<think>` 推理块可视化（DeepSeek / QwQ 等推理模型）
- 解析后进入**可编辑确认表单**，用户可修改分类、人设、示例后再生成

### 评论生成
- 两阶段流程：先**排产**（按比例分配任务清单），再**批量生成**
- **实时流式展示**：每解析出一条评论立即显示，无需等待整批完成
- 支持正向 / 反击 / 引导三类方向，按比例自动分配
- 已生成评论自动注入后续 prompt，减少重复率

### 质量校验
- **硬规则**（自动标记 `hard_fail`）：最小字数、禁用词/短语、空内容
- **软规则**（自动标记 `soft_flag`）：rapidfuzz token_sort_ratio > 70% 或 3-gram Jaccard > 0.6 的近重复对
- 自动修复句尾标点（。！？等）

### LLM 支持
| 服务商 | 说明 |
|--------|------|
| OpenAI / 兼容 | OpenAI、DeepSeek、Moonshot、硅基流动等任意 OpenAI 兼容接口 |
| Anthropic | Claude 系列模型（含 claude-opus-4-6 / sonnet-4-6 / haiku-4-5） |
| Ollama | 本地部署，无需 API Key，含 VRAM 优化参数 |

### Ollama 专项优化
- **动态 `num_ctx`**：解析前自动测量文档长度，计算所需 token，自动扩展上下文窗口（不会降低用户设定值）
- **模型切换自动卸载**：先查询 `/api/ps` 确认模型已加载，再发送 `keep_alive=0` 卸载，避免触发重新加载导致显存爆满
- **选项白名单过滤**：`flash_attn` / `kv_cache_type` 是 Ollama 环境变量而非 API 参数，会被自动剔除（发送无效选项会导致请求挂起）

### 导出
- UTF-8 BOM 编码 CSV（Excel 直接打开无乱码）
- 标准列：序号、评论内容、分类方向、主题、人设、字数、状态、备注
- 审计列：run_id、provider、model、prompt_version、生成时间

---

## 环境要求

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)（推荐，Windows 可由 `start.bat` 自动安装）

---

## 快速开始

### Windows（推荐）

```bat
start.bat
```

脚本会自动：检测并安装 `uv` → 从 `.env.example` 创建 `.env` → 同步依赖 → 启动应用。

首次运行会打开 `.env` 让你填写 API Key，保存后再次运行即可。

### 手动启动

```bash
# 1. 克隆项目
git clone <repo-url>
cd The-Marionette

# 2. 复制并填写 API Key
cp .env.example .env
# 编辑 .env，填入你使用的服务商 Key

# 3. 安装依赖
uv sync

# 4. 启动
uv run streamlit run app.py
```

浏览器访问 `http://localhost:8501`。

---

## 配置

### API Key（`.env`）

```env
# OpenAI 或任意兼容接口
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1

# Anthropic Claude
ANTHROPIC_API_KEY=sk-ant-...

# Ollama（本地，无需 Key）
OLLAMA_BASE_URL=http://localhost:11434
```

API Key **只存在 `.env`，不写入磁盘的其他位置，不进 git**。

### 运行时配置（`config.json`，自动生成）

通过 UI 侧边栏「⚙️ 配置 LLM」保存后自动写入，包含：

| 字段 | 说明 | 默认值 |
|------|------|--------|
| `provider` | 服务商（openai / anthropic / ollama） | `openai` |
| `model` | 模型名 | `gpt-4o-mini` |
| `temperature` | 生成温度 | `0.9` |
| `max_concurrency` | 最大并发数 | `5`（Ollama 自动限制为 2） |
| `batch_size` | 每批生成数量 | `8` |
| `ollama_num_ctx` | Ollama 上下文窗口基准值 | `8192` |
| `ollama_num_gpu` | GPU 层数（99 = 全 GPU） | `99` |

### Ollama VRAM 优化

`num_ctx` 和 `num_gpu` 通过每次请求的 `extra_body.options` 传递。

`flash_attn` 和 `kv_cache_type` 是 **Ollama 服务级环境变量**，需在启动 Ollama 服务前设置（UI 中提供复制命令）：

```powershell
# PowerShell — 在启动 Ollama 前执行
$env:OLLAMA_FLASH_ATTENTION = "1"
$env:OLLAMA_KV_CACHE_TYPE   = "q8_0"
ollama serve
```

---

## 使用流程

```
Step 1          Step 2              Step 3（侧边栏）     Step 3
上传 DOCX  →  解析结果确认  →    配置 LLM & 测试  →  生成 & 导出
或手动输入      可编辑表单          保存设置              实时查看评论
```

### Step 1 · 输入源

- **DOCX 模式**：上传 brief 文档，自动解析并流式显示解析过程
- **手动模式**：直接填写产品名、分类、人设等字段

### Step 2 · 解析确认

展示解析报告（识别到的分类数、人设数、禁用词数、比例）。所有字段均可在此页面编辑修改。确认无误后进入生成。

### Step 3（侧边栏）· LLM 配置

1. 选择服务商，填写 API Key / Base URL / 模型名
2. 点击「测试连接」验证
3. 保存设置（切换模型时会自动卸载旧 Ollama 模型）

### Step 3 · 生成 & 导出

- 设置生成总数，预览任务分配
- 点击「开始生成」，实时查看评论滚动出现（最新 30 条）
- 完成后查看校验摘要，筛选方向 / 人设 / 状态
- 下载 CSV 或保存到本地 `output/` 目录

---

## 项目结构

```
The-Marionette/
├── app.py                      # Streamlit 主入口（4步向导）
├── start.bat                   # Windows 一键启动脚本
├── .env.example                # API Key 模板
├── pyproject.toml              # 项目定义（uv / hatchling）
│
├── assets/                     # Brief 文档样本与截图
│
├── src/
│   ├── models.py               # Pydantic 数据契约（BriefSpec、CommentTask 等）
│   ├── config.py               # 配置管理（.env secrets + config.json）
│   ├── generator.py            # 生成引擎（排产 + 流式批量生成）
│   ├── validator.py            # 分层校验（硬规则 + rapidfuzz 去重）
│   ├── csv_writer.py           # CSV 导出（UTF-8 BOM + 审计列）
│   ├── parsers/
│   │   ├── base.py             # 解析器基类 + load_paragraphs
│   │   ├── classified.py       # ParserA：正向/反击分类型
│   │   ├── script_lib.py       # ParserB：话术库型
│   │   ├── fallback.py         # ParserFallback：兜底解析
│   │   └── llm_parser.py       # LLM 两阶段提取（动态 num_ctx）
│   └── llm/
│       ├── base.py             # 抽象基类（chat / stream_chat）
│       ├── openai_backend.py   # OpenAI 兼容 + Ollama（含 unload）
│       ├── anthropic_backend.py# Claude API（含 thinking block 流）
│       └── client.py           # 统一客户端（重试 / 并发 / 失败阈值）
│
├── prompts/
│   ├── system_prompt.py        # 系统提示词（注入产品背景与规则）
│   └── comment_prompt.py       # 评论生成提示词（注入已生成列表去重）
│
└── tests/
    ├── test_parsers.py         # 解析器单测
    ├── test_generator.py       # 生成管线集成测试（mock LLM）
    ├── test_csv.py             # CSV 输出快照测试
    ├── test_ollama_options.py  # Ollama extra_body 过滤测试
    └── test_ollama_unload.py   # Ollama 模型卸载正确性测试
```

---

## 测试

```bash
uv run pytest tests/ -v
```

37 个测试，覆盖：解析器置信度、生成管线（mock LLM）、CSV 编码、Ollama 选项过滤、Ollama 模型卸载逻辑。

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 前端 | Streamlit |
| 数据模型 | Pydantic v2 |
| DOCX 解析 | python-docx |
| LLM（OpenAI 兼容） | openai SDK |
| LLM（Claude） | anthropic SDK |
| 去重 | rapidfuzz |
| 配置安全 | python-dotenv |
| 包管理 | uv |

---

## 注意事项

- `.env` 已加入 `.gitignore`，API Key 不会被提交
- `config.json` 和 `output/` 同样被忽略
- Ollama 并发默认限制为 2，避免单机 VRAM 竞争
- 全局失败阈值：累计失败率 ≥ 15%（至少 10 次请求后）自动中止生成，提示检查配置
- 生成的评论仅供内部运营参考，使用时请遵守各平台社区规范
