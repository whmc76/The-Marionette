"""The Marionette V2 — LLM-driven comment generation wizard."""
from __future__ import annotations

import tempfile
import time

import streamlit as st

st.set_page_config(
    page_title="The Marionette · 评论生成器",
    page_icon="🎭",
    layout="wide",
)

from src.models import BriefSpec, CommentCategory, GeneratedComment
from src.presets import PRESETS, get_preset
from src.config import config, LLMSettings
from src.parsers import parse_brief
from src.llm.client import build_client, GenerationAborted
from src.llm.openai_backend import fetch_ollama_models, unload_ollama_model
from src.generator import schedule_tasks, run_generation, GenerationProgress
from src.validator import validate_comments
from src.csv_writer import to_csv_bytes, write_csv, make_filename
from prompts.comment_prompt import PROMPT_VERSION


# ══════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════

def _restore_client(llm: LLMSettings) -> None:
    """Rebuild LLMClient from saved settings (no network test). Called on startup."""
    try:
        ollama_opts = {
            "num_ctx": llm.ollama_num_ctx,
            "num_gpu": llm.ollama_num_gpu,
            "flash_attn": llm.ollama_flash_attn,
            "kv_cache_type": llm.ollama_kv_cache_type,
        } if llm.provider == "ollama" else None
        client = build_client(
            provider=llm.provider,
            api_key=config.env_api_key(llm.provider),
            base_url=llm.base_url,
            model=llm.model,
            max_concurrency=llm.max_concurrency,
            max_retries=llm.max_retries,
            failure_threshold=llm.failure_threshold,
            timeout=120.0 if llm.provider == "ollama" else 30.0,
            ollama_options=ollama_opts,
        )
        st.session_state.llm_client = client
        st.session_state.conn_verified = True
    except Exception:
        # Silent fail — user will see "未验证" and can re-test in settings
        st.session_state.conn_verified = False


def _init_state() -> None:
    defaults: dict = {
        # navigation
        "page": "wizard",       # "wizard" | "settings"
        "step": 1,              # 1 | 2 | 3
        # brief
        "spec": None,
        "parse_report": None,
        "all_reports": [],
        # generation
        "tasks": [],
        "comments": [],
        "validation_result": None,
        "run_id": "",
        "llm_config": {},       # active config snapshot used for last run
        # llm settings (loaded from disk, may be overridden in session)
        "llm": None,            # LLMSettings instance
        "llm_client": None,     # connected LLMClient
        "conn_verified": False, # whether test_connection() passed
        # settings page temp key storage (session-only, never persisted)
        "settings_api_key": "",
        # industry selection
        "industry": "general",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Load LLM settings from disk on first run, auto-restore client if previously verified
    if st.session_state.llm is None:
        llm = config.load_llm_settings()
        st.session_state.llm = llm
        if llm.conn_verified and st.session_state.llm_client is None:
            _restore_client(llm)


_init_state()


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════

def _render_sidebar() -> None:
    with st.sidebar:
        st.title("🎭 The Marionette")
        st.caption("LLM 驱动的用户评论批量生成工具")
        st.divider()

        # ── Wizard steps ─────────────────────────────────────────
        if st.session_state.page == "wizard":
            step = st.session_state.step
            steps = ["输入源", "解析确认", "生成 & 导出"]
            for i, name in enumerate(steps, 1):
                icon = "✅" if i < step else ("▶️" if i == step else "⬜")
                st.write(f"{icon} Step {i}：{name}")
        else:
            st.write("⚙️ **LLM 设置**")

        st.divider()

        # ── LLM settings entry point ──────────────────────────────
        llm: LLMSettings = st.session_state.llm
        conn_ok: bool = st.session_state.conn_verified

        status_icon = "✅" if conn_ok else "⚠️"
        status_text = "已验证" if conn_ok else "未验证"
        st.markdown(f"**⚙️ LLM 设置**")
        st.caption(
            f"{status_icon} {llm.provider_label()} · `{llm.model}`\n\n"
            f"状态：{status_text}"
        )

        if st.session_state.page == "settings":
            if st.button("← 返回向导", use_container_width=True):
                st.session_state.page = "wizard"
                st.rerun()
        else:
            if st.button("⚙️ 配置 LLM", use_container_width=True, type="secondary"):
                st.session_state.page = "settings"
                st.rerun()

        st.divider()
        if st.button("↩ 重新开始", use_container_width=True):
            keep = {"llm", "llm_client", "conn_verified", "settings_api_key"}
            for k in [k for k in st.session_state if k not in keep]:
                del st.session_state[k]
            st.rerun()


# ══════════════════════════════════════════════════════════════════
# SETTINGS PAGE
# ══════════════════════════════════════════════════════════════════

_PROVIDER_LABELS = {
    "openai": "OpenAI / 兼容接口（DeepSeek、Moonshot 等）",
    "anthropic": "Anthropic Claude",
    "ollama": "Ollama（本地模型）",
}
_DEFAULT_URLS = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "",
    "ollama": "http://localhost:11434",
}
_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "ollama": "qwen2.5:7b",
}


@st.cache_data(ttl=30, show_spinner=False)
def _cached_ollama_models(base_url: str) -> list[str]:
    """Fetch Ollama model list; cached 30s per base_url."""
    return fetch_ollama_models(base_url)


def _ollama_model_picker(base_url: str, saved_model: str) -> str:
    """Render Ollama model selector with auto-fetch and manual refresh."""
    col_status, col_refresh = st.columns([5, 1])

    try:
        models = _cached_ollama_models(base_url)
    except Exception as exc:
        col_status.warning(f"无法连接 Ollama（{exc}）— 请确认服务已启动，或手动输入模型名")
        return st.text_input(
            "模型名称（手动输入）",
            value=saved_model or "qwen2.5:7b",
            placeholder="qwen2.5:7b",
        )

    if not models:
        col_status.warning("Ollama 已连接，但本地没有已下载的模型。请先运行 `ollama pull <model>`")
        return st.text_input("模型名称", value=saved_model or "", placeholder="qwen2.5:7b")

    col_status.success(f"检测到 {len(models)} 个本地模型")
    if col_refresh.button("↺", help="刷新模型列表"):
        _cached_ollama_models.clear()
        st.rerun()

    default_idx = models.index(saved_model) if saved_model in models else 0
    return st.selectbox("选择模型", models, index=default_idx)


def _settings_page() -> None:
    st.header("⚙️ LLM 配置")
    st.caption("配置将自动保存到 `config.json`；API Key 仅在本次会话中生效，请在 `.env` 中长期保存。")
    st.divider()

    llm: LLMSettings = st.session_state.llm

    # ── 1. 服务商 ─────────────────────────────────────────────────
    st.subheader("1 · 服务商")
    provider_options = list(_PROVIDER_LABELS.keys())
    provider_idx = provider_options.index(llm.provider) if llm.provider in provider_options else 0
    provider = st.radio(
        "选择服务商",
        provider_options,
        index=provider_idx,
        format_func=lambda p: _PROVIDER_LABELS[p],
        horizontal=False,
        label_visibility="collapsed",
    )

    # ── 2. 连接配置 ───────────────────────────────────────────────
    st.subheader("2 · 连接配置")

    # Base URL
    if provider == "anthropic":
        base_url = ""
        st.info("Claude API 地址由 SDK 自动管理，无需填写")
    else:
        default_url = llm.base_url if llm.provider == provider else _DEFAULT_URLS[provider]
        base_url = st.text_input(
            "API 地址",
            value=default_url,
            placeholder=_DEFAULT_URLS[provider],
            help="OpenAI 兼容服务填入对应 base_url；Ollama 默认 http://localhost:11434",
        )

    # Model — Ollama: fetch dropdown; others: text input
    if provider == "ollama":
        model = _ollama_model_picker(base_url, saved_model=llm.model if llm.provider == "ollama" else "")
    else:
        default_model = llm.model if llm.provider == provider else _DEFAULT_MODELS[provider]
        model = st.text_input(
            "模型名称",
            value=default_model,
            placeholder=_DEFAULT_MODELS[provider],
        )

    # API Key (session-only)
    env_key = config.env_api_key(provider)
    if provider != "ollama":
        key_help = "已在 .env 中配置则留空；此处输入仅在本次会话有效" if env_key else "请输入 API Key（或在 .env 文件中配置）"
        key_placeholder = "已从 .env 读取，留空即可" if env_key else f"{provider.upper()}_API_KEY"
        api_key_input = st.text_input(
            "API Key",
            value=st.session_state.settings_api_key,
            type="password",
            placeholder=key_placeholder,
            help=key_help,
        )
        st.session_state.settings_api_key = api_key_input
        effective_key = api_key_input or env_key
    else:
        api_key_input = ""
        effective_key = ""
        st.caption("Ollama 本地服务无需 API Key")

    # ── 3. 生成参数 ───────────────────────────────────────────────
    st.subheader("3 · 生成参数")
    p1, p2, p3, p4 = st.columns(4)
    temperature = p1.slider(
        "Temperature",
        min_value=0.5, max_value=1.5,
        value=float(llm.temperature), step=0.05,
        help="越高越有创意，越低越稳定",
    )
    is_ollama = provider == "ollama"
    default_concurrency = 1 if is_ollama else llm.max_concurrency
    default_batch = 1 if is_ollama else int(llm.batch_size)
    max_concurrency = p2.number_input(
        "并发数",
        min_value=1, max_value=20,
        value=default_concurrency,
        help="Ollama 本地模型建议设为 1，大多数电脑同时只能跑一个推理线程",
    )
    batch_size = p3.number_input(
        "批次大小",
        min_value=1, max_value=20,
        value=default_batch,
        help="Ollama 建议设为 1，避免长时间占用导致超时",
    )
    if is_ollama and (max_concurrency > 1 or batch_size > 1):
        st.caption("⚠️ Ollama 并发 > 1 或批次 > 1 可能导致请求排队或超时，建议保持默认值 1")
    max_retries = p4.number_input(
        "重试次数",
        min_value=0, max_value=5,
        value=int(llm.max_retries),
        help="单条失败后的最大重试次数",
    )

    # ── 4. Ollama 显存优化（仅 Ollama 显示）───────────────────────
    if is_ollama:
        st.subheader("4 · Ollama 显存优化")

        ov1, ov2 = st.columns(2)
        _ctx_opts = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
        ollama_num_ctx = ov1.select_slider(
            "上下文窗口 (num_ctx) — 随请求下发",
            options=_ctx_opts,
            value=llm.ollama_num_ctx if llm.ollama_num_ctx in _ctx_opts else 8192,
            help="KV cache 随上下文线性增长。128K 时可达数十 GB；"
                 "本工具单次输入 ≤ 8 000 字，8 192 完全够用。"
                 "修改后保存 → 会自动卸载旧模型，下次请求以新 num_ctx 重新加载。",
        )
        ollama_num_gpu = ov2.number_input(
            "GPU 层数 (num_gpu) — 随请求下发",
            min_value=0, max_value=999,
            value=llm.ollama_num_gpu,
            help="99 = 全部层卸载到 GPU；0 = 强制 CPU 推理。",
        )

        # KV-cache estimate
        _kv_mb = int(ollama_num_ctx) * 28 * 32 * 128 * 2 * 2 / 1024 / 1024  # fp16 base
        st.caption(f"⚡ 预估 KV Cache（fp16 基准）：**{_kv_mb:.0f} MB**（7B 参考值）")

        # flash_attn / kv_cache_type are Ollama env vars, not API options
        ollama_flash_attn = llm.ollama_flash_attn
        ollama_kv_cache_type = llm.ollama_kv_cache_type

        with st.expander("🔧 进一步降低显存：Ollama 环境变量（需重启 Ollama 服务）"):
            _fa_val = "1" if ollama_flash_attn else "0"
            _kv_val = ollama_kv_cache_type
            st.code(
                f"# 在启动 Ollama 前设置（Windows PowerShell）\n"
                f"$env:OLLAMA_FLASH_ATTENTION = \"{_fa_val}\"   "
                f"# Flash Attention：降低长序列峰值显存\n"
                f"$env:OLLAMA_KV_CACHE_TYPE  = \"{_kv_val}\"  "
                f"# KV Cache 精度：q8_0 省 ~50%，q4_0 省 ~75%",
                language="powershell",
            )
            ev1, ev2 = st.columns(2)
            ollama_flash_attn = ev1.checkbox(
                "Flash Attention (OLLAMA_FLASH_ATTENTION)",
                value=llm.ollama_flash_attn,
                help="仅更新上方命令示例中的值，不会自动重启 Ollama。",
            )
            ollama_kv_cache_type = ev2.selectbox(
                "KV Cache 精度 (OLLAMA_KV_CACHE_TYPE)",
                options=["f16", "q8_0", "q4_0"],
                index=["f16", "q8_0", "q4_0"].index(
                    llm.ollama_kv_cache_type
                    if llm.ollama_kv_cache_type in ("f16", "q8_0", "q4_0")
                    else "q8_0"
                ),
            )
    else:
        ollama_num_ctx = llm.ollama_num_ctx
        ollama_num_gpu = llm.ollama_num_gpu
        ollama_flash_attn = llm.ollama_flash_attn
        ollama_kv_cache_type = llm.ollama_kv_cache_type

    st.divider()

    # ── 操作按钮 ──────────────────────────────────────────────────
    btn_test, btn_save, status_area = st.columns([1, 1, 2])

    if btn_test.button("🔌 测试连接", use_container_width=True):
        if not effective_key and provider != "ollama":
            st.error("请填写 API Key 或在 .env 中配置")
        else:
            with st.spinner("连接测试中…"):
                try:
                    # Only pass options that Ollama accepts as API params.
                    # flash_attn / kv_cache_type are env vars, not API options.
                    _ollama_opts = {
                        "num_ctx": int(ollama_num_ctx),
                        "num_gpu": int(ollama_num_gpu),
                    } if provider == "ollama" else None
                    client = build_client(
                        provider=provider,
                        api_key=effective_key,
                        base_url=base_url,
                        model=model,
                        max_concurrency=int(max_concurrency),
                        ollama_options=_ollama_opts,
                    )
                    ok = client.test_connection()
                    if ok:
                        st.session_state.llm_client = client
                        st.session_state.conn_verified = True
                        # Sync session llm with the tested form values so all
                        # labels (sidebar, step 2 headers) show the right model.
                        st.session_state.llm = LLMSettings(
                            provider=provider,
                            base_url=base_url,
                            model=model,
                            temperature=float(temperature),
                            max_concurrency=int(max_concurrency),
                            batch_size=int(batch_size),
                            max_retries=int(max_retries),
                            ollama_num_ctx=int(ollama_num_ctx),
                            ollama_num_gpu=int(ollama_num_gpu),
                            ollama_flash_attn=bool(ollama_flash_attn),
                            ollama_kv_cache_type=ollama_kv_cache_type,
                            conn_verified=True,
                        )
                        st.toast("✅ 连接成功", icon="✅")
                    else:
                        st.session_state.conn_verified = False
                        st.error("❌ 连接失败，请检查地址和 Key")
                except Exception as exc:
                    st.session_state.conn_verified = False
                    st.error(f"❌ {exc}")

    if btn_save.button("💾 保存设置", use_container_width=True, type="primary"):
        old_llm: LLMSettings = st.session_state.llm
        conn_params_changed = (
            provider != old_llm.provider
            or base_url != old_llm.base_url
            or model != old_llm.model
            # num_ctx / num_gpu changes require model reload to take effect
            or int(ollama_num_ctx) != old_llm.ollama_num_ctx
            or int(ollama_num_gpu) != old_llm.ollama_num_gpu
        )
        # If connection-critical params changed, the existing client is stale.
        # Reset verification so the user knows they must re-test.
        verified = st.session_state.conn_verified and not conn_params_changed
        new_llm = LLMSettings(
            provider=provider,
            base_url=base_url,
            model=model,
            temperature=float(temperature),
            max_concurrency=int(max_concurrency),
            batch_size=int(batch_size),
            max_retries=int(max_retries),
            ollama_num_ctx=int(ollama_num_ctx),
            ollama_num_gpu=int(ollama_num_gpu),
            ollama_flash_attn=bool(ollama_flash_attn),
            ollama_kv_cache_type=ollama_kv_cache_type,
            conn_verified=verified,
        )
        config.save_llm_settings(new_llm)
        st.session_state.llm = new_llm
        if conn_params_changed:
            # If the previous backend was Ollama, evict its model from VRAM
            if old_llm.provider == "ollama":
                unload_ollama_model(old_llm.base_url, old_llm.model)
            # Discard stale client so nothing accidentally uses the old model
            st.session_state.llm_client = None
            st.session_state.conn_verified = False
        if api_key_input and provider != "ollama":
            config.set_session_key(provider, api_key_input)
        st.toast("✅ 设置已保存到 config.json", icon="💾")
        st.rerun()

    with status_area:
        if st.session_state.conn_verified:
            st.success(f"✅ 已验证：{st.session_state.llm.model} ({st.session_state.llm.provider})")
        else:
            st.warning("⚠️ 尚未通过连接验证")

    # ── 当前配置预览 ───────────────────────────────────────────────
    with st.expander("📄 当前 config.json 内容", expanded=False):
        import json
        raw = config._file
        st.code(json.dumps(raw, ensure_ascii=False, indent=2), language="json")


# ══════════════════════════════════════════════════════════════════
# STEP 1 — INPUT SOURCE
# ══════════════════════════════════════════════════════════════════

def _step1() -> None:
    st.header("Step 1 · 选择输入源")

    # ── 行业选择器（两个分支共用）────────────────────────────────
    _industry_keys = [p.key for p in PRESETS]
    _default_idx = _industry_keys.index(st.session_state.industry) if st.session_state.industry in _industry_keys else _industry_keys.index("general")
    selected_industry = st.selectbox(
        "行业类型",
        options=_industry_keys,
        format_func=lambda k: f"{get_preset(k).icon} {get_preset(k).label}",
        index=_default_idx,
        key="industry_selector",
    )
    st.session_state.industry = selected_industry

    mode = st.radio("输入方式", ["📄 上传 DOCX Brief", "✏️ 手动输入"], horizontal=True)

    if mode == "📄 上传 DOCX Brief":
        _step1_docx()
    else:
        _step1_manual()


def _step1_docx() -> None:
    from src.parsers.base import load_paragraphs
    from src.parsers.classified import ClassifiedParser
    from src.parsers.script_lib import ScriptLibParser
    from src.parsers.fallback import FallbackParser
    from src.models import ParseReport

    uploaded = st.file_uploader("上传 Brief 文件（.docx）", type=["docx"])
    if not uploaded:
        return

    st.success(f"已选择：{uploaded.name}")

    conn_ok: bool = st.session_state.conn_verified
    llm: LLMSettings = st.session_state.llm
    if conn_ok:
        st.info(f"将使用 **LLM 解析**（{llm.model}）智能提取结构化信息")
    else:
        st.warning("未配置 LLM — 将使用正则规则解析（效果有限）。建议先在侧边栏「⚙️ 配置 LLM」完成配置。")

    if not st.button("解析 →", type="primary"):
        return

    # Save to temp file first
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    reports: list[ParseReport] = []
    client = st.session_state.llm_client if conn_ok else None

    with st.status("正在解析 Brief…", expanded=True) as status:

        # ── Step 1: Load document ─────────────────────────────────
        st.write("📄 读取文档内容…")
        try:
            paragraphs = load_paragraphs(tmp_path)
            char_count = sum(len(p) for p in paragraphs)
            st.write(f"&nbsp;&nbsp;&nbsp;✓ 共 {len(paragraphs)} 个段落 · {char_count} 字符")
        except Exception as exc:
            status.update(label="文档读取失败", state="error")
            st.error(str(exc))
            return

        # ── Step 2: LLM parse (two-phase: streaming tokens + field summary) ──
        if client:
            import re as _re
            from src.parsers.llm_parser import LLMParser

            # Streaming display: thinking box + JSON output box
            _stream_buf: list[str] = [""]
            _think_ph = st.empty()   # <think>...</think> reasoning content
            _out_ph = st.empty()     # regular JSON output tokens
            _phase_label_ph = st.empty()
            _field_ph = st.empty()
            _field_lines: list[str] = []

            _PHASE_LABELS = {
                "产品名称": f"🤖 **{llm.model}** — 第一阶段：提取基础信息",
                "评论分类": f"🤖 **{llm.model}** — 第二阶段：提取评论分类",
            }

            def _split_think(text: str) -> tuple[str, str]:
                """Return (think_text, output_text) by parsing <think> tags."""
                think_parts: list[str] = []
                out_parts: list[str] = []
                pos = 0
                for m in _re.finditer(r"<think>([\s\S]*?)</think>", text):
                    out_parts.append(text[pos:m.start()])
                    think_parts.append(m.group(1))
                    pos = m.end()
                tail = text[pos:]
                # Unclosed <think> at end means model is still reasoning
                open_m = _re.search(r"<think>([\s\S]*)$", tail)
                if open_m:
                    out_parts.append(tail[: open_m.start()])
                    think_parts.append(open_m.group(1) + "▌")
                else:
                    out_parts.append(tail)
                return "".join(think_parts), "".join(out_parts)

            def on_token(chunk: Optional[str]) -> None:
                if chunk is None:
                    # Phase boundary: reset both buffers
                    _stream_buf[0] = ""
                    _think_ph.empty()
                    _out_ph.empty()
                    return
                _stream_buf[0] += chunk
                think_text, out_text = _split_think(_stream_buf[0])

                # ── Thinking panel ────────────────────────────────────
                if think_text.strip():
                    lines = think_text.rstrip("\n").split("\n")
                    if len(lines) > 60:
                        lines = [f"…（省略 {len(lines) - 60} 行）…"] + lines[-60:]
                    _think_ph.markdown(
                        "💭 **思考过程**\n```\n" + "\n".join(lines) + "\n```"
                    )

                # ── JSON output panel ─────────────────────────────────
                out_stripped = out_text.strip()
                if out_stripped:
                    display = (
                        out_stripped
                        if len(out_stripped) <= 3000
                        else "…\n" + out_stripped[-3000:]
                    )
                    _out_ph.code(display, language="json")

            def on_step(field: str, result: str) -> None:
                # Switch phase header when first field of each phase arrives
                if field in _PHASE_LABELS:
                    _phase_label_ph.write(_PHASE_LABELS[field])
                    _field_lines.clear()
                icon = "⏳" if result == "…" else "✓"
                line = f"&nbsp;&nbsp;&nbsp;{icon} **{field}**：{result}"
                for i, l in enumerate(_field_lines):
                    parts = l.split("**")
                    if len(parts) >= 2 and parts[1] == field:
                        _field_lines[i] = line
                        break
                else:
                    _field_lines.append(line)
                _field_ph.markdown("\n\n".join(_field_lines))

            try:
                llm_report = LLMParser(
                    client, on_step=on_step, on_token=on_token,
                    industry=st.session_state.industry,
                ).parse(tmp_path)
                # Clear streaming boxes after completion; keep field summary
                _think_ph.empty()
                _out_ph.empty()
                reports.append(llm_report)
                if not (llm_report.spec and llm_report.category_count > 0):
                    st.warning("LLM 未提取到分类，将由规则解析器兜底")
            except Exception as exc:
                _think_ph.empty()
                _out_ph.empty()
                st.warning(f"LLM 解析异常：{exc}")

        # ── Step 3: Regex parsers ─────────────────────────────────
        label = "📐 运行规则解析器（备用对比）…" if client else "📐 运行规则解析器…"
        st.write(label)
        for parser in [ClassifiedParser(), ScriptLibParser(), FallbackParser()]:
            try:
                r = parser.parse(tmp_path)
                reports.append(r)
                if r.confidence > 0.1:
                    st.write(
                        f"&nbsp;&nbsp;&nbsp;· {r.parser_name}: "
                        f"{r.category_count} 分类 · 置信度 {r.confidence:.0%}"
                    )
            except Exception as exc:
                st.write(f"&nbsp;&nbsp;&nbsp;· {parser.name}: 异常 — {exc}")

        # ── Step 4: Select best ───────────────────────────────────
        if not reports or all(r.spec is None for r in reports):
            status.update(label="解析失败，无可用结果", state="error")
            st.error("所有解析器均未返回有效结果，请检查文件格式")
            return

        best = max(reports, key=lambda r: r.confidence)
        finish_label = (
            f"解析完成 — {best.parser_name} · {best.category_count} 个分类"
            if best.spec else "解析完成（结果较弱，请在下一步手动补充）"
        )
        status.update(label=finish_label, state="complete")

    # Advance to step 2
    st.session_state.parse_report = best
    st.session_state.all_reports = reports
    st.session_state.spec = best.spec
    st.session_state.step = 2
    st.rerun()


def _step1_manual() -> None:
    st.subheader("手动填写产品信息")
    _preset = get_preset(st.session_state.industry)
    with st.form("manual_brief"):
        product_name = st.text_input("产品名称 *", placeholder=f"如：{_preset.product_examples[0]}")
        product_bg = st.text_area("产品背景（简介）", height=100)
        min_chars = st.number_input("评论最短字数", min_value=10, max_value=100, value=20)
        general_rules_raw = st.text_area(
            "通用写作规则（每行一条）",
            placeholder="自然口语化\n不要出现品牌广告语",
            height=120,
        )
        forbidden_raw = st.text_area("禁用词/短语（逗号或换行分隔）", placeholder="官方用语, 最优惠")
        platforms_raw = st.text_input("目标平台（逗号分隔）", placeholder="微博, 小红书, 抖音")
        num_cats = st.number_input("添加几个分类？", min_value=1, max_value=10, value=2)
        submitted = st.form_submit_button("确认并下一步 →", type="primary")

    if submitted:
        if not product_name.strip():
            st.error("产品名称不能为空")
            return
        import re
        categories = [
            CommentCategory(
                direction="正向" if i % 2 == 0 else "反击",
                theme=f"分类 {i + 1}（请在下一步编辑）",
                personas=["真实用户"],
            )
            for i in range(int(num_cats))
        ]
        spec = BriefSpec(
            title=product_name,
            product_name=product_name,
            product_background=product_bg,
            general_rules=[r.strip() for r in general_rules_raw.splitlines() if r.strip()],
            forbidden_phrases=[w.strip() for w in re.split(r"[，,\n]+", forbidden_raw) if w.strip()],
            categories=categories,
            positive_ratio=0.5,
            negative_ratio=0.5,
            min_char_length=int(min_chars),
            platform_targets=[p.strip() for p in platforms_raw.split(",") if p.strip()],
            industry=st.session_state.industry,
        )
        st.session_state.spec = spec
        st.session_state.parse_report = None
        st.session_state.step = 2
        st.rerun()


# ══════════════════════════════════════════════════════════════════
# STEP 2 — PARSE CONFIRMATION
# ══════════════════════════════════════════════════════════════════

def _step2() -> None:
    st.header("Step 2 · 解析确认 & 编辑")
    spec: BriefSpec = st.session_state.spec

    report = st.session_state.parse_report
    if report:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("解析器", report.parser_name)
        c2.metric("置信度", f"{report.confidence:.0%}")
        c3.metric("分类数", report.category_count)
        c4.metric("人设数", report.persona_count)
        c5.metric("禁用词", report.forbidden_phrase_count)
        for w in report.warnings:
            st.warning(w)
        with st.expander("所有解析器报告"):
            for r in st.session_state.all_reports:
                tag = "🤖 LLM" if r.parser_name == "llm" else f"📐 {r.parser_name}"
                st.write(f"{tag} · 置信度 {r.confidence:.0%} · {r.category_count} 分类 · {r.example_count} 示例")
                for w in r.warnings:
                    st.caption(f"⚠ {w}")
        st.divider()

    with st.form("confirm_spec"):
        st.subheader("基本信息")
        c1, c2 = st.columns(2)
        product_name = c1.text_input("产品名称", value=spec.product_name)
        min_chars = c2.number_input("最短字数", min_value=5, max_value=200, value=spec.min_char_length)
        product_bg = st.text_area("产品背景", value=spec.product_background, height=80)
        c3, c4 = st.columns(2)
        pos_ratio = c3.slider("正向比例", 0.0, 1.0, float(spec.positive_ratio), 0.05)
        platforms_str = c4.text_input("目标平台（逗号分隔）", value=", ".join(spec.platform_targets))
        general_rules_str = st.text_area("通用规则（每行一条）", value="\n".join(spec.general_rules), height=100)
        forbidden_str = st.text_area("禁用词（逗号/换行分隔）", value=", ".join(spec.forbidden_phrases), height=80)

        st.subheader(f"评论分类（共 {len(spec.categories)} 个）")
        st.caption("展开每个分类可编辑")
        cat_edits: list[dict] = []
        for ci, cat in enumerate(spec.categories):
            with st.expander(f"[{cat.direction}] {cat.theme}", expanded=ci == 0):
                cc1, cc2 = st.columns(2)
                direction = cc1.selectbox(
                    "方向", ["正向", "反击", "引导"],
                    index=["正向", "反击", "引导"].index(cat.direction) if cat.direction in ["正向", "反击", "引导"] else 0,
                    key=f"dir_{ci}",
                )
                theme = cc2.text_input("主题名", value=cat.theme, key=f"theme_{ci}")
                sub_themes_str = st.text_input("子主题（逗号分隔）", value=", ".join(cat.sub_themes), key=f"sub_{ci}")
                description = st.text_area("描述", value=cat.description, height=60, key=f"desc_{ci}")
                personas_str = st.text_input("人设（逗号分隔）", value=", ".join(cat.personas), key=f"persona_{ci}")
                examples_str = st.text_area("示例评论（每行一条）", value="\n".join(cat.example_comments), height=80, key=f"ex_{ci}")
                cat_edits.append({
                    "direction": direction, "theme": theme,
                    "sub_themes_str": sub_themes_str, "description": description,
                    "personas_str": personas_str, "examples_str": examples_str,
                })

        submitted = st.form_submit_button("确认并下一步 →", type="primary")

    if submitted:
        import re
        new_categories = []
        for ce in cat_edits:
            new_categories.append(CommentCategory(
                direction=ce["direction"],
                theme=ce["theme"],
                sub_themes=[s.strip() for s in re.split(r"[,，]+", ce["sub_themes_str"]) if s.strip()],
                description=ce["description"],
                personas=[p.strip() for p in re.split(r"[,，\n]+", ce["personas_str"]) if p.strip()] or ["真实用户"],
                example_comments=[e.strip() for e in ce["examples_str"].splitlines() if e.strip()],
            ))
        new_spec = BriefSpec(
            title=spec.title,
            product_name=product_name,
            product_background=product_bg,
            general_rules=[r.strip() for r in general_rules_str.splitlines() if r.strip()],
            forbidden_phrases=[w.strip() for w in __import__("re").split(r"[,，\n]+", forbidden_str) if w.strip()],
            categories=new_categories,
            positive_ratio=pos_ratio,
            negative_ratio=round(1.0 - pos_ratio, 2),
            min_char_length=int(min_chars),
            platform_targets=[p.strip() for p in platforms_str.split(",") if p.strip()],
            industry=spec.industry,
        )
        st.session_state.spec = new_spec
        st.session_state.step = 3
        st.rerun()

    if st.button("← 返回 Step 1"):
        st.session_state.step = 1
        st.rerun()


# ══════════════════════════════════════════════════════════════════
# STEP 3 — GENERATE & EXPORT
# ══════════════════════════════════════════════════════════════════

def _step3() -> None:
    st.header("Step 3 · 生成 & 导出")
    spec: BriefSpec = st.session_state.spec
    llm: LLMSettings = st.session_state.llm

    # ── LLM config summary banner ─────────────────────────────────
    conn_ok = st.session_state.conn_verified
    if conn_ok:
        st.success(
            f"✅ LLM 就绪：**{llm.model}** · {llm.provider_label()} · "
            f"Temperature {llm.temperature} · 并发 {llm.max_concurrency}"
        )
    else:
        st.error(
            "❌ LLM 未配置或未通过连接验证，请先点击侧边栏「⚙️ 配置 LLM」完成设置",
            icon="❌",
        )
        st.stop()

    # ── Generation count ──────────────────────────────────────────
    st.divider()
    c1, c2 = st.columns([1, 3])
    total_count = c1.number_input(
        "生成总数", min_value=1, max_value=2000, value=20,
        help=f"按正向 {spec.positive_ratio:.0%} / 其他 {spec.negative_ratio:.0%} 分配",
    )
    c2.caption(
        f"分类数：{len(spec.categories)} · "
        f"正向 {spec.positive_ratio:.0%} / 其他 {spec.negative_ratio:.0%} · "
        f"批次大小：{llm.batch_size}"
    )

    # Task preview
    with st.expander("任务分配预览", expanded=False):
        import pandas as pd
        preview_tasks = schedule_tasks(spec, int(total_count), batch_size=llm.batch_size)
        if preview_tasks:
            df_tasks = pd.DataFrame([
                {"方向": t.category.direction, "主题": t.category.theme,
                 "人设": t.persona, "数量": t.target_count, "批次": t.batch_index}
                for t in preview_tasks
            ])
            st.dataframe(df_tasks, use_container_width=True, hide_index=True)
            st.caption(f"共 {len(preview_tasks)} 个任务批次")
        else:
            st.warning("无可用分类，请返回 Step 2 添加分类")

    st.divider()

    # ── Generate / re-generate ────────────────────────────────────
    already_done = bool(st.session_state.comments)

    if not already_done:
        if st.button("🚀 开始生成", type="primary", use_container_width=True):
            tasks = schedule_tasks(spec, int(total_count), batch_size=llm.batch_size)
            st.session_state.tasks = tasks
            _run_generation(spec, tasks, llm)
    else:
        st.success(f"已生成 {len(st.session_state.comments)} 条评论")
        if st.button("🔄 重新生成", use_container_width=True):
            st.session_state.comments = []
            st.session_state.validation_result = None
            st.rerun()

    if st.session_state.comments:
        _render_results()

    if st.button("← 返回 Step 2"):
        st.session_state.step = 2
        st.rerun()


def _run_generation(spec: BriefSpec, tasks: list, llm: LLMSettings) -> None:
    import pandas as pd

    client = st.session_state.llm_client
    if client is None:
        st.error("LLM 客户端未初始化，请在设置页测试连接")
        return

    total = sum(t.target_count for t in tasks)
    progress_bar = st.progress(0, text="准备中…")
    metrics_ph = st.empty()
    live_ph = st.empty()

    live_comments: list[GeneratedComment] = []

    def _refresh_live(p_done: int) -> None:
        """Re-render the live comment table inside live_ph."""
        with live_ph.container():
            st.caption(f"实时预览 — 已生成 {p_done} / {total} 条（最新 30 条）")
            if not live_comments:
                return
            recent = live_comments[-30:]
            offset = max(0, len(live_comments) - 30)
            df = pd.DataFrame([
                {
                    "#": offset + i + 1,
                    "评论内容": c.text,
                    "方向": c.category_direction,
                    "主题": c.theme,
                    "人设": c.persona,
                    "字数": c.char_count,
                }
                for i, c in enumerate(recent)
            ])
            st.dataframe(df, use_container_width=True, hide_index=True,
                         height=min(38 * len(recent) + 38, 420))

    def on_comment(comment: GeneratedComment) -> None:
        live_comments.append(comment)
        _refresh_live(len(live_comments))

    def on_progress(p: GenerationProgress) -> None:
        pct = min(1.0, p.done / max(total, 1))
        eta = f"{p.eta_seconds:.0f}s" if p.eta_seconds > 0 else "—"
        progress_bar.progress(pct, text=f"生成中… {p.done}/{total}  ETA {eta}")
        with metrics_ph.container():
            m1, m2, m3 = st.columns(3)
            m1.metric("成功", p.success)
            m2.metric("失败", p.failed)
            m3.metric("预计剩余", eta)

    try:
        result = run_generation(
            spec=spec, tasks=tasks, client=client,
            temperature=llm.temperature,
            on_progress=on_progress,
            on_comment=on_comment,
        )
        validated, val_result = validate_comments(result.comments, spec)
        st.session_state.comments = validated
        st.session_state.validation_result = val_result
        st.session_state.run_id = result.run_id
        st.session_state.llm_config = {
            "provider": llm.provider, "model": llm.model,
            "temperature": llm.temperature,
        }
        progress_bar.empty()
        metrics_ph.empty()
        live_ph.empty()
        st.success(
            f"生成完成！共 {val_result.total} 条 · "
            f"通过 {val_result.passed} · 硬失败 {val_result.hard_failed} · "
            f"软标记 {val_result.soft_flagged}"
        )
        st.rerun()
    except GenerationAborted as exc:
        st.error(f"生成中止（失败率过高）：{exc}")
    except Exception as exc:
        st.error(f"生成异常：{exc}")


def _render_results() -> None:
    import pandas as pd

    comments = st.session_state.comments
    val_result = st.session_state.validation_result
    llm_cfg = st.session_state.llm_config
    run_id = st.session_state.get("run_id", "")

    if val_result:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("总计", val_result.total)
        c2.metric("通过", val_result.passed)
        c3.metric("硬失败", val_result.hard_failed)
        c4.metric("软标记", val_result.soft_flagged)
        st.divider()

    directions = sorted({c.category_direction for c in comments})
    personas = sorted({c.persona for c in comments})
    fc1, fc2, fc3 = st.columns(3)
    filter_dir = fc1.multiselect("筛选方向", directions, default=directions)
    filter_persona = fc2.multiselect("筛选人设", personas, default=personas)
    filter_status = fc3.multiselect(
        "筛选状态", ["pass", "soft_flag", "hard_fail"], default=["pass", "soft_flag"]
    )

    filtered = [
        c for c in comments
        if c.category_direction in filter_dir
        and c.persona in filter_persona
        and c.validation_status in filter_status
    ]

    df = pd.DataFrame([
        {"序号": i + 1, "评论内容": c.text, "方向": c.category_direction,
         "主题": c.theme, "人设": c.persona, "字数": c.char_count,
         "状态": c.validation_status, "备注": "; ".join(c.errors)}
        for i, c in enumerate(filtered)
    ])
    st.dataframe(df, use_container_width=True, hide_index=True, height=400)
    st.divider()

    spec: BriefSpec = st.session_state.spec
    filename = make_filename(spec.product_name)
    csv_bytes = to_csv_bytes(
        filtered, run_id=run_id,
        provider=llm_cfg.get("provider", ""),
        model=llm_cfg.get("model", ""),
        prompt_version=PROMPT_VERSION,
        industry=spec.industry,
    )

    dl_col, save_col = st.columns(2)
    dl_col.download_button(
        "⬇️ 下载 CSV", data=csv_bytes, file_name=filename,
        mime="text/csv", use_container_width=True,
    )
    if save_col.button("💾 保存到 output/", use_container_width=True):
        out_path = config.ensure_output_dir() / filename
        write_csv(filtered, out_path, run_id=run_id,
                  provider=llm_cfg.get("provider", ""),
                  model=llm_cfg.get("model", ""),
                  prompt_version=PROMPT_VERSION,
                  industry=spec.industry)
        st.success(f"已保存到 {out_path}")


# ══════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    _render_sidebar()

    if st.session_state.page == "settings":
        _settings_page()
        return

    step = st.session_state.step
    if step == 1:
        _step1()
    elif step == 2:
        _step2()
    elif step == 3:
        _step3()


if __name__ == "__main__":
    main()
