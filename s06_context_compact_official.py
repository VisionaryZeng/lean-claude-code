#!/usr/bin/env python3
# Harness: compression -- keep the active context small enough to keep working.
"""
s06_context_compact.py - Context Compact
This teaching version keeps the compact model intentionally small:
1. Large tool output is persisted to disk and replaced with a preview marker.
2. Older tool results are micro-compacted into short placeholders.
3. When the whole conversation gets too large, the agent summarizes it and
   continues from that summary.
The goal is not to model every production branch. The goal is to make the
active-context idea explicit and teachable.
"""
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv
load_dotenv(override=True)
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]
SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Keep working step by step, and use compact if the conversation gets too long."
)
CONTEXT_LIMIT = 50000
KEEP_RECENT_TOOL_RESULTS = 3
PERSIST_THRESHOLD = 30000
PREVIEW_CHARS = 2000
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
TOOL_RESULTS_DIR = WORKDIR / ".task_outputs" / "tool-results"
@dataclass
class CompactState:
    has_compacted: bool = False
    last_summary: str = ""
    recent_files: list[str] = field(default_factory=list)
def estimate_context_size(messages: list) -> int:
    return len(str(messages))
# 维护一个始终包含最新、且不重复的 5 个文件的“快捷清单”
def track_recent_file(state: CompactState, path: str) -> None:
    if path in state.recent_files:
        state.recent_files.remove(path)
    state.recent_files.append(path)
    if len(state.recent_files) > 5:
        state.recent_files[:] = state.recent_files[-5:]
def safe_path(path_str: str) -> Path:
    path = (WORKDIR / path_str).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {path_str}")
    return path

# 在使用 bash 或 read 工具时，如果返回的结果很长，直接先持久化到硬盘中
def persist_large_output(tool_use_id: str, output: str) -> str:
    if len(output) <= PERSIST_THRESHOLD:
        return output
    TOOL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stored_path = TOOL_RESULTS_DIR / f"{tool_use_id}.txt"
    if not stored_path.exists():
        stored_path.write_text(output)
    preview = output[:PREVIEW_CHARS]
    rel_path = stored_path.relative_to(WORKDIR)
    return (
        "<persisted-output>\n"
        f"Full output saved to: {rel_path}\n"
        "Preview:\n"
        f"{preview}\n"
        "</persisted-output>"
    )

# 给所有 工具执行结果 建立索引
def collect_tool_result_blocks(messages: list) -> list[tuple[int, int, dict]]:
    blocks = []
    for message_index, message in enumerate(messages):
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, list):
            continue
        for block_index, block in enumerate(content):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                blocks.append((message_index, block_index, block))
    return blocks

# 压缩工具调用信息
def micro_compact(messages: list) -> list:
    tool_results = collect_tool_result_blocks(messages)
    if len(tool_results) <= KEEP_RECENT_TOOL_RESULTS:
        return messages
    for _, _, block in tool_results[:-KEEP_RECENT_TOOL_RESULTS]:
        content = block.get("content", "")
        if not isinstance(content, str) or len(content) <= 120:
            continue
        block["content"] = "[Earlier tool result compacted. Re-run the tool if you need full detail.]"
    return messages

# 将 Agent 宝贵的对话记录持久化到硬盘，并采用了一种非常适合日志记录的格式：JSONL (JSON Lines)
def write_transcript(messages: list) -> Path:
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with path.open("w") as handle:
        for message in messages:
            # json.dumps(message, ...)：序列化成 JSON，default=str 调用对象的 str() 方法把它转成字符串再序列化
            handle.write(json.dumps(message, default=str) + "\n")
    return path

# (三明治截断)：兼顾“目标”与“现状”，是性价比最高的上下文管理方式
# def compact_messages_object(messages: list, keep_head=2, keep_tail=8):
    if len(messages) <= (keep_head + keep_tail):
        return messages

    head = messages[:keep_head]
    tail = messages[-keep_tail:]

    # 插入一条占位消息，告知模型中间被省略了
    gap_info = [{"role": "user", "content": f"... [此处省略了 {len(messages) - keep_head - keep_tail} 条中间对话信息] ..."}]

    return head + gap_info + tail

def summarize_history(messages: list) -> str:
    # 这样截断前面80000会导致导致 LLM 理解 历史会话产生困扰
    # 后续优化 compact_messages_object 按消息进行 （三明治截断）：兼顾“目标”与“现状”，是性价比最高的上下文管理方式，最好配合 State Manager 实现对中间关键的变量的保留
    conversation = json.dumps(messages, default=str)[:80000]
    prompt = (
        "Summarize this coding-agent conversation so work can continue.\n"
        "Preserve:\n"
        "1. The current goal\n"
        "2. Important findings and decisions\n"
        "3. Files read or changed\n"
        "4. Remaining work\n"
        "5. User constraints and preferences\n"
        "Be compact but concrete.\n\n"
        f"{conversation}"
    )
    response = client.messages.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
    )
    return response.content[0].text.strip()
#  compact_history 执行完 messages 列表确实会被“清空”并替换为仅包含一条角色为 user 的消息
# focus 参数是压缩时的“导航指标”，当 AI 主动调用 compact 工具时，它可以传入一个 focus（比如：“我正在修复登录页面的 CSS 布局”）。
# 这个参数会被拼接到最终的总结中，强制 AI 在“遗忘”大部分细节时，依然死死记住当前的核心目标
def compact_history(messages: list, state: CompactState, focus: str | None = None) -> list:
    transcript_path = write_transcript(messages)
    print(f"[transcript saved: {transcript_path}]")
    summary = summarize_history(messages)
    if focus:
        summary += f"\n\nFocus to preserve next: {focus}"
    if state.recent_files:
        recent_lines = "\n".join(f"- {path}" for path in state.recent_files)
        summary += f"\n\nRecent files to reopen if needed:\n{recent_lines}"
    state.has_compacted = True
    state.last_summary = summary
    return [{
        "role": "user",
        "content": (
            "This conversation was compacted so the agent can continue working.\n\n"
            f"{summary}"
        ),
    }]
def run_bash(command: str, tool_use_id: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(item in command for item in dangerous):
        return "Error: Dangerous command blocked"
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    output = (result.stdout + result.stderr).strip() or "(no output)"
    return persist_large_output(tool_use_id, output)
def run_read(path: str, tool_use_id: str, state: CompactState, limit: int | None = None) -> str:
    try:
        track_recent_file(state, path)
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        output = "\n".join(lines)
        return persist_large_output(tool_use_id, output)
    except Exception as exc:
        return f"Error: {exc}"
def run_write(path: str, content: str) -> str:
    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as exc:
        return f"Error: {exc}"
def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        file_path = safe_path(path)
        content = file_path.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        file_path.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as exc:
        return f"Error: {exc}"
TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read file contents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace exact text in a file once.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
    {
        "name": "compact",
        "description": "Summarize earlier conversation so work can continue in a smaller context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "focus": {"type": "string"},
            },
        },
    },
]
def extract_text(content) -> str:
    if not isinstance(content, list):
        return ""
    texts = []
    for block in content:
        text = getattr(block, "text", None)
        if text:
            texts.append(text)
    return "\n".join(texts).strip()
def execute_tool(block, state: CompactState) -> str:
    if block.name == "bash":
        return run_bash(block.input["command"], block.id)
    if block.name == "read_file":
        return run_read(block.input["path"], block.id, state, block.input.get("limit"))
    if block.name == "write_file":
        return run_write(block.input["path"], block.input["content"])
    if block.name == "edit_file":
        return run_edit(block.input["path"], block.input["old_text"], block.input["new_text"])

    # 这里不是立即压缩上下文的原因：LLM 可能调用多个 命令，如果 compact 命令在前面，上下文丢失，导致后面的命令无法执行
    # 延迟压缩，等全部 命令执行完后，再执行压缩命令，可以有效压缩本轮 的 所有内容
    if block.name == "compact":
        return "Compacting conversation..."
    return f"Unknown tool: {block.name}"
def agent_loop(messages: list, state: CompactState) -> None:
    while True:

        # 这一处压缩是 工程压缩： 压缩工具调用 和 保证上下文在限制大小内 ，是为了保护 API 调用不崩溃
        # 切片赋值 (messages[:] = ...)： 保留原来的列表对象，但把里面的内容全部清空，然后填入 micro_compact 返回的新内容。
        messages[:] = micro_compact(messages)
        if estimate_context_size(messages) > CONTEXT_LIMIT:
            print("[auto compact]")
            messages[:] = compact_history(messages, state)
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return
        results = []
        manual_compact = False
        compact_focus = None
        for block in response.content:
            if block.type != "tool_use":
                continue
            output = execute_tool(block, state)
            # 这一处压缩是 LLM 整理思路用的，让它能主动清空不必要的干扰
            if block.name == "compact":
                manual_compact = True
                compact_focus = (block.input or {}).get("focus")
            print(f"> {block.name}: {str(output)[:200]}")
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": str(output),
            })
        messages.append({"role": "user", "content": results})
        if manual_compact:
            print("[manual compact]")
            messages[:] = compact_history(messages, state, focus=compact_focus)
if __name__ == "__main__":
    history = []
    compact_state = CompactState()
    while True:
        try:
            query = input("\033[36ms06 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history, compact_state)
        final_text = extract_text(history[-1]["content"])
        if final_text:
            print(final_text)
        print()