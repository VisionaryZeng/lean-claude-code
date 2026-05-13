#!/usr/bin/env python3
# Harness: persistence -- remembering across the session boundary.
"""
s09_memory_system.py - Memory System
This teaching version focuses on one core idea:
some information should survive the current conversation, but not everything
belongs in memory.
Use memory for:
  - user preferences
  - repeated user feedback
  - project facts that are NOT obvious from the current code
  - pointers to external resources
Do NOT use memory for:
  - code structure that can be re-read from the repo
  - temporary task state
  - secrets
Storage layout:
  .memory/
    MEMORY.md
    prefer_tabs.md
    review_style.md
    incident_board.md
Each memory is a small Markdown file with frontmatter.
The agent can save a memory through save_memory(), and the memory index
is rebuilt after each write.
An optional "Dream" pass can later consolidate, deduplicate, and prune
stored memories. It is useful, but it is not the first thing readers need
to understand.
Key insight: "Memory only stores cross-session information that is still
worth recalling later and is not easy to re-derive from the current repo."
"""

import os
import re
import subprocess
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv
load_dotenv(override=True)
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]
MEMORY_DIR = WORKDIR / ".memory"
MEMORY_INDEX = MEMORY_DIR / "MEMORY.md"
MEMORY_TYPES = ("user", "feedback", "project", "reference")
MAX_INDEX_LINES = 200


# 你的项目根目录/
# ├── .memory/                 <-- 核心记忆文件夹
# │   ├── MEMORY.md            <-- 自动生成的记忆索引（只读）
# │   ├── .dream_lock          <-- (可选) 梦境整合进程的 PID 锁文件
# │   ├── prefer_tabs.md       <-- 个体记忆文件 1 (Frontmatter 格式)
# │   ├── video_notes_specs.md <-- 个体记忆文件 2
# │   └── review_style.md      <-- 个体记忆文件 3
# ├── main.py
# └── .env

class MemoryManager:
    """
    Load, build, and save persistent memories across sessions.
    The teaching version keeps memory explicit:
    one Markdown file per memory, plus one compact index file.
    """
    def __init__(self, memory_dir: Path = None):
        self.memory_dir = memory_dir or MEMORY_DIR
        #　先定义 memories，
        self.memories = {}  # name -> {description, type, content}
    # 加载所有的 memory ，涉及大量的 IO 操作，不要在 __init__ 调用，方便出错好排查
    def load_all(self):
        """Load MEMORY.md index and all individual memory files."""
        # 清空 原来的 memories
        self.memories = {}
        if not self.memory_dir.exists():
            return
        # Scan all .md files except MEMORY.md
        # glob 默认是不递归的
        # md_file.name	prefer_tabs.md	完整文件名（带后缀）
        # md_file.stem	prefer_tabs	文件的主干名（去掉后缀）
        # md_file.suffix	.md	文件的后缀名
        # md_file.parent	/home/user/project/.memory	文件所在的目录
        for md_file in sorted(self.memory_dir.glob("*.md")):
            if md_file.name == "MEMORY.md":
                continue
            parsed = self._parse_frontmatter(md_file.read_text())
            if parsed:
                name = parsed.get("name", md_file.stem)
                self.memories[name] = {
                    "description": parsed.get("description", ""),
                    "type": parsed.get("type", "project"),
                    "content": parsed.get("content", ""),
                    "file": md_file.name,
                }
        count = len(self.memories)
        if count > 0:
            print(f"[Memory loaded: {count} memories from {self.memory_dir}]")

    # 全量加载 memory
    def load_memory_prompt(self) -> str:
        """Build a memory section for injection into the system prompt."""
        if not self.memories:
            return ""
        sections = []
        sections.append("# Memories (persistent across sessions)")
        # 空行 语义边界感
        sections.append("")
        # Group by type for readability
        for mem_type in MEMORY_TYPES:
            typed = {k: v for k, v in self.memories.items() if v["type"] == mem_type}
            if not typed:
                continue
            sections.append(f"## [{mem_type}]")
            for name, mem in typed.items():
                sections.append(f"### {name}: {mem['description']}")
                if mem["content"].strip():
                    sections.append(mem["content"].strip())
                # 空行 语义边界感
                sections.append("")
        return "\n".join(sections)

    def save_memory(self, name: str, description: str, mem_type: str, content: str) -> str:
        """
        Save a memory to disk and update the index.
        Returns a status message.
        """
        if mem_type not in MEMORY_TYPES:
            return f"Error: type must be one of {MEMORY_TYPES}"
        # Sanitize name for filename
        # 将文件名中除了[a-zA-Z0-9_-]的特殊字符全部转换为_,^ 在 [] 中表示取反
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", name.lower())
        if not safe_name:
            return "Error: invalid memory name"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        # Write individual memory file with frontmatter
        frontmatter = (
            f"---\n"
            f"name: {name}\n"
            f"description: {description}\n"
            f"type: {mem_type}\n"
            f"---\n"
            f"{content}\n"
        )
        file_name = f"{safe_name}.md"
        file_path = self.memory_dir / file_name
        file_path.write_text(frontmatter)
        # Update in-memory store
        self.memories[name] = {
            "description": description,
            "type": mem_type,
            "content": content,
            "file": file_name,
        }
        # Rebuild MEMORY.md index
        self._rebuild_index()
        return f"Saved memory '{name}' [{mem_type}] to {file_path.relative_to(WORKDIR)}"

    def _rebuild_index(self):
        """Rebuild MEMORY.md from current in-memory state, capped at 200 lines."""
        lines = ["# Memory Index", ""]
        for name, mem in self.memories.items():
            lines.append(f"- {name}: {mem['description']} [{mem['type']}]")
            if len(lines) >= MAX_INDEX_LINES:
                lines.append(f"... (truncated at {MAX_INDEX_LINES} lines)")
                break
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        # 会完全覆盖（Overwrite）掉 MEMORY_INDEX 文件原有的所有内容
        MEMORY_INDEX.write_text("\n".join(lines) + "\n")

    # 解析 memory 的 md 文件
    def _parse_frontmatter(self, text: str) -> dict | None:
        """Parse --- delimited frontmatter + body content."""
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
        if not match:
            return None
        header, body = match.group(1), match.group(2)
        result = {"content": body.strip()}
        for line in header.splitlines():
            if ":" in line:
                # partition(":") 类似 split(":")，但更安全更优雅
                key, _, value = line.partition(":")
                result[key.strip()] = value.strip()
        return result

class DreamConsolidator:
    """
    Auto-consolidation of memories between sessions ("Dream").
    This is an optional later-stage feature. Its job is to prevent the memory
    store from growing into a noisy pile by merging, deduplicating, and
    pruning entries over time.
    """
    COOLDOWN_SECONDS = 86400       # 24 hours between consolidations
    SCAN_THROTTLE_SECONDS = 600    # 10 minutes between scan attempts
    MIN_SESSION_COUNT = 5          # need enough data to consolidate
    LOCK_STALE_SECONDS = 3600      # PID lock considered stale after 1 hour
    PHASES = [
        "Orient: scan MEMORY.md index for structure and categories",
        "Gather: read individual memory files for full content",
        "Consolidate: merge related memories, remove stale entries",
        "Prune: enforce 200-line limit on MEMORY.md index",
    ]
    def __init__(self, memory_dir: Path = None):
        self.memory_dir = memory_dir or MEMORY_DIR
        self.lock_file = self.memory_dir / ".dream_lock"
        self.enabled = True
        self.mode = "default"
        self.last_consolidation_time = 0.0
        self.last_scan_time = 0.0
        self.session_count = 0

    # 顺序,   门禁,                     理由,                                                         成本
    # 1,    Gate 1: Enabled,            总开关，如果是关的，后面什么都不用看。,                           极低 (Bool)
    # 2,    Gate 5: Throttle,           防抖保护。如果 10 分钟内刚查过，直接跳过。它保护了下面所有的逻辑。,   极低 (Math)
    # 3,    Gate 3: Plan Mode,          逻辑状态检查。如果是预览模式，直接退出。,                         极低 (String)
    # 4,    Gate 6: Session Count,      简单的计数器检查。,                                            极低 (Int)
    # 5,    Gate 4: Cooldown,           业务逻辑的大钟表（24 小时）。,                                  低 (Math)
    # 6,    Gate 2: Directory/Files,    第一次触碰磁盘。确定真的有活干。,                                中 (Disk IO)
    # 7,    Gate 7: PID Lock,           最重的操作。读写文件 + 进程信号。,                               高 (Heavy IO)
    def should_consolidate(self) -> tuple[bool, str]:
        """
        Check 7 gates in sequence. All must pass.
        Returns (can_run, reason) where reason explains the first failed gate.
        """
        import time
        now = time.time()
        # Gate 1: enabled flag
        if not self.enabled:
            return False, "Gate 1: consolidation is disabled"
        # Gate 2: memory directory exists and has memory files
        if not self.memory_dir.exists():
            return False, "Gate 2: memory directory does not exist"
        memory_files = list(self.memory_dir.glob("*.md"))
        # Exclude MEMORY.md itself from the count
        memory_files = [f for f in memory_files if f.name != "MEMORY.md"]
        if not memory_files:
            return False, "Gate 2: no memory files found"
        # Gate 3: not in plan mode (only consolidate in active modes)
        if self.mode == "plan":
            # plan 模式意味着只读，不能修改记忆
            return False, "Gate 3: plan mode does not allow consolidation"
        # Gate 4: 24-hour cooldown since last consolidation
        time_since_last = now - self.last_consolidation_time
        if time_since_last < self.COOLDOWN_SECONDS:
            remaining = int(self.COOLDOWN_SECONDS - time_since_last)
            return False, f"Gate 4: cooldown active, {remaining}s remaining"
        # Gate 5: 10-minute throttle since last scan attempt
        time_since_scan = now - self.last_scan_time
        if time_since_scan < self.SCAN_THROTTLE_SECONDS:
            remaining = int(self.SCAN_THROTTLE_SECONDS - time_since_scan)
            return False, f"Gate 5: scan throttle active, {remaining}s remaining"
        # Gate 6: need at least 5 sessions worth of data
        if self.session_count < self.MIN_SESSION_COUNT:
            return False, f"Gate 6: only {self.session_count} sessions, need {self.MIN_SESSION_COUNT}"
        # Gate 7: no active lock file (check PID staleness)
        if not self._acquire_lock():
            return False, "Gate 7: lock held by another process"
        return True, "All 7 gates passed"
    def consolidate(self) -> list[str]:
        """
        Run the 4-phase consolidation process.
        The teaching version returns phase descriptions to make the flow
        visible without requiring an extra LLM pass here.
        """
        import time
        can_run, reason = self.should_consolidate()
        if not can_run:
            print(f"[Dream] Cannot consolidate: {reason}")
            return []
        print("[Dream] Starting consolidation...")
        self.last_scan_time = time.time()
        completed_phases = []
        # 1 表示 请从 1 开始计数
        for i, phase in enumerate(self.PHASES, 1):
            print(f"[Dream] Phase {i}/4: {phase}")
            completed_phases.append(phase)
        self.last_consolidation_time = time.time()
        self._release_lock()
        print(f"[Dream] Consolidation complete: {len(completed_phases)} phases executed")
        return completed_phases

    # 安全地获得锁
    def _acquire_lock(self) -> bool:
        """
        Acquire a PID-based lock file. Returns False if locked by another
        live process. Stale locks (older than LOCK_STALE_SECONDS) are removed.
        """
        import time
        if self.lock_file.exists():
            try:
                lock_data = self.lock_file.read_text().strip()
                pid_str, timestamp_str = lock_data.split(":", 1)
                pid = int(pid_str)
                lock_time = float(timestamp_str)
                # Check if lock is stale
                if (time.time() - lock_time) > self.LOCK_STALE_SECONDS:
                    print(f"[Dream] Removing stale lock from PID {pid}")
                    # 删除文件，等同于释放锁
                    self.lock_file.unlink()
                else:
                    # Check if owning process is still alive
                    try:
                        # 侦察进程的存活状态，并不是要杀掉进程
                        # 执行情况,     结果,                             含义
                        # 进程存在,     执行成功（不报错）,                  进程依然活跃。
                        # 进程不存在,   抛出 OSError (Errno 3),            进程已经结束，这是一个“死锁”。
                        # 无权限,      抛出 PermissionError (Errno 1),    进程存在，但你（当前用户）管不了它。
                        os.kill(pid, 0)
                        return False  # process alive, lock is valid
                    except OSError:
                        print(f"[Dream] Removing lock from dead PID {pid}")
                        self.lock_file.unlink()
            # ValueError 处理 lock_file 中错误的数值
            except (ValueError, OSError):
                # Corrupted lock file, remove it
                # 不让“脏数据”或“残留状态”变成系统的“永久路障”
                self.lock_file.unlink(missing_ok=True)
        # Write new lock
        try:
            self.memory_dir.mkdir(parents=True, exist_ok=True)
            self.lock_file.write_text(f"{os.getpid()}:{time.time()}")
            return True
        except OSError:
            return False

    # 释放锁
    def _release_lock(self):
        """Release the lock file if we own it."""
        try:
            if self.lock_file.exists():
                lock_data = self.lock_file.read_text().strip()
                pid_str = lock_data.split(":")[0]
                if int(pid_str) == os.getpid():
                    self.lock_file.unlink()
        except (ValueError, OSError):
            pass

    # 做梦 整合记忆文件
    # 1.在保存记忆时
    #   按类进行整理
    #   如果发现有矛盾的记忆，按 时间优先（最新为准）> 重要性优先 > 频率/一致性优先 > 场景区分 > 保留矛盾 + 标注 优先级处理矛盾
    #       记忆权重量化：
    #               维度,                         权重,     如何给新记忆打分（0~1）,                                      实现方式建议
    #               strength（语气强度）,          0.35,     是否用了“必须”“强烈”“绝对”“特别不喜欢”等强烈词语,                规则 + LLM
    #               criticality（业务关键度）,     0.25,      是否涉及项目核心功能、交付物、成本、合规、安全、用户体验关键点,      LLM 判断
    #               frequency（重复度）,          0.15,      该主题在近期是否已被多次提到,                                  向量检索统计
    #               emotion（情感强度）,          0.10,      是否表达了明显不满、兴奋、失望、赞赏等情绪,                       LLM + 规则
    #               recency（时效性）,            0.10,      新记忆默认给 0.95~1.0（越新越高）,                             时间计算
    #               category（类别权重）,         0.05,      预设不同类别的基准分（项目要求 > 用户反馈 > 一般偏好）,            固定映射表

    #   实际整合记忆时，要提示 LLM 当前任务要求
    # 2.额外再定时整合所有的记忆，进一步优化记忆
    def consolidation(self) -> str:
        # 获取当前最新的记忆文件夹

        # 在最新的记忆文件夹下获取所有写完的记忆文件，忽略还没写完的.tmp 文件

        # 创建 .dream 文件夹
        # 拼装成完整的 prompt ，提醒 LLM 按 记忆类型分类记忆
        # 调用 LLM 整合 记忆
        # 生成新的记忆文件 输出到 .dream 文件夹中



        pass

# -- Tool implementations --
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path
def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
def run_read(path: str, limit: int = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"
def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"
def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"
# Global memory manager
memory_mgr = MemoryManager()
def run_save_memory(name: str, description: str, mem_type: str, content: str) -> str:
    return memory_mgr.save_memory(name, description, mem_type, content)
TOOL_HANDLERS = {
    "bash":         lambda **kw: run_bash(kw["command"]),
    "read_file":    lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file":   lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":    lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "save_memory":  lambda **kw: run_save_memory(kw["name"], kw["description"], kw["type"], kw["content"]),
}
TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "save_memory", "description": "Save a persistent memory that survives across sessions.",
     "input_schema": {"type": "object", "properties": {
         "name": {"type": "string", "description": "Short identifier (e.g. prefer_tabs, db_schema)"},
         "description": {"type": "string", "description": "One-line summary of what this memory captures"},
         "type": {"type": "string", "enum": ["user", "feedback", "project", "reference"],
                  "description": "user=preferences, feedback=corrections, project=non-obvious project conventions or decision reasons, reference=external resource pointers"},
         "content": {"type": "string", "description": "Full memory content (multi-line OK)"},
     }, "required": ["name", "description", "type", "content"]}},
]
MEMORY_GUIDANCE = """
When to save memories:
- User states a preference ("I like tabs", "always use pytest") -> type: user
- User corrects you ("don't do X", "that was wrong because...") -> type: feedback
- You learn a project fact that is not easy to infer from current code alone
  (for example: a rule exists because of compliance, or a legacy module must
  stay untouched for business reasons) -> type: project
- You learn where an external resource lives (ticket board, dashboard, docs URL) -> type: reference
When NOT to save:
- Anything easily derivable from code (function signatures, file structure, directory layout)
- Temporary task state (current branch, open PR numbers, current TODOs)
- Secrets or credentials (API keys, passwords)
"""


def build_system_prompt() -> str:
    """Assemble system prompt with memory content included."""
    parts = [f"You are a coding agent at {WORKDIR}. Use tools to solve tasks."]
    # Inject memory content if available
    memory_section = memory_mgr.load_memory_prompt()
    if memory_section:
        parts.append(memory_section)
    parts.append(MEMORY_GUIDANCE)
    return "\n\n".join(parts)


def agent_loop(messages: list):
    """
    Agent loop with memory-aware system prompt.
    The system prompt is rebuilt each call so newly saved memories
    are visible in the next LLM turn within the same session.
    """
    while True:
        system = build_system_prompt()
        response = client.messages.create(
            model=MODEL, system=system, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            handler = TOOL_HANDLERS.get(block.name)
            try:
                output = handler(**(block.input or {})) if handler else f"Unknown: {block.name}"
            except Exception as e:
                output = f"Error: {e}"
            print(f"> {block.name}: {str(output)[:200]}")
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": str(output),
            })
        messages.append({"role": "user", "content": results})
if __name__ == "__main__":
    # Load existing memories at session start
    memory_mgr.load_all()
    mem_count = len(memory_mgr.memories)
    if mem_count:
        print(f"[{mem_count} memories loaded into context]")
    else:
        print("[No existing memories. The agent can create them with save_memory.]")
    history = []
    while True:
        try:
            query = input("\033[36ms09 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        # /memories command to list current memories
        if query.strip() == "/memories":
            if memory_mgr.memories:
                for name, mem in memory_mgr.memories.items():
                    print(f"  [{mem['type']}] {name}: {mem['description']}")
            else:
                print("  (no memories)")
            continue
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()