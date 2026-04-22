"""

每一次思考循环要做的事情：
    1. 获取 LLM 响应
    2. 状态记录 assistant 响应消息
    3. 判断是否有工具调用
        3.1 没有工具调用，结束本轮思考循环
    4. 需要调用工具，调用工具本返回调用结果
        4.1 调用工具没有返回结果，结束本轮思考循环
    5. 有调用结果，记录 user 消息
    6. 累加思考循环次数
    7. 设置 开启下一轮思考循环的理由
    8. 进入下一轮思考循环

1. 依赖 pip install anthropic python-dotenv
2. env 配置文件
3. 初始化 System Prompt，ToolS
4. 创建 LLM 客户端
5. 定义思考循环状态类
6. 设置一个用户与 LLM 交互的思考循环
    6.1  允许用户退出思考循环
    6.2 记录用户输入的请求并设置 本次请求的思考循环状态
    6.3 启动本次请求的思考循环
    6.4 思考循环结束后打印返回消息

"""
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv


TOOL_RSULT_LIMIT = 3
RECENT_FILE_LIMIT = 5
CONTEXT_LIMIT = 50000
PREVIEW_CHARS = 2000
PERSIST_THRESHOLD = 30000
WORKDIR = Path.cwd()
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
TOOL_RESULTS_DIR = WORKDIR / ".task_outputs" / "tool-results"

load_dotenv(override=True)
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]
SUBAGENT_SYSTEM = f"You are a coding subagent at {WORKDIR}. Complete the given task, then summarize your findings."


@dataclass
class CompactState:
    has_compacted: bool = False
    # 当前总结的摘要
    last_summary: str = ""
    # 最近操作的文件
    recent_files: list[str] = field(default=list)

@dataclass
class SkillManifest:
    name: str
    description: str
    path: Path


@dataclass
class SkillDocument:
    # 技能元数据
    manifest: SkillManifest
    # 技能正文
    body: str


"""

这个版本的 skill 比较松散，LLM 容易走神。工程优化方向： 只要加载 Skill 就必须强制生成 Todo

实现了下面3个步骤，保证 LLM 可以像调用工具一样加载 skill

1. 技能注册(_registry_skill)：建立“图书馆”索引
    程序启动时，SkillRegistry 会执行一次“全城搜索”。
    代码动作：_registry_skill() 扫描目录，_parse_frontmatter() 拆分元数据和正文。
    最终结果：在内存中形成一个 self.documents 字典。
    深度理解：这一步是离线完成的。它保证了当 AI 询问时，系统能立即知道“有没有这个技能”以及“它在哪里”。

2. 元数据描述(descript_skill)：给 LLM 看“书目清单”
    你不需要把书的内容全念给 AI 听，只需要给它一张导览表。
    代码动作：在 SYSTEM 提示词中插入了 {SKILL_REGISTRY.descript_skill()}。
    最终结果：AI 的初始记忆里只有类似 - ffmpeg: 视频处理专家 这样的短句。
    深度理解：这是为了节省 Token 并减少干扰。如果 AI 只是在写一个简单的 Python 脚本，它不需要知道 FFmpeg 的 50 个复杂命令参数。

3. 按需加载(load_skill)：实现“查阅手册”的动作
    这是最关键的闭环。
    代码动作：定义了 load_skill 工具，并在 TOOL_HANDLERS 中关联了 load_skill 函数。
    最终结果：当 AI 发现自己“知识不足”时，它会主动说：{"name": "load_skill", "arguments": {"name": "ffmpeg"}}。
    深度理解：加载后的正文被包裹在 <skill> 标签中回传。这在对话历史中产生了一个强烈的上下文信号，告诉 AI：“现在你已经学会了这个专业技能，请开始表演。”
"""


class SkillRegistry:
    def __init__(self, skill_dir: Path):
        self.skills_dir = skill_dir
        self.documents: dict[str, SkillDocument] = {}
        self._registry_skill()

    # 加载 skills 文件夹下 所有技能
    def _registry_skill(self) -> None:
        # 判空
        if not self.skills_dir.exists():
            return

        # 在指定的文件夹下递归查找所有 SKILL.md 文件
        # rglob() 代表 Recursive Glob（递归全局搜索）。
        # 搜索逻辑：它不仅在 skills/ 根目录下找，还会钻进每一个子文件夹（如 skills/ffmpeg/、skills/python/ 等）去寻找名为 SKILL.md 的文件。
        # 返回值：它返回的是一个 生成器（Generator），里面装满了 pathlib.Path 对象。
        # sorted() 会按照文件路径的字母顺序对所有发现的 Path 对象进行排列，确保系统行为是确定性的
        for path in sorted(self.skills_dir.rglob("SKILL.md")):
            meta, body = self._parse_frontmatter(path.read_text())
            name = meta.get("name", "")
            description = meta.get("description", "")
            manifest = SkillManifest(name=name, description=description, path=path)
            self.documents[name] = SkillDocument(manifest=manifest, body=body)

    # 解析 SKILL.md， 返回 SKILL.md 的元数据 和 操作流程正文
    def _parse_frontmatter(self, md_text: str) -> tuple[dict, str]:
        # 匹配整个 SKILL.md 文件的文本内容
        # (.*?) 匹配到的技能元数据
        # (.*) 匹配到的技能正文
        # re.DOTALL 表示篇匹配换行符
        match = re.match("^---\n(.*?)\n---\n(.*)", md_text, re.DOTALL)
        if not match:
            return {}, md_text

        meta = {}
        for line in match.group(1).strip().split("\n"):
            if ":" not in line:
                continue
            # “只切第一刀，剩下的部分不管里面有多少个冒号，都保留在一起。”
            key, value = line.strip().split(":", 1)
            meta[key.strip()] = value.strip()
        return meta, match.group(2).strip()

    # 给 system prompt 描述技能
    def descript_skill(self) -> str:
        if not self.documents:
            return "(no skills available)"

        skills = []
        for name, document in sorted(self.documents.items()):
            skills.append(f"- {name}: {document.manifest.description}")
        return "\n".join(skills)

    # 按需加载技能
    def load_skill(self, skill_name: str) -> str:
        if skill_name in self.documents:
            document = self.documents[skill_name]
            return f"""<skill name="{document.manifest.name}">{document.body}</skill>"""
        else:
            know = "".join(sorted(self.documents)) or "(none)"
            return f"Unknow skill '{skill_name}'. Available skills: {know}"


SKILL_REGISTRY = SkillRegistry(WORKDIR / "skills")
SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use load_skill when a task needs specialized instructions before you act.
Skills available:
{SKILL_REGISTRY.descript_skill()}
"""


@dataclass
class PlanItem:
    # 当前任务
    content: str
    # 当前任务状态
    status: str
    # 当前任务执行的具体内容
    active_form: str = ""


@dataclass
class PlanState:
    # 计划中的所有任务
    # field(default_factory=list) python 专用创建列表或字典的方法，直接用[] 会导致不同实例共享同一个列表
    plan_items: list[PlanItem] = field(default_factory=list)
    # 距离上一次更新 TODO 后多少轮思考循环没更新了
    round_since_update: int = 0


@dataclass
class TodoManager:

    def __init__(self):
        self.state = PlanState()

    # 全量更新 TODO 列表
    # 先做校验，然后规范化之后再装入
    # LLM 传过来的是一个 dict, 而不是 PlanItem，为了让 LLM 知道自己传的参数，先使用 dict ,
    # 未来可以使用 Dacite 或 Pydantic 等工具直接将dict 转换为 PlanItem， 类似与 Java 中的 Jackson

    def update(self, plan_items: list[dict]) -> str:
        # 任务不能超过 12 个
        if len(plan_items) > 12:
            raise ValueError("Keep the session plan short (max 12 items)")

        process_count = 0
        normalized = []
        # 校验每个任务中的元素
        # enumerate(items)：它像是一个工厂，把列表里的每个元素包装成一个元组（Tuple），长这样：(0, {"content": "A"}), (1, {"content": "B"})
        for index, raw_item in enumerate(plan_items):
            content = str(raw_item.get("content", "")).strip()
            status = str(raw_item.get("status", "")).lower()
            active_form = str(raw_item.get("active_form", "")).strip()
            # 校验任务目标是否存在
            if not content:
                raise ValueError(f"Plan item {index}: content required")

            # 校验 任务状态 是否是枚举值
            # ｛｝ 表示集合，() 表示元组，集合的查找算法复杂度是 O(1)，元组的算法复杂度是O(N)
            if status not in {"pending", "in_progress", "completed"}:
                raise ValueError(f"Plan item {index}: invalid status '{status}'")

            if status == "in_progress":
                process_count += 1

            # 正在进行中 的任务只能是一个
            if process_count > 1:
                raise ValueError("Only one plan item can be in_progress")

            # 通过校验就添加
            item = PlanItem(content=content, status=status, active_form=active_form)
            normalized.append(item)

        self.state.plan_items = normalized
        self.state.round_since_update = 0
        return self.render()

    # 更新之后 返回 更新的 TODO 列表状态字符串
    def render(self) -> str:
        if not self.state.plan_items:
            return "No session plan yet."

        lines = []
        # 统计 所有已完成的 任务
        completed = sum(1 for item in self.state.plan_items if item.status == "completed")
        lines.append(f"({completed} / {len(self.state.plan_items)} completed)")
        # 渲染每个任务的细节
        for item in self.state.plan_items:
            marker = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[√]",
            }[item.status]
            line = f"{marker} {item.content}"
            # 正在处理的任务需要展示正在执行的内容
            if item.status == "in_progress" and item.active_form:
                line += f": {item.active_form}"
            lines.append(line)

        return '\n'.join(lines)

    def note_round_without_update(self):
        self.state.round_since_update += 1

    def reminder(self) -> str | None:
        if len(self.state.plan_items) == 0 or self.state.round_since_update < 3:
            return None
        else:
            return '<reminder>Refresh your current plan before continuing.</reminder>'


TODO = TodoManager()


def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str, tool_use_id: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        output = (r.stdout + r.stderr).strip() or "(no output)"
        return persist_large_output(output, tool_use_id)
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"

# 维护一个始终包含最新、且不重复的 5 个文件的“快捷清单”
def record_recent_file(path, state):
    # python 中 remove(path) 方法必须确保 path 在 state.recent_files 里面
    if path in state.recent_files:
        state.recent_files.remove(path)
    state.recent_files.append(path)
    if len(state.recent_files) > RECENT_FILE_LIMIT:
        state.recent_files[:] = state.recent_files[-RECENT_FILE_LIMIT:]

# def atomic_persist(path: Path, content: str):
#     # 1. 在同一个文件系统分区创建一个临时文件
#     parent = path.parent
#     with tempfile.NamedTemporaryFile('w', dir=parent, delete=False) as tf:
#         temp_name = tf.name
#         tf.write(content)
#         # 强制将数据刷入磁盘硬件
#         tf.flush()
#         os.fsync(tf.fileno())
#
#     # 2. 原子性地重命名（这是一个文件系统级的原子操作）
#     # 在 Linux/Unix 上，os.replace 是原子的。如果目标已存在，它会被瞬间替换。
#     os.replace(temp_name, path)

def persist_large_output(output: str, tool_use_id: str) -> str:
    if len(output) < PERSIST_THRESHOLD:
        return output

    TOOL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stored_path = TOOL_RESULTS_DIR / f"{tool_use_id}.txt"

    # 这个代码仅作教学演示实际参考 atomic_persist
    if not stored_path.exists():
        stored_path.write_text(output)
    rel_path = stored_path.relative_to(WORKDIR)
    preview = output[:PREVIEW_CHARS]

    return (
        "<persisted-output>\n"
        f"Full output saved to: {rel_path}\n"
        "Preview:\n"
        f"{preview}\n"
        "</persisted-output>"
    )



def run_read(path: str, tool_use_id: str, state: CompactState, limit: int = None) -> str:
    try:
        record_recent_file(path, state)
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        output = "\n".join(lines)
        return persist_large_output(output, tool_use_id)
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
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


def run_subagent(prompt: str) -> str:
    subagent_messages = []
    subagent_messages.append({"role": "user", "content": prompt})
    # 只允许执行 30 轮
    for _ in range(30):
        response = client.messages.create(
            model=MODEL,
            system=SUBAGENT_SYSTEM,
            messages=normalize_messages(subagent_messages),
            tools=CHILD_TOOLS,
            max_tokens=8000,
        )
        subagent_messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            break
        subagent_results = []
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                print(f"[subagent] parameter> {block.name}: {block.input}")
                try:
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                except Exception as exc:
                    output = f"Error: {exc}"
                print(f"[subagent] result> {block.name}: {str(output)[:200]}")
                subagent_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    # anthropic 规定 content 必须是 字符串
                    "content": str(output)
                })
        subagent_messages.append({"role": "user", "content": subagent_results})
    # 只取最后的结果信息
    # 在 Python 中，for 循环和 if 语句不会开启新的作用域。
    # Python 里，变量的作用域通常只有两种：全局（Global）和函数级（Function）。
    # or 是最后兜底的作用，解决 > task: Error: 'ThinkingBlock' object has no attribute 'text'
    return "".join(
        block.text for block in response.content if hasattr(block, "text")) or "(subagent completed but no summary)"


# -- Concurrency safety classification --
# Read-only tools can safely run in parallel; mutating tools must be serialized.
CONCURRENCY_SAFE = {"read_file"}
CONCURRENCY_UNSAFE = {"write_file", "edit_file"}
# -- The dispatch map: {tool_name: handler} --
TOOL_HANDLERS = {
    "todo": lambda **kw: TODO.update(kw["plan_items"]),
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "load_skill": lambda **kw: SKILL_REGISTRY.load_skill(kw["skill_name"])
}
CHILD_TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
                      "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                      "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.", "input_schema": {"type": "object",
                                                                                         "properties": {
                                                                                             "path": {"type": "string"},
                                                                                             "old_text": {
                                                                                                 "type": "string"},
                                                                                             "new_text": {
                                                                                                 "type": "string"}},
                                                                                         "required": ["path",
                                                                                                      "old_text",
                                                                                                      "new_text"]}},
    {
        "name": "load_skill",
        "description": "Load the full body of a named skill into the current context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "skill_name": {"type": "string"},
            },
            "required": ["skill_name"]
        }
    }
]

PARENT_TOOLS = CHILD_TOOLS + [
    {
        "name": "task",
        "description": "Spawn a subagent with fresh context. It shares the filesystem but not conversation history.",
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "description": {"type": "string", "description": "Short description of the task"},
            }
        },
        "required": ["prompt"],
    },
    {
        "name": "todo",
        "description": "write the current session plan for multi-step work. ",
        "input_schema": {
            "type": "object",
            "properties": {
                "plan_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "the current step goal"
                            },
                            "status": {
                                "type": "string",
                                # 只能用 元组才能序列化成 JSON，不能用集合，否则： Object of type set is not JSON serializable
                                "enum": ["pending", "in_progress", "completed"]
                            },
                            "active_form": {
                                "type": "string",
                                "description": "Optional present-continuous label."
                            }
                        },
                        "required": ["content", "status"]
                    }
                }
            },
            "required": ["plan_items"]
        }
    },
]


def normalize_messages(messages: list) -> list:
    """Clean up messages before sending to the API.
    Three jobs:
    1. Strip internal metadata fields the API doesn't understand
    2. Ensure every tool_use has a matching tool_result (insert placeholder if missing)
    3. Merge consecutive same-role messages (API requires strict alternation)
    """
    cleaned = []
    for msg in messages:
        clean = {"role": msg["role"]}
        if isinstance(msg.get("content"), str):
            clean["content"] = msg["content"]
        elif isinstance(msg.get("content"), list):
            clean["content"] = [
                {k: v for k, v in block.items()
                 if not k.startswith("_")}
                for block in msg["content"]
                if isinstance(block, dict)
            ]
        else:
            clean["content"] = msg.get("content", "")
        cleaned.append(clean)
    # Collect existing tool_result IDs
    existing_results = set()
    for msg in cleaned:
        if isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    existing_results.add(block.get("tool_use_id"))
    # Find orphaned tool_use blocks and insert placeholder results
    for msg in cleaned:
        if msg["role"] != "assistant" or not isinstance(msg.get("content"), list):
            continue
        for block in msg["content"]:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and block.get("id") not in existing_results:
                cleaned.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": block["id"],
                     "content": "(cancelled)"}
                ]})
    # Merge consecutive same-role messages
    if not cleaned:
        return cleaned
    merged = [cleaned[0]]
    for msg in cleaned[1:]:
        if msg["role"] == merged[-1]["role"]:
            prev = merged[-1]
            prev_c = prev["content"] if isinstance(prev["content"], list) \
                else [{"type": "text", "text": str(prev["content"])}]
            curr_c = msg["content"] if isinstance(msg["content"], list) \
                else [{"type": "text", "text": str(msg["content"])}]
            prev["content"] = prev_c + curr_c
        else:
            merged.append(msg)
    return merged


# 给所有 工具执行结果 建立索引
def build_result_index(messages: list) -> list[tuple[int, int, dict]]:
    result_index = []
    for msg_idx, message in enumerate(messages):
        if message.get("role") != "user" or not isinstance(message.get("content"), list):
            return result_index

        for block_idx,block in enumerate(message.get("content")):
            if isinstance(block, dict) and block.get('type') == 'tool_result':
                result_index.append((msg_idx, block_idx, block))

    return result_index


def compact_tool_result(messages: list) -> list:
    # 先建立索引
    tool_result_index = build_result_index(messages)

    # 如果 工具使用 没有超出限制，不再压缩
    if len(tool_result_index) <= TOOL_RSULT_LIMIT:
        return messages

    # 超出限制，保留最近使用的工具结果，其余全部打标签
    for msg_idx, block_idx, block in tool_result_index[:-TOOL_RSULT_LIMIT]:
        content = block.get("content", "")
        if isinstance(content, str) and  len(content) > 120:
            block["content"] = "[Earlier tool result compacted. Re-run the tool if you need full detail.]"
    return messages

# 将 Agent 对话记录持久化到硬盘，并采用了一种非常适合日志记录的格式：JSONL (JSON Lines)
def write_transcript(messages: list) -> Path:
    #　确保目录存在
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    #  生成文件名
    path = TRANSCRIPT_DIR / f"transciption_{int(time.time())}.jsonl"
    # 写入文件
    with path.open("w") as handle:
        handle.write(json.dumps(messages, default=str) + "\n")
    return path


def summarize_history(messages: list) -> str:

    # 只截取没超出限制部分 消息
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

    """
    Context Window（上下文窗口） 是模型的“总肺活量”，而 max_tokens 是你为模型单次“呼气”设定的“长度上限”
    
    Input Tokens (Prompt) + Output Tokens (Completion) <= Context Window
    
    设置一个较小的 max_tokens，本质上是在逼迫模型：“别啰嗦，只把最重要的东西告诉我”
    """
    response = client.messages.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
    )

    return response.content[0].text.strip()


def compact_history(messages: list, state: CompactState, focus: str | None = None) -> list:
    # 转写历史消息到磁盘
    transcript_path = write_transcript(messages)
    print(f"[transcript saved: {transcript_path}]")
    # 生成摘要信息
    summary_parts = [summarize_history(messages)]
    if focus:
        summary_parts.append( f"Focus to preserve next: {focus}")
    if state.recent_files:
        lines = "\n".join( f"- {file}" for file in state.recent_files)
        summary_parts.append(f"Recent files to reopen if needed:\n{lines}")

    # filter(None) 确保空字符串不会导致多余换行
    state.last_summary = "\n\n".join(filter(None, summary_parts))
    # 记录为用户信息
    return [{
        "role": "user",
        "content": (
            "This conversation was compacted so the agent can continue working.\n\n"
            f"{state.last_summary}"
        )
    }]

def agent_loop(messages: list, state: CompactState) -> None:
    while True:
        messages[:] = normalize_messages(messages)
        # 例行压缩
        messages[:] = compact_tool_result(messages)

        if len(str(messages)) > CONTEXT_LIMIT:
            print("[auto compact]")
            messages[:] = compact_history(messages, state)

        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=PARENT_TOOLS,
            max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return
        results = []
        # 标记是否更新了 TODO 列表
        update_todo_flag = False

        compact_flag = False
        compact_goal = None
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                print(f"> {block.name} parameter: {block.input}")
                try:
                    if block.name == "task":
                        desc = str(block.input.get("description", "subtask"))
                        prompt = str(block.input.get("prompt", ""))
                        print("========== ========== subagent 开始 ========== ==========")
                        print(f"> task ({desc}): {prompt[:80]}")
                        output = run_subagent(prompt)
                        print("========== ========== subagent 结束 ========== ==========")
                    else:
                        output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                except Exception as exc:
                    output = f"Error: {exc}"
                print(f"> {block.name} result: {str(output)[:200]}")
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    # anthropic 规定 content 必须是 字符串
                    "content": str(output)
                })
                if block.name == "todo":
                    update_todo_flag = True

                if block.name == "compact":
                    compact_flag = True
                    compact_goal = (block.get("input") or {}).get("goal")
        # 没有更新时 要提醒 LLM 要更新
        if not update_todo_flag:
            # 　先更新　上一次更新后没有更新　TODO 的次数
            TODO.note_round_without_update()
            # 获取提醒
            reminder = TODO.reminder()
            if reminder:
                results.insert(0, {"type": "text", "text": reminder})
        else:
            TODO.state.round_since_update = 0

        messages.append({"role": "user", "content": results})

        # LLM 主动压缩
        if compact_flag:
            print("[manual compact]")
            messages[:] = compact_history(messages, state, compact_goal)


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms05 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print()
                    print('========== ========== 执行结果 ========== ==========')
                    print(block.text)
        print()
