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
import os
import subprocess
from dataclasses import dataclass, field
from email.policy import default
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]
SYSTEM = f"""
You are a coding agent at {WORKDIR}.
Use the todo tool for multi-step work.
Keep exactly one step in_progress when a task has multiple steps.
Refresh the plan as work advances. Prefer tools over prose.
"""


@dataclass
class Task:
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
    tasks: list[Task] = field(default_factory=list)
    # 距离上一次更新 TODO 后多少轮思考循环没更新了
    round_since_update: int = 0


@dataclass
class TodoManager:

    def __init__(self):
        self.state = PlanState()

    # 全量更新 TODO 列表
    # 先做校验，然后规范化之后再装入
    # LLM 传过来的是一个 dict, 而不是 Task，为了让 LLM 知道自己传的参数，先使用 dict ,
    # 未来可以使用 Dacite 或 Pydantic 等工具直接将dict 转换为 Task， 类似与 Java 中的 Jackson

    def update(self, tasks: list[dict]) -> str:
        # 任务不能超过 12 个
        if len(tasks) > 12:
            raise ValueError("Keep the session plan short (max 12 items)")

        process_count = 0
        normalized = []
        # 校验每个任务中的元素
        # enumerate(items)：它像是一个工厂，把列表里的每个元素包装成一个元组（Tuple），长这样：(0, {"content": "A"}), (1, {"content": "B"})
        for index, raw_task in enumerate(tasks):
            content = str(raw_task.get("content", "")).strip()
            status = str(raw_task.get("status", "")).lower()
            active_form = str(raw_task.get("active_form", "")).strip()
            # 校验任务目标是否存在
            if not content:
                raise ValueError(f"task {index}: content required")

            # 校验 任务状态 是否是枚举值
            # ｛｝ 表示集合，() 表示元组，集合的查找算法复杂度是 O(1)，元组的算法复杂度是O(N)
            if status not in {"pending", "in_progress", "completed"}:
                raise ValueError(f"task {index}: invalid status '{status}'")

            if status == "in_progress":
                process_count += 1

            # 正在进行中 的任务只能是一个
            if process_count > 1:
                raise ValueError("Only one plan task can be in_progress")

            # 通过校验就添加
            task = Task(content=content, status=status, active_form=active_form)
            normalized.append(task)

        self.state.tasks = normalized
        self.state.round_since_update = 0
        return self.render()

    # 更新之后 返回 更新的 TODO 列表状态字符串
    def render(self) -> str:
        if not self.state.tasks:
            return "No session plan yet."

        lines = []
        # 统计 所有已完成的 任务
        completed = sum(1 for task in self.state.tasks if task.status == "completed")
        lines.append(f"({completed} / {len(self.state.tasks)} completed)")
        # 渲染每个任务的细节
        for task in self.state.tasks:
            marker = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[√]",
            }[task.status]
            line = f"{marker} {task.content}"
            # 正在处理的任务需要展示正在执行的内容
            if task.status == "in_progress" and task.active_form:
                line += f": {task.active_form}"
            lines.append(line)

        return '\n'.join(lines)

    def note_round_without_update(self):
        self.state.round_since_update += 1

    def reminder(self) -> str | None:
        if len(self.state.tasks) == 0 or self.state.round_since_update < 3:
            return None
        else:
            return '<reminder>Refresh your current plan before continuing.</reminder>'


TODO = TodoManager()


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
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:50000]
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


# -- Concurrency safety classification --
# Read-only tools can safely run in parallel; mutating tools must be serialized.
CONCURRENCY_SAFE = {"read_file"}
CONCURRENCY_UNSAFE = {"write_file", "edit_file"}
# -- The dispatch map: {tool_name: handler} --
TOOL_HANDLERS = {
    "todo": lambda **kw: TODO.update(kw["tasks"]),
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}
TOOLS = [
    {
        "name": "todo",
        "description": "write the current session plan for multi-step work. ",
        "input_schema": {
            "type": "object",
            "properties": {
                "tasks": {
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
                                "enum": {"pending", "in_process", "completed"}
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
            "required": ["tasks"]
        }
    },

    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
                      "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                      "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"},
                                                       "new_text": {"type": "string"}},
                      "required": ["path", "old_text", "new_text"]}},
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


def agent_loop(messages: list):
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM,
            messages=normalize_messages(messages),
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return
        results = []
        # 标记是否更新了 TODO 列表
        update_todo_flag = False
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                print(f"> {block.name}: {block.input}")
                try:
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                except Exception as exc:
                    output = f"Error: {exc}"
                print(f"> {block.name}: {str(output)[:200]}")
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    # anthropic 规定 content 必须是 字符串
                    "content": str(output)
                })
                if block.name == "todo":
                    update_todo_flag = True
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

        messages.append({
            "role": "user",
            "content": results
        })


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms02 >> \033[0m")
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
                    print(block.text)
        print()
