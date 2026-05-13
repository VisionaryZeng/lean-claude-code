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
from datetime import datetime, timezone, timedelta
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from fnmatch import fnmatch
from os.path import join
from pathlib import Path, PurePath

from anthropic import Anthropic
from anthropic.types import TextBlock
from dotenv import load_dotenv


TOOL_RSULT_LIMIT = 3
RECENT_FILE_LIMIT = 5
CONTEXT_LIMIT = 50000
PREVIEW_CHARS = 2000
PERSIST_THRESHOLD = 30000
WORKDIR = Path.cwd()
print(f"WORKDIR: {WORKDIR}")
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
TOOL_RESULTS_DIR = WORKDIR / ".task_outputs" / "tool-results"
READ_ONLY_TOOLS = {"read_file", "bash_readonly"}
WRITE_TOOLS = {"write_file", "edit_file"}
TRUST_MARKER = WORKDIR / ".claude" / ".claude_trusted"
HOOK_TIMEOUT = 30

MEMORY_DIR = WORKDIR / ".memory"
MEMORY_INDEX_FILE = MEMORY_DIR / "MEMORY.md"
MAX_INDEX_LINES = 300
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
    recent_files: list[str] = field(default_factory=list)


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

class MODE(Enum):
    DEFAULT = "default"
    PLAN = "plan"
    AUTO = "auto"

MODES = (MODE.DEFAULT, MODE.PLAN, MODE.AUTO)

class Behavior(Enum):
    ASK = "ask"
    ALLOW = "allow"
    DENY = "deny"

DEFAULT_RULES = [
    {"tool": "bash", "constrains": {"command": "rm -rf /"}, "behavior": Behavior.DENY},
    {"tool": "bash", "constrains": {"command": "sudo *"}, "behavior": Behavior.DENY},
    {"tool": "read_file", "constrains": {"path": "*"}, "behavior": Behavior.ALLOW},
    {"tool": "bash", "constrains": {"command": "ls -la *"}, "behavior": Behavior.ALLOW},
]

@dataclass
class Decision:
    behavior: Behavior
    reason: str

class BashSecurityValidator:
    VALIDATORS = [
        ("shell_metachar", r"[;&|`$]"),  # shell metacharacters
        ("sudo", r"\bsudo\b"),  # privilege escalation
        ("rm_rf", r"\brm\s+(-[a-zA-Z]*)?r"),  # recursive delete
        ("cmd_substitution", r"\$\("),  # command substitution
        ("ifs_injection", r"\bIFS\s*="),  # IFS manipulation
    ]

    def validate(self, command: str) -> list:
        return [(name, pattern) for name, pattern in self.VALIDATORS if re.search(pattern, command)]

    def describe_failures(self, command: str) -> str:
        failures = self.validate(command)
        if not failures:
            return "No issues detected"

        parts = [f"{name} (pattern: {pattern})" for name, pattern in failures]
        return "Security flags: " + ", ".join(parts)

bash_validator = BashSecurityValidator()

class PermissionManager:

    def __init__(self, mode: MODE, rules: list = None):
        if mode not in MODES:
            raise ValueError(f"Unknow mode: {mode}, Choose from {list(MODES)}")

        self.mode = mode
        self.rules = rules or list(DEFAULT_RULES)
        # 连续出错的次数，设置这个计数器是为了优化用户体验，避免用户一直点拒绝
        self.consecutive_denials = 0
        self.max_consecutive_denials = 3

    # 一共5步，每一段代码都是一个关卡，只有前面的关卡放行了，后面的关卡才有意义
    def check(self, tool_name: str, tool_input: dict) -> Decision:

        # 危险命令直接拒绝
        if tool_name == "bash":
            command = tool_input.get("command")
            failures = bash_validator.validate(command)
            if failures:
                desc = bash_validator.describe_failures(command)
                if any(failure[0] in {"sudo", "rm_rf"} for failure in failures):
                    return Decision(behavior=Behavior.DENY, reason = f"Bash validator: {desc}")

                return Decision(behavior=Behavior.ASK, reason = f"Bash validator flagged: {desc}")


        # 黑名单：用户默认拒绝
        for rule in self.rules:
            if rule["behavior"] != Behavior.DENY:
                continue
            if self._matches(rule, tool_name, tool_input):
                return Decision(behavior=Behavior.DENY, reason = f"Blocked by deny rule: {rule}")

        # 按照当前模式下校验规则
        # 写成 mode，变成 main 方法中的 mode 变量
        if self.mode == MODE.PLAN:
            if tool_name in WRITE_TOOLS:
                return Decision(behavior=Behavior.DENY, reason = "Plan mode: write operations are blocked")
            return Decision(behavior=Behavior.ALLOW, reason="Plan mode: read-only allowed")

        if self.mode == MODE.AUTO:
            #if tool_name in READ_ONLY_TOOLS or tool_name == "read_file":
            return Decision(behavior=Behavior.ALLOW, reason="Auto mode: read-only tool auto-approved")

        # 白名单：用户默认允许
        for rule in self.rules:
            if rule["behavior"] != Behavior.ALLOW:
                continue
            if self._matches(rule, tool_name, tool_input):
                # 一旦 LLM 知道怎么合理使用工具后，将 consecutive_denials 重置为 0, 避免 LLM 由于前面被拒绝2次后一直正确使用工具但是一旦错误使用一次后，便立即建议切换模式。
                self.consecutive_denials = 0
                return Decision(behavior=Behavior.ALLOW, reason = f"Matched allow rule: {rule}")

        # 兜底：请求用户批准
        return Decision(behavior=Behavior.ASK, reason=f"No rule matched for {tool_name}, asking user")

    # 匹配工具所有参数，工具名和参数只要有一个不对，立即返回 False
    def _matches(self, rule: dict, tool_name: str, tool_input: dict) -> bool:
        # 匹配 工具名称
        if rule.get("tool") and rule["tool"] != "*":
            if  rule["tool"] != tool_name:
                return False


        constrains = rule.get("constrains", {})
        # constrains 限制是必填项，而 tool_input 中有可选项
        for field, pattern in constrains.items():
            if not field in tool_input or not fnmatch(str(tool_input[field]), pattern):
                return False
        return True

    # 用户体验与安全性的权衡: "给权限”是可以持久化的信任，但“拒权限”通常是针对当前特定场景的怀疑
    def ask(self, tool_name: str, tool_input: dict) -> Decision:
        # ensure_ascii=False 字符的原始 UTF-8 编码（方便人类阅读）： Python 会为了兼容性，把所有非 ASCII 字符强制转换成 Unicode 转义序列
        print(f"\n [Permission] {tool_name}: {json.dumps(tool_input, ensure_ascii=False)[:200]}]")

        answer = input(f" Allow? (y/n/always): ").strip().lower()

        if answer in ("y", "yes"):
            self.consecutive_denials = 0
            return Decision(behavior=Behavior.ALLOW, reason=f"user allowed {tool_name}")

        if answer == "always":
            self.consecutive_denials = 0
            self.rules.append({"tool": tool_name, "constrains": tool_input, "behavior": Behavior.ALLOW})
            return Decision(behavior=Behavior.ALLOW, reason=f"user always allowed {tool_name}")

        # 连续多次拒绝应提醒用户切换模式
        self.consecutive_denials += 1
        if self.consecutive_denials >= self.max_consecutive_denials:
            print(f"[{self.consecutive_denials} consecutive denials -- " "consider switching to plan mode]")

        return Decision(behavior=Behavior.DENY, reason=f"user denied {tool_name}")


class HookEvent(Enum):
    SESSION_START = "SessionStart"
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    SESSION_END = "SessionEnd"

class HookType(Enum):
    START = "start"
    INPUT="input"
    POLICY="policy"
    AUDIT="audit"
    END = "end"

class HookManager:

    # 初始化 并加载 hook
    def __init__(self, hook_path: Path = None, sdk_mode: bool = False):
        self.hooks = {HookEvent.PRE_TOOL_USE.value: [],HookEvent.POST_TOOL_USE.value: [],HookEvent.SESSION_START.value: []}
        self._sdk_mode = sdk_mode
        self._load_hooks(hook_path)

    # 实际加载 hook
    def _load_hooks(self, hook_path: Path):
        hook_path = hook_path or (WORKDIR / ".hooks.json")
        if hook_path.exists():
            try:
                hook_json = json.loads(hook_path.read_text())
                hook_config = hook_json.get("hooks", {})
                for even in HookEvent:
                    self.hooks[even.value] = hook_config.get(even.value, [])

                print(f"[Hooks loaded from {hook_path}]")
            except Exception as e:
                print(f"[Hook config error: {e}]")

    # 校验是否在信任的文件夹里面
    def _check_workspace(self) -> bool:
        # 当作为 SDK 使用时环境的安全性由你的主程序（宿主环境）担保，不再需要额外的 .claude_trusted 文件来多此一举
        if self._sdk_mode:
            return True

        # TRUST_MARKER 相当于自己显式授权可以执行外部的hooks
        return TRUST_MARKER.exists()

    def run_hooks(self, hookEvent: HookEvent, hookType: HookType, context: dict) -> dict:
        result = {"blocked": False, "messages": []}
        # 检查是否可信的 hook 目录
        if not self._check_workspace():
            return result

        # 获取 指定的 hook
        hooks = [event for event in self.hooks[hookEvent.value] if event["type"] == hookType.value]

        # 对每个hook
        for hook in hooks:
            # 匹配 工具名
            tool_pattern = hook.get("matcher", "")
            if tool_pattern and context:
                tool_name = context.get("tool_name", "")
                if tool_pattern != "*" and tool_pattern != tool_name:
                    continue
            # 校验 command
            command = hook.get("command", "")
            if not hook.get("command", ""):
                continue

            # 组装 环境变量
            # 使用环境变量传递参数，比直接拼接命令参数要更好维护
            env = dict(os.environ)
            if context:
                env["HOOK_EVENT"] = hookEvent.value
                env["HOOK_TOOL_NAME"] = context.get("tool_name", "")
                tool_input = context.get("tool_input", {})
                env["HOOK_TOOL_INPUT"] = json.dumps(tool_input, ensure_ascii=False)[:10000]

                if "tool_output" in context:
                    env["HOOK_TOOL_OUTPUT"] = str(context["tool_output"])[:10000]

            # 执行 工具
            try:
                #  执行 Hook 脚本看成在调用微服务 API
                ret = subprocess.run(
                    command,
                    shell=True,
                    cwd=WORKDIR,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=HOOK_TIMEOUT,
                )
                # 处理返回结果
                self._handle_ret(ret, hookEvent.value, result, context)
            except subprocess.TimeoutExpired:
                print(f"  [hook:{hookEvent.value}] Timeout ({HOOK_TIMEOUT}s)")
            except Exception as e:
                print(f"  [hook:{hookEvent.value}] Error: {e}")
        return result

    def _handle_ret(self, ret, event: str, result: dict, context: dict):
        # 状态码是独立于数据流之外的控制信号，类似于 HTTP状态码
        # 0 表示执行成功
        if ret.returncode == 0:
            res = ret.stdout.strip()
            if res:
                print(f"  [hook:{event}] {ret.stdout.strip()[:100]}")

            if not res.startswith("{"):
                return

            hook_output = json.loads(ret.stdout)
            if "updatedInput" in hook_output and context:
                context["tool_input"] = hook_output["updatedInput"]

            # 即使正常执行工具，也因为实际情况，提醒 LLM 调整
            if "additionalContext" in hook_output:
                result["messages"].append(hook_output["additionalContext"])

            # 即使技术上执行成功，但也可能没权限，例如 token 不足
            if "permissionDecision" in hook_output:
                result["permission_override"] = (hook_output["permissionDecision"])
        #  Hook 脚本主动拦截工具的执行， 类似 HTTP 503
        elif ret.returncode == 1:

            result["blocked"] = True
            reason = ret.stderr.strip() or "Blocked by hook"
            result["block_reason"] = reason
            print(f"  [hook:{event}] BLOCKED: {reason[:200]}")
            pass
        # 2 表示 执行失败, 类似 HTTP 500
        elif ret.returncode == 2:
            msg = ret.stderr.strip()
            if msg:
                result["messages"].append(msg)
                print(f"  [hook:{event}] INJECT: {msg[:200]}")

class MemType(Enum):
    USER="user"
    FEEDBACK="feedback"
    PROJECT="project"
    REFERENCE="reference"


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
  .memory/              <-- 核心记忆文件夹
    MEMORY.md           <-- 自动生成的记忆索引（只读）
    .dream_lock         <-- (可选) 梦境整合进程的 PID 锁文件
    prefer_tabs.md      <-- 个体记忆文件 1 (Frontmatter 格式)
    review_style.md     <-- 个体记忆文件 2
    incident_board.md   <-- 个体记忆文件 3
Each memory is a small Markdown file with frontmatter.
The agent can save a memory through save_memory(), and the memory index
is rebuilt after each write.
An optional "Dream" pass can later consolidate, deduplicate, and prune
stored memories. It is useful, but it is not the first thing readers need
to understand.
Key insight: "Memory only stores cross-session information that is still
worth recalling later and is not easy to re-derive from the current repo."
"""


@dataclass
class Memory:
    # 名称
    name: str = ""
    # 类型
    mem_type: MemType = MemType.PROJECT
    # 描述
    description: str = ""
    # 内容
    content: str = ""


# 输出北京东八区时间戳 20250515_22-40-00_123
def get_timestamp() -> str:
    return datetime.fromtimestamp(int(time.time()), timezone(timedelta(hours=8))).strftime('%Y%m%d_%H-%M-%S_%f')[:-3]


class MemoryManager:

    def __init__(self, memo_dir: Path = None):
        # self.memo_dir = memo_dir or MEMORY_DIR
        self.memories: dict[str, Memory] = {}

    # 获取 最新的 记忆文件夹
    def _get_memo_dir(self) -> Path :
        # 获取.memory 下所有的记忆文件夹，忽略做梦中的文件夹
        memo_dirs = [dir for dir in MEMORY_DIR.iterdir() if dir.is_dir() and not dir.name.endswith(".dream")]

        # 没有记忆文件夹就生成一个
        if len(memo_dirs) == 0:
            memo_dir = MEMORY_DIR / get_timestamp()
            memo_dir.mkdir(parents=True, exist_ok=True)
            return memo_dir
        # 按 创建日期排序
        memo_dirs = sorted(memo_dirs, key=lambda d: getattr(d.stat(), "st_birthtime", d.stat().st_mtime))
        # 返回最后创建的文件夹
        return memo_dirs[-1]

    # 加载所有的 memory ，涉及大量的 IO 操作，不要在 __init__ 调用，方便出错好排查
    def load_memories(self):

        memo_dir = self._get_memo_dir()

        # 判断 memo_dir 为空文件夹，就提前结束
        if not any(memo_dir.iterdir()):
            return

        # 清空 原来的 memories
        self.memories = {}
        # 遍历 memo_dir 下所有 md 文件
        for md_file in sorted(memo_dir.glob("*.md")):
            if md_file.name == "MEMORY.md":
                continue
            memory = self._parse_frontmatter(md_file)
            name = memory.name or md_file.name
            if memory:
                self.memories[name] = memory

        # 重建索引
        self._rebuild_index()

        count = len(self.memories)
        if count > 0:
            print(f"[Memory loaded: {count} memories from {memo_dir}]")


    def _rebuild_index(self):
        if not self.memories:
            return

        lines = ["# Memory Index", ""]
        for name, memory in self.memories.items():
            line = f"- {name}: {memory.description} [{memory.mem_type.value}]"
            lines.append(line)
            if len(lines) > MAX_INDEX_LINES:
                lines.append(f"... (truncated at {MAX_INDEX_LINES} lines)")
                break
        # 会完全覆盖（Overwrite）掉 MEMORY_INDEX 文件原有的所有内容
        MEMORY_INDEX_FILE.write_text("\n".join(lines) + "\n")

    # 解析 memory 的 md 文件
    def _parse_frontmatter(self, path: PurePath) -> Memory | None:

        # 正则解析
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", path.read_text(), re.DOTALL)

        if not match:
            return None

        # 获取 元数据 和 正文
        meta = match.group(1)
        body = match.group(2)

        # 拼装 memory
        memory = Memory()
        memory.content = body
        for line in meta.splitlines():
            if ":" in line:
                # partition(":") 类似 split(":")，但更安全更优雅
                key, _, value = line.partition(":")
                if "name" == key.strip():
                    memory.name = value.strip()
                elif "description" == key.strip():
                    memory.description = value.strip()
                elif "mem_type" == key.strip():
                    memory.mem_type = value.strip()
        return memory


    def save_memory(self, memory: Memory) -> str:
        # memory 校验 memory 类型

        if memory.mem_type not in MemType:
            return f"Error: type must be one of {MemType}"

        # 规范化 文件名
        # 将文件名中除了[a-zA-Z0-9_-]的特殊字符全部转换为_,^ 在 [] 中表示取反
        safe_name = re.sub(r"[^a-zA-Z0-9_-]",  "_", memory.name.lower())
        if not safe_name:
            return "Error: invalid memory name"

        # 按格式拼装 memory
        frontmatter = (
            f"---\n"
            f"name: {memory.name}\n"
            f"description: {memory.description}\n"
            f"type: {memory.mem_type.value}\n"
            f"---\n"
            f"{memory.content}\n"
        )
        # 未完成时，带上 tmp 点后缀，避免被加载
        memo_dir = self._get_memo_dir()
        tmp_path = memo_dir / f"{safe_name}.new.md.tmp"
        tmp_path.write_text(frontmatter)
        # 去掉 .tmp 后缀表示写完了
        final_path = memo_dir / f"{safe_name}.new.md"
        os.rename(tmp_path, final_path)
        # 添加到 memories
        self.memories[memory.name] = memory

        # 重建 索引
        self._rebuild_index()

        # 检查是否存最新的记忆文件夹或 .dream 文件夹

        dream_dir = [dir for dir in MEMORY_DIR.iterdir() if dir.is_dir() and dir.name.endswith(".dream")][0]
        if dream_dir:
            last_memo_dir = dream_dir
        else:
            last_memo_dir = self._get_memo_dir()

        # 拷贝到最新记忆文件夹中
        if last_memo_dir != memo_dir:
            self.copy_to_last_dir(final_path, last_memo_dir)

        return f"Saved memory '{memory.name} to {final_path.relative_to(WORKDIR)}'"

    # 安全复制记忆文件到最新的记忆文件夹
    def copy_to_last_dir(self, source_memo: Path, last_memo_dir: Path):
        try:
            dir_fd = os.open(last_memo_dir, os.O_RDONLY | os.O_DIRECTORY)
        except FileNotFoundError:
            # 避免打开过程中出现 .dream 文件夹改名
            last_memo_dir = self._get_memo_dir()
            dir_fd = os.open(last_memo_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            dest_fd = os.open(source_memo.name, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, dir_fd)
            with os.fdopen(dest_fd, "wb") as dest:
                dest.write(source_memo.read_bytes())
            print(f"{source_memo.name} 已经安全拷贝到 {last_memo_dir.name}")
        finally:
            os.close(dir_fd)

    # 全量加载 memory
    def _load_memory_prompt(self) -> str:
        # 判空
        if not self.memories:
            return ""

        sections = []
        sections.append(f"# Memories (persistent across sessions)")
        # 空行 语义边界感
        sections.append("")

        # 按 memory 类型遍历
        for memo_type in MemType:
            sections.append(f"## [{memo_type.value}]")
            for name, memo in self.memories.items():
                if memo_type == memo.mem_type:
                    sections.append(f"### {name}: {memo.description}")
                    if memo.content.strip():
                        sections.append(memo.content.strip())
                    # 空行 语义边界感
                    sections.append("")
        return "\n".join(sections)

    def build_system_prompt(self):
        sys_prompt = [f"You are a coding agent at {WORKDIR}. Use tools to solve tasks."]

        memory_section = self._load_memory_prompt()
        if memory_section:
            sys_prompt.append(memory_section)
        sys_prompt.append(MEMORY_GUIDANCE)
        return "\n\n".join(sys_prompt)

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
    def consolidation(self, mem_type: MemType = None) -> str:
        # 获取当前最新的记忆文件夹
        to_merge_memory: dict[MemType,list[Memory]] = {}
        if mem_type == None:
            for type in MemType:
                to_merge_memory[type] = []
                for _, memo in self.memories.items():
                    if type == memo.mem_type:
                        to_merge_memory[type].append(memo)
        else:
            to_merge_memory[mem_type] = []
            for _, memo in self.memories.items():
                if mem_type == memo.mem_type:
                    to_merge_memory[mem_type].append(memo)

        # 拼装成完整的 prompt ，提醒 LLM 按 记忆类型分类记忆
        merge_prompt = '''
        你是一个专业的记忆架构师，正在帮助 Agent 更好地服务用户当前的任务。

        【当前任务目标】
        {current_goal}
        
        【当前项目/阶段背景】
        {project_context}   // 可选：项目类型、当前阶段、关键约束等
        
        请将以下【{category}】类别的多条记忆进行高质量合并。
        
        合并原则：
        1. 优先保留与当前任务目标高度相关的记忆。
        2. 对与当前目标冲突或过时的记忆，进行合理弱化、标注或场景化处理。
        3. 突出那些能直接提升任务完成质量的关键偏好和规则。
        4. 显式标注矛盾，并根据当前目标给出推荐优先级。
        5. 输出控制在精炼、高信息密度。
        
        原始记忆：
        {memories_or_clusters}
        '''

        # 调用 LLM 整合 记忆
        # 创建 .dream 文件夹
        # 生成新的记忆文件 输出到 .dream 文件夹中

        pass

def safe_path(p: str) -> Path:
    # pathlib 的 / 操作符对绝对路径有特殊处理：如果右边p是绝对路径，会忽略左边WORKDIR的路径。
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
def record_recent_file(path, state: CompactState):
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
    },

    {
        "name": "save_memory",
        "description": "Save a persistent memory that survives across sessions.",
        "input_schema":
            {
                "type": "object",
                "properties":
                    {
                        "name":
                            {
                                "type": "string",
                                "description": "Short identifier (e.g. prefer_tabs, db_schema)"
                            },
                        "description":
                            {
                                "type": "string",
                                "description": "One-line summary of what this memory captures"
                            },
                        "type":
                            {
                                "type": "string",
                                "enum": ["user", "feedback", "project", "reference"],
                                "description": "user=preferences, feedback=corrections, project=non-obvious project conventions or decision reasons, reference=external resource pointers"
                            },
                        "content":
                            {
                                "type": "string",
                                "description": "Full memory content (multi-line OK)"
                            },
                    },
                "required": ["name", "description", "type", "content"]
            }
    },


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

        for block_idx, block in enumerate(message.get("content")):
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
        if isinstance(content, str) and len(content) > 120:
            block["content"] = "[Earlier tool result compacted. Re-run the tool if you need full detail.]"
    return messages


# 将 Agent 对话记录持久化到硬盘，并采用了一种非常适合日志记录的格式：JSONL (JSON Lines)
def write_transcript(messages: list) -> Path:
    # 　确保目录存在
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

    # 获取所有的 文本，避免使用ThinkingBlock时，缺少 text 属性
    summary = [block.text.strip() for block in response.content if isinstance(block, TextBlock)]
    return "\n".join(summary)


def compact_history(messages: list, state: CompactState, focus: str | None = None) -> list:
    # 转写历史消息到磁盘
    transcript_path = write_transcript(messages)
    print(f"[transcript saved: {transcript_path}]")
    # 生成摘要信息
    summary_parts = [summarize_history(messages)]
    if focus:
        summary_parts.append(f"Focus to preserve next: {focus}")
    if state.recent_files:
        lines = "\n".join(f"- {file}" for file in state.recent_files)
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


# 由于增加了 block.id 和 state，需要用适配器连接
def execute_tool(block, state: CompactState) -> str:
    fn = block.name

    if fn == "bash":
        output = run_bash(block.input["command"], block.id)
    elif fn == "read_file":
        output = run_read(block.input["path"], block.id, state, block.input.get("limit"))
    elif fn == "write_file":
        output = run_write(block.input["path"], block.input["content"])
    elif fn == "edit_file":
        output = run_edit(block.input["path"], block.input["old_text"], block.input["new_text"])
    elif fn == "load_skill":
        output = SKILL_REGISTRY.load_skill(block.input["skill_name"])
    elif fn == "todo":
        output = TODO.update(block.input["plan_items"])
    elif fn == "task":
        desc = str(block.input.get("description", "subtask"))
        prompt = str(block.input.get("prompt", ""))
        print("========== ========== subagent 开始 ========== ==========")
        print(f"> task ({desc}): {prompt[:80]}")
        output = run_subagent(block.input["prompt"])
        print("========== ========== subagent 结束 ========== ==========")
    elif fn == "save_memory":
        memory = Memory(name=block.input["name"], mem_type=MemType(block.input["type"]), description=block.input["description"], content=block.input["content"])
        output =  memoryManager.save_memory(memory)

    else:
        output = f"Unknown tool: {fn}"

    return output

def collect_hook_msg(pre_tool_result: dict, tool_result_content: list, hookEvent: HookEvent):
    prefix = {
        HookEvent.SESSION_START: "[hook]",
        HookEvent.PRE_TOOL_USE: "[hook message]",
        HookEvent.POST_TOOL_USE: "[hook note]",
        HookEvent.SESSION_END: "[hook]",
    }[hookEvent]

    for msg in pre_tool_result.get("messages", []):
        tool_result_content.append({
            "type": "text",
            "text": f"{prefix}: {msg}]"
        })

def agent_loop(messages: list, state: CompactState, perms: PermissionManager, hooks: HookManager, memories: MemoryManager) -> None:
    while True:
        messages[:] = normalize_messages(messages)
        # 例行压缩
        messages[:] = compact_tool_result(messages)

        if len(str(messages)) > CONTEXT_LIMIT:
            print("[auto compact]")
            messages[:] = compact_history(messages, state)
        system_prompt = memories.build_system_prompt()
        response = client.messages.create(
            model=MODEL,
            system=system_prompt,
            messages=messages,
            tools=PARENT_TOOLS,
            max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return
        results = []
        tool_result_content = []
        # 标记是否更新了 TODO 列表
        update_todo_flag = False

        compact_flag = False
        compact_goal = None
        for block in response.content:
            if block.type == "tool_use":
                tool_input = block.input or {}
                context = {"tool_name": block.name, "tool_input": tool_input}
                try:
                    print(f"> {block.name} parameter: {block.input}")
                    # PreToolUse 治理类型 参数纠错、路径补全、数据脱敏、格式转换
                    input_hook_result = hooks.run_hooks(HookEvent.PRE_TOOL_USE, HookType.INPUT, context)
                    collect_hook_msg(input_hook_result, tool_result_content, HookEvent.PRE_TOOL_USE)

                    decision = perms.check(block.name, tool_input)
                    if decision.behavior == Behavior.ASK:
                        decision = perms.ask(block.name, tool_input)
                    # 允许
                    if decision.behavior == Behavior.ALLOW:
                        # PreToolUse 决策类型 外部 API 授权、动态配额检查、复杂业务逻辑判断
                        policy_hook_result = hooks.run_hooks(HookEvent.PRE_TOOL_USE, HookType.POLICY, context)
                        if policy_hook_result.get("blocked"):
                            reason = policy_hook_result.get("block_reason", "Blocked by hook")
                            tool_result_content.append({
                                "type": "text",
                                "text": f"Tool blocked by PreToolUse hook: {reason}"
                            })
                            # 退出 工具执行
                            continue

                        output = execute_tool(block, state)
                    # 拒绝
                    else:
                        # decision['reason'] 是字典才这么访问的，decision 是对象了，再这样访问就会报错：'Decision' object is not subscriptable
                        output = f"Permission denied by user for {block.name}: {decision.reason}"
                        print(f"[USER DENIED] {block.name}: {decision.reason}")
                except Exception as exc:
                    output = f"Error: {exc}"
                print(f"> {block.name} result: {str(output)[:200]}")

                tool_result_content.append({
                    "type": "text",
                    "text": str(output)
                })
                # PostToolUse 审计反馈型: 结果记录、输出内容脱敏、给 Agent 提供改进建议
                context["tool_output"] = output
                audit_hook_result = hooks.run_hooks(HookEvent.POST_TOOL_USE, HookType.AUDIT, context)
                collect_hook_msg(audit_hook_result, tool_result_content, HookEvent.POST_TOOL_USE)

                if block.name == "todo":
                    update_todo_flag = True

                if block.name == "compact":
                    compact_flag = True
                    compact_goal = (block.get("input") or {}).get("goal")

                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": tool_result_content
                })

        # 没有更新时 要提醒 LLM 要更新
        # 只有 当计划没有完成时，才需要更改计划
        if any(item.status != "completed" for item in TODO.state.plan_items):
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

# main 函数块内定义的变量是模块级全局变量，其他函数或方法命名了和main函数一样的同名变量，只要赋值了就不会被覆盖。要是不赋值，直接读取就会读取到 main函数一样的同名变量
if __name__ == "__main__":


    hookManager = HookManager()
    hookManager.run_hooks(HookEvent.SESSION_START, HookType.START, {"tool_name": "", "tool_input": {}})

    memoryManager = MemoryManager()
    memoryManager.load_memories()

    # 选择 策略模式
    print("Permission modes: default, plan, auto")

    mode_input = input("Mode (default): ").strip().lower() or "default"
    if mode_input not in {mode.name for mode in MODE}:
        mode = MODE.DEFAULT
    mode = MODE(mode_input)
    perms = PermissionManager(mode=mode)
    print(f"[Permission mode: {mode_input}]")

    history = []
    state = CompactState()
    while True:
        try:
            query = input("\033[36ms09 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break

        if query.startswith("/mode"):
            # split 没有参数时，以任意连续的空白字符 作为分隔符
            parts = query.split()
            if len(parts) == 2 and parts[1] in {mode.value for mode in MODE}:
                perms.mode = MODE(parts[1])
                print(f"[Switched to {parts[1]} mode]")
            else:
                print(f"[Usage: /mode <{'|'.join({mode.value for mode in MODE})}>")
            continue

        if query.strip() == "/rules":
            for i, rule in enumerate(perms.rules):
                print(f" {i}: {rule}")
            # 这时停止往下执行，避免 将 /rules 输入到会话中
            continue

        if query.strip() == "/memory":
            if memoryManager.memories:
                for name, memory in memoryManager.memories.items():
                    print(f" [{memory.mem_type.value}]: {name}: {memory.description}")
            else:
                print(f"No memory available")
            continue
        history.append({"role": "user", "content": query})
        agent_loop(history, state, perms, hookManager, memoryManager)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print()
                    print('========== ========== 执行结果 ========== ==========')
                    print(block.text)
        print()
