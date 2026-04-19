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
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

WORKDIR = Path.cwd()

MODEL = os.getenv('MODEL_ID')

# 括号隐式拼接: 拼接多个字符串字面量，整洁、符合规范、不带多余换行、性能高。
SYSTEM = (
    f"you are a coding agent at {os.getcwd()}"  # 自动会获取当前路径
    "Use bash to inspect and change the workspace. Act first, then report clearly."
)

'''

工具元素列表： LLM  根据 description 决定要调用此工具或API, input_schema 提示 LLM 该如何传参数
即使没有设置调用的工具或 API，后面解析 LLM 返回的 response.content 时，会根据调用的工具名称(name)实现调用具体的工具或API

'''

TOOLS = [
    {  # 工具名称，解析 LLM 返回工具调用时用于区分工具
        "name": "bash",
        # 决定 LLM 是否使用这个工具
        "description": "Run a shell command in the current workspace.",

        # 参数提示信息
        "input_schema": {
            # 为参数定义一个容器，表示传递一个字典
            "type": "object",
            # 定义所有的参数类型，可以包含 type，description，enum 等元素
            "properties": {
                # 可以增加 "enum" 元素限制参数的选择，这对于控制 AI 的行为非常有用。 枚举类型 如： "enum": ["read", "write", "delete"]
                "command": {"type": "string"},
            },
            # 必填的参数
            "required": ["command"],
        }
    },
    {
        "name": "read_file",
        "description": "Read file content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description" : "Path from current workdir"},
                "limit": {"type": "integer"},
            },
            "required": ["path"],
        }
    },
    {
        "name": "write_file",
        "description": "Write file content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description" : "Path from current workdir"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        }
    },
    {
        "name": "edit_file",
        "description": "Replace exact text in file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description" : "Path from current workdir"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
        }
    },
]

client = Anthropic(base_url=os.getenv('ANTHROPIC_BASE_URL'))


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]

    # for d in dangerous: 它会遍历 dangerous 列表中的每一个元素，并将其临时赋值给变量 d
    # d in command：这是一个子字符串匹配操作。它在检查当前的危险词 d 是否出现在 AI 生成的字符串 command 之中
    # (d in command for d in dangerous)：这是一个生成器表达式,它不会立刻生成一个完整的列表，而是产生一个序列，里面包含了一连串的 True 或 False
    # any(...)：这是 Python 的内置函数。它的规则是：只要序列中有一个元素是 True，结果就是 True；只有全部都是 False 时，结果才是 False
    if any(d in command for d in dangerous):
        return f"Error: Dangerous command blocked"

    try:
        result = subprocess.run(
            # 运行的具体命令字符串（如 ls -la 或 python3 script.py）
            command,
            # 允许命令通过 Shell（Linux 下通常是 /bin/sh，Windows 下是 cmd.exe）执行。
            shell=True,
            # 确保 AI 所有的操作（读文件、写文件）都发生在你当前的工程目录下
            cwd=os.getcwd(),
            # 默认情况下，命令的输出会直接打印到你的终端。开启这个参数后，Python 会把命令执行产生的 stdout（标准输出）和 stderr（错误输出）统统“拦截”下来
            capture_output=True,
            # 计算机底层输出的是二进制字节（Bytes）。开启此参数后，Python 会自动根据系统编码将字节转换为我们能读懂的字符串。
            text=True,
            # 安全保险丝。如果 AI 运行了一个死循环脚本，或者一个需要跑很久的命令，程序不会无限期卡死。120 秒后，它会强行终止该进程并抛出一个错误
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"

    output = (result.stdout + result.stderr).strip()
    return output[:5000] if output else "(no output)"


def safe_path(path: str) -> Path:
    # 获取完整的绝对路径
    absolute_path = (WORKDIR / path).resolve()
    # 确保路径在工作路径下
    if not absolute_path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {path}")
    return absolute_path


def run_read(path: str, limit: int = None) -> str:
    text = safe_path(path).read_text()
    lines = text.splitlines()
    #  如果有限制就按限制行数读取
    if limit and limit < len(lines):
        lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
    # 没有限制就默认读取前 5w 个字符[:50000]
    return "\n".join(lines)[:50000]


def run_write(path: str, content: str) -> str:
    file_path = safe_path(path)
    # 确保目标文件所在的父目录一定存在，如果不存在就自动创建，如果已经存在也不报错。
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content)
    return f"Wrote {len(content)} bytes to {path}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    file_path = safe_path(path)
    # 先读取所有的文本
    content = file_path.read_text()
    # 如果 要替换的文本 在 原文本中,直接替换
    if old_text in content:
        # 替换之后再写入
        # 1 的意思只替换第一次遇到的匹配项，剩下的哪怕一模一样也不动
        content = content.replace(old_text, new_text, 1)
        file_path.write_text(content)
        return f"Edited {path}"
    # 没有找到要替换的文本，直接返回
    return f"Error: Text not found in {path}"


TOOL_HANDLERS = {
    # lambda (匿名函数) 是 Python 定义单行简易函数的关键字，充当了适配器
    # **kw (关键字参数收集) **（双星号）在 Python 参数列表中代表 “收集所有多余的关键字参数到一个字典中”
    # run_bash(kw["command"]) (函数体)  从刚才收集到的 kw 字典中，取出键名为 "command" 的值
    "bash": lambda **kw: run_bash(kw['command']),
    # 可选参数limit：kw.get('limit') 用 get 防止报错
    "read_file": lambda **kw: run_read(kw['path'], kw.get('limit')),
    "write_file": lambda **kw: run_write(kw['path'], kw['content']),
    "edit_file": lambda **kw: run_edit(kw['path'], kw['old_text'], kw['new_text']),
}


def normalize_messages(messages: list) -> list:
    # 剥离 LLM 不认识的内部元数据
    cleand = []
    for msg in messages:
        clean = {"role": msg['role']}
        if isinstance(msg["content"], str):
            clean["content"] = msg["content"]
        elif isinstance(msg["content"], list):
            # 类似于 Java 中的lambda 表达式，先获取msg["content"]的字典，然后再过滤掉字典中私有变量
            # 现在 装入 clean["content"] 的是一个列表，列表的每个元素都是一个字典
            clean["content"] = [{k: v for k, v in block.items() if not k.startswith('_')} for block in msg["content"] if isinstance(block, dict)]
        else:
            clean["content"] = msg.get("content", "")
        cleand.append(clean)
    # 确保每个工具使用都有一个执行结果返回
    # 收集 有 tool_result 的 tool_use ID
    existing_results = set()

    for msg in cleand:
        # msg.get("content") 如果取不到值，返回 None，不会报错
        if isinstance(msg["content"], list):
            # msg["content"] 如果取不到值，直接会报错
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    existing_results.add(block.get("tool_use_id"))
    # # 将 LLM 执行后但没有结果的 工具，设置为取消，将没有 tool_result 的 tool_use 补齐取消
    for msg in cleand:
        if msg["role"] != "assistant" and not isinstance(msg["content"], list):
            continue
        for block in msg["content"]:
            if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id") not in existing_results:
                cleand.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": block["id"],
                            "content": "(cancelled)"
                        }
                    ]
                })

    if len(cleand) == 0:
        return cleand

    # 合并相同角色的消息
    merged = [cleand[0]]
    for msg in cleand[1:]:
        pre_msg = merged[-1]
        if pre_msg["role"] == msg["role"]:
            pre_content = pre_msg["content"] if isinstance(pre_msg["content"], list) else [{"type": "text", "content": str(pre_msg["content"])}]
            cur_content = msg["content"] if isinstance(msg["content"], list) else [{"type": "text", "content": str(msg["content"])}]
            # 列表支持 用 + 直接合并
            pre_msg["content"] = pre_content + cur_content
        else:
            merged.append(msg)
    return merged

def agent_loop(messages: list) -> None:
    # 类似 do while
    while True:
        # 获取 LLM 返回结果
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            tools=TOOLS,
            messages=normalize_messages(messages),
            max_tokens=8000
        )

        # 规范化消息，记录 assistant 响应消息
        messages.append({'role': 'assistant', 'content': response.content})

        # 停止运行时没有要求调用工具，直接结束
        if response.stop_reason != 'tool_use':
            return

        # 需要调用工具时，支持执行多条命令
        results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"> {block.name}: {block.input}")
                tool_handler = TOOL_HANDLERS.get(block.name)
                output = tool_handler(**block.input) if tool_handler else f"Unknown tool: {block.name}"
                print(f" output: {output[:200]}")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
        messages.append({"role": "user", "content": results})


if __name__ == '__main__':
    # 设置 历史消息列表
    history = []
    while True:
        # 支持用户退出思考循环
        try:
            # 获取用户请求
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break

        history.append({'role': 'user', 'content': query})
        # 开启思考循环
        agent_loop(history)

        # 获取最后一条消息
        response_content = history[-1]['content']
        if isinstance(response_content, list):
            texts = ['========== ========== 执行结果 ========== ==========']
            for block in response_content:
                if getattr(block, 'text', None):
                    texts.append(block.text)
            print('\n'.join(texts).strip())
        print()
