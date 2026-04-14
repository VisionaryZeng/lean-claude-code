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
from dataclasses import dataclass
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

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
    }
]

client = Anthropic(base_url=os.getenv('ANTHROPIC_BASE_URL'))


@dataclass
class LoopState:
    # 定义消息列表，思考循环次数，继续思考循环的理由

    messages: list
    turn_count: int = 0
    transition_reason: str | None = None


def run_bash(command: str) -> str:
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


def execute_tool_calls(response_content) -> list[dict]:
    results = []

    '''
    block 格式类似下面这样

    {
        "type": "tool_use",
        "id": "toolu_01A2B3C4",
        "name": "bash",
        "input": {
            "command": "ls -la"
        }
    }
    '''

    for block in response_content:
        if block.type != "tool_use":
            continue

        command = block.input['command']
        # 打印 要执行的命令
        print(f"\033[33m$ {command}\033[0m")
        # 执行命令
        output = run_bash(command)
        # 打印 要执行的结果
        print(output[:200])
        results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": output
        })
    return results


def run_one_turn(state: LoopState) -> bool:
    # 获取 LLM 返回结果
    response = client.messages.create(
        model=MODEL,
        system=SYSTEM,
        tools=TOOLS,
        messages=state.messages,
        max_tokens=8000
    )

    # 记录 assistant 响应消息
    state.messages.append({'role': 'assistant', 'content': response.content})

    # 没有工具调用时应结束循环
    if response.stop_reason != 'tool_use':
        state.transition_reason = None
        return False

    # 执行工具调用
    result = execute_tool_calls(response.content)

    # 工具调用时没有结果应结束循环
    if not result:
        state.transition_reason = None
        return False

    # 准备进入下一轮思考循环
    state.messages.append({'role': 'user', 'content': result})
    state.transition_reason = 'tool_result'
    state.turn_count += 1
    return True


def agent_loop(state: LoopState) -> None:
    # 类似 do while
    while run_one_turn(state):
        pass


def extract_text(content) -> str:
    if not isinstance(content, list):
        return ''
    texts = ['========== ========== 执行结果 ========== ==========']
    for block in content:
        text = getattr(block, 'text', None)
        if text:
            texts.append(text)

    return '\n'.join(texts).strip()


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
        # 初始化状态
        state = LoopState(messages=history)

        # 开启思考循环
        agent_loop(state)

        # 获取最后一条消息
        finalMessage =extract_text(history[-1]['content'])
        if finalMessage:
            print(finalMessage)
        print()
