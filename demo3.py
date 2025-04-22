import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from IPython.display import Image, display
from langchain_core.runnables.graph_mermaid import MermaidDrawMethod

"""
创建一个简单的请假业务的多智能体
"""
load_dotenv(".env")
model_name = os.environ.get("model_name")
base_url = os.environ.get("base_url")
api_key = os.environ.get("api_key")

def get_chat_model(temperature=0.8, top_p=0.9):
    base_url = base_url
    api_key = api_key
    chat_model = ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key, 
        temperature=temperature, 
        max_tokens=8192,
        top_p=top_p, 
        max_retries=3)
    return chat_model

def replace(a, b):
    return b

@tool
def toolA(query, database=""):
    """
    从知识库中请假的规章制度信息 用于判断用户的请假是否符合规定
    args: 
        query: str 用户问题
        database: 知识库名称, 现有知识库为"规章制度知识库"
    return:
        str 与用户问题相关的知识
    """
    return "根据规定, 员工一年有5天年假。"

@tool
def toolB():
    """
    获取当前日前
    """
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def toolC(userId):
    """
    获取员工信息
    args:
        userId: str 员工工号
    """
    return "工号100001 姓名张三"

@tool
def toolD(userName, timeStart, timeEnd, type):
    """
    提交请假申请
    args:
        userName: str 员工姓名
        timeStart: str 请假开始时间
        timeEnd: str 请假结束时间
        type: str 请假类型
    return:
        str 请假申请结果
    """
    return "请假成功, 调用toolE结束流程"

@tool
def toolE(userId, timeStart, timeEnd):
    """
    不调用工具时使用
    """
    return ""

tools = [toolA, toolB, toolC, toolD, toolE]

class State(TypedDict):
    messages: Annotated[list, add_messages] = []
    knowledge: Annotated[str, replace] = ""
    userId: Annotated[str, replace] = ""
    plan: Annotated[str, replace] = ""
    userInfo: Annotated[str, replace] = ""
    tool_info: Annotated[str, replace] = ""
    time: Annotated[str, replace] = ""

llm = get_chat_model(temperature=0.1, top_p=0.1)
llm_with_tool = llm.bind_tools(tools, tool_choice="any")
# 读取用户输入
def input_node(state: State):
    user_input = input("请输入消息（输入 'exit' 退出）：")
    message = HumanMessage(content=user_input)
    return {"messages": message}

def plan(state: State):
    tool_info = str(tools)
    system_prompt = f"""
    角色: 你是一个规划者
    目的: 你的任务是将用户的请求拆解成多个子任务 子任务需要根据工具的内容决定
    输出格式为:
    1. 子任务1 调用工具xxx
    2. 子任务2 调用工具xxx
    ...
    工具信息如下:
    {tool_info}
    """
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(state["messages"])
    response = llm.invoke(messages)
    return {"messages": response, "plan": response.content}

def exector(state: State):
    plan = state["plan"]
    tool_info = state["tool_info"]
    userInfo = state["userInfo"]
    timeNow = state["time"]
    userId = state["userId"]
    knowledge = state["knowledge"]
    system_prompt = f"""
    角色: 你是一个执行者
    目的: 你的任务是根据规划者的计划调用工具执行任务, 并给出用户反馈
    当完成用户目的时,调用工具toolE
    当前的计划为:
    {plan}
    当前的用户信息为:
    用户信息: {userInfo}
    工号: {userId}
    当前的时间为:
    {timeNow}
    当前的工具的调用信息:
    {tool_info}
    知识库:
    {knowledge}
    """
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(state["messages"])
    response = llm_with_tool.invoke(messages)
    return {"messages": response}

def loop(state: State):
    last_message = state["messages"][-1]
    num_tools = len(last_message.tool_calls)
    if num_tools==0 or (num_tools == 1 and last_message.tool_calls[0]["name"] == "toolE"):
        return "chat"
    else:
        return "tool_execute"
    
def chat(state: State):
    plan = state["plan"]
    tool_info = state["tool_info"]
    userInfo = state["userInfo"]
    timeNow = state["time"]
    userId = state["userId"]
    knowledge = state["knowledge"]
    system_prompt = f"""
    角色: 你是请假助手
    目标: 以用户友好的格式给出反馈信息
    当前的计划为:
    {plan}
    当前的用户信息为:
    用户信息: {userInfo}
    工号: {userId}
    当前的时间为:
    {timeNow}
    当前的工具的调用信息:
    {tool_info}
    知识库:
    {knowledge}
    """
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(state["messages"])
    response = llm.invoke(messages)
    return {"messages": response}
    
def tool_execute(state: State):
    tool_call = state["messages"][-1].tool_calls
    for tool in tool_call:
        tool_name = tool["name"]
        print("调用工具:", tool_name, "\n")
        tool_func = globals()[tool_name]
        args = tool["args"]
        result =tool_func(args)
        state["tool_info"] += result + "\n"
        if tool_name == "toolA":
            state["knowledge"] = result
        elif tool_name == "toolB":
            state["time"] = result
        elif tool_name == "toolC":
            state["userInfo"] = result
    return state

# 创建条件边
def if_end(state: State):
    last_message = state["messages"][-1]
    if isinstance(last_message, HumanMessage) and last_message.content.lower() == "exit":
        return "end"
    return "input"

# 创建状态图
graph = StateGraph(State)
graph.add_node("input", input_node)
graph.add_node("plan_impl", plan)
graph.add_node("exector", exector)
graph.add_node("tool_execute", tool_execute)
graph.add_node("chat", chat)
graph.add_edge(START, "input")
graph.add_conditional_edges(
    "input",
    if_end,
    {
        "end": END,
        "input": "plan_impl"
    }
)
graph.add_edge("plan_impl", "exector")  # 规划后返回到执行节点
graph.add_conditional_edges(
    "exector",
    loop,
    {
        "chat": "chat",
        "tool_execute": "tool_execute"
    }
)
graph.add_edge("chat", "input")  # 聊天后返回到执行节点
graph.add_edge("tool_execute", "exector")  # 执行后返回到执行节点
# 编译并执行
workflow = graph.compile()

# 运行状态图
def run_workflow():
    print("您好, 我是请假助手, 有什么需要帮助的吗？")
    initial_state = {"messages": [], "userId": "4123"}
    # workflow.invoke() 非流式 只返回最后一个节点的状态
    for type, output in workflow.stream(initial_state, stream_mode=["messages"]):
        # output[0]是大模型返回的消息 output[1]是大模型调用的统计信息
        if isinstance(output[0], AIMessage):
            print(output[0].content, flush=True, end="")
        pass  # 流式处理输出，这里可以打印或处理每个步骤的输出



if __name__ == "__main__":
    run_workflow()