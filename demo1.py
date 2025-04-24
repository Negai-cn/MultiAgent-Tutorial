import os
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.graph_mermaid import MermaidDrawMethod
from IPython.display import Image, display
from dotenv import load_dotenv
"""
创建一个最简单的状态图
用户输入节点
包括大模型对话节点
"""
load_dotenv(".env")

def get_chat_model(temperature=0.8, top_p=0.9):
    model_name = os.environ.get("model_name")
    base_url = os.environ.get("base_url")
    api_key = os.environ.get("api_key")
    chat_model = ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key, 
        temperature=temperature, 
        max_tokens=8192,
        top_p=top_p, 
        max_retries=3)
    return chat_model

llm = get_chat_model(temperature=0.1, top_p=0.1)

class State(TypedDict):
    messages: Annotated[list, add_messages] = []

# 读取用户输入
def input_node(state: State):
    user_input = input("请输入内容（输入 'exit' 退出）：")
    message = HumanMessage(content=user_input)
    return {"messages": message}

# 创建交互节点
def interact_node(state: State):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": response}

# 创建条件边
def if_end(state: State):
    last_message = state["messages"][-1]
    if isinstance(last_message, HumanMessage) and last_message.content.lower() == "exit":
        return "end"
    return "interact"

# 创建状态图
graph = StateGraph(State)
graph.add_node("input", input_node)
graph.add_node("interact", interact_node)
graph.add_edge(START, "input")
graph.add_conditional_edges(
    "input",
    if_end,
    {
        "end": END,
        "interact": "interact"
    }
)
graph.add_edge("interact", "input")  # 交互后返回到输入节点

# 编译并执行
workflow = graph.compile()
# 运行状态图
def run_workflow():
    initial_state = {"messages": []}
    for output in workflow.stream(initial_state):
        # 以messages进行更新 当有新messages时会走到这里
        pass  # 流式处理输出，这里可以打印或处理每个步骤的输出
if __name__ == "__main__":
    run_workflow()
