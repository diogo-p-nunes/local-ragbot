import sys
import argparse
import uuid
from pathlib import Path

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from rich.console import Console
from rich.prompt import Prompt
from rich.text import Text

from const_prompt import SYSTEM_PROMPT

# Parse command line arguments
parser = argparse.ArgumentParser(description="Chatbot parameters")
parser.add_argument("--chat_model", type=str, default="Qwen/Qwen3-0.6B")
parser.add_argument("--query_model", type=str, default="")
parser.add_argument("--docs_dir", type=str, default="docs")
parser.add_argument("--openai_api_key", type=str, default="EMPTY")
parser.add_argument("--openai_api_base", type=str, default="http://localhost:8000/v1")
parser.add_argument("--max_tokens", type=int, default=1000)
parser.add_argument("--temperature", type=float, default=0.3)
parser.add_argument("--stream", action="store_true")
args = parser.parse_args()

# Query anc Chat models are the same by default
if args.query_model == "":
    args.query_model = args.chat_model

# Ensure the docs directory exists
Path(args.docs_dir).mkdir(exist_ok=True)

# Load chat and query models
chat_model = ChatOpenAI(
    model=args.chat_model,
    openai_api_key=args.openai_api_key,
    openai_api_base=args.openai_api_base,
    max_tokens=args.max_tokens,
    temperature=args.temperature,
    streaming=args.stream,
)

query_model = ChatOpenAI(
    model=args.query_model,
    openai_api_key=args.openai_api_key,
    openai_api_base=args.openai_api_base,
    max_tokens=200,
    temperature=0.7,
    streaming=False,
)

# Define the chat prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

def call_model(state: MessagesState):
    prompt = prompt_template.invoke(state["messages"])
    response = chat_model.invoke(prompt)
    return {"messages": response}

# Define the agent graph; compile agent with memory for chat history
workflow = StateGraph(state_schema=MessagesState)
workflow.add_node(call_model)
workflow.add_edge(START, "call_model")
agent = workflow.compile(checkpointer=MemorySaver())

def main(console: Console) -> None:
    EXIT = "/bye"
    # each launch of the script gets a unique thread ID
    thread_id = str(uuid.uuid4()) 

    while True:
        user_text = Prompt.ask("[bold magenta]>[/bold magenta]")
        if user_text.strip() == EXIT:
            console.print("\n[bold green]>[/bold green]: [green]Good-bye.[/green]")
            return
        input_message = [HumanMessage(content=user_text)]

        # --- run LangGraph ---
        label = Text(">:", style="bold green")
        if args.stream:
            console.print(label, end=" ")
            for chunk, metadata in agent.stream(
                {"messages": input_message},
                {"configurable": {"thread_id": thread_id}},
                stream_mode="messages",
            ):
                if metadata.get("langgraph_node") == "call_model" and isinstance(chunk, AIMessage):
                    partial_answer = Text(chunk.content, style="green")
                    console.print(partial_answer, style="green", end="")
            console.print("")
        else:
            result = agent.invoke(
                {"messages": input_message},
                {"configurable": {"thread_id": thread_id}},
            )
            answer = Text(result["messages"][-1].content, style="green")
            console.print(label, answer)

if __name__ == "__main__":
    console = Console()
    try: main(console)
    except KeyboardInterrupt:
        console.print("\n[bold green]>[/bold green]: [green]Good-bye.[/green]")