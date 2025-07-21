import sys
import argparse
import uuid
from pathlib import Path
import pprint

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools.retriever import create_retriever_tool

from rich.console import Console
from rich.prompt import Prompt
from rich.text import Text

from const_prompt import SYSTEM_PROMPT

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Parse command line arguments
parser = argparse.ArgumentParser(description="Chatbot parameters")
parser.add_argument("--chat_model", type=str, default="qwen3:8b")
parser.add_argument("--query_model", type=str, default="")
parser.add_argument("--docs_dir", type=str, default="docs")
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
chat_model = ChatOllama(
    model=args.chat_model,
    num_predict=args.max_tokens,
    temperature=args.temperature,
)

query_model = ChatOllama(
    model=args.query_model,
    num_predict=200,
    temperature=0.7,
)

# Define the chat prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Define vector store for retrieval
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = InMemoryVectorStore(embeddings)
retriever = vector_store.as_retriever()
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_pdf_content",
    "Search and return information from the provided PDF documents.",
)

# PDF document loader and chunking
def load_chunked_documents() -> list[Document]:
    documents = []
    files = list(Path(args.docs_dir).glob("*.pdf"))
    for file_path in files:
        loader = PyPDFLoader(file_path, mode='page')
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=50
    )
    documents = text_splitter.split_documents(documents)
    print(f"[INFO][load_chunked_documents] Loaded {len(files)} files from {args.docs_dir}. Converted to {len(documents)} documents.")
    return documents

def generate_query_or_respond(state: MessagesState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    prompt = prompt_template.invoke(state["messages"])
    response = (
        chat_model
        .bind_tools([retriever_tool]).invoke(prompt)
    )
    return {"messages": response}

def call_model(state: MessagesState):
    prompt = prompt_template.invoke(state["messages"])
    response = chat_model.invoke(prompt)
    return {"messages": response}

# Define the agent graph; compile agent with memory for chat history
workflow = StateGraph(state_schema=MessagesState)
workflow.add_node(generate_query_or_respond)
workflow.add_edge(START, "generate_query_or_respond")
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
    documents = load_chunked_documents()
    vector_store.add_documents(documents)

    console = Console()
    try: main(console)
    except KeyboardInterrupt:
        console.print("\n[bold green]>[/bold green]: [green]Good-bye.[/green]")