
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.memory import ConversationBufferMemory
from langchain.tools import tool
from API_key import GEMINI_API_KEY
import os
from tavily import TavilyClient
from API_key import TAVILY_API_KEY

# Math tools

@tool
def add(a:int, b:int) ->int:
    """Addition of two numbers"""
    return a+b

@tool
def sub(a:int, b:int) ->int:
    """subtraction of two numbers"""
    return a-b

@tool
def mul(a:int, b:int) ->int:
    """Multiplication of two numbers"""
    return a*b

@tool
def div(a:int, b:int) ->int:
    """division of two numbers"""
    return a/b


# LLM model 
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature = 0
)

# RAG
loader = PyPDFLoader("abc_company_policy.pdf")
document = loader.load()
print("Length of document: ", len(document))
print("\ndocumnets: ", document[0].page_content[:500])

#Splitting and chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 60
)

chunk = splitter.split_documents(
    document
)

print("\nLength of chunks: ", len(chunk))

embedding = HuggingFaceEmbeddings(
    model_name = "all-MiniLM-L6-V2"
)

vector_store = Chroma.from_documents(
    chunk,
    embedding,
    persist_directory="./ABC_company_database"
)

retriever = vector_store.as_retriever(
    search_kwargs = {"k":2}
)

retriever.invoke("what is the working Hour of ABC company?")


#RAG tool
@tool
def search_doc(query:str) ->str:
    """
    Search the local company knowledge base.
    
    Use this tool FIRST for:
    - ABC company
    - Company information
    - Internal documents
    - Terms and policies
    
    This tool contains official internal data.
    """
    doc = retriever.invoke(query)
    context = "\n\n".join(docs.page_content for docs in doc)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Use this context to answer the following question"),
        ("system", "context:{context}"),
        ("human", "Question: {query}")
    ])

    formatted_prompt = prompt.format(
        context = context,
        query = query
    )

    response = llm.invoke(formatted_prompt)

    return response.content

# Web search tool
from tavily import TavilyClient
from API_key import TAVILY_API_KEY

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
client = TavilyClient()

@tool
def tavily_search(query:str) ->str:
    """Search the internet for accurate and up to date answer"""
    response = client.search(
        query = query,
        search_depth="advanced",
        max_results = 3
    )

    results = ""

    for r in response["results"]:
        results += r["title"]+"\n"
        results += r["content"]+"\n\n"

    return results

# Memory
memory = ConversationBufferMemory(
    memory_key = "chat_history"
)
print("Memory length: ", len(memory.chat_memory.messages))

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistance. use these tools when it needed."),
    ("human", "{input}"),
    ("system", "chat_history: {chat_history}"),
    ("assistant", "{agent_scratchpad}")
])

#Tools
tools = [add, sub, mul, div , search_doc, tavily_search]
print("\nTotal number of tools: ", len(tools))
print("\n", tools)

#Agent
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

agent = create_tool_calling_agent(
    llm = llm,
    tools = tools,
    prompt= prompt
)

agent_executor = AgentExecutor(
    agent = agent,
    tools = tools,
    memory = memory,
    verbose = True
)

while True:

    user_input = input("\n\nEnter your query here...")

    if user_input.lower == "exit":
        break
    
    response = agent_executor.invoke({"input" : user_input})
    print(f"Results:\n" , response["output"])