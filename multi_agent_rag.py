from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import StackExchangeTool
from langchain_community.utilities import StackExchangeAPIWrapper

api_wrapper = StackExchangeAPIWrapper(site="stackoverflow",top_k_results=5)
stack = StackExchangeTool(api_wrapper = api_wrapper)
# response = stack.invoke("How does LangChain implement a mechanism where a particular tool must be used at the first call step before generating an answer?")
# print(response)
api_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper = api_wrapper)

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain

loader = WebBaseLoader("https://docs.smith.langchain.com/")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
embeddings = OllamaEmbeddings(model="gemma:2b")
vector_store = FAISS.from_documents(documents, embeddings)
retriever = vector_store.as_retriever()

from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(retriever, "langsmith_search", "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!")


from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper

api_wrapper = ArxivAPIWrapper(top_k_results=3)
arxiv = ArxivQueryRun(api_wrapper = api_wrapper)

tools=[wiki,arxiv,retriever_tool,stack]
from langchain_groq import ChatGroq
import os
os.environ["LANGCHAIN_API_KEY"]="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"
OPENAI_API_KEY="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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple RAG"

llm = ChatGroq(groq_api_key = "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",model_name = "llama3-8b-8192")


from langchain import hub
prompt = hub.pull("hwchase17/openai-functions-agent")

from langchain.agents import create_openai_functions_agent

agent = create_openai_functions_agent(llm, tools, prompt)
#%%
##agent executor
from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent,tools = tools,verbose=True)

response = agent_executor.invoke({"input":"How does LangChain implement a mechanism where a particular tool must be used at the first call step before generating an answer?"})

print(response["output"])q