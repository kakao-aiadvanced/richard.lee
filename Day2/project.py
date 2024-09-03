import bs4
import os
import getpass
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ["OPENAI_API_KEY"] = getpass.getpass()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=(urls),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(model = "text-embedding-3-small"))

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

user_query = "agent memory"
chunks = rag_chain.invoke(user_query).split(".")[:-1]

parser = JsonOutputParser()

prompt = PromptTemplate(
    template = "Answer yes or no about the relevance of query and each chunks.\n{format_instructions},\nquery: {query}, chunks\: {chunks}",
    input_variables = ["query", "chunks"],
    partial_variables = {"format_instructions": parser.get_format_instructions()},
)

chain = prompt | llm | parser

result = chain.invoke({"query": user_query, "chunks": chunks})

print(result)

prompt = PromptTemplate(
    template = "Answer yes or no about the hallucination of query and each chunks.\n{format_instructions},\nquery: {query}, chunks\: {chunks}",
    input_variables = ["query", "chunks"],
    partial_variables = {"format_instructions": parser.get_format_instructions()},
)

chain = prompt | llm | parser

result = chain.invoke({"query": user_query, "chunks": chunks})

print(result)
