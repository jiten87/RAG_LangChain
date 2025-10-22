from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

import chromadb

persist_directory= "./chroma_langchain_vectordb"
embedding = OllamaEmbeddings(model= "mistral")
# Create the Chroma DB
vectordb = Chroma(
    collection_name="my_local_vectorDB",
    embedding_function=embedding,
    persist_directory=persist_directory,
)


if __name__ == "__main__":
    #un-comment the below block while ingestion the info to vector DB. After that you can comment it.
    '''
    print("inside ingestion")
    loader = TextLoader("mediumblog.txt", encoding="utf-8")
    document = loader.load()

    print("splitting.....")
    text_spiltter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_spiltter.split_documents(documents=document)
    print(f"created {len(texts)} chunks ")

    embedding = OllamaEmbeddings(model= "mistral")

    index_name="langchain-test-index"

   

    #vectordb.embeddings = embedding
    vectordb.add_documents(texts)
    print(" ingestion done")
    '''

    print("Retrival part start")
    
    llm = ChatOllama(model= "mistral")

    query = "what is the pincone in ML world"
    chain = PromptTemplate.from_template(query) | llm
    result = chain.invoke(input={})
    print(result.content)

    template ="""
    Answer any use questions based solely on the context below:

    <context>
    {context}
    </context>
    {input}
    """
    # Create PromptTemplate object
    prompt = PromptTemplate(
    template=template,
    input_variables=["context", "input"]
    )

    combine_doc_chain = create_stuff_documents_chain(llm,prompt=prompt)
    retrival_chain = create_retrieval_chain(retriever=vectordb.as_retriever(search_type="similarity"),combine_docs_chain=combine_doc_chain)

    result = retrival_chain.invoke(input={"input": query})
    print("final answer after retrival:")
    print(result["answer"])
