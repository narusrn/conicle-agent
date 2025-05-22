import json
import sys
from datetime import datetime
from typing import Literal

import pandas as pd
from langchain.tools import Tool
from langchain_chroma import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, convert_to_messages
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def prepare_documents_from_excel(file_path: str, sheet_name: str) -> list[Document]:
    df = pd.read_excel(file_path, sheet_name=sheet_name)  # อ่าน sheet 'competencies', 'roles'
    
    # สร้าง Document โดยใช้ competency name + description รวมกันเป็น page_content
    documents = []
    for _, row in df.iterrows():
        if sheet_name == "competencies":
            content = f"{row['competency']}\n\n{row['description']}"
            metadata = {
                "competency": row["competency"]
            }
        elif sheet_name == "roles":
            content = f"{row['role']}\n\n{row['description']}"
            metadata = {
                "role": row["role"]
            }
        documents.append(Document(page_content=content, metadata=metadata))

    # ตัดข้อความเป็น chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    return splitter.split_documents(documents)


def get_retriever_from_excel() :
   
    documents1 = prepare_documents_from_excel("openai_result.xlsx", "competencies")
    documents2 = prepare_documents_from_excel("openai_result.xlsx", "roles")
    documents = documents1 + documents2
    vectorstore = Chroma(
        embedding_function=OpenAIEmbeddings(),
        persist_directory="chroma_db",
        collection_name="competency"
    )
    vectorstore = Chroma.from_documents(documents, embedding=OpenAIEmbeddings())

    return vectorstore.as_retriever(search_kwargs={"k": 10})


def document_search(question: str) -> dict:
    """
    Retrieve course documents
    """

    # Retrieval
    retriever = get_retriever_from_excel()
    print("QUESTION: ", question)
    documents = retriever.invoke(question)

    return {"documents": documents, "question": question}

def initialize_default_tools() : 

    # สร้าง DuckDuckGo search instance
    web_search_engine = DuckDuckGoSearchRun()

    return [
        # Tool.from_function(
        #     func=web_search_engine.run,
        #     name="web_search_engine",
        #     description="ใช้ค้นหา course จาก internet อย่าลืมแนบอ้างอิงด้วย"
        # ),
        Tool.from_function(
            func=document_search,
            name="document_search",
            description="ใช้สำหรับค้นหาข้อมูล course จากฐานข้อมูลภายใน",
        ),
    ]
