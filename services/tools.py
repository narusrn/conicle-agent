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


def prepare_documents_from_excel(file_path: str, sheet_name: str, group_size: int = 5) -> list[Document]:
    df = pd.read_csv(file_path)
    documents = []

    # แบ่งเป็นกลุ่ม ๆ ละ group_size
    for i in range(0, len(df), group_size):
        chunk = df.iloc[i:i+group_size]

        content_lines = []
        metadata = {}

        for _, row in chunk.iterrows():
            if sheet_name == "competencies":
                content_lines.append(f"{row['competency']}\n{row['description']}")
            elif sheet_name == "roles":
                content_lines.append(f"{row['role']}\n{row['description']}")

        content = "\n\n---\n\n".join(content_lines)  # คั่นด้วยเส้นแบ่ง
        documents.append(Document(page_content=content, metadata={"source_sheet": sheet_name}))

    # ตัด chunk ถ้าจำเป็น
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)

def get_retriever_from_excel() :
   
    documents1 = prepare_documents_from_excel("https://docs.google.com/spreadsheets/d/1_2BqDT6ipbu_nOguPlXMp0xO0vyBpvXyTTW-0T-_yPY/export?format=csv&gid=0", "competencies")
    documents2 = prepare_documents_from_excel("https://docs.google.com/spreadsheets/d/1_2BqDT6ipbu_nOguPlXMp0xO0vyBpvXyTTW-0T-_yPY/export?format=csv&gid=164019563", "roles")
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
