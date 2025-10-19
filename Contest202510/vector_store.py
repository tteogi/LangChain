from typing import List, Optional
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
import os


class VectorStoreManager:
    def __init__(self):
        self.vectorstore: Optional[Chroma] = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=250,
            length_function=len,
            separators=["\n[", "\n\n", "\n", "? ", ". ", " ", ""],
        )

    def create_vectorstore(
        self, documents: List[Document], openai_api_key: Optional[str]
    ):
        if not documents:
            return None

        splits = self.text_splitter.split_documents(documents)

        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

        return self.vectorstore

    def get_retriever(self):
        if not self.vectorstore:
            return None

        retriever = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        return retriever

    def get_llm(
        self,
        model_name: str,
        model_type: str,
        openai_api_key: Optional[str] = None,
        streaming: bool = False,
    ):
        if model_type == "OpenAI":
            if openai_api_key:
                os.environ["OPENAI_API_KEY"] = openai_api_key
            return ChatOpenAI(model=model_name, temperature=0.2, streaming=streaming)
        else:
            return Ollama(model=model_name, temperature=0.5)

    def get_qa_chain(
        self, model_name: str, model_type: str, openai_api_key: Optional[str] = None
    ):
        if not self.vectorstore:
            return None

        llm = self.get_llm(model_name, model_type, openai_api_key, streaming=False)

        prompt_template = """당신은 친절하고 전문적인 고객센터 상담원입니다.
제공된 FAQ 문서를 기반으로 고객의 질문에 정확하게 답변해주세요.

답변 규칙:
1. 문서에 있는 정보만 사용하여 정확하고 구체적으로 답변하세요
2. 수수료, 금액, 시간 등 숫자 정보는 문서에 나온 그대로 정확히 전달하세요
3. 제공된 여러 문서 조각에 관련 정보가 있다면 모두 확인하고 종합해서 답변하세요
4. 질문 표현이 달라도 같은 내용이면 답변하세요
5. 문서에 관련 정보가 전혀 없으면 "NOT_FOUND"라고만 답변하세요
6. 추측하지 말고 문서에 명시된 내용만 답변하세요
7. 친근하고 자연스러운 말투를 사용하세요

문서 내용:
{context}

질문: {question}

답변:"""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        retriever = self.get_retriever()
        if not retriever:
            return None

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT},
        )

        return qa_chain
