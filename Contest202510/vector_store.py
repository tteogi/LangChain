from typing import List, Optional
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
import os
import re


class VectorStoreManager:
    def __init__(self):
        self.vectorstore: Optional[Chroma] = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", "。", ". ", "? ", "! ", " ", ""],
        )

    def _is_comparison_query(self, query: str) -> bool:
        """
        질문이 비교/분석 질문인지 판단
        비교 질문: 여러 옵션을 비교해야 하는 질문
        단순 질문: 특정 항목에 대한 직접적인 질문
        """
        comparison_keywords = [
            "가장", "최고", "최저", "최선", "최악", "제일",
            "비교", "차이", "다른", "vs", "어느", "어떤",
            "전부", "모든", "전체", "모두", "다",
            "추천", "좋은", "나은", "유리한", "저렴한", "비싼"
        ]

        query_lower = query.lower()
        return any(keyword in query_lower for keyword in comparison_keywords)

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

    def get_retriever(self, query: str = ""):
        """
        질문 유형에 따라 최적화된 retriever 반환
        - 단순 질문: k=5 (빠름)
        - 비교 질문: k=18 (정확함)
        """
        if not self.vectorstore:
            return None

        is_comparison = self._is_comparison_query(query)

        if is_comparison:
            # 비교/분석 질문: 더 많은 문서 검색
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 25,         # 18개 반환 (속도와 정확도 균형)
                    "fetch_k": 40,   # 40개 후보 검색
                    "lambda_mult": 0.9  # 유사도 우선
                }
            )
        else:
            # 단순 질문: 적은 문서로 빠르게
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 5  # 5개만 검색 (빠름)
                }
            )

        return retriever

    def get_llm(
        self,
        model_name: str,
        model_type: str,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        streaming: bool = False,
    ):
        if model_type == "OpenAI":
            if openai_api_key:
                os.environ["OPENAI_API_KEY"] = openai_api_key
            return ChatOpenAI(model=model_name, temperature=0.2, streaming=streaming)
        elif model_type == "Claude":
            if anthropic_api_key:
                os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
            return ChatAnthropic(model=model_name, temperature=0.2, streaming=streaming)
        else:
            return Ollama(model=model_name, temperature=0.5)

    def get_qa_chain(
        self,
        model_name: str,
        model_type: str,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        query: str = ""
    ):
        """
        QA 체인 생성
        query를 받아서 질문 유형에 맞는 retriever 사용
        """
        if not self.vectorstore:
            return None

        llm = self.get_llm(model_name, model_type, openai_api_key, anthropic_api_key, streaming=False)

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

**비교/분석 질문 처리 방법:**
- "가장 저렴한", "가장 빠른", "어떤 것이 좋은" 등 비교 질문의 경우:
  1) 제공된 모든 관련 옵션을 빠짐없이 확인하세요
  2) 각 옵션의 수수료/조건을 명확히 비교하세요
  3) **중요**: 동일한 수수료를 가진 방법이 여러 개 있다면 모두 나열하세요
     - 예: "암호화폐 3%, 페이코인 3%" (둘 다 3%이지만 모두 언급)
     - 예: "전용계좌 1,000원, 케이뱅크 페이 1,000원" (둘 다 나열)
  4) 고정 금액(예: 1,000원)과 비율(예: 8%)이 섞여 있다면:
     - 충전 금액에 따라 달라진다는 점을 설명하세요
     - 구체적인 기준 금액을 계산하여 제시하세요 (예: "33,333원 기준으로...")
  5) 가능한 한 구조화된 형식으로 답변하세요

문서 내용:
{context}

질문: {question}

답변:"""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        # query에 따라 최적화된 retriever 사용
        retriever = self.get_retriever(query)
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
