from typing import List, Optional
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import os


class VectorStoreManager:
    def __init__(self):
        self.vectorstore: Optional[Chroma] = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=400,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def create_vectorstore(self, documents: List[Document], openai_api_key: Optional[str]):
        if not documents:
            return None
        
        splits = self.text_splitter.split_documents(documents)
        
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings
        )
        
        return self.vectorstore
    
    def get_retriever(self):
        if not self.vectorstore:
            return None
        
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 8
            }
        )
        
        return retriever
    
    def get_llm(self, model_name: str, model_type: str, openai_api_key: Optional[str] = None, streaming: bool = False):
        if model_type == "OpenAI":
            if openai_api_key:
                os.environ["OPENAI_API_KEY"] = openai_api_key
            return ChatOpenAI(
                model=model_name,
                temperature=0.5,
                streaming=streaming
            )
        else:
            return Ollama(model=model_name, temperature=0.5)
    
    def get_qa_chain(self, model_name: str, model_type: str, openai_api_key: Optional[str] = None):
        if not self.vectorstore:
            return None
        
        llm = self.get_llm(model_name, model_type, openai_api_key, streaming=False)
        
        prompt_template = """당신은 친절하고 전문적인 AI 어시스턴트입니다. 
사용자와 자연스럽게 대화하듯이 답변해주세요.

제공된 문서 내용을 참고하여 질문에 답변하되, 다음 사항을 지켜주세요:
- 친근하고 부드러운 말투를 사용하세요
- 문서에 있는 정보를 바탕으로 정확하고 구체적으로 설명해주세요
- 문서의 여러 부분에 관련 정보가 있다면 모두 종합해서 답변하세요
- 문서에 관련 정보가 전혀 없다면 "NOT_FOUND"라고만 답변하세요
- 확실하지 않은 내용은 추측하지 말고, 문서에 명시된 내용만 답변하세요

문서 내용:
{context}

질문: {question}

답변:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        retriever = self.get_retriever()
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        return qa_chain
