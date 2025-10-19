import streamlit as st
from dotenv import load_dotenv
import os
import time
from document_loader import load_documents
from vector_store import VectorStoreManager

load_dotenv()

APP_TITLE = os.getenv("APP_TITLE", "문서 기반 질의응답 챗봇")

st.set_page_config(page_title=APP_TITLE, page_icon="🤖", layout="wide")

header_container = st.container()

with header_container:
    st.markdown(
        f"""
        <style>
        .stContainer {{
            position: sticky;
            top: 0;
            background-color: white;
            z-index: 999;
            padding-top: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #f0f2f6;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title(f"🤖 {APP_TITLE}")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.welcome_shown = False

if "vectorstore_manager" not in st.session_state:
    st.session_state.vectorstore_manager = VectorStoreManager()

if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

with st.sidebar:
    st.title(f"🤖 {APP_TITLE}")
    st.divider()
    st.header("설정")

    model_type = st.selectbox("모델 유형", ["OpenAI", "Ollama"])

    if model_type == "OpenAI":
        openai_api_key = st.text_input(
            "OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", "")
        )
        model_name = st.selectbox("모델 선택", ["gpt-5", "gpt-5-mini"])
    else:
        openai_api_key = None
        model_name = st.text_input("Ollama 모델 이름", value="llama2")
        st.info("Ollama가 로컬에서 실행 중인지 확인하세요.")

    st.divider()

    st.header("문서 업로드")
    uploaded_files = st.file_uploader(
        "PDF, PowerPoint 또는 Markdown 파일을 업로드하세요",
        type=["pdf", "pptx", "md", "markdown"],
        accept_multiple_files=True,
    )

    if st.button("문서 처리", type="primary"):
        if uploaded_files:
            if model_type == "OpenAI" and not openai_api_key:
                st.error("OpenAI API 키를 먼저 입력해주세요! 🔑")
            else:
                with st.spinner("문서를 처리하는 중..."):
                    try:
                        documents = load_documents(uploaded_files)

                        if documents:
                            st.session_state.vectorstore_manager.create_vectorstore(
                                documents, openai_api_key
                            )
                            st.session_state.documents_loaded = True
                            st.session_state.messages = []
                            st.success(
                                f"좋아요! 📚 {len(documents)}개의 문서 청크가 준비되었어요. 이제 질문해주세요!"
                            )
                        else:
                            st.error("문서를 읽을 수 없어요. 파일 형식을 확인해주세요! 📄")
                    except Exception as e:
                        st.error(f"오류가 발생했어요: {str(e)}")
        else:
            st.warning("파일을 먼저 업로드해주세요! 📎")

    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.rerun()

if len(st.session_state.messages) == 0 and st.session_state.documents_loaded:
    welcome_msg = "안녕하세요! 👋 문서가 준비되었습니다. 궁금한 점을 편하게 물어보세요!"
    with st.chat_message("assistant"):
        st.markdown(welcome_msg)
    st.session_state.messages.append({"role": "assistant", "content": welcome_msg})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("질문을 입력하세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not st.session_state.documents_loaded:
            response = "앗, 아직 문서가 업로드되지 않았어요! 😊 왼쪽 사이드바에서 문서를 먼저 업로드해주시면 질문에 답변해드릴게요."
            st.markdown(response)
        else:
            if model_type == "OpenAI" and not openai_api_key:
                response = "OpenAI API 키가 필요해요. 왼쪽 사이드바에서 API 키를 입력해주세요! 🔑"
                st.markdown(response)
            else:
                try:
                    qa_chain = st.session_state.vectorstore_manager.get_qa_chain(
                        model_name, model_type, openai_api_key
                    )

                    if qa_chain:
                        start_time = time.time()
                        
                        with st.status("답변을 생성하고 있어요... 🤔", expanded=True) as status:
                            st.write("📄 문서 검색 중...")
                            search_start = time.time()
                            
                            result = qa_chain.invoke({"query": prompt})
                            response = result["result"]
                            
                            end_time = time.time()
                            total_time = end_time - start_time
                            
                            status.update(
                                label=f"답변 완료! ✨ (소요시간: {total_time:.2f}초)", 
                                state="complete", 
                                expanded=False
                            )

                        if (
                            not response
                            or response.strip() == ""
                            or "NOT_FOUND" in response
                        ):
                            response = "죄송해요, 업로드하신 문서에서 관련 정보를 찾지 못했어요. 😅 다른 질문을 해보시거나, 더 자세한 정보가 필요하시면 담당자 윤덕열님께 연락해주세요!"
                            st.markdown(response)
                        else:
                            st.markdown(response)
                            
                            st.caption(f"⏱️ 처리 시간: {total_time:.2f}초")

                            if result.get("source_documents"):
                                with st.expander("참조 문서"):
                                    for i, doc in enumerate(result["source_documents"]):
                                        source_file = doc.metadata.get("source", "알 수 없음")
                                        page = doc.metadata.get("page", "")
                                        st.markdown(f"**출처 {i+1}:** {source_file}" + (f" (페이지 {page + 1})" if page != "" else ""))
                                        st.markdown(doc.page_content[:300] + "...")
                    else:
                        response = "문서 처리에 문제가 있는 것 같아요. 문서를 다시 업로드해보시겠어요?"
                        st.markdown(response)
                except Exception as e:
                    response = f"앗, 처리 중에 문제가 발생했어요. 😅 질문을 다시 해보시거나 담당자 윤덕열님께 문의해주세요!"
                    st.markdown(response)
                    st.error(f"오류: {str(e)}")

        st.session_state.messages.append({"role": "assistant", "content": response})
