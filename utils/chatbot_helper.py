from typing import List, Dict, Tuple, Any
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_tavily import TavilySearch

def build_vector_store(slides_data: List[dict], all_contents: List[str], all_scripts: List[str], openai_key: str) -> FAISS:
    """슬라이드 정보와 요약본, 강의 스크립트 대본을 읽어 메모리 기반 FAISS 벡터 스토어를 빌드합니다."""
    documents = []
    for idx, slide in enumerate(slides_data):
        slide_summary = all_contents[idx] if idx < len(all_contents) else ""
        slide_script = all_scripts[idx] if idx < len(all_scripts) else ""
        
        content_block = (
            f"슬라이드 번호: {idx + 1}\n"
            f"슬라이드 제목: {slide.get('title', '없음')}\n"
            f"원본 텍스트 내용: {' '.join(slide.get('texts', []))}\n"
            f"요약 정보: {slide_summary}\n"
            f"매칭된 발표 대본: {slide_script}"
        )
        documents.append(Document(page_content=content_block, metadata={"slide": idx + 1}))
        
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

def rag_chatbot_query(
    user_query: str, 
    vector_store: FAISS, 
    openai_key: str, 
    tavily_key: str = ""
) -> Tuple[str, bool, Any]:
    """
    RAG 검색을 실행하여 질문이 슬라이드 내부 내용인지 판단하고 답변을 도출합니다.
    Tavily 연동 플래그 및 검색 데이터도 함께 반환합니다.
    """
    # 1. FAISS 유사도 점수와 함께 2개 도큐먼트 검색 (L2 distance 방식: 작을수록 우수)
    search_results = vector_store.similarity_search_with_score(user_query, k=2)
    
    context_docs = []
    min_score = 999.0
    for doc, score in search_results:
        context_docs.append(doc.page_content)
        if score < min_score:
            min_score = score
            
    context = "\n---\n".join(context_docs)
    
    # 임계값 (FAISS L2 score 기준 약 0.95 초과 시 관련 없는 내용으로 분류)
    is_relevant = min_score < 0.95
    
    # 2. LLM 질의 준비
    sys_instruction = """
    역할: 당신은 제공받은 PPT 슬라이드 지식 데이터베이스(Context)에만 철저히 기반하여 답변하는 교육 비서입니다.
    
    규칙:
    - 반드시 제공된 [Context] 정보에서만 정확하게 관련 팩트를 찾아 친절하고 정중한 한글로 대답하세요.
    - 만약 [Context] 정보에 질문에 대응하는 핵심 사실이 전혀 포함되어 있지 않거나, 엉뚱한 정보만 있다면 애써 억지로 지어내지 마세요.
    - 데이터베이스에 매칭되는 내용이 없을 경우에는 반드시 정확하게 다음 문장을 대답의 처음에 표출해야 합니다: "죄송합니다만, 질문하신 내용은 PPT 슬라이드에 수록되어 있지 않은 내용입니다."
    """
    
    prompt_content = f"[Context]\n{context}\n\n[사용자 질문]\n{user_query}"
    
    chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, openai_api_key=openai_key)
    response = chat.invoke([
        SystemMessage(content=sys_instruction),
        HumanMessage(content=prompt_content)
    ])
    
    ans_content = response.content
    
    # RAG에 정보가 없는지 판별
    needs_web_search = "수록되어 있지 않은 내용" in ans_content or not is_relevant
    
    return ans_content, needs_web_search, None
