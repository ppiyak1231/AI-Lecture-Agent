import os
import shutil
import tempfile
from pathlib import Path

import streamlit as st
import openai
from openai import OpenAI

# RAG 챗봇용 벡터 DB 빌더 임포트
from utils.chatbot_helper import build_vector_store

# LangGraph 컴파일된 Supervisor 그래프 임포트
from pipeline.graph import build_lecture_supervisor_graph, build_chat_supervisor_graph

# -----------------------------------------------------------------------------
# 1. UI 설정 및 프리미엄 HSL 테마 디자인 (Custom CSS)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI 강사 Agent v2.0",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Noto+Sans+KR:wght@300;400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', 'Noto Sans KR', sans-serif;
    background-color: #FAF8F5;
}

/* 사이드바 스타일링 */
[data-testid="stSidebar"] {
    background-color: #F3EFE9 !important;
    border-right: 1px solid #E6DFD5;
}

[data-testid="stSidebar"] .stMarkdown h2 {
    color: #4A3E3D;
    font-weight: 600;
}

/* 카드 UI */
.card {
    background-color: #FFFFFF;
    border: 1px solid #EBE6DD;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 4px 12px rgba(139, 94, 60, 0.03);
    transition: all 0.3s ease;
}

.card:hover {
    box-shadow: 0 8px 24px rgba(139, 94, 60, 0.06);
    transform: translateY(-2px);
}

/* 에러 및 경고 커스텀 */
.custom-warning {
    background-color: #FDF5EE;
    border-left: 4px solid #C4A882;
    padding: 16px;
    border-radius: 8px;
    color: #7A4F2D;
    margin-bottom: 15px;
}

/* 진행 상태 표시 바 */
.step-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background-color: #FFFFFF;
    border: 1px solid #EBE6DD;
    border-radius: 12px;
    padding: 16px 24px;
    margin-bottom: 24px;
}

.step-item {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 14px;
    color: #B09E93;
}

.step-item.active {
    color: #8B5E3C;
    font-weight: 600;
}

.step-dot {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    border: 2px solid #D9CDC4;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    color: #B09E93;
    font-weight: 600;
}

.step-item.active .step-dot {
    background-color: #8B5E3C;
    border-color: #8B5E3C;
    color: #FFFFFF;
}

.step-item.done .step-dot {
    background-color: #C4A882;
    border-color: #C4A882;
    color: #FFFFFF;
}

.step-line {
    flex-grow: 1;
    height: 2px;
    background-color: #E8E0D6;
    margin: 0 16px;
}

/* 다운로드 버튼 스타일링 */
div.stDownloadButton > button {
    background-color: #8B5E3C !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 10px rgba(139, 94, 60, 0.15) !important;
}

div.stDownloadButton > button:hover {
    background-color: #7A5032 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 14px rgba(139, 94, 60, 0.2) !important;
}

/* 헤더 & 로고 스타일 */
.main-title {
    font-size: 32px;
    font-weight: 700;
    color: #2C2420;
    margin-bottom: 8px;
}

.sub-title {
    font-size: 15px;
    color: #8C7D72;
    margin-bottom: 30px;
}

/* 실시간 디버그 로그용 스타일 */
.agent-log {
    background-color: #1E1E1E;
    color: #00FF00;
    font-family: 'Courier New', Courier, monospace;
    padding: 12px;
    border-radius: 8px;
    font-size: 13px;
    margin-top: 5px;
    margin-bottom: 15px;
    max-height: 180px;
    overflow-y: auto;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. 메인 타이틀 및 소개 정보
# -----------------------------------------------------------------------------
st.markdown("<div class='main-title'>🎓 AI 강사 Agent v2.0</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Supervisor Multi-Agent 아키텍처에 의해 각 전문 에이전트들이 유기적으로 협업하여 명품 강의 영상과 챗봇 RAG 서비스를 제공합니다.</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. 사이드바 UI 설정 & API KEY 관리
# -----------------------------------------------------------------------------
st.sidebar.markdown("## ⚙️ 설정 & API KEY")

openai_key = st.sidebar.text_input("OpenAI API Key", type="password", help="OpenAI API 키를 입력해주세요. (RAG 및 영상 제작용)")
tavily_key = st.sidebar.text_input("Tavily API Key (옵션)", type="password", help="Tavily API 키를 입력하면 챗봇이 PPT 외의 모르는 정보를 웹 실시간 검색하여 보충해 줍니다.")

st.sidebar.markdown("---")
st.sidebar.markdown("## 🎙️ 강의 성우 스타일 설정")
persona = st.sidebar.selectbox("강사 스타일 선택", [
    "친절한 AI 강사 (alloy/부드러움)", 
    "10년차 MLOps 현업 전문가 (onyx/자신감)", 
    "대학 컴퓨터공학과 교수님 (nova/격식)", 
    "동아리 IT 개발자 친한 선배 (shimmer/캐주얼)"
])
level = st.sidebar.radio("청중(교육 대상) 레벨", ["비전공자", "전공자"], index=0)

# 세션 상태 변수 초기화
if "processed" not in st.session_state:
    st.session_state["processed"] = False
if "final_video_path" not in st.session_state:
    st.session_state["final_video_path"] = None
if "final_script_text" not in st.session_state:
    st.session_state["final_script_text"] = None
if "ppt_structured_data" not in st.session_state:
    st.session_state["ppt_structured_data"] = None
if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "progress_status" not in st.session_state:
    st.session_state["progress_status"] = ""

# -----------------------------------------------------------------------------
# 4. 메인 화면 탭 구성
# -----------------------------------------------------------------------------
tab1, tab2 = st.tabs(["🚀 AI 강의 영상 & 대본 생성기", "💬 PPT 내용 AI 질의응답 (RAG 챗봇)"])

# -----------------------------------------------------------------------------
# TAB 1: 강의 생성 및 렌더링 화면
# -----------------------------------------------------------------------------
with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📥 프레젠테이션 파일 업로드")
    uploaded_file = st.file_uploader("강의 영상으로 변환할 PPTX 파일을 선택하세요.", type=["pptx"])
    st.markdown("</div>", unsafe_allow_html=True)
    
    if st.button("✨ AI 강의 영상 제작 시작", use_container_width=True):
        if not openai_key:
            st.error("⚠️ OpenAI API Key가 입력되지 않았습니다! 왼쪽 설정 창에서 먼저 입력해 주세요.")
        elif not uploaded_file:
            st.warning("⚠️ 업로드된 PPTX 파일이 존재하지 않습니다. 먼저 PPT 파일을 업로드해 주세요.")
        else:
            # 1. API Key 검증 (유효성 검사)
            st.session_state["progress_status"] = "🔑 OpenAI API Key의 신뢰도를 검증하는 중..."
            try:
                validation_client = OpenAI(api_key=openai_key)
                validation_client.models.list()
            except openai.AuthenticationError:
                st.error("🚨 입력하신 OpenAI API Key가 올바르지 않거나 활성화되지 않은 상태입니다. 키의 철자나 만료 여부를 꼭 다시 확인해 주세요!")
                st.stop()
            except Exception as e:
                st.error(f"🚨 OpenAI 서버와의 네트워크 연결 중 에러가 발생했습니다. (상세: {str(e)})")
                st.stop()
                
            # 2. 파이프라인 가동
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    temp_work_dir = Path(tmpdir) / "work"
                    temp_work_dir.mkdir(parents=True, exist_ok=True)
                    
                    temp_pptx_path = temp_work_dir / "uploaded_input.pptx"
                    with open(temp_pptx_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                        
                    # 초기 상태 정의
                    initial_state = {
                        "pptx_path": str(temp_pptx_path),
                        "work_dir": str(temp_work_dir),
                        "prompt": {
                            "persona": persona,
                            "voice": "alloy",
                            "tone": "부드럽고 명확하게 전달해 주세요.",
                            "level": level,
                            "style": "핵심 키워드 설명 및 명료하고 실용적인 예제 중심"
                        },
                        "openai_key": openai_key,
                        "tavily_key": tavily_key if tavily_key else "",
                        "all_contents": [],
                        "all_scripts": [],
                        "all_audios": [],
                        "all_videos": [],
                        "slide_index": 0
                    }
                    
                    # UI 피드백 영역 구축
                    progress_bar = st.progress(0.0)
                    status_area = st.empty()
                    log_console = st.empty()
                    
                    # 3. Lecture Supervisor Graph 선언 및 스트리밍 구동
                    lecture_graph = build_lecture_supervisor_graph()
                    
                    log_history = []
                    last_final_state = initial_state
                    
                    # Multi-Agent 루프이므로 recursion_limit을 150으로 여유 있게 설정
                    for event in lecture_graph.stream(initial_state, {"recursion_limit": 150}):
                        for node_name, updated_state in event.items():
                            last_final_state = updated_state
                            
                            # 1) 노드별 진행도 계산
                            n_slides = updated_state.get("n_slides", 1)
                            idx = updated_state.get("slide_index", 0)
                            
                            # UI 상태 업데이트
                            progress_val = 0.05
                            if n_slides > 0:
                                progress_val = min(0.05 + (idx / n_slides) * 0.9, 0.98)
                            progress_bar.progress(progress_val)
                            
                            # 2) 실시간 디버그 로그 콘솔 갱신
                            log_msg = f">> [Agent Activated] '{node_name}' 실행 중..."
                            if node_name == "supervisor":
                                next_w = updated_state.get("next_worker", "FINISH")
                                log_msg = f">> 🧠 [Lecture Supervisor] 분석 결과 다음 작업자로 '{next_w}' 지명!"
                            elif node_name == "parser":
                                log_msg = f">> 📄 [Parser Worker] PPTX 파싱 & {updated_state.get('n_slides', 0)}개 슬라이드 스냅샷 생성 완료."
                            elif node_name == "summarizer":
                                log_msg = f">> ✍️ [Summarizer Worker] 슬라이드 {idx+1} 지식 요약 완료."
                            elif node_name == "script_writer":
                                log_msg = f">> 📝 [ScriptWriter Worker] 슬라이드 {idx+1} 스피치 대본 완성."
                            elif node_name == "media_creator":
                                log_msg = f">> 🎬 [MediaCreator Worker] 슬라이드 {idx+1} TTS 음성 합성 및 비디오 인코딩 성공."
                            elif node_name == "accumulator":
                                log_msg = f">> 📥 [Accumulator Worker] 슬라이드 {idx} 데이터를 전역 버퍼에 저장 완료. 다음 슬라이드로 이동."
                            elif node_name == "concat_video":
                                log_msg = ">> 🔗 [ConcatVideo Worker] 모든 슬라이드 동영상을 완벽하게 조율하여 병합 중..."
                                
                            log_history.append(log_msg)
                            log_html = "<div class='agent-log'>" + "<br>".join(log_history[-7:]) + "</div>"
                            log_console.markdown(log_html, unsafe_allow_html=True)
                            
                            # Streamlit 상태창 메시지 실시간 반응성 극대화
                            status_area.info(st.session_state.get("progress_status", "에이전트가 통신 중입니다..."))
                            
                    progress_bar.progress(1.0)
                    status_area.success("🎉 [Multi-Agent 협업 완료] 모든 에이전트들이 공정을 완료하여 강의 영상과 대본이 탄생했습니다.")
                    
                    # 영구 저장소로 출력 파일 복사
                    output_perm_dir = Path("PPT_Agent_Output")
                    output_perm_dir.mkdir(exist_ok=True)
                    
                    perm_video_path = output_perm_dir / "final_lecture.mp4"
                    shutil.copy(last_final_state["final_video"], perm_video_path)
                    
                    # 대본 수집 및 텍스트 파일 저장
                    all_scripts_joined = ""
                    for idx, scr in enumerate(last_final_state["all_scripts"]):
                        all_scripts_joined += f"=== [제 {idx+1}강 슬라이드 스크립트] ===\n{scr}\n\n"
                        
                    perm_script_path = output_perm_dir / "script_transcript.txt"
                    with open(perm_script_path, "w", encoding="utf-8") as f:
                        f.write(all_scripts_joined)
                        
                    # 챗봇용 FAISS 벡터 스토어 구축 (chatbot_helper 유틸 이용)
                    status_area.info("🧠 [FAISS DB Worker] 질문에 언제든 대답할 수 있도록 PPT 내용 분석 인덱스를 Vector DB에 로드 중...")
                    vector_store = build_vector_store(
                        last_final_state["slides"],
                        last_final_state["all_contents"],
                        last_final_state["all_scripts"],
                        openai_key
                    )
                    
                    # 세션 상태 캐싱 저장
                    st.session_state["processed"] = True
                    st.session_state["final_video_path"] = str(perm_video_path)
                    st.session_state["final_script_text"] = all_scripts_joined
                    st.session_state["vector_store"] = vector_store
                    st.session_state["ppt_structured_data"] = last_final_state["slides"]
                    st.session_state["chat_history"] = [] # 채팅 기록 리셋
                    
                    st.balloons()
            except Exception as e:
                st.error(f"🚨 파이프라인 처리 중 치명적인 에러가 발생했습니다. API Key 상태 또는 시스템 코덱을 점검해주세요. (원인: {str(e)})")

    # 최종 결과물 렌더링 영역
    if st.session_state["processed"]:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🏆 최종 생성 완료된 명품 강의 산출물")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("#### 📺 강의 동영상 플레이어")
            if os.path.exists(st.session_state["final_video_path"]):
                with open(st.session_state["final_video_path"], "rb") as video_file:
                    st.video(video_file.read())
                    
                with open(st.session_state["final_video_path"], "rb") as vf:
                    st.download_button(
                        label="📥 최종 강의 동영상 다운로드 (.mp4)",
                        data=vf,
                        file_name="AI_lecture_video.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )
            else:
                st.error("오류: 최종 비디오 파일의 디렉토리 로드에 실패했습니다.")
                
        with col2:
            st.markdown("#### 📝 실시간 강의 스크립트 (대본) 미리보기")
            st.text_area("작성된 대본 텍스트", st.session_state["final_script_text"], height=350)
            
            st.download_button(
                label="📥 발표 강의 대본 다운로드 (.txt)",
                data=st.session_state["final_script_text"],
                file_name="AI_lecture_script.txt",
                mime="text/plain",
                use_container_width=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 2: AI 강의 질의응답 (RAG 챗봇)
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 💬 스마트 AI 강의 보조 챗봇")
    st.markdown("Supervisor Multi-Agent와 FAISS Vector DB가 결합되어 질문의 본질을 정확하게 파악하고 필요한 정보를 찾아 대화합니다.")
    
    if not st.session_state["processed"]:
        st.info("💡 챗봇을 이용하려면 먼저 **'Tab 1'에서 PPT 파일을 업로드하고 강의 영상을 제작**해 주세요. 분석된 정보를 기반으로 챗봇 두뇌가 활성화됩니다.")
    else:
        # 기존 대화 히스토리 출력
        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                
        # 대화 입력창
        user_query = st.chat_input("이 PPT와 대본의 구체적인 내용에 대해 무엇이든 질문하세요...")
        
        if user_query:
            # 1. 유저 질문 렌더링 및 세션 누적
            st.session_state["chat_history"].append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.write(user_query)
                
            # 2. Chat Supervisor Graph 구동 및 실시간 모니터링
            chat_graph = build_chat_supervisor_graph()
            
            chat_initial_state = {
                "user_query": user_query,
                "chat_history": st.session_state["chat_history"],
                "openai_key": openai_key,
                "tavily_key": tavily_key if tavily_key else "",
                "vector_store": st.session_state["vector_store"]
            }
            
            # 챗봇 에이전트 구동 중 상태 안내 피드백
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                status_log = st.empty()
                
                status_lines = []
                final_state_out = chat_initial_state
                
                try:
                    # LangGraph stream을 활용해 활성화된 Worker를 실시간 렌더링
                    for event in chat_graph.stream(chat_initial_state, {"recursion_limit": 20}):
                        for node_name, state in event.items():
                            final_state_out = state
                            
                            log_line = ""
                            if node_name == "chat_supervisor":
                                next_w = state.get("next_worker", "FINISH")
                                log_line = f"🧠 [Chat Supervisor] 의도 파악 ➔ 다음 작업자로 `{next_w}` 배정"
                            elif node_name == "ppt_retriever":
                                log_line = "🔍 [PPT Retriever Worker] FAISS Vector DB 문서 고속 조회 성공"
                            elif node_name == "web_searcher":
                                log_line = "🌐 [Web Searcher Worker] Tavily를 사용한 실시간 웹 지식 구글링 수집 성공"
                            elif node_name == "response_generator":
                                log_line = "✍️ [Response Generator Worker] 수집 지식 취합 및 개인화 답변 구성 완료"
                                
                            status_lines.append(log_line)
                            log_console_html = "<div class='agent-log'>" + "<br>".join(status_lines) + "</div>"
                            status_log.markdown(log_console_html, unsafe_allow_html=True)
                            
                    # 최종 답변 렌더링
                    final_ans = final_state_out.get("final_response", "죄송합니다. 에이전트 연동 중 답변 생성에 실패했습니다.")
                    message_placeholder.write(final_ans)
                    
                    # 챗 대화 기록 누적
                    st.session_state["chat_history"].append({"role": "assistant", "content": final_ans})
                    
                except Exception as err:
                    st.error(f"챗봇 에이전트 구동 중 오류가 발생했습니다. (상세: {str(err)})")
                    
    st.markdown("</div>", unsafe_allow_html=True)
