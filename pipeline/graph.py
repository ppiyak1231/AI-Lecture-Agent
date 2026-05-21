from langgraph.graph import StateGraph, END

# nodes 모듈에서 State 및 실행 노드들 임포트
from pipeline.nodes import (
    State,
    ChatState,
    node_lecture_supervisor,
    node_parse_all,
    node_generate_text,
    node_generate_script,
    node_generate_media,
    node_accumulate,
    node_concat_video,
    node_chat_supervisor,
    node_ppt_retriever,
    node_web_searcher,
    node_response_generator
)

# -----------------------------------------------------------------------------
# 1. 1번: 강의 제작용 Supervisor Graph 빌더
# -----------------------------------------------------------------------------
def route_lecture(state: State) -> str:
    """Lecture Supervisor가 지정한 next_worker 값을 기반으로 라우팅을 수행합니다."""
    worker = state.get("next_worker", "FINISH")
    if worker == "Parser":
        return "parser"
    elif worker == "Summarizer":
        return "summarizer"
    elif worker == "ScriptWriter":
        return "script_writer"
    elif worker == "MediaCreator":
        return "media_creator"
    elif worker == "Accumulator":
        return "accumulator"
    elif worker == "ConcatVideo":
        return "concat_video"
    return END

def build_lecture_supervisor_graph() -> StateGraph:
    """Lecture Supervisor StateGraph를 구성하고 컴파일하여 반환합니다."""
    builder = StateGraph(State)
    
    # 1. 노드 등록
    builder.add_node("supervisor", node_lecture_supervisor)
    builder.add_node("parser", node_parse_all)
    builder.add_node("summarizer", node_generate_text)
    builder.add_node("script_writer", node_generate_script)
    builder.add_node("media_creator", node_generate_media)
    builder.add_node("accumulator", node_accumulate)
    builder.add_node("concat_video", node_concat_video)
    
    # 2. 시작점 및 엣지 라우팅 설정
    builder.set_entry_point("supervisor")
    
    # Supervisor의 제어 라우팅 연결
    builder.add_conditional_edges(
        "supervisor",
        route_lecture,
        {
            "parser": "parser",
            "summarizer": "summarizer",
            "script_writer": "script_writer",
            "media_creator": "media_creator",
            "accumulator": "accumulator",
            "concat_video": "concat_video",
            END: END
        }
    )
    
    # 각 Worker 작업 완료 후 다시 Supervisor에게 의사결정 제어권 반환
    builder.add_edge("parser", "supervisor")
    builder.add_edge("summarizer", "supervisor")
    builder.add_edge("script_writer", "supervisor")
    builder.add_edge("media_creator", "supervisor")
    builder.add_edge("accumulator", "supervisor")
    
    # 최종 결합이 완료되면 워크플로우 END로 완결
    builder.add_edge("concat_video", END)
    
    return builder.compile()


# -----------------------------------------------------------------------------
# 2. 2번: RAG 챗봇용 Supervisor Graph 빌더
# -----------------------------------------------------------------------------
def route_chat(state: ChatState) -> str:
    """Chat Supervisor가 지정한 next_worker 값을 기반으로 라우팅을 수행합니다."""
    worker = state.get("next_worker", "FINISH")
    if worker == "PPT_Retriever":
        return "ppt_retriever"
    elif worker == "Web_Searcher":
        return "web_searcher"
    elif worker == "Response_Generator":
        return "response_generator"
    return END

def build_chat_supervisor_graph() -> StateGraph:
    """Chat Supervisor StateGraph를 구성하고 컴파일하여 반환합니다."""
    builder = StateGraph(ChatState)
    
    # 1. 노드 등록
    builder.add_node("chat_supervisor", node_chat_supervisor)
    builder.add_node("ppt_retriever", node_ppt_retriever)
    builder.add_node("web_searcher", node_web_searcher)
    builder.add_node("response_generator", node_response_generator)
    
    # 2. 시작점 및 엣지 라우팅 설정
    builder.set_entry_point("chat_supervisor")
    
    # Chat Supervisor의 제어 라우팅 연결
    builder.add_conditional_edges(
        "chat_supervisor",
        route_chat,
        {
            "ppt_retriever": "ppt_retriever",
            "web_searcher": "web_searcher",
            "response_generator": "response_generator",
            END: END
        }
    )
    
    # 각 Worker 작업 완료 후 다시 Chat Supervisor에게 판단 제어권 반환
    builder.add_edge("ppt_retriever", "chat_supervisor")
    builder.add_edge("web_searcher", "chat_supervisor")
    
    # 최종 답변 성형 완료 시 END로 완결
    builder.add_edge("response_generator", END)
    
    return builder.compile()
