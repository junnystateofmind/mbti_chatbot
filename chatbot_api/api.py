# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import agent  # agent.py 로드
import process # process.py 로드
from fastapi.middleware.cors import CORSMiddleware
import uuid
import copy
from asyncio import Lock
from pprint import pprint

# FastAPI 앱 초기화
app = FastAPI()

# CORS 설정
origins = [
    "http://192.168.0.22/*",
    "http://192.168.0.23:9000",
    "http://localhost:9000",  # 프론트엔드 URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 에이전트 초기화 함수 임포트
from process import initialize_agents

# 동기화를 위한 Lock
lock = Lock()

# 에이전트 초기화 이벤트 등록
@app.on_event("startup")
async def startup_event():
    initialize_agents()
    print("서버 시작 시 에이전트 초기화 완료.")

# 사용자 입력 모델 정의
class ChatInput(BaseModel):
    user_message: str
    user_mbti: str
    session_id: Optional[str] = None

# 세션별 에이전트 상태 저장소
session_agents = {}

# 대화 엔드포인트
# api.py 수정
@app.post("/chat")
async def chat_endpoint(chat_input: ChatInput):
    async with lock:
        try:
            if chat_input.session_id is None:
                session_id = str(uuid.uuid4())

                # 새 MBTI 에이전트 인스턴스 생성
                agents_copy = [
                    agent.MBTIAgent(
                        name=a.name,
                        persona=a.persona,
                        llm=a.llm,
                        retriever=a.retriever
                    ) for a in process.mbti_agents
                ]

                result = agent.simulate_conversation(
                    agents=agents_copy,
                    user_message=chat_input.user_message,
                    user_mbti=chat_input.user_mbti,
                    turns=3,
                    initial_run=True
                )

                # 세션 저장
                session_agents[session_id] = agents_copy
                return {"session_id": session_id, "response": result}

            else:
                session_id = chat_input.session_id
                if session_id not in session_agents:
                    raise HTTPException(status_code=400, detail="유효하지 않은 세션 ID입니다.")

                agents_copy = session_agents[session_id]
                result = agent.simulate_conversation(
                    agents=agents_copy,
                    user_message=chat_input.user_message,
                    user_mbti=chat_input.user_mbti,
                    turns=3,
                    initial_run=False
                )

                # 세션 업데이트
                session_agents[session_id] = agents_copy
                return {"session_id": session_id, "response": result}

        except Exception as e:
            import traceback
            print("서버 오류 발생!")
            print("요청 데이터:", chat_input.dict())
            print("에러 스택 트레이스:", traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")