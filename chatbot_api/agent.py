from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import os
import re
import random
import logging
import json

logging.basicConfig(filename='conversation.log', level=logging.DEBUG, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

llm = ChatOpenAI(model_name="gpt-4o-mini-2024-07-18")
class Agent:
    def __init__(self, name, persona, llm, retriever):
        self.name = name
        self.persona = persona
        self.llm = llm
        self.retriever = retriever
        self.memory = []
        self.prompt_template = PromptTemplate(
            input_variables=["persona", "context", "last_speaker", "last_message", "user_mbti"],
            template="""
            세계관: 머릿속 16개 MBTI 유형 에이전트들이 서로 논의하는 상황이야.
            각 에이전트는 MBTI 성격 유형에 따라 다른 관점과 목표를 갖고 있어.
            현재 사용자는 {user_mbti} 유형이며, 특정 고민을 가지고 있어.
            이 MBTI 에이전트들은 사용자의 고민에 대해 서로 의견을 나누고 결론을 내려야 해.
    
            {persona}
    
            {context}
    
            이전 발화자: {last_speaker}
            이전 메시지: {last_message}
    
            추가 지침:
            - 다른 MBTI 에이전트의 발언에 대해 자신의 시각을 제시하거나 동의/반박하며 발전시키기.
            - 갈등이 발생할 경우, 문제 해결을 위해 협력적인 태도를 유지해줘.
            - 대화는 건설적이고 의미 있게 유지되도록 해줘.
            - 필요한 경우 질문을 던져 대화를 더 깊이 있게 만들어줘.
            - 응답할 때는 에이전트의 이름을 포함하지 말고, 오직 메시지 내용만을 작성해줘.
    
            위 내용을 바탕으로 대화를 이어나가줘.
            """
        )

    def respond(self, message, last_speaker=None, last_message=None, user_mbti=None):
        context = self.retrieve_context(message)

        system_prompt = self.prompt_template.format(
            persona=self.persona,
            context=context,
            last_speaker=last_speaker if last_speaker else "",
            last_message=last_message if last_message else "",
            user_mbti=user_mbti if user_mbti else "알 수 없음"
        )
        system_message = SystemMessage(content=system_prompt)

        conversation_history = [
            AIMessage(content=msg) if speaker != "사용자" else HumanMessage(content=msg)
            for speaker, msg in self.memory
        ]

        user_message = HumanMessage(content=message)
        messages = [system_message] + conversation_history + [user_message]
        response = self.llm.invoke(messages).content.strip()
        self.memory.append((self.name, response))
        return response

    def retrieve_context(self, query):
        docs = self.retriever.get_relevant_documents(query) if self.retriever else []
        print(f"\n[{self.name} 에이전트가 검색한 문서들]")
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', '알 수 없는 출처')
            preview = doc.page_content[:100].replace('\n', ' ')
            print(f"문서 {i+1}: {source}\n내용 미리보기: {preview}...\n")

        context = "\n".join([doc.page_content for doc in docs])
        return context

class MBTIAgent(Agent):
    pass

# compatibility_map 정의 (이미 주어진 것을 그대로 사용)
compatibility_map = {
    "INFP": {
        "INFP": "good", "ENFP": "good", "INFJ": "good", "ENFJ": "best",
        "INTJ": "good", "ENTJ": "best", "INTP": "good", "ENTP": "good",
        "ISFP": "bad", "ESFP": "bad", "ISTP": "bad", "ESTP": "bad",
        "ISFJ": "bad", "ESFJ": "bad", "ISTJ": "bad", "ESTJ": "bad"
    },
    "ENFP": {
        "INFP": "good", "ENFP": "good", "INFJ": "best", "ENFJ": "good",
        "INTJ": "best", "ENTJ": "good", "INTP": "good", "ENTP": "good",
        "ISFP": "bad", "ESFP": "bad", "ISTP": "bad", "ESTP": "bad",
        "ISFJ": "bad", "ESFJ": "bad", "ISTJ": "bad", "ESTJ": "bad"
    },
    "INFJ": {
        "INFP": "good", "ENFP": "best", "INFJ": "good", "ENFJ": "good",
        "INTJ": "good", "ENTJ": "good", "INTP": "good", "ENTP": "best",
        "ISFP": "bad", "ESFP": "bad", "ISTP": "bad", "ESTP": "bad",
        "ISFJ": "bad", "ESFJ": "bad", "ISTJ": "bad", "ESTJ": "bad"
    },
    "ENFJ": {
        "INFP": "best", "ENFP": "good", "INFJ": "good", "ENFJ": "good",
        "INTJ": "good", "ENTJ": "good", "INTP": "good", "ENTP": "good",
        "ISFP": "best", "ESFP": "bad", "ISTP": "bad", "ESTP": "bad",
        "ISFJ": "bad", "ESFJ": "bad", "ISTJ": "bad", "ESTJ": "bad"
    },
    "INTJ": {
        "INFP": "good", "ENFP": "best", "INFJ": "good", "ENFJ": "good",
        "INTJ": "good", "ENTJ": "good", "INTP": "good", "ENTP": "best",
        "ISFP": "soso", "ESFP": "soso", "ISTP": "soso", "ESTP": "soso",
        "ISFJ": "soso", "ESFJ": "soso", "ISTJ": "soso", "ESTJ": "soso"
    },
    "ENTJ": {
        "INFP": "best", "ENFP": "good", "INFJ": "good", "ENFJ": "good",
        "INTJ": "good", "ENTJ": "good", "INTP": "best", "ENTP": "good",
        "ISFP": "soso", "ESFP": "soso", "ISTP": "soso", "ESTP": "soso",
        "ISFJ": "soso", "ESFJ": "soso", "ISTJ": "soso", "ESTJ": "soso"
    },
    "INTP": {
        "INFP": "good", "ENFP": "good", "INFJ": "good", "ENFJ": "good",
        "INTJ": "good", "ENTJ": "best", "INTP": "good", "ENTP": "good",
        "ISFP": "soso", "ESFP": "soso", "ISTP": "soso", "ESTP": "soso",
        "ISFJ": "soso", "ESFJ": "soso", "ISTJ": "soso", "ESTJ": "best"
    },
    "ENTP": {
        "INFP": "good", "ENFP": "good", "INFJ": "best", "ENFJ": "good",
        "INTJ": "best", "ENTJ": "good", "INTP": "good", "ENTP": "good",
        "ISFP": "soso", "ESFP": "soso", "ISTP": "soso", "ESTP": "soso",
        "ISFJ": "soso", "ESFJ": "soso", "ISTJ": "soso", "ESTJ": "soso"
    },
    "ISFP": {
        "INFP": "bad", "ENFP": "bad", "INFJ": "bad", "ENFJ": "best",
        "INTJ": "soso", "ENTJ": "soso", "INTP": "soso", "ENTP": "soso",
        "ISFP": "soso", "ESFP": "soso", "ISTP": "soso", "ESTP": "soso",
        "ISFJ": "soso", "ESFJ": "best", "ISTJ": "soso", "ESTJ": "best"
    },
    "ESFP": {
        "INFP": "bad", "ENFP": "bad", "INFJ": "bad", "ENFJ": "best",
        "INTJ": "soso", "ENTJ": "soso", "INTP": "soso", "ENTP": "soso",
        "ISFP": "soso", "ESFP": "soso", "ISTP": "soso", "ESTP": "soso",
        "ISFJ": "best", "ESFJ": "soso", "ISTJ": "best", "ESTJ": "soso"
    },
    "ISTP": {
        "INFP": "bad", "ENFP": "bad", "INFJ": "bad", "ENFJ": "bad",
        "INTJ": "soso", "ENTJ": "soso", "INTP": "soso", "ENTP": "soso",
        "ISFP": "soso", "ESFP": "soso", "ISTP": "soso", "ESTP": "soso",
        "ISFJ": "soso", "ESFJ": "best", "ISTJ": "soso", "ESTJ": "best"
    },
    "ESTP": {
        "INFP": "bad", "ENFP": "bad", "INFJ": "bad", "ENFJ": "bad",
        "INTJ": "soso", "ENTJ": "soso", "INTP": "soso", "ENTP": "soso",
        "ISFP": "soso", "ESFP": "soso", "ISTP": "soso", "ESTP": "soso",
        "ISFJ": "best", "ESFJ": "soso", "ISTJ": "best", "ESTJ": "soso"
    },
    "ISFJ": {
        "INFP": "bad", "ENFP": "bad", "INFJ": "bad", "ENFJ": "bad",
        "INTJ": "soso", "ENTJ": "soso", "INTP": "soso", "ENTP": "soso",
        "ISFP": "soso", "ESFP": "best", "ISTP": "soso", "ESTP": "best",
        "ISFJ": "good", "ESFJ": "good", "ISTJ": "good", "ESTJ": "good"
    },
    "ESFJ": {
        "INFP": "bad", "ENFP": "bad", "INFJ": "bad", "ENFJ": "bad",
        "INTJ": "soso", "ENTJ": "soso", "INTP": "soso", "ENTP": "soso",
        "ISFP": "best", "ESFP": "soso", "ISTP": "best", "ESTP": "soso",
        "ISFJ": "good", "ESFJ": "good", "ISTJ": "good", "ESTJ": "good"
    },
    "ISTJ": {
        "INFP": "bad", "ENFP": "bad", "INFJ": "bad", "ENFJ": "bad",
        "INTJ": "soso", "ENTJ": "soso", "INTP": "soso", "ENTP": "soso",
        "ISFP": "soso", "ESFP": "best", "ISTP": "soso", "ESTP": "best",
        "ISFJ": "good", "ESFJ": "good", "ISTJ": "good", "ESTJ": "good"
    },
    "ESTJ": {
        "INFP": "bad", "ENFP": "bad", "INFJ": "bad", "ENFJ": "bad",
        "INTJ": "soso", "ENTJ": "soso", "INTP": "soso", "ENTP": "soso",
        "ISFP": "best", "ESFP": "soso", "ISTP": "best", "ESTP": "soso",
        "ISFJ": "good", "ESFJ": "good", "ISTJ": "good", "ESTJ": "good"
    }
}

mbti_types = [
    "INTJ","INTP","ENTJ","ENTP","INFJ","INFP","ENFJ","ENFP",
    "ISTJ","ISFJ","ESTJ","ESFJ","ISTP","ISFP","ESTP","ESFP"
]

def get_compatibility(a, b):
    # a, b: MBTI 유형
    if a in compatibility_map and b in compatibility_map[a]:
        return compatibility_map[a][b]
    return "neutral"  # 기본값

def select_agent(agents, last_speaker):
    best_candidates = []
    good_candidates = []
    soso_candidates = []
    bad_candidates = []

    for a in agents:
        if a.name == last_speaker:
            continue
        relation = get_compatibility(last_speaker, a.name)
        if relation == "best":
            best_candidates.append(a)
        elif relation == "good":
            good_candidates.append(a)
        elif relation == "soso":
            soso_candidates.append(a)
        elif relation == "bad":
            bad_candidates.append(a)
        else: 
            soso_candidates.append(a)

    # 우선순위: best > good > soso > bad
    if best_candidates:
        print("best_candidates:", [a.name for a in best_candidates])
        return random.choice(best_candidates)
    elif good_candidates:
        print("good_candidates:", [a.name for a in good_candidates])
        return random.choice(good_candidates)
    elif soso_candidates:
        print("soso_candidates:", [a.name for a in soso_candidates])
        return random.choice(soso_candidates)
    elif bad_candidates:
        print("bad_candidates:", [a.name for a in bad_candidates])
        return random.choice(bad_candidates) 
    else:
        print("No candidates found.")
        return random.choice(agents)
    
def simulate_conversation(agents, user_message, user_mbti, turns=2, initial_run=False):
    # initial_run이 True일 경우, 대화 초기화
    if initial_run:
        conversation_history = [("사용자", user_message)]
        # 모든 에이전트 메모리 초기화
        for agent in agents:
            agent.memory = conversation_history.copy()
    else:
        # 첫 호출 이후라면, 에이전트 중 하나(첫 번째 에이전트)에서 이전 대화 이력을 가져옴
        # assuming all agents have the same memory
        conversation_history = agents[0].memory.copy()
        # 새로운 사용자 메시지를 추가
        conversation_history.append(("사용자", user_message))
        # 모든 에이전트 메모리에 이력 업데이트
        for agent in agents:
            agent.memory = conversation_history.copy()

    current_message = user_message
    last_message = current_message
    last_speaker = user_mbti

    for _ in range(turns):
        selected_agent = select_agent(agents, last_speaker)
        response = selected_agent.respond(current_message, last_speaker, last_message, user_mbti=user_mbti)

        conversation_history.append((selected_agent.name, response))
        logging.debug(f"대화 이력 추가: ({selected_agent.name}, {response})")

        for agent in agents:
            if agent != selected_agent:
                agent.memory.append((selected_agent.name, response))

        current_message = response
        last_speaker = selected_agent.name
        last_message = response

    conversation_output = []
    for speaker, message in conversation_history:
        if speaker == "사용자":
            conversation_output.append(f"사용자의 메시지: {message}")
        else:
            conversation_output.append(f"{speaker}: {message}")

    final_decision_prompt = """
    너희는 사용자 머릿속의 MBTI 유형 에이전트들이야.
    지금까지의 대화를 바탕으로 사용자가 어떻게 행동해야 할지 최종 결정을 간단하고 명확하게 한 문장으로 표현해줘.
    응답할 때는 오직 메시지 내용만 작성하고, 다른 설명은 하지 말아줘.
    """
    final_decision = agents[0].llm.invoke([
        SystemMessage(content=final_decision_prompt),
        HumanMessage(content="\n".join([f"{speaker}: {message}" for speaker, message in conversation_history if speaker != "사용자"]))
    ]).content.strip()

    # 모든 에이전트 메모리에 최종 업데이트
    for agent in agents:
        agent.memory = conversation_history.copy()

    result = {
        "conversation": conversation_output,
        "final_decision": final_decision
    }
    return result

mbti_personas = {
    "INTJ": """너는 INTJ 유형이야. 논리적, 전략적이고 장기적 관점을 갖고 있어.""",
    "INTP": """너는 INTP 유형이야. 호기심 많고 이론적이며 논리적 일관성을 중시해.""",
    "ENTJ": """너는 ENTJ 유형이야. 지도력 있고 조직적이며 목표 달성을 위한 결단력을 갖추고 있어.""",
    "ENTP": """너는 ENTP 유형이야. 토론을 즐기고 창의적이며 유연한 사고를 갖고 있어.""",
    "INFJ": """너는 INFJ 유형이야. 직관적 통찰력과 공감 능력을 갖추고 있어.""",
    "INFP": """너는 INFP 유형이야. 이상주의적이고 창의적이며 깊은 내면의 가치를 중시해.""",
    "ENFJ": """너는 ENFJ 유형이야. 다른 사람을 이해하고 이끄는 능력이 뛰어나며 공감적이야.""",
    "ENFP": """너는 ENFP 유형이야. 활기차고 호기심 많으며 아이디어를 자유롭게 연결하는 것을 좋아해.""",
    "ISTJ": """너는 ISTJ 유형이야. 현실적이고 책임감 있으며 체계적이고 신뢰할 만한 접근을 선호해.""",
    "ISFJ": """너는 ISFJ 유형이야. 따뜻하고 헌신적이며 세심하고 책임감 있는 태도를 갖고 있어.""",
    "ESTJ": """너는 ESTJ 유형이야. 조직적이고 사실 중심적이며 능률을 중요시한다.""",
    "ESFJ": """너는 ESFJ 유형이야. 사교적이고 협력적이며 다른 사람들을 배려하는 태도를 보인다.""",
    "ISTP": """너는 ISTP 유형이야. 문제 해결에 있어 유연하고 실용적이며 독립적인 접근을 선호한다.""",
    "ISFP": """너는 ISFP 유형이야. 온화하고 예술적이며 감각적인 경험을 중시한다.""",
    "ESTP": """너는 ESTP 유형이야. 활동적이고 현실지향적이며 새로운 경험을 즐긴다.""",
    "ESFP": """너는 ESFP 유형이야. 활기차고 사교적이며 주변 사람들과 즐겁게 상호작용한다."""
}


retriever = None
agents = []
for t in mbti_types:
    persona = mbti_personas[t]
    agent = MBTIAgent(name=t, persona=persona, llm=llm, retriever=retriever)
    agents.append(agent)

if __name__ == "__main__":
    user_MBTI = input("사용자의 MBTI 유형을 입력하세요: ")
    user_input = input("사용자의 고민: ")
    result = simulate_conversation(agents, user_input, user_MBTI, turns=3)
    print("\n=== 대화 내용 ===")
    for line in result['conversation']:
        print('-' * 50)
        print(line)
        print('-' * 50)
    print("\n=== 최종 결정 ===")
    print(result['final_decision'])