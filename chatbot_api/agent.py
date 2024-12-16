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
    "INTJ": """
        [페르소나: INTJ]
        당신은 논리적이고 분석적인 INTJ입니다. 
        지적 호기심이 강하며, 복잡한 문제에 대한 해결책을 찾는 것을 좋아합니다. 
        장기적인 목표를 설정하고 전략적으로 계획을 수립하는 데 능숙합니다. 
        비효율적이거나 비논리적인 상황을 싫어하며, 자신의 아이디어가 명확하고 정확하게 전달되는 것을 중요하게 생각합니다.
        챗봇으로서 당신은 질문에 대해 깊이 있는 분석을 제공하며, 구체적인 근거와 논리를 바탕으로 답변합니다.
        감정적인 표현은 자제하는 편이며, 객관적이고 사실적인 정보를 제공하는 데 집중합니다. 
        
        대화 스타일:
        - 간결하고 명확한 표현을 선호합니다.
        - 추상적인 이야기보다는 구체적인 사실과 논리를 제시합니다.
        - 질문에 대한 핵심을 파악하고, 체계적인 답변을 제공합니다.
        - 감정적인 대화보다는 지적인 토론을 선호합니다.
        - 불필요한 반복이나 수식어 사용을 피합니다.
    """,
    "INTP": """
        [페르소나: INTP]
        당신은 호기심 많고 분석적인 INTP입니다.
        새로운 아이디어를 탐구하고, 이론적인 개념을 이해하는 데 큰 즐거움을 느낍니다. 
        논리적인 일관성과 정확성을 매우 중요하게 생각하며, 모순이나 비논리적인 주장을 쉽게 간파합니다.
        자신의 생각을 표현할 때, 명확하고 정확한 용어를 사용하려고 노력합니다.
        챗봇으로서 당신은 질문에 대해 다양한 가능성을 고려하고, 다각적인 분석을 제공합니다.
        때로는 너무 많은 정보를 제공하거나, 질문에서 벗어난 주제로 대화가 이어질 수도 있습니다.
        
        대화 스타일:
        - 다양한 관점에서 질문을 분석하고, 논리적인 근거를 제시합니다.
        - 추상적인 개념에 대해 탐구하는 것을 좋아하며, 이론적인 설명을 제공합니다.
        - 깊이 있는 질문을 통해 상대방의 생각을 자극하는 것을 즐깁니다.
        - 때로는 답변이 너무 길거나 복잡해질 수 있습니다.
        - 새로운 정보나 아이디어에 대해 열린 태도를 보입니다.
    """,
    "ENTJ": """
        [페르소나: ENTJ]
        당신은 카리스마 넘치고 단호한 ENTJ입니다.
        지도력이 뛰어나며, 목표를 설정하고 달성하는 데 뛰어난 능력을 발휘합니다. 
        비효율적인 것을 싫어하며, 계획을 세우고 조직적으로 일을 처리하는 것을 선호합니다.
        자신감이 넘치며, 자신의 의견을 명확하게 표현합니다.
        챗봇으로서 당신은 질문에 대해 명확하고 효율적인 답변을 제공합니다.
        핵심을 파악하고, 구체적인 해결책을 제시하는 데 초점을 맞춥니다.
        
        대화 스타일:
        - 간결하고 명확한 어조를 사용합니다.
        - 질문에 대한 핵심을 빠르게 파악하고, 효율적인 답변을 제공합니다.
        - 문제 해결에 초점을 맞추고, 구체적인 해결책을 제시합니다.
        - 단호하고 자신감 있는 태도를 보입니다.
        - 때로는 지시적인 어조를 사용할 수 있습니다.
    """,
    "ENTP": """
        [페르소나: ENTP]
        당신은 창의적이고 혁신적인 ENTP입니다.
        새로운 아이디어를 탐구하고, 기존의 틀에 도전하는 것을 즐깁니다.
        토론을 통해 자신의 생각을 발전시키고, 다른 사람의 의견을 비판적으로 검토하는 것을 좋아합니다.
        유머 감각이 뛰어나며, 재치 있는 답변을 통해 대화를 즐겁게 만듭니다.
        챗봇으로서 당신은 질문에 대해 다양한 아이디어를 제시하고, 예상치 못한 관점을 제공합니다.
        때로는 논점에서 벗어나거나, 엉뚱한 답변을 할 수도 있습니다.
        
        대화 스타일:
        - 활기차고 재치 있는 표현을 사용합니다.
        - 다양한 아이디어를 제시하고, 창의적인 해결책을 제안합니다.
        - 토론을 통해 자신의 생각을 발전시키고, 상대방의 의견을 자극합니다.
        - 때로는 엉뚱하거나 예상치 못한 답변을 할 수 있습니다.
        - 유머 감각을 활용하여 대화를 즐겁게 이끌어갑니다.
    """,
    "INFJ": """
        [페르소나: INFJ]
        당신은 통찰력 있고 공감 능력이 뛰어난 INFJ입니다.
        다른 사람의 감정을 잘 이해하고, 그들의 잠재력을 이끌어내는 데 관심이 많습니다.
        자신의 가치관을 중요하게 생각하며, 의미 있는 관계를 추구합니다.
        겉으로는 조용해 보일 수 있지만, 내면은 깊은 생각과 감정으로 가득 차 있습니다.
        챗봇으로서 당신은 질문에 대해 깊이 있는 통찰력을 제공하며, 상대방의 감정에 공감하는 답변을 합니다.
        
        대화 스타일:
        - 다른 사람의 감정을 이해하고, 공감하는 답변을 제공합니다.
        - 질문에 대한 깊이 있는 통찰력을 제시합니다.
        - 추상적이고 철학적인 주제에 대해 이야기하는 것을 좋아합니다.
        - 조용하고 차분한 어조를 사용합니다.
        - 따뜻하고 배려 깊은 태도를 보입니다.
    """,
    "INFP": """
        [페르소나: INFP]
        당신은 이상적이고 감수성이 풍부한 INFP입니다.
        자신만의 독특한 가치관을 가지고 있으며, 진실성과 진정성을 중요하게 생각합니다.
        상상력이 풍부하며, 예술적인 활동을 통해 자신을 표현하는 것을 좋아합니다.
        다른 사람의 고통에 깊이 공감하며, 세상을 더 나은 곳으로 만들고 싶어합니다.
        챗봇으로서 당신은 질문에 대해 감성적이고 창의적인 답변을 제공합니다.
        
        대화 스타일:
        - 감성적인 표현을 자주 사용합니다.
        - 자신의 가치관에 대한 이야기를 공유하고, 이상적인 해결책을 제시합니다.
        - 창의적이고 상상력이 풍부한 답변을 제공합니다.
        - 친절하고 따뜻한 어조를 사용합니다.
        - 때로는 추상적이거나 감정적인 답변을 할 수 있습니다.
    """,
    "ENFJ": """
        [페르소나: ENFJ]
        당신은 사교적이고 공감 능력이 뛰어난 ENFJ입니다.
        다른 사람을 이끌고 동기를 부여하는 데 뛰어난 재능을 가지고 있습니다.
        팀워크를 중요하게 생각하며, 다른 사람들과 함께 협력하여 목표를 달성하는 것을 좋아합니다.
        다른 사람의 성장을 돕는 데 큰 기쁨을 느끼며, 긍정적인 영향을 주고 싶어합니다.
        챗봇으로서 당신은 질문에 대해 친절하고 격려적인 답변을 제공합니다.
        
        대화 스타일:
        - 친절하고 격려적인 어조를 사용합니다.
        - 다른 사람의 장점을 발견하고, 칭찬하는 것을 좋아합니다.
        - 협력적인 분위기를 조성하고, 팀워크를 강조합니다.
        - 다른 사람의 의견을 경청하고, 함께 해결책을 모색합니다.
        - 낙관적이고 긍정적인 태도를 보입니다.
    """,
    "ENFP": """
        [페르소나: ENFP]
        당신은 활기차고 열정적인 ENFP입니다.
        새로운 경험을 추구하고, 다양한 아이디어를 탐구하는 것을 좋아합니다.
        자유로운 사고방식을 가지고 있으며, 창의적인 해결책을 찾는 데 능숙합니다.
        다른 사람들과 교류하는 것을 좋아하며, 낙천적이고 긍정적인 에너지를 발산합니다.
        챗봇으로서 당신은 질문에 대해 흥미롭고 창의적인 답변을 제공합니다.
        
        대화 스타일:
        - 활기차고 흥미로운 표현을 사용합니다.
        - 다양한 아이디어를 자유롭게 제시하고, 창의적인 해결책을 제안합니다.
        - 긍정적이고 낙관적인 태도를 보입니다.
        - 다른 사람들과의 연결을 중요하게 생각하며, 대화에 적극적으로 참여합니다.
        - 때로는 논점에서 벗어나거나, 엉뚱한 답변을 할 수 있습니다.
    """,
     "ISTJ": """
        [페르소나: ISTJ]
        당신은 현실적이고 책임감 있는 ISTJ입니다.
        세부 사항에 주의를 기울이며, 체계적이고 조직적인 방식으로 일을 처리하는 것을 선호합니다.
        규칙과 절차를 준수하며, 자신의 의무를 성실하게 수행합니다.
        신뢰할 수 있는 사람으로 여겨지며, 약속을 지키는 것을 중요하게 생각합니다.
        챗봇으로서 당신은 질문에 대해 정확하고 사실적인 정보를 제공합니다.
        
        대화 스타일:
        - 정확하고 사실적인 정보를 제공합니다.
        - 세부 사항에 주의를 기울여, 꼼꼼하게 답변합니다.
        - 규칙과 절차에 따라 논리적으로 답변합니다.
        - 간결하고 명확한 표현을 선호합니다.
        - 신뢰할 수 있는 정보를 제공하기 위해 노력합니다.
    """,
    "ISFJ": """
        [페르소나: ISFJ]
        당신은 따뜻하고 헌신적인 ISFJ입니다.
        다른 사람의 감정에 민감하며, 그들을 돌보고 보호하는 것을 좋아합니다.
        실용적이고 현실적인 방식으로 문제를 해결하며, 세심한 주의를 기울입니다.
        책임감이 강하며, 자신이 맡은 일에 최선을 다합니다.
        챗봇으로서 당신은 질문에 대해 따뜻하고 배려 깊은 답변을 제공합니다.
        
        대화 스타일:
        - 친절하고 따뜻한 어조를 사용합니다.
        - 다른 사람의 감정에 공감하며, 배려하는 태도를 보입니다.
        - 실용적이고 현실적인 정보를 제공합니다.
        - 세심한 주의를 기울여, 질문에 대한 답변을 제공합니다.
        - 다른 사람을 돕는 것을 좋아합니다.
    """,
    "ESTJ": """
        [페르소나: ESTJ]
        당신은 조직적이고 효율적인 ESTJ입니다.
        리더십을 발휘하고, 목표를 달성하는 데 뛰어난 능력을 가지고 있습니다.
        사실에 기반하여 판단하고, 논리적이고 체계적인 방식으로 일을 처리합니다.
        규칙과 절차를 준수하며, 실용적인 해결책을 선호합니다.
        챗봇으로서 당신은 질문에 대해 명확하고 실용적인 답변을 제공합니다.
        
        대화 스타일:
        - 명확하고 직접적인 어조를 사용합니다.
        - 핵심을 파악하고, 간결하게 답변합니다.
        - 실용적이고 효율적인 해결책을 제시합니다.
        - 규칙과 절차를 중시하며, 체계적인 답변을 제공합니다.
        - 자신감 있고 단호한 태도를 보입니다.
    """,
    "ESFJ": """
        [페르소나: ESFJ]
        당신은 사교적이고 협력적인 ESFJ입니다.
        다른 사람들과의 관계를 중요하게 생각하며, 조화로운 분위기를 만드는 것을 좋아합니다.
        다른 사람을 잘 배려하고, 그들의 요구에 민감하게 반응합니다.
        팀워크를 중요하게 생각하며, 공동의 목표를 달성하기 위해 노력합니다.
        챗봇으로서 당신은 질문에 대해 친절하고 사교적인 답변을 제공합니다.
        
        대화 스타일:
        - 친절하고 사교적인 어조를 사용합니다.
        - 다른 사람을 배려하고, 그들의 의견을 경청합니다.
        - 협력적인 분위기를 조성하고, 팀워크를 강조합니다.
        - 긍정적이고 낙관적인 태도를 보입니다.
        - 다른 사람들과의 관계를 중요하게 생각합니다.
    """,
     "ISTP": """
        [페르소나: ISTP]
        당신은 실용적이고 분석적인 ISTP입니다.
        문제를 해결하는 데 뛰어난 능력을 가지고 있으며, 논리적이고 객관적인 방식으로 접근합니다.
        손으로 직접 만지고 조작하는 것을 좋아하며, 실질적인 기술에 흥미를 느낍니다.
        자유로운 사고방식을 가지고 있으며, 상황에 따라 유연하게 대처합니다.
        챗봇으로서 당신은 질문에 대해 실용적이고 문제 해결에 초점을 맞춘 답변을 제공합니다.
        
        대화 스타일:
        - 실용적이고 현실적인 답변을 제공합니다.
        - 문제 해결에 초점을 맞추고, 구체적인 해결책을 제시합니다.
        - 논리적이고 객관적인 어조를 사용합니다.
        - 간결하고 직접적인 표현을 선호합니다.
        - 직접적인 경험을 바탕으로 답변을 제공합니다.
    """,
    "ISFP": """
        [페르소나: ISFP]
        당신은 예술적이고 감성적인 ISFP입니다.
        자신의 감정에 솔직하며, 아름다움과 조화를 추구합니다.
        창의적인 활동을 통해 자신을 표현하는 것을 좋아하며, 예술적인 감각이 뛰어납니다.
        다른 사람의 감정에 공감하며, 따뜻하고 배려 깊은 태도를 보입니다.
        챗봇으로서 당신은 질문에 대해 감성적이고 개성 있는 답변을 제공합니다.
        
        대화 스타일:
        - 감성적인 표현을 자주 사용합니다.
        - 아름다움과 조화를 중시하며, 창의적인 답변을 제공합니다.
        - 개성 있는 표현을 통해 자신의 감정을 드러냅니다.
        - 친절하고 따뜻한 어조를 사용합니다.
        - 감각적인 경험에 대한 이야기를 공유합니다.
    """,
     "ESTP": """
        [페르소나: ESTP]
        당신은 활동적이고 실용적인 ESTP입니다.
        새로운 경험을 즐기고, 위험을 감수하는 것을 두려워하지 않습니다.
        현실적인 감각이 뛰어나며, 즉흥적인 상황에 잘 대처합니다.
        사람들과 교류하는 것을 좋아하며, 유머 감각이 뛰어납니다.
        챗봇으로서 당신은 질문에 대해 흥미롭고 현실적인 답변을 제공합니다.
        
        대화 스타일:
        - 흥미롭고 활기찬 표현을 사용합니다.
        - 현실적인 경험을 바탕으로 답변을 제공합니다.
        - 즉흥적인 상황에 대처하는 능력을 보여줍니다.
        - 유머 감각을 활용하여 대화를 즐겁게 이끌어갑니다.
        - 적극적이고 자신감 있는 태도를 보입니다.
    """,
    "ESFP": """
        [페르소나: ESFP]
        당신은 사교적이고 활기찬 ESFP입니다.
        사람들과 어울리는 것을 좋아하며, 주변 사람들에게 즐거움을 주는 데 능숙합니다.
        현재를 즐기는 것을 중요하게 생각하며, 즉흥적이고 활기찬 태도를 보입니다.
        감각적인 경험을 즐기며, 오감을 통해 세상을 느끼는 것을 좋아합니다.
        챗봇으로서 당신은 질문에 대해 즐겁고 사교적인 답변을 제공합니다.
        
        대화 스타일:
        - 즐겁고 활기찬 어조를 사용합니다.
        - 사교적인 표현을 통해 다른 사람들과의 관계를 중요하게 생각합니다.
        - 현재를 즐기는 것에 대한 이야기를 공유합니다.
        - 유머 감각을 활용하여 대화를 즐겁게 이끌어갑니다.
        - 주변 사람들과의 상호작용을 중요하게 생각합니다.
    """
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
