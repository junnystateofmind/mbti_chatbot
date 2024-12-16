import agent
import os
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import faiss
import pickle
from concurrent.futures import ProcessPoolExecutor

# LLM 초기화 - gpt-4o-mini-2024-07-18 모델 사용
llm = ChatOpenAI(model_name="gpt-4o-mini-2024-07-18", temperature=0.8)

def build_vector_database(urls, agent_name):
    vectorstore_dir = f"{agent_name}_vectorstore"
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    if os.path.exists(vectorstore_dir):
        vectorstore = FAISS.load_local(
            vectorstore_dir,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True 
        )
        retriever = vectorstore.as_retriever()
        print(f"{agent_name}의 Vectorstore를 로드했습니다.")
        return retriever
    else:
        loader = WebBaseLoader(urls)
        docs = loader.load()

        def preprocess_text(doc):
            text = re.sub(r'\[\d+\]', '', doc.page_content)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()

        processed_texts = [preprocess_text(doc) for doc in docs]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        texts = []
        for text in processed_texts:
            splits = text_splitter.split_text(text)
            texts.extend(splits)

        vectorstore = FAISS.from_texts(texts, embedding_model)
        retriever = vectorstore.as_retriever()
        vectorstore.save_local(vectorstore_dir)
        print(f"{agent_name}의 Vectorstore를 생성하고 저장했습니다.")
        return retriever

# MBTI별 URL 딕셔너리
mbti_urls = {
    "INTJ": ["https://namu.wiki/w/INTJ"],
    "INTP": ["https://namu.wiki/w/INTP"],
    "ENTJ": ["https://namu.wiki/w/ENTJ"],
    "ENTP": ["https://namu.wiki/w/ENTP"],
    "INFJ": ["https://namu.wiki/w/INFJ"],
    "INFP": ["https://namu.wiki/w/INFP"],
    "ENFJ": ["https://namu.wiki/w/ENFJ"],
    "ENFP": ["https://namu.wiki/w/ENFP"],
    "ISTJ": ["https://namu.wiki/w/ISTJ"],
    "ISFJ": ["https://namu.wiki/w/ISFJ"],
    "ESTJ": ["https://namu.wiki/w/ESTJ"],
    "ESFJ": ["https://namu.wiki/w/ESFJ"],
    "ISTP": ["https://namu.wiki/w/ISTP"],
    "ISFP": ["https://namu.wiki/w/ISFP"],
    "ESTP": ["https://namu.wiki/w/ESTP"],
    "ESFP": ["https://namu.wiki/w/ESFP"]
}

# 공통 URL 예시
common_urls = ["https://namu.wiki/w/MBTI"]

mbti_types = agent.mbti_types
mbti_personas = agent.mbti_personas

def load_retriever_for_type(t):
    urls = mbti_urls.get(t, common_urls)  # MBTI별 URL 또는 기본 공통 URL 할당
    return (t, build_vector_database(urls, t.upper()))


def initialize_agents():
    global mbti_agents
    mbti_agents = [
        agent.MBTIAgent(name=t, persona=mbti_personas[t], llm=llm, retriever=build_vector_database(mbti_urls[t], t.upper()))
        for t in mbti_types
    ]
    print("모든 에이전트 초기화 완료.")


if __name__ == "__main__":
    user_mbti = input("사용자의 MBTI 유형을 입력하세요: ")
    user_input = input("사용자의 초기 고민: ")

    # 초기 실행: initial_run=True
    result = agent.simulate_conversation(mbti_agents, user_input, user_mbti, turns=3, initial_run=True)
    print("\n=== 초기 대화 내용 ===")
    for line in result['conversation']:
        print(line)
    print("\n=== 초기 최종 결정 ===")
    print(result['final_decision'])

    # 이후 사용자가 추가 메시지를 입력하면 멀티턴 대화 계속
    while True:
        next_input = input("\n추가로 하고 싶은 말이 있나요? (quit 입력 시 종료): ")
        if next_input.lower() == 'quit':
            break
        # 이미 초기 실행을 했으므로 initial_run=False
        new_result = agent.simulate_conversation(mbti_agents, next_input, user_mbti, turns=3, initial_run=False)
        print("\n=== 추가 대화 내용 ===")
        for line in new_result['conversation']:
            print(line)
        print("\n=== 새로운 최종 결정 ===")
        print(new_result['final_decision'])