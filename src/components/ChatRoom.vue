<template>
  <div class="q-pa-md row justify-center" style="height: 100vh; display: flex; flex-direction: column;">
    <!-- 채팅 메시지 리스트 -->
    <q-virtual-scroll
      ref="virtualScroll"
      :items="messages"
      :virtual-scroll-item-size="60"
      class="scroll-area"
      style="flex: 1; overflow-y: auto; width: 100%; max-width: 100%;"
    >
      <template #default="props">
        <q-chat-message
          :name="props.item.email"
          :avatar="props.item.avatar"
          :text="props.item.text"
          :stamp="props.item.stamp"
          :sent="props.item.sent"
          :bg-color="props.item.bgColor"
        />
      </template>
    </q-virtual-scroll>
    <!-- 로딩 중 상태 표시 -->
    <q-loader v-if="loading" size="60px" color="primary" />
    <!-- 입력창 -->
    <div
      class="input-container"
      style="width: 100%; max-width: 100%; background: white; padding: 10px;"
    >
      <div class="row items-center">
        <q-input
          outlined
          v-model="newMessage"
          dense
          placeholder="Type your message..."
          class="col"
          @keyup.enter="sendMessage"
        />
        <q-btn
          color="primary"
          round
          icon="send"
          @click="sendMessage"
          class="q-ml-sm"
        />
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick, onMounted, onUnmounted } from "vue";
import { useRoute } from "vue-router";
import axios from "axios";
import jmj from 'assets/jmj.png'

// axios.defaults.baseURL = 'http://192.168.0.22:9000';  // 백엔드 URL
axios.defaults.withCredentials = true; // 쿠키 포함
axios.defaults.headers.common['Content-Type'] = 'application/json';

const messages = ref([]);
const newMessage = ref("");
const loading = ref(false);

const sendMessage = async () => {
if (newMessage.value.trim()) {
  loading.value = true;
  const messageData = {
    email: "고민이 있는 사람",
    text: newMessage.value,
    stamp: new Date().toISOString(),
  };

  // 클라이언트 메시지 추가
  messages.value.push({
    id: messages.value.length + 1,
    email: messageData.email,
    avatar: '',
    text: [newMessage.value],
    stamp: messageData.stamp,
    sent: true,
    bgColor: "amber-7",
  });

  // api를 위한 데이터
  // TODO : user mbti 입력 칸 추가하기
  const requestData = {
    user_mbti: 'ISTP',
    user_message: newMessage.value,
  }

  newMessage.value = '';

  try {
    const response = await axios.post("http://localhost:8000/chat", requestData)
    const conversation = response.data.response.conversation
    const final_decision = response.data.response.final_decision
    let responseText = ""

    for(let i=1; i<conversation.length ; i++) {
      messages.value.push( {
      id: messages.value.length + 1,
      email: "JMJ",
      avatar: jmj,
      text: [conversation[i]],
      stamp: new Date().toISOString(),
      sent: false,
      bgColor: "amber-4",
      })
    }
    messages.value.push( {
      id: messages.value.length + 1,
      email: "JMJ",
      avatar: jmj,
      text: [final_decision],
      stamp: new Date().toISOString(),
      sent: false,
      bgColor: "amber-4",
      })

  } catch (error) {
    console.error("Backend API 호출 실패", error)
  } finally {
    loading.value = false; // 로딩 끝
  }
  // Virtual scroll 스크롤 하단으로 이동
  nextTick(() => {
    const virtualScroll = document.querySelector(".q-virtual-scroll__content");
    if (virtualScroll) {
      virtualScroll.scrollTo({
        top: virtualScroll.scrollHeight,
        behavior: "smooth",
      });
    }
  });
}
};
</script>

<style scoped>

</style>
