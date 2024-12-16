import type { RouteRecordRaw } from 'vue-router';

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    component: () => import('layouts/MainLayout.vue'),
    children: [
      // 리다이렉트 설정 추가
      {
        path: '',
        redirect: '/chat',  // 기본 경로로 접속 시 '/chat'으로 리다이렉트
      },
      {
        path: 'chat',
        component: () => import('components/ChatRoom.vue'),
      },
    ],
  },

  // 기본 경로가 아닌 경우 에러 페이지 표시
  {
    path: '/:catchAll(.*)*',
    component: () => import('pages/ErrorNotFound.vue'),
  },
];

export default routes;
