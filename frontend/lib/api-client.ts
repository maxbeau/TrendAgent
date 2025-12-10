import axios from 'axios';

const apiBaseUrl = process.env.NODE_ENV === 'production' ? '/api' : 'http://localhost:8000/api';

// 报表/引擎计算可能超过 12s，这里放宽超时以减少无意义的前端失败提示。
export const apiClient = axios.create({
  baseURL: apiBaseUrl,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      // Keep the error object lightweight while surfacing backend message.
      error.message = error.response.data?.error ?? error.message;
    }
    return Promise.reject(error);
  },
);

export const getApiBaseUrl = () => apiBaseUrl;
