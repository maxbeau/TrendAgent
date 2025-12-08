import axios from 'axios';

const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://localhost:8000/api';

export const apiClient = axios.create({
  baseURL: apiBaseUrl,
  timeout: 12000,
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
