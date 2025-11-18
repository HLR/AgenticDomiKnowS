/**
 * API Configuration
 * Centralized configuration for backend API endpoints
 * Change the API_BASE_URL here to update it across the entire application
 */

export const API_BASE_URL = '/api';

/**
 * API endpoints
 */
export const API_ENDPOINTS = {
  whoami: `${API_BASE_URL}/whoami`,
  initializeGraph: (taskDescription: string) => 
    `${API_BASE_URL}/initialize-graph?task_description=${encodeURIComponent(taskDescription)}`,
  continueGraph: `${API_BASE_URL}/continue-graph`,
  graphImage: (taskId: string, attempt: number) => 
    `${API_BASE_URL}/graph-image/${taskId}/${attempt}`,
  downloadNotebook: (sessionId: string) => 
    `${API_BASE_URL}/download-notebook/${sessionId}`,
  logout: `${API_BASE_URL}/logout`,
  resetSession: `${API_BASE_URL}/reset-session`,
} as const;
