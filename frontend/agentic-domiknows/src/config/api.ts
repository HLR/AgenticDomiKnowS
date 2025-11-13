/**
 * API Configuration
 * Centralized configuration for backend API endpoints
 * Change the API_BASE_URL here to update it across the entire application
 */

export const API_BASE_URL = 'http://localhost:8000';

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
  logout: `${API_BASE_URL}/logout`,
} as const;
