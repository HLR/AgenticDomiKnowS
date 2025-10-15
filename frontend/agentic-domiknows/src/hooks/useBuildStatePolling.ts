import { useState, useEffect, useRef } from 'react';

interface BuildState {
  Task_ID?: string;
  Task_definition: string;
  graph_rag_examples: string[];
  graph_max_attempts: number;
  graph_attempt: number;
  graph_code_draft: string[];
  graph_visual_tools?: { [key: string]: any };
  graph_review_notes: string[];
  graph_reviewer_agent_approved: boolean;
  graph_exe_notes: string[];
  graph_exe_agent_approved: boolean;
  human_approved: boolean;
  human_notes: string;
}

interface PollingConfig {
  enabled: boolean;
  intervalMs?: number;
  onStateChange?: (oldState: BuildState | null, newState: BuildState) => void;
}

/**
 * Custom hook to poll the backend for BuildState updates
 * This enables real-time updates in the ProcessMonitor without modifying backend
 */
export function useBuildStatePolling(
  sessionId: string | null,
  config: PollingConfig
) {
  const [buildState, setBuildState] = useState<BuildState | null>(null);
  const [isPolling, setIsPolling] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const previousStateRef = useRef<BuildState | null>(null);

  useEffect(() => {
    if (!config.enabled || !sessionId) {
      // Stop polling if disabled or no session
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
        setIsPolling(false);
      }
      return;
    }

    setIsPolling(true);

    // Poll function - fetches current state from backend
    const pollState = async () => {
      try {
        const response = await fetch('http://localhost:8000/whoami', {
          credentials: 'include'
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        // Check if session has buildState data
        if (data.data && data.data.buildState) {
          const newState = data.data.buildState as BuildState;
          
          // Check if state actually changed
          if (JSON.stringify(previousStateRef.current) !== JSON.stringify(newState)) {
            setBuildState(newState);
            
            // Call onChange callback if provided
            if (config.onStateChange) {
              config.onStateChange(previousStateRef.current, newState);
            }
            
            previousStateRef.current = newState;
          }
        }

        setError(null);
      } catch (err) {
        console.error('Polling error:', err);
        setError(err instanceof Error ? err.message : 'Unknown error');
      }
    };

    // Initial poll
    pollState();

    // Set up interval for continuous polling
    const interval = config.intervalMs || 2000; // Default 2 seconds
    intervalRef.current = setInterval(pollState, interval);

    // Cleanup on unmount or config change
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      setIsPolling(false);
    };
  }, [config.enabled, sessionId, config.intervalMs]);

  return {
    buildState,
    isPolling,
    error,
    manualRefresh: async () => {
      // Allows manual refresh outside of interval
      if (!sessionId) return;
      
      try {
        const response = await fetch('http://localhost:8000/whoami', {
          credentials: 'include'
        });
        const data = await response.json();
        if (data.data && data.data.buildState) {
          setBuildState(data.data.buildState);
        }
      } catch (err) {
        console.error('Manual refresh error:', err);
      }
    }
  };
}
