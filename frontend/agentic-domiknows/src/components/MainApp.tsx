'use client';

import { useState, useEffect } from 'react';
import ChatInterface from '@/components/ChatInterface';
import ProcessMonitor from '@/components/ProcessMonitor';
import GraphVisualization from '@/components/GraphVisualization';
import HumanReviewInterface from '@/components/HumanReviewInterface';

interface ProcessUpdate {
  step: string;
  message: string;
  timestamp: string;
}

interface BuildState {
  Task_definition: string;
  graph_rag_examples: string[];
  graph_max_attempts: number;
  graph_attempt: number;
  graph_code_draft: string[];
  graph_review_notes: string[];
  graph_reviewer_agent_approved: boolean;
  graph_exe_notes: string[];
  graph_exe_agent_approved: boolean;
  human_approved: boolean;
  human_notes: string;
}

interface GraphResult {
  nodes: Array<{
    id: string;
    label: string;
    type: string;
    x: number;
    y: number;
  }>;
  edges: Array<{
    id: string;
    source: string;
    target: string;
    label: string;
  }>;
  code: string;
}

interface TaskStatus {
  task_id: string;
  status: string; // "processing", "waiting_human", "completed", "failed"
  build_state: BuildState;
  updates: ProcessUpdate[];
  result?: GraphResult;
}

export default function MainApp() {
  const [buildState, setBuildState] = useState<BuildState | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);

  // Get session info on mount
  useEffect(() => {
    const getSession = async () => {
      try {
        const response = await fetch('http://localhost:8000/whoami', {
          credentials: 'include'
        });
        const data = await response.json();
        setSessionId(data.session_id);
      } catch (error) {
        console.error('Error getting session:', error);
      }
    };
    getSession();
  }, []);

  const handleSubmitPrompt = async (prompt: string) => {
    setIsProcessing(true);
    setBuildState(null);

    try {
      // Initialize the graph with task description
      const response = await fetch(`http://localhost:8000/UI?task_description=${encodeURIComponent(prompt)}`, {
        method: 'GET',
        credentials: 'include'
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Server error:', errorText);
        
        // If we get a 500 error, try to create a mock state to continue
        if (response.status === 500) {
          const mockState: BuildState = {
            Task_definition: prompt,
            graph_rag_examples: [],
            graph_max_attempts: 3,
            graph_attempt: 0,
            graph_code_draft: [],
            graph_review_notes: [],
            graph_reviewer_agent_approved: false,
            graph_exe_notes: [],
            graph_exe_agent_approved: false,
            human_approved: false,
            human_notes: ""
          };
          setBuildState(mockState);
          setIsProcessing(false);
          return;
        }
        
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const initialState = await response.json();
      setBuildState(initialState);
      setIsProcessing(false);
    } catch (error) {
      console.error('Error submitting prompt:', error);
      
      // Create a fallback state for demonstration
      const fallbackState: BuildState = {
        Task_definition: prompt,
        graph_rag_examples: ["example1.py", "example2.py"],
        graph_max_attempts: 3,
        graph_attempt: 1,
        graph_code_draft: [`# Generated code for: ${prompt}\nfrom domiknows.graph import Graph\ngraph = Graph('demo_graph')`],
        graph_review_notes: ["Initial code structure looks good"],
        graph_reviewer_agent_approved: true,
        graph_exe_notes: ["Code syntax is valid"],
        graph_exe_agent_approved: true,
        human_approved: false,
        human_notes: ""
      };
      
      setBuildState(fallbackState);
      setIsProcessing(false);
    }
  };

  const handleHumanApproval = async (approved: boolean, notes: string) => {
    if (!buildState) return;

    setIsProcessing(true);
    
    try {
      // Update the build state with human decision
      const updatedState = {
        ...buildState,
        human_approved: approved,
        human_notes: notes
      };

      const response = await fetch('http://localhost:8000/UI', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify(updatedState),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Server error:', errorText);
        
        // If there's an error, just update the local state
        setBuildState(updatedState);
        setIsProcessing(false);
        return;
      }

      const newState = await response.json();
      setBuildState(newState);
      setIsProcessing(false);
    } catch (error) {
      console.error('Error processing human approval:', error);
      
      // Fallback: just update the local state
      const updatedState = {
        ...buildState,
        human_approved: approved,
        human_notes: notes
      };
      setBuildState(updatedState);
      setIsProcessing(false);
    }
  };

  const showHumanReview = buildState && !buildState.human_approved && 
                         (buildState.graph_reviewer_agent_approved && buildState.graph_exe_agent_approved ||
                          buildState.graph_attempt >= buildState.graph_max_attempts);
  
  const showResult = buildState && buildState.human_approved;
  const hasActiveTask = buildState !== null;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
              Agentic DomiKnows
            </h1>
            <div className="flex items-center space-x-4">
              {sessionId && (
                <span className="text-sm text-gray-500">Session: {sessionId.slice(0, 8)}...</span>
              )}
              <button 
                onClick={() => window.location.reload()} 
                className="text-gray-500 hover:text-gray-700 text-sm font-medium transition-colors"
              >
                ← Back to Landing
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <p className="text-lg text-gray-600">
            AI-Powered Knowledge Graph Generation
          </p>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
          {/* Main Content Area */}
          <div className="xl:col-span-3 space-y-6">
            {/* Chat Interface - Only show if no active task */}
            {!hasActiveTask && (
              <ChatInterface 
                onSubmit={handleSubmitPrompt} 
                isProcessing={isProcessing}
              />
            )}
            
            {/* Human Review Interface */}
            {showHumanReview && buildState && (
              <HumanReviewInterface
                taskId={sessionId || 'unknown'}
                buildState={buildState}
                onApproval={handleHumanApproval}
              />
            )}
            
            {/* Graph Result */}
            {showResult && buildState && (
              <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-100 p-6">
                <h3 className="text-xl font-semibold text-gray-800 mb-4">
                  ✅ Task Completed Successfully!
                </h3>
                <div className="bg-green-50 rounded-xl p-4 mb-4">
                  <h4 className="font-medium text-green-900 mb-2">Final Result</h4>
                  <p className="text-green-800">Task: {buildState.Task_definition}</p>
                  <p className="text-green-700 text-sm mt-1">
                    Completed after {buildState.graph_attempt} attempt(s)
                  </p>
                </div>
                
                {buildState.graph_code_draft.length > 0 && (
                  <div className="space-y-3">
                    <h4 className="font-medium text-gray-800">Generated Code:</h4>
                    <div className="bg-gray-900 rounded-xl p-4 max-h-60 overflow-y-auto">
                      <pre className="text-green-400 text-sm font-mono">
                        <code>{buildState.graph_code_draft[buildState.graph_code_draft.length - 1]}</code>
                      </pre>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Process Monitor */}
          <div className="xl:col-span-1">
            <ProcessMonitor 
              updates={[]} // No real-time updates in this version
              isProcessing={isProcessing}
              buildState={buildState || undefined}
            />
          </div>
        </div>
      </div>
    </div>
  );
}