'use client';

import { useState, useEffect, useRef } from 'react';
import ChatInterface from '@/components/ChatInterface';
import ProcessMonitor from '@/components/ProcessMonitor';
import GraphVisualization from '@/components/GraphVisualization';
import HumanReviewInterface from '@/components/HumanReviewInterface';
import SensorsWorkflow from '@/components/SensorsWorkflow';
import { useOptimisticProgress } from '@/hooks/useOptimisticProgress';
import { parseDomiKnowsCode, createFallbackGraph, type GraphResult } from '@/utils/graphParser';

interface ProcessUpdate {
  step: string;
  message: string;
  timestamp: string;
}

interface BuildState {
  Task_ID: string;
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
  graph_human_approved: boolean;
  graph_human_notes: string;
  sensor_code: string;
  sensor_rag_examples: string[];
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
  const [activeTab, setActiveTab] = useState<'graph' | 'sensors'>('graph');
  const humanReviewRef = useRef<HTMLDivElement>(null);

  // Debug buildState changes - only log when received
  useEffect(() => {
    if (buildState) {
      console.log('📥 === BUILDSTATE RECEIVED BY FRONTEND ===');
      console.log('� Full BuildState object:', buildState);
    }
  }, [buildState]);

  // Generate optimistic progress updates from BuildState changes
  const progressUpdates = useOptimisticProgress(buildState, isProcessing);

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

  // Listen for buildstate updates that may be dispatched by subcomponents (e.g. SensorsWorkflow)
  useEffect(() => {
    const handler = (e: Event) => {
      try {
        const ev = e as CustomEvent;
        if (ev?.detail) {
          console.log('📣 buildstate-updated event received');
          setBuildState(ev.detail as BuildState);
        }
      } catch (err) {
        // ignore
      }
    };
    window.addEventListener('buildstate-updated', handler as EventListener);
    return () => window.removeEventListener('buildstate-updated', handler as EventListener);
  }, []);

  const handleSubmitPrompt = async (prompt: string) => {
    setIsProcessing(true);
    setBuildState(null);

    try {
      // Step 1: Initialize the graph with task description
      const initResponse = await fetch(`http://localhost:8000/initialize-graph?task_description=${encodeURIComponent(prompt)}`, {
        method: 'GET',
        credentials: 'include'
      });

      if (!initResponse.ok) {
        const errorText = await initResponse.text();
        console.error('❌ Server error during init:', errorText);
        
        if (initResponse.status === 500) {
          const mockState: BuildState = {
            Task_ID: 'mock_task',
            Task_definition: prompt,
            graph_rag_examples: [],
            graph_max_attempts: 3,
            graph_attempt: 0,
            graph_code_draft: [],
            graph_visual_tools: {},
            graph_review_notes: [],
            graph_reviewer_agent_approved: false,
            graph_exe_notes: [],
            graph_exe_agent_approved: false,
            graph_human_approved: false,
            graph_human_notes: "",
            sensor_code: '',
            sensor_rag_examples: []
          };
          setBuildState(mockState);
          setIsProcessing(false);
          return;
        }
        
        throw new Error(`HTTP error! status: ${initResponse.status}`);
      }

      let currentState = await initResponse.json();
      setBuildState(currentState);

      // Step 2: Polling loop - keep calling continue-graph until both approvals are true
      let maxIterations = 20;
      let iterations = 0;

      while (iterations < maxIterations) {
        const bothApproved = currentState.graph_reviewer_agent_approved && currentState.graph_exe_agent_approved;
        
        if (bothApproved) {
          break;
        }

        if (currentState.graph_attempt >= currentState.graph_max_attempts) {
          break;
        }

        await new Promise(resolve => setTimeout(resolve, 1500));

        try {
          console.log('� === SENDING BUILDSTATE TO BACKEND ===');
          console.log('📤 BuildState being sent:', currentState);
          
          const stepResponse = await fetch('http://localhost:8000/continue-graph', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            credentials: 'include',
            body: JSON.stringify(currentState),
          });

          if (!stepResponse.ok) {
            const errorText = await stepResponse.text();
            console.error('❌ Server error during step:', errorText);
            break;
          }

          const newState = await stepResponse.json();
          currentState = newState;
          setBuildState(newState);
          
        } catch (stepError) {
          console.error('❌ Error during continue-graph:', stepError);
          break;
        }

        iterations++;
      }

      setIsProcessing(false);

    } catch (error) {
      console.error('❌ Error submitting prompt:', error);
      
      // Create a fallback state for demonstration
      const fallbackState: BuildState = {
        Task_ID: 'demo_task',
        Task_definition: prompt,
        graph_rag_examples: ["example1.py", "example2.py"],
        graph_max_attempts: 3,
        graph_attempt: 1,
        graph_code_draft: [`# Generated code for: ${prompt}\nfrom domiknows.graph import Graph\ngraph = Graph('demo_graph')`],
        graph_visual_tools: {},
        graph_review_notes: ["Initial code structure looks good"],
        graph_reviewer_agent_approved: true,
        graph_exe_notes: ["Code syntax is valid"],
        graph_exe_agent_approved: true,
        graph_human_approved: false,
        graph_human_notes: "",
        sensor_code: '',
        sensor_rag_examples: []
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
      // If human provided suggestions, reset human_approved to false to restart cycle
      const shouldRestart = notes && notes.trim() !== '';
      const updatedState = {
        ...buildState,
        graph_human_approved: shouldRestart ? false : approved,
        graph_human_notes: notes
      };

      console.log('� === SENDING BUILDSTATE TO BACKEND (Human Approval) ===');
      console.log('� BuildState being sent:', updatedState);

      const response = await fetch('http://localhost:8000/continue-graph', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify(updatedState),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('❌ Server error during human approval:', errorText);
        
        // If there's an error, just update the local state
        setBuildState(updatedState);
        setIsProcessing(false);
        return;
      }

      const newState = await response.json();
      setBuildState(newState);
      
      // If human provided suggestions (regardless of approval), restart the AI review cycle
      if (notes && notes.trim() !== '') {
        let currentState = newState;
        let maxIterations = 20;
        let iterations = 0;

        while (iterations < maxIterations) {
          const bothApproved = currentState.graph_reviewer_agent_approved && currentState.graph_exe_agent_approved;
          
          if (bothApproved) {
            break;
          }

          if (currentState.graph_attempt >= currentState.graph_max_attempts) {
            break;
          }

          await new Promise(resolve => setTimeout(resolve, 1500));

          try {
            console.log('� === SENDING BUILDSTATE TO BACKEND (Post-suggestion iteration) ===');
            console.log('📤 BuildState being sent:', currentState);
            
            const stepResponse = await fetch('http://localhost:8000/continue-graph', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              credentials: 'include',
              body: JSON.stringify(currentState),
            });

            if (!stepResponse.ok) {
              console.error('❌ Server error during post-rejection step');
              break;
            }

            const stepNewState = await stepResponse.json();
            currentState = stepNewState;
            setBuildState(stepNewState);
            
          } catch (stepError) {
            console.error('❌ Error during post-rejection continue-graph:', stepError);
            break;
          }

          iterations++;
        }
      }
      
      setIsProcessing(false);
    } catch (error) {
      console.error('❌ Error processing human approval:', error);
      
      // Fallback: just update the local state
      const updatedState = {
        ...buildState,
        graph_human_approved: approved,
        graph_human_notes: notes
      };
      setBuildState(updatedState);
      setIsProcessing(false);
    }
  };

  const showHumanReview = buildState && !buildState.graph_human_approved && !isProcessing &&
                         (buildState.graph_reviewer_agent_approved && buildState.graph_exe_agent_approved ||
                          buildState.graph_attempt >= buildState.graph_max_attempts);
  
  // Auto-scroll to human review interface when it becomes visible
  useEffect(() => {
    if (showHumanReview && humanReviewRef.current) {
      setTimeout(() => {
        humanReviewRef.current?.scrollIntoView({ 
          behavior: 'smooth', 
          block: 'center' 
        });
      }, 500);
    }
  }, [showHumanReview]);

  // Auto-switch to sensors tab when human approves (don't require agent flags here)
  useEffect(() => {
    if (buildState && buildState.graph_human_approved) {
      setActiveTab('sensors');
    }
  }, [buildState?.graph_human_approved]);
  
  // Only show final result when human has approved AND both agents approved
  const showResult = buildState && buildState.graph_human_approved && 
                     buildState.graph_reviewer_agent_approved && buildState.graph_exe_agent_approved;
  
  // Show intermediate results when both agents approved but human hasn't reviewed yet
  const showIntermediateResult = buildState && 
                                buildState.graph_reviewer_agent_approved && buildState.graph_exe_agent_approved &&
                                !buildState.graph_human_approved;
                                
  const hasActiveTask = buildState !== null;

  // Parse DomiKnows code to generate graph visualization
  const generateGraphResult = (code: string): GraphResult | null => {
    if (!code || code.trim().length === 0) {
      return null;
    }
    
    try {
      const result = parseDomiKnowsCode(code);
      
      // If parsing resulted in no nodes, return fallback
      if (result.nodes.length === 0) {
        const fallback = createFallbackGraph(code);
        return fallback;
      }
      
      return result;
    } catch (error) {
      console.error('❌ Error parsing graph code:', error);
      const fallback = createFallbackGraph(code);
      return fallback;
    }
  };

  // Generate graph from the latest code draft (updates on every iteration)
  const graphResult = buildState?.graph_code_draft && buildState.graph_code_draft.length > 0
    ? generateGraphResult(buildState.graph_code_draft[buildState.graph_code_draft.length - 1])
    : null;
  
  // Show graph whenever we have code, not just on final approval
  const showGraph = buildState && buildState.graph_code_draft.length > 0 && graphResult;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-6">
              <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                Agentic DomiKnows
              </h1>
              
              {/* Tab Navigation - Show only when human has approved */}
              {buildState && buildState.graph_human_approved && (
                <div className="flex space-x-2 bg-gray-100 p-1 rounded-lg">
                  <button
                    onClick={() => setActiveTab('graph')}
                    className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                      activeTab === 'graph'
                        ? 'bg-white text-blue-600 shadow-sm'
                        : 'text-gray-600 hover:text-gray-900'
                    }`}
                  >
                    📊 Graph Code
                  </button>
                  <button
                    onClick={() => setActiveTab('sensors')}
                    className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                      activeTab === 'sensors'
                        ? 'bg-white text-indigo-600 shadow-sm'
                        : 'text-gray-600 hover:text-gray-900'
                    }`}
                  >
                    ⚙️ Sensors
                  </button>
                </div>
              )}
            </div>
            
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

        {/* Conditionally render based on active tab */}
        {activeTab === 'graph' ? (
          /* Graph Code Tab Content */
          <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
            {/* Main Content Area */}
            <div className="xl:col-span-2 space-y-6">
              {/* Chat Interface - Only show if no active task */}
              {!hasActiveTask && (
                <ChatInterface 
                  onSubmit={handleSubmitPrompt} 
                  isProcessing={isProcessing}
                />
              )}
              
              {/* Human Review Interface */}
              {showHumanReview && buildState && (
                <div ref={humanReviewRef} key={`human-review-${buildState.graph_attempt}-${buildState.graph_reviewer_agent_approved}-${buildState.graph_exe_agent_approved}`}>
                  <HumanReviewInterface
                    taskId={sessionId || 'unknown'}
                    buildState={buildState}
                    onApproval={handleHumanApproval}
                  />
                </div>
              )}
              
              {/* Graph Visualization - Shows during process and on completion */}
              {showGraph && (
              <div className="space-y-6">
                {/* Show final completion message only when human approved AND both agents approved */}
                {showResult && (
                  <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-100 p-6">
                    <h3 className="text-xl font-semibold text-gray-800 mb-4">
                      ✅ Task Completed Successfully!
                    </h3>
                    <div className="bg-green-50 rounded-xl p-4">
                      <h4 className="font-medium text-green-900 mb-2">Final Result</h4>
                      <p className="text-green-800">Task: {buildState.Task_definition}</p>
                      <p className="text-green-700 text-sm mt-1">
                        Completed after {buildState.graph_attempt} attempt(s)
                      </p>
                      <p className="text-green-700 text-sm mt-1">
                        ✅ AI Review: Approved | ✅ Execution: Passed | ✅ Human: Approved
                      </p>
                    </div>
                  </div>
                )}
                
                {/* Show intermediate completion when both agents approved but awaiting human review */}
                {showIntermediateResult && (
                  <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-100 p-6">
                    <h3 className="text-xl font-semibold text-gray-800 mb-4">
                      🎯 Agents Completed - Awaiting Human Review
                    </h3>
                    <div className="bg-blue-50 rounded-xl p-4">
                      <h4 className="font-medium text-blue-900 mb-2">Ready for Review</h4>
                      <p className="text-blue-800">Task: {buildState.Task_definition}</p>
                      <p className="text-blue-700 text-sm mt-1">
                        Completed after {buildState.graph_attempt} attempt(s)
                      </p>
                      <p className="text-green-700 text-sm mt-1">
                        ✅ AI Review: Approved | ✅ Execution: Passed | ⏳ Human: Pending
                      </p>
                    </div>
                  </div>
                )}
                
                {/* Show work in progress message during processing */}
                {!showResult && !showIntermediateResult && buildState && (
                  <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-100 p-6">
                    <h3 className="text-xl font-semibold text-gray-800 mb-4">
                      🔄 Work in Progress
                    </h3>
                    <div className="bg-blue-50 rounded-xl p-4">
                      <h4 className="font-medium text-blue-900 mb-2">Current Draft</h4>
                      <p className="text-blue-800">Task: {buildState.Task_definition}</p>
                      <p className="text-blue-700 text-sm mt-1">
                        Attempt {buildState.graph_attempt} of {buildState.graph_max_attempts}
                      </p>
                      <div className="text-sm mt-2 space-y-1">
                        <p className={buildState.graph_reviewer_agent_approved ? 'text-green-700' : 'text-orange-700'}>
                          {buildState.graph_reviewer_agent_approved ? '✅' : '⏳'} AI Review: {buildState.graph_reviewer_agent_approved ? 'Approved' : 'In Progress'}
                        </p>
                        <p className={buildState.graph_exe_agent_approved ? 'text-green-700' : 'text-orange-700'}>
                          {buildState.graph_exe_agent_approved ? '✅' : '⏳'} Execution: {buildState.graph_exe_agent_approved ? 'Passed' : 'In Progress'}
                        </p>
                      </div>
                    </div>
                  </div>
                )}
                
                {/* Graph Visualization Component with Code - Merged View */}
                <GraphVisualization 
                  result={graphResult!} 
                  taskId={buildState?.Task_ID}
                  graphAttempt={buildState?.graph_attempt}
                  codeHistory={buildState?.graph_code_draft || []}
                />
              </div>
            )}
          </div>

          {/* Live Progress Tab - Back to sidebar */}
          <div className="xl:col-span-1">
            <ProcessMonitor 
              updates={progressUpdates}
              isProcessing={isProcessing}
              buildState={buildState || undefined}
            />
          </div>
        </div>
        ) : (
          /* Sensors Tab Content */
          activeTab === 'sensors' && buildState && (
            <SensorsWorkflow 
              buildState={buildState}
              sessionId={sessionId || 'unknown'}
            />
          )
        )}

        {/* Build Status Section - Full width underneath everything - Only show in Graph tab */}
        {activeTab === 'graph' && buildState && (
          <div className="w-full mt-6">
            <div className="bg-white/90 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-200 p-6">
              <h3 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                <span className="mr-3">📊</span>
                Build Status Details
              </h3>
              
              {/* Task Definition Display */}
              <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-4 border border-blue-200 mb-4">
                <div className="text-sm font-semibold text-blue-800 mb-2">📝 Current Task</div>
                <div className="text-sm text-blue-900 font-medium leading-relaxed">
                  {buildState.Task_definition || 'No task defined'}
                </div>
              </div>

              {/* Full-width Progress Grid */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm mb-4">
                <div className="bg-gray-50 rounded-lg p-3">
                  <div className="font-medium text-gray-600 mb-1">Attempt</div>
                  <div className="text-gray-800 font-semibold">
                    {buildState.graph_attempt}/{buildState.graph_max_attempts}
                  </div>
                </div>
                
                <div className="bg-gray-50 rounded-lg p-3">
                  <div className="font-medium text-gray-600 mb-1">Code Drafts</div>
                  <div className="text-gray-800 font-semibold">
                    {buildState.graph_code_draft.length}
                  </div>
                </div>
                
                <div className={`rounded-lg p-3 ${buildState.graph_reviewer_agent_approved ? 'bg-green-50' : 'bg-orange-50'}`}>
                  <div className="font-medium text-gray-600 mb-1">AI Review</div>
                  <div className={buildState.graph_reviewer_agent_approved ? 'text-green-800' : 'text-orange-800'}>
                    {buildState.graph_reviewer_agent_approved ? '✅ Approved' : '⏳ Pending'}
                  </div>
                </div>
                
                <div className={`rounded-lg p-3 ${buildState.graph_exe_agent_approved ? 'bg-green-50' : 'bg-red-50'}`}>
                  <div className="font-medium text-gray-600 mb-1">Execution</div>
                  <div className={buildState.graph_exe_agent_approved ? 'text-green-800' : 'text-red-800'}>
                    {buildState.graph_exe_agent_approved ? '✅ Passed' : '❌ Failed'}
                  </div>
                </div>
              </div>

              {/* RAG Examples Section */}
              {buildState.graph_rag_examples.length > 0 && (
                <div className="bg-purple-50 rounded-lg p-4 border border-purple-100 mb-4">
                  <div className="text-sm font-medium text-purple-800 mb-2 flex items-center">
                    <span className="mr-2">📚</span>
                    RAG Examples Used ({buildState.graph_rag_examples.length} reference{buildState.graph_rag_examples.length !== 1 ? 's' : ''})
                  </div>
                  
                  <details className="group">
                    <summary className="text-sm text-purple-600 cursor-pointer hover:text-purple-800 font-medium list-none flex items-center">
                      <span className="mr-2 group-open:rotate-90 transition-transform">▶</span>
                      View all examples
                    </summary>
                    <div className="mt-3 space-y-3 max-h-60 overflow-y-auto">
                      {buildState.graph_rag_examples.map((example, idx) => (
                        <div key={idx} className="bg-white rounded-lg p-3 border border-purple-200">
                          <div className="text-sm font-medium text-purple-800 mb-2">
                            Example {idx + 1}:
                          </div>
                          <div className="text-sm text-purple-700 font-mono bg-purple-25 p-2 rounded max-h-32 overflow-y-auto whitespace-pre-wrap">
                            {example}
                          </div>
                        </div>
                      ))}
                    </div>
                  </details>
                </div>
              )}

              {/* Review Notes Section */}
              {buildState.graph_review_notes.length > 0 && buildState.graph_review_notes[buildState.graph_review_notes.length - 1] && (
                <div className="bg-amber-50 rounded-lg p-4 border border-amber-100 mb-4">
                  <div className="text-sm font-medium text-amber-800 mb-2 flex items-center">
                    <span className="mr-2">📝</span>
                    Latest Review Note
                  </div>
                  <div className="text-sm text-amber-700 italic">
                    "{buildState.graph_review_notes[buildState.graph_review_notes.length - 1]}"
                  </div>
                </div>
              )}

              {/* Execution Notes Section */}
              {buildState.graph_exe_notes.length > 0 && buildState.graph_exe_notes[buildState.graph_exe_notes.length - 1] && (
                <div className={`rounded-lg p-4 border mb-4 ${buildState.graph_exe_agent_approved ? 'bg-green-50 border-green-100' : 'bg-red-50 border-red-100'}`}>
                  <div className={`text-sm font-medium mb-2 flex items-center ${buildState.graph_exe_agent_approved ? 'text-green-800' : 'text-red-800'}`}>
                    <span className="mr-2">⚡</span>
                    Latest Execution Note
                  </div>
                  <div className={`text-sm italic ${buildState.graph_exe_agent_approved ? 'text-green-700' : 'text-red-700'}`}>
                    "{buildState.graph_exe_notes[buildState.graph_exe_notes.length - 1]}"
                  </div>
                </div>
              )}

              {/* Human Feedback Section */}
              {buildState.graph_human_notes && buildState.graph_human_notes.trim() !== '' && (
                <div className="bg-pink-50 rounded-lg p-4 border border-pink-100">
                  <div className="text-sm font-medium text-pink-800 mb-2 flex items-center">
                    <span className="mr-2">👤</span>
                    Human Feedback
                  </div>
                  <div className="text-sm text-pink-700 italic">
                    "{buildState.graph_human_notes}"
                  </div>
                  <div className="text-sm text-pink-600 mt-2">
                    Status: {buildState.graph_human_approved ? '✅ Approved' : '🔄 Needs Revision'}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}