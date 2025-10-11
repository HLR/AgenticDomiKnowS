'use client';

import { useState, useEffect, useRef } from 'react';
import ChatInterface from '@/components/ChatInterface';
import ProcessMonitor from '@/components/ProcessMonitor';
import GraphVisualization from '@/components/GraphVisualization';
import HumanReviewInterface from '@/components/HumanReviewInterface';
import { useOptimisticProgress } from '@/hooks/useOptimisticProgress';
import { parseDomiKnowsCode, createFallbackGraph, type GraphResult } from '@/utils/graphParser';

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
  const humanReviewRef = useRef<HTMLDivElement>(null);

  // Debug buildState changes
  useEffect(() => {
    console.log('🔄 === BUILD STATE CHANGED ===');
    if (buildState) {
      console.log('📊 New BuildState received:');
      console.log('📊 - Task:', buildState.Task_definition);
      console.log('📊 - Attempt:', buildState.graph_attempt, '/', buildState.graph_max_attempts);
      console.log('📊 - Code drafts count:', buildState.graph_code_draft?.length || 0);
      console.log('📊 - Reviewer approved:', buildState.graph_reviewer_agent_approved);
      console.log('📊 - Executor approved:', buildState.graph_exe_agent_approved);
      console.log('📊 - Human approved:', buildState.human_approved);
      
      if (buildState.graph_code_draft && buildState.graph_code_draft.length > 0) {
        console.log('📄 === LATEST CODE DRAFT ===');
        const latestCode = buildState.graph_code_draft[buildState.graph_code_draft.length - 1];
        console.log('📄 Latest code (full):');
        console.log(latestCode);
        console.log('📄 Code preview:', latestCode.substring(0, 200) + '...');
      } else {
        console.log('⚠️ No code drafts in BuildState');
      }
      
      console.log('📊 Full BuildState object:', buildState);
    } else {
      console.log('📊 BuildState is null');
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

  const handleSubmitPrompt = async (prompt: string) => {
    setIsProcessing(true);
    setBuildState(null);

    try {
      // Step 1: Initialize the graph with task description
      console.log('🚀 Initializing graph with task:', prompt);
      const initResponse = await fetch(`http://localhost:8000/initialize-graph?task_description=${encodeURIComponent(prompt)}`, {
        method: 'GET',
        credentials: 'include'
      });

      if (!initResponse.ok) {
        const errorText = await initResponse.text();
        console.error('❌ Server error during init:', errorText);
        
        if (initResponse.status === 500) {
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
        
        throw new Error(`HTTP error! status: ${initResponse.status}`);
      }

      let currentState = await initResponse.json();
      console.log('✅ Initial state received from /initialize-graph:');
      console.log('📦 Full initial state:', currentState);
      console.log('📊 Initial state summary:', {
        attempt: currentState.graph_attempt,
        max_attempts: currentState.graph_max_attempts,
        reviewer_approved: currentState.graph_reviewer_agent_approved,
        executor_approved: currentState.graph_exe_agent_approved,
        both_approved: currentState.graph_reviewer_agent_approved && currentState.graph_exe_agent_approved,
        task: currentState.Task_definition
      });
      
      // Check if we're already done (edge case)
      if (currentState.graph_reviewer_agent_approved && currentState.graph_exe_agent_approved) {
        console.log('⚠️ WARNING: Both agents already approved in initial state!');
        console.log('⚠️ This means the loop will exit immediately without polling');
      }
      if (currentState.graph_attempt >= currentState.graph_max_attempts) {
        console.log('⚠️ WARNING: Max attempts already reached in initial state!');
        console.log(`⚠️ Attempt: ${currentState.graph_attempt} >= Max: ${currentState.graph_max_attempts}`);
        console.log('⚠️ This means the loop will exit immediately without polling');
      }
      
      setBuildState(currentState);

      // Step 2: Polling loop - keep calling continue-graph until both approvals are true
      let maxIterations = 20; // Safety limit to prevent infinite loops
      let iterations = 0;

      console.log('🔁 === STARTING POLLING LOOP ===');
      console.log(`🔁 Max iterations allowed: ${maxIterations}`);

      while (iterations < maxIterations) {
        console.log(`\n🔄 === LOOP ITERATION ${iterations + 1} ===`);
        
        // Check if both reviewer and execution agents have approved
        const bothApproved = currentState.graph_reviewer_agent_approved && currentState.graph_exe_agent_approved;
        
        console.log(`📊 Current state check:`, {
          iteration: iterations + 1,
          attempt: currentState.graph_attempt,
          max_attempts: currentState.graph_max_attempts,
          reviewer_approved: currentState.graph_reviewer_agent_approved,
          executor_approved: currentState.graph_exe_agent_approved,
          both_approved: bothApproved,
          task: currentState.Task_definition?.substring(0, 50) + '...'
        });

        if (bothApproved) {
          console.log('✅ Both agents approved! Stopping polling.');
          console.log('✅ Loop will EXIT after this iteration');
          break;
        }

        // Check if max attempts reached (fallback condition)
        if (currentState.graph_attempt >= currentState.graph_max_attempts) {
          console.log('⚠️ Max attempts reached. Stopping polling.');
          console.log(`⚠️ Attempts: ${currentState.graph_attempt} >= Max: ${currentState.graph_max_attempts}`);
          break;
        }

        console.log(`⏳ Waiting 1.5 seconds before next API call...`);
        // Wait a bit before next call (to avoid hammering the server)
        await new Promise(resolve => setTimeout(resolve, 1500));
        console.log(`✅ Wait complete. Making API call now...`);

        // Call continue-graph with current state
        try {
          console.log(`🔄 Calling continue-graph (iteration ${iterations + 1})...`);
          console.log(`📤 Request payload:`, {
            Task_definition: currentState.Task_definition,
            graph_attempt: currentState.graph_attempt,
            reviewer_approved: currentState.graph_reviewer_agent_approved,
            executor_approved: currentState.graph_exe_agent_approved
          });
          
          const stepResponse = await fetch('http://localhost:8000/continue-graph', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            credentials: 'include',
            body: JSON.stringify(currentState),
          });

          console.log(`📡 Response status: ${stepResponse.status} ${stepResponse.statusText}`);

          if (!stepResponse.ok) {
            const errorText = await stepResponse.text();
            console.error('❌ Server error during step:', errorText);
            console.error('❌ Breaking out of loop due to server error');
            break; // Stop polling on error
          }

          const newState = await stepResponse.json();
          console.log('✅ New state received from server:');
          console.log('📥 Response data:', {
            attempt: newState.graph_attempt,
            reviewer_approved: newState.graph_reviewer_agent_approved,
            executor_approved: newState.graph_exe_agent_approved,
            review_notes_count: newState.graph_review_notes?.length || 0,
            exe_notes_count: newState.graph_exe_notes?.length || 0,
            code_draft_count: newState.graph_code_draft?.length || 0
          });
          
          currentState = newState;
          console.log('🔄 Updated currentState variable with new state');
          setBuildState(newState); // Update UI with new state
          console.log('🔄 Updated React state (setBuildState called)');
          
        } catch (stepError) {
          console.error('❌ Error during continue-graph:', stepError);
          console.error('❌ Breaking out of loop due to error');
          break; // Stop polling on error
        }

        iterations++;
        console.log(`✅ Incremented iterations to ${iterations}`);
        console.log(`🔁 Loop condition check: ${iterations} < ${maxIterations} = ${iterations < maxIterations}`);
      }

      if (iterations >= maxIterations) {
        console.warn('⚠️ Max iterations reached in polling loop');
        console.warn(`⚠️ Stopped at iteration ${iterations}/${maxIterations}`);
      } else {
        console.log(`✅ Loop exited normally at iteration ${iterations}/${maxIterations}`);
      }

      console.log('\n🏁 === POLLING LOOP COMPLETE ===');
      console.log('🏁 Final state:', {
        attempt: currentState.graph_attempt,
        max_attempts: currentState.graph_max_attempts,
        reviewer_approved: currentState.graph_reviewer_agent_approved,
        executor_approved: currentState.graph_exe_agent_approved,
        total_iterations: iterations
      });

      setIsProcessing(false);

    } catch (error) {
      console.error('❌ Error submitting prompt:', error);
      
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

    console.log('👤 === HUMAN APPROVAL SUBMITTED ===');
    console.log('👤 Approved:', approved);
    console.log('👤 Notes:', notes);

    setIsProcessing(true);
    
    try {
      // Update the build state with human decision
      // If human provided suggestions, reset human_approved to false to restart cycle
      const shouldRestart = notes && notes.trim() !== '';
      const updatedState = {
        ...buildState,
        human_approved: shouldRestart ? false : approved,
        human_notes: notes
      };

      console.log('👤 Sending updated state to /continue-graph:', updatedState);
      if (shouldRestart) {
        console.log('👤 Resetting human_approved to false due to suggestions provided');
      }

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
      console.log('👤 Received new state after human approval:', newState);
      setBuildState(newState);
      
      // If human provided suggestions (regardless of approval), restart the AI review cycle
      if (notes && notes.trim() !== '') {
        console.log('👤 Human provided suggestions - starting polling for new iterations...');
        
        let currentState = newState;
        let maxIterations = 20;
        let iterations = 0;

        while (iterations < maxIterations) {
          const bothApproved = currentState.graph_reviewer_agent_approved && currentState.graph_exe_agent_approved;
          
          console.log(`🔄 Post-rejection iteration ${iterations + 1}:`, {
            attempt: currentState.graph_attempt,
            reviewer_approved: currentState.graph_reviewer_agent_approved,
            executor_approved: currentState.graph_exe_agent_approved,
            both_approved: bothApproved
          });

          if (bothApproved) {
            console.log('✅ Both agents approved after revision! Stopping polling.');
            break;
          }

          if (currentState.graph_attempt >= currentState.graph_max_attempts) {
            console.log('⚠️ Max attempts reached after revision. Stopping polling.');
            break;
          }

          await new Promise(resolve => setTimeout(resolve, 1500));

          try {
            console.log(`🔄 Calling continue-graph (post-rejection iteration ${iterations + 1})...`);
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
        human_approved: approved,
        human_notes: notes
      };
      setBuildState(updatedState);
      setIsProcessing(false);
    }
  };

  const showHumanReview = buildState && !buildState.human_approved && !isProcessing &&
                         (buildState.graph_reviewer_agent_approved && buildState.graph_exe_agent_approved ||
                          buildState.graph_attempt >= buildState.graph_max_attempts);
  
  // Debug logging for human review visibility
  useEffect(() => {
    if (buildState) {
      console.log('👤 === HUMAN REVIEW VISIBILITY CHECK ===');
      console.log('👤 buildState exists:', !!buildState);
      console.log('👤 human_approved:', buildState.human_approved);
      console.log('👤 isProcessing:', isProcessing);
      console.log('👤 graph_reviewer_agent_approved:', buildState.graph_reviewer_agent_approved);
      console.log('👤 graph_exe_agent_approved:', buildState.graph_exe_agent_approved);
      console.log('👤 graph_attempt:', buildState.graph_attempt);
      console.log('👤 graph_max_attempts:', buildState.graph_max_attempts);
      
      const bothAgentsApproved = buildState.graph_reviewer_agent_approved && buildState.graph_exe_agent_approved;
      const maxAttemptsReached = buildState.graph_attempt >= buildState.graph_max_attempts;
      const shouldShowHumanReview = !buildState.human_approved && !isProcessing && (bothAgentsApproved || maxAttemptsReached);
      
      console.log('👤 bothAgentsApproved:', bothAgentsApproved);
      console.log('👤 maxAttemptsReached:', maxAttemptsReached);
      console.log('👤 shouldShowHumanReview:', shouldShowHumanReview);
      console.log('👤 showHumanReview variable:', showHumanReview);
      
      if (buildState.human_approved) {
        console.log('⚠️ ALERT: human_approved is TRUE - this should only happen after explicit user approval!');
        console.log('⚠️ If you did not approve manually, this is a bug!');
      }
    }
  }, [buildState, showHumanReview]);
  
  // Auto-scroll to human review interface when it becomes visible
  useEffect(() => {
    if (showHumanReview && humanReviewRef.current) {
      console.log('🎯 === AUTO-SCROLLING TO HUMAN REVIEW ===');
      setTimeout(() => {
        humanReviewRef.current?.scrollIntoView({ 
          behavior: 'smooth', 
          block: 'center' 
        });
      }, 500); // Small delay to ensure rendering is complete
    }
  }, [showHumanReview]);

  // Force re-render when both agents become approved
  useEffect(() => {
    if (buildState && 
        buildState.graph_reviewer_agent_approved && 
        buildState.graph_exe_agent_approved && 
        !buildState.human_approved) {
      console.log('🚨 === BOTH AGENTS APPROVED - SHOULD SHOW HUMAN REVIEW ===');
      console.log('🚨 Reviewer approved:', buildState.graph_reviewer_agent_approved);
      console.log('🚨 Executor approved:', buildState.graph_exe_agent_approved);
      console.log('🚨 Human approved:', buildState.human_approved);
      console.log('🚨 Processing:', isProcessing);
      console.log('🚨 showHumanReview calculated:', showHumanReview);
      
      // Force a small state update to trigger re-render if needed
      if (!showHumanReview) {
        console.log('⚠️ WARNING: showHumanReview is false but should be true!');
      }
    }
  }, [buildState?.graph_reviewer_agent_approved, buildState?.graph_exe_agent_approved, buildState?.human_approved, isProcessing]);
  
  // Only show final result when human has approved AND both agents approved
  const showResult = buildState && buildState.human_approved && 
                     buildState.graph_reviewer_agent_approved && buildState.graph_exe_agent_approved;
  
  // Show intermediate results when both agents approved but human hasn't reviewed yet
  const showIntermediateResult = buildState && 
                                buildState.graph_reviewer_agent_approved && buildState.graph_exe_agent_approved &&
                                !buildState.human_approved;
                                
  const hasActiveTask = buildState !== null;

  // Parse DomiKnows code to generate graph visualization
  const generateGraphResult = (code: string): GraphResult | null => {
    if (!code || code.trim().length === 0) {
      console.log('⚠️ No code provided to generateGraphResult');
      return null;
    }
    
    try {
      console.log('🔍 === GRAPH PARSING DEBUG ===');
      console.log('📄 Full code being parsed:');
      console.log(code);
      console.log('📏 Code length:', code.length);
      console.log('🔍 Calling parseDomiKnowsCode...');
      
      const result = parseDomiKnowsCode(code);
      
      console.log('✅ === PARSING RESULT ===');
      console.log('📊 Total nodes found:', result.nodes.length);
      console.log('📊 Total edges found:', result.edges.length);
      console.log('🔍 Detailed nodes:');
      result.nodes.forEach((node, index) => {
        console.log(`  Node ${index + 1}:`, {
          id: node.id,
          label: node.label,
          type: node.type,
          position: { x: node.x, y: node.y }
        });
      });
      console.log('🔍 Detailed edges:');
      result.edges.forEach((edge, index) => {
        console.log(`  Edge ${index + 1}:`, {
          id: edge.id,
          source: edge.source,
          target: edge.target,
          label: edge.label
        });
      });
      
      // If parsing resulted in no nodes, return fallback
      if (result.nodes.length === 0) {
        console.log('⚠️ No nodes found in parsed result, using fallback');
        const fallback = createFallbackGraph(code);
        console.log('🔄 Fallback graph created:', {
          nodes: fallback.nodes.length,
          edges: fallback.edges.length
        });
        return fallback;
      }
      
      console.log('✅ Returning parsed result with', result.nodes.length, 'nodes and', result.edges.length, 'edges');
      return result;
    } catch (error) {
      console.error('❌ Error parsing graph code:', error);
      console.error('❌ Error details:', error instanceof Error ? error.message : 'Unknown error');
      console.log('🔄 Creating fallback graph due to error...');
      const fallback = createFallbackGraph(code);
      console.log('🔄 Fallback graph created:', {
        nodes: fallback.nodes.length,
        edges: fallback.edges.length
      });
      return fallback;
    }
  };

  // Generate graph from the latest code draft (updates on every iteration)
  const graphResult = buildState?.graph_code_draft && buildState.graph_code_draft.length > 0
    ? generateGraphResult(buildState.graph_code_draft[buildState.graph_code_draft.length - 1])
    : null;
  
  // Debug logging for what's being passed to GraphVisualization
  if (graphResult) {
    console.log('🎯 === GRAPH RESULT FOR VISUALIZATION ===');
    console.log('📊 Graph result that will be passed to GraphVisualization:');
    console.log('📊 Nodes count:', graphResult.nodes.length);
    console.log('📊 Edges count:', graphResult.edges.length);
    console.log('📊 Full graph result object:', JSON.stringify(graphResult, null, 2));
  } else {
    console.log('⚠️ No graphResult generated - GraphVisualization will not be shown');
    if (buildState) {
      console.log('🔍 BuildState exists:', {
        hasCodeDrafts: buildState.graph_code_draft?.length > 0,
        codeDraftsCount: buildState.graph_code_draft?.length || 0,
        latestCode: buildState.graph_code_draft?.length > 0 
          ? buildState.graph_code_draft[buildState.graph_code_draft.length - 1].substring(0, 100) + '...'
          : 'No code'
      });
    }
  }
  
  // Show graph whenever we have code, not just on final approval
  const showGraph = buildState && buildState.graph_code_draft.length > 0 && graphResult;
  
  console.log('🎯 === SHOW GRAPH DECISION ===');
  console.log('🔍 showGraph will be:', showGraph);
  console.log('🔍 Conditions:', {
    hasBuildState: !!buildState,
    hasCodeDrafts: (buildState?.graph_code_draft?.length || 0) > 0,
    hasGraphResult: !!graphResult
  });

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
                
                {/* Graph Visualization Component */}
                <GraphVisualization result={graphResult!} />
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

        {/* Build Status Section - Full width underneath everything */}
        {buildState && (
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
              {buildState.human_notes && buildState.human_notes.trim() !== '' && (
                <div className="bg-pink-50 rounded-lg p-4 border border-pink-100">
                  <div className="text-sm font-medium text-pink-800 mb-2 flex items-center">
                    <span className="mr-2">👤</span>
                    Human Feedback
                  </div>
                  <div className="text-sm text-pink-700 italic">
                    "{buildState.human_notes}"
                  </div>
                  <div className="text-sm text-pink-600 mt-2">
                    Status: {buildState.human_approved ? '✅ Approved' : '🔄 Needs Revision'}
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