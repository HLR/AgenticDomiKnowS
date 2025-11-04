'use client';

import { useState, useEffect, useRef } from 'react';
import ProcessMonitor from './ProcessMonitor';

interface ProcessUpdate {
  step: string;
  message: string;
  timestamp: string;
  status?: 'pending' | 'active' | 'completed';
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

interface SensorsWorkflowProps {
  buildState: BuildState;
  sessionId: string;
}

export default function SensorsWorkflow({ buildState, sessionId }: SensorsWorkflowProps) {
  const [sensorCode, setSensorCode] = useState<string[]>([
    `# Sensor code for: ${buildState.Task_definition}\nfrom domiknows.sensor import Sensor\n\n# Define your sensors here\nsensor = Sensor('demo_sensor')\n`
  ]);
  // Keep sensorCode in sync with server-provided buildState.sensor_code
  useEffect(() => {
    if (!buildState) return;
    const incoming = buildState.sensor_code || '';
    if (incoming && incoming.trim().length > 0) {
      const last = sensorCode[sensorCode.length - 1] || '';
      if (incoming !== last) {
        setSensorCode(prev => [...prev, incoming]);
        setProgressUpdates(prev => [...prev, {
          step: 'sensor_code_update',
          message: '🔁 Sensor code updated from build state',
          timestamp: new Date().toISOString(),
          status: 'active'
        }]);
      }
    }

    // Show reviewer note if present
    const latestReview = buildState.graph_review_notes && buildState.graph_review_notes.length > 0
      ? buildState.graph_review_notes[buildState.graph_review_notes.length - 1]
      : null;
    if (latestReview) {
      setProgressUpdates(prev => [...prev, {
        step: 'review_note',
        message: `📝 Reviewer: ${latestReview}`,
        timestamp: new Date().toISOString(),
        status: 'completed'
      }]);
    }

    // If there are sensor RAG examples, add an informational update
    if (buildState.sensor_rag_examples && buildState.sensor_rag_examples.length > 0) {
      setProgressUpdates(prev => [...prev, {
        step: 'sensor_rag',
        message: `📚 Sensor RAG examples available (${buildState.sensor_rag_examples.length})`,
        timestamp: new Date().toISOString(),
        status: 'completed'
      }]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [buildState?.sensor_code, buildState?.graph_review_notes, buildState?.sensor_rag_examples]);
  const [currentIteration, setCurrentIteration] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [isEditMode, setIsEditMode] = useState(false);
  const [editedCode, setEditedCode] = useState('');
  const [progressUpdates, setProgressUpdates] = useState<ProcessUpdate[]>([]);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const maxIterations = 5; // Dummy max iterations

  // Add initial progress update
  useEffect(() => {
    setProgressUpdates([
      {
        step: 'initialization',
        message: `🚀 Starting sensor workflow for: ${buildState.Task_definition}`,
        timestamp: new Date().toISOString(),
        status: 'completed'
      }
    ]);
  }, [buildState.Task_definition]);

  // Dummy sensor workflow loop
  useEffect(() => {
    if (isRunning && !isPaused && currentIteration < maxIterations) {
      intervalRef.current = setTimeout(() => {
        const newIteration = currentIteration + 1;
        setCurrentIteration(newIteration);
        
        // Add progress update
        const newUpdate: ProcessUpdate = {
          step: `sensor_iteration_${newIteration}`,
          message: `🔄 Sensor Iteration ${newIteration}/${maxIterations}: Processing sensor code and generating models...`,
          timestamp: new Date().toISOString(),
          status: 'completed'
        };
        
        setProgressUpdates(prev => [...prev, newUpdate]);

        // Simulate code generation
        const newCode = `# Sensor iteration ${newIteration}\n${sensorCode[sensorCode.length - 1]}\n\n# Updated sensor logic\nsensor_v${newIteration} = Sensor('sensor_iteration_${newIteration}')\n`;
        setSensorCode(prev => [...prev, newCode]);

        // Pause for user to edit after each iteration
        if (newIteration < maxIterations) {
          setIsPaused(true);
          setProgressUpdates(prev => [...prev, {
            step: 'waiting_edit',
            message: `⏸️ Paused at iteration ${newIteration}. Click on the code to edit, or click Resume to continue.`,
            timestamp: new Date().toISOString(),
            status: 'active'
          }]);
        } else {
          setIsRunning(false);
          setProgressUpdates(prev => [...prev, {
            step: 'completion',
            message: `✅ Sensor workflow completed after ${newIteration} iterations!`,
            timestamp: new Date().toISOString(),
            status: 'completed'
          }]);
        }
      }, 2000); // 2 second delay between iterations
    }

    return () => {
      if (intervalRef.current) {
        clearTimeout(intervalRef.current);
      }
    };
  }, [isRunning, isPaused, currentIteration, sensorCode, maxIterations]);

  const handleStartWorkflow = () => {
    setIsRunning(true);
    setIsPaused(false);
    setProgressUpdates(prev => [...prev, {
      step: 'start',
      message: '▶️ Starting sensor workflow...',
      timestamp: new Date().toISOString(),
      status: 'active'
    }]);
  };

  const handleResumeWorkflow = () => {
    setIsPaused(false);
    setProgressUpdates(prev => prev.filter(u => u.step !== 'waiting_edit'));
    setProgressUpdates(prev => [...prev, {
      step: 'resume',
      message: `▶️ Resuming workflow from iteration ${currentIteration}...`,
      timestamp: new Date().toISOString(),
      status: 'active'
    }]);
  };

  const handleEditCode = () => {
    setIsEditMode(true);
    setEditedCode(sensorCode[sensorCode.length - 1]);
  };

  const handleSaveEdit = () => {
    const updatedCode = [...sensorCode];
    updatedCode[updatedCode.length - 1] = editedCode;
    setSensorCode(updatedCode);
    setIsEditMode(false);
    setProgressUpdates(prev => [...prev, {
      step: 'code_edited',
      message: `✏️ Code edited at iteration ${currentIteration}`,
      timestamp: new Date().toISOString(),
      status: 'completed'
    }]);

    // Persist the edited sensor code to the backend via /continue-graph
    (async () => {
      try {
        const payload = {
          ...buildState,
          sensor_code: editedCode
        };

        const resp = await fetch('http://localhost:8000/continue-graph', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          credentials: 'include',
          body: JSON.stringify(payload)
        });

        if (!resp.ok) {
          const text = await resp.text();
          setProgressUpdates(prev => [...prev, {
            step: 'save_failed',
            message: `❌ Failed to save sensor code: ${text}`,
            timestamp: new Date().toISOString(),
            status: 'pending'
          }]);
          return;
        }

        const newState = await resp.json();

        // Notify other parts of the app (MainApp may listen) that buildState changed
        try {
          window.dispatchEvent(new CustomEvent('buildstate-updated', { detail: newState }));
        } catch (e) {
          // ignore if CustomEvent isn't supported in some env
        }

        setProgressUpdates(prev => [...prev, {
          step: 'save_ok',
          message: '💾 Sensor code saved to server',
          timestamp: new Date().toISOString(),
          status: 'completed'
        }]);
      } catch (err: any) {
        setProgressUpdates(prev => [...prev, {
          step: 'save_error',
          message: `❌ Error saving sensor code: ${err?.message || String(err)}`,
          timestamp: new Date().toISOString(),
          status: 'pending'
        }]);
      }
    })();
  };

  const handleCancelEdit = () => {
    setIsEditMode(false);
    setEditedCode('');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-800">Sensors Workflow</h1>
              <p className="text-sm text-gray-600">Define and configure sensors for your knowledge graph</p>
            </div>
            <div className="flex items-center space-x-4">
              <div className="text-sm text-gray-600">
                Session: <span className="font-mono font-medium">{sessionId}</span>
              </div>
              <div className="text-sm text-gray-600">
                Iteration: <span className="font-mono font-medium">{currentIteration}/{maxIterations}</span>
              </div>
            </div>
          </div>
        </div>
        {/* Top meta: reviewer note + sensor rag examples */}
        <div className="container mx-auto px-4">
          <div className="bg-white/90 rounded-xl p-4 border border-gray-200 mb-4">
            <div className="flex justify-between items-start">
              <div>
                <div className="text-sm text-gray-600 mb-1">Latest Reviewer Note</div>
                <div className="text-sm text-amber-800 italic">{(buildState.graph_review_notes && buildState.graph_review_notes.length > 0) ? buildState.graph_review_notes[buildState.graph_review_notes.length - 1] : 'No reviewer notes yet'}</div>
              </div>
              <div className="text-sm text-gray-600 text-right">
                <div className="font-medium">Sensor RAG</div>
                <div className="text-xs text-gray-500">{buildState.sensor_rag_examples.length} example(s)</div>
              </div>
            </div>

            {buildState.sensor_rag_examples && buildState.sensor_rag_examples.length > 0 && (
              <details className="mt-3">
                <summary className="text-sm text-purple-600 cursor-pointer">View sensor RAG examples</summary>
                <div className="mt-2 space-y-2 max-h-44 overflow-y-auto">
                  {buildState.sensor_rag_examples.map((ex: string, i: number) => (
                    <pre key={i} className="text-xs bg-gray-50 p-2 rounded text-gray-700 font-mono whitespace-pre-wrap">{ex}</pre>
                  ))}
                </div>
              </details>
            )}
          </div>
        </div>
      </div>

      <div className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          {/* Main Content Area */}
          <div className="xl:col-span-2 space-y-6">
            {/* Workflow Controls */}
            <div className="bg-white/90 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-200 p-6">
              <h3 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                <span className="mr-3">🎮</span>
                Workflow Controls
              </h3>
              <div className="flex space-x-4">
                {!isRunning && currentIteration === 0 && (
                  <button
                    onClick={handleStartWorkflow}
                    className="flex-1 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white font-medium py-3 px-6 rounded-xl transition-all duration-200 transform hover:scale-[1.02] shadow-lg"
                  >
                    <div className="flex items-center justify-center">
                      <span className="mr-2">▶️</span>
                      Start Sensor Workflow
                    </div>
                  </button>
                )}
                
                {isPaused && currentIteration < maxIterations && (
                  <button
                    onClick={handleResumeWorkflow}
                    className="flex-1 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-medium py-3 px-6 rounded-xl transition-all duration-200 transform hover:scale-[1.02] shadow-lg"
                  >
                    <div className="flex items-center justify-center">
                      <span className="mr-2">▶️</span>
                      Resume Workflow
                    </div>
                  </button>
                )}

                {isRunning && !isPaused && (
                  <div className="flex-1 bg-gradient-to-r from-orange-600 to-red-600 text-white font-medium py-3 px-6 rounded-xl flex items-center justify-center">
                    <div className="flex items-center">
                      <div className="animate-spin rounded-full h-5 w-5 border-2 border-white border-t-transparent mr-3"></div>
                      <span>Running...</span>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Code Display/Editor */}
            <div className="bg-white/90 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-200 overflow-hidden">
              <div className="bg-gradient-to-r from-green-50 to-emerald-50 border-b border-gray-200">
                <div className="flex items-center justify-between p-4">
                  <h3 className="text-lg font-semibold text-gray-800 flex items-center">
                    <span className="mr-3">💻</span>
                    Sensor Code {isEditMode ? '(Editing)' : ''}
                  </h3>
                  <div className="flex items-center space-x-3">
                    <div className="text-sm text-gray-600 bg-white px-3 py-1 rounded-full">
                      Draft #{sensorCode.length}
                    </div>
                    {!isEditMode && isPaused && (
                      <button
                        onClick={handleEditCode}
                        className="text-sm bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg transition-colors"
                      >
                        ✏️ Edit Code
                      </button>
                    )}
                  </div>
                </div>
              </div>
              
              <div className="p-6">
                {isEditMode ? (
                  <div className="space-y-4">
                    <textarea
                      value={editedCode}
                      onChange={(e) => setEditedCode(e.target.value)}
                      className="w-full h-96 p-4 bg-gray-900 text-green-400 font-mono text-sm rounded-xl focus:ring-2 focus:ring-blue-500 focus:outline-none"
                      spellCheck={false}
                    />
                    <div className="flex space-x-3">
                      <button
                        onClick={handleSaveEdit}
                        className="flex-1 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white font-medium py-2 px-4 rounded-lg"
                      >
                        💾 Save Changes
                      </button>
                      <button
                        onClick={handleCancelEdit}
                        className="flex-1 bg-gradient-to-r from-gray-600 to-gray-700 hover:from-gray-700 hover:to-gray-800 text-white font-medium py-2 px-4 rounded-lg"
                      >
                        ❌ Cancel
                      </button>
                    </div>
                  </div>
                ) : (
                  <div 
                    className={`bg-gray-900 rounded-xl p-4 max-h-96 overflow-y-auto ${isPaused ? 'cursor-pointer hover:ring-2 hover:ring-blue-500' : ''}`}
                    onClick={isPaused ? handleEditCode : undefined}
                    title={isPaused ? 'Click to edit code' : ''}
                  >
                    <pre className="text-green-400 text-sm font-mono whitespace-pre-wrap">
                      <code>{sensorCode[sensorCode.length - 1]}</code>
                    </pre>
                  </div>
                )}

                {/* Code History */}
                {sensorCode.length > 1 && !isEditMode && (
                  <details className="mt-4 bg-gray-50 rounded-lg p-3">
                    <summary className="text-sm text-gray-600 cursor-pointer hover:text-gray-800 font-medium">
                      📜 View previous versions ({sensorCode.length - 1} older)
                    </summary>
                    <div className="mt-3 space-y-3 max-h-60 overflow-y-auto">
                      {sensorCode.slice(0, -1).reverse().map((code, idx) => (
                        <div key={idx} className="border border-gray-300 rounded-lg overflow-hidden">
                          <div className="bg-gray-200 px-3 py-1 text-xs font-medium text-gray-700">
                            Draft #{sensorCode.length - idx - 1}
                          </div>
                          <div className="bg-gray-900 p-3 max-h-40 overflow-y-auto">
                            <pre className="text-green-400 text-xs font-mono whitespace-pre-wrap">
                              <code>{code}</code>
                            </pre>
                          </div>
                        </div>
                      ))}
                    </div>
                  </details>
                )}
              </div>
            </div>
          </div>

          {/* Live Progress Monitor */}
          <div className="xl:col-span-1">
            <ProcessMonitor 
              updates={progressUpdates}
              isProcessing={isRunning && !isPaused}
              buildState={buildState}
            />
          </div>
        </div>

        {/* Build Status Section */}
        <div className="w-full mt-6">
          <div className="bg-white/90 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-200 p-6">
            <h3 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
              <span className="mr-3">📊</span>
              Workflow Status
            </h3>
            
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div className="bg-blue-50 rounded-lg p-3">
                <div className="font-medium text-blue-600 mb-1">Task ID</div>
                <div className="text-blue-800 font-mono text-xs truncate">
                  {buildState.Task_ID || 'N/A'}
                </div>
              </div>
              
              <div className="bg-purple-50 rounded-lg p-3">
                <div className="font-medium text-purple-600 mb-1">Current Iteration</div>
                <div className="text-purple-800 font-semibold">
                  {currentIteration}/{maxIterations}
                </div>
              </div>
              
              <div className={`rounded-lg p-3 ${isRunning && !isPaused ? 'bg-green-50' : isPaused ? 'bg-orange-50' : 'bg-gray-50'}`}>
                <div className="font-medium text-gray-600 mb-1">Status</div>
                <div className={`font-semibold ${isRunning && !isPaused ? 'text-green-800' : isPaused ? 'text-orange-800' : 'text-gray-800'}`}>
                  {isRunning && !isPaused ? '🔄 Running' : isPaused ? '⏸️ Paused' : currentIteration >= maxIterations ? '✅ Complete' : '⏳ Ready'}
                </div>
              </div>
              
              <div className="bg-indigo-50 rounded-lg p-3">
                <div className="font-medium text-indigo-600 mb-1">Code Drafts</div>
                <div className="text-indigo-800 font-semibold">
                  {sensorCode.length}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
