'use client';

import { useState, useEffect } from 'react';
import { API_ENDPOINTS } from '@/config/api';

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
  sensor_attempt: number;
  sensor_codes: string[];
  sensor_human_changed: boolean;
  entire_sensor_codes: string[];
  sensor_code_outputs: string[];
  sensor_rag_examples: string[];
  property_human_text: string;
  final_code_text: string;
}

interface SensorsWorkflowProps {
  buildState: BuildState;
  sessionId: string;
  onSensorApproved: () => void;
}

export default function SensorsWorkflow({ buildState, sessionId, onSensorApproved }: SensorsWorkflowProps) {
  const [isEditMode, setIsEditMode] = useState(false);
  const [editedCode, setEditedCode] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [progressUpdates, setProgressUpdates] = useState<ProcessUpdate[]>([]);

  // Get the latest sensor code from buildState.sensor_codes array
  const latestSensorCode = buildState.sensor_codes && buildState.sensor_codes.length > 0
    ? buildState.sensor_codes[buildState.sensor_codes.length - 1]
    : '';

  const latestOutput = buildState.sensor_code_outputs && buildState.sensor_code_outputs.length > 0
    ? buildState.sensor_code_outputs[buildState.sensor_code_outputs.length - 1]
    : '';

  // Check if we're still waiting for sensor codes to be generated
  const isWaitingForSensorCode = !latestSensorCode && buildState.graph_human_approved;

  // Debug logging
  console.log('🔧 SensorsWorkflow render:');
  console.log('  - sensor_codes array:', buildState.sensor_codes);
  console.log('  - sensor_codes length:', buildState.sensor_codes?.length || 0);
  console.log('  - latestSensorCode:', latestSensorCode ? `EXISTS (${latestSensorCode.length} chars)` : 'EMPTY');
  console.log('  - isWaitingForSensorCode:', isWaitingForSensorCode);
  console.log('  - graph_human_approved:', buildState.graph_human_approved);

  // Add initial progress update
  useEffect(() => {
    setProgressUpdates([
      {
        step: 'initialization',
        message: `🚀 Sensor workflow initialized for: ${buildState.Task_definition}`,
        timestamp: new Date().toISOString(),
        status: 'completed'
      }
    ]);
  }, [buildState.Task_definition]);

  // Monitor sensor_codes changes and update progress
  useEffect(() => {
    if (buildState.sensor_codes && buildState.sensor_codes.length > 0) {
      setProgressUpdates(prev => [...prev, {
        step: 'sensor_code_update',
        message: `🔁 Sensor code generated (version ${buildState.sensor_codes.length})`,
        timestamp: new Date().toISOString(),
        status: 'completed'
      }]);
    }
  }, [buildState.sensor_codes?.length]);

  const handleEdit = () => {
    setIsEditMode(true);
    setEditedCode(latestSensorCode);
  };

  const handleCancelEdit = () => {
    setIsEditMode(false);
    setEditedCode('');
  };

  const handleSaveEdit = async () => {
    setIsSubmitting(true);
    setProgressUpdates(prev => [...prev, {
      step: 'saving_edit',
      message: '💾 Saving edited sensor code...',
      timestamp: new Date().toISOString(),
      status: 'active'
    }]);

    try {
      // Append edited code to sensor_codes array and set sensor_human_changed to true
      const updatedState = {
        ...buildState,
        sensor_codes: [...buildState.sensor_codes, editedCode],
        sensor_human_changed: true
      };

      const response = await fetch(API_ENDPOINTS.continueGraph, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify(updatedState),
      });

      if (!response.ok) {
        throw new Error('Failed to save edited sensor code');
      }

      const newState = await response.json();
      
      // Dispatch event to notify MainApp of state update
      window.dispatchEvent(new CustomEvent('buildstate-updated', { detail: newState }));

      setProgressUpdates(prev => [...prev, {
        step: 'saved_edit',
        message: '✅ Sensor code saved successfully. Waiting for backend processing...',
        timestamp: new Date().toISOString(),
        status: 'completed'
      }]);

      setIsEditMode(false);
      setEditedCode('');
    } catch (error) {
      console.error('Error saving edited sensor code:', error);
      setProgressUpdates(prev => [...prev, {
        step: 'save_error',
        message: `❌ Error saving sensor code: ${error instanceof Error ? error.message : String(error)}`,
        timestamp: new Date().toISOString(),
        status: 'pending'
      }]);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleApprove = () => {
    setProgressUpdates(prev => [...prev, {
      step: 'approved',
      message: '✅ Sensor code approved! Proceeding to final feedback...',
      timestamp: new Date().toISOString(),
      status: 'completed'
    }]);
    onSensorApproved();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-bold text-gray-800 flex items-center">
                <span className="mr-3">🔧</span>
                Sensor Workflow
              </h2>
              <p className="text-sm text-gray-600 mt-1">Review and refine your sensor code</p>
            </div>
            <div className="flex items-center space-x-4">
              <span className="text-sm text-gray-500">Session: {sessionId?.slice(0, 8)}...</span>
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          {/* Main Content Area */}
          <div className="xl:col-span-2 space-y-6">
            {/* Task Info */}
            <div className="bg-white/90 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center">
                <span className="mr-2">📋</span>
                Task Definition
              </h3>
              <p className="text-gray-700 bg-gray-50 p-4 rounded-lg">
                {buildState.Task_definition}
              </p>
            </div>

            {/* Sensor Code Display */}
            <div className="bg-white/90 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-200 p-6">
              {isWaitingForSensorCode && (
                <div className="flex flex-col items-center justify-center py-16">
                  <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-blue-500 mb-6"></div>
                  <h3 className="text-xl font-semibold text-gray-800 mb-2">Generating Sensor Code...</h3>
                  <p className="text-gray-600 text-center max-w-md">
                    The AI is now creating sensor code based on your approved graph structure. This may take a moment.
                  </p>
                  <div className="mt-6 w-full max-w-md">
                    <div className="bg-blue-100 rounded-full h-2 overflow-hidden">
                      <div className="bg-blue-500 h-full rounded-full animate-pulse" style={{ width: '70%' }}></div>
                    </div>
                  </div>
                </div>
              )}
              <div className={`${isWaitingForSensorCode ? 'hidden' : ''}`}>
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-800 flex items-center">
                  <span className="mr-2">💻</span>
                  Sensor Code
                  {buildState.sensor_codes && buildState.sensor_codes.length > 0 && (
                    <span className="ml-3 text-sm bg-blue-100 text-blue-700 px-3 py-1 rounded-full">
                      Version {buildState.sensor_codes.length}
                    </span>
                  )}
                </h3>
              </div>

              {latestSensorCode ? (
                <div className="space-y-4">
                  {isEditMode ? (
                    /* Edit Mode */
                    <div>
                      <textarea
                        value={editedCode}
                        onChange={(e) => setEditedCode(e.target.value)}
                        className="w-full h-96 p-4 bg-gray-900 text-green-400 font-mono text-sm rounded-lg border-2 border-blue-500 focus:outline-none focus:border-blue-600"
                        disabled={isSubmitting}
                      />
                      <div className="flex justify-end space-x-3 mt-4">
                        <button
                          onClick={handleCancelEdit}
                          disabled={isSubmitting}
                          className="px-6 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded-lg font-medium transition-colors disabled:opacity-50"
                        >
                          Cancel
                        </button>
                        <button
                          onClick={handleSaveEdit}
                          disabled={isSubmitting}
                          className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors disabled:opacity-50 flex items-center"
                        >
                          {isSubmitting ? (
                            <>
                              <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent mr-2"></div>
                              Saving...
                            </>
                          ) : (
                            <>
                              <span className="mr-2">💾</span>
                              Save Changes
                            </>
                          )}
                        </button>
                      </div>
                    </div>
                  ) : (
                    /* View Mode */
                    <div>
                      <div className="bg-gray-900 rounded-xl p-4 max-h-96 overflow-y-auto">
                        <pre className="text-green-400 text-sm font-mono whitespace-pre-wrap">
                          <code>{latestSensorCode}</code>
                        </pre>
                      </div>

                      {/* Output Display */}
                      {latestOutput && (
                        <div className="mt-4">
                          <h4 className="text-sm font-semibold text-gray-700 mb-2 flex items-center">
                            <span className="mr-2">📤</span>
                            Code Output
                          </h4>
                          <div className="bg-gray-900 rounded-xl p-4 max-h-48 overflow-y-auto">
                            <pre className="text-yellow-400 text-sm font-mono whitespace-pre-wrap">
                              <code>{latestOutput}</code>
                            </pre>
                          </div>
                        </div>
                      )}

                      {/* Approve/Edit Buttons */}
                      <div className="flex justify-end space-x-3 mt-6">
                        <button
                          onClick={handleEdit}
                          disabled={isSubmitting}
                          className="px-8 py-3 bg-orange-600 hover:bg-orange-700 text-white rounded-lg font-semibold transition-colors disabled:opacity-50 flex items-center shadow-lg"
                        >
                          <span className="mr-2">✏️</span>
                          Edit Code
                        </button>
                        <button
                          onClick={handleApprove}
                          disabled={isSubmitting}
                          className="px-8 py-3 bg-green-600 hover:bg-green-700 text-white rounded-lg font-semibold transition-colors disabled:opacity-50 flex items-center shadow-lg"
                        >
                          <span className="mr-2">✅</span>
                          Approve & Continue
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="flex items-center justify-center h-64 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
                  <div className="text-center">
                    <div className="text-6xl mb-4">⏳</div>
                    <p className="text-gray-600 font-medium mb-2">
                      Waiting for sensor code generation...
                    </p>
                    <p className="text-sm text-gray-500">
                      The backend is processing your graph. Sensor code will appear here.
                    </p>
                  </div>
                </div>
              )}
              </div>
            </div>

            {/* Code History */}
            {buildState.sensor_codes && buildState.sensor_codes.length > 1 && (
              <div className="bg-white/90 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-200 p-6">
                <details className="cursor-pointer">
                  <summary className="text-lg font-semibold text-gray-800 mb-3 flex items-center hover:text-blue-600">
                    <span className="mr-2">📜</span>
                    Code History ({buildState.sensor_codes.length - 1} previous versions)
                  </summary>
                  <div className="mt-4 space-y-3 max-h-96 overflow-y-auto">
                    {buildState.sensor_codes.slice(0, -1).reverse().map((code, idx) => {
                      const version = buildState.sensor_codes.length - idx - 1;
                      const output = buildState.sensor_code_outputs?.[version - 1];
                      
                      return (
                        <div key={idx} className="border border-gray-300 rounded-lg overflow-hidden">
                          <div className="bg-gray-100 px-4 py-2 text-sm font-medium text-gray-700 flex items-center justify-between">
                            <span>Version {version}</span>
                            {buildState.sensor_human_changed && idx === 0 && (
                              <span className="text-xs bg-orange-100 text-orange-700 px-2 py-1 rounded-full">
                                Human Edited
                              </span>
                            )}
                          </div>
                          <div className="bg-gray-900 p-3 max-h-48 overflow-y-auto">
                            <pre className="text-green-400 text-xs font-mono whitespace-pre-wrap">
                              <code>{code}</code>
                            </pre>
                          </div>
                          {output && (
                            <div className="bg-gray-800 p-3 max-h-32 overflow-y-auto border-t border-gray-700">
                              <p className="text-xs text-gray-400 mb-1">Output:</p>
                              <pre className="text-yellow-400 text-xs font-mono whitespace-pre-wrap">
                                <code>{output}</code>
                              </pre>
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </details>
              </div>
            )}
          </div>

          {/* Progress Monitor Sidebar */}
          <div className="xl:col-span-1">
            <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-100 p-6 sticky top-6">
              <div className="flex items-center mb-6">
                <div className="w-10 h-10 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-lg flex items-center justify-center mr-3">
                  <span className="text-white text-lg font-bold">📊</span>
                </div>
                <h3 className="text-xl font-semibold text-gray-800">
                  Progress
                </h3>
              </div>

              <div className="space-y-3 max-h-[600px] overflow-y-auto">
                {progressUpdates.map((update, index) => {
                  const isActive = update.status === 'active';
                  const isCompleted = update.status === 'completed';

                  return (
                    <div
                      key={index}
                      className={`border rounded-xl p-4 transition-all ${
                        isActive ? 'bg-gradient-to-r from-blue-50 to-indigo-50 border-blue-300 animate-pulse' :
                        isCompleted ? 'bg-green-50 border-green-200' :
                        'bg-gray-50 border-gray-200 opacity-60'
                      }`}
                    >
                      <div className="text-sm font-medium text-gray-800">
                        {update.message}
                      </div>
                      <div className="text-xs text-gray-600 mt-2">
                        {new Date(update.timestamp).toLocaleTimeString()}
                      </div>
                    </div>
                  );
                })}
              </div>

              {progressUpdates.length > 0 && (
                <div className="mt-4 pt-4 border-t border-gray-100 text-center">
                  <span className="text-xs text-gray-500 bg-gray-100 px-3 py-1 rounded-full">
                    {progressUpdates.length} step{progressUpdates.length !== 1 ? 's' : ''} completed
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
