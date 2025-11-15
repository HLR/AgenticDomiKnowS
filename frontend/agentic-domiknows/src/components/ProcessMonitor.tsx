'use client';

import { useState } from 'react';

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

interface ProcessMonitorProps {
  updates: ProcessUpdate[];
  isProcessing: boolean;
  buildState?: BuildState;
}

export default function ProcessMonitor({ updates, isProcessing, buildState }: ProcessMonitorProps) {
  const [expandedSteps, setExpandedSteps] = useState<Set<number>>(new Set());

  const toggleStep = (index: number) => {
    const newExpanded = new Set(expandedSteps);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedSteps(newExpanded);
  };
  const getStepIcon = (step: string) => {
    switch (step) {
      case 'initialization':
      case 'rag_selection':
        return '🚀';
      case 'ai_review_1':
      case 'ai_review_2':
      case 'ai_review_3':
      case 'ai_review':
        return '🤖';
      case 'code_generation':
        return '💻';
      case 'execution_check':
        return '⚡';
      case 'human_review':
        return '👤';
      case 'finalization':
      case 'completion':
        return '✅';
      default:
        return '⚡';
    }
  };

  const getStepColor = (step: string) => {
    switch (step) {
      case 'initialization':
      case 'rag_selection':
        return 'bg-blue-50 text-blue-700 border-blue-200';
      case 'ai_review_1':
      case 'ai_review_2':
      case 'ai_review_3':
      case 'ai_review':
        return 'bg-purple-50 text-purple-700 border-purple-200';
      case 'code_generation':
        return 'bg-green-50 text-green-700 border-green-200';
      case 'execution_check':
        return 'bg-orange-50 text-orange-700 border-orange-200';
      case 'human_review':
        return 'bg-pink-50 text-pink-700 border-pink-200';
      case 'finalization':
      case 'completion':
        return 'bg-emerald-50 text-emerald-700 border-emerald-200';
      default:
        return 'bg-gray-50 text-gray-700 border-gray-200';
    }
  };

  const formatTime = (timestamp: string) => {
    try {
      return new Date(timestamp).toLocaleTimeString('en-US', { 
        hour12: false, 
        hour: '2-digit', 
        minute: '2-digit', 
        second: '2-digit' 
      });
    } catch {
      return timestamp;
    }
  };

  return (
    <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-100 p-6 hover:shadow-2xl transition-all duration-300 h-fit sticky top-6">
      <div className="flex items-center mb-6">
        <div className="w-10 h-10 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-lg flex items-center justify-center mr-3">
          <span className="text-white text-lg font-bold">📊</span>
        </div>
        <h3 className="text-2xl font-semibold text-gray-800">
          Live Progress
        </h3>
      </div>
      
      {!isProcessing && updates.length === 0 && (
        <div className="text-center py-8">
          <div className="text-4xl mb-3">⏳</div>
          <p className="text-gray-500">
            Waiting for your task...
          </p>
        </div>
      )}

      <div className="space-y-3 max-h-[600px] overflow-y-auto custom-scrollbar">
        {updates.map((update, index) => {
          const isActive = update.status === 'active';
          const isCompleted = update.status === 'completed';
          const isExpanded = expandedSteps.has(index);
          // Make AI review steps always expandable, regardless of length
          const isAIReview = update.step.startsWith('ai_review') || update.message.includes('AI Review');
          const isLongMessage = update.message.length > 100 || isAIReview;
          
          return (
            <div 
              key={index} 
              className={`border rounded-xl transition-all duration-200 hover:scale-[1.02] ${
                isActive ? 'bg-gradient-to-r from-blue-50 to-indigo-50 border-blue-300 animate-pulse' :
                isCompleted ? getStepColor(update.step) :
                'bg-gray-50 border-gray-200 opacity-60'
              }`}
            >
              <div 
                className={`flex items-start space-x-3 p-4 ${isLongMessage ? 'cursor-pointer' : ''}`}
                onClick={() => isLongMessage && toggleStep(index)}
              >
                <span className="text-2xl flex-shrink-0 mt-0.5">{getStepIcon(update.step)}</span>
                <div className="flex-1 min-w-0">
                  <div className={`text-base font-medium leading-relaxed ${
                    isActive ? 'text-blue-900' : 
                    isCompleted ? 'text-gray-800' : 
                    'text-gray-600'
                  }`}>
                    {/* Make AI review steps expandable with more lines */}
                    {(update.step.startsWith('ai_review') && isLongMessage) ? (
                      <div>
                        {isExpanded ? (
                          <div className="whitespace-pre-wrap break-words">
                            {update.message}
                          </div>
                        ) : (
                          <div className="line-clamp-3">
                            {update.message}
                          </div>
                        )}
                      </div>
                    ) : isLongMessage ? (
                      <div>
                        {isExpanded ? (
                          <div className="whitespace-pre-wrap break-words">
                            {update.message}
                          </div>
                        ) : (
                          <div className="line-clamp-2">
                            {update.message}
                          </div>
                        )}
                      </div>
                    ) : (
                      <div>{update.message}</div>
                    )}
                  </div>
                  <div className="flex items-center mt-2">
                    <span className={`text-sm ${
                      isActive ? 'text-blue-700' : 
                      isCompleted ? 'text-gray-600' : 
                      'text-gray-500'
                    }`}>
                      {formatTime(update.timestamp)}
                    </span>
                    <div className="ml-auto flex items-center space-x-2">
                      {isLongMessage && (
                        <button 
                          onClick={(e) => {
                            e.stopPropagation();
                            toggleStep(index);
                          }}
                          className="text-xs text-gray-500 hover:text-gray-700 px-2 py-1 rounded hover:bg-gray-100 transition-colors"
                          title={isExpanded ? "Collapse" : "Expand"}
                        >
                          {isExpanded ? 'Show Less ▼' : 'Show More ▶'}
                        </button>
                      )}
                      {isActive && (
                        <div className="flex space-x-1">
                          <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                          <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                          <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                        </div>
                      )}
                      {isCompleted && (
                        <div className="w-2.5 h-2.5 bg-current rounded-full opacity-50"></div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
        
        {isProcessing && updates.length === 0 && (
          <div className="flex items-center space-x-3 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-xl">
            <div className="animate-spin rounded-full h-5 w-5 border-2 border-blue-600 border-t-transparent"></div>
            <div className="flex-1">
              <p className="text-sm text-blue-700 font-medium">
                Processing your request...
              </p>
              <div className="w-full bg-blue-200 rounded-full h-1 mt-2">
                <div className="bg-blue-600 h-1 rounded-full animate-pulse" style={{ width: '60%' }}></div>
              </div>
            </div>
          </div>
        )}
      </div>

      {updates.length > 0 && !isProcessing && (
        <div className="mt-4 pt-4 border-t border-gray-100 text-center">
          <span className="text-xs text-gray-500 bg-gray-100 px-3 py-1 rounded-full">
            {updates.length} step{updates.length !== 1 ? 's' : ''} completed
          </span>
        </div>
      )}
    </div>
  );
}