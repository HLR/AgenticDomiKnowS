'use client';

interface ProcessUpdate {
  step: string;
  message: string;
  timestamp: string;
  status?: 'pending' | 'active' | 'completed';
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

interface ProcessMonitorProps {
  updates: ProcessUpdate[];
  isProcessing: boolean;
  buildState?: BuildState;
}

export default function ProcessMonitor({ updates, isProcessing, buildState }: ProcessMonitorProps) {
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
    <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-100 p-6 hover:shadow-2xl transition-all duration-300 h-fit">
      <div className="flex items-center mb-6">
        <div className="w-8 h-8 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-lg flex items-center justify-center mr-3">
          <span className="text-white text-sm font-bold">📊</span>
        </div>
        <h3 className="text-xl font-semibold text-gray-800">
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

      <div className="space-y-3 max-h-80 overflow-y-auto custom-scrollbar">
        {updates.map((update, index) => {
          const isActive = update.status === 'active';
          const isCompleted = update.status === 'completed';
          
          return (
            <div 
              key={index} 
              className={`flex items-start space-x-3 p-4 rounded-xl border transition-all duration-200 hover:scale-[1.02] ${
                isActive ? 'bg-gradient-to-r from-blue-50 to-indigo-50 border-blue-300 animate-pulse' :
                isCompleted ? getStepColor(update.step) :
                'bg-gray-50 border-gray-200 opacity-60'
              }`}
            >
              <span className="text-xl flex-shrink-0 mt-0.5">{getStepIcon(update.step)}</span>
              <div className="flex-1 min-w-0">
                <p className={`text-sm font-medium leading-relaxed ${
                  isActive ? 'text-blue-900' : 
                  isCompleted ? 'text-gray-800' : 
                  'text-gray-600'
                }`}>
                  {update.message}
                </p>
                <div className="flex items-center mt-2">
                  <span className={`text-xs ${
                    isActive ? 'text-blue-700' : 
                    isCompleted ? 'text-gray-600' : 
                    'text-gray-500'
                  }`}>
                    {formatTime(update.timestamp)}
                  </span>
                  <div className="ml-auto flex items-center space-x-1">
                    {isActive && (
                      <div className="flex space-x-1">
                        <div className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                        <div className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                        <div className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                      </div>
                    )}
                    {isCompleted && (
                      <div className="w-2 h-2 bg-current rounded-full opacity-50"></div>
                    )}
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

      {/* BuildState Summary */}
      {buildState && (
        <div className="mt-4 pt-4 border-t border-gray-100 space-y-3">
          <h4 className="text-sm font-medium text-gray-800 flex items-center">
            <span className="mr-2">📊</span>
            Build Status
          </h4>
          
          {/* Task Definition Display */}
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-3 border border-blue-100">
            <div className="text-xs font-medium text-blue-800 mb-1">Current Task</div>
            <div className="text-xs text-blue-900 font-medium">
              {buildState.Task_definition || 'No task defined'}
            </div>
          </div>

          {/* Progress Grid */}
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="bg-gray-50 rounded-lg p-2">
              <div className="font-medium text-gray-600">Attempt</div>
              <div className="text-gray-800 font-semibold">
                {buildState.graph_attempt}/{buildState.graph_max_attempts}
              </div>
            </div>
            
            <div className="bg-gray-50 rounded-lg p-2">
              <div className="font-medium text-gray-600">Code Drafts</div>
              <div className="text-gray-800 font-semibold">
                {buildState.graph_code_draft.length}
              </div>
            </div>
            
            <div className={`rounded-lg p-2 ${buildState.graph_reviewer_agent_approved ? 'bg-green-50' : 'bg-orange-50'}`}>
              <div className="font-medium text-gray-600">AI Review</div>
              <div className={buildState.graph_reviewer_agent_approved ? 'text-green-800' : 'text-orange-800'}>
                {buildState.graph_reviewer_agent_approved ? '✅ Approved' : '⏳ Pending'}
              </div>
            </div>
            
            <div className={`rounded-lg p-2 ${buildState.graph_exe_agent_approved ? 'bg-green-50' : 'bg-red-50'}`}>
              <div className="font-medium text-gray-600">Execution</div>
              <div className={buildState.graph_exe_agent_approved ? 'text-green-800' : 'text-red-800'}>
                {buildState.graph_exe_agent_approved ? '✅ Passed' : '❌ Failed'}
              </div>
            </div>
          </div>

          {/* RAG Examples Section */}
          {buildState.graph_rag_examples.length > 0 && (
            <div className="bg-purple-50 rounded-lg p-3 border border-purple-100">
              <div className="text-xs font-medium text-purple-800 mb-1.5 flex items-center">
                <span className="mr-1">📚</span>
                RAG Examples Used
              </div>
              <div className="text-xs text-purple-700">
                {buildState.graph_rag_examples.length} reference{buildState.graph_rag_examples.length !== 1 ? 's' : ''} loaded
              </div>
              {buildState.graph_rag_examples.slice(0, 3).map((example, idx) => (
                <div key={idx} className="text-xs text-purple-600 mt-1 truncate">
                  • {example}
                </div>
              ))}
              {buildState.graph_rag_examples.length > 3 && (
                <div className="text-xs text-purple-500 mt-1">
                  +{buildState.graph_rag_examples.length - 3} more...
                </div>
              )}
            </div>
          )}

          {/* Review Notes Section */}
          {buildState.graph_review_notes.length > 0 && buildState.graph_review_notes[buildState.graph_review_notes.length - 1] && (
            <div className="bg-amber-50 rounded-lg p-3 border border-amber-100">
              <div className="text-xs font-medium text-amber-800 mb-1 flex items-center">
                <span className="mr-1">📝</span>
                Latest Review Note
              </div>
              <div className="text-xs text-amber-700 italic">
                "{buildState.graph_review_notes[buildState.graph_review_notes.length - 1]}"
              </div>
            </div>
          )}

          {/* Execution Notes Section */}
          {buildState.graph_exe_notes.length > 0 && buildState.graph_exe_notes[buildState.graph_exe_notes.length - 1] && (
            <div className={`rounded-lg p-3 border ${buildState.graph_exe_agent_approved ? 'bg-green-50 border-green-100' : 'bg-red-50 border-red-100'}`}>
              <div className={`text-xs font-medium mb-1 flex items-center ${buildState.graph_exe_agent_approved ? 'text-green-800' : 'text-red-800'}`}>
                <span className="mr-1">⚡</span>
                Latest Execution Note
              </div>
              <div className={`text-xs italic ${buildState.graph_exe_agent_approved ? 'text-green-700' : 'text-red-700'}`}>
                "{buildState.graph_exe_notes[buildState.graph_exe_notes.length - 1]}"
              </div>
            </div>
          )}

          {/* Human Feedback Section */}
          {buildState.human_notes && buildState.human_notes.trim() !== '' && (
            <div className="bg-pink-50 rounded-lg p-3 border border-pink-100">
              <div className="text-xs font-medium text-pink-800 mb-1 flex items-center">
                <span className="mr-1">👤</span>
                Human Feedback
              </div>
              <div className="text-xs text-pink-700 italic">
                "{buildState.human_notes}"
              </div>
              <div className="text-xs text-pink-600 mt-1">
                Status: {buildState.human_approved ? '✅ Approved' : '🔄 Needs Revision'}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}