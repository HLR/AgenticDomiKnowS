'use client';

interface ProcessUpdate {
  step: string;
  message: string;
  timestamp: string;
}

interface ProcessMonitorProps {
  updates: ProcessUpdate[];
  isProcessing: boolean;
}

export default function ProcessMonitor({ updates, isProcessing }: ProcessMonitorProps) {
  const getStepIcon = (step: string) => {
    switch (step) {
      case 'initialization':
        return '🚀';
      case 'ai_review_1':
      case 'ai_review_2':
      case 'ai_review_3':
        return '🤖';
      case 'code_generation':
        return '💻';
      case 'finalization':
        return '✅';
      default:
        return '⚡';
    }
  };

  const getStepColor = (step: string) => {
    switch (step) {
      case 'initialization':
        return 'bg-blue-50 text-blue-700 border-blue-200';
      case 'ai_review_1':
      case 'ai_review_2':
      case 'ai_review_3':
        return 'bg-purple-50 text-purple-700 border-purple-200';
      case 'code_generation':
        return 'bg-green-50 text-green-700 border-green-200';
      case 'finalization':
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
        {updates.map((update, index) => (
          <div 
            key={index} 
            className={`flex items-start space-x-3 p-4 rounded-xl border transition-all duration-200 hover:scale-[1.02] ${getStepColor(update.step)}`}
          >
            <span className="text-xl flex-shrink-0 mt-0.5">{getStepIcon(update.step)}</span>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium leading-relaxed">
                {update.message}
              </p>
              <div className="flex items-center mt-2">
                <span className="text-xs opacity-75">
                  {formatTime(update.timestamp)}
                </span>
                <div className="ml-auto">
                  <div className="w-2 h-2 bg-current rounded-full opacity-50"></div>
                </div>
              </div>
            </div>
          </div>
        ))}
        
        {isProcessing && (
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