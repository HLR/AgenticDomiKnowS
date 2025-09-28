'use client';

import { useState, useEffect } from 'react';
import ChatInterface from '@/components/ChatInterface';
import ProcessMonitor from '@/components/ProcessMonitor';
import GraphVisualization from '@/components/GraphVisualization';

interface ProcessUpdate {
  step: string;
  message: string;
  timestamp: string;
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

export default function MainApp() {
  const [taskId, setTaskId] = useState<string | null>(null);
  const [updates, setUpdates] = useState<ProcessUpdate[]>([]);
  const [result, setResult] = useState<GraphResult | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleSubmitPrompt = async (prompt: string) => {
    setIsProcessing(true);
    setUpdates([]);
    setResult(null);

    try {
      const response = await fetch('http://localhost:8000/api/process-task', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt }),
      });

      const data = await response.json();
      setTaskId(data.task_id);
    } catch (error) {
      console.error('Error submitting prompt:', error);
      setIsProcessing(false);
    }
  };

  useEffect(() => {
    if (!taskId) return;

    const pollStatus = async () => {
      try {
        const response = await fetch(`http://localhost:8000/api/process-status/${taskId}`);
        const data = await response.json();
        
        setUpdates(data.updates);
        
        if (data.status === 'completed' && data.result) {
          setResult(data.result);
          setIsProcessing(false);
        }
      } catch (error) {
        console.error('Error polling status:', error);
      }
    };

    const interval = setInterval(pollStatus, 1000);
    return () => clearInterval(interval);
  }, [taskId]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
              Agentic DomiKnows
            </h1>
            <button 
              onClick={() => window.location.reload()} 
              className="text-gray-500 hover:text-gray-700 text-sm font-medium transition-colors"
            >
              ← Back to Landing
            </button>
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
          {/* Chat Interface */}
          <div className="xl:col-span-3">
            <ChatInterface 
              onSubmit={handleSubmitPrompt} 
              isProcessing={isProcessing}
            />
            
            {result && (
              <div className="mt-6">
                <GraphVisualization result={result} />
              </div>
            )}
          </div>

          {/* Process Monitor */}
          <div className="xl:col-span-1">
            <ProcessMonitor updates={updates} isProcessing={isProcessing} />
          </div>
        </div>
      </div>
    </div>
  );
}