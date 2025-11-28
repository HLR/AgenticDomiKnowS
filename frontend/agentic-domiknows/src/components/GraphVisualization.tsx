'use client';

import { useState, useEffect } from 'react';
import { API_ENDPOINTS } from '@/config/api';

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

interface GraphVisualizationProps {
  result: GraphResult;
  taskId?: string;
  graphAttempt?: number;
  codeHistory: string[];
}

export default function GraphVisualization({ result, taskId, graphAttempt, codeHistory }: GraphVisualizationProps) {
  const [activeView, setActiveView] = useState<'graph' | 'code'>('graph');
  const [graphImageUrl, setGraphImageUrl] = useState<string | null>(null);
  const [imageError, setImageError] = useState<string | null>(null);
  const [isLoadingImage, setIsLoadingImage] = useState(false);
  const [lastSuccessfulImage, setLastSuccessfulImage] = useState<string | null>(null);

  // Fetch graph image when taskId and graphAttempt change
  useEffect(() => {
    const fetchGraphImage = async () => {
      if (!taskId || graphAttempt === undefined) {
        return;
      }

      setIsLoadingImage(true);
      setImageError(null);

      try {
        const imageUrl = API_ENDPOINTS.graphImage(taskId, graphAttempt);
        const response = await fetch(imageUrl, {
          credentials: 'include',
        });

        if (response.ok) {
          // Image exists, set it as the current and last successful image
          const timestamp = new Date().getTime();
          setGraphImageUrl(`${imageUrl}?t=${timestamp}`);
          setLastSuccessfulImage(`${imageUrl}?t=${timestamp}`);
          setImageError(null);
        } else if (response.status === 404) {
          // Image doesn't exist yet
          if (lastSuccessfulImage) {
            // Keep showing the last successful image
            setGraphImageUrl(lastSuccessfulImage);
            setImageError(`Graph image for attempt ${graphAttempt} not available yet. Showing previous image.`);
          } else {
            // No previous image to show
            setGraphImageUrl(null);
            setImageError(`Graph image for attempt ${graphAttempt} is being generated. Please wait...`);
          }
        } else {
          throw new Error(`Failed to fetch image: ${response.statusText}`);
        }
      } catch (error) {
        console.error('Error fetching graph image:', error);
        
        if (lastSuccessfulImage) {
          // Keep showing the last successful image on error
          setGraphImageUrl(lastSuccessfulImage);
          setImageError(`Error fetching latest image. Showing previous image.`);
        } else {
          setGraphImageUrl(null);
          setImageError(`Error loading graph image. The image may still be generating.`);
        }
      } finally {
        setIsLoadingImage(false);
      }
    };

    fetchGraphImage();
  }, [taskId, graphAttempt]);

  return (
    <div className="bg-white/90 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-200 overflow-hidden">
      {/* Header with Toggle */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border-b border-gray-200">
        <div className="flex items-center justify-between p-4">
          <h3 className="text-lg font-semibold text-gray-800 flex items-center">
            <span className="mr-3">🔗</span>
            DomiKnowS Conceptual Graph
          </h3>
          <div className="flex space-x-2">
            <button
              onClick={() => setActiveView('graph')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                activeView === 'graph'
                  ? 'bg-blue-500 text-white shadow-md'
                  : 'bg-white text-gray-600 hover:bg-gray-50'
              }`}
            >
            Visual Graph
            </button>
            <button
              onClick={() => setActiveView('code')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                activeView === 'code'
                  ? 'bg-green-500 text-white shadow-md'
                  : 'bg-white text-gray-600 hover:bg-gray-50'
              }`}
            >
              💻 Code
            </button>
          </div>
        </div>
      </div>

      <div className="p-6">
        {activeView === 'graph' ? (
          <div className="space-y-4">
            {/* Image View */}
            {isLoadingImage ? (
              <div className="flex items-center justify-center h-96 bg-gray-50 rounded-lg">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
                  <p className="text-gray-600">Loading graph image...</p>
                </div>
              </div>
            ) : graphImageUrl ? (
              <div className="space-y-3">
                {imageError && (
                  <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3">
                    <p className="text-sm text-yellow-800 flex items-center">
                      <span className="mr-2">⚠️</span>
                      {imageError}
                    </p>
                  </div>
                )}
                <div className="bg-white border border-gray-200 rounded-lg p-4 flex items-center justify-center">
                  <img 
                    src={graphImageUrl} 
                    alt="Graph Visualization" 
                    className="max-w-full h-auto rounded-lg shadow-md"
                    onError={() => {
                      if (!imageError) {
                        setImageError('Failed to load image');
                      }
                    }}
                  />
                </div>
                <div className="bg-gray-50 rounded-lg p-3 text-sm text-gray-600">
                  <p>Attempt: <span className="font-mono font-medium">{graphAttempt}</span></p>
                </div>
              </div>
            ) : (
              <div className="flex items-center justify-center h-96 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
                <div className="text-center">
                  <div className="text-6xl mb-4">🖼️</div>
                  <p className="text-gray-600 font-medium mb-2">
                    {imageError || 'No graph image available yet'}
                  </p>
                  <p className="text-sm text-gray-500">
                    The graph visualization image will appear here once generated
                  </p>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-4">
            {/* Code View */}
            {codeHistory.length > 0 ? (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h4 className="text-sm font-medium text-gray-700">
                    Latest Generated Code
                  </h4>
                  <div className="text-sm text-gray-600 bg-white px-3 py-1 rounded-full border border-gray-200">
                    Draft #{codeHistory.length}
                  </div>
                </div>
                <div className="bg-gray-900 rounded-xl p-4 max-h-[500px] overflow-y-auto">
                  <pre className="text-green-400 text-sm font-mono whitespace-pre-wrap">
                    <code>{codeHistory[codeHistory.length - 1]}</code>
                  </pre>
                </div>
                
                {/* Code History */}
                {codeHistory.length > 1 && (
                  <details className="bg-gray-50 rounded-lg p-3">
                    <summary className="text-sm text-gray-600 cursor-pointer hover:text-gray-800 font-medium">
                      📜 View previous versions ({codeHistory.length - 1} older)
                    </summary>
                    <div className="mt-3 space-y-3 max-h-60 overflow-y-auto">
                      {codeHistory.slice(0, -1).reverse().map((code, idx) => (
                        <div key={idx} className="border border-gray-300 rounded-lg overflow-hidden">
                          <div className="bg-gray-200 px-3 py-1 text-xs font-medium text-gray-700">
                            Draft #{codeHistory.length - idx - 1}
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
            ) : (
              <div className="flex items-center justify-center h-96 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
                <div className="text-center">
                  <div className="text-6xl mb-4">💻</div>
                  <p className="text-gray-600 font-medium mb-2">
                    No code generated yet
                  </p>
                  <p className="text-sm text-gray-500">
                    The generated code will appear here once available
                  </p>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}