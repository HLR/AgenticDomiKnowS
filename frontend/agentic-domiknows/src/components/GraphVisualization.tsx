'use client';

import { useState } from 'react';

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
}

export default function GraphVisualization({ result }: GraphVisualizationProps) {
  const [activeTab, setActiveTab] = useState<'graph' | 'code'>('graph');

  const getNodeColor = (type: string) => {
    switch (type) {
      case 'concept':
        return 'fill-blue-100 stroke-blue-400 text-blue-800';
      case 'relation':
        return 'fill-emerald-100 stroke-emerald-400 text-emerald-800';
      default:
        return 'fill-gray-100 stroke-gray-400 text-gray-800';
    }
  };

  return (
    <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-100 overflow-hidden hover:shadow-2xl transition-all duration-300">
      <div className="border-b border-gray-200 bg-gray-50/50">
        <nav className="flex">
          <button
            onClick={() => setActiveTab('graph')}
            className={`px-6 py-4 font-medium text-sm transition-all duration-200 relative ${
              activeTab === 'graph'
                ? 'text-blue-600 bg-white'
                : 'text-gray-500 hover:text-gray-700 hover:bg-white/50'
            }`}
          >
            {activeTab === 'graph' && (
              <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-blue-500 to-indigo-500"></div>
            )}
            <span className="flex items-center">
              <span className="mr-2">📊</span>
              Graph Visualization
            </span>
          </button>
          <button
            onClick={() => setActiveTab('code')}
            className={`px-6 py-4 font-medium text-sm transition-all duration-200 relative ${
              activeTab === 'code'
                ? 'text-blue-600 bg-white'
                : 'text-gray-500 hover:text-gray-700 hover:bg-white/50'
            }`}
          >
            {activeTab === 'code' && (
              <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-blue-500 to-indigo-500"></div>
            )}
            <span className="flex items-center">
              <span className="mr-2">💻</span>
              Generated Code
            </span>
          </button>
        </nav>
      </div>

      <div className="p-6">
        {activeTab === 'graph' && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h3 className="text-xl font-semibold text-gray-800">
                Knowledge Graph Structure
              </h3>
              <div className="text-sm text-gray-500 bg-gray-100 px-3 py-1 rounded-full">
                {result.nodes.length} nodes, {result.edges.length} edges
              </div>
            </div>
            
            {/* Enhanced SVG visualization */}
            <div className="bg-gradient-to-br from-gray-50 to-blue-50 rounded-xl p-6 border border-gray-200">
              <svg width="100%" height="350" viewBox="0 0 500 350" className="drop-shadow-sm">
                {/* Background grid */}
                <defs>
                  <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
                    <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#f1f5f9" strokeWidth="0.5"/>
                  </pattern>
                  <filter id="shadow" x="-50%" y="-50%" width="200%" height="200%">
                    <feDropShadow dx="2" dy="2" stdDeviation="3" floodOpacity="0.1"/>
                  </filter>
                  <marker
                    id="arrowhead"
                    markerWidth="12"
                    markerHeight="8"
                    refX="11"
                    refY="4"
                    orient="auto"
                    markerUnits="strokeWidth"
                  >
                    <path d="M0,0 L0,8 L12,4 z" fill="#64748b" />
                  </marker>
                </defs>
                
                <rect width="100%" height="100%" fill="url(#grid)" />
                
                {/* Render edges first */}
                {result.edges.map((edge) => {
                  const sourceNode = result.nodes.find(n => n.id === edge.source);
                  const targetNode = result.nodes.find(n => n.id === edge.target);
                  if (!sourceNode || !targetNode) return null;
                  
                  return (
                    <g key={edge.id}>
                      <line
                        x1={sourceNode.x * 1.2}
                        y1={sourceNode.y * 1.1}
                        x2={targetNode.x * 1.2}
                        y2={targetNode.y * 1.1}
                        stroke="#64748b"
                        strokeWidth="2.5"
                        markerEnd="url(#arrowhead)"
                        className="opacity-80"
                      />
                      <text
                        x={(sourceNode.x * 1.2 + targetNode.x * 1.2) / 2}
                        y={(sourceNode.y * 1.1 + targetNode.y * 1.1) / 2 - 8}
                        textAnchor="middle"
                        className="text-xs fill-gray-600 font-medium"
                      >
                        {edge.label}
                      </text>
                    </g>
                  );
                })}
                
                {/* Render nodes */}
                {result.nodes.map((node) => (
                  <g key={node.id} filter="url(#shadow)">
                    <rect
                      x={node.x * 1.2 - 50}
                      y={node.y * 1.1 - 20}
                      width="100"
                      height="40"
                      rx="12"
                      className={`${getNodeColor(node.type)} stroke-2`}
                    />
                    <text
                      x={node.x * 1.2}
                      y={node.y * 1.1 + 5}
                      textAnchor="middle"
                      className="text-sm font-semibold fill-current"
                    >
                      {node.label}
                    </text>
                  </g>
                ))}
              </svg>
            </div>

            {/* Enhanced Node legend */}
            <div className="bg-gray-50 rounded-xl p-5 space-y-4">
              <h4 className="font-semibold text-gray-800 flex items-center">
                <span className="mr-2">🏷️</span>
                Graph Elements
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {result.nodes.map((node) => (
                  <div key={node.id} className="flex items-center space-x-3 bg-white p-3 rounded-lg border border-gray-200">
                    <div className={`w-5 h-5 rounded-lg border-2 ${getNodeColor(node.type).replace('fill-', 'bg-').replace('stroke-', 'border-')}`}></div>
                    <div>
                      <span className="text-sm font-medium text-gray-800">{node.label}</span>
                      <p className="text-xs text-gray-500 capitalize">{node.type}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'code' && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-xl font-semibold text-gray-800">
                Generated DomiKnows Code
              </h3>
              <button 
                onClick={() => navigator.clipboard.writeText(result.code)}
                className="text-sm text-blue-600 hover:text-blue-800 font-medium flex items-center px-3 py-1 rounded-lg hover:bg-blue-50 transition-colors"
              >
                <span className="mr-1">📋</span>
                Copy Code
              </button>
            </div>
            <div className="relative">
              <pre className="bg-gradient-to-br from-gray-900 to-gray-800 text-gray-100 p-6 rounded-xl overflow-x-auto text-sm leading-relaxed shadow-2xl border border-gray-700">
                <code className="font-mono">{result.code}</code>
              </pre>
              <div className="absolute top-4 right-4 text-xs text-gray-400 bg-gray-800 px-2 py-1 rounded">
                Python
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}