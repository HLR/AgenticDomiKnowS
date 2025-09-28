'use client';

import { useState } from 'react';

interface LandingPageProps {
  onChooseNewApp: () => void;
}

export default function LandingPage({ onChooseNewApp }: LandingPageProps) {
  const [isHovering, setIsHovering] = useState<string | null>(null);

  const handleOldWebsiteRedirect = () => {
    // Replace with your actual old DomiKnows website URL
    window.open('https://domiknows.github.io/', '_blank');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 flex items-center justify-center p-4">
      <div className="max-w-4xl mx-auto text-center">
        {/* Header */}
        <div className="mb-12">
          <h1 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent mb-4">
            DomiKnows
          </h1>
          <p className="text-xl md:text-2xl text-gray-600 mb-2">
            PlaceHolder
          </p>
          <p className="text-gray-500 max-w-2xl mx-auto">
            Choose your experience: Try our new agentic interface or continue with the classic version
          </p>
        </div>

        {/* Options */}
        <div className="grid md:grid-cols-2 gap-8 max-w-3xl mx-auto">
          {/* New App Option */}
          <div
            className={`bg-white rounded-2xl p-8 shadow-lg border-2 transition-all duration-300 cursor-pointer ${
              isHovering === 'new' 
                ? 'border-blue-500 shadow-xl scale-105' 
                : 'border-gray-200 hover:border-blue-300 hover:shadow-xl'
            }`}
            onMouseEnter={() => setIsHovering('new')}
            onMouseLeave={() => setIsHovering(null)}
            onClick={onChooseNewApp}
          >
            {/* <div className="text-4xl mb-4">🚀</div> */}
            <h3 className="text-2xl font-bold text-gray-900 mb-3">
              New Agentic Experience
            </h3>
            <p className="text-gray-600 mb-6 leading-relaxed">
              Experience our AI-powered interface that automatically generates knowledge graphs 
              from natural language descriptions using the DomiKnows framework.
            </p>
            <div className="space-y-2 text-sm text-gray-500">
              <div className="flex items-center justify-center">
                <span className="w-2 h-2 bg-green-500 rounded-full mr-2"></span>
                Interactive Chat Interface
              </div>
              <div className="flex items-center justify-center">
                <span className="w-2 h-2 bg-green-500 rounded-full mr-2"></span>
                Real-time Process Monitoring
              </div>
              <div className="flex items-center justify-center">
                <span className="w-2 h-2 bg-green-500 rounded-full mr-2"></span>
                Visual Graph Generation
              </div>
            </div>
          </div>

          {/* Old Website Option */}
          <div
            className={`bg-white rounded-2xl p-8 shadow-lg border-2 transition-all duration-300 cursor-pointer ${
              isHovering === 'old' 
                ? 'border-indigo-500 shadow-xl scale-105' 
                : 'border-gray-200 hover:border-indigo-300 hover:shadow-xl'
            }`}
            onMouseEnter={() => setIsHovering('old')}
            onMouseLeave={() => setIsHovering(null)}
            onClick={handleOldWebsiteRedirect}
          >
            {/* <div className="text-4xl mb-4">📚</div> */}
            <h3 className="text-2xl font-bold text-gray-900 mb-3">
              Classic Interface
            </h3>
            <p className="text-gray-600 mb-6 leading-relaxed">
              Placholder
            </p>
            <div className="space-y-2 text-sm text-gray-500">
              <div className="flex items-center justify-center">
                <span className="w-2 h-2 bg-blue-500 rounded-full mr-2"></span>
                Placholder
              </div>
              <div className="flex items-center justify-center">
                <span className="w-2 h-2 bg-blue-500 rounded-full mr-2"></span>
                Placholder
              </div>
              <div className="flex items-center justify-center">
                <span className="w-2 h-2 bg-blue-500 rounded-full mr-2"></span>
                Placholder
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-12 text-gray-500 text-sm">
          <p>
            {/* Need help deciding? The new experience is perfect for quick experimentation, 
            while the classic interface offers comprehensive control. */}
          </p>
        </div>
      </div>
    </div>
  );
}