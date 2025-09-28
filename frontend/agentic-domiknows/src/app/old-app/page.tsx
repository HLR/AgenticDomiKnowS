'use client';

import { useEffect } from 'react';

export default function OldApp() {
  useEffect(() => {
    // Redirect to the original DomiKnows website
    window.location.href = 'https://github.com/HLR/DomiKnowS';
  }, []);

  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center">
      <div className="max-w-2xl mx-auto p-8 bg-white rounded-2xl shadow-xl text-center">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">
          Redirecting to Classic DomiKnows
        </h1>
        <p className="text-lg text-gray-600 mb-6">
          You will be redirected to the original DomiKnows application...
        </p>
        <div className="text-6xl mb-6">�</div>
        <p className="text-gray-500">
          If you're not redirected automatically,{' '}
          <a 
            href="https://github.com/HLR/DomiKnowS" 
            className="text-blue-600 hover:text-blue-800 underline"
          >
            click here
          </a>
        </p>
      </div>
    </div>
  );
}