'use client';

import { useState } from 'react';

interface ChatInterfaceProps {
  onSubmit: (prompt: string) => void;
  isProcessing: boolean;
}

export default function ChatInterface({ onSubmit, isProcessing }: ChatInterfaceProps) {
  const [prompt, setPrompt] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (prompt.trim() && !isProcessing) {
      onSubmit(prompt.trim());
      setPrompt('');
    }
  };

  return (
    <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-100 p-6 hover:shadow-2xl transition-all duration-300">
      <div className="flex items-center mb-6">
        <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-indigo-500 rounded-lg flex items-center justify-center mr-3">
          <span className="text-white text-sm font-bold">💬</span>
        </div>
        <h2 className="text-xl font-semibold text-gray-800">
          Describe Your Task
        </h2>
      </div>
      
      <form onSubmit={handleSubmit} className="space-y-5">
        <div className="relative">
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Describe the conceptual graph you want to create using the DomiKnows framework. Be as detailed as possible..."
            className="w-full p-4 border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none transition-all duration-200 bg-gray-50 focus:bg-white text-gray-900 placeholder-gray-500 font-medium"
            rows={5}
            disabled={isProcessing}
          />
          <div className="absolute bottom-3 right-3 text-xs text-gray-400">
            {prompt.length}/1000
          </div>
        </div>
        
        <button
          type="submit"
          disabled={!prompt.trim() || isProcessing}
          className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 disabled:from-gray-300 disabled:to-gray-400 text-white font-medium py-3.5 px-6 rounded-xl transition-all duration-200 transform hover:scale-[1.02] disabled:scale-100 disabled:cursor-not-allowed shadow-lg hover:shadow-xl"
        >
          {isProcessing ? (
            <div className="flex items-center justify-center">
              <div className="animate-spin rounded-full h-5 w-5 border-2 border-white border-t-transparent mr-3"></div>
              <span>Generating Graph...</span>
            </div>
          ) : (
            <div className="flex items-center justify-center">
              
              <span>Generate Graph</span>
            </div>
          )}
        </button>
      </form>

      {/* Example prompts */}
      {!isProcessing && !prompt && (
        <div className="mt-6 pt-6 border-t border-gray-100">
          <p className="text-sm text-gray-500 mb-3">Here are some examples of task descriptions:</p>
          <div className="space-y-2">
            {[
              "In our dataset of emails labeled spam or legitimate, each record includes a header, body, and spam label. We want to build two independent models that, given an email header and body, each predict whether the email is spam.	No constraints are needed.",
              "The Belief-Consistent Question Answering task in NLP aims to develop a system that can accurately answer questions based on the BeliefBank dataset while ensuring the answers are consistent with the beliefs expressed in the dataset.Here, we have a collection of entities and a list of sentences that describe those entities. Some of these sentences are correct, and some are not. Additionally, we have a graph of constraints that describe a relation between two sentence that describe a positive or negative correlation. For example, if an entity is a bird, then this entity can also fly. On the other hand, if an entity is a reptile, then it cannot fly.  The constraints here are the relationships between sentences that are provided in the constarint graph. These constraints are either negative or positive. Suppose a model decides an sentence applies to an entity. In that case, all sentences with a positive correlation must also be correct, and all those with a negative relationship with this attribute must be false.",
                "This task aims to solve the Eight Queens Puzzle using AI/Machine learning techniques in the domain of Constraint Satisfaction Problems (CSP) based on a Partially Filled Chess Board dataset.\n" +
                "\n" +
                "Here, the input would be a chess board with some queens already placed on it. The chess board with its cells should be modeled, and the remaining queens should be correctly placed."+
                "No two queens should be able to attack each other.",
            ].map((example, index) => (
              <button
                key={index}
                onClick={() => setPrompt(example)}
                className="block w-full text-left text-sm text-blue-600 hover:text-blue-800 hover:bg-blue-50 p-2 rounded-lg transition-colors"
              >
                "{example}"
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}