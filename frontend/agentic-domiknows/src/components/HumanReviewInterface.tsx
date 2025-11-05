'use client';

import { useState } from 'react';

interface BuildState {
  Task_ID?: string;
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
}

interface HumanReviewInterfaceProps {
  taskId: string;
  buildState: BuildState;
  onApproval: (approved: boolean, notes: string) => void;
}

export default function HumanReviewInterface({ taskId, buildState, onApproval }: HumanReviewInterfaceProps) {
  // Initialize feedback with existing graph_human_notes if any
  const [feedback, setFeedback] = useState(buildState.graph_human_notes || '');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleApproval = async (approved: boolean) => {
    setIsSubmitting(true);
    try {
  // Pass the feedback as graph_human_notes to the backend
  onApproval(approved, feedback);
  // Don't clear feedback - it will be preserved in buildState.graph_human_notes
    } catch (error) {
      console.error('Error submitting approval:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const latestCode = buildState.graph_code_draft[buildState.graph_code_draft.length - 1] || '';
  const latestReview = buildState.graph_review_notes[buildState.graph_review_notes.length - 1] || '';
  const latestExeNotes = buildState.graph_exe_notes[buildState.graph_exe_notes.length - 1] || '';

  return (
    <div className="bg-white/90 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-200 p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center">
          <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg flex items-center justify-center mr-3">
            <span className="text-white text-sm font-bold">👤</span>
          </div>
          <h3 className="text-xl font-semibold text-gray-800">Human Review Required</h3>
        </div>
        <div className="text-sm text-gray-500 bg-purple-100 px-3 py-1 rounded-full">
          Attempt {buildState.graph_attempt} of {buildState.graph_max_attempts}
        </div>
      </div>

      {/* Task Definition */}
      <div className="bg-blue-50 rounded-xl p-4">
        <h4 className="font-medium text-blue-900 mb-2">Original Task</h4>
        <p className="text-blue-800">{buildState.Task_definition}</p>
      </div>

      {/* Current Status */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-green-50 rounded-xl p-4">
          <h4 className="font-medium text-green-900 mb-2 flex items-center">
            <span className="mr-2">🤖</span>
            AI Reviewer Status
          </h4>
          <p className={`text-sm ${buildState.graph_reviewer_agent_approved ? 'text-green-800' : 'text-orange-800'}`}>
            {buildState.graph_reviewer_agent_approved ? '✅ Approved' : '⏳ Needs Review'}
          </p>
          {latestReview && (
            <p className="text-xs text-green-700 mt-1 italic">"{latestReview}"</p>
          )}
          {buildState.graph_review_notes.length > 1 && (
            <details className="mt-2">
              <summary className="text-xs text-green-600 cursor-pointer hover:text-green-800">
                View all {buildState.graph_review_notes.length} review notes
              </summary>
              <div className="mt-2 space-y-1 max-h-32 overflow-y-auto">
                {buildState.graph_review_notes.slice(0, -1).reverse().map((note, idx) => (
                  <p key={idx} className="text-xs text-green-600 border-l-2 border-green-300 pl-2">
                    #{buildState.graph_review_notes.length - idx - 1}: {note}
                  </p>
                ))}
              </div>
            </details>
          )}
        </div>

        {/* <div className="bg-purple-50 rounded-xl p-4">
          <h4 className="font-medium text-purple-900 mb-2 flex items-center">
            <span className="mr-2">⚡</span>
            Execution Status
          </h4>
          <p className={`text-sm ${buildState.graph_exe_agent_approved ? 'text-purple-800' : 'text-orange-800'}`}>
            {buildState.graph_exe_agent_approved ? '✅ Passed' : '❌ Failed'}
          </p>
          {latestExeNotes && (
            <p className="text-xs text-purple-700 mt-1 italic">"{latestExeNotes}"</p>
          )}
          {buildState.graph_exe_notes.length > 1 && (
            <details className="mt-2">
              <summary className="text-xs text-purple-600 cursor-pointer hover:text-purple-800">
                View all {buildState.graph_exe_notes.length} execution notes
              </summary>
              <div className="mt-2 space-y-1 max-h-32 overflow-y-auto">
                {buildState.graph_exe_notes.slice(0, -1).reverse().map((note, idx) => (
                  <p key={idx} className="text-xs text-purple-600 border-l-2 border-purple-300 pl-2">
                    #{buildState.graph_exe_notes.length - idx - 1}: {note}
                  </p>
                ))}
              </div>
            </details>
          )}
        </div> */}
      </div>

      {/* Generated Code */}
      {latestCode && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="font-medium text-gray-800 flex items-center">
              <span className="mr-2">💻</span>
              Latest Generated Code (Draft #{buildState.graph_code_draft.length})
            </h4>
            {buildState.graph_code_draft.length > 1 && (
              <span className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded-full">
                {buildState.graph_code_draft.length} versions
              </span>
            )}
          </div>
          <div className="bg-gray-900 rounded-xl p-4 max-h-60 overflow-y-auto">
            <pre className="text-green-400 text-sm font-mono">
              <code>{latestCode}</code>
            </pre>
          </div>
          
          {/* Show previous code drafts */}
          {buildState.graph_code_draft.length > 1 && (
            <details className="bg-gray-50 rounded-lg p-3">
              <summary className="text-sm text-gray-600 cursor-pointer hover:text-gray-800 font-medium">
                📜 View previous code versions ({buildState.graph_code_draft.length - 1} older)
              </summary>
              <div className="mt-3 space-y-3">
                {buildState.graph_code_draft.slice(0, -1).reverse().map((code, idx) => (
                  <div key={idx} className="border border-gray-300 rounded-lg overflow-hidden">
                    <div className="bg-gray-200 px-3 py-1 text-xs font-medium text-gray-700">
                      Draft #{buildState.graph_code_draft.length - idx - 1}
                    </div>
                    <div className="bg-gray-900 p-3 max-h-40 overflow-y-auto">
                      <pre className="text-green-400 text-xs font-mono">
                        <code>{code}</code>
                      </pre>
                    </div>
                  </div>
                ))}
              </div>
            </details>
          )}
        </div>
      )}

      {/* Feedback Input */}
      <div className="space-y-3">
        <label className="block text-sm font-medium text-gray-700">
          Your Feedback {!buildState.graph_reviewer_agent_approved || !buildState.graph_exe_agent_approved ? '(Required for revision)' : '(Optional)'}
        </label>
        <textarea
          value={feedback}
          onChange={(e) => setFeedback(e.target.value)}
          placeholder="Provide specific feedback on what needs to be improved or changed."
          className="w-full p-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-purple-500 focus:border-purple-500 resize-none bg-white text-gray-900 placeholder-gray-500"
          rows={3}
          disabled={isSubmitting}
        />
        <div className="text-xs text-gray-500 flex items-center justify-between">
          <span>💡 This feedback helps the AI understand what to improve</span>
          <span>{feedback.length} characters</span>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex space-x-4">
        <button
          onClick={() => handleApproval(true)}
          disabled={isSubmitting}
          className="flex-1 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 disabled:from-gray-400 disabled:to-gray-500 text-white font-medium py-3 px-6 rounded-xl transition-all duration-200 transform hover:scale-[1.02] disabled:scale-100"
        >
          {isSubmitting ? (
            <div className="flex items-center justify-center">
              <div className="animate-spin rounded-full h-5 w-5 border-2 border-white border-t-transparent mr-2"></div>
              Processing...
            </div>
          ) : (
            <div className="flex items-center justify-center">
              <span className="mr-2">✅</span>
              Approve & Complete
            </div>
          )}
        </button>

        <button
          onClick={() => handleApproval(false)}
          disabled={isSubmitting}
          className="flex-1 bg-gradient-to-r from-orange-600 to-red-600 hover:from-orange-700 hover:to-red-700 disabled:from-gray-400 disabled:to-gray-500 text-white font-medium py-3 px-6 rounded-xl transition-all duration-200 transform hover:scale-[1.02] disabled:scale-100"
        >
          <div className="flex items-center justify-center">
            <span className="mr-2">🔄</span>
            Request Revision
          </div>
        </button>
      </div>

      {/* RAG Examples */}
      {buildState.graph_rag_examples && buildState.graph_rag_examples.length > 0 && (
        <div className="border-t border-gray-200 pt-4">
          <h4 className="font-medium text-gray-800 mb-2 flex items-center">
            <span className="mr-2">📚</span>
            Reference Examples Used
          </h4>
          <div className="flex flex-wrap gap-2">
            {buildState.graph_rag_examples.map((example, index) => (
              <span key={index} className="text-xs bg-gray-100 text-gray-700 px-2 py-1 rounded-full">
                {example}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}