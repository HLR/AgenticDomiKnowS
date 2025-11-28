'use client';

import { useState } from 'react';
import { API_ENDPOINTS } from '@/config/api';

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

interface FinalFeedbackPageProps {
  buildState: BuildState;
  sessionId: string;
}

export default function FinalFeedbackPage({ buildState, sessionId }: FinalFeedbackPageProps) {
  const [feedback, setFeedback] = useState(buildState.property_human_text || '');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  const latestGraphCode = buildState.graph_code_draft && buildState.graph_code_draft.length > 0
    ? buildState.graph_code_draft[buildState.graph_code_draft.length - 1]
    : '';

  const latestSensorCode = buildState.entire_sensor_codes && buildState.entire_sensor_codes.length > 0
    ? buildState.entire_sensor_codes[buildState.entire_sensor_codes.length - 1]
    : buildState.sensor_codes && buildState.sensor_codes.length > 0
    ? buildState.sensor_codes[buildState.sensor_codes.length - 1]
    : '';

  const handleSubmitFeedback = async () => {
    setIsSubmitting(true);
    
    try {
      // Send the property_human_text to the backend
      const updatedState = {
        ...buildState,
        property_human_text: feedback
      };

      const response = await fetch('/api/continue-graph', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify(updatedState),
      });

      if (!response.ok) {
        throw new Error('Failed to submit feedback');
      }

      const newState = await response.json();
      
      // Dispatch event to notify MainApp of state update with final_code_text
      window.dispatchEvent(new CustomEvent('buildstate-updated', { detail: newState }));
      
      setSubmitted(true);
    } catch (error) {
      console.error('Error submitting feedback:', error);
      alert('Failed to submit feedback. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleDownloadCode = async () => {
    try {
      // Download the notebook file from the backend
      const response = await fetch(API_ENDPOINTS.downloadNotebook(sessionId), {
        method: 'GET',
        credentials: 'include',
      });

      if (!response.ok) {
        throw new Error('Failed to download notebook');
      }

      // Create a blob from the response
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${sessionId}.ipynb`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Error downloading notebook:', error);
      alert('Failed to download notebook. Please try again.');
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-bold text-gray-800 flex items-center">
                <span className="mr-3"></span>
                Model Declaration II: Property Designation
              </h2>
              <p className="text-sm text-gray-600 mt-1">Here, you should explain how to map the dataset features to graph concepts as properties.</p>
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-4 py-8">
        <div className="max-w-6xl mx-auto space-y-6">
          {/* Success Banner */}
          {/* {submitted && (
            <div className="bg-green-50 border-2 border-green-500 rounded-2xl p-6 shadow-lg animate-bounce">
              <div className="flex items-center">
                <span className="text-4xl mr-4"></span>
                <div>
                  <h3 className="text-xl font-bold text-green-800">Feedback Submitted Successfully!</h3>
                  <p className="text-green-700">Thank you for your feedback. The workflow is complete.</p>
                </div>
              </div>
            </div>
          )} */}


          {/* Graph Code */}
          <div className="bg-white/90 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-200 p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-800 flex items-center">
                <span className="mr-2">🔗</span>
                Graph Code
              </h3>
              <span className="text-sm bg-green-100 text-green-700 px-3 py-1 rounded-full">
                ✅ Approved
              </span>
            </div>
            <div className="bg-gray-900 rounded-xl p-4 max-h-96 overflow-y-auto">
              <pre className="text-green-400 text-sm font-mono whitespace-pre-wrap">
                <code>{latestGraphCode || '# No graph code generated'}</code>
              </pre>
            </div>
          </div>

          {/* Sensor Code */}
          {/* <div className="bg-white/90 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-200 p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-800 flex items-center">
                <span className="mr-2">🔧</span>
                Sensor Code
              </h3>
              <span className="text-sm bg-green-100 text-green-700 px-3 py-1 rounded-full">
                Approved
              </span>
            </div>
            <div className="bg-gray-900 rounded-xl p-4 max-h-96 overflow-y-auto">
              <pre className="text-green-400 text-sm font-mono whitespace-pre-wrap">
                <code>{latestSensorCode || '# No sensor code generated'}</code>
              </pre>
            </div>
          </div> */}

          {/* Feedback Section */}
          <div className="bg-white/90 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
              <span className="mr-2">💭</span>
              Specify how each dataset feature should be designated to the concepts as properties.
            </h3>
            <p className="text-sm text-gray-600 mb-4">
            Assign every feature in the dataset to at least one concept, and explain how each assigned feature is used to predict the labels.            </p>
            <textarea
              value={feedback}
              onChange={(e) => setFeedback(e.target.value)}
              className="w-full h-32 p-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none bg-white text-gray-900"
              disabled={submitted}
            />

            {/* Example prompts */}
            {!submitted && !feedback && (
              <div className="mt-6 pt-6 border-t border-gray-100">
                <p className="text-sm text-gray-500 mb-3">Here are some examples of Property Descriptions:</p>
                <div className="space-y-2">
                  {[
                    "Subject and facts concepts should read features called subject_text and facts_text respectively. Both these features should be used to predict if a fact is true or false.",
                    "The context and question concepts should read features called context_text and question_text respectively and question should use both of these features to predict its label.",
                    "The image concept should read a feature called image_pixels that is used to predict its number and sum of 2 digits.",
                  ].map((example, index) => (
                    <button
                      key={index}
                      onClick={() => setFeedback(example)}
                      className="block w-full text-left text-sm text-blue-600 hover:text-blue-800 hover:bg-blue-50 p-2 rounded-lg transition-colors"
                    >
                      "{example}"
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>


          {/* Action Buttons - Only Submit Button */}
          <div className="bg-white/90 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-200 p-6">
            <div className="flex flex-col md:flex-row gap-4">
              <button
                onClick={handleSubmitFeedback}
                disabled={isSubmitting || submitted}
                className="w-full bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 disabled:from-gray-400 disabled:to-gray-500 text-white font-semibold py-4 px-6 rounded-xl transition-all duration-200 transform hover:scale-[1.02] disabled:scale-100 shadow-lg flex items-center justify-center disabled:cursor-not-allowed"
              >
                {isSubmitting ? (
                  <>
                    <div className="animate-spin rounded-full h-5 w-5 border-2 border-white border-t-transparent mr-3"></div>
                    Submitting...
                  </>
                ) : submitted ? (
                  <>
                    <span className="mr-2 text-xl">✅</span>
                    Property Description Submitted
                  </>
                ) : (
                  <>
                    <span className="mr-2 text-xl">🚀</span>
                    Submit the Property Description
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Final Code Result */}
          {submitted && buildState.final_code_text && (
            <div className="bg-white/90 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
                <span className="mr-2">🎯</span>
                Download the final DomiKnowS code in a Jupyter Notebook
              </h3>
              {/* <div className="bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200 rounded-xl p-6">
                <div className="prose max-w-none text-gray-800">
                  {buildState.final_code_text}
                </div>
              </div> */}

              {/* Download Button - Below Final Code */}
              <div className="mt-6 bg-blue-50 border border-blue-200 rounded-xl p-6">
                <p className="text-sm text-gray-700 mb-4 text-center">
                  📓   For simplicity, we recommend running the notebook in{" "}
                  <a
                    href="https://colab.research.google.com/"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Google Colab
                  </a>.
                </p>
                <button
                  onClick={handleDownloadCode}
                  className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-semibold py-4 px-6 rounded-xl transition-all duration-200 transform hover:scale-[1.02] shadow-lg flex items-center justify-center"
                >
                  <span className="mr-2 text-xl">📥</span>
                  Download the final code (Jupyter Notebook)
                </button>
              </div>
            </div>
          )}

          {/* Completion Message */}
          {/* {submitted && (
            <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-2xl p-6 text-center">
              <div className="text-4xl mb-3">🎉</div>
              <h3 className="text-xl font-semibold text-gray-800 mb-2">Thank you for your feedback!</h3>
              <p className="text-gray-600">
                {buildState.final_code_text 
                  ? 'Your complete code package is ready above!' 
                  : 'Your feedback has been recorded and will help improve future generations.'}
              </p>
            </div>
          )} */}
        </div>
      </div>
    </div>
  );
}
