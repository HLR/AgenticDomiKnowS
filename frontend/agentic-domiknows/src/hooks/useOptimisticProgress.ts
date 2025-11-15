'use client';

import { useState, useEffect, useCallback, useRef } from 'react';

interface ProcessStep {
  step: string;
  message: string;
  timestamp: string;
  status: 'pending' | 'active' | 'completed';
}

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

/**
 * Hook to create optimistic progress updates based on BuildState changes
 * Shows simulated progress during processing, then real progress from BuildState
 */
export function useOptimisticProgress(
  buildState: BuildState | null,
  isProcessing: boolean
) {
  const [progressSteps, setProgressSteps] = useState<ProcessStep[]>([]);
  const simulationTimerRef = useRef<NodeJS.Timeout[]>([]);

  // Generate progress steps based on actual BuildState
  const generateProgressFromState = useCallback((state: BuildState | null) => {
    if (!state) return [];

    const steps: ProcessStep[] = [];
    const now = new Date().toISOString();

    // Step 1: Task Initialization
    if (state.Task_definition) {
      steps.push({
        step: 'initialization',
        message: `Task initialized: "${state.Task_definition}"`,
        timestamp: now,
        status: 'completed'
      });
    }

    // Step 2: RAG Selection
    if (state.graph_rag_examples.length > 0) {
      steps.push({
        step: 'rag_selection',
        message: `Selected ${state.graph_rag_examples.length} relevant examples from database`,
        timestamp: now,
        status: 'completed'
      });
    }

    // Step 3: Code Generation Attempts
    const totalAttempts = Math.max(state.graph_attempt, state.graph_max_attempts);
    for (let i = 0; i < state.graph_attempt; i++) {
      steps.push({
        step: `code_generation_${i + 1}`,
        message: `Attempt ${i + 1}/${totalAttempts}: Generated code draft`,
        timestamp: now,
        status: 'completed'
      });

      // Show review if exists
      if (state.graph_review_notes[i]) {
        const reviewNote = state.graph_review_notes[i];
        const isApproved = reviewNote.toLowerCase().includes('approve');
        steps.push({
          step: `ai_review_${i + 1}`,
          message: isApproved 
            ? `AI Review (Attempt ${i + 1}): ✅ Approved - ${reviewNote.substring(0, 50)}${reviewNote.length > 50 ? '...' : ''}`
            : `AI Review (Attempt ${i + 1}): 🔄 ${reviewNote.substring(0, 60)}${reviewNote.length > 60 ? '...' : ''}`,
          timestamp: now,
          status: 'completed'
        });
      }

      // Show execution if exists
      if (state.graph_exe_notes[i]) {
        const exeNote = state.graph_exe_notes[i];
        const passed = !exeNote.toLowerCase().includes('error') && !exeNote.toLowerCase().includes('failed');
        steps.push({
          step: `execution_check_${i + 1}`,
          message: passed 
            ? `Execution (Attempt ${i + 1}): ✅ Passed validation` 
            : `Execution (Attempt ${i + 1}): ❌ ${exeNote.substring(0, 50)}${exeNote.length > 50 ? '...' : ''}`,
          timestamp: now,
          status: 'completed'
        });
      }
    }

    // Step 3.5: Show if currently waiting for next iteration
    const bothApproved = state.graph_reviewer_agent_approved && state.graph_exe_agent_approved;
    const maxAttemptsReached = state.graph_attempt >= state.graph_max_attempts;
    
    if (!bothApproved && !maxAttemptsReached && state.graph_attempt > 0) {
      const totalAttempts = Math.max(state.graph_attempt + 1, state.graph_max_attempts);
      steps.push({
        step: 'iterating',
        message: `Processing next iteration (${state.graph_attempt + 1}/${totalAttempts})...`,
        timestamp: now,
        status: 'active'
      });
    }

    // Step 4: Human Review Stage
    const needsHumanReview = (state.graph_reviewer_agent_approved && state.graph_exe_agent_approved) ||
                            state.graph_attempt >= state.graph_max_attempts;
    
    if (needsHumanReview && !state.graph_human_approved) {
      steps.push({
        step: 'human_review',
        message: 'Awaiting human review and approval...',
        timestamp: now,
        status: 'active'
      });
    }

  // Step 5: Completion - ONLY when ALL THREE conditions are met
  // Must have: graph_human_approved: true AND graph_reviewer_agent_approved: true AND graph_exe_agent_approved: true
  const isFullyCompleted = state.graph_human_approved === true && 
               state.graph_reviewer_agent_approved === true && 
               state.graph_exe_agent_approved === true;
    
  // Debug logging for completion logic
  console.log('🏁 === COMPLETION CHECK ===');
  console.log('🏁 graph_human_approved:', state.graph_human_approved);
    console.log('🏁 graph_reviewer_agent_approved:', state.graph_reviewer_agent_approved);
    console.log('🏁 graph_exe_agent_approved:', state.graph_exe_agent_approved);
  console.log('🏁 isFullyCompleted (all three must be true):', isFullyCompleted);
    
    if (isFullyCompleted) {
      console.log('✅ All three approvals confirmed - showing completion step');
      steps.push({
        step: 'completion',
        message: '✅ Task completed successfully!',
        timestamp: now,
        status: 'completed'
      });
    } else {
      console.log('⏳ Not all approvals completed yet - no completion step');
    }

    return steps;
  }, []);

  // Simulate progress during API call
  const simulateProgress = useCallback(() => {
    // Clear any existing timers
    simulationTimerRef.current.forEach(timer => clearTimeout(timer));
    simulationTimerRef.current = [];

    const simulatedSteps: ProcessStep[] = [];
    
    // Step 1: Initialize (immediate)
    simulatedSteps.push({
      step: 'initializing',
      message: 'Initializing graph generation...',
      timestamp: new Date().toISOString(),
      status: 'active'
    });
    setProgressSteps([...simulatedSteps]);

    // Step 2: RAG Selection (after 1s)
    const timer1 = setTimeout(() => {
      simulatedSteps[0].status = 'completed';
      simulatedSteps.push({
        step: 'rag_selecting',
        message: 'Searching for relevant examples...',
        timestamp: new Date().toISOString(),
        status: 'active'
      });
      setProgressSteps([...simulatedSteps]);
    }, 1000);
    simulationTimerRef.current.push(timer1);

    // Step 3: Code Generation (after 2.5s)
    const timer2 = setTimeout(() => {
      simulatedSteps[1].status = 'completed';
      simulatedSteps.push({
        step: 'generating',
        message: 'Generating knowledge graph code...',
        timestamp: new Date().toISOString(),
        status: 'active'
      });
      setProgressSteps([...simulatedSteps]);
    }, 2500);
    simulationTimerRef.current.push(timer2);

    // Step 4: AI Review (after 4s)
    const timer3 = setTimeout(() => {
      simulatedSteps[2].status = 'completed';
      simulatedSteps.push({
        step: 'reviewing',
        message: 'AI agent reviewing generated code...',
        timestamp: new Date().toISOString(),
        status: 'active'
      });
      setProgressSteps([...simulatedSteps]);
    }, 4000);
    simulationTimerRef.current.push(timer3);

    // Step 5: Execution Check (after 5s)
    const timer4 = setTimeout(() => {
      simulatedSteps[3].status = 'completed';
      simulatedSteps.push({
        step: 'executing',
        message: 'Validating code execution...',
        timestamp: new Date().toISOString(),
        status: 'active'
      });
      setProgressSteps([...simulatedSteps]);
    }, 5000);
    simulationTimerRef.current.push(timer4);

  }, []);

  // Update progress when buildState changes (real-time updates during polling)
  useEffect(() => {
    if (buildState) {
      const steps = generateProgressFromState(buildState);
      setProgressSteps(steps);
    } else if (!isProcessing) {
      setProgressSteps([]);
    }
  }, [buildState, isProcessing, generateProgressFromState]);

  // Show initial processing step when started
  useEffect(() => {
    if (isProcessing && !buildState) {
      setProgressSteps([{
        step: 'processing',
        message: 'Initializing task...',
        timestamp: new Date().toISOString(),
        status: 'active'
      }]);
    }
  }, [isProcessing, buildState]);

  // Cleanup timers on unmount
  useEffect(() => {
    return () => {
      simulationTimerRef.current.forEach(timer => clearTimeout(timer));
    };
  }, []);

  return progressSteps;
}
