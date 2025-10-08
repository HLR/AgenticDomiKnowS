'use client';

import { useState, useEffect, useCallback, useRef } from 'react';

interface ProcessStep {
  step: string;
  message: string;
  timestamp: string;
  status: 'pending' | 'active' | 'completed';
}

interface BuildState {
  Task_definition: string;
  graph_rag_examples: string[];
  graph_max_attempts: number;
  graph_attempt: number;
  graph_code_draft: string[];
  graph_review_notes: string[];
  graph_reviewer_agent_approved: boolean;
  graph_exe_notes: string[];
  graph_exe_agent_approved: boolean;
  human_approved: boolean;
  human_notes: string;
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
    for (let i = 0; i < state.graph_attempt; i++) {
      steps.push({
        step: `code_generation_${i + 1}`,
        message: `Attempt ${i + 1}/${state.graph_max_attempts}: Generated code draft`,
        timestamp: now,
        status: 'completed'
      });

      // Show review if exists
      if (state.graph_review_notes[i]) {
        steps.push({
          step: `ai_review_${i + 1}`,
          message: `AI Review: ${state.graph_review_notes[i].substring(0, 60)}${state.graph_review_notes[i].length > 60 ? '...' : ''}`,
          timestamp: now,
          status: 'completed'
        });
      }

      // Show execution if exists
      if (state.graph_exe_notes[i]) {
        const passed = state.graph_exe_agent_approved;
        steps.push({
          step: `execution_check_${i + 1}`,
          message: passed 
            ? `Execution Check: ✅ Passed` 
            : `Execution Check: ❌ ${state.graph_exe_notes[i].substring(0, 50)}...`,
          timestamp: now,
          status: 'completed'
        });
      }
    }

    // Step 4: Human Review Stage
    const needsHumanReview = (state.graph_reviewer_agent_approved && state.graph_exe_agent_approved) ||
                            state.graph_attempt >= state.graph_max_attempts;
    
    if (needsHumanReview && !state.human_approved) {
      steps.push({
        step: 'human_review',
        message: 'Awaiting human review and approval...',
        timestamp: now,
        status: 'active'
      });
    }

    // Step 5: Completion
    if (state.human_approved) {
      steps.push({
        step: 'completion',
        message: '✅ Task completed successfully!',
        timestamp: now,
        status: 'completed'
      });
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

  // When processing starts, show simulated progress
  useEffect(() => {
    if (isProcessing && !buildState) {
      simulateProgress();
    }
  }, [isProcessing, buildState, simulateProgress]);

  // When buildState arrives, replace simulated with real progress
  useEffect(() => {
    if (buildState) {
      // Clear simulation timers
      simulationTimerRef.current.forEach(timer => clearTimeout(timer));
      simulationTimerRef.current = [];
      
      // Show real progress
      const steps = generateProgressFromState(buildState);
      setProgressSteps(steps);
    } else if (!isProcessing) {
      setProgressSteps([]);
    }
  }, [buildState, isProcessing, generateProgressFromState]);

  // Cleanup timers on unmount
  useEffect(() => {
    return () => {
      simulationTimerRef.current.forEach(timer => clearTimeout(timer));
    };
  }, []);

  return progressSteps;
}
