from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import json
from typing import List, Dict
import uvicorn

app = FastAPI(title="Agentic DomiKnows Backend")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TaskRequest(BaseModel):
    prompt: str

class ProcessUpdate(BaseModel):
    step: str
    message: str
    timestamp: str

class GraphResult(BaseModel):
    nodes: List[Dict]
    edges: List[Dict]
    code: str

# Store active processes
active_processes = {}

@app.get("/")
async def root():
    return {"message": "Agentic DomiKnows Backend is running"}

@app.post("/api/process-task")
async def process_task(request: TaskRequest):
    task_id = f"task_{len(active_processes) + 1}"
    
    # Initialize process
    active_processes[task_id] = {
        "status": "processing",
        "updates": [],
        "result": None
    }
    
    # Start background processing
    asyncio.create_task(simulate_processing(task_id, request.prompt))
    
    return {"task_id": task_id, "status": "started"}

@app.get("/api/process-status/{task_id}")
async def get_process_status(task_id: str):
    if task_id not in active_processes:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return active_processes[task_id]

async def simulate_processing(task_id: str, prompt: str):
    """Simulate the AI processing with updates"""
    updates = [
        {"step": "initialization", "message": "Initializing DomiKnows framework analysis...", "timestamp": "0s"},
        {"step": "ai_review_1", "message": "AI Reviewer 1: Analyzing task requirements...", "timestamp": "2s"},
        {"step": "ai_review_1", "message": "AI Reviewer 1: Identified key concepts and relationships", "timestamp": "4s"},
        {"step": "ai_review_2", "message": "AI Reviewer 2: Validating ontology structure...", "timestamp": "6s"},
        {"step": "ai_review_2", "message": "AI Reviewer 2: Checking logical constraints", "timestamp": "8s"},
        {"step": "code_generation", "message": "Generating DomiKnows graph code...", "timestamp": "10s"},
        {"step": "ai_review_3", "message": "AI Reviewer 3: Code review in progress...", "timestamp": "12s"},
        {"step": "ai_review_3", "message": "AI Reviewer 3: Code approved with minor suggestions", "timestamp": "14s"},
        {"step": "finalization", "message": "Generating final graph visualization...", "timestamp": "16s"},
    ]
    
    for i, update in enumerate(updates):
        await asyncio.sleep(2)  # Simulate processing time
        active_processes[task_id]["updates"].append(update)
        
        # If this is the last update, add the result
        if i == len(updates) - 1:
            active_processes[task_id]["status"] = "completed"
            active_processes[task_id]["result"] = {
                "nodes": [
                    {"id": "1", "label": "Sentence", "type": "concept", "x": 100, "y": 100},
                    {"id": "2", "label": "Word", "type": "concept", "x": 300, "y": 100},
                    {"id": "3", "label": "Label", "type": "concept", "x": 200, "y": 200},
                    {"id": "4", "label": "contains", "type": "relation", "x": 200, "y": 150}
                ],
                "edges": [
                    {"id": "e1", "source": "1", "target": "4", "label": "subject"},
                    {"id": "e2", "source": "4", "target": "2", "label": "object"},
                    {"id": "e3", "source": "2", "target": "3", "label": "hasLabel"}
                ],
                "code": f"""# Generated DomiKnows Graph for: {prompt}
                        from domiknows.graph import Graph, Concept, Relation

                        # Create ontology graph
                        graph = Graph('task_graph')

                        # Define concepts
                        Sentence = Concept('Sentence')
                        Word = Concept('Word') 
                        Label = Concept('Label')

                        # Define relations
                        contains = Relation('Sentence', 'Word', name='contains')
                        hasLabel = Relation('Word', 'Label', name='hasLabel')

                        # Add to graph
                        graph.addConcept(Sentence)
                        graph.addConcept(Word)
                        graph.addConcept(Label)
                        graph.addRelation(contains)
                        graph.addRelation(hasLabel)
                        """
            }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)