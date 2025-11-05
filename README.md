# Agentic DomiKnows

AI-powered knowledge graph generation system with human-in-the-loop validation. Build DomiKnows graphs using LLM agents with automated review, execution validation, and sensor workflow management.

## Prerequisites

- Python 3.9 or above
- Node.js 16+ and npm
- OpenAI API KEY
  
## Setup & Run

### Backend Setup
0. Add your OpenAI key by creating a file named `.env` in the folder `Agent/LLM/` and set your key by writing the following:
```
OPENAI_API_KEY=<openai key>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install Python dependencies:
```bash
pip install -r backend-requirements.txt
pip install -r domiknows-requirements.txt
pip install -r agent-requirements.com
```

3. Run the backend server:
```bash
cd server
python main.py
```

The backend API will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd frontend/agentic-domiknows
```

2. Install dependencies:
```bash
npm install
```

3. Run development server:
```bash
npm run dev
```

Or build and run production server:
```bash
npm run build
npm run start
```

The frontend will be available at `http://localhost:49790`

## Project Structure

- `Agent/` - LLM agents for graph generation, review, and sensors
- `server/` - FastAPI backend server
- `frontend/agentic-domiknows/` - Next.js frontend application
- `Examples/` - Sample DomiKnows implementations created for the agents

## Usage

1. Start both backend and frontend servers
2. Open `http://localhost:49790` in your browser
3. Enter a task description to generate a knowledge graph
4. Review AI-generated code and provide feedback
5. Approve to proceed to sensor workflow configuration
