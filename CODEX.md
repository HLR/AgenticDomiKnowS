# Codex Project Instructions

## Context
- Always treat the **current working directory (`.`)** as the project context.
- Load and reference all files in:
  - `src/`
  - `examples/`
  - `tests/`
  - `Doc/`
- Assume the entire project folder is available for imports, edits, and new file creation.

## Behavior Rules
- Auto-accept all file generations, edits, and commands until the full codebase is created and runnable.
- Do not ask for confirmation while creating files, directories, or code.
- Generate the **complete codebase** based on the initial user prompt without requiring additional input.
- Continue automatically until:
  1. Datasets (if applicable) are created and split into train, dev, and test subsets.
  2. All required source code files are written.
  3. A main entry point (`main.py`) exists and ties everything together.
  4. The program trains/runs successfully, evaluates on dev/test splits, and executes without errors.

## Codebase Goals
- Implement a runnable end-to-end example using the **DomiKnowS framework**.
- Include:
  - Dataset creation with proper splits.
  - Graph definition with relevant concepts and constraints.
  - Sensors, learners, and program logic.
  - Training and evaluation pipeline.
  - Testing and inference on held-out data.
  - An entrypoint (`main.py`) that runs with `python main.py`.
- Follow coding conventions from:
  - `Doc/` (documentation)
  - `Examples/` (working examples)
  - `domiknows/` (framework code)

## Style Guidelines
- Clearly prefix each generated file with its filename in output.
- Favor completeness and correctness over brevity.
- Follow existing naming patterns and directory layout.

## Validation and Testing
- After generating code, automatically run training and evaluation.
- Verify the program executes end-to-end on train/dev/test splits.
- Report test performance and confirm constraints are respected.

## Termination
- Once the program is generated, trained, evaluated, and tested successfully:
  - Stop generating new content.
  - Return a final success confirmation.
