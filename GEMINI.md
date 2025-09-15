# Gemini Project Instructions

## Behavior Rules
- Auto-accept all file generations, edits, and commands until the full codebase is created and verified runnable.
- Do not ask for confirmation while creating files, directories, or code.
- Generate the complete codebase based on the initial user prompt without requiring additional input.
- Continue automatically until:
  1. Dataset(s) are created and split into train, dev, and test subsets.
  2. All required code files are written.
  3. A main entry point (`main.py`) exists.
  4. The program trains successfully, evaluates on the dev/test split, and runs without errors.

## Codebase Goals
- Implement a runnable end-to-end example using the **DomiKnowS** framework.
- Include:
  - Dataset creation with proper train/dev/test splits.
  - Graph definition with relevant concepts and constraints.
  - Sensors, learners, and program logic.
  - Training and evaluation pipeline.
  - Testing and inference on held-out data.
  - An entrypoint (`main.py`) that ties it all together.
- Ensure the generated code can be executed directly with `python main.py` or equivalent.

## Style Guidelines
- Follow coding conventions and patterns found in:
  - `Doc/` (documentation)
  - `Examples/` (working examples)
  - `domiknows/` (core framework code)
- Clearly prefix each generated file with its filename in output.
- Favor completeness and correctness over brevity.

## Validation and Testing
- After generating code, automatically run training and evaluation steps.
- Verify the program executes end-to-end on train/dev/test splits.
- Report test performance and confirm constraints are respected.

## Termination
- Once the program has been generated, trained, evaluated, and tested successfully, stop generating new content and return a success confirmation.
