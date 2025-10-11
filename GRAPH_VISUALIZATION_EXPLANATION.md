# Graph Visualization Calculation Explanation

## How the Graph Visualization Works

### Step 1: Code Parsing
The frontend receives `BuildState.graph_code_draft[]` from the backend, which contains Python code like:
```python
Email = Concept(name='email')
Spam = Concept(name='spam') 
hasLabel = Relation(Email, Spam, name='hasLabel')
```

### Step 2: Regex Pattern Matching

**Concepts Extraction** (`extractConcepts` function):
- Pattern 1: `(\w+)\s*=\s*Concept\s*\(\s*name\s*=\s*['"]([^'"]+)['"]`
  - Matches: `Email = Concept(name='email')`
  - Captures: varName="Email", conceptName="email"
- Pattern 2: `Concept\s*\(\s*name\s*=\s*['"]([^'"]+)['"]`
  - Matches: `Concept(name='email')` (without assignment)

**Relations Extraction** (`extractRelations` function):
- Pattern 1: `(\w+)\s*=\s*Relation\s*\(\s*(\w+)\s*,\s*(\w+)\s*(?:,\s*name\s*=\s*['"]([^'"]+)['"])?\s*\)`
  - Matches: `hasLabel = Relation(Email, Spam, name='hasLabel')`
  - Captures: varName="hasLabel", source="Email", target="Spam", relationName="hasLabel"
- Pattern 2: `Relation\s*\(\s*(\w+)\s*,\s*(\w+)\s*,\s*name\s*=\s*['"]([^'"]+)['"]\s*\)`
  - Matches: `Relation(Email, Spam, name='hasLabel')` (without assignment)

### Step 3: Validation
Relations are only created if:
- Both source and target exist in the concepts Map
- `concepts.has(source) && concepts.has(target)`

### Step 4: Layout Calculation (`calculateLayout` function)
- **Circular Layout**: Arranges concepts in a circle
- **Center**: (250, 175) - middle of 500x350 SVG viewBox
- **Radius**: `Math.min(150, 100 + conceptArray.length * 10)`
- **Angle**: `(2 * Math.PI * index) / conceptArray.length - Math.PI / 2`
- **Position**: `x = centerX + radius * Math.cos(angle)`, `y = centerY + radius * Math.sin(angle)`

### Step 5: SVG Rendering
- **Nodes**: Rectangles at calculated positions
- **Edges**: Lines between source and target node positions
- **Scaling**: Positions multiplied by 1.2 (x-axis) and 1.1 (y-axis) for spacing

## Why Edges Might Not Appear
1. **Regex not matching**: Relation syntax doesn't match expected patterns
2. **Validation failing**: Source/target concepts not found in concepts Map
3. **Case sensitivity**: Variable names must match exactly
4. **Syntax variations**: Code might use different Relation syntax than expected

## Example Working Code
```python
# This should work:
Email = Concept(name='email')
Spam = Concept(name='spam')
hasLabel = Relation(Email, Spam, name='hasLabel')

# This might not work:
email = Concept(name='email')  # lowercase
spam = Concept(name='spam')
hasLabel = Relation(Email, Spam, name='hasLabel')  # Email != email
```