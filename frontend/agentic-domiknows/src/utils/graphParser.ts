/**
 * Parser for DomiKnows Python code to extract graph structure
 * Parses Concept and Relation definitions from the generated code
 */

interface GraphNode {
  id: string;
  label: string;
  type: 'concept' | 'relation';
  x: number;
  y: number;
}

interface GraphEdge {
  id: string;
  source: string;
  target: string;
  label: string;
}

export interface GraphResult {
  nodes: GraphNode[];
  edges: GraphEdge[];
  code: string;
}

/**
 * Extract Concepts from DomiKnows code
 * Matches patterns like: Email = Concept(name='email')
 * or: Concept(name='email')
 */
function extractConcepts(code: string): Map<string, string> {
  console.log('🔍 === EXTRACTING CONCEPTS ===');
  console.log('📄 Code to analyze for concepts:');
  console.log(code);
  
  const concepts = new Map<string, string>();
  
  // Pattern 1: VariableName = Concept(name='concept_name')
  const pattern1 = /(\w+)\s*=\s*Concept\s*\(\s*name\s*=\s*['"]([^'"]+)['"]/gi;
  let match;
  
  console.log('🔍 Testing pattern 1: VariableName = Concept(name=\'concept_name\')');
  while ((match = pattern1.exec(code)) !== null) {
    const varName = match[1];
    const conceptName = match[2];
    console.log(`✅ Found concept: ${varName} = Concept(name='${conceptName}')`);
    concepts.set(varName, conceptName);
  }
  
  // Pattern 2: Just Concept(name='concept_name') without assignment
  const pattern2 = /Concept\s*\(\s*name\s*=\s*['"]([^'"]+)['"]/gi;
  console.log('🔍 Testing pattern 2: Concept(name=\'concept_name\') without assignment');
  while ((match = pattern2.exec(code)) !== null) {
    const conceptName = match[1];
    if (!Array.from(concepts.values()).includes(conceptName)) {
      console.log(`✅ Found standalone concept: Concept(name='${conceptName}')`);
      concepts.set(conceptName, conceptName);
    } else {
      console.log(`⚠️ Concept '${conceptName}' already exists, skipping`);
    }
  }
  
  console.log('📊 Total concepts extracted:', concepts.size);
  console.log('📊 Concepts map:', Array.from(concepts.entries()));
  
  return concepts;
}

/**
 * Extract Relations from DomiKnows code
 * Matches patterns like: hasLabel = Relation(Sentence, Label, name='hasLabel')
 * or: Relation(Concept1, Concept2, name='relates_to')
 */
function extractRelations(code: string, concepts: Map<string, string>): Array<{
  varName: string;
  relationName: string;
  source: string;
  target: string;
}> {
  console.log('🔍 === EXTRACTING RELATIONS ===');
  console.log('📄 Code to analyze for relations:');
  console.log(code);
  console.log('📊 Available concepts for relations:', Array.from(concepts.keys()));
  
  const relations: Array<{
    varName: string;
    relationName: string;
    source: string;
    target: string;
  }> = [];
  
  // Pattern: VariableName = Relation(Source, Target, name='relation_name')
  const pattern1 = /(\w+)\s*=\s*Relation\s*\(\s*(\w+)\s*,\s*(\w+)\s*(?:,\s*name\s*=\s*['"]([^'"]+)['"])?\s*\)/gi;
  let match;
  
  console.log('🔍 Testing pattern 1: VariableName = Relation(Source, Target, name=\'relation_name\')');
  while ((match = pattern1.exec(code)) !== null) {
    const varName = match[1];
    const source = match[2];
    const target = match[3];
    const relationName = match[4] || varName;
    
    console.log(`🔍 Found potential relation: ${varName} = Relation(${source}, ${target}, name='${relationName}')`);
    
    // Verify source and target are concepts
    if (concepts.has(source) && concepts.has(target)) {
      console.log(`✅ Valid relation: ${source} -> ${target} (${relationName})`);
      relations.push({
        varName,
        relationName,
        source,
        target
      });
    } else {
      console.log(`❌ Invalid relation: ${source} (exists: ${concepts.has(source)}) -> ${target} (exists: ${concepts.has(target)})`);
    }
  }
  
  // Pattern 2: Just Relation(Source, Target, name='relation_name') without assignment
  const pattern2 = /Relation\s*\(\s*(\w+)\s*,\s*(\w+)\s*,\s*name\s*=\s*['"]([^'"]+)['"]\s*\)/gi;
  console.log('🔍 Testing pattern 2: Relation(Source, Target, name=\'relation_name\') without assignment');
  while ((match = pattern2.exec(code)) !== null) {
    const source = match[1];
    const target = match[2];
    const relationName = match[3];
    
    console.log(`🔍 Found potential standalone relation: Relation(${source}, ${target}, name='${relationName}')`);
    
    if (concepts.has(source) && concepts.has(target)) {
      // Check if not already added
      const exists = relations.some(r => 
        r.source === source && r.target === target && r.relationName === relationName
      );
      if (!exists) {
        console.log(`✅ Valid standalone relation: ${source} -> ${target} (${relationName})`);
        relations.push({
          varName: relationName,
          relationName,
          source,
          target
        });
      } else {
        console.log(`⚠️ Relation already exists: ${source} -> ${target} (${relationName})`);
      }
    } else {
      console.log(`❌ Invalid standalone relation: ${source} (exists: ${concepts.has(source)}) -> ${target} (exists: ${concepts.has(target)})`);
    }
  }
  
  console.log('📊 Total relations extracted:', relations.length);
  console.log('📊 Relations:', relations);
  
  return relations;
}

/**
 * Calculate graph layout using force-directed approach
 */
function calculateLayout(
  concepts: Map<string, string>,
  relations: Array<{ source: string; target: string }>
): Map<string, { x: number; y: number }> {
  const positions = new Map<string, { x: number; y: number }>();
  const conceptArray = Array.from(concepts.keys());
  
  if (conceptArray.length === 0) {
    return positions;
  }
  
  // Simple circular layout
  const centerX = 250;
  const centerY = 175;
  const radius = Math.min(150, 100 + conceptArray.length * 10);
  
  if (conceptArray.length === 1) {
    positions.set(conceptArray[0], { x: centerX, y: centerY });
  } else if (conceptArray.length === 2) {
    positions.set(conceptArray[0], { x: centerX - 100, y: centerY });
    positions.set(conceptArray[1], { x: centerX + 100, y: centerY });
  } else {
    conceptArray.forEach((concept, index) => {
      const angle = (2 * Math.PI * index) / conceptArray.length - Math.PI / 2;
      const x = centerX + radius * Math.cos(angle);
      const y = centerY + radius * Math.sin(angle);
      positions.set(concept, { x, y });
    });
  }
  
  return positions;
}

/**
 * Main parser function that converts DomiKnows code to graph structure
 */
export function parseDomiKnowsCode(code: string): GraphResult {
  console.log('🔍 === STARTING DOMIKNOWS CODE PARSING ===');
  console.log('📄 Input code:');
  console.log(code);
  console.log('📏 Code length:', code.length);
  
  if (!code || code.trim().length === 0) {
    console.log('⚠️ Empty or null code provided');
    return {
      nodes: [],
      edges: [],
      code: code || ''
    };
  }
  
  // Extract concepts and relations
  console.log('🔍 Starting concept extraction...');
  const concepts = extractConcepts(code);
  
  console.log('🔍 Starting relation extraction...');
  const relations = extractRelations(code, concepts);
  
  console.log('🔍 Starting layout calculation...');
  // Calculate layout
  const positions = calculateLayout(concepts, relations);
  console.log('📊 Calculated positions:', Array.from(positions.entries()));
  
  // Build nodes array
  console.log('🔍 Building nodes array...');
  const nodes: GraphNode[] = Array.from(concepts.entries()).map(([varName, conceptName]) => {
    const pos = positions.get(varName) || { x: 250, y: 175 };
    const node = {
      id: varName,
      label: conceptName,
      type: 'concept' as const,
      x: pos.x,
      y: pos.y
    };
    console.log(`📊 Created node:`, node);
    return node;
  });
  
  // Build edges array
  console.log('🔍 Building edges array...');
  const edges: GraphEdge[] = relations.map((rel, index) => {
    const edge = {
      id: `edge_${index}`,
      source: rel.source,
      target: rel.target,
      label: rel.relationName
    };
    console.log(`📊 Created edge:`, edge);
    return edge;
  });
  
  const result = {
    nodes,
    edges,
    code
  };
  
  console.log('✅ === PARSING COMPLETE ===');
  console.log('📊 Final result summary:');
  console.log(`📊 - Nodes: ${result.nodes.length}`);
  console.log(`📊 - Edges: ${result.edges.length}`);
  console.log('📊 Final result object:', result);
  
  return result;
}

/**
 * Fallback function for when parsing fails or code is invalid
 */
export function createFallbackGraph(code: string): GraphResult {
  return {
    nodes: [
      { id: 'placeholder', label: 'Parsing...', type: 'concept', x: 250, y: 175 }
    ],
    edges: [],
    code: code || ''
  };
}
