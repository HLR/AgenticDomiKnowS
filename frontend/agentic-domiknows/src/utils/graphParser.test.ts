import { parseDomiKnowsCode } from './graphParser';

// Test with sample DomiKnows code
const sampleCode = `
from domiknows.graph import Graph, Concept, Relation

# Create concepts
Email = Concept(name='email')
Spam = Concept(name='spam')
Ham = Concept(name='ham')
Label = Concept(name='label')

# Create relations
hasLabel = Relation(Email, Label, name='hasLabel')
isSpam = Relation(Email, Spam, name='isSpam')
isHam = Relation(Email, Ham, name='isHam')

# Build graph
graph = Graph('email_spam_graph')
`;

console.log('Testing DomiKnows parser...');
const result = parseDomiKnowsCode(sampleCode);
console.log('Parsed result:', JSON.stringify(result, null, 2));
console.log('Nodes:', result.nodes.length);
console.log('Edges:', result.edges.length);
