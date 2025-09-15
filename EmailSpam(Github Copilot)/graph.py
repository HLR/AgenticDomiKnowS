from domiknows.graph import ConceptClass, Graph

# Define concepts
Email = ConceptClass("Email")
Header = ConceptClass("Header")
Body = ConceptClass("Body")
Label = ConceptClass("Label")
Model1Label = ConceptClass("Model1Label")
Model2Label = ConceptClass("Model2Label")

# Build the graph
graph = Graph("EmailSpamGraph")
graph.addConcept(Email)
graph.addConcept(Header, parent=Email)
graph.addConcept(Body, parent=Email)
graph.addConcept(Label, parent=Email)
graph.addConcept(Model1Label, parent=Email)
graph.addConcept(Model2Label, parent=Email)