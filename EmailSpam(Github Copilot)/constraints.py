from domiknows.graph.logicalConstrain import LogicalConstraint

# Constraint: If Model1Label is "spam", Model2Label must not be "not spam"
constraint = LogicalConstraint(
    "ConsistencyConstraint",
    lambda model1, model2: not (model1 == "spam" and model2 == "legitimate"),
    concepts=["Model1Label", "Model2Label"]
)