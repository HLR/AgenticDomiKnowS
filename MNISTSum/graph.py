
from domiknows.graph import Graph, Concept, Relation, Property
from domiknows.graph.logicalConstrain import ifL, andL, eqL

Graph.clear()
Concept.clear()

with Graph('mnist_sum') as graph:
    # Concepts
    pair = Concept(name='pair')
    digit = Concept(name='digit')
    digits = {i: digit(name=str(i)) for i in range(10)}
    
    concept_sum = Concept(name='sum')
    sums = {i: concept_sum(name=f'sum_{i}') for i in range(19)} # 0 through 18

    # Relations (Predictions)
    digit_a = Relation(pair, digit, argument_name='digit_a')
    digit_b = Relation(pair, digit, argument_name='digit_b')
    predicted_sum = Relation(pair, concept_sum, argument_name='predicted_sum')

    # Sensor for ground truth sum
    sum_label = Property(pair, name='sum_label')

    # Points of Interest
    (p, digit_a, digit_b, predicted_sum, sum_label) = pair.poi

    # --- Constraints ---

    # 1. Arithmetic Constraint: digit_a + digit_b = predicted_sum
    # This is modeled by enumerating all possibilities.
    for i in range(10):
        for j in range(10):
            s = i + j
            ifL(
                andL(
                    digit_a.is_a(digits[i]),
                    digit_b.is_a(digits[j])
                ),
                then=predicted_sum.is_a(sums[s])
            )

    # 2. Supervision Constraint: predicted_sum must equal the ground truth sum_label
    # This is the key constraint that enforces the sum.
    # We use eqL to compare the concept 'predicted_sum' with the sensor 'sum_label'
    for s_val in range(19):
        ifL(sum_label.eq(s_val), then=predicted_sum.is_a(sums[s_val]))

