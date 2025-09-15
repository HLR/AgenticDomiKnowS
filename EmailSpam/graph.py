
from domiknows.graph import Graph, Concept
from domiknows.graph.logicalConstrain import ifL, nandL

Graph.clear()
Concept.clear()

with Graph('email_graph') as graph:
    email = Concept(name='email')
    header = Concept(name='header')
    body = Concept(name='body')
    email.contains(header)
    email.contains(body)

    spam = email(name='spam')
    not_spam = email(name='not_spam')

    nandL(spam, not_spam)

    model1_spam = email(name='model1_spam')
    model2_spam = email(name='model2_spam')

    ifL(model1_spam, spam)
    ifL(model2_spam, spam)

    # Constraint: If Model 1 predicts spam, Model 2 must not predict not_spam.
    ifL(model1_spam, model2_spam)
