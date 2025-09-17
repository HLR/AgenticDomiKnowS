"""Knowledge graph and logical constraints for the email spam example."""
from domiknows.graph import Concept, Graph
from domiknows.graph.logicalConstrain import ifL, nandL

Graph.clear()
Concept.clear()

with Graph("email_spam_graph") as graph:
    email = Concept(name="email")
    header = Concept(name="header")
    body = Concept(name="body")

    email.contains(header)
    email.contains(body)

    label = email(name="label")

    model1_logits = email(name="model1_logits")
    model2_logits = email(name="model2_logits")

    model1_spam = email(name="model1_spam")
    model1_not_spam = email(name="model1_not_spam")
    model2_spam = email(name="model2_spam")
    model2_not_spam = email(name="model2_not_spam")

    nandL(model1_spam, model1_not_spam)
    nandL(model2_spam, model2_not_spam)
    nandL(model1_spam, model2_not_spam)
    nandL(model2_spam, model1_not_spam)

    ifL(model1_spam, model2_spam)
    ifL(model2_spam, model1_spam)

__all__ = [
    "graph",
    "email",
    "header",
    "body",
    "label",
    "model1_logits",
    "model2_logits",
    "model1_spam",
    "model1_not_spam",
    "model2_spam",
    "model2_not_spam",
]
