

# Graph Declaration Guide

DomiKnowS lets you declare knowledge about a domain and use it during training and inference with deep learning models. Every graph must define:
1. Concepts
2. Relations
3. Constraints over those concepts

Build the smallest graph that expresses exactly what you need. Avoid unused concepts/relations.

---

## Concepts

Start with a single root/domain concept. Everything else connects to it directly or indirectly.

```text
# Root (domain-specific is fine)
batch = Concept(name='batch')

# Another root-style context
sentence = Concept(name='sentence')
```

Create child concepts by calling a parent concept (this implicitly creates an `is_a` relation):

```python
parent = Concept(name='parent')
child = parent(name='child')  # child is_a parent
```

### Label concepts

- Binary (two classes) using `EnumConcept`:

```python
label = parent(name='label', ConceptClass=EnumConcept, values=['false', 'true'])
# Access via: label.false, label.true or label.getattr('false'), label.getattr('true')
```

- Binary as boolean concept (true vs `notL(concept)`):

```python
is_target = parent(name='is_target')  # true when is_target holds, false via notL(is_target)
```

- Multi-class options:
  - Enumerated (mutually exclusive by construction):

    ```python
    cls = parent(name='class', ConceptClass=EnumConcept, values=['label1', 'label2', 'label3'])
    # Use: cls.label1, cls.label2, ... or cls.getattr('label1')
    ```

  - Multiple binary concepts (needs an exclusivity constraint to avoid multi-hot):

    ```python
    l1 = parent(name='label1')
    l2 = parent(name='label2')
    l3 = parent(name='label3')
    # Enforce exactly one true within scope (see Path scoping below for path usage):
    ifL(parent('x'), exactL(l1(path=('x')), l2(path=('x')), l3(path=('x')), 1))
    ```

Notes:
- All concepts are categorical. For numeric targets, bucketize into categories and, if needed, add ordering constraints.
- Do not leave dangling concepts (unused by constraints or structure).

---

## Relations

### `is_a` (implicit)
Created when you define a child concept via `parent(name='child')`.

```python
A = Concept(name='A')
B = A(name='B')    # B is_a A
C = B(name='C')    # C is_a B
# If A is root with one instance, B and C mirror that instance count.
```

### `contains` (one-to-many hierarchy)

```python
A = Concept(name='A')
B = Concept(name='B')
(A_contains_B,) = A.contains(B)
# Effect: An A instance contains zero or more B instances.
```

### `has_a` (many-to-many links across instances)

```python
X = Concept(name='X')
Y = Concept(name='Y')
(y_arg1, y_arg2) = Y.has_a(arg1=X, arg2=X)
# Y instances link pairs of X instances; Y can carry its own labels.
```

Notes:
- The linked concepts (e.g., X above) must have multiple instances (usually by being contained under another concept).
- The linking concept (e.g., Y) is often where relation labels live.

---

## Constraints

Logical predicates:
- `notL(X)`, `andL(X, Y)`, `orL(X, Y)`, `nandL(X, Y)`, `norL(X, Y)`, `xorL(X, Y)`, `ifL(X, Y)`
  - `notL` takes one argument; `ifL` takes two; others take 2+ arguments.

Counting / aggregation operators:
- `existsL(A)` – exists one true instance of A within the current group (defined by `contains` or any other scope)
- `existsAL(A)` – exists one true instance of A anywhere in the entire graph (ignores group)
- `exactL(A, k)` – exactly `k` true instance of A within the current group
- `atLeastL(A, k)`, `atMostL(A, k)` – within current group
- `atLeastAL(A, k)`, `atMostAL(A, k)` – across the entire graph

---

## Path scoping (how `path=...` works)

Path scoping tells a label predicate which specific instances it should apply to. You pass a `path` argument to a label or concept call inside constraints. The key ideas are:

1) Bind a variable to a parent instance, then refer to labels relative to it defined by is_a relationship (parent → children).

```python
# this is a common mutual exclusion pattern
ifL(parent_concept('x'), exactL(label1_concept(path=('x')), label2_concept(path=('x')), label3_concept(path=('x')), 1))
# Here, parent_concept('x') binds variable x to a parent_concept instance.
# label1_concept(path=('x')) means: "label1_concept on the same parent_concept instance bound to x".
```

2) Scope within a `contains` group defined by a contrains relation.

```python
# Example pattern for when A contains B and Blabel is_a B.
ifL(A('x'), atMostL(Blabel(path=('x', A_contain_B_relationship)), 5))
ifL(A('x'), atLeastL(Blabel(path=('x', A_contain_B_relationship)), 2))
# Interpreted as: inside the group of B instances bound to A,
# at most 5 B instances have label1_concept, at least 2 do.
```

3) Traverse `has_a` argument positions from a bound relation instance.

```python
# Example pattern for when B has_a (arg1=A, arg2=A) and Alabel2 is_a A.
# Follow arg1 from relation r, then arg2 from relation r.
ifL(B('r'), notL(existsL(andL(
    Alabel2(path=('r', arg1)),  # follow arg1 from relation r
    Alabel2(path=('r', arg2))   # follow arg2 from relation r
))))
# Interpreted as: no two A instances have Alabel2 that are linked by a B relation instance.
```

4) Aggregations respect scope: the `L`-suffixed aggregators (`existsL`, `exactL`, ...) operate within the current group implied by the nearest containing context; the `AL` variants ignore group and search the entire graph.

```python
# Group vs global existence
existsL(label(path=('x')))   # within x's group
existsAL(label(path=('x')))  # anywhere in the whole graph
```

5) Use `.reversed` to traverse from a node to incident relation instances (inverse direction).

- An example for a contains relation:
```python
ifL(A(path=('x', A_contain_B_relationship.reversed)), atMostL(Blabel("x"), 5))
ifL(A(path=('x', A_contain_B_relationship.reversed)), atLeastL(Blabel("x"), 2))
```

- An example for a has_a relation:
```python
ifL(andL(Alabel('x'), existsL(has_a_relation_concept('s', path=('x', i_arg1.reversed)))),Alabel(path=('s', i_arg2)))
```

Tips:
- Always bind a variable (e.g., `'x'`, `'r'`, `'s'`) before using it in `path=...`.
- When you need relation instances incident to a node, use `argX.reversed` in the path.
---

## Design guidance

- Build the smallest graph that can express the needed constraints.
- Choose relations thoughtfully:
  - Use `contains` for one-to-many hierarchy.
  - Use `has_a` for many-to-many associations you need to constrain.
- Ensure every concept is connected to the single root domain.
- Prefer `EnumConcept` for mutually exclusive multi-class labels; otherwise add `exactL` constraints.

---

## Complete example (pattern)

```python
from domiknows.graph import *
from domiknows.graph.logicalConstrain import *

with Graph('Conll') as graph:
    # Root
    batch = Concept(name='batch')
    sentence = Concept(name='sentence')
    (batch_contains_sentence,) = batch.contains(sentence)

    phrase = Concept(name='phrase')
    (sentence_contains_phrase,) = sentence.contains(phrase)

    # Pairwise links among phrases in the same sentence
    pair = Concept(name='pair')
    (pair_arg1, pair_arg2) = pair.has_a(arg1=phrase, arg2=phrase)

    # Entity labels on phrases
    entity = phrase(name='entity')
    people = entity(name='people')
    organization = entity(name='organization')
    location = entity(name='location')
    other = entity(name='other')
    o = entity(name='O')

    # Relation labels on pairs
    work_for = pair(name='work_for')
    located_in = pair(name='located_in')
    live_in = pair(name='live_in')
    org_based_on = pair(name='org_based_on')
    kill = pair(name='kill')

    # Constraints
    ifL(entity('x'), exactL(
        people(path=('x')), location(path=('x')), organization(path=('x')),
        other(path=('x')), o(path=('x')), 1
    ))

    ifL(pair('x'), exactL(
        work_for(path=('x')), located_in(path=('x')), live_in(path=('x')),
        org_based_on(path=('x')), kill(path=('x')), 1
    ))

    ifL(work_for('x'), andL(
        people(path=('x', pair_arg1)), organization(path=('x', pair_arg2))
    ))
    ifL(located_in('x'), andL(
        location(path=('x', pair_arg1)), location(path=('x', pair_arg2))
    ))
    ifL(live_in('x'), andL(
        people(path=('x', pair_arg1)), location(path=('x', pair_arg2))
    ))
    ifL(org_based_on('x'), andL(
        organization(path=('x', pair_arg1)), location(path=('x', pair_arg2))
    ))
    ifL(kill('x'), andL(
        people(path=('x', pair_arg1)), people(path=('x', pair_arg2))
    ))
```
