
def get_graph_prompt(task_definition):

    graph_instructions = f"""
    You are an expert DomiKnowS knowledge-graph author.
    
    The DomiKnowS library lets you **declare knowledge** about a domain and use it during training and inference. Each graph must define (1) concepts, (2) relations, and (3) constraints over those concepts.
    
    Problem definition: {task_definition}
    
    Framework basics
    - A `Graph` is a container for `Concept`s, relations, and constraints. Each concept can have multiple properties bu they are not defined in the graph and are assigned later
    - Relation types you may use:
      • `contains` — one-to-many parent → children. Using it implies a way to generate or connect parent/child instances of the same child type.
      • `has_a` — many-to-many linkage between concepts for candidate generation.
      
    
    Goal
    - Write a **minimal** graph tailored to the task.
    - Include only the concepts necessary to express the required constraints. Prefer clear, descriptive names; avoid unused objects.
       - A concept should either has a label or be used in other parts of the graph like as a parent of another concept.
       - A concept is different from a property. A concept can assume multiple properties later but they will not be defined in the graph.
          - For example a car's type can be a concept, but the number of wheels or the car color are properties that can be later added to the car concept and dont need to be defined in the graph.
    - You have decomposed the task into a set of concepts and relations in a way the makes sense for the task. 
       - For example in a dataset for each Feature A there might be many Feature B related to it. In this case you can define concept A and concept B as separate concepts and then define a contains relation between them.
       - On the other hand if A and B have a many-to-many relationship and you want to define constraints on them, you can define concept A and concept B as separate concepts and then define a has_a relation between them.
       - You have to consider the limitations of DomiKnowS while writing your code. 
          - For example, each concept in DomiKnowS is categorical. As a result you cant set their label as a regression number. 
          - If you have a concept with a number as its label you can set them as categories. You can use a forloop to define constraints on them.
    
    Logical constraints
    
    Compose constraints with: notL(), andL(), orL(), nandL(), ifL(), norL(), xorL().
    
    Counting / aggregation operators:
    
    existsAL(A) — there exists a candidate classified as A in the entire graph.
    existsL(A) — there exists a candidate classified as A in the given group on concepts.
    exactL(A, 2) - there are exactly 2 candidates classified as A in the given group on concepts.
    atLeastL(A, 2) — there are atleast 2 candidates classified as A in the given group on concepts.
    atLeastAL(A, 2) — there are atleast 2 candidates classified as A in the entire graph.
    atMostL and atMostAL are similar.
    
    Self-check before emitting the code (do not print this checklist):
    
    Minimality: every concept/relation is used by at least one constraint or relation; nothing superfluous.
    Semantics: contains is used only for one-to-many parent→child; has_a only for many-to-many candidate links.
    Completeness: constraints fully capture the task requirements.
    Hygiene: consistent naming; no redundant comments; code is syntactically valid.
    If any self-check fails, revise silently and only then output the final single Python code block.
    """

    graph_examples= ["""
    Problem definition: The primary task for the CoNLL-2003 dataset is to identify and classify entities within a sentence. For example, in the sentence "Barack Obama was born in Hawaii," the model should recognize "Barack Obama" as a PERSON and "Hawaii" as a LOCATION. Furthermore, the dataset can be used for relation extraction, where the objective is not only to identify entities but also to detect relationships between them. For instance, if the sentence was "Barack Obama was born in Hawaii," the potential relation might be a "born-in" relation between "Barack Obama" and "Hawaii."
    Output Graph:
    with Graph('Conll') as graph:
        word = Concept(name='word')
        phrase = Concept(name='phrase')
        sentence = Concept(name='sentence')
        (rel_sentence_contains_word,) = sentence.contains(word)
        (rel_sentence_contains_phrase,) = sentence.contains(phrase)
        (rel_phrase_contains_word,) = phrase.contains(word)
    
        pair = Concept(name='pair')
        (rel_pair_phrase1, rel_pair_phrase2) = pair.has_a(arg1=phrase, arg2=phrase)
    
        entity = phrase(name='entity')
        people = entity(name='people', auto_constraint=True)
        organization = entity(name='organization', auto_constraint=False)
        location = entity(name='location', auto_constraint=None)
        other = entity(name='other')
        o = entity(name='O')
    
        work_for = pair(name='work_for')
        located_in = pair(name='located_in')
        live_in = pair(name='live_in')
        orgbase_on = pair(name='orgbase_on')
        kill = pair(name='kill')
        
        ifL(work_for('x'), andL(people(path=('x', rel_pair_phrase1.name)), organization(path=('x', rel_pair_phrase2.name))))
        ifL(located_in('x'), andL(location(path=('x', rel_pair_phrase1.name)), location(path=('x', rel_pair_phrase2.name))))
        ifL(live_in('x'), andL(people(path=('x', rel_pair_phrase1.name)), location(path=('x', rel_pair_phrase2.name))))
        ifL(orgbase_on('x'), andL(organization(path=('x', rel_pair_phrase1.name)), location(path=('x', rel_pair_phrase2.name))))
        ifL(kill('x'), andL(people(path=('x', rel_pair_phrase1.name)), people(path=('x', rel_pair_phrase2.name))))
        
    """, """
    Problem definition: We have a collections of images each with some regions. each region has some clues and some inferences. we want to find the correct inference for each region. If there are at least 3 useful clues related to an inference, that inference must be correct.
    Output Graph:
    with Graph('sherlockgraph') as graph:
        image = Concept(name='image')

        region = Concept(name='region')
        image_contains_region, = image.contains(region)

        clue = Concept(name='clue')
        usefull_clue = clue(name='usefull_clue')
        region_contains_clue, = region.contains(clue)

        inference = Concept(name='inference')
        true_inference = inference(name='true_inference')
        region_contains_inference, = region.contains(inference)

        helper = Concept(name='helper')
        cluearg, inferencearg = helper.has_a(cluearg=clue, inferencearg=inference)
        related = helper(name='related')


        ifL(notL(usefull_clue("x")),
            notL(existsL(related("y",path=("x",cluearg.reversed))))
        )

        orL(true_inference("x"),
            notL(atLeastL(related("y",path=("x",inferencearg.reversed)),3))
        )

        ifL(region("x"),
            exactL(true_inference("y",path=("x",region_contains_inference)),1),
        )
    """]
    additional_graph_examples= "" # TODO
    return graph_instructions + "\n\n" + "Examples:\n\n" + "\n\n".join(graph_examples) + "\n\n"