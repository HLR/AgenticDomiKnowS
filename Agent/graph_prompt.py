
def get_graph_prompt(task_definition):

    graph_instructions = f"""
    You are an expert DomiKnowS knowledge-graph author.
    The DomiKnowS library lets you **declare knowledge** about a domain and use it during training and inference with deep learning models. Each graph must define (1) concepts, (2) relations, and (3) constraints over those concepts.
    Problem definition: {task_definition}
    
    Framework basics
    - A `Graph` is a container for `Concept`s, relations, and constraints. 
    - A concept either represents a label in the dataset that has to be predicted by a model or is created to help in the formation of the graph structure e.g. as the parent of another concept. Each concept can have multiple properties but they are not defined in the graph and are assigned later.
    For example consider the CONLL Graph:
    
    with Graph('Conll') as graph:
        phrase = Concept(name='phrase')
        sentence = Concept(name='sentence')
        (rel_sentence_contains_phrase,) = sentence.contains(phrase)
        
        pair = Concept(name='pair')
        (rel_pair_phrase1, rel_pair_phrase2) = pair.has_a(arg1=phrase, arg2=phrase)
        
        entity = phrase(name='entity')
        people = entity(name='people')
        organization = entity(name='organization')
        location = entity(name='location')
        other = entity(name='other')
        o = entity(name='O')
        
        work_for = pair(name='work_for')
        located_in = pair(name='located_in')
        live_in = pair(name='live_in')
        orgbase_on = pair(name='orgbase_on')
        kill = pair(name='kill')
        
    In this graph the concepts `people`, `organization`, `location`, `other`, `O`, `work_for`, `located_in`, `live_in`, `orgbase_on`, `kill` are all labels that have to predicted by a model.
    The other concepts are created to help in the formation of the graph structure. Sentence is created to contain multiple instances of the phrase concept. pair is created to connected two instances of phrase concept.
    entity is created as a parent of entity labels. Each of these concepts whther they are labels or parents can be assgined a label by a model but they are not always assigned one.
    Each concept can have multiple properties but they are not defined in the graph and are assigned later. For example each phrase concept have multiple words that form it and an assgined embedding based on the words.
    But these are not shown in the graph since they are not labels nor defined in the formation of the graph.
    
    A concept can either be binary or multi-class.
    A multi-class concept is defined as such: concept_var = question(name="concept_name", ConceptClass=EnumConcept, values=["label1", "label2","label3"])
    You can refer to them later in the constraints as : concept_var.label1, concept_var.label2, concept_var.label3
    
    - Relation types you may use: 
        1) is_a : this relationship is implicit and in the above graph is defined between these concepts (phrase, entity), (entity, people), (pair, work_for), ...
        2) contains : one-to-many relationship between concepts. For example in the above graph sentence contains phrase because there are multiple phrases related to a sentence. 
        3) has_a : many-to-many linkage between multiple concepts. For example in the above graph pair has_a two phrases.
    
    Goal
    - Write a **minimal** graph tailored to the task.
    - Include only the concepts necessary to express the required constraints. Prefer clear, descriptive names; avoid unused objects.
    - You have decomposed the task into a set of concepts and relations in a way the makes sense for the task. 
       - For example in a dataset for each Feature A there might be many Feature B related to it. In this case you can define concept A and concept B as separate concepts and then define a contains relation between them.
       - On the other hand if A and B have a many-to-many relationship and you want to define constraints on them, you can define concept A and concept B as separate concepts and then define a has_a relation between them.
       - You have to consider the limitations of DomiKnowS while writing your code. 
          - For example, each concept in DomiKnowS is categorical. As a result you cant set their label as a regression number. 
          - If you have a concept with a number as its label you can set them as categories. You can use a forloop to define constraints on them.
    
    The constraints in the graph are defined using predefined predicates such as notL(), andL(), orL(), nandL(), ifL(), norL(), xorL() which can take either one (notL) or two arguments.
    
    For the above graph the constratints are as follows:
    
        ifL(work_for('x'), andL(people(path=('x', rel_pair_phrase1)), organization(path=('x', rel_pair_phrase2))))
        ifL(located_in('x'), andL(location(path=('x', rel_pair_phrase1)), location(path=('x', rel_pair_phrase2))))
        ifL(live_in('x'), andL(people(path=('x', rel_pair_phrase1)), location(path=('x', rel_pair_phrase2))))
        ifL(orgbase_on('x'), andL(organization(path=('x', rel_pair_phrase1)), location(path=('x', rel_pair_phrase2))))
        ifL(kill('x'), andL(people(path=('x', rel_pair_phrase1)), people(path=('x', rel_pair_phrase2))))
    
    There are also counting / aggregation operators:
    
        existsAL(A) — there exists a candidate classified as A in the entire graph.
        existsL(A) — there exists a candidate classified as A in the given group on concepts.
        exactL(A, 2) - there are exactly 2 candidates classified as A in the given group on concepts.
        atLeastL(A, 2) — there are atleast 2 candidates classified as A in the given group on concepts.
        atLeastAL(A, 2) — there are atleast 2 candidates classified as A in the entire graph.
        atMostL and atMostAL are similar.
    """

    graph_examples= ["""
    Problem definition: We have a collections of images with given pixels and dimentions each with some regions. each region has a bounding box, some textual clues and some textual inferences. we want to find the correct inference for each region. If there are at least 3 useful clues related to an inference, that inference must be correct.
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