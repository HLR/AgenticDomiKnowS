

def property_agent(llm, property_human_text, entire_sensor_code, property_rag_examples):

    instructions = """
    # Property
    
    The input code uses simple DummyLearner placeholders that predict labels from IDs. The output code should integrate real “properties” (text, images, pairs, etc.) via ReaderSensors and use LLMLearner with a natural‑language prompt to produce the predictions.
    The steps below are the common recipe.
    
    
    ## General conversion checklist
    
    1) Keep the graph/constraints the same
    
    2) Add raw input features to the dataset dict
       - Provide the property values the LLM will read, e.g.:
         - Text: "..._text": ["...", ...]
         - Image surrogate (IDs or pixels): "image_pixels": [...]
         - Structured board/string features
    
    3) Expose those features via ReaderSensor
       - Attach ReaderSensor to the Concept that holds the feature, e.g.:
         - news['news_text'] = ReaderSensor(keyword='news_text')
         - image['image_pixels'] = ReaderSensor(keyword='image_pixels')
    
    4) Keep ground‑truth labels via LabelReaderSensor
    
    5) Replace DummyLearner with LLMLearner
       - For any concept that does not have a has_a relation:
         - node[target] = LLMLearner(node1["feature1"], node2["feature2"], ... , prompt="...", classes=[...]) # classes should match EnumConcept values of the target node or be "false" and "true" for binary labels
       - For a concept with a has_a relation ( this is where node.has_a or target.is_a is used ):
         - node[target] = LLMLearner(node["feature"], node["feature"], ... , prompt="...", classes=[...], rel='has_a_relation_keyword')
       - relation_keyword should match the dataset keyword for the has_a relation
       - Write a concise prompt that instructs the model how to map the input(s) to the target label.
    
    6) Leave edge/relation ReaderSensors unchanged
       - EdgeReaderSensor and ManyToManyReaderSensor setup usually stays the same as in main.py.
    
    7) Keep the rest of the code unchanged.
    """

    msgs = [{"role": "system", "content": instructions}]

    if property_rag_examples:
        i = 0
        n = len(property_rag_examples)
        while i < n:
            msgs.append({"role": "user", "content": property_rag_examples[i]})
            if i + 1 < n:
                msgs.append({"role": "assistant", "content": property_rag_examples[i + 1]})
            i += 2

    msgs.append({"role": "user", "content": property_human_text + "\n" + entire_sensor_code})
    return llm(msgs)