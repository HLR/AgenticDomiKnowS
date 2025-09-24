from typing import List, Tuple
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import hashlib
import os
import shutil


def upsert_examples(task_id, examples: List[str], api_key: str, forced: bool = False):
    persist_directory = f"./rag_examples_{task_id}/"
    collection_name = f"graph_example_bank{task_id}"

    if forced and os.path.exists(persist_directory):
        shutil.rmtree(persist_directory, ignore_errors=True)

    EMB = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=api_key)

    if (not forced) and os.path.exists(persist_directory):
        return Chroma(collection_name=collection_name, persist_directory=persist_directory, embedding_function=EMB)

    DB = Chroma(collection_name=collection_name, persist_directory=persist_directory, embedding_function=EMB)
    texts, metas, ids = [], [], []


    for i in range(0, len(examples), 2):
        desc, code = examples[i], examples[i+1]
        _id = hashlib.sha1(desc.encode("utf-8")).hexdigest()
        texts.append(desc)
        metas.append({"desc": desc, "code": code})
        ids.append(_id)
    DB.add_texts(texts=texts, metadatas=metas, ids=ids)
    return DB


def select_graph_examples(DB: Chroma, task_desc: str, k: int) -> List[str]:
    if not k or k<=0:
        return []
    results = DB.similarity_search(task_desc or "", k=k)
    out = []
    for d in results:
        md = d.metadata or {}
        out.extend([md.get("desc", d.page_content)] + ([md["code"]] if md.get("code") else []))
    return out