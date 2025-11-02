from typing import List, Tuple
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import hashlib
import os
from dotenv import load_dotenv
load_dotenv()

def upsert_examples(task_id, examples: List[str], forced: bool = False):

    EMB = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=os.environ["OPENAI_API_KEY"])
    DB = Chroma(embedding_function=EMB)
    texts, metas, ids = [], [], []
    for i in range(0, len(examples), 2):
        desc, code = examples[i], examples[i+1]
        _id = hashlib.sha1(desc.encode("utf-8")).hexdigest()
        texts.append(desc)
        metas.append({"desc": desc, "code": code})
        ids.append(_id)
    DB.add_texts(texts=texts, metadatas=metas, ids=ids)
    return DB

def select_graph_examples(DB: Chroma, task_desc: str, k: int, review: bool = False) -> List[str]:
    if not k or k<=0:
        return []
    results = DB.similarity_search(task_desc or "", k=k)
    out = []
    for d in results:
        md = d.metadata or {}
        if not review:
            out.extend([md.get("desc", d.page_content)] + ([md["code"]] if md.get("code") else []))
        else:
            out.extend([md.get("desc", d.page_content) +"\n"+ md["code"], "approve"])
    return out