from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

class LLM:
    def __init__(self,test_run=True):
        self.test_run = test_run
        self.api_key = os.environ["OPENAI_API_KEY"]
        if self.test_run:
            self.swe_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=self.api_key)
            self.reviewer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=self.api_key)
        else:
            self.swe_llm = ChatOpenAI(model="gpt-5", reasoning={"effort": "medium"}, api_key=self.api_key)
            self.reviewer_llm = ChatOpenAI(model="gpt-5", reasoning={"effort": "medium"}, api_key=self.api_key)

    def __call__(self, msgs):
        msg = self.swe_llm.invoke(msgs).content
        if isinstance(msg, list): return msg[0]["text"]
        return msg

    def review(self, msgs):
        msg = self.reviewer_llm.invoke(msgs).content
        if isinstance(msg, list): return msg[0]["text"]
        return msg
