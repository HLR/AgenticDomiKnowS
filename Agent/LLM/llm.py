from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

class LLM:
    def __init__(self,reasoning_effort="medium"):
        self.reasoning_effort = reasoning_effort
        self.api_key = os.environ["OPENAI_API_KEY"]
        self.swe_llm = ChatOpenAI(model="gpt-5", reasoning={"effort": self.reasoning_effort}, api_key=self.api_key)
        self.reviewer_llm = ChatOpenAI(model="gpt-5", reasoning={"effort": self.reasoning_effort}, api_key=self.api_key)

    def __call__(self, msgs):
        msg = self.swe_llm.invoke(msgs).content
        if isinstance(msg, list): return msg[0]["text"]
        return msg

    def review(self, msgs):
        msg = self.reviewer_llm.invoke(msgs).content
        if isinstance(msg, list): return msg[0]["text"]
        return msg
