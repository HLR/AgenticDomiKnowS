from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings
load_dotenv()

class LLM:
    def __init__(self,reasoning_effort="medium"):
        self.reasoning_effort = reasoning_effort
        self.api_key = os.environ["OPENAI_API_KEY"]
        self.embedder = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=self.api_key)
        if not type(reasoning_effort) == list:
            self.swe_llm = ChatOpenAI(model="gpt-5-2025-08-07", reasoning={"effort": self.reasoning_effort}, api_key=self.api_key)
            self.reviewer_llm = ChatOpenAI(model="gpt-5-2025-08-07", reasoning={"effort": self.reasoning_effort}, api_key=self.api_key)
            self.sensor_llm = ChatOpenAI(model="gpt-5-2025-08-07", reasoning={"effort": self.reasoning_effort},api_key=self.api_key)
            self.property_llm = ChatOpenAI(model="gpt-5-2025-08-07", reasoning={"effort": self.reasoning_effort},api_key=self.api_key)
        else:
            self.swe_llm = ChatOpenAI(model="gpt-5-2025-08-07", reasoning={"effort": self.reasoning_effort[0]},api_key=self.api_key)
            self.reviewer_llm = ChatOpenAI(model="gpt-5-2025-08-07", reasoning={"effort": self.reasoning_effort[0]},api_key=self.api_key)
            self.sensor_llm = ChatOpenAI(model="gpt-5-2025-08-07", reasoning={"effort": self.reasoning_effort[1]},api_key=self.api_key)
            self.property_llm = ChatOpenAI(model="gpt-5-2025-08-07", reasoning={"effort": self.reasoning_effort[2]},api_key=self.api_key)

    def __call__(self, msgs):
        msg = self.swe_llm.invoke(msgs).content
        if isinstance(msg, list): return msg[0]["text"]
        return msg

    def review(self, msgs):
        msg = self.reviewer_llm.invoke(msgs).content
        if isinstance(msg, list): return msg[0]["text"]
        return msg

    def make_sensor(self, msgs):
        msg = self.sensor_llm.invoke(msgs).content
        if isinstance(msg, list): return msg[0]["text"]
        return msg

    def assign_property(self, msgs):
        msg = self.property_llm.invoke(msgs).content
        if isinstance(msg, list): return msg[0]["text"]
        return msg
