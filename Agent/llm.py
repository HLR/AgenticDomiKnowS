from langchain_openai import ChatOpenAI

class LLM:
    def __init__(self,test_run=True, api_key=""):
        self.test_run = test_run
        self.api_key = api_key
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
