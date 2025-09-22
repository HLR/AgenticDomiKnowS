from langchain_openai import ChatOpenAI

class LLM:
    def __init__(self,test_run=True, api_key=""):
        self.test_run = test_run
        if not self.test_run:
            self.api_key = api_key
            self.swe_llm = ChatOpenAI(model="gpt-5", temperature=0,api_key=self.api_key)
            self.reviewer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0,api_key=self.api_key)
        else:
            self.swe_llm = None
            self.reviewer_llm = None

    def __call__(self, msgs):
        return self.swe_llm.invoke(msgs).content if not self.test_run else "Print('Code Test')"

    def review(self, msgs):
        return self.reviewer_llm.invoke(msgs).content if not self.test_run else "Print('Review Test')"