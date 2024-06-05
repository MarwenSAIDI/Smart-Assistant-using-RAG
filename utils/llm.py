from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
import chromadb
from langchain_core.prompts import PromptTemplate

class LLM():
    def __init__(self) -> None:
        self.llm = ChatOllama(model='phi3:latest', stop=["<|end|>","<|user|>","<|assistant|>"])
        self.embedding = OllamaEmbeddings(model='phi3:latest')
        self.chroma_client = chromadb.PersistentClient(path="./db")
        self.collection = self.chroma_client.get_or_create_collection(name="open-chat-DB")

        template = """
        <|system|>You are a cheerful medical professional. You will answer any question but only if you have contents
        and titles. Make sure to format your answer from the contents in a short and comprehesible way so that
        even a 12 year old can understand. After each answer, make sure to return the titles of where you got this information. 
        contents:\n\n{contents}\n\n
        titles:\n\n{docTitles}\n\n
        <|end|>\n<|user|>\n{question}<|end|>\n<|assistant|>
        """
        self.prompt = PromptTemplate.from_template(template)


    def request(self, q:str, n:int = 1) -> str:
        
        results = self.collection.query(
        query_embeddings=self.embedding.embed_query("What is Atopic dermatitis?"), # Chroma will embed this for you
        n_results=3 # how many results to return

        )

        ## Get the closes n result
        listContents = results["documents"][0][:n]
        listTitles = [object["title"] for object in results["metadatas"][0][:n]]

        answer = self.llm.invoke(self.prompt.format(question=q, contents=listContents, docTitles=listTitles))

        return answer.content