from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
   model="models/gemini-embedding-001"
)

result = embeddings.embed_query("hello")

print(len(result))