from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Sample meeting notes
sample_notes = [
    "Meeting on 2025-06-01: Discussed project timelines for Q3. Team agreed to prioritize feature X. Action item: John to finalize specs by 2025-06-10.",
    "Meeting on 2025-06-03: Client feedback on prototype. Need to improve UI responsiveness. Action item: Sarah to update designs by 2025-06-08.",
    "Meeting on 2025-06-05: Budget review for Q4. Allocated 20K for marketing. Action item: Mike to contact vendors by 2025-06-12."
]

# Create documents
documents = [Document(page_content=note, metadata={"id": i+1}) for i, note in enumerate(sample_notes)]

# Initialize vector store
vector_store = PGVector(
    connection_string=DATABASE_URL,
    embedding_function=embeddings,
    collection_name="meeting_notes"
)

# Add documents to vector store
vector_store.add_documents(documents)

print("Sample meeting notes added to database.")