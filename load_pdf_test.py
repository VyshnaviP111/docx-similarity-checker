from langchain_community.document_loaders import PyPDFLoader
# loading the document --------------------------------------------
loader = PyPDFLoader("data/text/hospitalreadmission.pdf")
docs = loader.load()
print(f"Loaded {len(docs)}")
print(docs[0].page_content[100:200])
print(docs[1].page_content[100:200])

# # splitting the document---------------------------------------------
from langchain_text_splitters import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size = 500 , chunk_overlap = 50)

splits = text_splitter.split_documents(docs)
print("total splits:" , len(splits))
# print(splits[0].page_content)

# for i , split in enumerate(splits[:len(splits)] , 1):
#     print(f"\n--- Split {i} ---")
#     print(split.page_content[:300])

#embedding the document splits---------------------------------------------
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

import numpy as np

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

doc_vectors = embeddings.embed_documents([splits[i].page_content for i in range(len(splits))])

print("Embedding vector dimension:" , len(doc_vectors[0]))
print("First 5 values of the first embedding vector:" , np.array(doc_vectors[0])[:5])


queries = "What is the factors affecting hospital readmission rate?"

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

db = FAISS.from_documents(splits , embeddings)
db.save_local("faiss_index")
print("FAISS index saved locally.")


##loading from the vector db 

db = FAISS.load_local("faiss_index" , embeddings , allow_dangerous_deserialization=True)
print("FAISS index loaded from local storage.")

results = db.similarity_search(queries , k=3)
print(f"\n{'='*80}")
print(f"Query: {queries}")  

for i , res in enumerate(results , 1):
    print(f"\n--- Result {i} ---")
    print(res.page_content[:300])