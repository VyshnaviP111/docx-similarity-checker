from langchain_community.document_loaders import UnstructuredWordDocumentLoader

# loading the document --------------------------------------------
loader = UnstructuredWordDocumentLoader("data/docx/patient_readmission_case.docx")
docs = loader.load()

print(f"Loaded {len(docs)} documents")
# print(docs[0].page_content[100:200])   
# print(docs[0].metadata)
# splitting the document---------------------------------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
splits = text_splitter.split_documents(docs)
# print(splits)
print("total splits created:", len(splits))

# print(splits[0].page_content)
# print(splits[1].page_content) 
# print(splits[2].page_content) 

for i, split in enumerate(splits[:4], 1):
    print(f"\n--- Split {i} ---")
    print(split.page_content[:300])

#embedding the document splits---------------------------------------------
from sentence_transformers import SentenceTransformer
import numpy as np
embeddings = SentenceTransformer('all-MiniLM-L6-v2')
doc_vectors = embeddings.encode([split.page_content for split in splits])

# print("Embedding vector dimension:", len(doc_vectors[0]))
# print("First 5 values of the first embedding vector:", np.array(doc_vectors[0])[:5])

# print("Embedding vector dimension:", len(doc_vectors[0]))
queries = ["What are the main factors contributing to patient readmission?"]

def cosine_similarity(vec1 , vec2):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1 , vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

for query in queries:
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print(f"{'='*80}\n")

    query_embedding = embeddings.encode(query , convert_to_numpy=True)

    similarities = []
    for i , doc_embedding in enumerate(doc_vectors):
        similarity = cosine_similarity(query_embedding , doc_embedding)
        similarities.append((i , similarity , splits[i]))

    similarities.sort(key=lambda x: x[1] , reverse=True)

    print("Top 3 most similar document sections:")
    for rank , (idx , score , text) in enumerate(similarities[:3] , 1):
        print(f"\nRank {rank}: chunk index {idx}")
        print(f"Similarity Score: {score:.4f}")
        print(f"Document Section:\n{text.page_content[:500]}")  # Print first 500 characters of the section