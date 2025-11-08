# 🧠 Local Document Similarity Checker

A simple, self-contained project that loads a document, splits it into chunks, creates embeddings using a **local SentenceTransformer model**, and performs **semantic similarity search** — all **without any API keys**.

---

## 🚀 Features
- 📄 Load `.docx` or `.txt` files easily
- ✂️ Split documents into meaningful chunks
- 🧩 Create embeddings locally using `sentence-transformers/all-MiniLM-L6-v2`
- 🔍 Perform in-memory semantic similarity search
- 🧠 No OpenAI API — runs fully offline!

---

## 🧰 Setup

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

python -m venv .venv
.venv\Scripts\activate       # (Windows)
# source .venv/bin/activate  # (Mac/Linux)

pip install -r requirements.txt
