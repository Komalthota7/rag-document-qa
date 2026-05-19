import anthropic
import chromadb
from sentence_transformers import SentenceTransformer

# ── SETUP ──────────────────────────────────────────────────────
# Replace the text below with your actual API key
import os
claude = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# This creates a local database folder on your computer
chroma_client = chromadb.PersistentClient(path="./chroma_storage")
collection = chroma_client.get_or_create_collection(name="my_documents")

# This loads the embedding model onto your computer
print("Loading embedding model... please wait...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model ready!")


# ── STEP 1: LOAD AND INDEX THE DOCUMENT ────────────────────────
def load_and_index_document(file_path):
    print(f"\nReading document from {file_path}...")

    # Read the text file
    with open(file_path, "r") as f:
        text = f.read()

    print(f"Document loaded. Total characters: {len(text)}")

    # Split into chunks by paragraph
    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    print(f"Document split into {len(chunks)} chunks")

    # Convert each chunk into an embedding
    print("Creating embeddings for each chunk...")
    embeddings = embedding_model.encode(chunks).tolist()

    # Store everything in ChromaDB
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

    print("Document indexed and stored in database successfully!")


# ── STEP 2: FIND RELEVANT CHUNKS FOR A QUESTION ────────────────
def retrieve_relevant_chunks(question):

    # Convert the question into an embedding
    question_embedding = embedding_model.encode([question]).tolist()

    # Search the database for the 3 most similar chunks
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=3
    )

    return results["documents"][0]


# ── STEP 3: GENERATE ANSWER USING CLAUDE ───────────────────────
def generate_answer(question, chunks):

    # Combine all retrieved chunks into one block of text
    context = "\n\n".join(chunks)

    # Build the prompt
    prompt = f"""Here is information retrieved from a document:

{context}

Using only the information above, answer this question:
{question}

If the answer is not in the information provided, say 
'I don't have enough information to answer that.'"""

    # Send to Claude
    response = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.content[0].text


# ── PUTTING IT ALL TOGETHER ─────────────────────────────────────
def ask(question):
    print(f"\nYou asked: {question}")
    chunks = retrieve_relevant_chunks(question)
    answer = generate_answer(question, chunks)
    print(f"Answer: {answer}")
    print("-" * 60)


# ── RUN THE PROGRAM ─────────────────────────────────────────────
if __name__ == "__main__":

    # Index the document — only needs to happen once
    load_and_index_document("document.txt")

    # Start the question answering loop
    print("\n RAG System is ready!")
    print("Type your question and press Enter.")
    print("Type 'quit' to stop the program.\n")

    while True:
        question = input("Your question: ").strip()

        if question.lower() == "quit":
            print("Goodbye!")
            break

        if question:
            ask(question)