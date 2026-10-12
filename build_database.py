import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma


DATA_PATH = "data/markdown"
CHROMA_PATH = "chroma"



def load_documents():
    documents = []

    for filename in os.listdir(DATA_PATH):

        if filename.endswith(".md"):
            file_path = os.path.join(DATA_PATH, filename)

            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

            document = Document(
                page_content=text,
                metadata={
                    "source": filename
                }
            )

            documents.append(document)

    return documents



def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )

    return text_splitter.split_documents(documents)



def calculate_chunk_ids(chunks):
    source_chunk_counts = {}

    for chunk in chunks:
        source = chunk.metadata.get("source")

        if source not in source_chunk_counts:
            source_chunk_counts[source] = 0

        chunk_index = source_chunk_counts[source]

        chunk_id = f"{source}:{chunk_index}"

        chunk.metadata["id"] = chunk_id

        source_chunk_counts[source] += 1

    return chunks




def add_to_chroma(chunks):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )

    # Έλεγχος των chunks που υπάρχουν ήδη στη βάση
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])

    print(f"Existing chunks in database: {len(existing_ids)}")

    # Επιλογή μόνο των νέων chunks
    new_chunks = [
        chunk
        for chunk in chunks
        if chunk.metadata["id"] not in existing_ids
    ]

    if not new_chunks:
        print("No new chunks to add.")
        return

    print(f"Adding new chunks: {len(new_chunks)}")

    batch_size = 100

    for i in range(0, len(new_chunks), batch_size):
        batch_chunks = new_chunks[i:i + batch_size]

        batch_ids = [
            chunk.metadata["id"]
            for chunk in batch_chunks
        ]

        # Αποθήκευση του συγκεκριμένου batch
        db.add_documents(
            documents=batch_chunks,
            ids=batch_ids
        )

        # Έλεγχος του πραγματικού αριθμού στη βάση
        current_count = db._collection.count()

        print(
            f"Batch {i // batch_size + 1}: "
            f"Sent {len(batch_chunks)} chunks | "
            f"Database total: {current_count}"
        )

    # Τελικός έλεγχος
    final_count = db._collection.count()

    print(f"\nFinal database chunks: {final_count}")
    print("Database update completed.")


documents = load_documents()
chunks = split_documents(documents)
chunks = calculate_chunk_ids(chunks)

# πληροφοριες
print(f"Documents loaded: {len(documents)}")
print(f"Chunks created: {len(chunks)}")


add_to_chroma(chunks)


