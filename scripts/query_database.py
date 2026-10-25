from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function


CHROMA_PATH = "chroma"


def query_database(query_text, source_filter=None):
    # Σύνδεση με τη βαση
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )

    # Αναζήτηση σε όλα τα αρχεία ή σε συγκεκριμένο αρχείο
    results = db.similarity_search_with_score(
    query_text,
    k=5,
    filter={"source": source_filter} if source_filter else None
)

    return results


