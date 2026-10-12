from langchain_ollama import ChatOllama
from query_database import query_database


MODEL_NAME = "llama3"

#αλλαζω τα ονοματα των αρχειων για ευκολια στο φιλτράρισμα
SOURCES = {
    "μαθηματικά": "22-0273-01_Mathimatika-Teuchos-B_G-Lykeiou-Thetikon-Spoudon-kai-Spoudon-Oikonomias-Pliroforikis_Vivlio-Mathiti.md",

    "πληροφορική": "22-0279-01_Pliroforiki_G-Lykeiou-SpOikPlir_Sympliromatiko-Ekpaideutiko-Yliko.md",

    "φυσική": "22-0282-01_Fysiki-Teuchos-G_G-Lykeiou-Thetikon-Spoudon-kai-Spoudon-Ygeias_Vivlio-Mathiti.md",

    "οικονομία": "22-0299-01_V1_Arches-Oikonomikis-Theorias_G-Lykeiou-Spoudon-Oikonomias-Pliroforikis_Vivlio-Mathiti.md",
}


def generate_answer(query_text, source_filter=None):
    # 1. κανω retrive τα chunks που ταιριαζουν με την ερωτηση
    results = query_database(query_text, source_filter)

    # 2. Ενώνω τα chunks σε ένα context
    context = "\n\n---\n\n".join(
        document.page_content
        for document, score in results
    )

    # 3. οριζω το LLM
    model = ChatOllama(
        model=MODEL_NAME,
        temperature=0
    )

    # 4. Prompt στο LLM
    messages = [
        (
            "system",
            """
Είσαι εκπαιδευτικός βοηθός. 
Απάντησε στην ερώτηση χρησιμοποιώντας μόνο τις πληροφορίες που υπάρχουν στα αποσπάσματα του εκπαιδευτικού υλικού. 
Αν τα αποσπάσματα δεν περιέχουν αρκετές πληροφορίες, πες ότι δεν υπάρχει επαρκής πληροφορία στο διαθέσιμο υλικό. 
Μην επινοείς πληροφορίες. 
Απάντησε στα ελληνικά, με σαφή και κατανοητό τρόπο.
"""
        ),
        (
            "human",
            f"""
Ερώτηση:
{query_text}

Αποσπάσματα:
{context}

Απάντηση:
"""
        )
    ]

    # 5. Δημιουργία απάντησης
    response = model.invoke(messages)

    return response.content, results

if __name__ == "__main__":
  query = "Τι ονομάζεται αλγόριθμος;"

  # None = αναζήτηση σε όλα τα αρχεία
  #SOURCES["πληροφορική"] = για να ψαξω μόνο στο αρχείο της πληροφορικής
  source_filter = None

  answer, results = generate_answer(query, source_filter)

  print(f"\nΕρώτηση:\n{query}")

  print("\n--------------- RETRIEVED CHUNKS -----------------")

  for i, (document, score) in enumerate(results, start=1):
      print(f"\n--- Chunk {i} ---")
      print(f"Source: {document.metadata.get('source')}")
      print(f"Chunk ID: {document.metadata.get('id')}")
      print(f"Distance: {score:.4f}")
      print()
      print(document.page_content)

  print("\n------------------------------------------------")

  print(f"\nΑπάντηση:\n{answer}")