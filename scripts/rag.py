from langchain_ollama import ChatOllama
from query_database import query_database


MODEL_NAME = "qwen3:4b-instruct"


def generate_answer(query_text, source_filter=None, mode="Εξήγηση", history=None):

    history = history or []

    # Κρατάω μόνο τα 6 τελευταία μηνύματα
    recent_history = history[-6:]


    # 1. Retrieval με βάση την τωρινή ερώτηση
    results = query_database(query_text, source_filter)


    # 2. Ενώνω τα chunks σε context
    context = "\n\n---\n\n".join(
        document.page_content
        for document, score in results
    )


    # 3. Ορίζω το LLM
    model = ChatOllama(
        model=MODEL_NAME,
        temperature=0
    )


    # 4. Prompt ανάλογα με το mode
    if mode == "Εξήγηση":

        system_prompt = """
Είσαι εκπαιδευτικός βοηθός.
Απάντησε χρησιμοποιώντας μόνο τις πληροφορίες που υπάρχουν στο εκπαιδευτικό υλικό που σου παρέχεται.
Λάβε υπόψη σου και την προηγούμενη συζήτηση.
Αν το εκπαιδευτικό υλικό δεν περιέχει αρκετές πληροφορίες, πες ότι δεν υπάρχει επαρκής πληροφορία στο διαθέσιμο υλικό.
Μην επινοείς πληροφορίες.
Απάντησε αποκλειστικά στα ελληνικά, με σαφή και κατανοητό τρόπο.
"""

    elif mode == "Καθοδήγηση":

        system_prompt = """
Είσαι εκπαιδευτικός βοηθός που λειτουργεί με την Σωκρατική μέθοδο.
Μην δίνεις αμέσως την τελική απάντηση ή λύση.
Κάνε μια ερώτηση την κάθε φορά.
Καθοδήγησε τον μαθητή βήμα προς βήμα χρησιμοποιώντας ερωτήσεις και μικρές υποδείξεις.
Χρησιμοποίησε αποκλειστικά τις πληροφορίες που υπάρχουν στο εκπαιδευτικό υλικό.
Μην επινοείς πληροφορίες.
Απάντησε αποκλειστικά στα ελληνικά.
"""


    # 5. Ξεκινάω τα messages
    messages = [
        ("system", system_prompt)
    ]


    # 6. Προσθέτω το ιστορικό
    for message in recent_history:

        if isinstance(message, dict):

            role = message.get("role")
            content = message.get("content", "")

            if role == "user":
                messages.append(
                    ("human", str(content))
                )

            elif role == "assistant":
                messages.append(
                    ("ai", str(content))
                )


    # 7. Προσθέτω τη νέα ερώτηση
    messages.append(
        (
            "human",
            f"""
Νέο μήνυμα του μαθητή:
{query_text}

Εκπαιδευτικό υλικό:
{context}

Απάντηση:
"""
        )
    )


    # 8. Δημιουργία απάντησης
    response = model.invoke(messages)

    return response.text, results



if __name__ == "__main__":
  query = "Τι ονομάζεται αλγόριθμος;"

  # None = αναζήτηση σε όλα τα αρχεία
  #22-0273-01_Mathimatika-Teuchos-B_G-Lykeiou-Thetikon-Spoudon-kai-Spoudon-Oikonomias-Pliroforikis_Vivlio-Mathiti.md
  #22-0279-01_Pliroforiki_G-Lykeiou-SpOikPlir_Sympliromatiko-Ekpaideutiko-Yliko.md
  #22-0282-01_Fysiki-Teuchos-G_G-Lykeiou-Thetikon-Spoudon-kai-Spoudon-Ygeias_Vivlio-Mathiti.md
  #22-0299-01_V1_Arches-Oikonomikis-Theorias_G-Lykeiou-Spoudon-Oikonomias-Pliroforikis_Vivlio-Mathiti.md
  
  source_filter = None
  mode = "Καθοδήγηση" # "Εξήγηση" ή "Καθοδήγηση"
  answer, results = generate_answer(query, source_filter, mode)

  print(f"\nΕρώτηση:\n{query}")

  print("\n--------------- RETRIEVED CHUNKS -----------------")

  for i, (document, score) in enumerate(results, start=1):
      print(f"\n--- Chunk {i} ---")
      print(f"Source: {document.metadata.get('source')}")
      print(f"Chunk ID: {document.metadata.get('id')}")
      print(f"Distance: {score:.4f}")
      print(f"Mode: {mode}")
      print()
      print(document.page_content)

  print("\n------------------------------------------------")

  print(f"\nΑπάντηση:\n{answer}")