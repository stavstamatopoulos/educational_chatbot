import gradio as gr
from rag import generate_answer


import shutil
from pathlib import Path


#folders
pdf_folder = Path("data/pdf")
markdown_folder = Path("data/markdown")



# Επιστρέφει όλα τα διαθέσιμα αρχεία
def get_source_choices():

    choices = [
        ("Όλα")
    ]

    for file in sorted(markdown_folder.glob("*.md")):

        choices.append(
            (file.stem, file.name)
        )

    return choices



def chat(message, history, subject, mode):
    # Επιλογή βιβλίου
    if subject == "Όλα":
        source_filter = None
    else:
        source_filter = subject

    # RAG
    answer, results = generate_answer(message, source_filter, mode, history)

    # Προσθήκη στο ιστορικό του chatbot
    history = history or []

    # Προσθήκη ερώτησης στο ιστορικό
    history.append({
        "role": "user",
        "content": message
    })

    # Προσθήκη απάντησης στο ιστορικό
    history.append({
        "role": "assistant",
        "content": answer
    })

    # Καθαρίζει το textbox και επιστρέφει το νέο ιστορικό
    return "", history


#ανεβασμα αρχειων απο το ui
def upload_pdf(file_path):

    from build_database import update_database
    from pdf_to_md import convert_new_pdfs

    if file_path is None:
        return (
            "Δεν επιλέχθηκε αρχείο.",
            gr.Dropdown(
                choices=get_source_choices(),
                value="Όλα"
            )
        )
    
    try:
        source = Path(file_path)

        destination = pdf_folder / source.name

        # Έλεγχος αν υπάρχει ήδη
        if destination.exists():
           return (
                f"Το αρχείο '{source.name}' υπάρχει ήδη.",
                gr.Dropdown(
                    choices=get_source_choices(),
                    value="Όλα"
                )
            )
        # Αποθήκευση PDF
        shutil.copy2(source, destination)

        # PDF -> Markdown
        convert_new_pdfs()

        # Markdown -> Chunks -> Embeddings -> ChromaDB
        update_database()

        # Το Markdown που δημιουργήθηκε
        new_source = f"{source.stem}.md"

        # Ξαναδιαβάζω τα αρχεία μετά τη δημιουργία του νέου .md
        new_choices = get_source_choices()


        # Επιστρέφουμε μήνυμα + ανανεωμένο dropdown
        return (
            f"Το αρχείο '{source.name}' προστέθηκε επιτυχώς στη βάση δεδομένων.",
            gr.Dropdown(
                choices=get_source_choices(),
                value=new_source
            )
        )

    except Exception as error:
        return (
            f"Σφάλμα κατά την προσθήκη του αρχείου: {error}",
            gr.Dropdown(
                choices=get_source_choices(),
                value="Όλα"
            )
        )


#διαγραφη αρχειου απο την βαση
def delete_source(source):

    from langchain_chroma import Chroma

    # Προστασία: δεν διαγράφουμε την επιλογή "Όλα"
    if source == "Όλα":
        return (
            "Επίλεξε συγκεκριμένο αρχείο για διαγραφή.",
            gr.Dropdown(
                choices=get_source_choices(),
                value="Όλα"
            )
        )

    try:
        # Σύνδεση με ChromaDB
        db = Chroma(
            persist_directory="chroma"
        )

        # Βρίσκω όλα τα chunks του συγκεκριμένου source
        existing = db.get(
            where={"source": source},
            include=[]
        )

        ids = existing["ids"]

        # Διαγραφή chunks από τη βάση
        if ids:
            db.delete(ids=ids)

        # Διαγραφή Markdown
        markdown_file = markdown_folder / source

        if markdown_file.exists():
            markdown_file.unlink()

        # Διαγραφή PDF
        pdf_file = pdf_folder / f"{Path(source).stem}.pdf"

        if pdf_file.exists():
            pdf_file.unlink()

        # Νέα λίστα επιλογών μετά τη διαγραφή
        new_choices = get_source_choices()

        return (
            f"Το αρχείο '{source}' διαγράφηκε επιτυχώς "
            f"μαζί με {len(ids)} chunks.",
            gr.Dropdown(
                choices=new_choices,
                value="Όλα"
            )
        )

    except Exception as error:
        return (
            f"Σφάλμα κατά τη διαγραφή: {error}",
            gr.Dropdown(
                choices=get_source_choices(),
                value="Όλα"
            )
        )



with gr.Blocks() as demo:

    gr.Markdown(
        """
        <h1 style="text-align: center;">
            🎓 Εκπαιδευτικό Chatbot
        </h1>
        """
    )

    # Επιλογή αρχείου
    subject = gr.Dropdown(
        choices=get_source_choices(),
        value="Όλα",
        label="Επιλογή μαθήματος"
    )

    delete_button = gr.Button(
        "Διαγραφή επιλεγμένου μαθήματος"
    )

    # Επιλογή τρόπου λειτουργίας
    mode = gr.Radio(
        choices=[
            "Εξήγηση",
            "Καθοδήγηση"
        ],
        value="Εξήγηση",
        label="Τρόπος λειτουργίας"
    )
    
    chatbot = gr.Chatbot(
        height=300
    )

    # Πεδίο ερώτησης
    message = gr.Textbox(
        label="Ερώτηση",
        placeholder="Γράψε την ερώτησή σου εδώ..."
    )

    clear_button = gr.ClearButton(
        [message, chatbot],
        value="Καθαρισμός συνομιλίας"
    )

     # Upload PDF
    file_upload = gr.File(
        label="Ανέβασε νέο PDF",
        file_types=[".pdf"],
        type="filepath"
    )

    upload_button = gr.Button(
        "Προσθήκη στη βάση δεδομένων"
    )

    upload_status = gr.Textbox(
        label="Κατάσταση",
        interactive=False
    )

    # Αποστολή με Enter
    message.submit(
        fn=chat,
        inputs=[message, chatbot, subject, mode],
        outputs=[message, chatbot]
    )

    # ανεβασμα νέου μαθήματος
    upload_button.click(
        fn=upload_pdf,
        inputs=file_upload,
        outputs=[upload_status, subject]
    )

    #διαγραφή επιλεγμένου μαθήματος
    delete_button.click(
    fn=delete_source,
    inputs=subject,
    outputs=[
        upload_status,
        subject
    ]
)

if __name__ == "__main__":
    demo.launch(
            server_name="127.0.0.1",
            inbrowser=True
        )