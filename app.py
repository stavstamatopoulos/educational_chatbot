
import gradio as gr

from rag import generate_answer, SOURCES


def chat(message, history, subject):
    # Επιλογή βιβλίου
    if subject == "Όλα":
        source_filter = None
    else:
        source_filter = SOURCES[subject.lower()]

    # RAG
    answer, results = generate_answer(message, source_filter)

    # Προσθήκη στο ιστορικό του chatbot
    history = history or []

    history.append({
        "role": "user",
        "content": message
    })

    history.append({
        "role": "assistant",
        "content": answer
    })

    # Καθαρίζει το textbox και επιστρέφει το νέο ιστορικό
    return "", history


with gr.Blocks() as demo:

    gr.Markdown(
        """
        <h1 style="text-align: center;">
            🎓 Εκπαιδευτικό Chatbot
        </h1>
        """
    )

    subject = gr.Dropdown(
        choices=[
            "Όλα",
            "Μαθηματικά",
            "Φυσική",
            "Πληροφορική",
            "Οικονομία"
        ],
        value="Όλα",
        label="Επιλογή μαθήματος"
    )

    chatbot = gr.Chatbot(
        height=300
    )

    message = gr.Textbox(
        label="Ερώτηση",
        placeholder="Γράψε την ερώτησή σου εδώ..."
    )

    submit_button = gr.Button("Αποστολή")

    clear_button = gr.ClearButton(
        [message, chatbot],
        value="Καθαρισμός συνομιλίας"
    )

    # Αποστολή με το κουμπί
    submit_button.click(
        fn=chat,
        inputs=[message, chatbot, subject],
        outputs=[message, chatbot]
    )

    # Αποστολή και με Enter
    message.submit(
        fn=chat,
        inputs=[message, chatbot, subject],
        outputs=[message, chatbot]
    )


if __name__ == "__main__":
    demo.launch(
            server_name="127.0.0.1",
            inbrowser=True
        )