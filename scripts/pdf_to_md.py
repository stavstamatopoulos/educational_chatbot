import os
from dotenv import load_dotenv

load_dotenv()
os.environ["TORCH_COMPILE_DISABLE"] = "1" #to apenorgopoiw giati moy dinei error me to torch.compile
from pathlib import Path

#folders
pdf_folder = Path("data/pdf")
markdown_folder = Path("data/markdown")


# Περιγραφή εικόνων μέσω Gemini
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption


def convert_with_image_annotation(input_doc_path):
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found")

    model = "gemini-3.5-flash-lite"

    picture_desc_api_option = PictureDescriptionApiOptions(
        url="https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        prompt=(
            "Περιέγραψε αυτή την εικόνα αναλυτικά σε μία παράγραφο στα ελληνικά. "
            "Εστίασε στις εκπαιδευτικές πληροφορίες, στα διαγράμματα, στα γραφήματα, "
            "στις επιγραφές, στο κείμενο και στις σχέσεις μεταξύ των στοιχείων. "
            "Αν η εικόνα περιέχει ελληνικό κείμενο, διάβασέ το και συμπερίλαβε "
            "τις σημαντικές πληροφορίες στην περιγραφή."
        ),
        params={
            "model": model,
        },
        headers={
            "Authorization": "Bearer " + api_key,
        },
        timeout=120,
        concurrency=1,
    )

    pipeline_options = PdfPipelineOptions(
        do_picture_description=True,
        picture_description_options=picture_desc_api_option,
        enable_remote_services=True,
        generate_picture_images=True,
        images_scale=2,
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    conv_res = converter.convert(source=input_doc_path)

    return conv_res

#τα παρακατω τα βαζω σε συναρτηση για να υλοποιησω την λειτουργια "προσθηκη αρχειων στην βαση"
def convert_new_pdfs():

    #anazhthsh arxeiwn
    pdf_files = list(pdf_folder.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in the folder.")

    
    # επεξεργασία όλων των PDF
    for pdf_file in pdf_files:
        output_file = markdown_folder / f"{pdf_file.stem}.md"

        # Αν έχει ήδη μετατραπεί παράλειψε το
        if output_file.exists():
            print(f"Skipping: {pdf_file.name} (already converted)")
            continue

        print(f"Processing: {pdf_file.name}")

        result = convert_with_image_annotation(pdf_file)

        annotated_doc = result.document
        markdown_text = annotated_doc.export_to_markdown()

        output_file = markdown_folder / f"{pdf_file.stem}.md"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(markdown_text)

        print(f"Saved markdown to: {output_file}")


if __name__ == "__main__":
    convert_new_pdfs()

