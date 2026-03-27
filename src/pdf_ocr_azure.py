import os
from dotenv import load_dotenv
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

load_dotenv()

endpoint = os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT")
key = os.getenv("AZURE_DOC_INTELLIGENCE_KEY")

if not endpoint or not key:
    raise ValueError("Missing Azure Document Intelligence credentials in .env")

client = DocumentIntelligenceClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
)


def extract_text_from_pdf(pdf_path, model="prebuilt-layout"):
    """
    Extract text from scanned/image-based PDFs using Azure Document Intelligence.

    Args:
        pdf_path (str): Path to PDF file
        model (str): Azure prebuilt model ("prebuilt-read" or "prebuilt-layout")

    Returns:
        str: Extracted text
    """
    with open(pdf_path, "rb") as f:
        poller = client.begin_analyze_document(
            model_id=model,
            body=f
        )
        result = poller.result()

    lines = []

    for page in result.pages:
        lines.append(f"\n--- PAGE {page.page_number} ---\n")

        for line in page.lines:
            lines.append(line.content)

    return "\n".join(lines).strip()