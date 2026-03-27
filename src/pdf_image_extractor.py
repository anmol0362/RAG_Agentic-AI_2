import fitz  # pymupdf
import base64
import json
from pathlib import Path
from anthropic import Anthropic

client = Anthropic()


# =========================
# HELPER
# =========================
def image_to_base64(image_bytes):
    return base64.standard_b64encode(image_bytes).decode("utf-8")


# =========================
# CLAUDE CALL
# =========================
def describe_image_with_claude(image_base64, page_num, source):
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_base64,
                        },
                    },
                    {
                        "type": "text",
                        "text": """Extract ALL readable text from this image.
Also briefly describe diagrams/tables if present.

Return ONLY JSON:
{
  "description": "short description",
  "extracted_text": "all text from image"
}"""
                    }
                ],
            }
        ],
    )

    raw = response.content[0].text
    raw = raw.strip().replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw)
    except:
        return {"description": "", "extracted_text": raw}


# =========================
# 1️⃣ EMBEDDED IMAGE EXTRACTION
# =========================
def extract_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    source_name = Path(pdf_path).name
    results = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]

            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_base64 = image_to_base64(image_bytes)

                print(f"  Image page {page_num+1}, img {img_index+1}")

                vision_result = describe_image_with_claude(
                    image_base64, page_num + 1, source_name
                )

                results.append({
                    "doc_id": f"{source_name}_p{page_num+1}_img{img_index+1}",
                    "source": source_name,
                    "file_type": "pdf_image",
                    "page": page_num + 1,
                    "text": f"{vision_result.get('description','')}\n\n{vision_result.get('extracted_text','')}".strip()
                })

            except Exception as e:
                print(f"[ERROR] Image {img_index+1}: {e}")

    doc.close()
    return results


# =========================
# 2️⃣ FULL PAGE OCR (IMPORTANT)
# =========================
def extract_pages_with_claude(pdf_path):
    doc = fitz.open(pdf_path)
    source_name = Path(pdf_path).name
    results = []

    for page_num in range(len(doc)):
        page = doc[page_num]

        try:
            print(f"  OCR page {page_num+1}/{len(doc)}")

            # render full page as image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            image_bytes = pix.tobytes("png")
            image_base64 = image_to_base64(image_bytes)

            vision_result = describe_image_with_claude(
                image_base64, page_num + 1, source_name
            )

            results.append({
                "doc_id": f"{source_name}_p{page_num+1}",
                "source": source_name,
                "file_type": "pdf_page",
                "page": page_num + 1,
                "text": f"{vision_result.get('description','')}\n\n{vision_result.get('extracted_text','')}".strip()
            })

        except Exception as e:
            print(f"[ERROR] Page {page_num+1}: {e}")

    doc.close()
    return results