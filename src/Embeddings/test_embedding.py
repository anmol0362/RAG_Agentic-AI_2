import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

print("\n" + "="*80)
print("RAW PARAMS")
print("="*80)
print("ENDPOINT:", endpoint)
print("API VERSION:", api_version)
print("DEPLOYMENT:", deployment)
print("KEY PREFIX:", api_key[:6] if api_key else "NO KEY")

# -------------------------------
# STEP 1: Check endpoint only
# -------------------------------
print("\n[TEST 1] Endpoint sanity check...")

if "openai.azure.com" not in endpoint:
    print("❌ WRONG ENDPOINT FORMAT")
else:
    print("✅ Endpoint format looks correct")

# -------------------------------
# STEP 2: Create client
# -------------------------------
print("\n[TEST 2] Creating client...")

client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=endpoint,
)

print("✅ Client created")

# -------------------------------
# STEP 3: Try embedding
# -------------------------------
print("\n[TEST 3] Trying embedding call...")

try:
    response = client.embeddings.create(
        model=deployment,
        input=["test embedding"]
    )
    print("✅ EMBEDDING SUCCESS")
    print("Vector length:", len(response.data[0].embedding))

except Exception as e:
    print("❌ EMBEDDING FAILED")
    print("ERROR:", e)