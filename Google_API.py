from google.oauth2 import service_account
from googleapiclient.discovery import build

# Load the credentials
creds = service_account.Credentials.from_service_account_file("path/to/credentials.json")

# Build the API client
docs_service = build("docs", "v1", credentials=creds)

# Extract text from a Google Doc
document_id = "your-document-id"
doc = docs_service.documents().get(documentId=document_id).execute()
doc_content = doc.get("body").get("content")
text = ""

for value in doc_content:
    if "textRun" in value.get("paragraph").get("elements")[0]:
        text += value.get("paragraph").get("elements")[0].get("textRun").get("content").strip()

print(text)
