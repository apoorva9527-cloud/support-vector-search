import chromadb
from sentence_transformers import SentenceTransformer
from pprint import pprint

# Create or reuse persistent Chroma database
client = chromadb.PersistentClient(path="./chroma_store")

# Open or create collection (no automatic embedding function)
collection = client.get_or_create_collection(
    name="support_knowledge_base",
    embedding_function=None
)

# Knowledge base data
ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]

documents = [
    "Customers can return products within 30 days of delivery.",
    "Refunds are processed within 5 to 7 business days after the return is approved.",
    "Orders above 499 rupees qualify for free shipping.",
    "You can reset your password from the account settings page.",
    "Express delivery orders usually arrive within 24 to 48 hours."
]

metadatas = [
    {"category": "returns"},
    {"category": "returns"},
    {"category": "shipping"},
    {"category": "account"},
    {"category": "shipping"}
]

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# The same embedding model must encode both stored FAQs and user queries
# so they live in the same vector space and can be compared meaningfully.
document_embeddings = model.encode(documents).tolist()

# Upsert data into Chroma
collection.upsert(
    ids=ids,
    documents=documents,
    metadatas=metadatas,
    embeddings=document_embeddings
)

print("\n=== Collection Information ===")
print("Collection Name:", collection.name)
print("Document Count:", collection.count())

print("\n=== Collection Peek ===")
pprint(collection.peek())

print("\n=== Get doc4 ===")
# get() retrieves records by exact id and does not perform semantic search.
pprint(collection.get(ids=["doc4"]))


def run_query(query_text, n_results):
    print("\n" + "=" * 70)
    print("User Query:", query_text)
    print("=" * 70)

    query_embedding = model.encode(query_text).tolist()

    # query() performs vector similarity search and returns nearest matches.
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    ids_result = results["ids"][0]
    docs_result = results["documents"][0]
    metas_result = results["metadatas"][0]

    distances_result = results.get("distances", [[]])
    distances_result = distances_result[0] if distances_result else []

    for rank, doc_id in enumerate(ids_result, start=1):
        print(f"\nRank {rank}")
        print("ID:", doc_id)
        print("Document:", docs_result[rank - 1])
        print("Metadata:", metas_result[rank - 1])

        if distances_result:
            print("Distance:", distances_result[rank - 1])

    return results


# Query 1
run_query(
    "I want to return my shoes and get my money back",
    3
)

# Query 2
run_query(
    "How do I change my login password?",
    2
)

# Query 3
query3_results = run_query(
    "Can I pay with UPI?",
    3
)

# Gap analysis
top_id = query3_results["ids"][0][0]
top_category = query3_results["metadatas"][0][0]["category"]

print("\n--- Gap analysis ---")
print(
    f"The highest-ranked result was {top_id} from the '{top_category}' category."
)
print(
    "This may still be a weak business answer because the knowledge base contains no payment or UPI FAQ, so the closest vector match is only semantically related rather than actually answering the payment question."
)