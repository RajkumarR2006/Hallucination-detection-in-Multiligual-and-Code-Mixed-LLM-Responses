
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

pairs = [
    # Likely equivalent
    ("Interactive proof and Automated proof",
     "Interactive proof and automated proof."),
    ("P versus NP", "P vs. NP"),
    ("Interactive proof", "interactive proof system"),

    # Related, but not equivalent
    ("Recursion", "Recurrence relation"),
    ("lower bounds", "upper bounds"),
    ("P", "PSPACE"),
    ("NP-complete", "NP-hard"),

    # Tamil examples from the dataset
    ("PSPACE", "P"),
    ("அனைத்து வரைபடங்கள்", "தொகுப்பிற்கான சொல்"),
    ("ரால்ஃப்", "என்னோ"),

    # Tanglish examples
    ("Time and space", "Time-space trade-off"),
    ("Reduction to a simple case", "Reduction to basics"),
]

print("Loading model...")
model = SentenceTransformer(MODEL_NAME)

texts = [text for pair in pairs for text in pair]
embeddings = model.encode(texts, normalize_embeddings=True)

for i, (a, b) in enumerate(pairs):
    score = float(
        cosine_similarity(
            embeddings[2 * i].reshape(1, -1),
            embeddings[2 * i + 1].reshape(1, -1),
        )[0, 0]
    )

    print(f"\nPair {i + 1}: similarity = {score:.4f}")
    print("  A:", a)
    print("  B:", b)
