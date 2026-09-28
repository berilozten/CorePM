from foundry_local_sdk import Configuration, FoundryLocalManager
import math
import os
import re


# ============================================================
# 0. DEMO START SCREEN
# ============================================================

print("=" * 60)
print("COREPM — LOCAL PRODUCT INTELLIGENCE")
print("=" * 60)
print("Analyze customer feedback with local embeddings, semantic retrieval, and a local LLM.\n")


# ============================================================
# 1. PROJECT CONFIGURATION & DATA
# ============================================================

FILE_PATH = "docs/feedback.txt"

EMBEDDING_MODEL_ID = "qwen3-embedding-0.6b-generic-cpu:1"
GENERATION_MODEL_ID = "qwen3-0.6b-generic-gpu:2"

TOP_K = 3

DEFAULT_QUESTION = (
    "What problems are users experiencing during onboarding "
    "and what should the product team prioritize?"
)

# Error Handling: Verify feedback file exists
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(
        f"Feedback file not found at: '{FILE_PATH}'. "
        f"Please ensure the file exists before running."
    )

# Interactive User Question (fallback to DEFAULT_QUESTION if empty)
print("Enter your product question (Press Enter for default):")
user_input = input("> ").strip()
USER_QUESTION = user_input if user_input else DEFAULT_QUESTION

print(f"\nActive Question:\n{USER_QUESTION}")


# ============================================================
# 2. LOAD CUSTOMER FEEDBACK & UNIT-LEVEL RECORD CHUNKING
# ============================================================

with open(FILE_PATH, "r", encoding="utf-8") as file:
    document_text = file.read().strip()

if not document_text:
    raise ValueError(f"Feedback file '{FILE_PATH}' is empty.")


def parse_feedback_records(raw_text: str):
    """
    Parse document into independent, structured feedback records.
    Extracts User ID, Feature, and Feedback fields robustly.
    Falls back to paragraph separation if headers are absent.
    """
    raw_text = raw_text.strip()
    if not raw_text:
        return []

    # Check for structured 'User ID:' records
    if re.search(r"(?:^|\n)\s*User ID:", raw_text, re.IGNORECASE):
        blocks = [b.strip() for b in re.split(r"(?=(?:^|\n)\s*User ID:)", raw_text, flags=re.IGNORECASE) if b.strip()]
        records = []
        for idx, block in enumerate(blocks, start=1):
            uid_match = re.search(r"User ID:\s*([^\n]+)", block, re.IGNORECASE)
            feat_match = re.search(r"Feature:\s*([^\n]+)", block, re.IGNORECASE)
            fb_match = re.search(r"Feedback:\s*(.*)", block, re.IGNORECASE | re.DOTALL)

            user_id = uid_match.group(1).strip() if uid_match else f"User-{idx}"
            feature = feat_match.group(1).strip() if feat_match else "General"
            if fb_match:
                feedback = fb_match.group(1).strip()
            else:
                lines = [l for l in block.splitlines() if not re.match(r"^\s*(User ID|Feature):", l, re.IGNORECASE)]
                feedback = " ".join(lines).strip()

            if not feedback:
                feedback = block

            records.append({
                "record_id": idx,
                "user_id": user_id,
                "feature": feature,
                "feedback": feedback
            })
        return records
    else:
        # Fallback: Split by double newline or treat as separate paragraphs
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", raw_text) if p.strip()]
        records = []
        for idx, para in enumerate(paragraphs, start=1):
            feat_match = re.search(r"Feature:\s*([^\n]+)", para, re.IGNORECASE)
            feature = feat_match.group(1).strip() if feat_match else "General"
            lines = [l for l in para.splitlines() if not re.match(r"^\s*Feature:", l, re.IGNORECASE)]
            feedback = " ".join(lines).strip()
            records.append({
                "record_id": idx,
                "user_id": f"User-{idx}",
                "feature": feature,
                "feedback": feedback
            })
        return records


def get_feedback_chunks(records, max_words=80):
    """
    Keep each customer feedback as its own natural retrieval unit.
    Only divide into sub-chunks if the feedback exceeds max_words.
    Prepares text with Feature and Feedback context for optimal embedding.
    """
    chunks = []
    for record in records:
        feature = record["feature"]
        user_id = record["user_id"]
        feedback = record["feedback"]
        rec_id = record["record_id"]
        words = feedback.split()
        if len(words) <= max_words:
            chunks.append({
                "record_id": rec_id,
                "user_id": user_id,
                "feature": feature,
                "feedback": feedback,
                "text": f"Feature: {feature} | Feedback: {feedback}"
            })
        else:
            raw_sentences = re.split(r"(?<=[.!?])\s+", feedback.strip())
            sentences = [s.strip() for s in raw_sentences if s.strip()]
            curr_chunk = []
            curr_len = 0
            for s in sentences:
                s_words = s.split()
                if curr_len + len(s_words) > max_words and curr_chunk:
                    sub_text = " ".join(curr_chunk)
                    chunks.append({
                        "record_id": rec_id,
                        "user_id": user_id,
                        "feature": feature,
                        "feedback": sub_text,
                        "text": f"Feature: {feature} | Feedback: {sub_text}"
                    })
                    curr_chunk = [s]
                    curr_len = len(s_words)
                else:
                    curr_chunk.append(s)
                    curr_len += len(s_words)
            if curr_chunk:
                sub_text = " ".join(curr_chunk)
                chunks.append({
                    "record_id": rec_id,
                    "user_id": user_id,
                    "feature": feature,
                    "feedback": sub_text,
                    "text": f"Feature: {feature} | Feedback: {sub_text}"
                })
    return chunks


records = parse_feedback_records(document_text)
chunks = get_feedback_chunks(records)

print(f"\nTotal feedback records parsed: {len(records)}")
print(f"Total retrieval units (chunks) generated: {len(chunks)}")


# ============================================================
# 3. COSINE SIMILARITY
# ============================================================

def cosine_similarity(vector_a, vector_b):
    """
    Calculate cosine similarity between two vectors.
    """

    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


# ============================================================
# 4. INITIALIZE MICROSOFT FOUNDRY LOCAL
# ============================================================

print("\nInitializing Microsoft Foundry Local...")

config = Configuration(app_name="CorePM")

FoundryLocalManager.initialize(config)

manager = FoundryLocalManager.instance

print("Embedding model:", EMBEDDING_MODEL_ID)
print("Generation model:", GENERATION_MODEL_ID)


# ============================================================
# 5. LOAD EMBEDDING MODEL
# ============================================================

embedding_model = manager.catalog.get_model_variant(
    EMBEDDING_MODEL_ID
)

if embedding_model is None:
    raise RuntimeError(
        f"Embedding model not found: {EMBEDDING_MODEL_ID}"
    )

print("\nLoading embedding model...")

if not embedding_model.is_cached:
    print("Downloading embedding model...")
    embedding_model.download()

embedding_model.load()

embedding_client = embedding_model.get_embedding_client()


# ============================================================
# 6. CREATE EMBEDDINGS FOR CUSTOMER FEEDBACK
# ============================================================

print("\nCreating embeddings for customer feedback...")

chunk_embeddings = []

for index, chunk in enumerate(chunks):

    result = embedding_client.generate_embedding(chunk["text"])

    embedding = result.data[0].embedding

    chunk_embeddings.append(embedding)

    print(
        f"Record {chunk['record_id']} (User: {chunk['user_id']}): "
        f"{len(embedding)} dimensions"
    )


# ============================================================
# 7. EMBED USER QUESTION
# ============================================================

print("\nUser question:")
print(USER_QUESTION)

print("\nCreating embedding for user question...")

question_result = embedding_client.generate_embedding(
    USER_QUESTION
)

question_embedding = question_result.data[0].embedding


# ============================================================
# 8. SEMANTIC SEARCH
# ============================================================

results = []

for index, chunk in enumerate(chunks):

    similarity = cosine_similarity(
        question_embedding,
        chunk_embeddings[index]
    )

    results.append({
        "chunk": chunk,
        "record_id": chunk["record_id"],
        "user_id": chunk["user_id"],
        "feature": chunk["feature"],
        "feedback": chunk["feedback"],
        "similarity": similarity
    })


# Sort from most relevant to least relevant
results.sort(
    key=lambda item: item["similarity"],
    reverse=True
)


# ============================================================
# 9. DISPLAY RETRIEVED INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("RETRIEVING RELEVANT CUSTOMER FEEDBACK...")
print("=" * 60)

for rank, result in enumerate(results, start=1):
    print(f"\nRank {rank}")
    print(
        f"Similarity: "
        f"{result['similarity']:.4f}"
    )
    print(
        f"User ID: "
        f"{result['user_id']}"
    )
    print(
        f"Feature: "
        f"{result['feature']}"
    )
    print(
        f"Feedback: "
        f"{result['feedback']}"
    )


# ============================================================
# 10. BUILD RAG CONTEXT & DETERMINISTIC PM PRIORITY RUBRIC
# ============================================================

def determine_priority(retrieved_results):
    """
    CorePM PM Impact Rubric:
    Determines priority deterministically based on product impact:
    - HIGH:
        * Financial loss, double charge, transaction failure ("charged twice", "double charge")
        * App crash, infinite spin/hang blocking action ("crashes", "crash", "spins infinitely")
        * Explicit privacy concern or security risk ("privacy concern", "security risk")
        * Authentication / login block preventing app access ("cannot log in", "rejecting password")
        * Forced core blocking action ("forced to")
    - MEDIUM:
        * Significant onboarding friction or UX breakdown ("too much information", "confusing")
        * Search failure or filter reset ("unrelated results", "filters reset", "takes too long")
        * Degraded performance (10s load, blank charts, lingering pending state)
        * Stuck loading spinner in payment/checkout flow ("loading spinner", "afraid to try again")
        * Recurring friction across multiple users on the same feature (feature count >= 2)
    - LOW:
        * Minor convenience or cosmetic preferences ("too many notifications", "spending summary")
        * Profile photo adjustment or minor layout polish ("hidden below the keyboard")
        * Notification volume or category controls
    """
    if not retrieved_results:
        return "Medium"

    # High severity patterns: severe user impact, data/financial risk, crash, hard blocks
    high_patterns = [
        r"\bcharged\s+twice\b",
        r"\bdouble\s+charge\b",
        r"\bcrashes?\b",
        r"\bcrash\b",
        r"\bprivacy\s+concern\b",
        r"\bsecurity\s+risk\b",
        r"\bforced\s+to\b",
        r"\bcannot\s+log\s+in\b",
        r"\brejecting\s+(?:my\s+)?(?:correct\s+)?password\b",
        r"\bspins?\s+infinitely\b",
        r"\bnever\s+received.*(?:confirmation|subscription)\b"
    ]

    # Medium severity patterns: meaningful degradation, stuck spinner, or friction
    medium_patterns = [
        r"\btoo\s+much\s+information\b",
        r"\bunrelated\s+results\b",
        r"\bfilters?\s+reset\b",
        r"\btakes?\s+(?:around\s+)?(?:ten|\d+)\s+seconds?\b",
        r"\bblank\b",
        r"\bpending\s+for\s+(?:the\s+)?entire\s+day\b",
        r"\bcancelled.*still\s+shows\b",
        r"\bcannot\s+find\s+a\s+clear\s+way\s+to\s+contact\b",
        r"\bconfusing\b",
        r"\btakes?\s+too\s+long\b",
        r"\bloading\s+spinner\b",
        r"\bafraid\s+to\s+try\s+again\b"
    ]

    # 1. Check for immediate HIGH triggers in any retrieved record
    for r in retrieved_results:
        text = (r.get("feature", "") + " " + r.get("feedback", "")).lower()
        for pat in high_patterns:
            if re.search(pat, text, re.IGNORECASE):
                return "High"

    # 2. Count feature occurrences to assess widespread friction
    features = [r.get("feature", "").lower().strip() for r in retrieved_results if r.get("feature")]
    feature_counts = {}
    for f in features:
        feature_counts[f] = feature_counts.get(f, 0) + 1

    has_multi_complaint = any(count >= 2 for count in feature_counts.values())

    # 3. Check for MEDIUM triggers
    has_medium = False
    for r in retrieved_results:
        text = (r.get("feature", "") + " " + r.get("feedback", "")).lower()
        for pat in medium_patterns:
            if re.search(pat, text, re.IGNORECASE):
                has_medium = True
                break

    if has_medium or has_multi_complaint:
        return "Medium"

    return "Low"


# Use strictly the top-K retrieved feedback records as evidence for the generation model
top_k_results = results[:TOP_K]
rubric_priority = determine_priority(top_k_results)

retrieved_context_items = []
for idx, r in enumerate(top_k_results, start=1):
    item_str = (
        f"[Feedback {idx}]\n"
        f"User ID: {r['user_id']}\n"
        f"Feature: {r['feature']}\n"
        f"Feedback: {r['feedback']}"
    )
    retrieved_context_items.append(item_str)

retrieved_context = "\n\n".join(retrieved_context_items)


# ============================================================
# 11. UNLOAD EMBEDDING MODEL
# ============================================================

embedding_model.unload()

print("\nEmbedding model unloaded.")


# ============================================================
# 12. LOAD LOCAL GENERATION MODEL
# ============================================================

print("\nLoading local generation model...")

generation_model = manager.catalog.get_model_variant(
    GENERATION_MODEL_ID
)

if generation_model is None:
    raise RuntimeError(
        f"Generation model not found: {GENERATION_MODEL_ID}"
    )

if not generation_model.is_cached:
    print("Downloading generation model...")
    generation_model.download()

generation_model.load()

chat_client = generation_model.get_chat_client()

# Configure inference parameters to prevent repetition and hallucination
chat_client.settings.temperature = 0.2
chat_client.settings.max_tokens = 500


# ============================================================
# 13. GROUNDED GENERATION PROMPT (SYSTEM + USER)
# ============================================================

system_prompt = """You are CorePM, a local AI assistant for Product Managers.
Analyze retrieved customer feedback records and synthesize concise, professional product insights.

PRIORITY GUIDELINES:
- High: Completely blocks onboarding or core flows, causes financial/transaction failure, or creates severe privacy/security risks.
- Medium: Causes significant friction, confusion, or degradation without completely preventing task completion.
- Low: Minor cosmetic flaws, polish suggestions, or small UX inconveniences.

STRICT RULES:
1. Base your answer ONLY on the provided RETRIEVED FEEDBACK records. Synthesize common issues if multiple records are present.
2. Never repeat the same sentence, phrase, or idea across sections.
3. Do not restate the Evidence quote in the Why section.
4. Keep each section strictly concise:
   - Problem: Exactly 1 concise sentence summarizing the core user problem.
   - Evidence: Exactly 1 direct quote from the most relevant retrieved feedback.
   - Priority: Output ONLY one word: High, Medium, or Low based on the priority guidelines.
   - Why: At most 1-2 concise sentences justifying priority. No repeated ideas, no assumptions outside evidence.
   - Recommendation: Exactly 1 concise, actionable sentence for the product team.
5. Output only the requested five sections. Do not add numbered lists, commentary, or math symbols ($)."""

user_prompt = f"""RETRIEVED FEEDBACK:
{retrieved_context}

USER QUESTION:
{USER_QUESTION}

Output your insight using this exact structure:

Problem:
Users are confused by the onboarding flow and cannot skip the bank connection step.

Evidence:
"I couldn't find the 'Skip' button anywhere, so I was forced to connect my bank account before even seeing the dashboard."

Priority:
High

Why:
The issue creates both onboarding friction and a privacy concern before users reach the dashboard.

Recommendation:
Make the bank connection step optional and provide a clearly visible Skip action.

Now generate the final insight for the feedback above following this exact format:
Problem:
/no_think"""


# ============================================================
# 14. GENERATE PRODUCT INSIGHT
# ============================================================

messages = [
    {
        "role": "system",
        "content": system_prompt
    },
    {
        "role": "user",
        "content": user_prompt
    }
]


print("\n" + "=" * 60)
print("COREPM PRODUCT INSIGHT")
print("=" * 60)
print("Priority determined using CorePM PM impact rubric.\n")

response = chat_client.complete_chat(messages)
raw_answer = response.choices[0].message.content


def clean_pm_insight(text: str, override_priority: str = None) -> str:
    """Clean reasoning tags, strip repetitions, placeholders, and format the final PM insight."""
    # 1. Remove closed or open think blocks
    if "</think>" in text:
        text = text.split("</think>")[-1]
    else:
        text = re.sub(r"<\s*think\s*>", "", text, flags=re.IGNORECASE)

    # 2. Clean LaTeX notation like $ \text{Header:} $ or markdown asterisks
    text = re.sub(r"\$\s*\\text\{([A-Za-z]+):?\}\s*\$?:?", r"\1:", text, flags=re.IGNORECASE)
    text = re.sub(r"[%*#$+=^~@&]+", " ", text)
    text = re.sub(r"\b(was|is|the|to|so|in|on|at|for|were|are)[,;:\-–—\(\)\[\]%]+\s+", r"\1 ", text, flags=re.IGNORECASE)
    text = re.sub(r"  +", " ", text)

    # 3. Extract sections using regex
    pattern = r"(Problem|Evidence|Priority|Why|Recommendation):\s*(.*?)(?=(?:Problem|Evidence|Priority|Why|Recommendation):|$)"
    matches = list(re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL))

    data = {}
    for m in matches:
        sec_name = m.group(1).capitalize()
        content = m.group(2).strip()
        if sec_name not in data:
            data[sec_name] = content

    def deduplicate_sentences(s_text: str, max_s: int = 2) -> str:
        raw_s = re.split(r"(?<=[.!?])\s+", s_text.strip())
        seen = set()
        cleaned = []
        for s in raw_s:
            sc = s.strip()
            if not sc:
                continue
            norm = re.sub(r"[^\w\s]", "", sc.lower())
            words_curr = set(norm.split())
            is_dup = False
            for prev in seen:
                words_prev = set(prev.split())
                if len(words_curr) > 3 and len(words_curr & words_prev) / len(words_curr) > 0.65:
                    is_dup = True
                    break
            if not is_dup:
                seen.add(norm)
                cleaned.append(sc)
            if len(cleaned) >= max_s:
                break
        return " ".join(cleaned)

    # Clean Problem: max 1 sentence
    prob = data.get("Problem", "")
    prob = re.sub(r"[\[\]\(\)]", "", prob).strip()
    prob = re.sub(r"\s+-\s+|\s+-\b", " ", prob)
    prob = re.sub(r"(?<=[a-zA-Z'\"])\d+(?=[a-zA-Z\s.,'\"])", "", prob)
    prob = deduplicate_sentences(prob, max_s=1)

    # Clean Evidence: clean prefix and keep 1 complete sentence/quote
    evid = data.get("Evidence", "")
    evid = re.sub(r"[\[\]\(\)]", "", evid).strip()
    evid = re.sub(r"\s+-\s+|\s+-\b", " ", evid)
    evid = re.sub(r"^(?:Direct\s+quote(?:\s+from\s+.*?feedback)?:\s*)+", "", evid, flags=re.IGNORECASE).strip()
    evid = deduplicate_sentences(evid, max_s=1)
    evid = re.sub(r"(?<=[a-zA-Z'\"])\d+(?=[a-zA-Z\s.,'\"])", "", evid)

    # Clean Priority: authoritative determination via PM impact rubric
    if override_priority:
        prio = override_priority
    else:
        prio_raw = data.get("Priority", "").strip()
        prio_lower = prio_raw.lower()
        if re.search(r"\bhigh\b", prio_lower):
            prio = "High"
        elif re.search(r"\bmedium\b", prio_lower):
            prio = "Medium"
        elif re.search(r"\blow\b", prio_lower):
            prio = "Low"
        else:
            prio = "High"

    # Clean Why: max 1-2 sentences, deduplicated
    why = data.get("Why", "")
    why = re.sub(r"[\[\]\(\)]", "", why).strip()
    why = re.sub(r"\s+-\s+|\s+-\b", " ", why)
    why = re.sub(r"(?<=[a-zA-Z'\"])\d+(?=[a-zA-Z\s.,'\"])", "", why)
    why = deduplicate_sentences(why, max_s=2)

    # Clean Recommendation: max 1 sentence, strip prefix
    rec = data.get("Recommendation", "")
    rec = re.sub(r"[\[\]\(\)]", "", rec).strip()
    rec = re.sub(r"\s+-\s+|\s+-\b", " ", rec)
    rec = re.sub(r"^(?:Actionable\s+next\s+step(?:\s+for\s+.*?team)?:\s*)+", "", rec, flags=re.IGNORECASE).strip()
    rec = re.sub(r"(?<=[a-zA-Z'\"])\d+(?=[a-zA-Z\s.,'\"])", "", rec)
    rec = deduplicate_sentences(rec, max_s=1)

    output = f"""Problem:
{prob}

Evidence:
{evid}

Priority:
{prio}

Why:
{why}

Recommendation:
{rec}"""
    return output.strip()


answer = clean_pm_insight(raw_answer, override_priority=rubric_priority)

print(answer)


# ============================================================
# 15. CLEAN UP
# ============================================================

generation_model.unload()

print("\n" + "=" * 60)
print("CorePM RAG pipeline completed successfully!")
print("=" * 60)