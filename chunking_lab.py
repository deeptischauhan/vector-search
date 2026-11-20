# Keel Vector Search — Chunking Lab
# Run in a Databricks notebook (Python). Pure-Python: no index or endpoint needed.
# Goal: SEE how chunk size + overlap change how a long document is split, so you
# understand the Section-2 exam tradeoff by doing it, not reading it.
#
# Source: Deepti's own Medium post "Six Stages of Deploying Enterprise AI" (she owns it).

SAMPLE_DOC = """
Deploying an analytical AI system is not like shipping software, and treating it like software is where most of these projects go wrong.

MIT's 2025 NANDA study, The GenAI Divide, put a number on the damage: around 95% of enterprise generative-AI pilots deliver no measurable business impact. The researchers traced the cause to weak integration and organisational fit, not model quality. The model is rarely what fails.

Software is correct when it behaves the way the spec says. An analytical system has no such luxury. There is rarely a crisp spec, and "correct" means something harder: the output has to be true, and it has to mean what the person reading it believes it means. A broken feature is a bug you fix next sprint. A number that is slightly wrong but trusted is a bad decision someone has already made.

Stage 1 - Discovery: agree the definitions before you build anything. Enterprise analytics is definitions, definitions, definitions - the model is the easy part. In any organisation of a reasonable size, different teams quietly define the same metric differently. Build on top of that disagreement and you ship something precisely, confidently wrong, which every stakeholder rejects for a different reason.

Stage 2 - Pilot: sequence by business impact, not coverage. Take the narrowest slice where the pain is greatest and the stakes are highest, and solve that first. It earns you credibility and keeps the surface area small enough that when a number looks off, you are checking one thing, not a hundred.

Stage 3 - Build: accuracy is the product, so make it checkable. There is no green test suite that certifies a metric is right; correctness is whether the number faithfully represents reality, which is a judgement. So the system has to make that judgement checkable by a human: cite the source behind a claim, show how a number was derived, and let someone verify rather than just trust.

Stage 4 - Rollout: put it where people already work. A system that depends on someone pulling numbers on request will not survive a busy organisation. Access is not usage. The job is to make the new way genuinely easier than the old one, not just available alongside it.

Stage 5 - Adoption: trust is the real launch. The signal is not the launch announcement. It is people changing their habits: leadership reading it as part of their routine, and the competing homemade versions disappearing because everyone now relies on the one trusted source. Measure adoption, not delivery.

Stage 6 - Automate: only once it is trusted, and never without alarms. Automate the repetitive, identical work and let it run on a schedule, but only under two conditions. The process must already be correct, because automating an untrusted one just means being wrong faster and at scale. And you must wire in failure alerts, so you hear about a break before the stakeholder does.

The thread through all six: your job is oversight, not execution. The tools keep getting better at producing the work. What they still cannot do is decide what good means for your business, or catch the quiet mistake inside a confident answer. That judgement is the durable skill.
""".strip()


# --- A simple word-based chunker with overlap ---
# (Embedding models limit by *tokens*; tokens are roughly 0.75 words, so a
#  512-token model limit is about ~380 words. We chunk by words here to keep it
#  intuitive; the size/overlap tradeoff is identical.)
def chunk_text(text, chunk_size, overlap):
    words = text.split()
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + chunk_size]
        if piece:
            chunks.append(" ".join(piece))
        if start + chunk_size >= len(words):
            break
    return chunks


total_words = len(SAMPLE_DOC.split())
print(f"Document length: {total_words} words\n")

# --- Experiment: same document, three chunking settings ---
settings = [
    ("Small chunks, no overlap", 40, 0),
    ("Medium chunks, some overlap", 90, 15),
    ("Large chunks, no overlap", 200, 0),
]

for label, size, overlap in settings:
    chunks = chunk_text(SAMPLE_DOC, size, overlap)
    print(f"=== {label}  (size={size} words, overlap={overlap}) ===")
    print(f"  -> {len(chunks)} chunks")
    print(f"  first chunk: {chunks[0][:160]}...")
    print()

# ---------------------------------------------------------------------------
# WHAT TO OBSERVE (the exam tradeoff):
#
#  - Smaller chunks  -> MORE chunks, each a tight single idea. Retrieval is
#    precise (you get the exact passage), but a chunk may lack surrounding
#    context to be useful on its own.
#  - Larger chunks   -> FEWER chunks, more context per chunk, but a search can
#    return a big blob where only one sentence was relevant (less precise), and
#    you risk exceeding the embedding model's token limit.
#  - Overlap         -> repeats a few words across neighbouring chunks so an idea
#    split across a boundary isn't lost. Costs a little duplication for better recall.
#
#  Rule of thumb: chunk on natural boundaries (paragraphs/sections) near the
#  model's token limit, with ~10-15% overlap. Then HAND-READ retrieval quality
#  and adjust - same discipline as your eval hand-reads.
# ---------------------------------------------------------------------------


# ===========================================================================
# OPTIONAL - the "real Databricks" version (write chunks to a table).
# Note: Free Edition allows ONE AI Search index on the single search unit, so
# to build a chunked index you'd DELETE the existing articles_index first, then
# create a new index over the chunks table below. Skip unless you want the full
# end-to-end; the learning above is the point.
# ===========================================================================
#
# from pyspark.sql import Row
# rows = []
# for i, ch in enumerate(chunk_text(SAMPLE_DOC, 90, 15)):
#     rows.append(Row(chunk_id=f"sixstages_{i:03d}",
#                     article_id="a08",
#                     chunk_index=i,
#                     category="business_acumen",
#                     chunk_text=ch))
# df = spark.createDataFrame(rows)
# df.write.mode("overwrite").option("delta.enableChangeDataFeed", "true") \
#     .saveAsTable("workspace.cpt.article_chunks")
# # then: Catalog -> article_chunks -> Create vector search index
# #   primary key = chunk_id, embedding source column = chunk_text, model = databricks-gte-large-en
