from loaders.loaders_factory import DocumentLoader
from processing import analyzer
from processing.text_cleaner import TextCleaner
from processing.headings import HeadingSplitter
from processing.splitters import splitter

from vectorstore.faiss_store import vector_store
from retriever.retriever import RetrieverService
from processing.analyzer import DocumentAnalyzer
from processing.semantic_chunks import SemanticChunker
from processing.validator import ChunkValidator
from chains.rag_chain import build_rag_chain
from llm.huggingface_llm import llm

docs = DocumentLoader.load("uploads/sample.pdf")

docs = TextCleaner.clean(docs)

analyzed = analyzer.DocumentAnalyzer().analyze(docs)

semantic_docs = SemanticChunker.build(analyzed)

chunks = splitter.split(semantic_docs)

ChunkValidator.validate(chunks)

chunks = splitter.split(docs)


db = vector_store.create(chunks)


retriever = RetrieverService(db).get_retriever(

    search_type="mmr",

    k=4,

    fetch_k=20

)


rag = build_rag_chain(
    retriever,
    llm
)

# ==========================================================
# TEST CASES
# ==========================================================

test_cases = [

    {
        "question": "What is Organizational Structure?",

        "expected_keywords": [
            "hierarchy",
            "roles",
            "responsibilities",
            "authority"
        ]
    },


    {
        "question": "Why do tech companies need organizational structure?",

        "expected_keywords": [
            "communication",
            "decision-making",
            "growth",
            "innovation"
        ]
    },


    {
        "question": "What are advantages of Functional Structure?",

        "expected_keywords": [
            "efficiency",
            "expertise",
            "management"
        ]
    },


    {
        "question": "What are disadvantages of Functional Structure?",

        "expected_keywords": [
            "silos",
            "communication",
            "innovation"
        ]
    },


    {
        "question": "What is the salary of a software engineer?",

        "expected_keywords": [
            "not_found"
        ]
    }

]



# ==========================================================
# EVALUATION
# ==========================================================


total = len(test_cases)

passed = 0



for index, test in enumerate(test_cases, start=1):


    print("\n")
    print("=" * 90)

    print(f"TEST CASE {index}")

    print("=" * 90)


    question = test["question"]


    print("\nQuestion:")
    print(question)


    results = retriever.invoke(question)
    



    retrieved_text = " ".join(

        [
            doc.page_content.lower()

            for doc in results
        ]

    )



    print("\nRetrieved Chunks:")

    for i, doc in enumerate(results, start=1):

        print("-" * 50)

        print(f"Chunk {i}")

        print(doc.metadata)

        print(doc.page_content[:300])



    print("\nEvaluation:")



    if "not_found" in test["expected_keywords"]:


        # For unknown questions,
        # we expect weak/no matching retrieval

        if len(results) == 0:

            print("[PASS] No documents retrieved")

            passed += 1

        else:

            print(
                "⚠ REVIEW - Documents retrieved"
            )



    else:


        matched = []


        for keyword in test["expected_keywords"]:

            if keyword.lower() in retrieved_text:

                matched.append(keyword)



        score = (

            len(matched)

            /

            len(test["expected_keywords"])

        )



        print(
            f"Keyword Match: {score*100:.2f}%"
        )


        print(
            "Matched:",
            matched
        )


        if score >= 0.5:

            print("[PASS]")

            passed += 1

        else:

            print("[FAIL]")



print("\n")

print("=" * 90)

print("FINAL RESULT")

print("=" * 90)


print(
    f"Passed {passed}/{total} tests"
)


print(
    f"Accuracy: {(passed/total)*100:.2f}%"
)
