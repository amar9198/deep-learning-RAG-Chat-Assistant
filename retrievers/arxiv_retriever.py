import requests
import feedparser

from langchain_core.documents import Document


class ArxivRetriever:

    def __init__(self, load_max_docs=2):
        self.load_max_docs = load_max_docs

    def invoke(self, query):

        api_url = "https://export.arxiv.org/api/query"

        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": self.load_max_docs,
        }

        headers = {
            "User-Agent": "RAG-Project/1.0"
        }

        response = requests.get(
            api_url,
            params=params,
            headers=headers,
            timeout=30
        )

        response.raise_for_status()

        feed = feedparser.parse(response.content)

        documents = []

        for entry in feed.entries:

            title = entry.get("title", "").strip()

            authors = ", ".join(
                author.get("name", "")
                for author in entry.get("authors", [])
            )

            summary = entry.get("summary", "").strip()

            published = entry.get("published", "")

            paper_url = entry.get("link", "")

            content = f"""
Title: {title}

Authors: {authors}

Published: {published}

Summary:
{summary}
"""

            documents.append(
                Document(
                    page_content=content.strip(),
                    metadata={
                        "Title": title,
                        "Authors": authors,
                        "Published": published,
                        "URL": paper_url,
                        "source": "arXiv",
                    }
                )
            )

        return documents


# Create retriever
retriever = ArxivRetriever(
    load_max_docs=2
)

# Search arXiv
docs = retriever.invoke("large language models")


# Print results
for i, doc in enumerate(docs, start=1):

    print("\n" + "=" * 80)
    print(f"Result {i}")

    print("\nTitle:")
    print(doc.metadata.get("Title"))

    print("\nAuthors:")
    print(doc.metadata.get("Authors"))

    print("\nSummary:")
    print(doc.page_content[:500])

    print("\nURL:")
    print(doc.metadata.get("URL"))

    print("=" * 80)
