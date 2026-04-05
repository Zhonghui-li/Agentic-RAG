import os
import re
import json
import argparse
from dotenv import load_dotenv
load_dotenv()
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

def split_text_by_sentence(text, chunk_size=300, chunk_overlap=50):
    """基于句子边界进行文本切分"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["。", "？", "！", "\n", ".", "?", "!"]
    )
    return text_splitter.split_text(text)

def split_by_paragraph(context, max_chunk_size=800):
    """按段落切分 HotpotQA context，保持段落完整性。

    每个段落格式为 "Title: sentence1 sentence2..."，以 \\n\\n 分隔。
    仅对超过 max_chunk_size 的段落做二次切分。

    Returns:
        list[dict]: 每个元素包含 "text" 和 "paragraph_title"。
    """
    paragraphs = context.split("\n\n")
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=100,
        separators=[". ", "? ", "! ", "\n", " "],
    )

    results = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Extract title from "Title: content" format
        title_match = re.match(r"^(.+?):\s", para)
        paragraph_title = title_match.group(1).strip() if title_match else ""

        if len(para) <= max_chunk_size:
            results.append({"text": para, "paragraph_title": paragraph_title})
        else:
            sub_chunks = fallback_splitter.split_text(para)
            for chunk in sub_chunks:
                results.append({"text": chunk, "paragraph_title": paragraph_title})

    return results

def load_hotpot_mini(json_path="data-hotpot/hotpot_mini_corpus.json"):
    """加载Hotpot mini corpus (json)"""
    with open(json_path, 'r') as f:
        corpus = json.load(f)
    return corpus

def create_vectorstore(json_path="data-hotpot/hotpot_mini_corpus.json",
                       persist_path="vectorstore-hotpot/hotpotqa_faiss",
                       chunk_strategy="paragraph"):
    """从Hotpot mini corpus创建FAISS向量库"""

    # 1. 加载小型corpus
    print("Loading mini corpus...")
    corpus = load_hotpot_mini(json_path)
    print(f"Loaded {len(corpus)} items.")

    # 2. 做chunking
    print(f"Splitting into chunks (strategy={chunk_strategy})...")
    docs = []
    for item in corpus:
        if "context" not in item:
            print(f"Warning: Missing 'context' in item with question: {item.get('question', 'Unknown')}")
            continue

        context = item['context']
        question = item.get('question', '')
        answer = item.get('answer', '')

        if chunk_strategy == "paragraph":
            chunks = split_by_paragraph(context)
            for chunk in chunks:
                if chunk["text"].strip():
                    docs.append(Document(
                        page_content=chunk["text"],
                        metadata={
                            "question": question,
                            "answer": answer,
                            "title": chunk["paragraph_title"],
                        }
                    ))
        else:  # fixed
            chunks = split_text_by_sentence(context)
            for chunk in chunks:
                if chunk.strip():
                    docs.append(Document(
                        page_content=chunk,
                        metadata={
                            "question": question,
                            "answer": answer,
                            "title": chunk.split(':', 1)[0] if ':' in chunk else ""
                        }
                    ))

    print(f"Prepared {len(docs)} chunks from {len(corpus)} items.")

    # 打印一些示例chunks以便验证
    if docs:
        print("\nSample chunks:")
        for i in range(min(3, len(docs))):
            print(f"Chunk {i+1}:")
            print(f"  Content: {docs[i].page_content[:100]}...")
            print(f"  Metadata: {docs[i].metadata}")
        print()

    # 3. 建立Embedding和Vectorstore
    print("Embedding documents...")
    emb_model = os.getenv("EMB_MODEL", "text-embedding-3-large")
    embeddings = OpenAIEmbeddings(model=emb_model)

    vectorstore = FAISS.from_documents(docs, embeddings)

    # 4. 保存
    print(f"Saving FAISS index to {persist_path}...")
    os.makedirs(persist_path, exist_ok=True)
    vectorstore.save_local(persist_path)

    print("Vectorstore created successfully.")
    print(f"  - Contains {len(docs)} vectors")
    print(f"  - Saved to: {persist_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS vectorstore from HotpotQA corpus")
    parser.add_argument("--chunk_strategy", choices=["paragraph", "fixed"], default="paragraph",
                        help="Chunking strategy: 'paragraph' (split by \\n\\n) or 'fixed' (300-char RecursiveCharacterTextSplitter)")
    parser.add_argument("--output_path", default="vectorstore-hotpot/hotpotqa_faiss",
                        help="Output path for the FAISS index")
    parser.add_argument("--json_path", default="data-hotpot/hotpot_mini_corpus.json",
                        help="Path to the HotpotQA corpus JSON")
    args = parser.parse_args()

    create_vectorstore(
        json_path=args.json_path,
        persist_path=args.output_path,
        chunk_strategy=args.chunk_strategy,
    )
