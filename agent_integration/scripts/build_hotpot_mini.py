import os
import json
from datasets import load_dataset
import pprint

def build_hotpot_mini(
    output_path="data-hotpot/hotpot_mini_corpus.json",
    num_questions=1000
):
    """从 HotpotQA 'fullwiki' 构建小规模 'context + question + answer' mini corpus"""

    # 1. 加载 HotpotQA fullwiki (只取 1% 节省时间)
    print("🚀 Loading small subset of HotpotQA...")
    dataset = load_dataset('hotpot_qa', 'fullwiki', split='train[:1%]')
    print(f"✅ Loaded {len(dataset)} examples.")

    # 调试：检查数据集结构
    print("\n🔍 Examining dataset structure...")
    example = dataset[0]
    print("Available keys:", list(example.keys()))

    # 打印示例条目的完整结构以便理解
    print("\nSample data item structure:")
    for key in example.keys():
        print(f"{key}: {type(example[key])}")
        if key == 'context':
            print(f"  - Context type: {type(example[key])}")
            if isinstance(example[key], list) and len(example[key]) > 0:
                print(f"  - First context item: {example[key][0]}")
        elif key == 'supporting_facts':
            print(f"  - Supporting facts type: {type(example[key])}")
            if isinstance(example[key], dict):
                print(f"  - Supporting facts keys: {example[key].keys()}")
                for sf_key in example[key].keys():
                    print(f"    - {sf_key}: {type(example[key][sf_key])}")
                    if hasattr(example[key][sf_key], '__len__') and len(example[key][sf_key]) > 0:
                        print(f"      - First item: {example[key][sf_key][0]}")
            elif isinstance(example[key], list) and len(example[key]) > 0:
                print(f"  - First supporting fact: {example[key][0]}")

    # 2. 打乱并取前 num_questions 条
    dataset = dataset.shuffle(seed=42).select(range(min(len(dataset), num_questions)))

    processed_examples = []

    print("\n🔍 Processing examples...")
    for i, item in enumerate(dataset):
        if i % 20 == 0:
            print(f"Processing example {i}/{len(dataset)}...")

        question = item['question']
        answer = item['answer']

        context_texts = []

        try:
            # 根据数据卡结构，context可能有不同的格式
            if 'context' in item:
                contexts = item['context']

                # 调试第一个条目的结构
                if i == 0:
                    print("\nFirst item context structure:")
                    print(f"Type: {type(contexts)}")
                    if isinstance(contexts, list) and len(contexts) > 0:
                        print(f"First element: {contexts[0]}")
                    elif isinstance(contexts, dict):
                        print(f"Context keys: {contexts.keys()}")
                        for context_key in contexts.keys():
                            if hasattr(contexts[context_key], '__len__') and len(contexts[context_key]) > 0:
                                print(f"First {context_key}: {contexts[context_key][0]}")

                # 处理情况1: context是列表，每个元素是[title, sentences]
                if isinstance(contexts, list):
                    for context_item in contexts:
                        if isinstance(context_item, list) and len(context_item) == 2:
                            title, sentences = context_item
                            if isinstance(sentences, list):
                                paragraph = f"{title}: " + " ".join([s if isinstance(s, str) else s[0] if isinstance(s, list) and len(s) > 0 else "" for s in sentences])
                                context_texts.append(paragraph)

                # 处理情况2: context是字典，有'sentences'和'title'字段
                elif isinstance(contexts, dict) and 'sentences' in contexts and 'title' in contexts:
                    titles = contexts['title']
                    all_sentences = contexts['sentences']

                    # 确保标题和句子列表长度匹配
                    if len(titles) == len(all_sentences):
                        for idx, (title, sentences) in enumerate(zip(titles, all_sentences)):
                            if sentences:
                                # 处理sentences可能是嵌套列表的情况
                                sentence_texts = []
                                for sent in sentences:
                                    if isinstance(sent, str):
                                        sentence_texts.append(sent)
                                    elif isinstance(sent, list) and len(sent) > 0:
                                        sentence_texts.append(sent[0])

                                paragraph = f"{title}: " + " ".join(sentence_texts)
                                context_texts.append(paragraph)

            # 如果context处理后为空，尝试使用supporting_facts
            if not context_texts and 'supporting_facts' in item:
                supporting_facts = item['supporting_facts']

                # 调试supporting_facts结构
                if i == 0:
                    print("\nSupporting facts structure:")
                    print(f"Type: {type(supporting_facts)}")
                    if isinstance(supporting_facts, dict):
                        print(f"Keys: {supporting_facts.keys()}")
                        for sf_key in supporting_facts.keys():
                            if hasattr(supporting_facts[sf_key], '__len__') and len(supporting_facts[sf_key]) > 0:
                                print(f"First {sf_key}: {supporting_facts[sf_key][0]}")
                    elif isinstance(supporting_facts, list) and len(supporting_facts) > 0:
                        print(f"First item: {supporting_facts[0]}")

                # 处理supporting_facts为字典的情况
                if isinstance(supporting_facts, dict) and 'title' in supporting_facts:
                    titles = supporting_facts['title']
                    # 尝试从context中找到对应的内容
                    if isinstance(contexts, dict) and 'sentences' in contexts:
                        all_sentences = contexts['sentences']
                        for title in titles:
                            # 在context的title中找到对应项
                            if title in contexts['title']:
                                idx = contexts['title'].index(title)
                                if idx < len(all_sentences) and all_sentences[idx]:
                                    paragraph = f"{title}: " + " ".join(all_sentences[idx])
                                    context_texts.append(paragraph)

                # 处理supporting_facts为列表的情况
                elif isinstance(supporting_facts, list):
                    for fact in supporting_facts:
                        if isinstance(fact, list) and len(fact) >= 1:
                            title = fact[0]
                            # 尝试在contexts中找到对应内容
                            for context_item in contexts if isinstance(contexts, list) else []:
                                if isinstance(context_item, list) and len(context_item) == 2 and context_item[0] == title:
                                    sentences = context_item[1]
                                    if sentences:
                                        paragraph = f"{title}: " + " ".join(sentences)
                                        context_texts.append(paragraph)
                                        break

        except Exception as e:
            print(f"Error processing item {i}: {e}")
            if i < 3:  # 只为前几个条目提供更详细的错误信息
                print(f"Question: {question}")
                try:
                    print(f"Context type: {type(item.get('context', 'No context'))}")
                    print(f"Supporting facts type: {type(item.get('supporting_facts', 'No supporting_facts'))}")
                except:
                    pass
            continue

        full_context = "\n\n".join(context_texts)

        if not full_context.strip():
            print(f"⚠️ Skipping item with empty context: {question}")
            # 为空context的条目提供更详细的调试信息
            if i < 3:
                try:
                    if isinstance(item.get('context', {}), dict):
                        print(f"  - Context keys: {list(item['context'].keys())}")
                    else:
                        print(f"  - Context type: {type(item.get('context', 'Not found'))}")

                    if isinstance(item.get('supporting_facts', {}), dict):
                        print(f"  - Supporting facts keys: {list(item['supporting_facts'].keys())}")
                    else:
                        print(f"  - Supporting facts type: {type(item.get('supporting_facts', 'Not found'))}")
                except Exception as e:
                    print(f"  - Error printing debug info: {e}")
            continue

        # 添加成功处理的示例
        processed_examples.append({
            "question": question,
            "answer": answer,
            "context": full_context
        })

        # 打印第一个成功处理的示例
        if len(processed_examples) == 1:
            print("\n✅ First successful example:")
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print(f"Context (first 150 chars): {full_context[:150]}...")

    # 3. 保存为 JSON
    if processed_examples:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(processed_examples, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Saved {len(processed_examples)} examples to {output_path}")

        # 打印统计信息
        print("\n📊 Statistics:")
        print(f"- Total processed examples: {len(processed_examples)}")
        avg_context_len = sum(len(ex['context']) for ex in processed_examples) / len(processed_examples)
        print(f"- Average context length: {avg_context_len:.1f} characters")
        print(f"- Sample questions: {processed_examples[0]['question'][:50]}...")
    else:
        print("\n⚠️ WARNING: No examples were processed! The output file is empty.")
        print("Please check the HotpotQA dataset structure and update the processing logic.")

if __name__ == "__main__":
    build_hotpot_mini()