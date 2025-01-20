import json
import random
from collections import defaultdict

import config

SPECIAL_TOKENS = ["</uk>", "<pad>", "<s>", "</s>", "<cls>", "<sep>"]
STOP_WORDS = {"the", "is", "in","and","a", "的", "了", "在", "和", "是"}

def process_stop(texts):
    return [' '.join(word for word in text.split() if word not in STOP_WORDS) for text in texts ]

def get_vocab(texts):
    """将单词分解为字符列表并构建初始词汇表"""
    vocab = defaultdict(int)
    for text in texts:
        for word in text.split():
            word = " ".join(list(word.lower())) + " </w>"  # 添加结尾标记
            vocab[word] += 1
    return vocab


def get_stats(vocab):
    """统计相邻字符对频率"""
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def merge_vocab(pair, vocab):
    """合并最高频的字符对"""
    new_vocab = {}
    bigram = " ".join(pair)
    replacement = "".join(pair)
    for word in vocab:
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = vocab[word]
    return new_vocab

def save_merge_list(merge_list):
    with open(config.MERGE_ADDR,'w',encoding='utf-8') as merge_file:
        for pair in merge_list:
            merge_file.write(" ".join(pair) + "\n")

def bpe(texts, num_merges):
    """BPE 主函数"""
    texts = process_stop(texts)
    vocab = get_vocab(texts)
    merge_list=[]
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            return
        best_pair = max(pairs, key=pairs.get)  # 找到最高频的字符对
        vocab = merge_vocab(best_pair, vocab)
        merge_list.append(best_pair)
        print(f"Step {i + 1}: Merge {best_pair}")
    save_merge_list(merge_list)

    # 构建最终词汇表
    final_vocab = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
    idx = len(final_vocab)
    for word in vocab:
        for token in word.split():
            if token not in final_vocab:
                final_vocab[token] = idx
                idx += 1
    return final_vocab



# 读取输入文本
texts = []

with open(config.CORPUS_ADDR,'r',encoding='utf-8') as corpus_file:
    for line in corpus_file:
        doc = json.loads(line.strip())
        texts.append(doc['text'])
    corpus_file.close()

texts=random.sample(texts,10000)
# 运行 BPE
num_merges = config.MERGE_NUM  # 合并次数
final_vocab = bpe(texts, num_merges)

with open(config.VOCAB_ADDR,'w+',encoding='utf-8') as vocab_file:
    for token, idx in final_vocab.items():
        vocab_file.write(f"{token}:{idx}\n")
    vocab_file.close()
print('finish')
