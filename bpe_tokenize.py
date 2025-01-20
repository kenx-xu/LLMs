import numpy as np

import config


def load_vocab():
    """加载词典"""
    vocab={}
    with open(config.VOCAB_ADDR,'r',encoding='utf-8') as vocab_file:
        for line in vocab_file:
            token ,idx = line.strip().split(':')
            vocab[token] = int(idx)
    return vocab

# 分词器未使用txt
def bpe_tokenize(text ,vocab):
    """分词器 """
    tokens=[]
    for word in text.split():
        word=' '.join(list(word))+'</w>'
        subwords=[]
        while len(word)>0:
            matches=False
            for token in sorted(vocab.keys(), key=len, reverse=True):
                if word.startswith(token):
                    subwords.append(word)
                    word=word[len(token):].strip()
                    matches=True
                    break
            if not matches:
                subwords.append('</uk>')
                break
        tokens.extend(subwords)
    return tokens

# 使用txt后的分词器
def bpe_tokenize_by_txt(text,vocab):
    tokens=[]
    while text:
        longest_match=None
        for token in vocab:
            if text.startswith(token):
                if not longest_match or len(token)>len(longest_match):
                    longest_match=token
            if longest_match:
                tokens.append(longest_match)
                text=text[len(longest_match):]
            else:
                tokens.append(token[0])
                text=text[1:]
    return tokens



# 转化为索引数列表
def to_ind(texts,max_len,vocab,padding='post'):
    """转化为索引然后padding"""
    # 加载txt
    merge_list=load_merges()
    texts=[apply_merge(text,merge_list) for text in texts]
    tokenize_seq=[bpe_tokenize_by_txt(text,vocab) for text in texts]
    ind_seq=[[vocab.get(token,vocab.get('</uk>')) for token in tokens] for tokens in tokenize_seq]
    padded_seq=[]
    for seq in ind_seq:
        if len(seq) < max_len:
            if padding=='post':
                seq = seq+[vocab['<pad>']]*(max_len-len(seq))
            elif padding == 'pre':
                seq = [vocab['<pad>']]*(max_len-len(seq))+seq
        else:
            seq=seq[:max_len]
        padded_seq.append(seq)
    return np.array(padded_seq)

def load_merges():
    merge_list=[]
    with open(config.MERGE_ADDR, 'r', encoding='utf-8') as merge_file:
        for line in merge_file:
            merge_list.append(tuple(line.strip().split()))
    return merge_list

def apply_merge(text,merge_list):
    for pair in merge_list:
        merge_token=' '.join(pair)
        replace_token=''.join(pair)
        text=text.replace(replace_token,merge_token)
    return text
