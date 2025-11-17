import sentencepiece as spm
import re

# This is ChatGLM model
# https://huggingface.co/zai-org/chatglm2-6b/tree/main
if 1==1:
    sp = spm.SentencePieceProcessor(model_file='./model_chatglm/tokenizer.model')
    print(sp)

    vocabs = [sp.IdToPiece(id) for id in range(sp.GetPieceSize())]
    print(vocabs[800:1000])

    special_tokens = ["<s>", "</s>", "<unk>"]
    for st in special_tokens:
        print(sp.PieceToId(st))


    ids = sp.Encode("Hello this is a test.", out_type=str, enable_sampling=False, alpha=0.1, nbest_size=-1)
    print(ids)

    # 筛选并打印所有中文字符
    print("\n词汇表中的所有中文字符:")
    chinese_chars = []
    chinese_chars_long = []
    
    '''
    中文字符+标点正则表达式 (借助deepseek实现)
    \u4e00-\u9fff 中文基本区
    \u3000 全角空格 \u3001 顿号 \u3002 句号 \uff0c 全角逗号
    \uff1b 全角分号 \uff1a 全角冒号 \uff01 全角感叹号 \uff1f 全角问号
    \u2018\u2019 单引号 \u201c\u201d 双引号 \uff08\uff09 全角括号 \u3010\u3011 黑括号
    \u300a\u300b 书名号
    '''

    chinese_pattern = r'^[\u4e00-\u9fff\u3000\u3001\u3002\uff0c]+$' #\uff1b\uff1a\uff01\uff1f\u2018\u2019\u201c\u201d\uff08\uff09\u3010\u3011\u300a\u300b ]+$'

    # 使用Unicode范围匹配中文字符 (\u4e00-\u9fff是中文基本区)
    for id, token in enumerate(vocabs):
        try:
            # 将token解码为文本
            decoded_text = token #tokenizer.decode([token_id])
            
            # 检查解码后的文本是否只包含中文字符或逗号和句号的编码
            if re.match(chinese_pattern, decoded_text):
                chinese_chars.append((token, decoded_text))
                if len(decoded_text) >= 3:
                    chinese_chars_long.append((token, decoded_text))
                    print(f"Token: {token} -> '{decoded_text}' {len(decoded_text)}")
        except:
            continue



    # 将结果保存到文件
    with open('out_chatglm_chinese_chars.txt', 'w', encoding='utf-8') as f:
        for token, text in chinese_chars:
            f.write(f"{token}\t{text}\n")
    print("结果已保存到 out_chatglm_chinese_chars.txt")

    with open('out_chatglm_chinese_chars_long.txt', 'w', encoding='utf-8') as f:
        for token, text in chinese_chars_long:
            f.write(f"{token}\t{text}\n")
    print("结果已保存到 out_chatglm_chinese_chars_long.txt")
    

from transformers import AutoTokenizer

# This is Qwen3 model
# https://huggingface.co/Qwen/Qwen3-0.6B/tree/main
if 2==2:
    tokenizer = AutoTokenizer.from_pretrained('./model_qwen3')

    # 获取词汇表
    vocab = tokenizer.get_vocab()
    print("词汇表大小:", len(vocab))

    # 筛选并打印所有中文字符
    print("\n词汇表中的所有中文字符:")
    chinese_chars = []
    chinese_chars_long = []
    chinese_pattern = r'^[\u4e00-\u9fff\u3000\u3001\u3002\uff0c]+$' #\uff1b\uff1a\uff01\uff1f\u2018\u2019\u201c\u201d\uff08\uff09\u3010\u3011\u300a\u300b ]+$'

    # 使用Unicode范围匹配中文字符 (\u4e00-\u9fff是中文基本区)
    for token, token_id in sorted(vocab.items(), key=lambda x: x[1]):
        try:
            # 将token解码为文本
            decoded_text = tokenizer.decode([token_id])
            
            # 检查解码后的文本是否只包含中文字符或逗号和句号的编码
            if re.match(chinese_pattern, decoded_text):
                chinese_chars.append((token, decoded_text, token_id))
                if len(decoded_text) >= 3:
                    chinese_chars_long.append((token, decoded_text, token_id))
                    print(f"Token: {token} -> '{decoded_text}' -> ID: {token_id} {len(decoded_text)}")
        except:
            continue

    print(f"\n找到的中文字符总数: {len(chinese_chars)}")
    print(f"\n找到的>=4的中文字符总数: {len(chinese_chars_long)}")

    # 将结果保存到文件
    with open('out_qwen3_chinese_chars.txt', 'w', encoding='utf-8') as f:
        for token, text, token_id in chinese_chars:
            f.write(f"{token}\t{text}\t{token_id}\n")
    print("结果已保存到 out_qwen3_chinese_chars.txt")

    with open('out_qwen3_chinese_chars_long.txt', 'w', encoding='utf-8') as f:
        for token, text, token_id in chinese_chars_long:
            f.write(f"{token}\t{text}\t{token_id}\n")
    print("结果已保存到 out_qwen3_chinese_chars_long.txt")
    
    
    
    
