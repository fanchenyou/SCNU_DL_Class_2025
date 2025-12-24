import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 设置
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 加载tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained("/data/LLM_MODEL/Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained(
    "/data/LLM_MODEL/Qwen/Qwen3-0.6B",
    device_map="cuda:0")

# 原始句子
sentences: list[str] = ["Talk is cheap.", "白日依山尽。", "我们都在华师。"]
print("=== Qwen模型Token级别Logits生成 ===")


# -------------------------- 1. 循环方式生成每个单词的Logits --------------------------
print("\n\n=== 1. 循环方式生成Token Logits ===")

all_token_logits_loop = []
all_tokens_loop = []

for i, sentence in enumerate(sentences):
    print(f"\n句子 {i+1}: {sentence}")
    
    # 编码句子，获取token列表
    tokens = tokenizer.encode(sentence, add_special_tokens=False)
    tokens_str = tokenizer.convert_ids_to_tokens(tokens)
    all_tokens_loop.append(tokens_str)
    
    # 转换为张量
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    
    # 前向传播获取logits
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, return_dict=True)
        logits = outputs.logits[0]  # 形状: (1, seq_len, vocab_size)
   

    token_logits = []
    for i in range(len(tokens) - 1):
        next_token_logits = logits[i]  # 当前token预测下一个token的logits
        correct_next_token = tokens[i + 1]
        correct_logit = next_token_logits[correct_next_token].item()
        
        token_str = tokenizer.decode([tokens[i]])
        next_token_str = tokenizer.decode([tokens[i + 1]])
        print(f"  '{token_str}' -> '{next_token_str}': {correct_logit:.4f}")
        
        token_logits.append(next_token_logits)
    
    all_token_logits_loop.append(token_logits)


# -------------------------- 2. Packing Sequence方式生成每个单词的Logits --------------------------
print("\n\n=== 2. Packing Sequence方式生成Token Logits ===")

# 1. 分别编码每个句子
tokenized_sentences = []
sentence_lengths = []
all_tokens_packed = []

for sentence in sentences:
    tokens = tokenizer.encode(sentence, add_special_tokens=False)
    tokenized_sentences.append(tokens)
    sentence_lengths.append(len(tokens))
    all_tokens_packed.append(tokenizer.convert_ids_to_tokens(tokens))

# 2. 合并所有句子为一个长序列
merged_input_ids = []
for tokens in tokenized_sentences:
    merged_input_ids.extend(tokens)

# 转换为张量
merged_input_ids = torch.tensor([merged_input_ids], dtype=torch.long, device=device)
total_length = merged_input_ids.shape[1]

# 3. 创建 Block Diagonal Attention Mask
# 形状: (1, 1, total_length, total_length)
# 初始化全0矩阵
# attention_mask = torch.zeros(1, 1, total_length, total_length, device=device)
attention_mask = torch.full(
    (1, 1, total_length, total_length), 
    float('-inf'), 
    device=device
)


# 计算句子边界
start_positions = [0]
for length in sentence_lengths[:-1]:
    start_positions.append(start_positions[-1] + length)

# 设置每个句子的注意力掩码为下三角矩阵（只有句子内部可以互相看到）
for i, start in enumerate(start_positions):
    end = start + sentence_lengths[i]
    # 对于每个句子，创建下三角矩阵并设置到对应的区域
    # 上三角区域(不包括对角线)设为-inf，其余为0
    attention_mask[0, 0, start:end, start:end] = torch.triu(
        torch.full((sentence_lengths[i], sentence_lengths[i]), float('-inf'), device=device), 
        diagonal=1
    )
    
    
# 4. 创建正确的Position IDs（每个句子从0开始重新计数）
merged_position_ids = []
for length in sentence_lengths:
    # 每个句子的position_ids从0开始到length-1
    merged_position_ids.extend(list(range(length)))

# 转换为张量
merged_position_ids = torch.tensor([merged_position_ids], dtype=torch.long, device=device)

# print(f"\n注意力掩码和位置编码信息:")
# print(f"  合并后序列长度: {total_length}")
# print(f"  句子长度: {sentence_lengths}")
# print(f"  句子起始位置: {start_positions}")
# print(f"  位置编码: {merged_position_ids}")

# # 可视化注意力掩码的对角线结构
# print("\n注意力掩码对角线结构:")
# for i, (start, length) in enumerate(zip(start_positions, sentence_lengths)):
#     end = start + length
#     print(f"  句子 {i+1}: [{start}:{end}] -> 对角线子矩阵大小: {length}x{length}")

# 5. 一次性前向传播获取所有logits（使用正确的position_ids）
model.eval()
with torch.no_grad():
    outputs = model(
        input_ids=merged_input_ids,
        attention_mask=attention_mask,  # 使用正确的block diagonal掩码
        position_ids=merged_position_ids,  # 使用重新索引的位置编码
        return_dict=True
    )
    merged_logits = outputs.logits  # 形状: (1, total_length, vocab_size)


# 6. 提取每个句子每个token的logits
all_token_logits_packed = []

for i, (start, length) in enumerate(zip(start_positions, sentence_lengths)):
    end = start + length
    print(f"\n句子 {i+1}: {sentences[i]}")
    
    # 获取当前句子的tokens和logits
    sentence_tokens = all_tokens_packed[i]
    sentence_logits = merged_logits[:, start:end, :]
    
    # 记录每个token的logits
    token_logits = []
    for token_idx, token_str in enumerate(sentence_tokens):
        if token_idx < sentence_logits.shape[1] - 1:
            # 获取当前token对应的下一个token的logits
            next_token_logits = sentence_logits[0, token_idx, :]
            token_logits.append(next_token_logits)
            
            # 计算当前token对应的正确下一个token的logits值
            correct_next_token = tokenized_sentences[i][token_idx + 1]
            correct_logit = next_token_logits[correct_next_token].item()
            
            # 使用tokenizer.decode获取可读文本，格式与循环方式一致
            token_str = tokenizer.decode([tokenized_sentences[i][token_idx]])
            next_token_str = tokenizer.decode([tokenized_sentences[i][token_idx + 1]])
            print(f"  '{token_str}' -> '{next_token_str}': {correct_logit:.4f}")
    
    all_token_logits_packed.append(token_logits)


# -------------------------- 3. 验证两种方式结果一致性 --------------------------
print("\n\n=== 3. 验证两种方式结果一致性 ===")


print("\n\n=== 4. 查看mask ===")

print(attention_mask)
