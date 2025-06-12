from transformers import AutoTokenizer
import regex as re
import json
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from typing import Optional

# 模型路径
model_name = '/user/gaoziheng/nlp/distil/Qwen2.5-0.5B-distil'

# Tokenizer 初始化
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

SYSTEM_PROMPT = '''你是一个中文人工智能助手。请根据要求，完成下面的数学竞赛题目。 以下是中国数学竞赛中的题目，请根据题目的要求和所提供的信息计算得出答案。请在最后以“所以最终答案是\\boxed{答案}。”显式给出结果。过程解答过程和结果中使用的变量和公式请使用LaTeX格式表示。'''

# 提取 boxed 答案（\boxed{...}）
def extract_boxed_answer(text: str) -> Optional[str]:
    # 支持嵌套 { } 的正则表达式
    pattern = r"\\boxed$\s*([^$]+?(?:$(?1)*?$)?[^$]*?)$\s*"
    match = re.search(pattern, text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

# 正确性奖励函数
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_boxed_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}", f"\nAnswer:\n{answer[0]}")
    return [1.0 if r == a.strip('$') else 0.0 for r, a in zip(extracted_responses, answer)]

# soft 格式奖励（是否存在 \boxed）
def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"\\boxed\{.*?\}"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    return [2 if match else 0.0 for match in matches]

# strict 格式奖励（是否包含完整“所以最终答案是 \boxed{...}。”）
def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"所以最终答案是\s*\\boxed\{.*?\}。"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    return [4 if match else 0.0 for match in matches]

# 加载 OlympiadBench 数据集
def get_olympiadbench_dataset(path: str = '/home/gaoziheng/nlp/data/OlympiadBench_Dataset/data/combined.json') -> Dataset:
    with open(path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    dataset = Dataset.from_list([
        {
            'prompt': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': item['question']}
            ],
            'answer': item['final_answer'][0]
        }
        for item in raw_data
    ])
    return dataset

# 读取数据集
olympiad_dataset = get_olympiadbench_dataset()

# 打印样本检查
#print(olympiad_dataset[0])

# GRPO 配置
training_args = GRPOConfig(
    use_vllm=False,
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_generations=2,
    max_prompt_length=512,
    max_completion_length=2048,
    num_train_epochs=1,
    save_steps=100,
    max_grad_norm=0.1,
    vllm_gpu_memory_utilization=0.2,
    report_to="tensorboard",
    output_dir="/user/gaoziheng/nlp/grpo/logs",
)

# 启动训练器
trainer = GRPOTrainer(
    model=model_name,
    processing_class=tokenizer,
    reward_funcs=[
        soft_format_reward_func,
        strict_format_reward_func,
        correctness_reward_func,
    ],
    args=training_args,
    train_dataset=olympiad_dataset,
)

# 训练
trainer.train()
trainer.save_model("/user/gaoziheng/nlp/grpo/Qwen2.5-0.5B-grpo")

os.makedirs("/user/gaoziheng/nlp/grpo/logs", exist_ok=True)
loss_kl_log_path = "/user/gaoziheng/nlp/grpo/logs/loss_kl_history.json"

# 提取 log_history 中的 loss 和 kl_divergence
loss_kl_history = []
for log in trainer.state.log_history:
    if "loss" in log and "kl" in log:
        loss_kl_history.append({
            "step": log["step"],
            "loss": log["loss"],
            "kl": log["kl"]
        })

# 保存为 JSON 文件
with open(loss_kl_log_path, "w", encoding="utf-8") as f:
    json.dump(loss_kl_history, f, indent=2, ensure_ascii=False)

print(f"Loss and KL divergence saved to {loss_kl_log_path}")