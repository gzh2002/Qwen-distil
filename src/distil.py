import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import Dataset
from textbrewer import GeneralDistiller, TrainingConfig, DistillationConfig
import logging

# 日志设置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载本地 JSON 数据集
dataset_path = "/home/gaoziheng/nlp/data/train.json"
logger.info(f"Loading dataset from: {dataset_path}")
with open(dataset_path, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

SYSTEM_PROMPT = '''你是一个中文人工智能助手。请根据要求，完成下面的数学竞赛题目。 以下是中国数学竞赛中的题目，请根据题目的要求和所提供的信息计算得出答案。请在最后以“所以最终答案是\\boxed{答案}。”显式给出结果。过程解答过程和结果中使用的变量和公式请使用LaTeX格式表示。'''

# 构造 Dataset 对象（只保留 question 和 solution）
def extract_qa(example):
    return {
        "text": SYSTEM_PROMPT + example["question"] + "\n\n" + example["solution"]
    }

processed_data = [extract_qa(item) for item in raw_data]
dataset = Dataset.from_list(processed_data)
train_dataset = dataset.train_test_split(test_size=0.1)["train"]
eval_dataset = dataset.train_test_split(test_size=0.1)["test"]

# 加载模型
teacher_model_path = "/home/gaoziheng/nlp/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"  # 教师模型路径
student_model_path = "/home/gaoziheng/nlp/models/Qwen/Qwen2___5-0___5B"  # 学生模型路径

teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path, local_files_only=True)
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_path, local_files_only=True)

student_tokenizer = AutoTokenizer.from_pretrained(student_model_path, local_files_only=True)
student_model = AutoModelForCausalLM.from_pretrained(student_model_path, local_files_only=True)

# 预处理函数
def preprocess_function(examples):
    return teacher_tokenizer(examples["text"], truncation=True, padding="max_length", max_length=2048)

train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# 数据收集器
data_collator = DataCollatorForLanguageModeling(tokenizer=teacher_tokenizer, mlm=False)

# 训练参数
training_args = TrainingArguments(
    output_dir="/user/gaoziheng/nlp/distil/Qwen2.5-0.5B-distil",
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="/user/gaoziheng/nlp/distil/logs",
    logging_steps=10,
    fp16=True,
    gradient_accumulation_steps=4,
    report_to="tensorboard",
)

# 蒸馏配置
distill_config = DistillationConfig(
    temperature=2.0,
    hard_label_weight=0.5,
    kd_loss_type="ce",
    intermediate_matches=[
        {"layer_T": 0, "layer_S": 0, "feature": "hidden", "weight": 1.0, "loss": "mse"},
        {"layer_T": 8, "layer_S": 3, "feature": "hidden", "weight": 1.0, "loss": "mse"},
        {"layer_T": 16, "layer_S": 6, "feature": "hidden", "weight": 1.0, "loss": "mse"},
        {"layer_T": 32, "layer_S": 12, "feature": "hidden", "weight": 1.0, "loss": "mse"}
    ]
)

# 训练配置
train_config = TrainingConfig(
    device="cuda" if torch.cuda.is_available() else "cpu",
    log_dir="/user/gaoziheng/nlp/distil/logs",
    output_dir="/user/gaoziheng/nlp/distil/outputs"
)

def simple_adaptor(model, batch):
    """
    Adaptor function to convert model outputs into logits for distillation.
    """
    outputs = model(**batch)
    return {
        "logits": outputs.logits,
        "hidden": tuple(outputs.hidden_states) if outputs.hidden_states else None,
        "attentions": tuple(outputs.attentions) if outputs.attentions else None
    }

# 创建蒸馏器
distiller = GeneralDistiller(
    train_config=train_config,
    distill_config=distill_config,
    model_T=teacher_model,
    model_S=student_model,
    adaptor_T=simple_adaptor,
    adaptor_S=simple_adaptor
)

# 启动训练
with distiller:
    trainer = Trainer(
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )
    trainer.train()
    trainer.save_model()

logger.info("Training complete.")

os.makedirs("/user/gaoziheng/nlp/distil/logs", exist_ok=True)

loss_log_path = "/user/gaoziheng/nlp/distil/logs/logs_history.json"
training_losses = [
    {"step": log["step"], "loss": log["loss"], "epoch": log["epoch"]}
    for log in trainer.state.log_history if "loss" in log and "eval_loss" not in log
]
with open(loss_log_path, "w", encoding="utf-8") as f:
    json.dump(training_losses, f, indent=2, ensure_ascii=False)

logger.info(f"Training losses saved to {loss_log_path}")