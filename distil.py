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
dataset_path = "D:\\code\\py\\class_code\\nlp\\OlympiadBench_Dataset\\data\\OE_TO_maths_zh_COMP.json"
logger.info(f"Loading dataset from: {dataset_path}")
with open(dataset_path, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

# 构造 Dataset 对象（只保留 question 和 solution）
def extract_qa(example):
    return {
        "text": example["question"] + "\n\n" + (example["solution"][0] if example["solution"] else example["final_answer"][0])
    }

processed_data = [extract_qa(item) for item in raw_data]
dataset = Dataset.from_list(processed_data)
train_dataset = dataset.train_test_split(test_size=0.1)["train"]
eval_dataset = dataset.train_test_split(test_size=0.1)["test"]

# 加载模型
script_dir = os.path.dirname(os.path.abspath(__file__))
teacher_model_path = os.path.join(script_dir, "DeepSeek-R1-Distill-Qwen-1.5B")
student_model_path = os.path.join(script_dir, "Qwen2.5-0.5B")

teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path, local_files_only=True)
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_path, local_files_only=True)

student_tokenizer = AutoTokenizer.from_pretrained(student_model_path, local_files_only=True)
student_model = AutoModelForCausalLM.from_pretrained(student_model_path, local_files_only=True)

# 预处理函数
def preprocess_function(examples):
    return teacher_tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# 数据收集器
data_collator = DataCollatorForLanguageModeling(tokenizer=teacher_tokenizer, mlm=False)

# 训练参数
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
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
        {
            "layer_T": 6,
            "layer_S": 6,
            "feature": "hidden",
            "weight": 1.0,
            "loss": "mse"
        }
    ]
)

# 训练配置
train_config = TrainingConfig(
    device="cuda" if torch.cuda.is_available() else "cpu",
    log_dir="./logs",
    output_dir="./outputs"
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
>>>>>>> c7de23d (Initial commit)
