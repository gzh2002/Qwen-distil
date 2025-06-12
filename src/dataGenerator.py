from distilabel.models.llms import vLLM
from distilabel.steps import GeneratorStep, Step
from distilabel.steps.tasks.base import Task
from distilabel.steps.base import StepInput
from distilabel.typing import GeneratorStepOutput, StepOutput
from distilabel.steps.tasks.text_generation import TextGeneration, check_column_in_template
from distilabel.models.llms.utils import prepare_output
from distilabel.pipeline import Pipeline
from distilabel.typing import FormattedInput, GenerateOutput, ChatType
from distilabel.typing import LLMOutput
from jinja2 import Template
from transformers import AutoTokenizer
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
#from qwen_vl_utils import process_vision_info
from pydantic import PrivateAttr
from pydantic import BaseModel, Field
from pydantic import validate_call
import uuid
import re
import os
import random
from typing import Generator
from typing import Any, List, Union, Dict, Literal, Callable
import logging
import json
import difflib

subjects = ["math"]
maths_subfields = ['Algebra', 'Combinatorics', 'Complex Numbers', 'Conic Sections', 'Derivative', 'Elementary Functions', 'Inequality', 'Logic', 'Number Theory', 'Probability and Statistics', 'Sequence', 'Set Theory', 'Trigonometric Functions']
physics_subfields = ['Electromagnetism', 'Mechanics', 'Modern Physics', 'Optics', 'Thermodynamics']

GEN_PROMPT = '''你是一个{subject}竞赛老师，现在需要你出一道{subfield}方向的高中竞赛题，题目中的变量和公式必须使用LaTeX格式表示，必须保证题目的合理性，同时你要保证题目的多样性，同一个方向的题目你要给出多种不同的题型。你的回答只需要给出完整的题目不需要给出提示。例如“题目: \n\n”'''
CHECK_PROMPT = '''你是一个{subject}竞赛老师，现在有一道{subfieled}方向的高中竞赛题，题目如下：{gen}。请你给出这道题的解法，公式和变量使用LaTeX格式表示。'''

gen_model_path = "/home/gaoziheng/nlp/models/Qwen/Qwen2.5-7B-Instruct"
check_model_path = "/home/gaoziheng/nlp/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
data_path = "/user/gaoziheng/nlp/data/train.json"

def is_similar(text1: str, text2: str, threshold: float = 0.1) -> bool:
    """
    判断两个字符串是否相似（基于 SequenceMatcher）
    
    :param text1: 第一个字符串
    :param text2: 第二个字符串
    :param threshold: 相似度阈值
    :return: 是否相似
    """
    ratio = difflib.SequenceMatcher(None, text1, text2).ratio()
    return ratio >= threshold

def load_existing_questions(file_path: str) -> List[str]:
    """
    从 JSON 文件中加载已存在的 question 字段
    
    :param file_path: JSON 文件路径
    :return: 已有问题列表
    """
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            return [item["question"] for item in data if "question" in item]
        except (json.JSONDecodeError, KeyError):
            return []

class PromptGenerator(GeneratorStep):
    data_num:int
    batch_size:int

    def process(self, offset: int = 0) -> GeneratorStepOutput:
        data_num = self.data_num
        batch_size = self.batch_size
        num_returned = 0
        total_batch_cnt = data_num // batch_size
        for batch_num in range(total_batch_cnt):
            batch = []
            for i in range(batch_size):
                subject = random.choice(subjects)
                if subject == "math": subfield = random.choice(maths_subfields)
                elif subject == "physics": subfield = random.choice(physics_subfields)
                id = str(uuid.uuid4())
                prompt = GEN_PROMPT.replace("{subject}", subject).replace("{subfield}", subfield)
                batch.append(
                    {
                        "id": id,
                        "subject": subject,
                        "subfield": subfield,
                        "instruction": prompt
                    }
                )
            num_returned += batch_size
            print(f"num_returned:{num_returned}")
            yield batch, num_returned >= data_num

    @property
    def outputs(self) -> list[str]:
        return ["id", "subject", "subfield", "instruction"]
    
class ToCheck(Step):
    data_path:str
    @property
    def inputs(self) -> list:
        # 该步骤的输入需要包含的字段，用于静态检查
        return ['instruction', 'generation']
    
    def process(self, inputs: StepInput) -> "StepOutput":
        outputs = []
        json_list = []

        if not os.path.exists(data_path):
            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump([], f)
        existing_questions = load_existing_questions(self.data_path)
        print(len(inputs))
        for input in inputs:
            subject = input["subject"]
            subfield = input["subfield"]
            gen = input["generation"]

            # 检查相似度
            if any(is_similar(gen, old_q, threshold=0.8) for old_q in existing_questions):
                logging.warning("检测到高度相似的题目，已跳过：\n%s", gen)
                continue

            json_data = {
                "id": input["id"],
                "subject": subject,
                "subfield": subfield,
                "question": gen
            }
            json_list.append(json_data)
            prompt = gen.replace("{subject}", "subject").replace("{subfield}", "subfield").replace("{gen}", "gen")
            outputs.append({
                "id": input["id"],
                "instruction": prompt
            })
        
        if not os.path.exists(self.data_path):
            with open(self.data_path, 'w', encoding='utf-8') as f:
                json.dump(json_list, f, ensure_ascii=False, indent=4)
        else:
            # 如果文件存在，则读取现有数据并追加新的数据
            with open(self.data_path, 'r+', encoding='utf-8') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    # 如果文件为空或不是有效的 JSON，则将其视为空列表
                    existing_data = []
                
                # 将新数据追加到现有数据后面
                existing_data.extend(json_list)
                
                # 将指针移动到文件开头，清空文件内容后写入更新后的数据
                f.seek(0)
                f.truncate()
                json.dump(existing_data, f, ensure_ascii=False, indent=4)
        yield outputs
        
class FinalDeal(Step):
    data_path: str
    @property
    def inputs(self) -> list:
        # 该步骤的输入需要包含的字段，用于静态检查
        return ['id', 'instruction', 'generation']
    
    def process(self, inputs: StepInput) -> "StepOutput":
        with open(self.data_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        for input in inputs:
            gen = input['generation']
            id = input['id']
            for item in json_data:
                json_data[solution] = gen
        with open(self.data_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        yield inputs
  
if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(gen_model_path, trust_remote_code=True)
    # LLM 的参数,
    params_dict_1 = {
        "presence_penalty": 0.5,  # 默认为0.0
        "frequency_penalty": 0.2,
        "max_new_tokens": 1024,
        "temperature": 2,  # 默认为1.0
        "top_p": 0.5,  # 默认为1.0
        "top_k": -1
    }
    
    tokenizer = AutoTokenizer.from_pretrained(check_model_path, trust_remote_code=True)
    params_dict_2 = {
        "presence_penalty": 0.5,  # 默认为0.0
        "frequency_penalty": 0.2,
        "max_new_tokens": 1024,
        "temperature": 0.7,  # 默认为1.0
        "top_p": 0.5,  # 默认为1.0
        "top_k": -1
    }
    
    with Pipeline(
        name='simple-sticker-pipeline-test',
        description='A simple pipeline for sticker dataset',
        cache_dir='/user/gaoziheng/nlp/temp'
    ) as pipeline:
        gen_prompt = PromptGenerator(
            data_num = 2000,
            batch_size = 50
        )
        
        gen_data = TextGeneration(
            # 指定使用的 LLM 以及对应的参数
            # extra_kwargs 是构造 vLLM 实例时的额外参数，generation_kwargs 是调用 `generate` 方法时的额外参数
            llm=vLLM(
                model=gen_model_path,
                generation_kwargs=params_dict_1,
                trust_remote_code=True,
                extra_kwargs={
                    'tensor_parallel_size': 4,
                    'max_model_len': 4096,
                    'enforce_eager': True,
                    'gpu_memory_utilization': 0.95,
                    'enable_prefix_caching': True,
                },
                cuda_devices=[0,1,2,3]
            )
        )
        
        to_check = ToCheck(data_path = data_path)
        
        check_data = TextGeneration(
            llm=vLLM(
                model=check_model_path,
                generation_kwargs=params_dict_2,
                trust_remote_code=True,
                extra_kwargs={
                    'tensor_parallel_size': 2,
                    'max_model_len': 4096,
                    'enforce_eager': True,
                    'gpu_memory_utilization': 0.95,
                    'enable_prefix_caching': True,
                },
                cuda_devices=[4,5,6,7]
            )
        )

        final = FinalDeal(data_path = data_path)
        
        gen_prompt >> gen_data >> to_check >> check_data >> final
        
    result = pipeline.run()