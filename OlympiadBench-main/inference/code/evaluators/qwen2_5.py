from evaluators.evaluator import Evaluator
import os, re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Qwen_25_Evaluator(Evaluator):
    def __init__(self, model_name, k=-1):
        super(Qwen_25_Evaluator, self).__init__(model_name, k)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def split_markdown(self, md):
        # use regex to extract image property
        items = re.split('(<img_\d+>)', md)

        # 从分割后的字符串列表中去除空的元素（完全由空白符、换行符或制表符构成的字符串）
        items = [item for item in items if item and not item.isspace()]
        message_items = []
        for item in items:
            if item[:5] == '<img_':
                image_path = os.path.join(self.image_parent_dir, item[1:-1]+'.jpg')
                if not os.path.exists(image_path):
                    print('Image file not found!')
                image_abs_path = os.path.abspath(image_path)
                message_items.append({
                    'image': f'file://{image_abs_path}'
                })
            else:
                message_items.append({
                    'text': item.strip()
                })

        return message_items

    def make_input(self, prompt, question_content):
        if self.is_chinese:
            subject = '数学' if self.is_math else '物理'
            question_message = self.split_markdown(prompt + '\n' + question_content)
            messages = [
                {
                    'role': 'system',
                    'content': [{'text': f'你是一个中文人工智能助手。请根据要求，完成下面的{subject}竞赛题目。'}]
                },
                {
                    'role': 'user',
                    'content': question_message
                }
            ]
        else:
            subject = 'Math' if self.is_math else 'Physics'
            question_message = self.split_markdown(prompt + '\n' + question_content)
            messages = [
                {
                    'role': 'system',
                    'content': [{'text': f'You are an AI assistant. Please answer the following {subject} competition problems as required.'}]
                },
                {
                    'role': 'user',
                    'content': question_message
                }
            ]
        return messages

    def get_answer(self, input):
        # Convert input messages to a single string
        input_text = ""
        for message in input:
            if message['role'] == 'system':
                input_text += message['content'][0]['text'] + "\n"
            elif message['role'] == 'user':
                for item in message['content']:
                    if 'text' in item:
                        input_text += item['text'] + "\n"
                    elif 'image' in item:
                        input_text += f"![image]({item['image']})\n"

        # Tokenize the input text
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        # Generate the response
        with torch.no_grad():
            outputs = self.model.generate(**inputs)

        # Decode the output tokens to text
        model_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return model_response