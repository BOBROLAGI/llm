from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from torch import cuda
import torch
import transformers
import os
from llama_cpp import Llama


class LanguageModel:
    def __init__(self, model_name):
        self.device = f'cuda:{cuda.current_device()}' \
                      if cuda.is_available() else 'cpu'

        self.model_name = model_name
        self.temperature = 0.3

        self.model = Llama(
            model_path=model_name,
            n_batch=2048,
            n_gpu_layers=100,
            n_ctx=5000,
            temperature=self.temperature,
            verbose=True,
        )

        self.prompt_template = """
                        НЕ ОТВЕЧАЙ НА ВОПРОСЫ КОТОРЫЕ НЕ ОТНОСЯТСЯ К ТЕМЕ БАНКОВ
                        Ты - Бедолага, русскоязычный асистент, ты разговариаешь с людьми и помогаешь решить вопросы 
                        связанные с работой банка Используйте следующие занния, чтобы ответить на вопрос 
                        в конце. Если вы не знаете ответа, просто скажите, что не знаете, НЕ ПЫТАЙСЯ ПРИДУМАТЬ ОТВЕТ.
                        НЕ ОТВЕЧАЙ НА ВОПРОСЫ КОТОРЫЕ НЕ ОТНОСЯТСЯ К ТЕМЕ БАНКОВ.
                        Старайся отвечать наиболее точно, укладывайся в 5000 токенов ответа.
                        
                        {context}
                        {links}
                        """

        self.summarization_template = """"
                        Создайте лаконичную фразу из {k} слов в качестве {header} для следующего запроса,
                        строго придерживаясь ограничения в {k} слов и избегая использования слова «заголовок»: 
                        {user_prompt}
                         """

    def generate_text(self, user_prompt, max_tokens=100, top_p=0.1, echo=True, stop=("Q", "\n")):
        output = self.model(user_prompt, top_p=top_p,
                            max_tokens=max_tokens,
                            temperature=self.temperature,
                            echo=echo, stop=stop)

        return output["choices"][0]["text"].strip()
