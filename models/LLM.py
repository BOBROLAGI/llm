from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from torch import cuda
import torch
from llama_cpp import Llama
import transformers
import os


class LanguageModel:
    def __init__(self, model_name):

        self.device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
        self.model_name = model_name

        self.bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.hf_token = os.environ.get("HUGGING_FACE_TOKEN")

        # self.model_config = transformers.AutoConfig.from_pretrained(
        #     model_name,
        #     use_auth_token=self.hf_token
        # )

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote=True,
            #config=self.model_config,
            quantization_config=self.bnb_config,
            device_map='auto',
            use_auth_token=self.hf_token
        )

    def generate_text(self) -> transformers.pipeline:
        generate_text = transformers.pipeline(
            self.model_name,
            tokenizer=transformers.AutoTokenizer.from_pretrained(self.model_name,use_auth_token=self.hf_token),
            return_full_text=True,
            task='text-generation',
            temperature=0.0,
            max_new_tokens=512,
            repetition_penalty=1.1
        )
        return generate_text

    def download_model(self):
        self.model.eval()
        print(f'Model_loaded on {self.device}')
