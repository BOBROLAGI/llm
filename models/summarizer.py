import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class Summarizer:
    def __init__(self, model_name: str = "IlyaGusev/rugpt3medium_sum_gazeta"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate_summarization(self, text: str):
        text_tokens = self.tokenizer(
                                    text,
                                    max_length=20,
                                    add_special_tokens=False,
                                    padding=False,
                                    truncation=True)["input_ids"]
        input_ids = text_tokens + [self.tokenizer.sep_token_id]
        input_ids = torch.LongTensor([input_ids])

        output_ids = self.model.generate(
            input_ids=input_ids,
            no_repeat_ngram_size=4
        )

        summary = self.tokenizer.decode(output_ids[0], skip_special_tokens=False)
        summary = summary.split(self.tokenizer.sep_token)[1]
        summary = summary.split(self.tokenizer.eos_token)[0]
        return summary
