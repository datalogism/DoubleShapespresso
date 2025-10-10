from .models import GenerationModel


from transformers import AutoModelForCausalLM, AutoTokenizer


class LiquiAIModel(GenerationModel):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="bfloat16",
            #    attn_implementation="flash_attention_2" <- uncomment on compatible GPU
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def model_response(self, prompt):
        input_ids = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
        ).to(self.model.device)

        output = self.model.generate(
            input_ids,
            do_sample=True,
            temperature=0.3,
            min_p=0.15,
            repetition_penalty=1.05,
            max_new_tokens=512,
        )

        return self.tokenizer.decode(output[0], skip_special_tokens=False)

    def structured_response(self, prompt, response_model, stream=False):
        raise NotImplementedError


