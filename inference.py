from datasets import load_dataset

dataset = load_dataset("code_x_glue_ct_code_to_text", "ruby")
print(dataset['test'])

test_example = dataset['test'][0]
print("Code:", test_example['code'])

from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-small")

save_directory = '.'
from transformers import T5ForConditionalGeneration

model = T5ForConditionalGeneration.from_pretrained(save_directory)

# prepare for the model
input_ids = tokenizer(test_example['code'], return_tensors='pt').input_ids
# generate
outputs = model.generate(input_ids)
print("Generated docstring:", tokenizer.decode(outputs[0], skip_special_tokens=True))

print("Ground truth:", test_example['docstring'])