from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline

tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
model = GPT2LMHeadModel.from_pretrained("./test_trainer/checkpoint-2500")
text_generator = TextGenerationPipeline(model, tokenizer)
result = text_generator("话说大宋仁宗天子在位", max_length=100, do_sample=True)
print(result)
