from transformers import TextStreamer, AutoTokenizer, AutoModelForCausalLM
model_name = "/mnt/data/chatglm3-6b" # 本地路径
prompt = "请说出以下两句话区别在哪里？ 1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少"
tokenizer = AutoTokenizer.from_pretrained(
 model_name,
 trust_remote_code=True
 )
model = AutoModelForCausalLM.from_pretrained(
 model_name,
 trust_remote_code=True,
 torch_dtype="auto" # 自动选择 float32/float16（根据模型配置）
).eval().cuda()
inputs = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
streamer = TextStreamer(tokenizer)
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)





