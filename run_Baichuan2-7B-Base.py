from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

model = pipeline(task=Tasks.text_generation, model='baichuan-inc/Baichuan2-7B-Chat')
input_text = "领导：你这是什么意思？ 小明：没什么意思。意思意思。 领导：你这就不够意思了。 小明：小意思，小意思。领导：你这人真有意思。 小明：其实也没有别的意思。 领导：那我就不好意思了。 小明：是我不好意思。请问：以上“意思”分别是什么意思。"
result = model(input_text)
print(result['text'])