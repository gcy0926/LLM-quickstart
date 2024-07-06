import os
from datasets import load_dataset, DatasetDict

model_name_or_path = "openai/whisper-large-v2"
model_dir = "models/whisper-large-v2-asr-int8"
data_dir = "/root/dataDisk/.cache/"

language = "Chinese (China)"
language_abbr = "zh-CN"
task = "transcribe"
dataset_name = "mozilla-foundation/common_voice_11_0"

batch_size=64

# 方法一：设置 HF_HOME 环境变量，修改数据下载地址
os.environ['HF_HOME'] = data_dir
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_TIMEOUT'] = '300'

common_voice = DatasetDict()

common_voice["train"] = load_dataset(dataset_name, language_abbr, split="train", trust_remote_code=True)
common_voice["validation"] = load_dataset(dataset_name, language_abbr, split="validation", trust_remote_code=True)

common_voice["train"][0]