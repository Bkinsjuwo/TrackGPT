from openai import OpenAI
import pandas as pd


import os
import time
from tenacity import retry,stop_after_attempt, wait_random_exponential
from tqdm import tqdm

# os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
client = OpenAI(
    # 输入转发API Key

    api_key="***",
    base_url="https://api.chatanywhere.com.cn"
)

@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(6))
def generate_summary(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt}
        ],
        stream=True,  # 开启流式输出
        temperature=1.1,
        # max_tokens=200
    )

    # 初始化一个空字符串来累加响应的内容
    res = ""

    # 遍历响应的每个片段
    for chunk in response:
        # 获取每个片段的文本内容
        # text = chunk.choices[0].delta.content
        text = chunk.choices[0].delta.content if chunk.choices[0].delta.content is not None else ''
        res += text

        # 延时以确保获取完整响应
        time.sleep(0.01)

    return res

# 输出文件路径
output_file_path = r"D:\桌面\大学\和利时\data\GPT-3.5_地铁_问答数据集-2000.xlsx"

# 检测输出文件是否存在
if os.path.exists(output_file_path):
    # 如果文件存在，则加载已有的数据
    new_data = pd.read_excel(output_file_path)
    # 找到上次处理到的最后一个index
    start_index = new_data.index[-1] + 1
else:
    # 如果文件不存在，创建一个新的DataFrame
    new_data = pd.DataFrame(columns=["问答数据"])
    start_index = 0

# 设置要生成的数据量
total_data = 2000

# 使用tqdm显示进度条
for i in tqdm(range(start_index, total_data), desc="Generating QA Data"):
    # 生成的prompt
    prompt = "请你生成一条地铁领域的问答数据。输出格式为：问题：***。回答：***。"
    try:
        # 调用函数生成数据
        qa_data = generate_summary(prompt)
        # 将生成的数据添加到DataFrame中
        new_data = new_data.append({"问答数据": qa_data}, ignore_index=True)
        # 每次调用后保存数据
        new_data.to_excel(output_file_path, index=False)
    except Exception as e:
        print(f"An error occurred: {e}")
        break