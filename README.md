# TrackGPT

## fine-tune文件夹

### fine-tune.py——微调脚本

用于执行模型微调，不需要设置输入参数

### train.sh——微调参数文件

在该文件修改模型微调的参数，包括路径等。运行此文件来进行模型微调。

### test.py——测试文件

用于加载微调后模型并获取模型在评估数据集上的回答文本。需在该文件中修改微调后模型的路径和数据集路径，以及生成文件（对话）路径。

### eva_bleu.py——bleu性能评估文件

用于加载对话文件（包含问题，标准答案，模型生成答案），计算bleu指标。

### eva_rouge.py——rouge性能评估文件

用于加载对话文件，计算rouge指标
## Score.py 使用说明文档

### 概述

该文件处理和评分问答数据集，提供录入评分和输出未评分数据的功能。

### 导入依赖

```python
import pandas as pd
import json
```

### 函数

#### `load_dataset(filename)`

- **功能**: 加载JSON数据集。
- **参数**: 
  - `filename` (str): JSON文件路径。
- **返回**: 数据集（字典）。

#### `load_existing_scores()`

- **功能**: 加载已有评分数据，若文件不存在则创建空DataFrame。
- **返回**: 
  - `existing_scores` (DataFrame): 评分数据。
  - `completed_ids` (set): 已评分ID集合。

#### `save_scores(results)`

- **功能**: 保存评分结果到CSV文件。
- **参数**:
  - `results` (DataFrame): 评分结果。

#### `output_to_txt(data, start_index=0)`

- **功能**: 输出未评分数据到TXT文件。
- **参数**: 
  - `data` (list): 数据集。
  - `start_index` (int): 起始索引，默认为0。
- **返回**: 下一个开始索引。

### 主程序

#### `main()`

- **功能**: 用户选择模式（录入评分、输出未评分数据、退出）。
- **流程**:
  1. 加载数据集。
  2. 提示选择模式：
     - `1`: 录入评分
     - `2`: 输出未评分数据
     - `0`: 退出程序

### 使用方法

1. 运行脚本：`python score.py`
2. 输入模式选择：`1` 录入评分，`2` 输出未评分数据，`0` 退出程序。

## data-prepare.py 使用说明文档

### 概述

该文件提供了一组工具函数，用于处理文本文件并将其内容整理成CSV格式。主要功能包括分割和清理文本、将处理后的文本附加到CSV文件中，以及批量处理当前文件夹中的所有文本文件。

## 数据处理代码

## 1. transgpt内容提取.py

从TransGPT官网下载下来的问答型数据json文件提取出表头为introduction和output的表格文件。将file_path替换成合适的json文件就可以直接使用。

```python
import json
import pandas as pd

file_path = "path_to_your_json_file.json"

with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

rows = []
for item in data:
    rows.append({
        'introduction': item['introduction'],
        'output': item['output']
    })

df = pd.DataFrame(rows)
df.to_csv('output.csv', index=False, encoding='utf-8')
```

## 2. 关键词提取.py

提取transgpt问答数据里的“instruction”列或“output”列里含有关键词“轨道交通”的数据。

```python
import pandas as pd

file_path = "path_to_your_csv_file.csv"
df = pd.read_csv(file_path, encoding='utf-8')

keyword = "轨道交通"
filtered_df = df[df['instruction'].str.contains(keyword, na=False) | df['output'].str.contains(keyword, na=False)]

filtered_df.to_csv('filtered_output.csv', index=False, encoding='utf-8')
```

## 3. GPT-3.5-数据集产生.py

使用GPT-3.5模型获取地铁领域的问答数据，设置输入模型的prompt为：“请你生成一条地铁领域的问答数据。输出格式为：问题：***。回答：***。”设置要生成的数据量total_data = 2000。使用方法： 设置OpenAI的api之后就可以直接使用。

```python
import openai

openai.api_key = "your_openai_api_key"

prompt = "请你生成一条地铁领域的问答数据。输出格式为：问题：***。回答：***。"
total_data = 2000
results = []

for _ in range(total_data):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100
    )
    results.append(response.choices[0].text.strip())

with open('gpt35_dataset.txt', 'w', encoding='utf-8') as file:
    for result in results:
        file.write(result + '\n')
```

# 调用模型生成回答代码

## 1. chatglm2-6b.py

使用百度千帆大模型，设置好key，调用模型“ChatGLM2-6B-32K”，使用的prompt为“输出文本不要超过80个字。” + “问题”，这里的问题是测试文件里的问题列的问题数据。

```python
import requests

api_key = "your_baidu_api_key"
url = "https://qianfan.baidu.com/api/v1/model/chatglm2-6b-32k/infer"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

questions = ["测试文件里的问题数据1", "测试文件里的问题数据2", ...]

for question in questions:
    prompt = f"输出文本不要超过80个字。{question}"
    payload = {"prompt": prompt}
    
    response = requests.post(url, headers=headers, json=payload)
    answer = response.json().get('result')
    
    print(f"问题: {question}\n回答: {answer}\n")
```

## 2. ERNIE-3.5-8K.py

使用百度千帆大模型，设置好key，调用模型：ERNIE-Bot-turbo，使用的prompt为“输出文本不要超过80个字。” + “问题”，这里的问题是测试文件里的问题列的问题数据。

```python
import requests

api_key = "your_baidu_api_key"
url = "https://qianfan.baidu.com/api/v1/model/ernie-bot-turbo/infer"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

questions = ["测试文件里的问题数据1", "测试文件里的问题数据2", ...]

for question in questions:
    prompt = f"输出文本不要超过80个字。{question}"
    payload = {"prompt": prompt}
    
    response = requests.post(url, headers=headers, json=payload)
    answer = response.json().get('result')
    
    print(f"问题: {question}\n回答: {answer}\n")
```

## 3. Llama.py

使用百度千帆大模型，设置好key，调用模型：Qianfan-Chinese-Llama-2-7B，使用的prompt为“输出文本不要超过80个字。” + “问题”，这里的问题是测试文件里的问题列的问题数据。

```python
import requests

api_key = "your_baidu_api_key"
url = "https://qianfan.baidu.com/api/v1/model/qianfan-chinese-llama-2-7b/infer"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

questions = ["测试文件里的问题数据1", "测试文件里的问题数据2", ...]

for question in questions:
    prompt = f"输出文本不要超过80个字。{question}"
    payload = {"prompt": prompt}
    
    response = requests.post(url, headers=headers, json=payload)
    answer = response.json().get('result')
    
    print(f"问题: {question}\n回答: {answer}\n")
```

## 4. 通义千问.py

调用模型：qwen-turbo，使用的prompt为“输出文本不要超过80个字。” + “问题”，这里的问题是测试文件里的问题列的问题数据。

```python
import requests

api_key = "your_baidu_api_key"
url = "https://qianfan.baidu.com/api/v1/model/qwen-turbo/infer"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

questions = ["测试文件里的问题数据1", "测试文件里的问题数据2", ...]

for question in questions:
    prompt = f"输出文本不要超过80个字。{question}"
    payload = {"prompt": prompt}
    
    response = requests.post(url, headers=headers, json=payload)
    answer = response.json().get('result')
    
    print(f"问题: {question}\n回答: {answer}\n")
```
