from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import os
import asyncio

# 创建OpenAI客户端
client = OpenAI(
    api_key="sk-9a2d0ee05c9d43b98a7e7eea4f670d5b",  # 替换成真实DashScope的API_KEY
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 填写DashScope服务endpoint
)

async def generate_answer_qianfan(prompt):
    limited_token_prompt = "输出文本不要超过80个字。"
    try:
        completion = client.chat.completions.create(
            model="qwen-turbo",
            messages=[{'role': 'system', 'content': 'You are a helpful assistant.'},
                      {'role': 'user', 'content': prompt + limited_token_prompt}],
            stream=True
        )

        combined_result = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                combined_result += chunk.choices[0].delta.content
        return combined_result
    except OpenAI.BadRequestError:
        print(f"BadRequestError for prompt '{prompt}'. Skipping...")
        return "错误：输入数据可能包含不当内容，已跳过此问题。"

def main():
    file_path = "D:\\桌面\\大学\\和利时\\KG_test_set.csv"
    data = pd.read_csv(file_path,encoding='gbk')

    output_file_path = "D:\\桌面\\大学\\和利时\\data\\测试数据-KG-test\\通义千问-v2.xlsx"
    if not os.path.exists(output_file_path):
        # 如果文件不存在，首先创建文件和表头
        pd.DataFrame(columns=["问题", "回答"]).to_excel(output_file_path, index=False)

    existing_data = pd.read_excel(output_file_path)
    processed_questions = existing_data['问题'].tolist()

    progress_bar = tqdm(total=len(data))

    for index, row in data.iterrows():
        prompt = row['问题']

        if prompt in processed_questions:
            continue  # 如果问题已处理，跳过

        answer = asyncio.run(generate_answer_qianfan(prompt))

        if "错误" in answer:
            continue  # 如果返回错误信息则跳过保存此问题

        new_row = {"问题": prompt, "回答": answer}
        new_row_df = pd.DataFrame([new_row])

        # 使用ExcelWriter以追加模式写入数据
        with pd.ExcelWriter(output_file_path, mode='a', if_sheet_exists='overlay') as writer:
            new_row_df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)

        progress_bar.set_description(f"Processing {index}")
        progress_bar.update(1)

    progress_bar.close()

if __name__ == "__main__":
    main()

