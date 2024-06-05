import qianfan
import pandas as pd
from tqdm import tqdm
import os
from tenacity import retry, wait_random_exponential, stop_after_attempt
import asyncio

# 设置千帆大模型的key
chat_comp = qianfan.ChatCompletion(ak="slNUmkcQ2DDk156NGwDmg9Od", sk="BO5EBPEtAhc9OAwarNkyRJsvdSDPqYog")


@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(10))
async def generate_answer_qianfan(prompt):
    limited_token_prompt = "输出文本不要超过80个字。"
    resp = await chat_comp.ado(
        model="ERNIE-Bot-turbo",  # 修改为有效的模型名
        messages=[{
            "role": "user",
            "content": prompt + limited_token_prompt
        }],
        stream=True
    )

    combined_result = ""
    async for r in resp:
        combined_result += r.body['result']
    return combined_result


def main():
    file_path = 'D:\\桌面\\大学\\和利时\\KG_test_set.csv'
    data = pd.read_csv(file_path)

    output_file_path = "D:\\桌面\\大学\\和利时\\data\\测试数据-KG-test\\ERNIE-bot-turbo-v0.xlsx"
    if os.path.exists(output_file_path):
        new_data = pd.read_excel(output_file_path)
    else:
        new_data = pd.DataFrame(columns=["问题", "回答"])

    progress_bar = tqdm(total=min(797, len(data)))

    for index, row in data.iterrows():
        # if index >= 5:
        #     break  # 只处理前10条数据

        progress_bar.set_description(f"Processing {index}")
        progress_bar.update(1)

        prompt = row['问题']
        answer = asyncio.run(generate_answer_qianfan(prompt))
        # print(answer)  # 输出答案进行调试

        new_row = {"问题": prompt, "回答": answer}
        # 将 new_row 转换为 DataFrame
        new_row_df = pd.DataFrame([new_row])

        # 使用 concat 方法添加新行
        new_data = pd.concat([new_data, new_row_df], ignore_index=True)
        new_data.to_excel(output_file_path, index=False)
    progress_bar.close()


if __name__ == "__main__":
    main()
