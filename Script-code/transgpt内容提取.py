import json
import pandas as pd

# 读取文件的路径
file_path = 'D:\桌面\大学\和利时\data\TransGPT-sft (1) - 副本.txt'

# 读取文件内容
with open(file_path, 'r', encoding='utf-8') as file:
    data = file.read()

# 将文件内容分割为单独的行
lines = data.split('\n')

# 处理每一行，去除额外的封装和转义字符
processed_lines = []
for line in lines:
    if line:
        # 去除最外层的双引号
        trimmed_line = line[1:-1]
        # 替换转义字符
        unescaped_line = trimmed_line.encode().decode('unicode_escape')
        processed_lines.append(unescaped_line)

# 处理编码以正确显示中文字符
correctly_decoded_lines = []
for line in processed_lines:
    try:
        # 使用utf-8解码
        decoded_line = line.encode('latin1').decode('utf-8')
        correctly_decoded_lines.append(decoded_line)
    except UnicodeDecodeError as e:
        print(f"解码错误：{e}")

# 解析每行数据，并确保数据是字典类型
parsed_data = []
for line in correctly_decoded_lines:
    try:
        # 直接从字符串解析JSON
        json_data = json.loads(line)
        if isinstance(json_data, dict):  # 确保是字典类型
            parsed_data.append(json_data)
    except json.JSONDecodeError as e:
        print(f"解析错误：{e}")

# 提取 'instruction' 和 'output'
instructions = [item.get('instruction', '') for item in parsed_data]  # 使用get避免类型错误
outputs = [item.get('output', '') for item in parsed_data]

# 添加函数以清洗字符串中的非法字符
def clean_text(text):
    # 这里仅作为示例，您可以根据需要自定义清洗规则
    return ''.join(char for char in text if char.isprintable())



# 创建DataFrame
df = pd.DataFrame({
    'instruction': instructions,
    'output': outputs
})
# 清洗DataFrame中的数据
df['instruction'] = df['instruction'].apply(clean_text)
df['output'] = df['output'].apply(clean_text)
# 保存到Excel文件的路径
output_file_path = 'D:\桌面\大学\和利时\data\问答数据集-v2.xlsx'

# 保存DataFrame到Excel
df.to_excel(output_file_path, index=False)
