import pandas as pd

file_path = r"D:\桌面\大学\和利时\data\问答数据集-v2.xlsx"
df = pd.read_excel(file_path)

filtered_df = df[df['instruction'].str.contains('轨道交通') | df['output'].str.contains('轨道交通')]

saved_file_path = r"D:\桌面\大学\和利时\data\关键词_轨道交通_问答数据集(introduction&output).xlsx"
filtered_df.to_excel(saved_file_path,index=False)

print(saved_file_path)