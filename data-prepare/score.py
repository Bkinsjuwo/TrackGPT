import pandas as pd
import json

# 加载数据集
def load_dataset(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)

# 加载已存在的评分数据
def load_existing_scores():
    try:
        existing_scores = pd.read_csv('qa_scores.csv')
        completed_ids = set(existing_scores['id'].astype(str))
    except FileNotFoundError:
        # 如果不存在评分文件，初始化一个空的DataFrame
        existing_scores = pd.DataFrame(columns=['id', 'question', 'answer', 'score'])
        completed_ids = set()
    return existing_scores, completed_ids

# 保存评分到CSV
def save_scores(results):
    results.to_csv('qa_scores.csv', index=False)

# 输出未评分问答回答到txt
def output_to_txt(data, start_index=0):
    with open('output.txt', 'w', encoding='utf-8') as file:  # 使用写入模式，覆盖原文件
        end_index = start_index + 90
        for item in data[start_index:end_index]:
            question = item['conversations'][0]['value']
            answer = item['conversations'][1]['value']
            file.write(f"ID: {item['id']}\n问题: {question}\n回答: {answer}\n\n")
    return end_index

# 主程序
def main():
    dataset = load_dataset('subway_rail_dataset.json')

    while True:  # 循环让用户选择操作
        mode = input("请选择模式：1-录入模式，2-输出模式，0-退出程序：")
        existing_scores, completed_ids = load_existing_scores()  # 每次选择模式时重新加载评分数据

        if mode == '1':
            # 批量录入
            filtered_data = [item for item in dataset if item['id'] not in completed_ids]
            batch_size = 30
            for i in range(0, len(filtered_data), batch_size):
                batch = filtered_data[i:i + batch_size]
                for item in batch:
                    question = item['conversations'][0]['value']
                    answer = item['conversations'][1]['value']
                    print(f"ID: {item['id']}\n问题: {question}\n回答: {answer}\n")

                scores = input("请连续输入这30条的评分（使用空格分隔，0=低，1=中，2=高，输入back返回主菜单）：").split()
                if 'back' in scores:
                    break  # 返回主菜单
                while len(scores) != len(batch) or any(score not in ['0', '1', '2'] for score in scores):
                    print("输入有误，请确保评分数量正确并且只包含0, 1, 或2.")
                    scores = input("请重新输入这30条的评分（使用空格分隔，0=低，1=中，2=高，输入back返回主菜单）：").split()
                    if 'back' in scores:
                        break  # 返回主菜单
                if 'back' in scores:
                    continue  # 继续外层循环，返回主菜单

                for item, score in zip(batch, scores):
                    existing_scores = existing_scores.append(
                        {'id': item['id'], 'question': item['conversations'][0]['value'],
                         'answer': item['conversations'][1]['value'], 'score': score}, ignore_index=True)

                save_scores(existing_scores)
                print("进度已自动保存.")
        elif mode == '2':
            # 输出模式，输出未完成评分的数据
            filtered_data = [item for item in dataset if item['id'] not in completed_ids]
            next_index = 0
            continue_output = 'yes'
            while continue_output.lower() == 'yes' and next_index < len(filtered_data):
                next_index = output_to_txt(filtered_data, next_index)
                continue_output = input("继续输出下90条？输入yes继续，其他退出：")
        elif mode == '0':
            print("退出程序。")
            break

if __name__ == "__main__":
    main()
