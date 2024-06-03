
import json
from rouge_score import rouge_scorer

# 加载JSON数据
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# 计算ROUGE分数
def calculate_rouge(data):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    results = []
    for item in data:
        scores = scorer.score(item["答案"], item["model"])
        results.append(scores)
    return results

# 计算平均ROUGE分数
def calculate_average_rouge(results):
    total_rouge1 = {'precision': 0, 'recall': 0, 'fmeasure': 0}
    total_rouge2 = {'precision': 0, 'recall': 0, 'fmeasure': 0}
    total_rougeL = {'precision': 0, 'recall': 0, 'fmeasure': 0}
    n = len(results)

    for scores in results:
        for key in total_rouge1.keys():
            total_rouge1[key] += getattr(scores['rouge1'], key)
            total_rouge2[key] += getattr(scores['rouge2'], key)
            total_rougeL[key] += getattr(scores['rougeL'], key)

    average_rouge1 = {key: val / n for key, val in total_rouge1.items()}
    average_rouge2 = {key: val / n for key, val in total_rouge2.items()}
    average_rougeL = {key: val / n for key, val in total_rougeL.items()}

    return average_rouge1, average_rouge2, average_rougeL


# 主函数
def main():
    file_path = './new/test_subway/subway_test_8epoch.json'  # JSON文件路径
    data = load_data(file_path)
    results = calculate_rouge(data)
    average_rouge1, average_rouge2, average_rougeL = calculate_average_rouge(results)
    print("Average ROUGE-1:", average_rouge1)
    print("Average ROUGE-2:", average_rouge2)
    print("Average ROUGE-L:", average_rougeL)

if __name__ == "__main__":
    main()
