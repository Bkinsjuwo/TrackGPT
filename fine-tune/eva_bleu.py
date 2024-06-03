
import jieba
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data
# def compute_bleu_chinese(ref, hyp):
#     reference = [list(jieba.cut(ref.strip()))]  # 使用jieba进行分词
#     hypothesis = list(jieba.cut(hyp.strip()))
#     return sentence_bleu(reference, hypothesis, smoothing_function=SmoothingFunction().method2)


def compute_bleu_chinese(ref, hyp):
    reference = [list(ref.strip())]
    hypothesis = list(hyp.strip())
    return sentence_bleu(reference, hypothesis, smoothing_function=SmoothingFunction().method2)

def calculate_average_bleu(data):
    bleu_scores = [compute_bleu_chinese(item['答案'], item['model']) for item in data]
    print(len(bleu_scores))
    average_bleu = sum(bleu_scores) / len(bleu_scores)
    return average_bleu

# Example usage
def main():
    file_path = './new/test_subway/subway_test_8epoch.json'
    data = load_data(file_path)
    average_bleu_score = calculate_average_bleu(data)
    print(f"Average BLEU Score: {average_bleu_score:.3f}")
    
if __name__ == "__main__":
    main()