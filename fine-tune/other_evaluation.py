import json
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import jieba

# 读取JSON数据
def load_data(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

# 计算BLEU和ROUGE指标
def calculate_metrics(data):
    bleu_scores = []
    rouge_scores = []
    rouge = Rouge()
    
    for entry in data:
        print(type(entry))
        print(entry)
        reference = entry['gpt']
        candidate = entry['model']
        
        # Tokenize reference and output text using jieba
        reference_tokens = list(jieba.cut(reference))
        output_tokens = list(jieba.cut(candidate))
                

        # Calculate BLEU score
        bleu_score = sentence_bleu(reference_tokens, output_tokens, weights=(0.25, 0.25, 0.25, 0.25))
        bleu_scores.append(bleu_score)
       # Calculate ROUGE score
        rouge_score = rouge.get_scores(candidate, reference)[0]
        rouge_scores.append(rouge_score)

     # Save bleu_scores and rouges to JSON file
    bleus_path = "./new/bleus.json"
    with open(bleus_path, 'w') as f:
        json.dump(bleu_scores, f, indent=4)
    print("bleu_scores saved to:", bleus_path)
    
    # Save rouge_scores and rouges to JSON file
    rouges_path = "./new/rouges.json"
    with open(rouges_path, 'w') as f:
        json.dump(rouge_scores, f, indent=4)
    print("rouge_scores saved to:", rouges_path)
        
        
      # Calculate average BLEU score
    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
    
    # Calculate average ROUGE score
    avg_rouge_score = {
        'rouge-1': sum(score['rouge-1']['f'] for score in rouge_scores) / len(rouge_scores),
        'rouge-2': sum(score['rouge-2']['f'] for score in rouge_scores) / len(rouge_scores),
        'rouge-l': sum(score['rouge-l']['f'] for score in rouge_scores) / len(rouge_scores)
    }
    
    

    # Save results to JSON file
    results = {
        "Bleu": {
            "Average bleu score": avg_bleu_score,
            "Max bleu": max(bleu_scores),
            "Min bleu": min(bleu_scores),
            "Variance": sum((bleu_score - sum(bleu_scores) / len(bleu_scores)) ** 2 for bleu_score in bleu_scores) / len(bleu_scores)
        },
        "average_rouge_score": avg_rouge_score
        # "Losses": {
        #     "Average Loss": sum(losses) / len(losses),
        #     "Max Loss": max(losses),
        #     "Min Loss": min(losses),
        #     "Variance": sum((loss - sum(losses) / len(losses)) ** 2 for loss in losses) / len(losses)
        # }
    }
    results_path = "./new/generated_dialogues_evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print("Evaluation results saved to:", results_path)   
                

    return bleu_scores, rouge_scores

# 主程序
def main():
    filename = 'generated_dialogues.json'
    data = load_data(filename)
    bleu_scores, rouge_scores = calculate_metrics(data)
    
    print("BLEU Scores:", sum(bleu_scores)/len(bleu_scores))

if __name__ == "__main__":
    main()