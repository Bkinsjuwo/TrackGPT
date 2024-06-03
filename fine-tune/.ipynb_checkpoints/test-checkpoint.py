from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, GenerationConfig,AutoModelForCausalLM
import torch
import json

# Function to load dataset
def load_dataset(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Function to generate dialogue responses
def generate_dialogues(model, tokenizer, data, device):
    model.eval()  # Set the model to evaluation mode
    model.to(device)  # Ensure the model is on the correct device
    all_dialogues = []
    
    # 设置模型的生成配置，假设你有相应的配置文件或使用与原始模型相同的配置
    model.generation_config = GenerationConfig.from_pretrained("baichuan-inc/Baichuan2-7B-Chat")

    for item in data:
        dialog = []
        # human_input = item['conversations'][0]['value'] + "（请使用最简短的回答）" if item['conversations'][0]['from'] == 'human' else ""
        human_input = item['conversations'][0]['value'] + "输出文本不要超过80个字。" if item['conversations'][0]['from'] == 'human' else ""
        gpt_response = item['conversations'][1]['value'] if len(item['conversations']) > 1 and item['conversations'][1]['from'] == 'gpt' else ""

        # human_input = item['conversations'][0]['value'] if item['conversations'][0]['from'] == 'human' else ""
        # gpt_response = item['conversations'][1]['value'] if len(item['conversations']) > 1 and item['conversations'][1]['from'] == 'gpt' else ""
        
        messages = []
        messages.append({"role": "user", "content":human_input})
        model_output= model.chat(tokenizer, messages)

        all_dialogues.append({ '问题': human_input,'答案': gpt_response,'model': model_output})

    # Save dialogues to a JSON file
    with open("./new/medium/checkpoint8.json", 'w') as f:
        json.dump(all_dialogues, f, indent=4)
    print("Generated dialogues saved to './new/medium/checkpoint1.json'")


# # Main function to load model and data, then generate dialogues
def main(model_path, data_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    model = AutoPeftModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    data = load_dataset(data_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generate_dialogues(model, tokenizer, data, device)

if __name__ == "__main__":
    main("output_7B_CHAT_8epochs/checkpoint-7168", "data/subway_rail_test_dataset.json")
