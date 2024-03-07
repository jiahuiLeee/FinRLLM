import re
import os
import torch
import platform
from colorama import Fore, Style
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from transformers.generation.utils import GenerationConfig
from peft import PeftModel

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SYSTEM_PROMPT = "You are an experienced financial risk analyst. Your task is to predict whether the user is at risk of credit default based on the basic financial data given, as follows: " \
    "Your answer format should be as follows:\n\n[Prediction]:\nYes or No\n\n[Analysis]:\n...\n"

def construct_prompt(prompt):
    
    prompt = B_INST + B_SYS + SYSTEM_PROMPT + E_SYS + prompt + E_INST
    
    return prompt

def predict(model, tokenizer, streamer, prompt):
    # 打印GPU使用情况
    # print_gpu_utilization()

    prompt = construct_prompt(prompt)
    
    inputs = tokenizer(prompt, return_tensors='pt', padding=False)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    # print("Inputs loaded onto devices.")
        
    res = model.generate(
        **inputs, max_length=256, do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True, streamer=streamer
    )
    output = tokenizer.decode(res[0], skip_special_tokens=True)
    answer = re.sub(r'.*\[/INST\]\s*', '', output, flags=re.DOTALL)

    torch.cuda.empty_cache()
    
    return answer
    
def init_model():
    print("Initializing model...")
    # model_path = "Go4miii/DISC-FinLLM"
    # model_path = "/home/nist3/Fin4LLM/hf-models/llmaa-2-7b-chat-hf/"
    model_path = "meta-llama/Llama-2-7b-chat-hf"
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True, 
    )
    # model.generation_config = GenerationConfig.from_pretrained(model_path)
    model = PeftModel.from_pretrained(base_model, 'FinGPT/fingpt-forecaster_dow30_llama2-7b_lora')
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    streamer = TextStreamer(tokenizer)
    return model, tokenizer, streamer


def clear_screen():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")
    print(
        Fore.YELLOW
        + Style.BRIGHT
        + "欢迎使用广州大学 FinLLM，输入进行对话，clear 清空历史，Ctrl+C 中断生成，"
        + "stream 开关流式生成，exit 结束。"
    )
    return []


def main(stream=True):
    model, tokenizer, streamer = init_model()

    messages = clear_screen()
    while True:
        prompt = input(Fore.GREEN + Style.BRIGHT + "\n用户：" + Style.NORMAL)
        if prompt.strip() == "exit":
            break
        if prompt.strip() == "clear":
            messages = clear_screen()
            continue
        print(Fore.CYAN + Style.BRIGHT + "\nFinLLM：" + Style.NORMAL, end="")

        if prompt.strip() == "stream":
            stream = not stream
            print(
                Fore.YELLOW + "({}流式生成)\n".format("开启" if stream else "关闭"),
                end="",
            )
            continue
        messages.append({"role": "user", "content": prompt})

        if stream:
            position = 0
            try:
                for response in predict(model, tokenizer, streamer, prompt):
                    print(response[position:], end="", flush=True)
                    position = len(response)
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
            except KeyboardInterrupt:
                pass
            print()
        else:
            response = predict(model, tokenizer, streamer, prompt)
            print(response)
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        messages.append({"role": "assistant", "content": response})

    print(Style.RESET_ALL)


if __name__ == "__main__":
    main()