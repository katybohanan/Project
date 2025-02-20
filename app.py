import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_name = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='cpu')

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors='pt').to('cpu')
    outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Gradio interface
iface = gr.Interface(fn=generate_response, 
                     inputs=gr.Textbox(label="Enter your message:"), 
                     outputs=gr.Textbox(label="Response"), 
                     live=True,
                     title="🗨️ AI Chatbot",
                     description="Start a conversation!")

iface.launch()
