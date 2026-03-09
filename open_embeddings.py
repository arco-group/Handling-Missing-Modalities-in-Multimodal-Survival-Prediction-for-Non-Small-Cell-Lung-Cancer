import torch
from transformers import AutoModelForCausalLM

olmo = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-0425-1B")


pass