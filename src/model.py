import torch

from peft import LoraConfig, get_peft_model
from transformers import BlipProcessor, BlipForQuestionAnswering


lora_config = LoraConfig(
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.05,
                    bias="none",
                    target_modules=["query", "key"]
                    )

model_id = "Salesforce/blip-vqa-base"
processor = BlipProcessor.from_pretrained(model_id)
model = BlipForQuestionAnswering.from_pretrained(model_id)


model = get_peft_model(model, lora_config )
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.print_trainable_parameters()