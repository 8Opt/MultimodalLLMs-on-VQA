import os, json
from PIL import Image

import torch


from transformers import BlipProcessor, BlipForQuestionAnswering


retrain_model_dir = './save_model'
processor = BlipProcessor.from_pretrained(retrain_model_dir)
model = BlipForQuestionAnswering.from_pretrained(retrain_model_dir).to(device)


test_data_dir = f"{data_path}/test_data/test_data"
samples = os.listdir(test_data_dir)
sample_path = os.path.join(test_data_dir, samples[0])
json_path = os.path.join(sample_path, "data.json")

with open(json_path, "r") as json_file:
    data = json.load(json_file)
    question = data["question"]
    image_id = data["id"]

image_path = os.path.join(test_data_dir, f"{image_id}", "image.png")
image = Image.open(image_path).convert("RGB")
image


encoding = processor(image,
                     question,
                     return_tensors="pt").to(device, torch.float16)

outputs = model.generate(**encoding)

generated_text = processor.decode(outputs[0], skip_special_tokens=True)

generated_text