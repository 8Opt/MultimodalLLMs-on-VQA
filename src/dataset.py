import os
from PIL import Image
from torch.utils.data import Dataset

class VQADataset(Dataset):

    def __init__(self, dataset, processor, data_path):
        self.dataset = dataset
        self.processor = processor
        self.data_path = data_path

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        question = self.dataset[idx]['question']
        answer = self.dataset[idx]['answer']
        image_id = self.dataset[idx]['pid']
        image_path = os.path.join(self.data_path, f"train_fill_in_blank/train_fill_in_blank/{image_id}/image.png")
        image = Image.open(image_path).convert("RGB")
        text = question

        encoding = self.processor(image, text, padding="max_length", truncation=True, return_tensors="pt")
        labels = self.processor.tokenizer.encode(answer, max_length=8, pad_to_max_length=True, return_tensors="pt")
        encoding['labels'] = labels
        for k, v in encoding.items():
            encoding[k] = v.squeeze()

        return encoding