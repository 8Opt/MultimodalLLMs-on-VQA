from model import model, processor
from dataset import VQADataset

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch import optim



train_dataset = VQADataset(dataset=ds["train"],
                           processor=processor,
                           data_path=data_path)
val_dataset = VQADataset(dataset=ds["train"],
                           processor=processor,
                           data_path=data_path)


batch_size=8
train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True)
val_dataloader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=True)


optimizer = optim.AdamW(model.parameters(),
                        lr=4e-5)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                             gamma=0.9,
                                             last_epoch=-1,
                                             verbose=True)
n_epochs = 3
min_val_loss = float("inf")
scaler = torch.cuda.amp.GradScaler()


for epoch in range(n_epochs):

    # Training setting
    train_loss = []
    model.train()
    for idx, batch in zip(tqdm(range(len(train_dataloader)),
                               desc=f"Training Batch {epoch+1}"), train_dataloader):
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device)
        attention_masked = batch.pop('attention_mask').to(device)
        labels = batch.pop('labels').to(device)

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            labels=labels)

        loss = outputs.loss
        train_loss.append(loss.item())

        ## Backward propogation
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # Validating setting
    val_loss = []
    model.eval()
    for idx, batch in zip(tqdm(range(len(val_dataloader)),
                               desc=f"Validating Batch {epoch+1}"), val_dataloader):
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device)
        attention_masked = batch.pop('attention_mask').to(device)
        labels = batch.pop('labels').to(device)

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            attention_mask=attention_masked,
                            labels=labels)

        loss = outputs.loss
        val_loss.append(loss.item())

        avg_train_loss = sum(train_loss)/len(train_loss)
        avg_val_loss = sum(val_loss)/len(val_loss)
        lr_per_epoch = optimizer.param_groups[0]["lr"]
    print(f"Epoch: {epoch + 1} - Training Loss: {avg_train_loss} - Eval Loss: {avg_val_loss} - LR: {lr_per_epoch}")

    scheduler.step()

    if avg_val_loss < min_val_loss:
        model.save_pretrained('./save_model', from_pt=True)
        print ("Saved model to ./save_model")
        min_eval_loss = avg_val_loss
processor.save_pretrained ("./save_model", from_pt=True )