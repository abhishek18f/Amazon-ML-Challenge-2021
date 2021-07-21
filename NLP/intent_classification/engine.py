import config
import torch
import torch.nn as nn
from tqdm import tqdm

def loss_fn(outputs , targets):
    return nn.BCEWithLogitsLoss()(outputs , targets.view(-1,1))

def train_fn(data_loader , model , optimizer , device , accumulation_steps,schedular):
    model.train()

    #loop through each batch
    for batch_index , data_batch in tqdm(enumerate(data_loader) , total = len(data_loader)):
        ids = data_batch["ids"]
        mask =  data_batch["mask"]
        targets = data_batch["targets"]

        ids = ids.to(device, dtype = torch.long)
        mask = mask.to(device, dtype = torch.long)
        targets = targets.to(device, dtype = torch.float)

        optimizer.zero_grad()
        outputs = model(
            ids = ids,
            mask = mask
        )

        # print(f"inputs {len(ids) } , mask {len(mask)}  target {targets.shape}")
        loss = loss_fn(outputs, targets)
        loss.backward()
        print(f"loss = {loss.item()}")
        # optimizer.step()
        # schedular.step()     

        if (batch_index + 1) % accumulation_steps == 0:
            optimizer.step()
            schedular.step()


def eval_fn(data_loader , model, device):
    model.eval()
    final_targets = []
    final_outputs = []

    #loop through each batch
    with torch.no_grad():   #??
        for batch_index , data_batch in tqdm(enumerate(data_loader) , total = len(data_loader)):
            ids = data_batch['ids']
            mask =  data_batch['mask']
            targets = data_batch['targets']

            ids = ids.to(device, dtype = torch.long)
            mask = mask.to(device, dtype = torch.long)
            targets = targets.to(device, dtype = torch.float)

            outputs = model(
                ids = ids,
                mask = mask
            )

            final_targets.extend(targets.cpu().detach().numpy().tolist())
            final_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())    #change this in case of multiple outputs

    return final_outputs , final_targets