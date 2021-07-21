import config
from dataLoader import BERTDataset
from model import BERTBaseUncased
from engine import train_fn , eval_fn

import numpy as np
import pandas as pd
import torch

from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

def getModelArchitecture(model):
    # Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())
    print('The BERT model has {:} different named parameters.\n'.format(len(params)))
    print('==== Embedding Layer ====\n')

    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    
    print('\n==== First Transformer ====\n')
    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== Output Layer ====\n')
    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

def run():
    df = pd.read_csv(config.TRAINING_FILE).fillna("none")
    print(df.head())

    #change this lambda mapping in multiple intent classification
    df.sentiment = df.sentiment.apply(
        lambda x : 1 if x == 'positive' else 0
    )


    df_train, df_valid = model_selection.train_test_split(
        df,
        test_size = 0.1,
        random_state = 2000,
        stratify = df.sentiment.values
    )

    df_train = df_train.reset_index(drop = True)
    df_valid = df_valid.reset_index(drop = True)

    train_dataset = BERTDataset(
        sentence = df_train.review.values , 
        target = df_train.sentiment.values
    )

    valid_dataset = BERTDataset(
        sentence = df_valid.review.values , 
        target = df_valid.sentiment.values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset ,
        batch_size = config.TRAIN_BATCH_SIZE, 
        num_workers = 1
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset ,
        batch_size = config.VALIDATION_BATCH_SIZE, 
        num_workers = 1
    )

    model = BERTBaseUncased()

    #to get model architecture information
    getModelArchitecture(model)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("There are %s GPU's." %torch.cuda.device_count())
        print("GPU Name: " , torch.cuda.get_device_name(0))

    else:
        print("No GPU's Available :(")
        device = torch.device("cpu")
    
    model.to(device)



    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
 
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]   

    num_train_steps = len(df_train)/(config.TRAIN_BATCH_SIZE) * (config.EPOCHS)
    optimizer = AdamW(
        optimizer_parameters,
        lr = 2e-5
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_accuracy = 0
    for epoch in range(config.EPOCHS):
        train_fn(train_data_loader , model , optimizer , device , config.ACCUMULATION , scheduler)

        outputs , targets = eval_fn(valid_data_loader , model, device )
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets , outputs)
        print(f"Accuracy Score = {accuracy}")

        # #for saving the model
        # if accuracy > best_accuracy:
        #     torch.save(model.state_dict(), config.MODEL_PATH)
        #     best_accuracy = accuracy

if __name__ == "__main__":
    run()