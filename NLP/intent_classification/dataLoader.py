import config
import torch

#https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class BERTDataset:
    def __init__(self , sentence , target):
        """
            sentence : list of strings(sentences)
            target : list of ints
        """
        self.sentence = sentence
        self.target = target
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    #total len of dataset
    def __len__(self):
        return len(self.sentence)

    def __getitem__(self , idx):
        sentence = str(self.sentence[idx])   #just to make sure everything is string and not ints or UTF
        sentence = " ".join(sentence.split())

        #tokeizing the sentences
        inputs = self.tokenizer.encode_plus(
            text = sentence,
            add_special_tokens = True,
            max_length = self.max_len,
            padding='max_length',
            truncation = True
            # return_attention_mask = True
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        # print(f"inputs {len(ids) } , mask {len(mask)}  len {len(sentence.split())}  target {self.target[idx]}")

        return {
            'ids' : torch.tensor(ids  , dtype = torch.long),
            'mask' : torch.tensor(mask , dtype = torch.long),
            'targets' : torch.tensor(self.target[idx] , dtype = torch.float)
        }