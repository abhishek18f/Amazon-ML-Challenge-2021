import config
import transformers
import torch.nn as nn

class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.bert_drop = nn.Dropout(p = 0.3)
        self.out = nn.Linear(768 , 1)   #change 1 to number of intnents and also add actication functions

    def forward(self, ids , mask):
        #out1 = (batch_size, sequence_length, 786) – Sequence of hidden-states at the output of the last layer of the model.
        #out2 = (batch_size, 786) – Last layer hidden-state of the first token of the sequence (classification token) (?? not sure what this is)
        #                         – Gives a vector of size 768 for each sample in batch
        #https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        _ , out2 = self.bert(
            input_ids = ids,
            attention_mask = mask,
            return_dict=False
            # token_type_ids = token_type_ids     #not sure if it's necessary for this task
        )

        bert_output = self.bert_drop(out2)
        output = self.out(bert_output)
        return output 