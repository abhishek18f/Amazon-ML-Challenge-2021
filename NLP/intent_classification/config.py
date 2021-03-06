import transformers

MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
VALIDATION_BATCH_SIZE = 4
EPOCHS = 10
ACCUMULATION = 2
BERT_PATH = "../bert_base_uncased/"
MODEL_PATH = "../bert_base_uncased/"
TRAINING_FILE = "../input/IMDB Dataset.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH, 
    do_lower_case = True
)