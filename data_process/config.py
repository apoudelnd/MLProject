import transformers

MAX_LEN = 256
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS =1
ACCUMULATION =2 
MODEL_PATH = "model.bin"
TRAINING_FILE = "./data/final_data.csv"
TEST_FILE = "./data/test.csv"
VAL_FILE = "./data/val.csv"
PRE_TRAINED_MODEL = 'bert-base-uncased'
TOKENIZER = transformers.BertTokenizer.from_pretrained(PRE_TRAINED_MODEL)

# /afs/crc.nd.edu/user/a/apoudel/projects/MLProject/data/sample.csv