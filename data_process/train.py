import engine
import config
from model import Models
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import transformers
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer
import torch
import dataset
from collections import defaultdict
import logging
import os
transformers.logging.set_verbosity_error()

def run():
    df = pd.read_csv(config.TRAINING_FILE).fillna("None")
    df_train, df_valid = train_test_split(
        df,
        test_size= 0.2,
        random_state = 42,
        stratify = df.labels.values
    )
    print('shape of the training dataset is {}'.format(len(df_train)))

    df_valid, df_test = train_test_split(
        df_valid, 
        test_size = 0.1,
        random_state = 42, 
    )


    


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    #list of supported models to train on!

    supported_models = ['nghuyong/ernie-2.0-en','bert-base-uncased' , 'roberta-base']

    # logger = logging.getLogger(__name__)

    for mdl in supported_models:
        # print(directory)
        tokenizer = AutoTokenizer.from_pretrained(mdl)
        train_data_loader = dataset.create_data_loader(df_train, config.TRAIN_BATCH_SIZE,tokenizer, num_workers = 4)
    # data = next(iter(train_data_loader))
    # print(data['input_ids'].shape)

        valid_data_loader = dataset.create_data_loader(df_valid, config.VALID_BATCH_SIZE,tokenizer, num_workers = 2)

        test_data_loader = dataset.create_data_loader(df_test, config.VALID_BATCH_SIZE,tokenizer, num_workers = 2)
    
        print ("Working on {} model" .format(mdl))
        models = Models(mdl)
        models.to(device)

        optimizer = AdamW(models.parameters(), lr = 2e-5, correct_bias = False)

        num_train_steps = len(train_data_loader) * config.EPOCHS  # len(df_train)/len(train_data_loader) * EPOCHS

        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps = 0,
            num_training_steps = num_train_steps
        )

        history = defaultdict(list)
        best_acc =  0
        for epoch in range(config.EPOCHS):
            dir_path = os.path.join(os.getcwd(),mdl[0:4])

            if not os.path.exists(dir_path):
                os.mkdir(dir_path)

            path = os.path.join(dir_path, 'model.pth')
            print(path)

            print (f'Epoch {epoch + 1}/{config.EPOCHS}')
            print ('-'*10)

            train_acc, train_loss = engine.train_epoch(train_data_loader, models, optimizer, scheduler, device, len(df_train))
            val_acc, val_loss = engine.eval_epoch(valid_data_loader, models, device, len(df_valid))

            print (f'Train loss {train_loss} accuracy {train_acc}')

            print(f'Val loss {val_acc} accuracy {val_loss}')
            print()

            history['train_acc'].append(train_acc)
            history['train_loss'].append(train_loss)
        
            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss)

            if val_acc > best_acc:
                output = open(path, mode="wb")
                torch.save(models.state_dict(), output)
                best_acc = val_acc

        history['train_acc'] = torch.stack(history['train_acc']).cpu()
        history['val_acc'] = torch.stack(history['val_acc']).cpu()


        engine.trainingvsvalid(history['train_acc'].cpu(), history['val_acc'].cpu(), mdl)

        test_acc, _ = engine.eval_epoch(
                            test_data_loader,
                            models,
                            device,
                            len(df_test)
                            )
        print("The test accuracy of the {} is {}".format(mdl , test_acc.item()))

        y_review_texts, y_pred, y_pred_probs, y_test = engine.get_prediction(test_data_loader, 
                                                                        models, device, len(df_test))

        class_names = ['0','1']
        print("Classification report for the test dataset on {} model".format(mdl))
        print(classification_report(y_test, y_pred, target_names= class_names))

        cm = confusion_matrix(y_test, y_pred)
        df_cm = pd.DataFrame(cm, index = class_names, columns = class_names)
        engine.show_confusion_matrix(df_cm)
    
        break

if __name__ == "__main__":
    run()