import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModel, BertTokenizer, BertModel
import config
import torch.nn as nn


class Models(nn.Module):

    def __init__(self, mdl):
        super(Models, self).__init__()
        self.model = AutoModel.from_pretrained(mdl, return_dict = False)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(self.model.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        #pooled_ouput is the pooler layer -- simply put the output from the CLS token 
        _, pooled_output = self.model(
            input_ids = input_ids,
            attention_mask = attention_mask
        )

        output = self.drop(pooled_output)
        output = self.out(output)

        return output

