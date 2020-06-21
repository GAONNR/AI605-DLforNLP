import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertPreTrainedModel, BertModel, BertConfig


class BertForSquad(BertPreTrainedModel):
    """ BERT model for Squad dataset
    Implement proper a question and answering model based on BERT.
    We are not going to check whether your model is properly implemented.
    If the model shows proper performance, it doesn't matter how it works.

    BertPretrinedModel Examples:
    https://huggingface.co/transformers/_modules/transformers/modeling_bert.html#BertForQuestionAnswering
    """

    def __init__(self, config: BertConfig):
        """ Model Initializer
        You can declare and initialize any layer if you want.
        """
        super().__init__(config)
        # YOUR CODE HERE
        self.bert = BertModel(config)
        self.linear = nn.Linear(config.hidden_size, 2)
        # END YOUR CODE

        # Don't forget initializing the weights
        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor
    ):
        """ Model Forward Function
        There is no format for the return values.
        However, the input must be in the prescribed form.

        Arguments:
        input_ids -- input_ids is a tensor 
                    in shape (batch_size, sequence_length)
        attention_mask -- attention_mask is a tensor
                    in shape (batch_size, sequence_length)
        token_type_ids -- token_type ids is a tensor
                    in shape (batch_size, sequence_length)

        Returns:
        FREE-FORMAT
        """
        # YOUR CODE HERE
        outputs: torch.Tensor = self.bert(input_ids,
                                          attention_mask=attention_mask,
                                          token_type_ids=token_type_ids)
        seq_out: torch.Tensor = outputs[0]

        logits: torch.Tensor = self.linear(seq_out)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits
        # END YOUR CODE
