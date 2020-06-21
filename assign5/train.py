from tqdm import tqdm, trange
from dataset import SquadDataset, SquadFeatureDataset
from model import BertForSquad
from typing import List, Tuple
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler

# YOUR LIBRARIES HERE
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# END YOUR LIBRARIES


def train():
    """ Training function for Squad QA BERT model
    Implement the Squad QA trainer which trains the model you have made.

    Note: There are useful tools for your implementation below.

    Memory tip 1: If you delete the output tensors explictly after every loss calculation like "del out, loss",
                  tensors are garbage-collected before next loss calculation so you can cut memory usage.

    Memory tip 2: If you want to keep batch_size while reducing memory usage,
                  creating a virtual batch is a good solution.
    Explanation: https://medium.com/@davidlmorton/increasing-mini-batch-size-without-increasing-memory-6794e10db672

    Useful readings: https://blog.paperspace.com/pytorch-memory-multi-gpu-debugging/ 
    """
    # Below options are just our recommendation. You can choose your own options if you want.
    epochs = 5
    learning_rate = 5e-5
    batch_size = 4
    bert_type = 'bert-base-uncased'

    # Change the lazy option if you want fast debugging.
    dataset = SquadFeatureDataset(
        SquadDataset(), bert_type=bert_type, lazy=True)

    model = BertForSquad.from_pretrained(bert_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # YOUR CODE HERE
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              num_workers=4,
                                              collate_fn=squad_feature_collate_fn)
    # rec = open('record.txt', 'w')

    model = model.to(device)

    for e in trange(epochs):
        model.train()

        loss_per_epoch = 0.

        for data in tqdm(iter(data_loader)):
            (input_ids,
             attention_mask,
             token_type_ids,
             start_pos,
             end_pos) = data

            # print('=======')
            # print('len(input_ids)', len(input_ids))
            # print('start_pos', start_pos)
            # print('end_pos', end_pos)

            optimizer.zero_grad()
            out = model(
                input_ids.to(device),
                attention_mask.to(device),
                token_type_ids.to(device))
            start_logits, end_logits = out

            # print('start_logits', start_logits.cpu())
            # print('end_logits', end_logits.cpu())

            ignore_index = start_logits.size(1)
            loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

            start_loss = loss_fn(start_logits, start_pos.to(device))
            end_loss = loss_fn(end_logits, end_pos.to(device))
            loss = start_loss + end_loss

            # print(start_loss)
            # print(end_loss)

            loss.backward()
            optimizer.step()

            loss_per_epoch += loss.cpu().item()

            del out, loss

        # rec.write('epoch %d: %f\n' % (e, loss_per_epoch))

    # rec.close()
    # END YOUR CODE

    # Save the model in the checkpoint folder
    model.save_pretrained('./checkpoint')

#############################################################
# Useful utils below.                                       #
# You can modify or adapt these utils for your trainer.     #
#############################################################


def squad_feature_collate_fn(
    samples: List[Tuple[List[int], List[int], int, int]]
):
    """ Squad sample sollate function
    This function also generates attention mask

    How to Use:
    data_loader = Dataloader(squad_feature_dataset, ..., collate_fn=squad_feature_collate_fn)
    """
    input_ids, token_type_ids, start_pos, end_pos = zip(*samples)
    attention_mask = [[1] * len(input_id) for input_id in input_ids]

    input_ids = pad_sequence([torch.Tensor(input_id).to(torch.long) for input_id in input_ids],
                             padding_value=0, batch_first=True)
    token_type_ids = pad_sequence([torch.Tensor(token_type_id).to(torch.long) for token_type_id in token_type_ids],
                                  padding_value=1, batch_first=True)
    attention_mask = pad_sequence([torch.Tensor(mask).to(torch.long) for mask in attention_mask],
                                  padding_value=0, batch_first=True)

    start_pos = torch.Tensor(start_pos).to(torch.long)
    end_pos = torch.Tensor(end_pos).to(torch.long)

    return input_ids, attention_mask, token_type_ids, start_pos, end_pos


class SquadBucketSampler(Sampler):
    """ Squad dataset bucketed batch sampler

    How to Use:
    squad_feature_dataset(squad_dataset, lazy=False)
    batch_sampler = SquadBucketSampler(squad_feature_dataset, batch_size, shuffle=True)
    data_loader = DataLoader(squad_feature_dataset, ..., batch_size=1, batch_sampler=batch_sampler, ...)
    """

    def __init__(self, dataset: SquadFeatureDataset, batch_size, shuffle=False):
        super().__init__(dataset)
        self.shuffle = shuffle

        _, indices = zip(*sorted((len(input_ids), index) for index,
                                 (input_ids, _, _, _) in enumerate(tqdm(dataset, desc="Bucketing"))))
        self.batched_indices = [indices[index: index+batch_size]
                                for index in range(0, len(indices), batch_size)]

    def __len__(self):
        return len(self.batched_indices)

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batched_indices)

        for batch in self.batched_indices:
            yield batch


if __name__ == "__main__":
    train()
