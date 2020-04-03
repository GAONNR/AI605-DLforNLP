from abc import ABC
from typing import List, Dict, Tuple, Set
import random

### You may import any Python standard library here.
import torch.nn.functional as F
### END YOUR LIBRARIE

import torch
from dataset import SkipgramDataset


def naive_softmax_loss(
    center_vectors: torch.Tensor, outside_vectors: torch.Tensor, 
    center_word_index: torch.Tensor, outside_word_indices: torch.Tensor
) -> torch.Tensor:
    """ Naive softmax loss function for word2vec models

    Implement the naive softmax losses between a center word's embedding and an outside word's embedding.
    When using GPU, it is efficient to perform a large calculation at once, so batching is used generally.
    In addition, using a large batch size reduces the variance of samples in SGD, making training process more effective and accurate.
    To practice this, let's calculate batch-sized losses of skipgram at once.
    <PAD> tokens are appended for batching if the number of outside words is less than 2 * window_size. 
    However, these arbitrarily inserted <PAD> tokens have no meaning so should NOT be included in the loss calculation.

    !!!IMPORTANT: Do NOT forget eliminating the effect of <PAD> tokens!!!

    Note: Try not to use 'for' iteration as you can. It may degrade your performance score. You can complete this file without any for iteration.
    Use built-in functions in pyTorch library. They must be faster than your hard-coded script. You can use any funtion in pyTorch library.

    Hint: torch.index_select function would be helpful

    Arguments:
    center_vectors -- center vectors is
                        in shape (num words in vocab, word vector length)
                        for all words in vocab (V in the pdf handout)
    outside_vectors -- outside vector is
                        in shape (num words in vocab, word vector length)
                        for all words in vocab (U in the pdf handout)
    center_word_index -- the index of the center word
                        in shape (batch size,)
                        (c of v_c in the pdf handout)
    outside_word_indices -- the indices of the outside words
                        in shape (batch size, window size * 2)
                        (all o of u_o in the pdf handout.
                        <PAD> tokens are inserted for padding if the number of outside words is less than window size * 2)

    Return:
    losses -- naive softmax loss for each (center_word_index, outsied_word_indices) pair in a batch
                        in shape (batch size,)
    """
    assert center_word_index.shape[0] == outside_word_indices.shape[0]

    n_tokens, word_dim = center_vectors.shape
    batch_size, outside_word_size = outside_word_indices.shape
    PAD = SkipgramDataset.PAD_TOKEN_IDX

    ### YOUR CODE HERE (~4 lines)
    centers: torch.Tensor = center_vectors[center_word_index].unsqueeze(2)
    losses: torch.Tensor = torch.matmul(outside_vectors, centers).squeeze()
    losses[:, PAD] = -1 * float('inf')
    losses = F.log_softmax(losses, dim=1) * -1
    losses[:, PAD] = 0.
    losses = torch.gather(losses, 1, outside_word_indices)
    losses = losses.sum(dim=1).squeeze()
    ### END YOUR CODE
    assert losses.shape == torch.Size([batch_size])
    return losses


def neg_sampling_loss(
    center_vectors: torch.Tensor, outside_vectors: torch.Tensor,
    center_word_index: torch.Tensor, outside_word_indices: torch.Tensor,
    negative_sampler, K: int=10
) -> torch.Tensor:
    """ Negative sampling loss function for word2vec models

    Implement the negative sampling loss for each pair of (center_word_index, outside_word_indices) in a batch.
    As same with naive_softmax_loss, all inputs are batched with batch_size.

    !!!IMPORTANT: Do NOT forget eliminating the effect of <PAD> tokens!!!

    Note: Implementing negative sampler is a quite tricky job so we pre-implemented this part. See below comments to check how to use it.
    If you want to know how the sampler works, check SkipgramDataset.negative_sampler code in dataset.py file

    Hint: torch.gather function would be helpful

    Arguments/Return Specifications: same as naiveSoftmaxLoss

    Additional arguments:
    negative_sampler -- the negative sampler
    K -- the number of negative samples to take
    """
    assert center_word_index.shape[0] == outside_word_indices.shape[0]

    n_tokens, word_dim = center_vectors.shape
    batch_size, outside_word_size = outside_word_indices.shape
    PAD = SkipgramDataset.PAD_TOKEN_IDX

    ##### Sampling negtive indices #####
    # Because each outside word needs K negatives samples,
    # negative_sampler takes a tensor in shape [batch_size, outside_word_size] and gives a tensor in shape [batch_size, outside_word_size, K]
    # where values in last dimension are the indices of sampled negatives for each outside_word.
    negative_samples: torch.Tensor = negative_sampler(outside_word_indices, K)
    assert negative_samples.shape == torch.Size([batch_size, outside_word_size, K])

    ###  YOUR CODE HERE (~5 lines)
    centers: torch.Tensor = center_vectors[center_word_index].unsqueeze(2)

    vec_dots: torch.Tensor = torch.matmul(outside_vectors, centers).squeeze()

    losses_front = F.logsigmoid(torch.gather(vec_dots, 1, outside_word_indices)) * -1
    tmp: torch.Tensor = negative_samples.reshape(batch_size, -1)
    losses_back: torch.Tensor = F.logsigmoid(torch.gather(vec_dots, 1, tmp) * -1)\
        .reshape(batch_size, outside_word_size, -1)\
        .sum(dim=2)
    
    losses: torch.Tensor = (losses_front - losses_back)
    losses[outside_word_indices == 0] = 0.
    losses = losses.sum(dim=1).squeeze()
    ### END YOUR CODE
    assert losses.shape == torch.Size([batch_size])
    return losses

#############################################
# Helper functions below. DO NOT MODIFY!    #
#############################################

class Word2Vec(torch.nn.Module, ABC):
    """
    A helper class that wraps your word2vec losses.
    """
    def __init__(self, n_tokens: int, word_dimension: int):
        super().__init__()

        self.center_vectors = torch.nn.Parameter(torch.empty([n_tokens, word_dimension]))
        self.outside_vectors = torch.nn.Parameter(torch.empty([n_tokens, word_dimension]))

        self.init_weights()

    def init_weights(self):
        torch.nn.init.normal_(self.center_vectors.data)
        torch.nn.init.normal_(self.outside_vectors.data)

class NaiveWord2Vec(Word2Vec):
    def forward(self, center_word_index: torch.Tensor, outside_word_indices: torch.Tensor):
        return naive_softmax_loss(self.center_vectors, self.outside_vectors, center_word_index, outside_word_indices)

class NegSamplingWord2Vec(Word2Vec):
    def __init__(self, n_tokens: int, word_dimension: int, negative_sampler, K: int=10):
        super().__init__(n_tokens, word_dimension)

        self._negative_sampler = negative_sampler
        self._K = K

    def forward(self, center_word_index: torch.Tensor, outside_word_indices: torch.Tensor):
        return neg_sampling_loss(self.center_vectors, self.outside_vectors, center_word_index, outside_word_indices, self._negative_sampler, self._K)

#############################################
# Testing functions below.                  #
#############################################

def test_naive_softmax_loss():
    print ("======Naive Softmax Loss Test Case======")
    center_word_index = torch.randint(1, 100, [10])
    outside_word_indices = []
    for _ in range(10):
        random_window_size = random.randint(3, 6)
        outside_word_indices.append([random.randint(1, 99) for _ in range(random_window_size)] + [0] * (6 - random_window_size))
    outside_word_indices = torch.Tensor(outside_word_indices).to(torch.long)

    model = NaiveWord2Vec(n_tokens=100, word_dimension=3)

    loss = model(center_word_index, outside_word_indices).mean()
    loss.backward()

    # first test
    assert (model.center_vectors.grad[0, :] == 0).all() and (model.outside_vectors.grad[0, :] == 0).all(), \
        "<PAD> token should not affect the result."
    print("The first test passed! Howerver, this test doesn't guarantee you that <PAD> tokens really don't affects result.")    

    # Second test
    temp = model.center_vectors.grad.clone().detach()
    temp[center_word_index] = 0.
    assert (temp == 0.).all() and (model.center_vectors.grad[center_word_index] != 0.).all(), \
        "Only batched center words can affect the center_word embedding."
    print("The second test passed!")

    # third test
    assert loss.detach().allclose(torch.tensor(26.86926651)), \
        "Loss of naive softmax do not match expected result."
    print("The third test passed!")

    # forth test
    expected_grad = torch.Tensor([[-0.07390384, -0.14989397,  0.03736909],
                                  [-0.00191219,  0.00386495, -0.00311787],
                                  [-0.00470913,  0.00072215,  0.00303244]])
    assert model.outside_vectors.grad[1:4, :].allclose(expected_grad), \
        "Gradients of naive softmax do not match expected result."
    print("The forth test passed!")

    print("All 4 tests passed!")

def test_neg_sampling_loss():
    print ("======Negative Sampling Loss Test Case======")
    center_word_index = torch.randint(1, 100, [5])
    outside_word_indices = []
    for _ in range(5):
        random_window_size = random.randint(3, 6)
        outside_word_indices.append([random.randint(1, 99) for _ in range(random_window_size)] + [0] * (6 - random_window_size))
    outside_word_indices = torch.Tensor(outside_word_indices).to(torch.long)

    neg_sampling_prob = torch.ones([100])
    neg_sampling_prob[0] = 0.

    dummy_database = type('dummy', (), {'_neg_sample_prob': neg_sampling_prob})

    sampled_negatives = list()
    def negative_sampler_wrapper(outside_word_indices, K):
        result = SkipgramDataset.negative_sampler(dummy_database, outside_word_indices, K)
        sampled_negatives.clear()
        sampled_negatives.append(result)
        return result

    model = NegSamplingWord2Vec(n_tokens=100, word_dimension=3, negative_sampler=negative_sampler_wrapper, K=5)

    loss = model(center_word_index, outside_word_indices).mean()
    loss.backward()

    # first test
    assert (model.center_vectors.grad[0, :] == 0).all() and (model.outside_vectors.grad[0, :] == 0).all(), \
        "<PAD> token should not affect the result."
    print("The first test passed! Howerver, this test dosen't guarantee you that <PAD> tokens really don't affects result.")    

    # Second test
    temp = model.center_vectors.grad.clone().detach()
    temp[center_word_index] = 0.
    assert (temp == 0.).all() and (model.center_vectors.grad[center_word_index] != 0.).all(), \
        "Only batched center words can affect the centerword embedding."
    print("The second test passed!")

    # Third test
    sampled_negatives = sampled_negatives[0]
    sampled_negatives[outside_word_indices.unsqueeze(-1).expand(-1, -1, 5) == 0] = 0
    affected_indices = list((set(sampled_negatives.flatten().tolist()) | set(outside_word_indices.flatten().tolist())) - {0})
    temp = model.outside_vectors.grad.clone().detach()
    temp[affected_indices] = 0.
    assert (temp == 0.).all() and (model.outside_vectors.grad[affected_indices] != 0.).all(), \
        "Only batched outside words and sampled negatives can affect the outside word embedding."
    print("The third test passed!")

    # forth test
    assert loss.detach().allclose(torch.tensor(35.82903290)), \
        "Loss of negative sampling do not match expected result."
    print("The forth test passed!")

    # fifth test
    expected_grad = torch.Tensor([[ 0.08583137, -0.40312022, -0.05952500],
                                  [ 0.14896543, -0.53478962, -0.18037169],
                                  [ 0.03650964,  0.24137473, -0.21831468]])
    assert model.outside_vectors.grad[affected_indices[:3], :].allclose(expected_grad), \
        "Gradient of negative sampling do not match expected result."
    print("The fifth test passed!")

    print("All 5 tests passed!")


if __name__ == "__main__":
    torch.set_printoptions(precision=8)
    torch.manual_seed(4321)
    random.seed(4321)

    test_naive_softmax_loss()
    test_neg_sampling_loss()
