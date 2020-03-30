from typing import List, Dict, Tuple 
from random import randint

### You may import any Python standard library here
import re
### END YOUR LIBRARIES

import torch
from torch.utils.data import IterableDataset

def tokenize(
    sentence: str
) -> List[str]:
    """ Sentence Tokenizer

    Implement simple sentence tokenizer
    Punctuation marks should be a seperate token: . , ! ?

    Note: You can import any Python standard library during this assignment,
    but do not use other external libraries except torch (such as nltk.)
    For this tokenizing function, 're' library might be helpful but it is not mandatory.

    Example: 'Don\'t be fooled, but be clever.'
    ==> ['Don\'t', 'be', 'fooled', ',', 'but', 'be', 'clever', '.']

    Arguments:
    sentence -- The sentence to be tokenized

    Return:
    tokens -- The list of tokens
    """

    ### YOUR CODE HERE (~3 lines, this is not a mandatory requirement, but try to make efficent codes)
    tokens: List[str] = re.findall('([.,!?]|[\'\+\-\w\d]+)', sentence)
    ### END YOUR CODE

    return tokens

def build_vocab(
    sentences: List[List[str]],
    min_freq: int
) -> Tuple[List[str], Dict[str, int], List[int]]:
    """ Vocabulary Builder

    Implement vocab builder that makes word2idx and idx2word from sentences.
    Words with too few frequencies can cause overfitting, so we will replace these words to <UNK> tokens.
    Because negative sampling needs the freqency of each word, you will calculate this also in here.
    <PAD> token will be used later for batching, but don't mind it now.

    Hint: Counter in collection library would be helpful

    Arguments:
    sentences -- The list of sentence to build vocab
    min_freq -- The minimum frequency of a word to be a seperate token.
                A word whose frequency is less than this number should be treated as <UNK> token.

    Return:
    idx2word -- A list which takes a index and gives its matched word 
    word2idx -- A dictionary which maps a word to its indices
    word_freq -- The list of the number of appearance count of each word through entire sentences.
                 The frequency of <UNK> token should be calculated properly.
    """

    # Special tokens, insert these tokens properly
    PAD = SkipgramDataset.PAD_TOKEN
    PAD_idx = SkipgramDataset.PAD_TOKEN_IDX # The index of <PAD> must be 0
    UNK = SkipgramDataset.UNK_TOKEN
    UNK_idx = SkipgramDataset.UNK_TOKEN_IDX # The index of <UNK> must be 1

    ### YOUR CODE HERE (~5 lines)
    idx2word: List[str] = [PAD, UNK]
    word2idx: Dict[str, int] = {PAD: PAD_idx, UNK: UNK_idx}
    word_freq: List[int] = [0, 0]

    ### END YOUR CODE

    assert idx2word[PAD_idx] == PAD and word2idx[PAD] == PAD_idx, \
        "PAD token should be placed properly"
    assert idx2word[UNK_idx] == UNK and word2idx[UNK] == UNK_idx, \
        "UNK token should be placed properly"
    assert len(idx2word) == len(word2idx) and len(idx2word) == len(word_freq), \
        "Size of idx2word, word2idx and word_freq should be same"
    return idx2word, word2idx, word_freq

def skipgram(
    sentence: List[str],
    window_size: int,
    center_word_loc: int
) -> Tuple[str, List[str]]:
    """ Function to generate a (center_word, outside_words) pair for skipgram

    Implement the function that generate a (center_word, outside_words) pair from the given sentence and the location of center word.

    Argument:
    sentence -- A sentence where the center word and the outside words come from
    window_size -- Integer, context window size
    center_word_loc -- The location of the center word within the sentence

    Return:
    center_word -- String, a center word
    outside_words -- List of string, words within the window centered on the center word.
                     outside_words dose not include center_word.
                     The number of outside words could be less than 2 * window_size.
    """

    ### YOUR CODE HERE (~3 lines)
    center_word: str = ""
    outside_words: List[str] = list()

    ### END YOUR CODE
    return center_word, outside_words

#############################################
# Helper functions below. DO NOT MODIFY!    #
#############################################

class SkipgramDataset(IterableDataset):
    PAD_TOKEN = '<PAD>'
    PAD_TOKEN_IDX = 0
    UNK_TOKEN = '<UNK>'
    UNK_TOKEN_IDX = 1
    
    def __init__(self, path="sentences.txt", window_size=2, min_freq=2, device=torch.device('cpu')):
        self._window_size = window_size

        with open(path, "r") as f:
            sentences = [tokenize(line.strip().lower()) for line in f]

        idx2word, word2idx, word_freq = build_vocab(sentences, min_freq=min_freq)

        self._sentences = sentences
        self._idx2word = idx2word
        self._word2idx = word2idx
        self._neg_sample_prob = torch.Tensor(word_freq).to(device) ** .75

    @property
    def n_tokens(self):
        return len(self._idx2word)

    def negative_sampler(self, outside_word_indices, K):
        indices = outside_word_indices.flatten()

        if outside_word_indices.device == torch.device('cpu'):
            negatives = []
            for index in indices:
                temp = self._neg_sample_prob[index].clone()
                self._neg_sample_prob[index] = 0.
                negatives.append(torch.multinomial(self._neg_sample_prob, num_samples=K, replacement=True))
                self._neg_sample_prob[index] = temp
            negatives = torch.stack(negatives)

        else:
            probs = self._neg_sample_prob.repeat(indices.shape[0], 1)
            probs.scatter_(dim=-1, index=indices.unsqueeze(-1), src=outside_word_indices.new_zeros([]))
            negatives = torch.distributions.categorical.Categorical(probs).sample([K]).T

        return negatives.reshape(list(outside_word_indices.shape) + [K]).detach()
    
    def idx2word(self, index: int) -> str:
        return self._idx2word[index]

    def word2idx(self, word: str) -> int:
        return self._word2idx[word]

    def __iter__(self):
        while True:
            index = randint(0, len(self._sentences)-1)
            sentence = self._sentences[index]
            center_word, outside_words = skipgram(sentence, self._window_size, randint(0, len(sentence)-1))

            center_word_index = self._word2idx.get(center_word, SkipgramDataset.UNK_TOKEN_IDX)
            outside_word_indices = list(map(lambda word: self._word2idx.get(word, SkipgramDataset.UNK_TOKEN_IDX), outside_words)) 
            outside_word_indices += [SkipgramDataset.PAD_TOKEN_IDX] * (self._window_size * 2 - len(outside_word_indices))

            yield center_word_index, torch.Tensor(outside_word_indices).to(torch.long)
    
#############################################
# Testing functions below.                  #
#############################################

def test_tokenizer():
    print ("======Tokenizer Test Cases======")

    # First test
    sentence = "This sentence should be tokenized properly."
    tokens = tokenize(sentence)
    assert tokens == ['This', 'sentence', 'should', 'be', 'tokenized', 'properly', '.'], \
        "Your tokenized list do not match expected result"
    print("The first test passed!")

    # Second test
    sentence = "Jhon's book is not popular, but he loves his book."
    tokens = tokenize(sentence)
    assert tokens == ["Jhon's", "book", "is", "not", "popular", ",", "but", "he", "loves", "his", "book", "."], \
        "Your tokenized list do not match expected result"
    print("The second test passed!")

    # Third test
    sentence = "  .,! ?,,'-4.  ! "
    tokens = tokenize(sentence)
    assert tokens == ['.', ',', '!', '?', ',', ',', "'-4", '.', '!'], \
        "Your tokenized list do not match expected result"
    print("The third test passed!")
    
    print("All 3 tests passed!")

def test_vocab_builder():
    print ("======Vocabulary Builder Test Cases======")

    # First test
    sentences = [["This", "sentence", "be", "tokenized", "propery", "."],
                 ["Jhon", "'s", "book", "is", "not", "popular", ",", "but", "he", "loves", "his", "book", "."]]
    
    idx2word, word2idx, word_freq = build_vocab(sentences, min_freq=1)
    assert sentences == [[idx2word[word2idx[word]] for word in sentence] for sentence in sentences], \
        "Your word2idx and idx2word do not show consistency"
    print("The first test passed!")

    # Second test
    sentences = [['a', 'a', 'b'], ['b', 'c']]

    idx2word, word2idx, word_freq = build_vocab(sentences, min_freq=1)
    assert len(idx2word) == 5 and \
        word_freq[word2idx[SkipgramDataset.UNK_TOKEN]] == 0 and word_freq[word2idx[SkipgramDataset.PAD_TOKEN]] == 0 and \
        word_freq[word2idx['a']] == 2 and word_freq[word2idx['b']] == 2 and word_freq[word2idx['c']] == 1, \
        "Result of word_freq do not match expected result"
    print("The second test passed!")

    # Third test
    sentences = [["a", "b", "c", "d", "e"],
                 ["c", "d", "f", "g"],
                 ["d", "e", "g", "h"]]

    idx2word, word2idx, word_freq = build_vocab(sentences, min_freq=2)
    assert set(word2idx.keys()) == {'<PAD>', '<UNK>', 'c', 'd', 'e', 'g'} and len(idx2word) == 6 and \
        word_freq[word2idx['c']] == 2 and word_freq[word2idx['d']] == 3 and word_freq[word2idx['e']] == 2 and word_freq[word2idx['g']] == 2 and \
        word_freq[word2idx[SkipgramDataset.UNK_TOKEN]] == 4 and word_freq[word2idx[SkipgramDataset.PAD_TOKEN]] == 0, \
        "Your vocabulary builder does not work with min_freq argument"
    print("The third test passed!")
    
    print("All 3 tests passed!")

def test_skipgram():
    print ("======Skipgram Test Cases======")

    # first test
    sentence = ["Jhon's", "book", "is", "not", "popular", ",", "but", "he", "loves", "his", "book", "."]
    center_word, outside_words = skipgram(sentence, window_size=2, center_word_loc=3)
    assert center_word == 'not' and len(outside_words) == 4 and set(outside_words) == {"book", "is", "popular", ","}, \
        "Your skipgram does not work"
    print("The first test passed!")

    # second test
    sentence = ["Jhon's", "book", "is", "not", "popular", ",", "but", "he", "loves", "his", "book", "."]
    center_word, outside_words = skipgram(sentence, window_size=3, center_word_loc=1)
    assert center_word == 'book' and len(outside_words) == 4 and set(outside_words) == {"Jhon's", "is", "not", "popular"}, \
        "Your skipgram does not work when the front of window is cut out"
    print("The second test passed!")

    # third test
    sentence = ["Jhon's", "book", "is", "not", "popular", ",", "but", "he", "loves", "his", "book", "."]
    center_word, outside_words = skipgram(sentence, window_size=2, center_word_loc=11)
    assert center_word == '.' and len(outside_words) == 2 and set(outside_words) == {"his", "book"}, \
        "Your skipgram does not work when the rear of window is cut out"
    print("The third test passed!")

    # forth test
    sentence = ["Jhon's", "book", "is", "popular", "."]
    center_word, outside_words = skipgram(sentence, window_size=3, center_word_loc=2)
    assert center_word == 'is' and len(outside_words) == 4 and set(outside_words) == {"Jhon's", "book", "popular", "."}, \
        "Your skipgram does not work when the both side of window is cut out"
    print("The forth test passed!")

    print("All 4 tests passed!")

def test_dataset():
    print ("======Database Test======")

    dataset = SkipgramDataset()
    center_word_index, outside_word_indices = next(iter(dataset))
    negative_samples = dataset.negative_sampler(outside_word_indices, 5)

    print("Sampled center word index is %d" % center_word_index)
    print("Sampled outside word indices are {}".format(outside_word_indices.tolist()))
    print("Sampled negative indices with K=5 are {}".format(negative_samples.tolist()))

    print("Database test passed!")

if __name__ == "__main__":
    test_tokenizer()
    test_vocab_builder()
    test_skipgram()
    test_dataset()