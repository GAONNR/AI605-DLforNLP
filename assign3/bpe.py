from typing import List, Dict, Set
from itertools import chain

# You can import any Python standard libraries here.
# Do not import external library such as numpy, torchtext, etc.
import re
from collections import Counter, defaultdict
# END YOUR LIBRARIES


def build_bpe(
    corpus: List[str],
    max_vocab_size: int
) -> List[int]:
    """ BPE Vocabulary Builder
    Implement vocabulary builder for byte pair encoding.
    Please sort your idx2word by subword length in decsending manner.

    Hint: Counter in collection library would be helpful

    Note: If you convert sentences list to word frequence dictionary,
          building speed is enhanced significantly because duplicated words are preprocessed together

    Arguments:
    corpus -- List of words to build vocab
    max_vocab_size -- The maximum size of vocab

    Return:
    idx2word -- Subword list
    """
    # Special tokens
    PAD = BytePairEncoding.PAD_token  # Index of <PAD> must be 0
    UNK = BytePairEncoding.UNK_token  # Index of <UNK> must be 1
    CLS = BytePairEncoding.CLS_token  # Index of <CLS> must be 2
    SEP = BytePairEncoding.SEP_token  # Index of <SEP> must be 3
    MSK = BytePairEncoding.MSK_token  # Index of <MSK> must be 4
    SPECIAL = [PAD, UNK, CLS, SEP, MSK]

    WORD_END = BytePairEncoding.WORD_END  # Use this token as the end of a word

    # YOUR CODE HERE (~22 lines)
    real_corpus = [x for x in corpus]
    idx2word: List[str] = SPECIAL
    words: Counter = Counter(
        [' '.join(list(x)) + ' ' + WORD_END for x in real_corpus])

    initial_words: Set = set()
    for word in real_corpus:
        initial_words.update(word)
    initial_words.add(WORD_END)

    subwords: List[str] = list(initial_words)
    while len(subwords) < max_vocab_size - len(SPECIAL):
        pairs: defaultdict = defaultdict(int)
        for word, freq in words.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq

        if len(pairs) == 0:
            break

        max_freq_pair = max(pairs, key=pairs.get)
        new_subword = ''.join(max_freq_pair)

        new_words: Counter = Counter()
        bigram = r'(?!\s)(%s %s)(?!\S)' % (max_freq_pair[0].replace(
            '.', '\.'), max_freq_pair[1].replace('.', '\.'))
        for word in words:
            new_words[re.sub(bigram, new_subword, word)] = words[word]

        if words == new_words:
            break
        words = new_words
        subwords.append(new_subword)
    subwords.sort(key=len, reverse=True)
    idx2word += subwords
    # END YOUR CODE

    return idx2word


def encode(
    sentence: List[str],
    idx2word: List[str]
) -> List[int]:
    """ BPE encoder
    Implement byte pair encoder which takes a sentence and gives the encoded tokens

    Arguments:
    sentence -- The list of words which need to be encoded.
    idx2word -- The vocab that you have made on the above build_bpe function.

    Return:
    tokens -- The list of the encoded tokens
    """
    WORD_END = BytePairEncoding.WORD_END

    # YOUR CODE HERE (~10 lines)
    tokens: List[int] = list()
    modified_sentence: List[str] = [word + WORD_END for word in sentence]

    for word in modified_sentence:
        candidates: List[List[int]] = [list() for x in word]
        for curr in range(0, len(word)):
            for past in range(curr + 1):
                subword = word[past:curr + 1]
                if subword in idx2word[5:]:
                    if past > 0 and len(candidates[past - 1]) == 0:
                        continue
                    candidate = [idx2word.index(
                        subword)] if past == 0 else candidates[past - 1] + [idx2word.index(subword)]
                    if len(candidate) < len(candidates[curr]) or len(candidates[curr]) == 0:
                        candidates[curr] = candidate
        tokens += candidates[-1]
    # END YOUR CODE

    return tokens


def decode(
    tokens: List[int],
    idx2word: List[str]
) -> List[str]:
    """ BPE decoder
    Implement byte pair decoder which takes tokens and gives the decoded sentence.

    Arguments:
    tokens -- The list of tokens which need to be decoded
    idx2word -- the vocab that you have made on the above build_bpe function.

    Return:
    sentence  -- The list of the decoded words
    """
    WORD_END = BytePairEncoding.WORD_END

    # YOUR CODE HERE (~1 lines)
    sentence: List[str] = list()
    for token in tokens:
        sentence.append(idx2word[token])
    sentence = ''.join(sentence).split(WORD_END)[:-1]
    # END YOUR CODE
    return sentence


#############################################
# Helper functions below. DO NOT MODIFY!    #
#############################################

class BytePairEncoding(object):
    """ Byte Pair Encoding class
    We aren't gonna use this class for encoding. Because it is too slow......
    We will use sentence piece Google have made.
    Thus, this class is just for special token index reference.
    """
    PAD_token = '<pad>'
    PAD_token_idx = 0
    UNK_token = '<unk>'
    UNK_token_idx = 1
    CLS_token = '<cls>'
    CLS_token_idx = 2
    SEP_token = '<sep>'
    SEP_token_idx = 3
    MSK_token = '<msk>'
    MSK_token_idx = 4

    WORD_END = '_'

    def __init__(self, corpus: List[List[str]], max_vocab_size: int) -> None:
        self.idx2word = build_bpe(corpus, max_vocab_size)

    def encode(self, sentence: List[str]) -> List[int]:
        return encode(sentence, self.idx2word)

    def decoder(self, tokens: List[int]) -> List[str]:
        return decode(tokens, self.idx2word)

#############################################
# Testing functions below.                  #
#############################################


def test_build_bpe():
    print("======Building BPE Vocab Test Case======")
    PAD = BytePairEncoding.PAD_token
    UNK = BytePairEncoding.UNK_token
    CLS = BytePairEncoding.CLS_token
    SEP = BytePairEncoding.SEP_token
    MSK = BytePairEncoding.MSK_token
    WORD_END = BytePairEncoding.WORD_END

    # First test
    corpus = ['abcde']
    vocab = build_bpe(corpus, max_vocab_size=15)
    assert vocab[:5] == [PAD, UNK, CLS, SEP, MSK], \
        "Please insert the special tokens properly"
    print("The first test passed!")

    # Second test
    assert sorted(vocab[5:], key=len, reverse=True) == vocab[5:], \
        "Please sort your idx2word by subword length in decsending manner."
    print("The second test passed!")

    # Third test
    corpus = ['low'] * 5 + ['lower'] * 2 + ['newest'] * 6 + ['widest'] * 3
    vocab = set(build_bpe(corpus, max_vocab_size=24))
    assert vocab > {PAD, UNK, CLS, SEP, MSK, 'est_', 'low', 'newest_',
                    'i', 'e', 'n', 't', 'd', 's', 'o', 'l', 'r', 'w', WORD_END} and \
        "low_" not in vocab and "wi" not in vocab and "id" not in vocab, \
        "Your bpe result does not match expected result"
    print("The third test passed!")

    # forth test
    corpus = ['aaaaaaaaaaaa', 'abababab']
    vocab = set(build_bpe(corpus, max_vocab_size=13))
    assert vocab == {PAD, UNK, CLS, SEP, MSK, 'aaaaaaaa', 'aaaa', 'abab', 'aa', 'ab', 'a', 'b', WORD_END}, \
        "Your bpe result does not match expected result"
    print("The forth test passed!")

    # fifth test
    corpus = ['abc', 'bcd']
    vocab = build_bpe(corpus, max_vocab_size=10000)
    assert len(vocab) == 15, \
        "Your bpe result does not match expected result"
    print("The fifth test passed!")

    print("All 5 tests passed!")


def test_encoding():
    print("======Encoding Test Case======")
    PAD = BytePairEncoding.PAD_token
    UNK = BytePairEncoding.UNK_token
    CLS = BytePairEncoding.CLS_token
    SEP = BytePairEncoding.SEP_token
    MSK = BytePairEncoding.MSK_token
    SPECIAL = [PAD, UNK, CLS, SEP, MSK]
    WORD_END = BytePairEncoding.WORD_END

    # First test
    vocab = SPECIAL + ['bcc', 'bb', 'bc', 'a', 'b', 'c', WORD_END]
    assert encode(['abbccc'], vocab) == [8, 9, 5, 10, 11], \
        "Your bpe encoding does not math expected result"
    print("The first test passed!")

    # Second test
    vocab = SPECIAL + ['aaaa', 'aa', 'a', WORD_END]
    assert len(encode(['aaaaaaaa', 'aaaaaaa'], vocab)) == 7, \
        "Your bpe encoding does not math expected result"
    print("The second test passed!")

    print("All 2 tests passed!")


def test_decoding():
    print("======Decoding Test Case======")
    PAD = BytePairEncoding.PAD_token
    UNK = BytePairEncoding.UNK_token
    CLS = BytePairEncoding.CLS_token
    SEP = BytePairEncoding.SEP_token
    MSK = BytePairEncoding.MSK_token
    SPECIAL = [PAD, UNK, CLS, SEP, MSK]
    WORD_END = BytePairEncoding.WORD_END

    # First test
    vocab = SPECIAL + ['bcc', 'bb', 'bc', 'a', 'b', 'c', WORD_END]
    assert decode([8, 9, 5, 10, 11], vocab) == ['abbccc'], \
        "Your bpe decoding does not math expected result"
    print("The first test passed!")

    # Second test
    vocab = SPECIAL + ['aaaa', 'aa', 'a', WORD_END]
    assert decode([5, 5, 8, 5, 6, 7, 8], vocab) == ['aaaaaaaa', 'aaaaaaa'], \
        "Your BPE decoding does not math expected result"
    print("The second test passed!")


def test_consistency():
    print("======Consistency Test Case======")
    corpus = ['this is test corpus .',
              'we will check the consistency of your byte pairing encoding .',
              'you have to pass this test to get full scores .',
              'we hope you to pass tests wihtout any problem .',
              'good luck .']

    vocab = build_bpe(chain.from_iterable(sentence.split()
                                          for sentence in corpus), 80)

    sentence = 'this is another sentence to test encoding and decoding .'.split()

    assert decode(encode(sentence, vocab), vocab) == sentence, \
        "Your BPE does not show consistency."
    print("The consistency test passed!")


if __name__ == "__main__":
    test_build_bpe()
    test_encoding()
    test_decoding()
    test_consistency()
