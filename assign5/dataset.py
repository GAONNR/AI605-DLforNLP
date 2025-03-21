from typing import List, Tuple, Any, Dict, Union
from multiprocessing import Process
import json

# YOUR LIBRARIES HERE

# END YOUR LIBRARIES

from transformers import BertTokenizerFast
from tqdm import trange

# You may implment your own classes HERE


class SquadData(object):
    def __init__(self, context: str, qa: dict):
        self.id_str: str = qa['id']
        self.context: str = context
        self.question = str = qa['question']
        self.answer = str = qa['answers'][0]['text']
        self.start_pos = int = qa['answers'][0]['answer_start']

# END YOUR CLASSES


class SquadDataset(object):
    """ Squad Dataset
    Implement the interface for the Standford Question Answerting Dataset 
    """

    def __init__(self, json_file: str = 'data/train-v1.1-TA.json'):
        """ Squad Dataset Initializer
        You can use this part as you like.
        Load the given json file properly and provide a simple interface.

        Arguments:
        json_file -- the name of the json file which have to be processed.
        """
        # YOUR CODE HERE
        f: Any = open(json_file, 'r')
        json_data: dict = json.load(f)

        self.version: str = json_data['version']
        json_data = json_data['data']

        self.data: List[SquadData] = list()

        for title_para in json_data:
            paragraphs = title_para['paragraphs']
            for paragraph in paragraphs:
                context = paragraph['context']
                qas = paragraph['qas']

                self.data += list([SquadData(context, qa) for qa in qas])

        f.close()
        # END YOUR CODE

    def __len__(self) -> int:
        """ Squad Dataset Length
        Return:
        length -- length of the dataset
        """
        # YOUR CODE HERE
        length: int = len(self.data)
        # END YOUR CODE

        return length

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """ Squad Dataset Indexing
        Arguments:
        index -- The index of the question
                 Each question should have a unique index number in the order.

        Return:
        id_str -- The id of the question, which is saved in the json file.
                  You can find it when you check the file structure.
                  This id is needed when evaluating your model.
        context -- The corresponding context of the question.
        question -- The question
        answer -- The corresponding answer of the question.
                  Just select the first answer of the answers list.
        start_pos -- The position of the answer in terms of character indexing level.
                     This information is also saved in the file in raw.
        """
        # YOUR CODE HERE
        item: SquadData = self.data[index]
        id_str: str = item.id_str
        context: str = item.context
        question: str = item.question
        answer: str = item.answer
        start_pos: int = item.start_pos

        # END YOUR CODE

        sample: Dict[str, Any] = {
            'id': id_str,
            'context': context,
            'question': question,
            'answer': answer,
            'start_pos': start_pos
        }

        return sample


def squad_features(
    context: str,
    question: str,
    answer: Union[str, None],
    start_char_pos: Union[int, None],
    tokenizer: BertTokenizerFast
) -> Tuple[List[int], List[int], int, int]:
    """ Squad feature extractor
    Implement the feature extractor from a Squad sample for your model
    Return values should follow [CLS + question + SEP + context + SEP] form.
    In addition, because start_char_pos is based on character index, you should convert it to proper token index.
    Check the test cases to know the functionality in detail.

    Note: input_ids and token_type_ids follows the transfomer library documentation 
    https://huggingface.co/transformers/glossary.html

    Arguments:
    context -- Context string
    question -- Question string
    anwser -- Answer string. If the answer is None, return None for start_token_pos and end_token_pos
    start_char_pos -- Character index which the answer starts from in the context.
                      If the answer is None, this argument is also None.
    tokenizer -- Tokenizer to encode text strings.
                 Explanation: https://huggingface.co/transformers/model_doc/bert.html#berttokenizerfast

    Returns:
    input_ids -- Input ids
    token_type_ids -- Token type ids 
    start_token_pos -- Token index which the answer starts from in the input_ids list. 
                       None if no answer is given.
    end_token_pos -- Token index which the answer ends by in the input_ids list.
                     This includes the last token which located in the index.
                     None if no answer is given.
    """
    # YOUR CODE HERE (~18 lines)
    encoded: dict = tokenizer.encode_plus(question, context)
    input_ids: List[int] = None
    token_type_ids: List[int] = None
    start_token_pos: int = None
    end_token_pos: int = None

    encoded_question: List[int] = tokenizer.encode(
        question, add_special_tokens=False)
    decoded_question: List[str] = tokenizer.convert_ids_to_tokens(
        encoded_question)
    encoded_context: List[int] = tokenizer.encode(
        context, add_special_tokens=False)
    decoded_context: List[str] = tokenizer.convert_ids_to_tokens(
        encoded_context)

    if answer:
        # find answer position
        curr_tok_idx = 0
        curr_char_idx = 0
        start_token_pos = 0
        end_char_pos = start_char_pos + len(answer)

        while curr_char_idx < len(context):
            if curr_char_idx == start_char_pos:
                start_token_pos = curr_tok_idx
            if curr_char_idx >= end_char_pos:
                end_token_pos = curr_tok_idx
                break

            real_curr_token = decoded_context[curr_tok_idx]
            if real_curr_token[:2] == '##':
                real_curr_token = real_curr_token[2:]

            if context[curr_char_idx:curr_char_idx + len(real_curr_token)].lower() == real_curr_token:
                curr_char_idx += len(real_curr_token)
                curr_tok_idx += 1
            else:
                curr_char_idx += 1

        if curr_char_idx >= len(context):
            end_token_pos = curr_tok_idx
        else:
            if curr_char_idx > end_char_pos:
                front = decoded_context[:end_token_pos - 1]
                back = decoded_context[end_token_pos:]
                curr = decoded_context[end_token_pos - 1]

                diff = curr_char_idx - end_char_pos

                new_curr = [curr[: -1 * diff], '##%s' % curr[-1 * diff:]]

                decoded_context = front + new_curr + back

        start_token_pos += 2 + len(encoded_question)
        end_token_pos += 1 + len(encoded_question)

    pretokenized_str = ['[CLS]'] + decoded_question + \
        ['[SEP]'] + decoded_context + ['[SEP]']

    input_ids = tokenizer.convert_tokens_to_ids(pretokenized_str)
    token_type_ids = [0] * (2 + len(decoded_question)) + \
        [1] * (1 + len(decoded_context))
    # END YOUR CODE

    return input_ids, token_type_ids, start_token_pos, end_token_pos


########################################################
# Helper functions below. DO NOT MODIFY!               #
# Read helper classes to implement your code properly! #
########################################################

class SquadFeatureDataset(object):
    """ Squad Feature Dataset
    The wrapper class for the squad_feature function
    """

    def __init__(self, dataset: SquadDataset, bert_type: str, lazy=False, return_sample=False, eval=False):
        self.dataset = dataset
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_type)
        self.return_sample = return_sample
        self.eval = eval

        if not lazy:
            self.lazy = True
            self.dataset = [self[index] for index in trange(
                0, len(self.dataset), desc="Preprocessing")]

        self.lazy = lazy

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        if not self.lazy:
            return self.dataset[index]

        sample = self.dataset[index]
        context = sample['context']
        question = sample['question']
        answer = sample['answer']
        start_pos = sample['start_pos']

        if self.eval:
            out = squad_features(context, question, None, None, self.tokenizer)
        else:
            out = squad_features(context, question, answer,
                                 start_pos, self.tokenizer)

        if self.return_sample:
            out = (out, sample)

        return out

#############################################
# Testing functions below.                  #
#############################################


def test_wrapper(dataset):
    [dataset[sample_id]['id'] for sample_id in range(0, len(dataset))]


def test_squad_dataset(dataset):
    print("======Squad Dataset Test Case======")
    # First test
    assert len(dataset) == 87474, \
        "Your dataset have some missing or duplicated samples."
    print("The first test passed!")

    # Second test
    expected = {
        'id': '5733be284776f41900661182',
        'context': 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.',
        'question': 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?',
        'answer': 'Saint Bernadette Soubirous',
        'start_pos': 515
    }
    assert dataset[0] == expected, \
        "Your indexing form does not match the expected form."
    print("The second test passed!")

    # Third test
    expected = {
        'id': '57302039b2c2fd14005688db',
        'context': 'On February 29, 2012, Microsoft released Windows 8 Consumer Preview, the beta version of Windows 8, build 8250. Alongside other changes, the build removed the Start button from the taskbar for the first time since its debut on Windows 95; according to Windows manager Chaitanya Sareen, the Start button was removed to reflect their view that on Windows 8, the desktop was an "app" itself, and not the primary interface of the operating system. Windows president Steven Sinofsky said more than 100,000 changes had been made since the developer version went public. The day after its release, Windows 8 Consumer Preview had been downloaded over one million times. Like the Developer Preview, the Consumer Preview expired on January 15, 2013.',
        'question': 'When was the beta version of Windows 8 made available to the public?',
        'answer': 'February 29, 2012',
        'start_pos': 3
    }
    assert dataset[80000] == expected, \
        "Your indexing result does not match the expedted result."
    print("The third test passed!")

    result = True

    process = Process(target=test_wrapper, args=(dataset,))
    process.start()
    process.join(timeout=1)

    if process.is_alive():
        process.terminate()
        process.join()
        result = False

    # Forth test
    assert result, \
        "Your indexing is too slow."
    print("The forth test passed!")

    # Fifth test
    assert not process.exitcode, \
        "An error occured when indexing your dataset."
    print("The fifth test passed!")

    print("All 5 tests passed!")


def test_squad_feature_extractor(dataset):
    print("======Squad Feature Test Case======")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # First test
    context = 'This is a sample context. BERT will find the answer words in the context by pointing the start and end token positions.'
    question = 'Where are the answer words?'
    answer = 'in the context'
    start_pos = context.find(answer)
    input_ids, token_type_ids, start_pos, end_pos = squad_features(
        context, question, answer, start_pos, tokenizer)

    assert tokenizer.convert_ids_to_tokens(input_ids) == \
        ['[CLS]', 'where', 'are', 'the', 'answer', 'words', '?', '[SEP]',
         'this', 'is', 'a', 'sample', 'context', '.',
         'bert', 'will', 'find', 'the', 'answer', 'words', 'in', 'the', 'context',
         'by', 'pointing', 'the', 'start', 'and', 'end', 'token', 'positions', '.', '[SEP]'], \
        "Your tokenized result does not match the expected result."

    assert token_type_ids == \
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], \
        "Your sentence type ids do not math the expected result"

    assert tokenizer.convert_ids_to_tokens(input_ids[start_pos: end_pos+1]) == ['in', 'the', 'context'], \
        "The start and end tokens do not point the answer position."

    print("The first test passed!")

    # Second test
    context = 'Sometimes, the answer could be subwords so you may need to split them manually.'
    question = 'What should the answer consist of'
    answer = 'word'
    start_pos = context.find(answer)
    input_ids, token_type_ids, start_pos, end_pos = squad_features(
        context, question, answer, start_pos, tokenizer)

    assert tokenizer.convert_ids_to_tokens(input_ids) == \
        ['[CLS]', 'what', 'should', 'the', 'answer', 'consist', 'of', '[SEP]',
         'sometimes', ',', 'the', 'answer', 'could', 'be', 'sub', '##word', '##s',
         'so', 'you', 'may', 'need', 'to', 'split', 'them', 'manually', '.', '[SEP]'], \
        "Your tokenized result does not match the expected result."

    assert token_type_ids == \
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], \
        "Your sentence type ids do not math the expected result"

    assert tokenizer.convert_ids_to_tokens(input_ids[start_pos: end_pos+1]) == ['##word'], \
        "The start and end tokens do not point the answer position."

    print("The second test passed!")

    # Third test
    context = 'When the answer is not given, you should return None for start_pos and end_pos.'
    question = 'This test case does not need a question'
    input_ids, token_type_ids, start_pos, end_pos = squad_features(
        context, question, None, None, tokenizer)

    assert len(input_ids) == 33, \
        "Your tokenized result does not match the expected result."

    assert start_pos is None and end_pos is None, \
        "You should return None for start_pos and end_pos when the answer is not given."

    print("The third test passed!")

    # Forth test
    sample = dataset[0]
    context = sample['context']
    question = sample['question']
    answer = sample['answer']
    start_pos = sample['start_pos']

    input_ids, token_type_ids, start_pos, end_pos = squad_features(
        context, question, answer, start_pos, tokenizer)

    assert len(input_ids) == 176, \
        "Your tokenized result does not match the expected result."

    assert tokenizer.convert_ids_to_tokens(input_ids[start_pos: end_pos+1]) == tokenizer.tokenize(answer), \
        "The start and end tokens do not point the answer position."

    print("The forth test passed!")

    # Fifth test
    sample = dataset[80000]
    context = sample['context']
    question = sample['question']
    answer = sample['answer']
    start_pos = sample['start_pos']

    input_ids, token_type_ids, start_pos, end_pos = squad_features(
        context, question, answer, start_pos, tokenizer)

    assert len(input_ids) == 165, \
        "Your tokenized result does not match the expected result."

    assert tokenizer.convert_ids_to_tokens(input_ids[start_pos: end_pos+1]) == tokenizer.tokenize(answer), \
        "The start and end tokens do not point the answer position."

    print("The fifth test passed!")

    print("All 5 tests passed!")


def test_squad_feature_dataset(dataset):
    print("======Squad Feature Dataset Test Case======")
    dataset = SquadFeatureDataset(
        dataset, bert_type='bert-base-uncased', lazy=True)

    input_ids, token_type_ids, start_pos, end_pos = dataset[0]
    print("input_ids:", input_ids)
    print("token_type_ids:", token_type_ids)
    print("start_pos:", start_pos)
    print("end_pos:", end_pos)

    print("The test passed!")


if __name__ == "__main__":
    dataset = SquadDataset()

    test_squad_dataset(dataset)
    test_squad_feature_extractor(dataset)
    test_squad_feature_dataset(dataset)
