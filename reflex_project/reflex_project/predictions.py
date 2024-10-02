import pandas as pd
import numpy as np
import re
import random

from typing import List, Dict, Union
from collections import defaultdict, Counter

UNK, EOS = "[UNK]", "[EOS]"
PREDICTION_LENGHT_CONSTANT = 3

def clean_email(email_text):

    cleaned_text = re.sub(
        r'^(Message-ID|Date|From|Sent|To|Subject|RE|Mime-Version|Content-Type|Content-Transfer-Encoding|X-[^\n]*|X-From|X-To|X-cc|X-bcc|X-Folder|X-Origin|X-FileName).*\n?',
        '',
        email_text,
        flags=re.MULTILINE,
    )

    cleaned_text = re.sub(r'[*-]', '', cleaned_text)

    cleaned_text = re.sub(r'\d+', '', cleaned_text)

    cleaned_text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', '', cleaned_text)

    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
    cleaned_text = re.sub(r'^\s+', '', cleaned_text, flags=re.MULTILINE)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = re.sub(r'(?<=Subject:)\s+', '', cleaned_text)

    cleaned_text.replace('Message-ID', '')
    cleaned_text.replace('RE', '')
    cleaned_text.replace('Content-Type', '')
    cleaned_text.replace('Content-Transfer-Encoding', '')
    cleaned_text.replace('X-FileName', '')
    cleaned_text.replace('X-Folder', '')
    cleaned_text.replace('X-Origin', '')
    cleaned_text.replace('X-cc', '')
    cleaned_text.replace('X-To', '')
    return cleaned_text

def tokenize(text):
    reg = re.compile(r'\w+')
    return reg.findall(text.lower())

from typing import List

class PrefixTreeNode:
    def __init__(self, prefix_value=''):
        # словарь с буквами, которые могут идти после данной вершины
        self.children: dict[str, PrefixTreeNode] = {}
        self.is_end_of_word = False
        self.word = prefix_value

class PrefixTree:
    def __init__(self, vocabulary: List[str], tree_limit=10000):
        """
        vocabulary: список всех уникальных токенов в корпусе
        """
        self.root = PrefixTreeNode()
        self.tree_limit = tree_limit
        for word in vocabulary:
          self.add_word(word)
          if self.tree_limit <= 0:
            return

    def add_word(self, word: str):
      cur_node = self.root
      for idx, letter in enumerate(word):
        if letter not in cur_node.children:
          cur_node.children[letter] = PrefixTreeNode(prefix_value=word[:idx + 1])
        cur_node = cur_node.children[letter]
      cur_node.is_end_of_word = True
      self.tree_limit -= 1

    def search_words(
        self,
        cur_node: PrefixTreeNode,
        ans_arr: List[str],
      ):
      if cur_node.is_end_of_word:
        ans_arr.append(cur_node.word)

      if len(cur_node.children) == 0:
        return

      for letter, node in cur_node.children.items():
        self.search_words(node, ans_arr)


    def search_prefix(self, prefix) -> List[str]:
        """
        Возвращает все слова, начинающиеся на prefix
        prefix: str – префикс слова
        """
        cur_node = self.root
        for letter in prefix:
          if letter in cur_node.children:
            cur_node = cur_node.children[letter]
          else:
            print("Letter is not in vocabulary")

        ans_arr: List[str] = []
        self.search_words(cur_node, ans_arr)

        return ans_arr

from typing import Dict, Union
from itertools import chain

class WordCompletor:
    def __init__(self, corpus):
        """
        corpus: list – корпус текстов
        """
        # your code here
        # self.vocabulary = np.concatenate(corpus)
        self.vocabulary = list(chain.from_iterable(corpus))
        self.prefix_tree = PrefixTree(list(chain.from_iterable(corpus)))
        self.count_dct = self.construct_count_dct()

    def construct_count_dct(self) -> Dict[str, int]:
      count_dct: Dict[str, float] = {}
      for word in self.vocabulary:
        if word in count_dct:
          count_dct[word] += 1
        else:
          count_dct[word] = 1

      return count_dct

    def get_words_and_probs(self, prefix: str) -> (List[str], List[float]):
        """
        Возвращает список слов, начинающихся на prefix,
        с их вероятностями (нормировать ничего не нужно)
        """
        words = self.prefix_tree.search_prefix(prefix)
        probs = [self.count_dct[word] / len(self.vocabulary) for word in words]
        return words, probs

class NGramLanguageModel:
    def __init__(self, corpus, n):
        counts = self.count_ngrams(corpus, n, tokenize=tokenize)
        self.n = n

        self.probs = defaultdict(Counter)

        for prefix, token_count in counts.items():
            token_sum = sum(token_count.values())
            for token, count in token_count.items():
                self.probs[prefix][token] = count / token_sum
    
    def count_ngrams(self, lines, n, tokenize=tokenize):
        """
        Count how many times each word occured after (n - 1) previous words
        Input: a list of strings with space-separated tokens
        :returns: a dictionary { tuple(prefix_tokens): {next_token_1: count_1, next_token_2: count_2}}

        If the prefix is too short, it should be padded with [UNK].
        Add [EOS] at the end of each sequence and consider it as all other token
        """
        counts = defaultdict(Counter)

        for line in lines:
            line = ' '.join(line)
            tokenized = [UNK] * (n) + tokenize(line) + [EOS]
            for i in range(n - 1, len(tokenized)):
                # print(tokenized, '   :   ', tokenized[i-n:i], '     :      ', tokenized[i])
                counts[tuple(tokenized[i-n:i])][tokenized[i]] += 1

        return counts
    

    def process_prefix(self, prefix):
        if self.n == 1:
            prefix = []
        else:
            prefix = prefix[-(self.n):]
            prefix = [UNK] * (self.n - len(prefix)) + prefix
            
        return prefix

    def get_next_words_and_probs(self, prefix: list) -> (List[str], List[float]):
        """
        Возвращает список слов, которые могут идти после prefix,
        а так же список вероятностей этих слов
        """
        # print("PrePrefix: ", prefix)
        prefix = self.process_prefix(prefix)
        # print("Prefix: ", prefix)

        possible_tokens = self.probs[tuple(prefix)]
        # print(possible_tokens)

        next_words = list(possible_tokens.keys())
        probs = list(possible_tokens.values())

        return next_words, probs

class TextSuggestion:
    def __init__(self):
        # with open(', 'rb') as file:     # используется файл полученный на предыдущем шаге
        #     obj = dill.load(file)/model.pik'
        emails = pd.read_csv('../required_data/emails.csv')
        email_texts = emails['message'][:10000]
        corpus = [tokenize(clean_email(t).lower()) for t in email_texts]
        self.word_completor = WordCompletor(corpus)
        self.n_gram_model = NGramLanguageModel(corpus=corpus, n=PREDICTION_LENGHT_CONSTANT)
    
    def complete_word(self, current_word: str):

        words, probs = self.word_completor.get_words_and_probs(current_word)
        # probs.append(0)
        # words.append(UNK)
        if len(probs) == 0:
            words, probs = self.word_completor.get_words_and_probs('')
    
        # print(probs, " --- ")
        word_prediction_ = words[max(enumerate(probs), key=lambda x: x[1])[0]]
        # print("current - ", word_prediction_)
        return word_prediction_

    def suggest_next_word(self, n_prefix: List[str]):        
        next_words, probs = self.n_gram_model.get_next_words_and_probs(n_prefix)
        # probs.append(0)
        # next_words.append(UNK)
        # print(probs, " --vre- ")
        if len(probs) == 0:
            next_words, probs = self.word_completor.get_words_and_probs('')
        # print(probs, " --vre- ", max(enumerate(probs), key=lambda x: x[1]))
            random.shuffle(probs)
        word_prediction = next_words[max(enumerate(probs), key=lambda x: x[1])[0]]
        # print("next - ", word_prediction)
        return word_prediction

    def suggest_text(self, text: List[str], n_words=3, n_texts=1) -> list[list[str]]:
        """
        Возвращает возможные варианты продолжения текста (по умолчанию только один)

        text: строка или список слов – написанный пользователем текст
        n_words: число слов, которые дописывает n-граммная модель
        n_texts: число возвращаемых продолжений (пока что только одно)

        return: list[list[srt]] – список из n_texts списков слов, по 1 + n_words слов в каждом
        Первое слово – это то, которое WordCompletor дополнил до целого.
        """
        suggestions = []
        # print("VALUE: ", text, " : ", type(text))

        if len(text) < self.n_gram_model.n:
            tmp = ([UNK] * (self.n_gram_model.n - len(text)))
            tmp.extend(text)
            text = tmp
        
        # print(self.n_gram_model.n, "---------",len(text))
        # print([UNK] * (self.n_gram_model.n - len(text) + 1), "---------",([UNK] * (self.n_gram_model.n - len(text))).extend(text))
        # print(text)

        for _ in range(n_texts):
            suggestions_for_text = [self.complete_word(text[-1].lower())]
            n_prefix = text[-self.n_gram_model.n:]
            for _ in range(n_words):
                print("n_ : ",self.n_gram_model.n ,  " | n_Text: ", n_prefix, " | Text: ", text)
                next_word = self.suggest_next_word(n_prefix)
                suggestions_for_text.append(next_word)
                n_prefix.append(next_word)
                n_prefix = n_prefix[-self.n_gram_model.n:]


            suggestions.append(suggestions_for_text)
        
        return suggestions