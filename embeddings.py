import numpy as np

from typing import List, Union, Dict
from pathlib import Path

import os
import torch
import logging
import flair
import gensim

from flair.data import Sentence
from flair.embeddings.token import TokenEmbeddings
from flair.embeddings import StackedEmbeddings
from flair.embeddings import WordEmbeddings, CharacterEmbeddings
from flair.file_utils import cached_path

log = logging.getLogger("flair")


def get_embedding(embedding, finetune_bert=False):
    embeddings = embedding.split('+')
    result = [CaseEmbedding()]
    # skip updating to new flair version
    old_base_path = "https://flair.informatik.hu-berlin.de/resources/embeddings/token/"
    cache_dir = Path("embeddings")
    cached_path(f"{old_base_path}glove.gensim.vectors.npy", cache_dir=cache_dir)
    cached_path(
        f"{old_base_path}glove.gensim", cache_dir=cache_dir
    )

    cached_path(f"https://flair.informatik.hu-berlin.de/resources/characters/common_characters", cache_dir="datasets")

    for embedding in embeddings:
        if embedding == 'char':
            result.append(CustomCharacterEmbeddings())
        if embedding == 'glove':
            result.append(LargeGloveEmbeddings('./data/glove'))

    return StackedEmbeddings(embeddings=result)


class LargeGloveEmbeddings(WordEmbeddings):
    """Standard static word embeddings, such as GloVe or FastText."""

    def __init__(self, glove_dir):
        """
        Initializes classic word embeddings - made for large glove embedding
        """

        super().__init__('glove')
        embeddings = '840b-300d-glove'
        self.field = ""
        self.embeddings = embeddings
        self.static_embeddings = True

        # Large Glove embeddings
        embeddings = os.path.join(glove_dir, 'glove.bin')
        self.name: str = str(embeddings)
        self.precomputed_word_embeddings = gensim.models.KeyedVectors.load(embeddings)
        self.__embedding_length: int = self.precomputed_word_embeddings.vector_size

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for i, sentence in enumerate(sentences):
            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                word_embedding = self.get_cached_vec(word=token.text)
                token.set_embedding(self.name, word_embedding)

        return sentences

    @property
    def embedding_length(self) -> int:
        return 300


class CaseEmbedding(TokenEmbeddings):
    """Static Case Embedding"""

    def __init__(self):
        self.name: str = 'case-embedding-shun'
        self.static_embeddings = False
        self.__embedding_length: int = 3
        super().__init__()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def embed(self, sentences: Union[Sentence, List[Sentence]]) -> List[Sentence]:
        for sentence in sentences:
            for token in sentence:
                text = token.text
                is_lower = 1 if text == text.lower() else 0
                is_upper = 1 if text == text.upper() else 0
                is_mix = 1 if is_lower + is_upper == 0 else 0
                word_embedding = torch.tensor(
                    np.array([is_lower, is_upper, is_mix]), device=flair.device, dtype=torch.float
                )
                token.set_embedding('case-embedding-shun', word_embedding)

        return sentences

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        return sentences

    def __str__(self):
        return self.name


class CustomCharacterEmbeddings(CharacterEmbeddings):
    """Batched-version of CharacterEmbeddings. """

    def _add_embeddings_internal(self, sentences: List[Sentence]):

        token_to_embeddings = {}

        for sentence in sentences:
            for token in sentence.tokens:
                token_to_embeddings[token.text] = None

        tokens_char_indices = []
        for token in token_to_embeddings:
            char_indices = [
                self.char_dictionary.get_idx_for_item(char) for char in token
            ]
            tokens_char_indices.append(char_indices)

        # sort words by length, for batching and masking
        tokens_sorted_by_length = sorted(
            tokens_char_indices, key=lambda p: len(p), reverse=True
        )
        d = {}
        for i, ci in enumerate(tokens_char_indices):
            for j, cj in enumerate(tokens_sorted_by_length):
                if ci == cj:
                    d[j] = i
                    continue
        chars2_length = [len(c) for c in tokens_sorted_by_length]
        longest_token_in_sentence = max(chars2_length)
        tokens_mask = torch.zeros(
            (len(tokens_sorted_by_length), longest_token_in_sentence),
            dtype=torch.long,
            device=flair.device,
        )

        for i, c in enumerate(tokens_sorted_by_length):
            tokens_mask[i, : chars2_length[i]] = torch.tensor(
                c, dtype=torch.long, device=flair.device
            )

        # chars for rnn processing
        chars = tokens_mask

        character_embeddings = self.char_embedding(chars).transpose(0, 1)

        packed = torch.nn.utils.rnn.pack_padded_sequence(
            character_embeddings, chars2_length
        )

        lstm_out, self.hidden = self.char_rnn(packed)

        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
        outputs = outputs.transpose(0, 1)
        chars_embeds_temp = torch.zeros(
            (outputs.size(0), outputs.size(2)),
            dtype=torch.float,
            device=flair.device,
        )
        for i, index in enumerate(output_lengths):
            chars_embeds_temp[i] = outputs[i, index - 1]
        character_embeddings = chars_embeds_temp.clone()
        for i in range(character_embeddings.size(0)):
            character_embeddings[d[i]] = chars_embeds_temp[i]

        for token_number, token in enumerate(token_to_embeddings.keys()):
            token_to_embeddings[token] = character_embeddings[token_number]

        for sentence in sentences:
            for token in sentence.tokens:
                token.set_embedding(self.name, token_to_embeddings[token.text])
