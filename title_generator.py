import random

import pandas as pd
import spacy

get_nlp_engine = spacy.load('en_core_web_sm')


class BookTitleGenerator:
    def load(self):
        self._get_subtitles()
        self._get_title_token_bank_and_structures()
        self._get_subtitle_token_bank_and_structures()

    def get_reconstituted_title(self) -> str:
        titles = [
            random.choice(self._title_tokens[self._title_tokens.pos == pos].text.values)
            for pos in random.choice(self._title_structures)
        ]
        subtitles = [
            random.choice(self._subtitle_tokens[self._subtitle_tokens.pos == pos].text.values)
            for pos in random.choice(self._subtitle_structures)
        ]

        _add_spaces = lambda data: [f' {i}' if i.isalnum() else i for i in data]
        titles = _add_spaces(titles)
        subtitles = _add_spaces(subtitles)
        reconstituted_title = ': '.join(tuple(map(lambda x: ''.join(x).strip(), (titles, subtitles))))
        return reconstituted_title

    def _get_subtitles(self):
        self._just_titles = []
        self._subtitles = []
        for title in self._titles:
            sections = [i.strip() for i in title.replace('&', 'and').strip().split(':', 1)]
            self._just_titles.append(sections[0])
            if len(sections) > 1:
                self._subtitles.append(sections[1])

    def _get_title_token_bank_and_structures(self):
        self._title_structures, self._title_tokens = _get_token_bank_and_structures(self._just_titles)

    def _get_subtitle_token_bank_and_structures(self):
        self._subtitle_structures, self._subtitle_tokens = _get_token_bank_and_structures(self._subtitles)

    @property
    def _titles(self) -> list:
        titles = open('book_titles.txt').read().strip().splitlines()
        return titles


def _get_token_bank_and_structures(titles_or_subtitles: list) -> tuple:
    token_bank = []
    structures = []

    for i in titles_or_subtitles:
        doc = get_nlp_engine(i)
        if len(doc) <= 2:
            continue
        token_bank.extend((token.text, token.pos_) for token in doc)
        structures.append(tuple(token.pos_ for token in doc if token.pos_ != 'PUNCT'))

    tokens = pd.DataFrame(token_bank, columns=['text', 'pos'])
    tokens = tokens[tokens.pos != 'PUNCT'].copy()
    tokens.text = tokens.text.apply(lambda x: x.upper())
    return structures, tokens
