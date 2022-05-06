import random

import pandas as pd
import spacy

get_nlp_engine = spacy.load('en_core_web_sm')


class Tagger:
    def run(self):
        self._split_subtitles()
        self._save_tokens_and_structures()

    def _split_subtitles(self):
        self._just_titles = []
        self._subtitles = []
        for title in self._titles:
            sections = [i.strip() for i in title.replace('&', 'and').strip().split(':', 1)]
            self._just_titles.append(sections[0])
            if len(sections) > 1:
                self._subtitles.append(sections[1])

    def _save_tokens_and_structures(self):
        _save_tokens_and_structures(self._just_titles, 'titles')
        _save_tokens_and_structures(self._subtitles, 'subtitles')

    @property
    def _titles(self) -> list:
        titles = open('titles.txt').read().strip().splitlines()
        return titles


class BookTitleGenerator:
    def __init__(self):
        self._add_spaces = lambda data: [f' {i}' if i.isalnum() else i for i in data]
        self._choose_random = lambda tokens, structures: [
            random.choice(tokens[tokens.pos == pos].text.values) for pos in random.choice(structures)]

        self._title_tokens = pd.read_csv('data/tokens_titles.csv', dtype=str)
        self._subtitle_tokens = pd.read_csv('data/tokens_subtitles.csv', dtype=str)
        self._title_structures = _read_structures_from_disk('titles')
        self._subtitle_structures = _read_structures_from_disk('subtitles')

    def get_reconstituted_title(self) -> str:
        title_components = [
            random.choice(self._title_tokens[self._title_tokens.pos == pos].text.values)
            for pos in random.choice(self._title_structures)
        ]
        subtitle_components = [
            random.choice(self._subtitle_tokens[self._subtitle_tokens.pos == pos].text.values)
            for pos in random.choice(self._subtitle_structures)
        ]
        reconstituted_title = ': '.join(tuple(map(lambda x: ''.join(self._add_spaces(x)).strip(), (
            title_components, subtitle_components))))
        return reconstituted_title


def _save_tokens_and_structures(titles_or_subtitles: list, label: str):
    token_bank = []
    structures = []

    for i in titles_or_subtitles:
        doc = get_nlp_engine(i)
        if len(doc) <= 2:
            continue
        token_bank.extend((token.text, token.pos_) for token in doc)
        structures.append(tuple(token.pos_ for token in doc if token.pos_ != 'PUNCT'))

    structures = pd.DataFrame(structures)
    tokens = pd.DataFrame(token_bank, columns=['text', 'pos'])
    tokens = tokens[tokens.pos != 'PUNCT'].copy()
    tokens.text = tokens.text.apply(lambda x: x.upper())

    structures.to_csv(f'data/structures_{label}.csv', index=False)
    tokens.to_csv(f'data/tokens_{label}.csv', index=False)


def _read_structures_from_disk(label: str) -> list:
    df = pd.read_csv(f'data/structures_{label}.csv', dtype=str)
    structures = [tuple(i for i in record if i and pd.notna(i)) for record in df.to_records(index=False)]
    return structures
