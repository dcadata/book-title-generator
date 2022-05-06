import random

import pandas as pd
import spacy

get_nlp_engine = spacy.load('en_core_web_sm')


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
        title_components = self._choose_random(self._title_tokens, self._title_structures)
        subtitle_components = self._choose_random(self._subtitle_tokens, self._subtitle_structures)
        reconstituted_title = ': '.join(tuple(map(lambda x: ''.join(self._add_spaces(x)).strip(), (
            title_components, subtitle_components))))
        return reconstituted_title


def _split_subtitles() -> tuple[list, list]:
    just_titles = []
    subtitles = []
    titles = open('data/titles.txt').read().strip().splitlines()

    for title in titles:
        sections = [i.strip() for i in title.replace('&', 'and').strip().split(':', 1)]
        just_titles.append(sections[0])
        if len(sections) > 1:
            subtitles.append(sections[1])

    return just_titles, subtitles


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


def tag_and_save():
    just_titles, subtitles = _split_subtitles()
    _save_tokens_and_structures(just_titles, 'titles')
    _save_tokens_and_structures(subtitles, 'subtitles')


def _read_structures_from_disk(label: str) -> list:
    df = pd.read_csv(f'data/structures_{label}.csv', dtype=str)
    structures = [tuple(i for i in record if i and pd.notna(i)) for record in df.to_records(index=False)]
    return structures
