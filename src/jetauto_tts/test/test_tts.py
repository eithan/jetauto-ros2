"""Unit tests for the TTS node sentence building logic."""

import pytest


class TestBuildSentence:
    """Test TTSNode._build_sentence behavior via a standalone implementation."""

    def test_single(self):
        assert _build('cup') == 'I can see a cup'

    def test_pair(self):
        assert _build('cup', 'laptop') == 'I can see a cup and a laptop'

    def test_triple(self):
        assert _build('cup', 'laptop', 'phone') == (
            'I can see a cup, a laptop, and a phone'
        )

    def test_vowel_article(self):
        assert _build('apple') == 'I can see an apple'
        assert _build('elephant') == 'I can see an elephant'
        assert _build('umbrella') == 'I can see an umbrella'

    def test_consonant_article(self):
        assert _build('banana') == 'I can see a banana'

    def test_empty(self):
        assert _build() == ''


class TestWithArticle:
    """Test article selection."""

    def test_vowel_start(self):
        assert _with_article('orange') == 'an orange'

    def test_consonant_start(self):
        assert _with_article('table') == 'a table'

    def test_uppercase(self):
        assert _with_article('Apple') == 'an Apple'

    def test_empty_string(self):
        assert _with_article('') == 'an unknown object'


# -- Standalone helpers matching the node methods --

def _with_article(label: str) -> str:
    if not label:
        return 'an unknown object'
    article = 'an' if label[0].lower() in 'aeiou' else 'a'
    return f'{article} {label}'


def _build(*labels, greeting='I can see') -> str:
    if not labels:
        return ''
    items = [_with_article(l) for l in labels]
    if len(items) == 1:
        return f'{greeting} {items[0]}'
    elif len(items) == 2:
        return f'{greeting} {items[0]} and {items[1]}'
    else:
        return f'{greeting} {", ".join(items[:-1])}, and {items[-1]}'
