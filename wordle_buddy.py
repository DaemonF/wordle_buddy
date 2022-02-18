#!/usr/bin/env python3

import multiprocessing
import os
import sys

from argparse import ArgumentParser
from datetime import datetime
from enum import Enum
from functools import partial
from time import time
from tqdm import tqdm


# Characters used to display the score.
green_char = '!'
yellow_char = '?'
gray_char = 'x'

# Conversion to emoji for official output format.
score_char_to_emoji = {
  green_char: '\U0001F7E9',
  yellow_char: '\U0001F7E8',
  gray_char: '\U00002B1B',
}

# Letters from 'a' to 'z', for convenience.
letters = [chr(ordinal) for ordinal in range(ord('a'), ord('z') + 1)]

def occurrences(word, letter):
  return sum(1 for l in word if l == letter)

def fmt_real(stat):
  return f"{stat: >5.3f}"
def fmt_perc(stat):
  return f"{stat: >5.1%}"

def fmt_stats(items, fmt_item=fmt_perc):
  return ', '.join(fmt_item(i) for i in items)
def fmt_blocks(items, fmt_item=fmt_perc):
  return f"[ {' ][ '.join(fmt_item(i) for i in items)} ]"


class PropEnum(Enum):
  """Enum that allows properties that don't affect the enum identity."""
  def __new__(cls, *args, **kwds):
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj


class Game(PropEnum):
  """All games that can be played."""

  WORDLE = (
    'wordle', 'Wordle', 'wordlists/wordle_answers.txt',
    5, 6, datetime(2021, 6, 19),
    '*', ' on hard')
  JAYDLE = (
    'jaydle', 'Jaydle', 'wordlists/wordle_answers.txt',
    5, 6, datetime(2021, 6, 19),
    '*', ' on hard')
  LEWDLE = (
    'lewdle', 'Lewdle', 'wordlists/lewdle_answers.txt',
    5, 6, datetime(2022, 1, 17, hour=21),
    '', '')

  def __init__(self, name, display_name, default_dict_file, word_length, max_tries, epoch, mode_symbol, mode_desc):
    self.display_name = display_name
    self.default_dict_file = default_dict_file
    self.word_length = word_length
    self.max_tries = max_tries
    self.epoch = epoch
    self.mode_symbol = mode_symbol
    self.mode_desc = mode_desc

  def fmt_results(self, results):
    number = (datetime.now() - self.epoch).days
    return "\n".join((
      f"{self.display_name} {number} {len(results)}/{self.max_tries}{self.mode_symbol}",
      "",
      self._fmt_results_emoji(results),
      "",
      f"(Bot play{self.mode_desc})"
    ))

  def _fmt_results_emoji(self, results):
    return "\n".join(
      " ".join((
        "".join(score_char_to_emoji[c] for c in result['score']),
        f"{result['len_wordlist']} word{'s' if result['len_wordlist'] > 1 else ''}"
      )) for result in results
    )


class Strategy(PropEnum):
  """All strategies that can be used."""

  FREQ = 'freq', "positional letter frequency"
  CLUES = 'clues', "potential clue value"
  BIFUR = 'bifur', "maximum wordlist bifurcation"

  def __init__(self, name, description):
    self.description = description


class Guess:
  def __init__(self, game, wordlist, word):
    self.game = game
    self.wordlist = wordlist
    self.word = word

    self.score: list = [None for _ in range(self.game.word_length)]  # The actual result from the game.

  def compute_score(self, answer):
    if self.wordlist.scoring_table is not None:
      self.score = list(self.wordlist.scoring_table[self.word][answer])
      return self.score

    answer = list(answer)
    for index, letter in enumerate(self.word):
      if letter == answer[index]:
        self.score[index] = green_char
        answer[index] = None
      else:
        self.score[index] = gray_char
    for index, letter in enumerate(self.word):
      if self.score[index] != green_char and letter in answer:
        self.score[index] = yellow_char
        answer[answer.index(letter)] = None
    return self.score

  def __str__(self):
    stats = ', '.join(f"{strategy.value}: {fmt_real(self.wordlist.grade(self.word, strategy))}" for strategy in Strategy)
    return f"{self.word} ({stats})"


class LetterStats():
  def __init__(self, game):
    self.green_chance = [0 for _ in range(game.word_length)]
    self.yellow_chance = [0 for _ in range(game.word_length)]
    self.gray_chance = [0 for _ in range(game.word_length)]
    self.dupe_chance = [0 for _ in range(game.word_length)]

  def freeze(self):
    self.green_chance = tuple(self.green_chance)
    self.yellow_chance = tuple(self.yellow_chance)
    self.gray_chance = tuple(self.gray_chance)
    self.dupe_chance = tuple(self.dupe_chance)

class WordList(list):
  def __init__(self, game, wordlist, scoring_table=None):
    super(WordList, self).__init__(wordlist)
    self.game = game
    self.scoring_table = scoring_table

    self.stats = {letter:LetterStats(self.game) for letter in letters}
    for word in self:
      # Track whether a given guess would be green, yellow, or gray for this word.
      # TODO: This assumes all guesses are equally likely. Use wordlist?
      for index, letter in enumerate(word):
        for guess in letters:
          stats = self.stats[guess]
          if guess == letter:
            stats.green_chance[index] += 1
          elif guess in word:
            stats.yellow_chance[index] += 1
          else:
            stats.gray_chance[index] += 1
      # Track how often letters appear within a word.
      for letter in set(word):
        self.stats[letter].dupe_chance[occurrences(word, letter) - 1] += 1

    # Normalize
    total_words = len(self)
    for letter in letters:
      stats = self.stats[letter]
      total_occurrences = sum(stats.dupe_chance)
      for index in range(self.game.word_length):
        stats.green_chance[index] /= total_words
        stats.yellow_chance[index] /= total_words
        stats.gray_chance[index] /= total_words
        if total_occurrences > 0:
          stats.dupe_chance[index] /= total_occurrences
      stats.freeze()

  def sublist(self, scored_guess):
    '''Returns a new WordList by removing all incompatible words from this wordlist.
    '''
    if scored_guess.wordlist != self:
      raise ValueError(f"Guess is not associated with this wordlist.")
    filter = lambda w: True
    guess = scored_guess.word
    score = scored_guess.score
    for index, letter in enumerate(guess):
      if score[index] == green_char:
        filter = partial(lambda f,i,l,w: f(w) and
          w[i] == l,
          filter, index, letter)
      elif score[index] == yellow_char:
        at_least_count = sum(1 for j in range(self.game.word_length) if guess[j] == letter and score[j] in (yellow_char, green_char))
        filter = partial(lambda f,i,l,c,w: f(w) and
          w[i] != l and occurrences(w, l) >= c,
          filter, index, letter, at_least_count)
      elif score[index] == gray_char:
        at_most_count = occurrences(guess, letter) - sum(1 for j in range(self.game.word_length) if guess[j] == letter and score[j] == gray_char)
        filter = partial(lambda f,i,l,c,w: f(w) and
          w[i] != l and occurrences(w, l) <= c,
          filter, index, letter, at_most_count)
      else:
        raise ValueError(f"Unknown score character: '{score[index]}'.")
    return WordList(self.game, (w for w in self if filter(w)), self.scoring_table)

  def _build_scoring_table_part(self, word):
    guess = Guess(self.game, self, word)
    return (word, {
      answer: ''.join(guess.compute_score(answer)) for answer in self
    })

  def build_scoring_table(self):
    print("Building scoring table...")
    start = time()
    parallelism = os.cpu_count()
    chunk_size = 5
    with multiprocessing.Pool(parallelism) as pool:
      parts = pool.imap(self._build_scoring_table_part, self, chunk_size)
      self.scoring_table = { word: scores for word, scores in parts }
    stop = time()
    print(f"Built scoring table in {stop - start:.3f} seconds.")

  def grade(self, word, strategy: Strategy):
    if strategy == Strategy.FREQ:
      return self._grade_by_frequency(word)
    elif strategy == Strategy.CLUES:
      return self._grade_by_potential_clues(word)
    elif strategy == Strategy.BIFUR:
      return self._grade_by_bifurcation(word)

  def _grade_by_frequency(self, word):
    '''Grades a guess base on positional letter frequency in the wordlist.
    '''
    grade = 0
    for index, letter in enumerate(word):
      stats = self.stats[letter]
      grade += (stats.green_chance[index]
                * self._dupe_modifier(word, index, letter, stats)
               ) / self.game.word_length
    return grade

  def _grade_by_potential_clues(self, word):
    '''Grades a guess by how many potential clues it could give based on the wordlist.
    '''
    grade = 0
    for index, letter in enumerate(word):
      stats = self.stats[letter]
      # The number of words that would make this guess green, yellow, or gray.
      greens = stats.green_chance[index]
      yellows = stats.yellow_chance[index]
      grays = stats.gray_chance[index]

      # In Wordle, duplicate letters only count as yellow if the answer has the same or more duplicates. Model that.
      yellows *= self._dupe_modifier(word, index, letter, stats)
      # Only the first gray for a given letter matters.
      grays *= 1 if word.index(letter) == index else 0

      # Weight each category by how much it would split up the wordlist.
      green_weight = 1/2 - abs(1/2 - greens)
      yellow_weight = 1/2 - abs(1/2 - yellows)
      gray_weight = 1/2 - abs(1/2 - grays)

      grade += (greens * green_weight
                + yellows * yellow_weight
                + grays * gray_weight
               ) / self.game.word_length
    return grade

  def _grade_by_bifurcation(self, word):
    '''Grades a guess based on how closely it would split the wordlist in equal halves.
    '''
    buckets = {}
    guess = Guess(self.game, self, word)
    for answer in self:
      if self.scoring_table is not None:
        key = self.scoring_table[word][answer]
      else:
        key = ''.join(guess.compute_score(answer))
      buckets[key] = buckets.get(key, 0) + 1
    return len(self) / max(buckets.values())

  def _dupe_modifier(self, word, index, letter, stats):
    '''If the Guess contains duplicate letters, discount later occurrences based on the dupe chance.
    '''
    return (1 if word.index(letter) == index
      else sum(stats.dupe_chance[occurrences(word[:index], letter):]))


def show_stats_interactive(game, wordlist):
  for strategy in Strategy:
    print(f"By {strategy.description}:")
    for guess in sorted(wordlist, key=lambda word: wordlist.grade(word, strategy), reverse=True)[:5]:
      print(f"  {Guess(game, wordlist, guess)}")
    print()

  while True:
    entry = input("Enter a letter or word: ")
    if entry == "":
      break
    elif len(entry) == 1:
      stats = wordlist.stats[entry]

      print(f"  Appears anywhere in word: {fmt_perc(sum(stats.green_chance))}")
      print()
      print(f"  Positional chance of green:\n    {fmt_blocks(stats.green_chance)}")
      print(f"  Positional chance of yellow:\n    {fmt_blocks(stats.yellow_chance)}")
      print(f"  Positional chance of gray:\n    {fmt_blocks(stats.gray_chance)}")
      print()
      print(f"  Distribution of count per word it appears in:\n    {fmt_stats(stats.dupe_chance)}")
    elif len(entry) == game.word_length:
      print(f"  {Guess(game, wordlist, entry)}")
      if entry not in wordlist:
        print("  Note: Not in wordlist.")
    else:
      print(f"ERROR: Invalid length. Must be 1 or {game.word_length} characters.")
    print()

def play_game(game, wordlist, strategy, scoring_func, results=None, quiet=False):
  if not quiet:
    print(f"Strategy: {strategy.value}")

  results = []
  while True:
    if not quiet:
      print()
      print(f"List has {len(wordlist)} words: {', '.join(wordlist[:3])}")

    guess = Guess(game, wordlist, max(wordlist, key=lambda word: wordlist.grade(word, strategy)))
    if not quiet:
      print(f"Try: {guess}")

    scoring_func(guess)
    results.append({
      'score': guess.score,
      'len_wordlist': len(wordlist)
    })
    if occurrences(guess.score, green_char) == game.word_length:
      break

    wordlist = wordlist.sublist(guess)

  if not quiet:
    print()
    print()
    print(game.fmt_results(results))
  return results

def play_game_interactive(game, wordlist, strategy):
  def scoring_func(guess):
    while None in guess.score:
      resp = input("What was the score? ")
      if len(resp) == game.word_length:
        guess.score = list(resp)
  return play_game(game, wordlist, strategy, scoring_func)

def play_game_with_answer(game, wordlist, strategy, answer, quiet=False):
  if answer not in wordlist:
    if not quiet:
      print(f"'{answer}' is not in wordlist. Exiting...")
    return

  def scoring_func(guess):
    guess.compute_score(answer)
    if not quiet:
      print(f"Score: {''.join(guess.score)}")
  return play_game(game, wordlist, strategy, scoring_func, quiet=quiet)


class RegressionTest:
  def __init__(self, game, wordlist, strategy):
    self.game = game
    self.wordlist = wordlist
    self.strategy = strategy

  def __call__(self, answer):
    try:
      return play_game_with_answer(self.game, self.wordlist, self.strategy, answer, quiet=True)
    except:
      return answer


def regression_test(game, wordlist, strategy, sampling, answerlist):
  if answerlist is None:
    answerlist = wordlist
  if sampling != 1:
    answerlist = [answer for index, answer in enumerate(answerlist) if index % sampling == 0]

  games = len(answerlist)
  wins = [0 for _ in range(20)]
  crashes = []

  start = time()
  parallelism = os.cpu_count()
  chunk_size = 5
  with multiprocessing.Pool(parallelism) as pool:
    all_results = pool.imap(RegressionTest(game, wordlist, strategy), answerlist, chunk_size)

    for index, results in tqdm(enumerate(all_results), total=games):
      if type(results) is str:
        crashes.append(results)
      else:
        wins[len(results) - 1] += 1
  stop = time()

  print(f"Regression test")
  print(f"  List: {len(wordlist)} words")
  print(f"  Games: {games} games of {game.display_name}{'' if sampling == 1 else f' (sampling: 1/{sampling})'}")
  print(f"  Strategy: {strategy.value}")
  print()
  print(f"Stats of {games} games:")
  if len(crashes) > 0:
    print(f"  Crashes: {len(crashes)} {len(crashes) / games:.2%}")
    for crash in crashes:
      print(f"    {crash}")
  print(f"  Wins:")
  for index, count in enumerate(wins):
    def perc(n, d):
      return f"{n/d:.1%}"
    print(f"    {index+1:>3} {count:>4}  {perc(count, games):>6}  {perc(sum(wins[:index+1]), games):>6}")
  print()
  print(f"Total time: {stop - start:.3f} seconds (parallelism: {parallelism}, chunk size: {chunk_size}).")


def main():
  parser = ArgumentParser()
  parser.add_argument('--game', default='jaydle', type=Game)
  parser.add_argument('--strategy', default='bifur', type=Strategy)
  parser.add_argument('--dict_file', default=None)
  # Play game with known answer.
  parser.add_argument('--answer')
  # Play game interactively.
  parser.add_argument('-i', dest='mode', action='store_const', const='interactive')
  # Run a regression test.
  parser.add_argument('-t', dest='mode', action='store_const', const='test')
  parser.add_argument('--sampling', default=1, type=int)
  parser.add_argument('--answer_file', default=None)
  args = parser.parse_args()

  game = args.game
  dict_file = args.dict_file if args.dict_file is not None else game.default_dict_file
  wordlist = None
  with open(dict_file, 'r') as f:
    wordlist = WordList(game, (entry.strip() for entry in f.readlines() if len(entry.strip()) == game.word_length))

  answerlist = None
  if args.answer_file is not None:
    answerlist = []
    with open(args.answer_file, 'r') as f:
      for l in f.readlines():
        entry = l.strip()
        if len(entry) == game.word_length:
          answerlist.append(entry)

  strategy = args.strategy
  if strategy == Strategy.BIFUR:
    # TODO: Currently, this massively slows down the other strategies, which is
    # unexpexcted. It seems related to the use of Multiprocessing. Investigate.
    wordlist.build_scoring_table()

  if args.mode == 'test':
    regression_test(game, wordlist, strategy, args.sampling, answerlist)
  elif args.mode == 'interactive':
    play_game_interactive(game, wordlist, strategy)
  elif args.answer is not None:
    play_game_with_answer(game, wordlist, strategy, args.answer)
  else:
    print("Showing wordlist and starting-word stats...")
    print()
    show_stats_interactive(game, wordlist)


if __name__ == '__main__':
  main()
