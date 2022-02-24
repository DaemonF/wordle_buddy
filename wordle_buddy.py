#!/usr/bin/env python3

import argparse
import cProfile as profile
import multiprocessing
import pickle
import pstats
import os
import sys

from datetime import datetime
from enum import Enum
from ordered_set import OrderedSet
from time import time, perf_counter_ns
from tqdm import tqdm


# Characters used to display the score.
green_char = "!"
yellow_char = "?"
gray_char = "x"

# Conversion to emoji for official output format.
score_char_to_emoji = {
  green_char: "\U0001F7E9",
  yellow_char: "\U0001F7E8",
  gray_char: "\U00002B1B",
}

# Letters from 'a' to 'z', for convenience.
letters = tuple(
  chr(ordinal) for ordinal in range(ord("a"), ord("z") + 1)
)


def inds(score):
  i = 0
  for s in score:
    i *= 3
    if s == gray_char:
      i += 0
    elif s == yellow_char:
      i += 1
    else:
      i += 2
  return i


def occurrences(word, letter):
  return sum(1 for l in word if l == letter)


def fmt_real(stat):
  return f"{stat: >5.3f}"


def fmt_perc(stat):
  return f"{stat: >5.1%}"


def fmt_stats(items, fmt_item=fmt_perc):
  return ", ".join(fmt_item(i) for i in items)


def fmt_blocks(items, fmt_item=fmt_perc):
  return f"[ {' ][ '.join(fmt_item(i) for i in items)} ]"


class PoolFunc:
  def __init__(self, func, *args):
    self.func = pickle.dumps(func)
    self.args = ()
    self.set_up = False

  def partial(self, *args):
    assert not self.set_up
    self.args = tuple(pickle.dumps(arg) for arg in args)
    return self

  def setup(self):
    global func, args, set_up
    func = pickle.loads(self.func)
    args = tuple(pickle.loads(arg) for arg in self.args)
    set_up = self.set_up = True

  @staticmethod
  def __call__(arg):
    global func, args, set_up
    assert set_up
    return func(*args, arg)


class PropEnum(Enum):
  """Enum that allows properties that don't affect the enum identity."""

  def __new__(cls, *args, **kwds):
    obj = object.__new__(cls)
    obj._value_ = args[0]
    return obj


class Game(PropEnum):
  """All games that can be played."""

  WORDLE = (
    "wordle",
    "Wordle",
    "wordlists/wordle_answers.txt",
    5,
    6,
    datetime(2021, 6, 19),
    False,
    "",
    "",
  )
  WORDLE_HARD = (
    "wordle_hard",
    "Wordle (hard mode)",
    "wordlists/wordle_answers.txt",
    5,
    6,
    datetime(2021, 6, 19),
    True,
    "*",
    " on hard",
  )
  JAYDLE = (
    "jaydle",
    "Jaydle",
    "wordlists/wordle_answers.txt",
    5,
    6,
    datetime(2021, 6, 19),
    False,
    "",
    "",
  )
  JAYDLE_HARD = (
    "jaydle_hard",
    "Jaydle (hard mode)",
    "wordlists/wordle_answers.txt",
    5,
    6,
    datetime(2021, 6, 19),
    True,
    "*",
    " on hard",
  )
  LEWDLE = (
    "lewdle",
    "Lewdle",
    "wordlists/lewdle_answers.txt",
    5,
    6,
    datetime(2022, 1, 17, hour=21),
    False,
    "",
    "",
  )

  def __init__(
    self,
    name,
    display_name,
    default_dict_file,
    word_length,
    max_tries,
    epoch,
    hard_mode,
    mode_symbol,
    mode_desc,
  ):
    self.display_name = display_name
    self.default_dict_file = default_dict_file
    self.word_length = word_length
    self.max_tries = max_tries
    self.epoch = epoch
    self.hard_mode = hard_mode
    self.mode_symbol = mode_symbol
    self.mode_desc = mode_desc

  def fmt_results(self, results):
    def _fmt_result(result):
      score = "".join(
        score_char_to_emoji[c] for c in result["score"]
      )
      words = result["len_wordlist"]
      return f"{score} {words} word{'' if words == 1 else 's'}"

    number = (datetime.now() - self.epoch).days
    tries = len(results)
    return "\n".join(
      (
        f"{self.display_name} {number} {tries}/{self.max_tries}{self.mode_symbol}",
        "",
        *(_fmt_result(result) for result in results),
        "",
        f"(Bot play{self.mode_desc})",
      )
    )


class Strategy(PropEnum):
  """All strategies that can be used."""

  FREQ = "freq", "positional letter frequency"
  CLUES = "clues", "potential clue value"
  BIFUR = "bifur", "maximum wordlist bifurcation"

  def __init__(self, name, description):
    self.description = description


class Guess:
  def __init__(self, word, wordlist):
    self.word = word
    self.wordlist = wordlist

  def __str__(self):
    stats = ", ".join(
      f"{strategy.value}: {fmt_real(self.wordlist.grade(self.word, strategy))}"
      for strategy in Strategy
    )
    return f"{self.word} ({stats})"


class LetterStats:
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


class WordList(OrderedSet):
  def __init__(self, wordlist, game, scoring_table=None):
    super(WordList, self).__init__(wordlist)
    self.game = game
    self.scoring_table = scoring_table

    self._score = [None for _ in range(game.word_length)]

    self.stats = {
      letter: LetterStats(self.game) for letter in letters
    }
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
        self.stats[letter].dupe_chance[
          occurrences(word, letter) - 1
        ] += 1

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

  def __getstate__(self):
    return (
      list(self),
      self.game,
      self.scoring_table,
      self._score,
      self.stats,
    )

  def __setstate__(self, state):
    super(WordList, self).__init__(state[0])
    (
      _,
      self.game,
      self.scoring_table,
      self._score,
      self.stats,
    ) = state

  def sublist(self, guess, score):
    """Returns a new WordList by removing all incompatible words from this wordlist."""
    f = lambda word: True
    for i, (g, s) in enumerate(zip(guess, score)):
      if s == green_char:
        f = lambda word, i=i, g=g, f=f: (
          word[i] == g and f(word)
        )
      elif s == yellow_char:
        count = sum(
          1
          for g2, s2 in zip(guess, score)
          if g2 == g and s2 != gray_char
        )
        f = lambda word, i=i, g=g, c=count, f=f: (
          word[i] != g and f(word) and occurrences(word, g) >= c
        )

      elif s == gray_char:
        count = sum(
          1
          for g2, s2 in zip(guess, score)
          if g2 == g and s2 != gray_char
        )
        f = lambda word, i=i, g=g, c=count, f=f: (
          word[i] != g and f(word) and occurrences(word, g) <= c
        )

      else:
        assert False

    return WordList(
      (word for word in self if f(word)),
      self.game,
      self.scoring_table,
    )

  def compute_score(self, guess, answer):
    """WARNING: Return value is mutable for performance reasons. Must use or copy before calling this method again."""
    answer = list(answer)
    for index, letter in enumerate(guess):
      if letter == answer[index]:
        self._score[index] = green_char
        answer[index] = None
      else:
        self._score[index] = gray_char
    for index, letter in enumerate(guess):
      if self._score[index] != green_char and letter in answer:
        self._score[index] = yellow_char
        answer[answer.index(letter)] = None
    return self._score

  @staticmethod
  def build_scoring_table_part(wordlist, word):
    return (
      word,
      {
        answer: inds(wordlist.compute_score(word, answer))
        for answer in wordlist
      },
    )

  def build_scoring_table(self):
    print("Building scoring table...")
    start = time()
    func = PoolFunc(self.build_scoring_table_part).partial(self)
    with multiprocessing.Pool(initializer=func.setup) as pool:
      parts = pool.imap(func, self, chunksize=20)
      self.scoring_table = {
        word: scores for word, scores in parts
      }
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
    """Grades a guess base on positional letter frequency in the wordlist."""
    grade = 0
    for index, letter in enumerate(word):
      stats = self.stats[letter]
      grade += (
        stats.green_chance[index]
        * self._dupe_modifier(word, index, letter, stats)
      ) / self.game.word_length

    # For hard mode, make words in the wordlist slightly more attractive.
    modifier = -0.01 if word not in self else 0
    return grade + modifier

  def _grade_by_potential_clues(self, word):
    """Grades a guess by how many potential clues it could give based on the wordlist."""
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
      green_weight = 0.5 - abs(0.5 - greens)
      yellow_weight = 0.5 - abs(0.5 - yellows)
      gray_weight = 0.5 - abs(0.5 - grays)

      grade += (
        greens * green_weight
        + yellows * yellow_weight
        + grays * gray_weight
      ) / self.game.word_length

    # For hard mode, make words in the wordlist slightly more attractive.
    modifier = -0.01 if word not in self else 0
    return grade + modifier

  def _grade_by_bifurcation(self, word):
    """Grades a guess based on how closely it would split the wordlist in equal halves."""
    buckets = [0 for _ in range(3 ** self.game.word_length)]
    if self.scoring_table is not None:
      scores = self.scoring_table[word]
      for answer in self:
        buckets[scores[answer]] += 1
    else:
      guess = Guess(word, self)
      for answer in self:
        buckets[inds(self.compute_score(word, answer))] += 1

    # For hard mode, make words in the wordlist slightly more attractive.
    modifier = 0.5 if word not in self else 0
    return len(self) / (max(buckets) + modifier)

  def _dupe_modifier(self, word, index, letter, stats):
    """If the Guess contains duplicate letters, discount later occurrences based on the dupe chance."""
    return (
      1
      if word.index(letter) == index
      else sum(
        stats.dupe_chance[occurrences(word[:index], letter) :]
      )
    )


def show_stats_interactive(game, wordlist):
  for strategy in Strategy:
    print(f"By {strategy.description}:")
    for guess in sorted(
    wordlist,
    key=lambda word: wordlist.grade(word, strategy),
    reverse=True,
  )[:5]:
      print(f"  {Guess(guess, wordlist)}")
    print()

  while True:
    entry = input("Enter a letter or word: ")
    if entry == "":
      break
    elif len(entry) == 1:
      stats = wordlist.stats[entry]

      print(
        f"  Appears anywhere in word: {fmt_perc(sum(stats.green_chance))}"
      )
      print()
      print("  Positional chance of green:")
      print(f"    {fmt_blocks(stats.green_chance)}")
      print("  Positional chance of yellow:")
      print(f"    {fmt_blocks(stats.yellow_chance)}")
      print("  Positional chance of gray:")
      print(f"    {fmt_blocks(stats.gray_chance)}")
      print()
      print("  Distribution of count per word it appears in:")
      print(f"    {fmt_stats(stats.dupe_chance)}")
    elif len(entry) == game.word_length:
      print(f"  {Guess(entry, wordlist)}")
      if entry not in wordlist:
        print("  Note: Not in wordlist.")
    else:
      print(
        f"ERROR: Invalid length. Must be 1 or {game.word_length} characters."
      )
    print()


def play_game(
  game,
  wordlist,
  strategy,
  scoring_func,
  results=None,
  quiet=False,
):
  if not quiet:
    print(f"Strategy: {strategy.value}")

  guesslist = wordlist
  results = []
  while True:
    if not quiet:
      words = ""
      for index, word in enumerate(wordlist):
        if index >= 3:
          words += "..."
          break
        words += ("" if index == 0 else ", ") + word
      print()
      print(f"List has {len(wordlist)} words: {words}")
    guess = max(
      guesslist,
      key=lambda word: wordlist.grade(word, strategy),
    )
    if not quiet:
      print(f"Try: {guess}")

    score = scoring_func(guess)
    results.append(
      {"score": tuple(score), "len_wordlist": len(wordlist)}
    )
    if occurrences(score, green_char) == game.word_length:
      break

    wordlist = wordlist.sublist(guess, score)
    if game.hard_mode:
      guesslist = wordlist

  if not quiet:
    print()
    print()
    print(game.fmt_results(results))
  return results


def play_game_interactive(game, wordlist, strategy):
  def scoring_func(guess):
    while True:
      resp = input("What was the score? ")
      if len(resp) == game.word_length:
        return list(resp)

  return play_game(game, wordlist, strategy, scoring_func)


def play_game_with_answer(
  game, wordlist, strategy, answer, quiet=False
):
  if answer not in wordlist:
    if not quiet:
      print(f"'{answer}' is not in wordlist. Exiting...")
    return

  def scoring_func(guess):
    score = wordlist.compute_score(guess, answer)
    if not quiet:
      print(f"Score: {''.join(score)}")
    return score

  return play_game(
    game, wordlist, strategy, scoring_func, quiet=quiet
  )


def regression_test_case(wordlist, strategy, answer):
  try:
    return play_game_with_answer(
      wordlist.game, wordlist, strategy, answer, quiet=True
    )
  except:
    return answer


def regression_test(
  game, wordlist, strategy, sampling, answerlist
):
  if answerlist is None:
    answerlist = wordlist
  if sampling != 1:
    answerlist = [
      answer
      for index, answer in enumerate(answerlist)
      if index % sampling == 0
    ]

  games = len(answerlist)
  wins = [0 for _ in range(10)]
  crashes = []

  start = time()
  func = PoolFunc(regression_test_case).partial(
    wordlist, strategy
  )
  parallelism = os.cpu_count()
  with multiprocessing.Pool(
    parallelism, initializer=func.setup
  ) as pool:
    all_results = pool.imap(func, answerlist)
    for index, results in tqdm(
      enumerate(all_results), total=games
    ):
      if type(results) is str:
        crashes.append(results)
      else:
        wins[len(results) - 1] += 1
  stop = time()

  print(f"Regression test")
  print(f"  List: {len(wordlist)} words")
  print(
    f"  Games: {games} games of {game.display_name}{'' if sampling == 1 else f' (sampling: 1/{sampling})'}"
  )
  print(f"  Strategy: {strategy.value}")
  print()
  print(f"Stats of {games} games:")
  if len(crashes) > 0:
    print(
      f"  Crashes: {len(crashes)} {len(crashes) / games:.2%}"
    )
    for crash in crashes:
      print(f"    {crash}")
  print(f"  Wins:")
  for index, count in enumerate(wins):

    def perc(n, d):
      return f"{n/d:.1%}"

    print(
      f"    {index+1:>3} {count:>4}  {perc(count, games):>6}  {perc(sum(wins[:index+1]), games):>6}"
    )
  print()
  print(
    f"Total time: {stop - start:.3f} seconds (parallelism: {parallelism})."
  )


def _main(
  *,
  game,
  strategy,
  dict_file,
  profile,
  answer,
  mode,
  sampling,
  answer_file,
):
  with open(dict_file or game.default_dict_file, "r") as f:
    wordlist = WordList(
      (
        entry
        for entry in (line.strip() for line in f.readlines())
        if len(entry) == game.word_length
      ),
      game,
    )

  answerlist = None
  if answer_file is not None:
    with open(answer_file, "r") as f:
      answerlist = [
        entry
        for entry in (line.strip() for line in f.readlines())
        if len(entry) == game.word_length
      ]

  if strategy == Strategy.BIFUR:
    # TODO: Currently, this massively slows down the other strategies, which is
    # unexpexcted. It seems related to the use of Multiprocessing. Investigate.
    wordlist.build_scoring_table()

  if mode == "test":
    regression_test(
      game, wordlist, strategy, sampling, answerlist
    )
  elif mode == "interactive":
    play_game_interactive(game, wordlist, strategy)
  elif answer is not None:
    play_game_with_answer(game, wordlist, strategy, answer)
  else:
    print("Showing wordlist and starting-word stats...")
    print()
    show_stats_interactive(game, wordlist)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--game", default="jaydle", type=Game)
  parser.add_argument(
    "--strategy", default="bifur", type=Strategy
  )
  parser.add_argument("--dict_file", default=None)
  parser.add_argument(
    "--profile", action=argparse.BooleanOptionalAction
  )

  # Play game with known answer.
  parser.add_argument("--answer")

  # Play game interactively.
  parser.add_argument(
    "-i", dest="mode", action="store_const", const="interactive"
  )

  # Run a regression test.
  parser.add_argument(
    "-t", dest="mode", action="store_const", const="test"
  )
  parser.add_argument("--sampling", default=1, type=int)
  parser.add_argument("--answer_file", default=None)

  args = parser.parse_args()
  if args.profile:
    with profile.Profile(
      timer=perf_counter_ns,
      timeunit=1e-9,
      subcalls=True,
      builtins=True,
    ) as pr:
      _main(**vars(args))
    p = pstats.Stats(pr)
    p.sort_stats(pstats.SortKey.TIME).print_stats()
  else:
    _main(**vars(args))


if __name__ == "__main__":
  main()
