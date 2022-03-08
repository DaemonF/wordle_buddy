#!/usr/bin/env python3
# cython: language_level=3
# distutils: language=c++

import cProfile as profile
import cython as c
import multiprocessing
import pickle
import os
import sys

from argparse import ArgumentParser, BooleanOptionalAction
from contextlib import contextmanager, ExitStack
from datetime import datetime
from enum import Enum
from ordered_set import OrderedSet
from pstats import Stats, SortKey
from time import perf_counter, perf_counter_ns
from tqdm import tqdm

if not c.compiled:
  import re
  import subprocess

  from colorama import Fore, Style

# Constants for the possible scores.
GREEN = 0
YELLOW = 1
GRAY = 2

# Mappings to display formats.
score_chars = ("!", "?", "x")
score_emoji = ("\U0001F7E9", "\U0001F7E8", "\U00002B1B")

# Letters from 'a' to 'z', for convenience.
letters = "abcdefghijklmnopqrstuvwxyz"


def inds(score):
  i = 0
  for s in score:
    i = i * 3 + s
  return i


def fmt_perc(value):
  return f"{value:.1%}"


def fmt_stat_real(stat):
  return f"{stat: >5.3f}"


def fmt_stat_perc(stat):
  return f"{stat: >5.1%}"


def fmt_stats(items, fmt_item=fmt_stat_perc, sep=", "):
  return sep.join(fmt_item(i) for i in items)


def fmt_stats_block(items, fmt_item=fmt_stat_perc):
  return f"[ {fmt_stats(items, fmt_item, sep=' ][ ')} ]"


class FakeMultiprocessing:

  @contextmanager
  def Pool(self, initializer=None):
    initializer()
    yield self

  @staticmethod
  def imap_unordered(func, it, **_):
    return (func(i) for i in it)


class PoolFunc:

  def __init__(self, func):
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
  """Enum that allows properties that don't affect the enum
  identity."""

  def __new__(cls, *args):
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
  HERMETIC = (
    "hermetic",
    "Hermetic testing",
    "wordlists/hermetic_testing.txt",
    5,
    6,
    datetime.now(),
    False,
    "**",
    " for testing",
  )

  def __init__(
    self,
    _,
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

  def fmt_result(self, result):
    game_id = (datetime.now() - self.epoch).days
    return (
      f"{self.display_name} {game_id} {len(result.tries)}"
      f"/{self.max_tries}{self.mode_symbol}\n\n"
      f"{result}\n\n"
      f"(Bot play{self.mode_desc})\n")


class Strategy(PropEnum):
  """The strategy to use when playing a game."""

  FREQ = ("freq", "positional letter frequency")
  CLUES = ("clues", "value of likely clues")
  DIV = ("div", "max division of the wordlist")

  def __init__(self, _, description):
    self.description = description


class Result:
  """The outcome of a game."""

  def __init__(self, tries):
    self.tries = tries

  def __str__(self):
    return "\n".join(
      score.fmt_emoji_and_stats() for score in self.tries)


class Score:
  """The outcome of a single try in a game."""

  def __init__(self, score, wordlist):
    self.score = score
    self.words = len(wordlist)

  def __str__(self):
    return "".join(s.char for s in self.score)

  def fmt_emoji_and_stats(self):
    score = "".join(score_emoji[s] for s in self.score)
    return f"{score} {self.words} word{'s'[:self.words^1]}"


class Guess:

  def __init__(self, word, wordlist):
    self.word = word
    self.wordlist = wordlist

  def __str__(self):
    stats = ", ".join(
      f"{strategy.value}: {fmt_stat_real(grade)}"
      for strategy, grade in (
        (strategy, self.wordlist.grade(self.word, strategy))
        for strategy in Strategy))
    return f"{self.word} ({stats})"


class LetterStats:

  def __init__(self, game):
    self.green_chance = [0] * game.word_length
    self.yellow_chance = [0] * game.word_length
    self.gray_chance = [0] * game.word_length
    self.dupe_chance = [0] * game.word_length

  def freeze(self):
    self.green_chance = tuple(self.green_chance)
    self.yellow_chance = tuple(self.yellow_chance)
    self.gray_chance = tuple(self.gray_chance)
    self.dupe_chance = tuple(self.dupe_chance)


class WordList(OrderedSet):

  def __init__(
      self, wordlist, game, strategy, scoring_table=None):
    super(WordList, self).__init__(wordlist)
    self.game = game
    self.strategy = strategy
    self.scoring_table = scoring_table

    self.stats = None
    self._score = [0] * game.word_length

    if strategy in (Strategy.FREQ, Strategy.CLUES):
      self.build_stats()
    elif strategy is Strategy.DIV and not scoring_table:
      self.build_scoring_table()

  def __getstate__(self):
    return (
      list(self),
      self.game,
      self.strategy,
      self.scoring_table,
      self._score,
      self.stats,
    )

  def __setstate__(self, state):
    super(WordList, self).__init__(state[0])
    (
      _,
      self.game,
      self.strategy,
      self.scoring_table,
      self._score,
      self.stats,
    ) = state

  def sublist(self, guess, score):
    """Returns a new WordList by removing all incompatible words
    from this wordlist."""
    f = lambda _: True
    for i, (g, s) in enumerate(zip(guess, score)):
      if s == GRAY:
        count = sum(
          1 for g2, s2 in zip(guess, score)
          if g2 == g and s2 != GRAY)
        f = lambda word, i=i, g=g, c=count, f=f: (
          word[i] != g and f(word) and word.count(g) <= c)
      elif s == YELLOW:
        count = sum(
          1 for g2, s2 in zip(guess, score)
          if g2 == g and s2 != GRAY)
        f = lambda word, i=i, g=g, c=count, f=f: (
          word[i] != g and f(word) and word.count(g) >= c)
      elif s == GREEN:
        f = lambda word, i=i, g=g, f=f: (
          word[i] == g and f(word))
      else:
        assert False

    return WordList(
      (word for word in self if f(word)),
      self.game,
      self.strategy,
      self.scoring_table,
    )

  def compute_score(self, guess, answer):
    return tuple(self._compute_score(guess, answer))

  def _compute_score(self, guess, answer):
    """WARNING: Return value is mutable for performance reasons.
    Must use or copy before calling this method again."""
    answer = list(answer)
    for index, letter in enumerate(guess):
      if letter == answer[index]:
        self._score[index] = GREEN
        answer[index] = None
      else:
        self._score[index] = GRAY
    for index, letter in enumerate(guess):
      if self._score[index] != GREEN and letter in answer:
        self._score[index] = YELLOW
        answer[answer.index(letter)] = None
    return self._score

  def build_stats(self):
    self.stats = {
      letter: LetterStats(self.game)
      for letter in letters
    }
    for word in self:
      # Track how a given guess would score for this answer.
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
        self.stats[letter].dupe_chance[word.count(letter)
                                       - 1] += 1

    # Normalize
    total_words = len(self)
    for stats in self.stats.values():
      total_count = sum(stats.dupe_chance) or 1
      for index in range(self.game.word_length):
        stats.green_chance[index] /= total_words
        stats.yellow_chance[index] /= total_words
        stats.gray_chance[index] /= total_words
        stats.dupe_chance[index] /= total_count
      stats.freeze()

  @staticmethod
  def build_scoring_table_part(wordlist, word):
    return (
      word,
      {
        answer: inds(wordlist._compute_score(word, answer))
        for answer in wordlist
      },
    )

  def build_scoring_table(self):
    print("Building scoring table...")
    start = perf_counter()
    func = PoolFunc(self.build_scoring_table_part).partial(self)
    with multiprocessing.Pool(initializer=func.setup) as pool:
      parts = pool.imap_unordered(func, self, chunksize=20)
      self.scoring_table = {
        word: scores
        for word, scores in parts
      }
    elapsed = perf_counter() - start
    if self.game is Game.HERMETIC:
      elapsed = 1.234
    print(f"Built scoring table in {elapsed:.3f} seconds.")

  def grade(self, word, strategy: Strategy):
    if strategy is Strategy.FREQ:
      return self._grade_by_freq(word)
    elif strategy is Strategy.CLUES:
      return self._grade_by_clues(word)
    elif strategy is Strategy.DIV:
      return self._grade_by_div(word)
    else:
      assert False

  def _grade_by_freq(self, word):
    """Grades a guess base on positional letter frequency in
    the wordlist."""
    if not self.stats:
      return 0
    grade = 0
    for index, letter in enumerate(word):
      stats = self.stats[letter]
      grade += (
        stats.green_chance[index] * self._dupe_modifier(
          word[:index], letter, stats)) / self.game.word_length

    # Make words outside the wordlist slightly less attractive.
    if not (self.game.hard_mode or word in self):
      grade -= 0.01
    return grade

  def _grade_by_clues(self, word):
    """Grades a guess by how many potential clues it could give
    based on the wordlist."""
    if not self.stats:
      return 0
    grade = 0
    for index, letter in enumerate(word):
      stats = self.stats[letter]
      prefix = word[:index]

      # Find the portion of answers that would yeild each score.
      greens = stats.green_chance[index]
      # In Wordle, duplicate letters only count as yellow if the
      # answer has the same or more duplicates. Model that.
      yellows = stats.yellow_chance[index] * self._dupe_modifier(
        prefix, letter, stats)
      # Only the first gray for a given letter matters.
      grays = 0 if letter in prefix else stats.gray_chance[index]

      # Weight each group by how it would split up the wordlist.
      green_weight = 0.5 - abs(0.5 - greens)
      yellow_weight = 0.5 - abs(0.5 - yellows)
      gray_weight = 0.5 - abs(0.5 - grays)

      grade += (
        greens * green_weight + yellows * yellow_weight
        + grays * gray_weight) / self.game.word_length

    # Make words outside the wordlist slightly less attractive.
    if not (self.game.hard_mode or word in self):
      grade -= 0.01
    return grade

  def _grade_by_div(self, word):
    """Grades a guess based on how well it divides the wordlist
    into smaller parts."""
    buckets = [0] * (3**self.game.word_length)
    if self.scoring_table and word in self.scoring_table:
      scores = self.scoring_table[word]
      for answer in self:
        buckets[scores[answer]] += 1
    else:
      for answer in self:
        buckets[inds(self._compute_score(word, answer))] += 1
    biggest_bucket = max(buckets)

    # Make words outside the wordlist slightly less attractive.
    if not (self.game.hard_mode or word in self):
      biggest_bucket += 0.5
    return len(self) / biggest_bucket

  @staticmethod
  def _dupe_modifier(prefix, letter, stats):
    """If the Guess contains duplicate letters, discount later
    occurrences based on the dupe chance."""
    if letter not in prefix:
      return 1
    return sum(stats.dupe_chance[prefix.count(letter):])


def show_stats_interactive(game, wordlist, strategy):
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
    if not sys.stdin.isatty():
      print(entry)
    if entry == "":
      break
    elif len(entry) == 1:
      if entry not in letters:
        print(f"ERROR: Invalid letter.")
      elif strategy in (Strategy.FREQ, Strategy.CLUES):
        stats = wordlist.stats[entry]
        print(
          f"  Chance of appearing in the answer:\n"
          f"    {fmt_perc(sum(stats.green_chance))}\n"
          f"  Positional chance of being green:\n"
          f"    {fmt_stats_block(stats.green_chance)}\n"
          f"  Positional chance of being yellow:\n"
          f"    {fmt_stats_block(stats.yellow_chance)}\n"
          f"  Chance of being gray:\n"
          f"    {fmt_perc(stats.gray_chance[0])}\n"
          f"\n"
          f"  Distribution of count per word it appears in:\n"
          f"    {fmt_stats(stats.dupe_chance)}")
      elif strategy is Strategy.DIV:
        print(f"\n  {strategy} doesn't support letter stats.")
      else:
        assert False
    elif len(entry) == game.word_length:
      print(f"  {Guess(entry, wordlist)}")
      if entry not in wordlist:
        print("  Note: Not in wordlist.")
    else:
      print(
        f"ERROR: Invalid length. Must be 1 or "
        f"{game.word_length} characters.")
    print()


def play_game(
  game,
  wordlist,
  strategy,
  scoring_func,
  quiet=False,
):
  if not quiet:
    print(f"Strategy: {strategy.value}")

  guesslist = wordlist
  tries = []
  while True:
    if not quiet:
      words = ""
      for index, word in enumerate(wordlist):
        if index >= 3:
          words += "..."
          break
        words += ("" if index == 0 else ", ") + word
      print(f"\nList has {len(wordlist)} words: {words}")
    guess = max(
      guesslist,
      key=lambda word: wordlist.grade(word, strategy),
    )
    if not quiet:
      print(f"Try: {guess}")

    score = scoring_func(guess)
    tries.append(Score(score, wordlist))
    if score.count(GREEN) == game.word_length:
      break

    wordlist = wordlist.sublist(guess, score)
    if game.hard_mode:
      guesslist = wordlist

  result = Result(tries)
  if not quiet:
    print(f"\n\n{game.fmt_result(result)}")
  return result


def play_game_interactive(game, wordlist, strategy):

  def scoring_func(_):
    while True:
      entry = input("What was the score? ")
      if not sys.stdin.isatty():
        print(entry)
      if len(entry) == game.word_length:
        return tuple(score_chars.index(c) for c in entry)

  return play_game(game, wordlist, strategy, scoring_func)


def play_game_with_answer(
    game, wordlist, strategy, answer, quiet=False):
  assert answer in wordlist

  def scoring_func(guess):
    score = wordlist.compute_score(guess, answer)
    if not quiet:
      print(f"Score: {''.join(score_chars[s] for s in score)}")
    return score

  return play_game(
    game, wordlist, strategy, scoring_func, quiet=quiet)


def regression_test_case(wordlist, strategy, answer):
  try:
    return play_game_with_answer(
      wordlist.game, wordlist, strategy, answer, quiet=True)
  except Exception:
    print(
      f"Crash:\n"
      f"  game: {wordlist.game}\n"
      f"  strategy: {strategy}\n"
      f"  answer: {answer}")
    raise


def regression_test(
    game, wordlist, strategy, sampling, answerlist):
  answerlist = answerlist or wordlist
  if sampling != 1:
    answerlist = [
      answer for index, answer in enumerate(answerlist)
      if index % sampling == 0
    ]

  games = len(answerlist)
  wins = [0] * 10

  start = perf_counter()
  func = PoolFunc(regression_test_case).partial(
    wordlist, strategy)
  parallelism = os.cpu_count()
  with multiprocessing.Pool(parallelism,
                            initializer=func.setup) as pool:
    results = pool.imap_unordered(func, answerlist)
    for result in tqdm(results, total=games):
      wins[len(result.tries) - 1] += 1
  elapsed = perf_counter() - start
  if game is Game.HERMETIC:
    elapsed = 12.345

  sampling_desc = f"1/{sampling} of "
  print(
    f"Test details:\n"
    f"  Games:     playing {games} games of {game.display_name}\n"
    f"  Sampling:  playing {'' if sampling == 1 else sampling_desc}all possible games\n"
    f"  Guesslist: starting with {len(wordlist)} potential words\n"
    f"  Strategy:  guessing based on {strategy.description}\n"
    f"\n"
    f"Distribution of tries required to win:\n"
    f"  Tries   Wins  % of games     Cum. %")
  for index, count in enumerate(wins):
    if index == game.max_tries:
      print("    --- would lose below this line ---")
    print(
      f"  {index+1:>5} {count:>6} {fmt_perc(count / games):>11} "
      f"{fmt_perc(sum(wins[:index+1]) / games):>10}")
  print(
    f"\n"
    f"Total time: {elapsed:.3f} seconds "
    f"(parallelism: {parallelism}).")


def run_wordle_buddy(
  *,
  game,
  strategy,
  dict_file,
  answer,
  mode,
  sampling,
  answer_file,
  **_,
):
  wordlist = None
  with open(dict_file or game.default_dict_file, "r") as f:
    wordlist = WordList(
      (
        entry
        for entry in (line.strip() for line in f.readlines())
        if len(entry) == game.word_length),
      game,
      strategy,
    )

  answerlist = None
  if answer_file:
    with open(answer_file, "r") as f:
      answerlist = [
        entry
        for entry in (line.strip() for line in f.readlines())
        if len(entry) == game.word_length
      ]

  if mode == "test":
    regression_test(
      game, wordlist, strategy, sampling, answerlist)
  elif mode == "interactive":
    play_game_interactive(game, wordlist, strategy)
  elif answer:
    play_game_with_answer(game, wordlist, strategy, answer)
  else:
    print("Showing wordlist and starting-word stats...\n")
    show_stats_interactive(game, wordlist, strategy)


def main():
  parser = ArgumentParser()
  parser.add_argument("--game", default="jaydle", type=Game)
  parser.add_argument("--strategy", default="div", type=Strategy)
  parser.add_argument("--dict_file", default=None)
  parser.add_argument("--profile", action=BooleanOptionalAction)
  parser.add_argument("--cython", action=BooleanOptionalAction)

  # Play game with known answer.
  parser.add_argument("--answer")

  # Play game interactively.
  parser.add_argument(
    "-i", dest="mode", action="store_const", const="interactive")

  # Run a regression test.
  parser.add_argument(
    "-t", dest="mode", action="store_const", const="test")
  parser.add_argument("--sampling", default=1, type=int)
  parser.add_argument("--answer_file", default=None)

  args = parser.parse_args()
  if not c.compiled and args.cython:
    directives = {
      "boundscheck": False,
      "wraparound": False,
      "initializedcheck": False,
      "nonecheck": False,
      "profile": bool(args.profile),
      "infer_types": True,
      # "warn.undeclared": True,
      "warn.maybe_uninitialized": True,
      "warn.unused": True,
      "warn.unused_arg": True,
      "warn.unused_result": True,
    }
    directives = ",".join(
      f"{name}={val}" for name, val in directives.items())

    flags = ("--build", "--annotate", "-X", directives)
    sources = ("wordle_buddy.py", )
    with subprocess.Popen(
      ["/usr/bin/cythonize", *flags, *sources],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
    ) as p:

      def build_ignored_warnings_re():
        paths = (r"/usr/lib/", )
        paths = r"|".join(re.escape(path) for path in paths)

        messages = (r"Unused entry 'genexpr'", )
        messages = r"|".join(
          re.escape(path) for path in messages)

        return re.compile(
          fr"warning:("
          fr" ({paths})|"
          fr" [^:]+:[0-9]+:[0-9]+: ({messages})"
          fr")")

      compiling_re = re.compile(r"Compiling ")
      build_ext_re = re.compile(r"running build_ext")
      gcc_re = re.compile(r"(gcc|g\+\+)")
      warning_re = re.compile(r"warning: ")
      ignored_warnings_re = build_ignored_warnings_re()

      msg = lambda *args: print(*args, file=sys.stderr)
      recompiling = False
      for line in p.stdout:
        line = line.rstrip()
        if build_ext_re.match(line) or ignored_warnings_re.match(
            line):
          continue
        elif (compiling_re.match(line) or gcc_re.match(line)
              or (not recompiling and warning_re.match(line))):
          msg()
          recompiling = True

        if warning_re.match(line):
          line = (
            f"{Fore.YELLOW}WARNING:{Style.RESET_ALL}"
            f" {warning_re.sub('', line)}")
        msg(line)
      error = p.wait()
      if error:
        msg(
          "{Fore.RED}ERROR:{Style.RESET_ALL}"
          " Compilation failed. Exiting.")
        sys.exit(error)
      if recompiling:
        msg("\nDone. Running program...\n")
    from wordle_buddy import main

    # It is critical to reparse args within the compiled version
    # to avoid duplicate definition issues, such as with enums.
    return main()

  prof = None
  with ExitStack() as stack:
    if args.profile:
      global multiprocessing
      multiprocessing = FakeMultiprocessing()
      timer = perf_counter_ns
      if args.game is Game.HERMETIC:
        timer = lambda: 0
      prof = stack.enter_context(
        profile.Profile(timer=timer, timeunit=1e-9))

    run_wordle_buddy(**vars(args))

  if prof:
    if args.game is Game.HERMETIC:
      print(
        "\tCollected profile successfully. Supressing "
        "non-hermetic output.")
    else:
      Stats(prof).sort_stats(SortKey.TIME).print_stats()


if __name__ == "__main__":
  main()
