#!/usr/bin/env python3

import multiprocessing
import os
import sys

from argparse import ArgumentParser
from enum import Enum
from functools import cache, partial
from time import time
from tqdm import tqdm


# The word size for the game.
word_length = 5

# Characters used to display the score.
green_char = '!'
yellow_char = '?'
gray_char = 'x'

# Letters from 'a' to 'z', for convenience.
letters = [chr(ordinal) for ordinal in range(ord('a'), ord('z') + 1)]

# All strategies that can be used.
class Strategy(Enum):
  FREQ = 'freq', "positional letter frequency"
  CLUES = 'clues', "potential clue value"
  BIFUR = 'bifur', "maximum wordlist bifurcation"

  def __new__(cls, *args, **kwds):
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj

  def __init__(self, name, description):
    self.description = description


def occurrences(word, letter):
  return sum(1 for l in word if l == letter)

def fmt_stat(stat):
  return f"{stat: >5.3f}"

def fmt_stats(stats):
  return ', '.join(fmt_stat(e) for e in stats)


class Guess:
  def __init__(self, wordlist, word):
    self.wordlist = wordlist
    self.word = word

    self.score: list = [None for _ in range(word_length)]  # The actual result from the game.

  def compute_score(self, answer):
    answer = list(answer)
    for index in range(word_length):
      self.score[index] = gray_char
    for index in range(word_length):
      if self.word[index] == answer[index]:
        self.score[index] = green_char
        answer[index] = None
    for index in range(word_length):
      if self.word[index] in answer and self.score[index] == gray_char:
        self.score[index] = yellow_char
        answer[answer.index(self.word[index])] = None

  def __str__(self):
    stats = ', '.join(f"{strategy.value}: {fmt_stat(self.wordlist.grade(self.word, strategy))}" for strategy in Strategy)
    return f"{self.word} ({stats})"


class LetterStats():
  def __init__(self):
    self.green_chance = [0 for _ in range(word_length)]
    self.yellow_chance = [0 for _ in range(word_length)]
    self.gray_chance = [0 for _ in range(word_length)]
    self.dupe_chance = [0 for _ in range(word_length)]

  def freeze(self):
    self.green_chance = tuple(self.green_chance)
    self.yellow_chance = tuple(self.yellow_chance)
    self.gray_chance = tuple(self.gray_chance)
    self.dupe_chance = tuple(self.dupe_chance)

class WordList(list):
  def __init__(self, wordlist):
    super(WordList, self).__init__(wordlist)

    self.stats = {letter:LetterStats() for letter in letters}

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
      for index in range(word_length):
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
      raise(f"Guess is not associated with this wordlist.")
    filter = lambda w: True
    guess = scored_guess.word
    score = scored_guess.score
    for index, letter in enumerate(guess):
      if score[index] == green_char:
        filter = partial(lambda f,i,l,w: f(w) and
          w[i] == l,
          filter, index, letter)
      elif score[index] == yellow_char:
        at_least_count = sum(1 for j in range(word_length) if guess[j] == letter and score[j] in (yellow_char, green_char))
        filter = partial(lambda f,i,l,c,w: f(w) and
          w[i] != l and occurrences(w, l) >= c,
          filter, index, letter, at_least_count)
      elif score[index] == gray_char:
        at_most_count = occurrences(guess, letter) - sum(1 for j in range(word_length) if guess[j] == letter and score[j] == gray_char)
        filter = partial(lambda f,i,l,c,w: f(w) and
          w[i] != l and occurrences(w, l) <= c,
          filter, index, letter, at_most_count)
      else:
        raise(f"Unknown score character: '{score}'.")
    return WordList(w for w in self if filter(w))

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
               ) / word_length
    return grade

  def _grade_by_potential_clues(self, word):
    '''Grades a guess by how many potential clues it could give based on the wordlist.
    '''
    debug = False
    grade = 0
    for index, letter in enumerate(word):
      stats = self.stats[letter]
      # The number of words that would make this guess green, yellow, or gray.
      greens = stats.green_chance[index]
      yellows = stats.yellow_chance[index]
      grays = stats.gray_chance[index]
      if debug:
        print(f"{word}[{index}]")
        print(f"pre%: {fmt_stats([greens, yellows, grays])} (sum: {fmt_stat(greens + yellows + grays)})")

      # In Wordle, duplicate letters only count as yellow if the answer has the same or more duplicates. Model that.
      yellows *= self._dupe_modifier(word, index, letter, stats)
      # Only the first gray for a given letter matters.
      grays *= 1 if word.index(letter) == index else 0
      if debug:
        print(f"adj%: {fmt_stats([greens, yellows, grays])} (sum: {fmt_stat(greens + yellows + grays)})")

      # Weight each category by how much it would split up the wordlist.
      green_weight = 1/2 - abs(1/2 - greens)
      yellow_weight = 1/2 - abs(1/2 - yellows)
      gray_weight = 1/2 - abs(1/2 - grays)
      if debug:
        print(f"weights: {fmt_stats([green_weight, yellow_weight, gray_weight])} (sum: {fmt_stat(green_weight + yellow_weight + gray_weight)})")

      grade += (greens * green_weight
                + yellows * yellow_weight
                + grays * gray_weight
               ) / word_length
    return grade

  def _grade_by_bifurcation(self, word):
    '''Grades a guess based on how closely it would split the wordlist in equal halves.
    '''
    # TODO
    return 0

  def _dupe_modifier(self, word, index, letter, stats):
    '''If the Guess contains duplicate letters, discount later occurrences based on the dupe chance.
    '''
    return (1 if word.index(letter) == index
      else sum(stats.dupe_chance[occurrences(word[:index], letter):]))


def show_stats_interactive(wordlist):
  for strategy in Strategy:
    print(f"By {strategy.description}:")
    for guess in sorted(wordlist, key=lambda word: wordlist.grade(word, strategy), reverse=True)[:5]:
      print(f"  {Guess(wordlist, guess)}")
    print()

  while True:
    entry = input("Enter a letter or word: ")
    if entry == "":
      break
    elif len(entry) == 1:
      def fmt_stat(stat):
        return f"{stat: >5.1%}"
      def fmt_stats(stats):
        return ', '.join(fmt_stat(e) for e in stats)
      stats = wordlist.stats[entry]

      print(f"  Appears anywhere in word: {fmt_stat(sum(stats.green_chance))}")
      print()
      print(f"  Positional chance of green:\n    {fmt_stats(stats.green_chance)}")
      print(f"  Positional chance of yellow:\n    {fmt_stats(stats.yellow_chance)}")
      print(f"  Positional chance of gray:\n    {fmt_stats(stats.gray_chance)}")
      print()
      print(f"  Distribution of count per word it appears in:\n    {fmt_stats(stats.dupe_chance)}")
    elif len(entry) == word_length:
      print(f"  {Guess(wordlist, entry)}")
      if entry not in wordlist:
        print("  Note: Not in wordlist.")
    else:
      print(f"ERROR: Invalid length. Must be 1 or {word_length} characters.")
    print()

def play_game(wordlist, strategy, scoring_func, quiet=False):
  def _print(*args):
    if not quiet:
      print(*args)

  _print(f"Strategy: {strategy.value}")
  _print()
  tries = 0
  while True:
    tries += 1
    _print(f"List has {len(wordlist)} words: {', '.join(wordlist[:3])}")

    guess = Guess(wordlist, max(wordlist, key=lambda word: wordlist.grade(word, strategy)))
    _print(f"Try: {guess}")

    scoring_func(guess)

    if occurrences(guess.score, green_char) == word_length:
      break

    wordlist = wordlist.sublist(guess)

  _print(f"Got it in {tries}/6 tries.")
  return tries

def play_game_interactive(wordlist, strategy):
  def scoring_func(guess):
    while None in guess.score:
      resp = input("What was the score? ")
      if len(resp) == word_length:
        for index, char in enumerate(resp):
          guess.score[index] = char
    print()
  return play_game(wordlist, strategy, scoring_func)

def play_game_with_answer(wordlist, strategy, answer, quiet=False):
  def _print(*args):
    if not quiet:
      print(*args)

  if answer not in wordlist:
    _print(f"'{answer}' is not in wordlist. Exiting...")
    return

  def scoring_func(guess):
    guess.compute_score(answer)
    _print(f"Score: {''.join(guess.score)}")
    _print()
  return play_game(wordlist, strategy, scoring_func, quiet=quiet)

def _regression_test(wordlist, strategy, answer):
  try:
    return play_game_with_answer(wordlist, strategy, answer, quiet=True)
  except ZeroDivisionError:
    return answer

def regression_test(wordlist, strategy, sampling, answerlist):
  answers = answerlist
  if answers is None:
    answers = [word for index, word in enumerate(wordlist) if index % sampling == 0]
  total = len(answers)
  wins = [0 for _ in range(20)]
  crashes = []

  start = time()
  parallelism = os.cpu_count()
  chunk_size = 5
  with multiprocessing.Pool(parallelism, maxtasksperchild=25) as pool:
    results = pool.imap(partial(_regression_test, wordlist, strategy), answers, chunk_size)

    for index, result in tqdm(enumerate(results), total=total):
      if type(result) is str:
        crashes.append(result)
      else:
        wins[result - 1] += 1
  stop = time()

  print(f"Regression test")
  print(f"  List: {len(wordlist)} words")
  if answerlist is None:
    print(f"  Games: {len(answers)} answers (sampling: 1/{sampling})")
  else:
    print(f"  Games: {len(answers)} answers")
  print(f"  Strategy: {strategy.value}")
  print()
  print(f"Stats of {total} games:")
  if len(crashes) > 0:
    print(f"  Crashes: {len(crashes)} {len(crashes) / total:.2%}")
    for crash in crashes:
      print(f"    {crash}")
  print(f"  Wins:")
  for index, count in enumerate(wins):
    def perc(n, d):
      return f"{n/d:.1%}"
    print(f"    {index+1:>3} {count:>4}  {perc(count, total):>6}  {perc(sum(wins[:index+1]), total):>6}")
  print()
  print(f"Total time: {stop - start:.3f} seconds (parallelism: {parallelism}, chunk size: {chunk_size}).")


def main():
  parser = ArgumentParser()
  parser.add_argument('--dict_file', default='wordlists/sowpods.txt')
  parser.add_argument('--strategy', default='freq', type=Strategy)
  # Play game with known answer.
  parser.add_argument('--answer')
  # Play game interactively.
  parser.add_argument('-i', dest='mode', action='store_const', const='interactive')
  # Run a regression test.
  parser.add_argument('-t', dest='mode', action='store_const', const='test')
  parser.add_argument('--sampling', default=10, type=int)
  parser.add_argument('--answer_file', default=None)
  args = parser.parse_args()

  raw_wordlist = []
  with open(args.dict_file, 'r') as f:
    for l in f.readlines():
      entry = l.strip()
      if len(entry) == word_length:
        raw_wordlist.append(entry)
  wordlist = WordList(raw_wordlist)

  answerlist = None
  if args.answer_file is not None:
    answerlist = []
    with open(args.answer_file, 'r') as f:
      for l in f.readlines():
        entry = l.strip()
        if len(entry) == word_length:
          answerlist.append(entry)

  if args.mode == 'test':
    regression_test(wordlist, args.strategy, args.sampling, answerlist)
  elif args.mode == 'interactive':
    play_game_interactive(wordlist, args.strategy)
  elif args.answer is not None:
    play_game_with_answer(wordlist, args.strategy, args.answer)
  else:
    print("Showing wordlist and starting-word stats...")
    print()
    show_stats_interactive(wordlist)


if __name__ == '__main__':
  main()
