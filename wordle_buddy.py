#!/usr/bin/env python3

import argparse
import functools
import multiprocessing
import os
import sys


# The word size for the game.
word_length = 5

# Characters used to display the score.
green_char = '!'
yellow_char = '?'
gray_char = 'x'

# Letters from 'a' to 'z', for convenience.
letters = [chr(x) for x in range(ord('a'), ord('z') + 1)]


def occurrences(word, letter):
  return len(list(filter(lambda x: x == letter, word)))

def fmt_stat(stat):
  return f"{stat: >5.3f}"

def fmt_stats(list):
  return ', '.join(fmt_stat(e) for e in list)


class Guess:
  def __init__(self, wordlist, word):
    self.wordlist = wordlist
    self.word = word

    self.grades = {
      'freq': self._grade_by_frequency(),
      'clues': self._grade_by_potential_clues(),
      'bifur': self._grade_by_bifurcation(),
    }

    self.score = [None for x in range(word_length)]  # The actual result from Wordle.

  def _grade_by_frequency(self):
    '''Grades a guess base on positional letter frequency in the wordlist.
    '''
    grade = 0
    for index, letter in enumerate(self.word):
      grade += (self.wordlist.letter_pos_freq[letter][index]
                * self._dupe_modifier(letter, index)
               ) / word_length
    return grade

  def _grade_by_potential_clues(self):
    '''Grades a guess by how many potential clues it could give based on the wordlist.
    '''
    debug = False
    grade = 0
    for index, letter in enumerate(self.word):
      # The number of words that would make this guess green, yellow, or gray.
      greens = self.wordlist.green_chance[letter][index]
      yellows = self.wordlist.yellow_chance[letter][index]
      grays = self.wordlist.gray_chance[letter][index]
      if debug:
        print(f"{self.word}[{index}]")
        print(f"pre%: {fmt_stats([greens, yellows, grays])} (sum: {fmt_stat(greens + yellows + grays)})")

      # In Wordle, duplicate letters only count as yellow if the answer has the same or more duplicates. Model that.
      yellows *= self._dupe_modifier(letter, index)
      # Only the first gray for a given letter matters.
      grays *= 0 if self._dupe_modifier(letter, index) < 1.0 else 1
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

  def _grade_by_bifurcation(self):
    '''Grades a guess based on how closely it would split the wordlist in equal halves.
    '''
    # TODO
    return 0

  def _dupe_modifier(self, letter, index):
    '''If the Guess contains duplicate letters, discount later occurrences based on the dupe chance.
    '''
    # TODO: Need to handle greens, both before and after index and not penalize. (ie. green_chance == 1.0)
    prev_dupes = occurrences(self.word[:index], letter)
    return (1 if prev_dupes <= 0
      else sum(self.wordlist.letter_dupe_chance[letter][prev_dupes:]))

  def __str__(self):
    stats = ', '.join(f"{k}: {fmt_stat(v)}" for k, v in self.grades.items())
    return f"{self.word} ({stats})"


class WordList(list):
  def __init__(self, wordlist):
    super(WordList, self).__init__(wordlist)

    self.letter_freq = {letter:0 for letter in letters}
    self.letter_pos_freq = {letter:[0 for x in range(word_length)] for letter in letters}

    self.green_chance = {letter:[0 for x in range(word_length)] for letter in letters}
    self.yellow_chance = {letter:[0 for x in range(word_length)] for letter in letters}
    self.gray_chance = {letter:[0 for x in range(word_length)] for letter in letters}

    self.letter_dupe_chance = {letter:[0 for x in range(word_length)] for letter in letters}

    for word in self:
      for index, letter in enumerate(word):
        # Track how often the letter appears, and in this index.
        self.letter_freq[letter] += 1
        self.letter_pos_freq[letter][index] += 1

        # Track whether a given guess would be green, yellow, or gray for this word.
        # TODO: This assumes all guesses are equally likely. Use wordlist?
        for guess in letters:
          if guess == letter:
            self.green_chance[guess][index] += 1
          elif guess in word:
            self.yellow_chance[guess][index] += 1
          else:
            self.gray_chance[guess][index] += 1

      # Track how often letters appear within a word.
      for letter in set(word):
        self.letter_dupe_chance[letter][occurrences(word, letter) - 1] += 1

    # Normalize
    total_words = len(self)
    for letter in letters:
      self.letter_freq[letter] /= total_words

      total_occurrences = sum(self.letter_dupe_chance[letter])
      for index in range(word_length):
        self.letter_pos_freq[letter][index] /= total_words

        self.green_chance[letter][index] /= total_words
        self.yellow_chance[letter][index] /= total_words
        self.gray_chance[letter][index] /= total_words

        if total_occurrences > 0:
          self.letter_dupe_chance[letter][index] /= total_occurrences


  def sublist(self, scored_guess):
    '''Returns a new WordList by removing all incompatible words from this wordlist.
    '''
    new_list = self
    for index, letter in enumerate(scored_guess.word):
      score = scored_guess.score[index]
      if score == green_char:
        new_list = [w for w in new_list if w[index] == letter]
      elif score == yellow_char:
        new_list = [w for w in new_list if w[index] != letter and letter in w]
      elif score == gray_char:
        # TODO: Unless the char is yellow or green in word.
        new_list = [w for w in new_list if letter not in w]
      else:
        throw(f"Unknown score character: '{score}'.")
    return WordList(new_list)


def show_stats_interactive(list):
  if True:  # FIXME: de-indent
    guesses = [Guess(list, word) for word in list]

    def _print_top_guesses(guesses, title, strategy, limit=5):
      print(title)
      top = sorted(guesses, key=lambda x: x.grades[strategy], reverse=True)
      for guess in top[:limit]:
        print(f"  {guess}")
      print()

    _print_top_guesses(guesses, "By positional letter frequency:", 'freq')
    _print_top_guesses(guesses, "By potential for clues:", 'clues')
    _print_top_guesses(guesses, "By maximum wordlist bifurcation:", 'bifur')

    while True:
      entry = input("Enter a letter or word: ")
      if entry == "":
        break
      elif len(entry) == 1:
        def fmt_stat(stat):
          return f"{stat: >5.1%}"
        def fmt_stats(list):
          return ', '.join(fmt_stat(e) for e in list)

        print(f"  Overall frequency: {fmt_stat(list.letter_freq[entry])}")
        print(f"  Positional frequency:\n    {fmt_stats(list.letter_pos_freq[entry])}")
        print()
        print(f"  Positional chance of green:\n    {fmt_stats(list.green_chance[entry])}")
        print(f"  Positional chance of yellow:\n    {fmt_stats(list.yellow_chance[entry])}")
        print(f"  Positional chance of gray:\n    {fmt_stats(list.gray_chance[entry])}")
        print()
        print(f"  Distribution of count per word it appears in:\n    {fmt_stats(list.letter_dupe_chance[entry])}")
      elif len(entry) == word_length:
        print(f"  {Guess(list, entry)}")
        if entry not in list:
          print("  Note: Not in word list.")
      else:
        print(f"ERROR: Invalid length. Must be 1 or {word_length} characters.")
      print()

def play_game_interactive(list, strategy):
  if True:  # FIXME: de-indent
    print(f"Strategy: {strategy}")
    print()
    tries = 0
    while True:
      tries += 1
      print(f"List has {len(list)} words: {', '.join(list[:3])}")

      guesses = [Guess(list, word) for word in list]
      guess = sorted(guesses, key=lambda x: x.grades[strategy], reverse=True)[0]
      print(f"Try: {guess}")

      while None in guess.score:
        resp = input("What was the score? ")
        if len(resp) == word_length:
          for index, char in enumerate(resp):
            guess.score[index] = char
      print()

      if occurrences(guess.score, green_char) == word_length:
        break

      list = list.sublist(guess)

    print(f"Got it in {tries}/6 tries.")

def play_game_with_answer(list, strategy, answer, quiet=False):
  def _print(*args):
    if not quiet:
      print(*args)

  if answer not in list:
    _print(f"'{answer}' is not in word list. Exiting...")
    return

  _print(f"Strategy: {strategy}")
  _print()
  tries = 0
  while True:
    tries += 1
    _print(f"List has {len(list)} words: {', '.join(list[:3])}")

    guesses = [Guess(list, word) for word in list]
    guess = sorted(guesses, key=lambda x: x.grades[strategy], reverse=True)[0]
    _print(f"Try: {guess}")

    for index, letter in enumerate(guess.word):
      if letter == answer[index]:
        guess.score[index] = green_char
      # TODO: Need to subtract greens or yellows elsewhere in the word ('rebus' with a guess of 'seres').
      elif letter in answer and occurrences(answer, letter) > occurrences(guess.word[:index], letter):
        guess.score[index] = yellow_char
      else:
        guess.score[index] = gray_char

    _print(f"Score: {''.join(guess.score)}")
    _print()

    if occurrences(guess.score, green_char) == word_length:
      break

    list = list.sublist(guess)

  _print(f"Got it in {tries}/6 tries.")
  return tries

def _regression_test(list, strategy, answer):
  try:
    return play_game_with_answer(list, strategy, answer, quiet=True)
  except ZeroDivisionError:
    return 0

def regression_test(list, strategy, sampling):
  answers = [word for index, word in enumerate(list) if index % sampling == 0]
  total = len(answers)
  wins = [0 for x in range(20)]
  crashes = 0

  with multiprocessing.Pool(os.cpu_count()) as pool:
    results = pool.imap(functools.partial(_regression_test, list, strategy), answers)

    interval = max(1, total // 100)
    for index, result in enumerate(results):
      if index % interval == interval - 1:
        print(f"{(index + 1) / total:.0%}", file=sys.stderr)

      if result == 0:
        crashes += 1
      else:
        wins[result - 1] += 1

  print(f"Regression test")
  print(f"  List: {len(list)} words")
  print(f"  Games: {len(answers)} answers (sampling: 1/{sampling})")
  print(f"  Strategy: {strategy}")
  print()
  print(f"Stats of {total} games:")
  print(f"  Crashes: {crashes} {crashes / total:.2%}")
  print(f"  Wins:")
  for index, count in enumerate(wins):
    def perc(n, d):
      return f"{n/d:.1%}"
    print(f"    {index+1:>3} {count:>4}  {perc(count, total):>6}  {perc(sum(wins[:index+1]), total):>6}")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dict_file', default='sowpods.txt')
  parser.add_argument('--strategy', default='freq')
  # Play game with known answer.
  parser.add_argument('--answer')
  # Play game interactively.
  parser.add_argument('-i', dest='mode', action='store_const', const='interactive')
  # Run a regression test.
  parser.add_argument('-t', dest='mode', action='store_const', const='test')
  parser.add_argument('--sampling', default=10, type=int)
  args = parser.parse_args()

  raw_list = []
  with open(args.dict_file, 'r') as f:
    for l in f.readlines():
      entry = l.strip()
      if len(entry) == 5:
        raw_list.append(entry)
  list = WordList(raw_list)

  if args.mode == 'test':
    regression_test(list, args.strategy, args.sampling)
  elif args.mode == 'interactive':
    play_game_interactive(list, args.strategy)
  elif args.answer is not None:
    play_game_with_answer(list, args.strategy, args.answer)
  else:
    print("Showing wordlist and starting-word stats...")
    print()
    show_stats_interactive(list)


if __name__ == '__main__':
  main()
