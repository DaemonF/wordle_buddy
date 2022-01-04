#!/usr/bin/env python3

import sys

dict_file = 'words_alpha.txt'
#dict_file = 'scrabble_words.txt'
word_length = 5

# Weights used when assesing the relative value of certain results.
green_weight = 1 / word_length
yellow_weight = green_weight / 5
gray_weight = yellow_weight / 5

# Characters used to display the score.
green_char = '!'
yellow_char = '?'
gray_char = 'x'

def c_range(start, end):
  '''Generates the characters from `start` to `end`, inclusive.
  '''
  for c in range(ord(start), ord(end)+1):
    yield chr(c)

def letters():
  return c_range('a', 'z')

def occurrences(word, letter):
  return len(list(filter(lambda x: x == letter, word)))

class Guess:
  def __init__(self, wordlist, word):
    self.wordlist = wordlist
    self.word = word

    self.freq_grade = self._grade_by_frequency()
    self.clues_grade = self._grade_by_potential_clues()
    self.bifur_grade = self._grade_by_bifurcation()

    self.score = ['?' for x in range(word_length)]  # The actual result from Wordle.

  def _grade_by_frequency(self):
    '''Grades a guess base on positional letter frequency in the wordlist.
    '''
    grade = 0
    for index, letter in enumerate(self.word):
      grade += (self.wordlist.letter_pos_freq[letter][index]
                * self._dupe_modifier(index)
                / word_length)
    return grade

  def _grade_by_potential_clues(self):
    '''Grades a guess by how many potential clues it could give based on the wordlist.
    '''
    grade = 0
    for index, letter in enumerate(self.word):
      # The number of words that would make this guess green, yellow, or gray.
      greens = self.wordlist.letter_pos_freq[letter][index]
      yellows = self.wordlist.letter_freq[letter] - greens
      grays = 1 - greens - yellows

      # In Wordle, duplicate letters only count as yellow if the answer has the same or more duplicates. Model that.
      yellows *= self._dupe_modifier(index)
      # Only the first gray for a given letter matters.
      grays *= 0 if self._dupe_modifier(index) < 1.0 else 1

      # Weight each category by how valuable it would be at this index.
      grade += (greens * green_weight
                + yellows * yellow_weight
                + grays * gray_weight)
    return grade

  def _grade_by_bifurcation(self):
    '''Grades a guess based on how closely it would split the wordlist in equal halves.
    '''
    # TODO
    return 0

  def _dupe_modifier(self, index):
    '''If the Guess contains duplicate letters, discount later occurrences based on the dupe chance.
    '''
    letter = self.word[index]
    occurrence = occurrences(self.word[:index], letter) + 1
    return (1 if occurrence <= 1
      else sum(self.wordlist.letter_dupe_chance[letter][occurrence:]))

  def __str__(self):
    return f"{self.word} (freq: {self.freq_grade:.3f}, clues: {self.clues_grade:.3f}, bifur: {self.bifur_grade:.3f})"


class WordList(list):
  def __init__(self, wordlist):
    super(WordList, self).__init__(wordlist)

    self.letter_freq = {letter:0 for letter in letters()}
    self.letter_pos_freq = {letter:[0 for x in range(word_length)] for letter in letters()}

    self.green_weights = {letter:[0 for x in range(word_length)] for letter in letters()}
    self.yellow_weights = {letter:[0 for x in range(word_length)] for letter in letters()}
    self.gray_weights = {letter:[0 for x in range(word_length)] for letter in letters()}

    self.letter_dupe_chance = {letter:[0 for x in range(word_length)] for letter in letters()}

    for word in self:
      for index, letter in enumerate(word):
        # Track how often the letter appears, and in this index.
        self.letter_freq[letter] += 1
        self.letter_pos_freq[letter][index] += 1

        # Track whether a given guess would be green, yellow, or gray for this word.
        for guess in letters():
          if guess == letter:
            self.green_weights[guess][index] += 1
          elif guess in word:
            self.yellow_weights[guess][index] += 1
          else:
            self.gray_weights[guess][index] += 1

      # Track how often letters appear within a word.
      for letter in set(word):
        self.letter_dupe_chance[letter][occurrences(word, letter)] += 1

    # Normalize
    total_words = len(self)
    #total_letters = sum(self.letter_freq.values())
    for letter in letters():
      self.letter_freq[letter] /= total_words

      total_occurrences = sum(self.letter_dupe_chance[letter])
      for index in range(word_length):
        self.letter_pos_freq[letter][index] /= total_words

        # TODO: This is just incorrect...
        for weights in [self.green_weights, self.yellow_weights, self.gray_weights]:
          weights[letter][index] = weights[letter][index] / total_words

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
        new_list = [w for w in new_list if letter not in w]
      else:
        throw('Huh?')
    return WordList(new_list)


def main():
  raw_list = []
  with open(dict_file, 'r') as f:
    for l in f.readlines():
      entry = l.strip()
      if len(entry) == 5:
        raw_list.append(entry)

  list = WordList(raw_list)

  if len(sys.argv) < 2:
    print("No answer given as an argument. Showing wordlist and starting-word stats...")
    print()

    guesses = [Guess(list, word) for word in list]

    print("By positional letter frequency:")
    by_freq = sorted(guesses, key=lambda x: x.freq_grade, reverse=True)
    for guess in by_freq[:5]:
      print(f"  {guess}")
    print()

    print("By potential for clues:")
    by_clues = sorted(guesses, key=lambda x: x.clues_grade, reverse=True)
    for guess in by_clues[:5]:
      print(f"  {guess}")
    print()

    print("By maximum wordlist bifurcation:")
    by_bifur = sorted(guesses, key=lambda x: x.bifur_grade, reverse=True)
    for guess in by_bifur[:5]:
      print(f"  {guess}")

    return

  answer = sys.argv[1]

  tries = 0
  while True:
    tries += 1
    print(f"List has {len(list)} words: {', '.join(list[:3])}")

    guesses = [Guess(list, word) for word in list]

    guess = sorted(guesses, key=lambda x:
      x.freq_grade, reverse=True)[0]
    print(f"Try: {guess}")

    for index, letter in enumerate(guess.word):
      if letter == answer[index]:
        guess.score[index] = green_char
      elif letter in answer and occurrences(answer, letter) > occurrences(guess.word[:index], letter):
        guess.score[index] = yellow_char
      else:
        guess.score[index] = gray_char

    print(f"Score: {''.join(guess.score)}")
    print()

    if occurrences(guess.score, '!') == word_length:
      break

    list = list.sublist(guess)

  print(f"Got it in {tries}/6 tries.")


if __name__ == '__main__':
  main()
