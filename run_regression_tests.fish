#!/usr/bin/env fish

for strat in {freq,clues,bifur}
  ./wordle_buddy.py --strategy=$strat -t \
    > regression_tests/$strat/wordle_answers.txt
  ./wordle_buddy.py --strategy=$strat -t \
    --dict_file=wordlists/lewdle_answers.txt \
    > regression_tests/$strat/lewdle_answers.txt
end
