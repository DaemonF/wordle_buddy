#!/usr/bin/env fish

for strat in {freq,clues,bifur}
  ./wordle_buddy.py --strategy=$strat -t \
    > regression_tests/$strat/wordle_answers.txt
end
