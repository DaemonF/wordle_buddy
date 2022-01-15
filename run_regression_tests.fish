#!/usr/bin/env fish

for strat in {freq,clues}
  ./wordle_buddy.py --strategy=$strat -t \
    --answer_file=wordlists/wordle_answers.txt \
    > regression_tests/$strat/wordle_answers.txt
  ./wordle_buddy.py --strategy=$strat -t \
    --sampling=10 \
    > regression_tests/$strat/sampling_10.txt
  if [ "$argv[1]" = '--full' ]
    ./wordle_buddy.py --strategy=$strat -t \
      --sampling=1 \
      > regression_tests/$strat/full.txt
  end
end
if [ "$argv[1]" = '--full' ]
  cat regression_tests/*/sampling_10.txt \
    | email_me 'Wordle buddy regression tests done.'
end
