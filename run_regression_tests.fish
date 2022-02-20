#!/usr/bin/env fish

for game in {lewdle, wordle_hard, wordle}
  for strat in {freq, clues, bifur}
    set path regression_tests/$game-$strat.txt
    echo "Running $path:"
    ./wordle_buddy.py -t \
      --game=$game \
      --strategy=$strat \
      > $path
  end
end
