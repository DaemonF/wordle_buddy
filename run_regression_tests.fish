#!/usr/bin/env fish

for strat in {freq,clues,bifur}
  for game in {lewdle,wordle}
    ./wordle_buddy.py -t \
        --game=$game --strategy=$strat \
      > regression_tests/$game-$strat.txt
  end
end
