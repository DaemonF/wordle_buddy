#!/usr/bin/env fish

set games lewdle wordle_hard wordle
set strategies freq clues bifur

set dir regression_tests
for game in $games
  for strat in $strategies
    set path $dir/$game-$strat.txt
    echo "Running regression test $game-$strat..."
    ./wordle_buddy.py -t \
      --game=$game \
      --strategy=$strat \
      > $dir/$game-$strat.txt
  end
end
