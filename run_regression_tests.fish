#!/usr/bin/env fish

set games lewdle wordle_hard wordle
set strategies freq clues div

set dir regression_tests
for game in $games
  for strat in $strategies
    set path $dir/$game-$strat.txt
    echo "Running regression test $game-$strat..."
    ./wordle_buddy.py -t \
      --cython \
      --game=$game \
      --strategy=$strat \
      > $path.tmp \
      && mv $path.tmp $path
  end
end
