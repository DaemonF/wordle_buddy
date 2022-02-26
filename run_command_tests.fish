#!/usr/bin/env fish

set strategies freq clues bifur

set dir command_tests
for strat in $strategies
  set common_args --strategy=$strat

  echo "Checking default mode $strat..."
  ./wordle_buddy.py \
    $common_args \
    < $dir/default_mode.input \
    > $dir/default_mode-$strat.output \
    || exit

  echo "Checking interactive mode $strat..."
  ./wordle_buddy.py -i \
    $common_args \
    < $dir/interactive_mode.input \
    > $dir/interactive_mode-$strat.output \
    || exit

  echo "Checking answer mode $strat..."
  ./wordle_buddy.py --answer robot \
    $common_args \
    > $dir/answer_mode-$strat.output \
    || exit

  echo "Checking test mode $strat..."
  ./wordle_buddy.py -t --sampling=99999 \
    $common_args \
    > $dir/test_mode-$strat.output \
    || exit
end
