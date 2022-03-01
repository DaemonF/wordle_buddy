#!/usr/bin/env fish

set strategies freq clues bifur

set dir command_tests
for strat in $strategies
  set common_args \
    --game=hermetic \
    --strategy=$strat

  echo "Checking default mode $strat..."
  ./wordle_buddy.py \
    $common_args \
    < $dir/default_mode.input \
    > $dir/default_mode-$strat.output \
    || exit

  echo "Checking interactive mode $strat..."
  ./wordle_buddy.py -i \
    $common_args \
    < $dir/interactive_mode-$strat.input \
    > $dir/interactive_mode-$strat.output \
    || exit

  echo "Checking answer mode $strat..."
  ./wordle_buddy.py --answer canoe \
    $common_args \
    > $dir/answer_mode-$strat.output \
    || exit

  echo "Checking test mode $strat..."
  ./wordle_buddy.py -t --sampling 2 \
    $common_args \
    > $dir/test_mode-$strat.output \
    || exit

  echo "Checking profile mode $strat..."
  ./wordle_buddy.py --answer mason --profile \
    $common_args \
    > $dir/profile_mode-$strat.output \
    || exit

  echo "Checking cython mode $strat..."
  ./wordle_buddy.py --answer ingot --cython \
    $common_args \
    > $dir/cython_mode-$strat.output \
    || exit
end
