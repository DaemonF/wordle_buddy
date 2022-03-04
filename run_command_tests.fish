#!/usr/bin/env fish

set strategies freq clues div

function test_case --argument-names case_name input output flags
  test -n "$case_name" || echo_error 'Must provide a test case name.' || return
  test -n "$input" || echo_error 'Must provide an input.' || return
  test -n "$output" || echo_error 'Must provide an output.' || return
  set -l flags $argv[4..-1]

  echo "Checking $case_name..."
  ./wordle_buddy.py $flags \
    < $input \
    > $output.tmp \
    && mv $output.tmp $output
end

set dir command_tests
for strat in $strategies
  set common_args \
    --game=hermetic \
    --strategy=$strat

  test_case "default mode $strat " \
    $dir/default_mode.input \
    $dir/default_mode-$strat.output \
    $common_args \
    || exit

  test_case "interactive mode $strat" \
    $dir/interactive_mode-$strat.input \
    $dir/interactive_mode-$strat.output \
    -i $common_args \
    || exit

  test_case "answer mode $strat" \
    /dev/null \
    $dir/answer_mode-$strat.output \
    --answer canoe $common_args \
    || exit

  test_case "test mode $strat" \
    /dev/null \
    $dir/test_mode-$strat.output \
    -t --sampling 2 $common_args \
    || exit

  test_case "profile mode $strat" \
    /dev/null \
    $dir/profile_mode-$strat.output \
    --answer mason --profile $common_args \
    || exit

  test_case "cython mode $strat" \
    /dev/null \
    $dir/cython_mode-$strat.output \
    --answer ingot --cython $common_args \
    || exit
end
