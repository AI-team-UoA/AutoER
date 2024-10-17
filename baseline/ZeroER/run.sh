#!/bin/bash

log=$1


nohup python zeroer.py rest1_rest2 D1 "$log" --run_transitivity --LR_dup_free --sep "|"  > D1.log 2>&1 &
wait
nohup python zeroer.py abt_buy D2 "$log" --run_transitivity --LR_dup_free --sep "|"  > D2.log 2>&1 &
wait
echo "Running ZeroER on D3"
nohup python zeroer.py amazon_gp D3 "$log" --run_transitivity --LR_dup_free --sep "#"  > D3.log 2>&1 &
wait
nohup python zeroer.py dblp_acm D4 "$log" --run_transitivity --LR_dup_free --sep "%"  > D4.log 2>&1 &
wait
echo "Running ZeroER on D5"
nohup python -u  zeroer.py imdb_tvdb D5 "$log" --run_transitivity --LR_dup_free --sep "|" > D5.log 2>&1 &
wait
echo "Done with D5"
echo "Running ZeroER on D6"
nohup python -u zeroer.py tmdb_tvdb D6 "$log" --run_transitivity --LR_dup_free --sep "|" > D6.log 2>&1 &
wait
echo "Done with D6"
echo "Running ZeroER on D7"
nohup python -u zeroer.py imdb_tmdb D7 "$log" --run_transitivity --LR_dup_free --sep "|" > D7.log 2>&1 &
wait
echo "Done with D7"
echo "Running ZeroER on D8"
nohup python -u zeroer.py walmart_amazon D8 "$log" --run_transitivity --LR_dup_free --sep "|" > D8.log 2>&1 &
wait
echo "Done with D8"
nohup python zeroer.py dblp_scholar D9 "$log" --run_transitivity --LR_dup_free --sep ">" > D9.log 2>&1 &
wait
nohup python zeroer.py imdb_dbpedia D10 "$log" --run_transitivity --LR_dup_free --sep "|" > D10.log 2>&1 &
wait
