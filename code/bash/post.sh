gpu_id=-1
species='human'
atlas='Glasser'
metric='fiber_count'
approach='edr'
for seed in {1..100}
do
    for step in 4 5
    do
        python ../script/running.py $gpu_id $species $atlas $metric $approach $seed $step 2>> "../logs/err-1hm.log" 
    done
done