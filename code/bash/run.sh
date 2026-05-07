gpu_id=1
species='human'
atlas='Glasser'
metric='fiber_count'
approach='edr'
for seed in {1..100}
do
    for step in 0 1 2 3
    do
        python ../script/running.py $gpu_id $species $atlas $metric $approach $seed $step 2>> "../logs/err1hm.log" 
    done
done