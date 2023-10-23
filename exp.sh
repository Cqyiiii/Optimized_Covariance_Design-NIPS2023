for data in "stf" "cornell"
do
for res in 2 5 10
do
for md in "linear" "multi"
do
for gamma in 0.5 1 2
do
python main.py --model $md --gamma $gamma --resolution $res --dataset $data
done
done
done
done

