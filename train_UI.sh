CUDA_VISIBLE_DEVICES=4 python main.py\
 --model_name corediff\
 --run_name UI_pet\
 --batch_size 4\
 --max_iter 150000\
 --patch_size 64\
 --train_dataset UI\
 --test_dataset UI\
 --test_id 294\
 --context\
 --only_adjust_two_step\
 --dose 25\
 --save_freq 2500


#!/bin/bash
#
## 循环三次执行命令
#for i in {1..3}
#do
#  echo "Running iteration $i..."
#  CUDA_VISIBLE_DEVICES=1 python main.py \
#    --model_name corediff \
#    --run_name UI_pet_run_$i \
#    --batch_size 4 \
#    --max_iter 50000 \
#    --patch_size 64 \
#    --train_dataset UI \
#    --test_dataset UI \
#    --test_id 294 \
#    --context \
#    --only_adjust_two_step \
#    --dose 25 \
#    --save_freq 2500
#  echo "Iteration $i completed."
#done



#for i in {1..3}
#do
#    echo "Running iteration $i..."
#    CUDA_VISIBLE_DEVICES=1 python main.py \
#        --model_name corediff \
#        --run_name UI_pet \
#        --batch_size 4 \
#        --max_iter 50000 \
#        --patch_size 64 \
#        --train_dataset UI \
#        --test_dataset UI \
#        --test_id 294
#        --context \
#        --only_adjust_two_step \
#        --dose 25 \
#        --save_freq 2500
#done