CUDA_VISIBLE_DEVICES=1 python main.py\
 --model_name corediff\
 --run_name UI_pet\
 --batch_size 1\
 --max_iter 400000\
 --test_dataset UI\
 --test_batch_size 1\
 --test_id 294\
 --context\
 --only_adjust_two_step\
 --dose 25\
 --save_freq 2500\
 --mode test\
 --test_iter 90000

# CUDA_VISIBLE_DEVICES=0 python main.py\
# --model_name corediff\
# --run_name UI_pet\
# --batch_size 1\
# --max_iter 400000\
# --test_dataset UI\
# --test_batch_size 1\
# --test_id 294\
# --context\
# --only_adjust_two_step\
# --dose 25\
# --save_freq 2500\
# --mode test\
# --test_iter 92500
#
# CUDA_VISIBLE_DEVICES=0 python main.py\
# --model_name corediff\
# --run_name UI_pet\
# --batch_size 1\
# --max_iter 400000\
# --test_dataset UI\
# --test_batch_size 1\
# --test_id 294\
# --context\
# --only_adjust_two_step\
# --dose 25\
# --save_freq 2500\
# --mode test\
# --test_iter 95000
