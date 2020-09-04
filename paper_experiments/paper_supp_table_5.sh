## Ablation Study
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2000000000 --budget 0.01 --restarts 8 --ensemble 1 --name ablation_steps --attackiter 50
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2100000000 --budget 0.01 --restarts 8 --ensemble 1 --name ablation_steps --attackiter 50
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2110000000 --budget 0.01 --restarts 8 --ensemble 1 --name ablation_steps --attackiter 50
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111000000 --budget 0.01 --restarts 8 --ensemble 1 --name ablation_steps --attackiter 50
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111100000 --budget 0.01 --restarts 8 --ensemble 1 --name ablation_steps --attackiter 50
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111110000 --budget 0.01 --restarts 8 --ensemble 1 --name ablation_steps --attackiter 50
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111111000 --budget 0.01 --restarts 8 --ensemble 1 --name ablation_steps --attackiter 50
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111111100 --budget 0.01 --restarts 8 --ensemble 1 --name ablation_steps --attackiter 50
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111111110 --budget 0.01 --restarts 8 --ensemble 1 --name ablation_steps --attackiter 50
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111111111 --budget 0.01 --restarts 8 --ensemble 1 --name ablation_steps --attackiter 50

python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2000000000 --budget 0.01 --restarts 8 --ensemble 1 --name ablation_cw --target_criterion cw
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2100000000 --budget 0.01 --restarts 8 --ensemble 1 --name ablation_cw --target_criterion cw
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2110000000 --budget 0.01 --restarts 8 --ensemble 1 --name ablation_cw --target_criterion cw
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111000000 --budget 0.01 --restarts 8 --ensemble 1 --name ablation_cw --target_criterion cw
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111100000 --budget 0.01 --restarts 8 --ensemble 1 --name ablation_cw --target_criterion cw
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111110000 --budget 0.01 --restarts 8 --ensemble 1 --name ablation_cw --target_criterion cw
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111111000 --budget 0.01 --restarts 8 --ensemble 1 --name ablation_cw --target_criterion cw
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111111100 --budget 0.01 --restarts 8 --ensemble 1 --name ablation_cw --target_criterion cw
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111111110 --budget 0.01 --restarts 8 --ensemble 1 --name ablation_cw --target_criterion cw
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111111111 --budget 0.01 --restarts 8 --ensemble 1 --name ablation_cw --target_criterion cw

python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2000000000 --budget 0.01 --restarts 8 --ensemble 1 --name ablation_l2 --loss SE
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2100000000 --budget 0.01 --restarts 8 --ensemble 1 --name ablation_l2 --loss SE
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2110000000 --budget 0.01 --restarts 8 --ensemble 1 --name ablation_l2 --loss SE
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111000000 --budget 0.01 --restarts 8 --ensemble 1 --name ablation_l2 --loss SE
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111100000 --budget 0.01 --restarts 8 --ensemble 1 --name ablation_l2 --loss SE
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111110000 --budget 0.01 --restarts 8 --ensemble 1 --name ablation_l2 --loss SE
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111111000 --budget 0.01 --restarts 8 --ensemble 1 --name ablation_l2 --loss SE
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111111100 --budget 0.01 --restarts 8 --ensemble 1 --name ablation_l2 --loss SE
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111111110 --budget 0.01 --restarts 8 --ensemble 1 --name ablation_l2 --loss SE
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111111111 --budget 0.01 --restarts 8 --ensemble 1 --name ablation_l2 --loss SE
