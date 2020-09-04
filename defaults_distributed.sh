# ResNet distault
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29501 dist_brew_poison.py  --net ResNet18 --vruns 2 --name dist --poisonkey 1000000000
python -m torch.distributed.launch --nproc_per_node=4 --master_port=28502 dist_brew_poison.py  --net ResNet18 --vruns 2 --name dist --poisonkey 1100000000
python -m torch.distributed.launch --nproc_per_node=4 --master_port=27503 dist_brew_poison.py  --net ResNet18 --vruns 2 --name dist --poisonkey 1110000000
python -m torch.distributed.launch --nproc_per_node=4 --master_port=26504 dist_brew_poison.py  --net ResNet18 --vruns 2 --name dist --poisonkey 1111000000
python -m torch.distributed.launch --nproc_per_node=4 --master_port=25505 dist_brew_poison.py  --net ResNet18 --vruns 2 --name dist --poisonkey 1111100000
python -m torch.distributed.launch --nproc_per_node=4 --master_port=24506 dist_brew_poison.py  --net ResNet18 --vruns 2 --name dist --poisonkey 1111110000
python -m torch.distributed.launch --nproc_per_node=4 --master_port=23507 dist_brew_poison.py  --net ResNet18 --vruns 2 --name dist --poisonkey 1111111000
python -m torch.distributed.launch --nproc_per_node=4 --master_port=22508 dist_brew_poison.py  --net ResNet18 --vruns 2 --name dist --poisonkey 1111111100
python -m torch.distributed.launch --nproc_per_node=4 --master_port=21509 dist_brew_poison.py  --net ResNet18 --vruns 2 --name dist --poisonkey 1111111110
python -m torch.distributed.launch --nproc_per_node=4 --master_port=20510 dist_brew_poison.py  --net ResNet18 --vruns 2 --name dist --poisonkey 1111111111