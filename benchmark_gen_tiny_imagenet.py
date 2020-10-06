"""poisoning-benchmark runscript creator."""


# ######## BREW POISONS #################### 250
file_name = 'poison_brewing250'
benchmark_file = f'benchmark_{file_name}.sh'
experiments = []

def _command(idx, name, ensemble, model, add_options):
    setup = f'--name bm_{name} --benchmark ../poisoning-benchmark/scaling_setups/tinyimagenet/num_poisons=250/setups.pickle --save benchmark --vruns 0'
    return f'python brew_poison.py {setup} --dataset TinyImageNet --data_path /fs/cml-datasets/tiny_imagenet --cache_dataset --ensemble {ensemble} --net {model} {add_options} --eps 8 --benchmark_idx {idx}'

with open(benchmark_file, 'w+') as file:
    source_model = 'ResNet34'
    add_options = ''
    ensemble = 1
    experiment_name = 'base250'
    experiments.append((experiment_name, source_model))
    for i in range(100):
        file.write(_command(i, experiment_name, ensemble, source_model, add_options) + '\n')



# ######## BREW POISONS #################### 100
file_name = 'poison_brewing100'
benchmark_file = f'benchmark_{file_name}.sh'
experiments = []

def _command(idx, name, ensemble, model, add_options):
    setup = f'--name bm_{name} --benchmark ../poisoning-benchmark/scaling_setups/tinyimagenet/num_poisons=100/setups.pickle --save benchmark --vruns 0'
    return f'python brew_poison.py {setup} --dataset TinyImageNet --data_path /fs/cml-datasets/tiny_imagenet --cache_dataset --ensemble {ensemble} --net {model} {add_options} --eps 8 --benchmark_idx {idx}'

with open(benchmark_file, 'w+') as file:
    source_model = 'ResNet34'
    add_options = ''
    ensemble = 1
    experiment_name = 'base100'
    experiments.append((experiment_name, source_model))
    for i in range(100):
        file.write(_command(i, experiment_name, ensemble, source_model, add_options) + '\n')

# ###### AVI will be running the evaluation ##########################
