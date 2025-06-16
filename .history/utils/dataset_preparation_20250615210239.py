dataset_file = open("../.dependencies/pretraining_dataset.txt", "r")
dataset = dataset_file.read()
chars = sorted(list(set(dataset_file.read())))
