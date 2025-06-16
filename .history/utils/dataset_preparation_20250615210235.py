dataset_file = open("../.dependencies/pretraining_dataset.txt", "r")

chars = sorted(list(set(dataset_file.read())))
