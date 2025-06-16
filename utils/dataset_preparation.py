dataset_file = open("./.dependencies/pretraining_dataset.txt", "r")
dataset = dataset_file.read()

chars = sorted(list(set(dataset)))
charset_size = len(chars)

print("Number of unique characters:", charset_size)
print(chars)
