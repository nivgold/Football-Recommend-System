import preprocessing.prepare_dataset as prepare

leagues_datasets = prepare.preprocess()

for league_name, dataset in leagues_datasets.items():
    print(dataset)
    print(dataset.iloc[:, 0:-1])
    break