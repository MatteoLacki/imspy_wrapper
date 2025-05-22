import pandas as pd

train = pd.read_csv(
    "/home/matteo/Projects/koina/dlomix-resources/example_datasets/RetentionTime/proteomeTools_train.csv"
)
val = pd.read_csv(
    "/home/matteo/Projects/koina/dlomix-resources/example_datasets/RetentionTime/proteomeTools_val.csv"
)
train_val = pd.read_csv(
    "/home/matteo/Projects/koina/dlomix-resources/example_datasets/RetentionTime/proteomeTools_train_val.csv"
)

train_val_comb = pd.concat([train, val], ignore_index=True).sort_values(
    ["sequence", "irt"], ignore_index=True
)
train_val = train_val.sort_values(["sequence", "irt"], ignore_index=True)

all(train_val_comb == train_val)
