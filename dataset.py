import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor


M = torch.tensor([3801.7295, 1594.2151, 3851.0134, 1507.4651], dtype=torch.float64)
STD = torch.tensor([2200.4813, 775.9291, 2175.2612, 775.4867], dtype=torch.float64)


def scatterplot_tensor_dataset(X: torch.Tensor, skip: float = 0.5):
    sns.set()

    every_n = round(1 / skip)

    X_smaller = X[::every_n]

    fig, ax = plt.subplots(figsize=(10, 10))

    sns.scatterplot(
        x=X_smaller[:, 0], y=X_smaller[:, 1], color="blue", size=4, ax=ax
    )

    sns.scatterplot(
        x=X_smaller[:, 2], y=X_smaller[:, 3], color="red", size=4, alpha=0.5, ax=ax
    )
    plt.show()


def create_table(test: bool):
    whole_dataset = pd.DataFrame()

    substr = "" #"dynamic" if test else "stat"

    path = os.path.join("Task", "F8")
    files_at_path = [
        f for f in os.listdir(path) if f.endswith("xlsx") and (substr in f)
    ]
    for file in tqdm.tqdm(files_at_path):
        file_path = os.path.join(path, file)
        df = pd.read_excel(file_path)[
            [
                "data__coordinates__x",
                "data__coordinates__y",
                "reference__x",
                "reference__y",
            ]
        ]
        whole_dataset = whole_dataset.append(df, ignore_index=True)

    return whole_dataset


def create_train_test_dataloaders(fake_examles: bool = False):
    """If `fake_examles` is set to True fake data will be added to the dataset."""
    
    dataset = Dataset(fake_examles=fake_examles)

    n = len(dataset)
    train_n = round(.70 * n)
    val_n = (n - train_n) // 2
    test_n = n - train_n - val_n

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_n, val_n, test_n])

    train_dataloader = DataLoader(train_dataset, batch_size=4096 * 8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4096 * 8, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=4096 * 8, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


def generate_fake_examples(original: torch.Tensor, n=4):
    """Generates fake learning examples with labels equal to mean of n closest neighbors"""

    k_neighbors_regr = KNeighborsRegressor(n_neighbors=n, weights="uniform", n_jobs=12)

    estimated_pos = original[:, [0, 1]]
    real_pos = original[:, [2, 3]]

    k_neighbors_regr.fit(estimated_pos, real_pos - estimated_pos)

    all_x = (torch.rand(200) - 0.5) * 4
    all_y = (torch.rand(200) - 0.5) * 4

    new_data_coor = torch.tensor([[x, y] for x in all_x for y in all_y])

    # To prevent oom errors
    step = 1000
    corrections = torch.zeros_like(new_data_coor)
    for i in tqdm.tqdm(range(0, len(corrections), step)):
        if i + step < len(corrections):
            corrections[i : i + step] = torch.tensor(
                k_neighbors_regr.predict(new_data_coor[i : i + step])
            )
        else:
            corrections[i:] = torch.tensor(k_neighbors_regr.predict(new_data_coor[i:]))

    new_references = new_data_coor + corrections

    tensor_dataset = torch.cat((new_data_coor, new_references), dim=1)

    scatterplot_tensor_dataset(tensor_dataset)

    return tensor_dataset


class Dataset(Dataset):
    def __init__(self, fake_examles):
        """The main dataset. Will add fake learning examples if `fake` is True."""

        if "tensor_dataset.pt" not in os.listdir():
            self.dataset_table = create_table(test=False)

            print(f"Appended {len(self.dataset_table)} rows in total.")

            self.dataset_table = self.dataset_table.reset_index(drop=True)
            self.dataset_table = self.dataset_table.dropna(axis="rows")

            print(f"\tAfter deleting NaNs: {len(self.dataset_table)}.")

            self.dataset_table = self.dataset_table.applymap(float)
            self.tensor_dataset = torch.tensor(self.dataset_table.values)

            print("Standardizing dataset")

            self.tensor_dataset = (self.tensor_dataset - M) / STD

            if fake_examles:
                fake_examples_tensor = generate_fake_examples(self.tensor_dataset)

                self.tensor_dataset = torch.cat((self.tensor_dataset, fake_examples_tensor), dim=0)

            torch.save(self.tensor_dataset, "tensor_dataset.pt")

        else:
            self.tensor_dataset = torch.load("tensor_dataset.pt")

    def __len__(self):
        return len(self.tensor_dataset)

    def __getitem__(self, idx):
        real_position = self.tensor_dataset[idx, [2, 3]]
        est_position = self.tensor_dataset[idx, [0, 1]]
        difference = real_position - est_position

        return {"diff": difference, "est": est_position}