import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import tqdm
import matplotlib.pyplot as plt


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

    substr = "dynamic" if test else "stat"

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


def create_train_test_dataloaders(fake: bool = False):
    """If `fake` is set to True fake data will be added to the dataset."""
    
    train_dataset = Dataset(fake)
    test_dataset = TestDataset()

    train_dataloader = DataLoader(train_dataset, batch_size=32_000, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32_000, shuffle=False)

    return train_dataloader, test_dataloader


def generate_fake_examples(original: torch.Tensor, n=4):
    """Generates fake learning examples with labels equal to mean of n closest neighbors"""

    k_neighbors_regr = KNeighborsRegressor(n_neighbors=n, weights="uniform", n_jobs=12)

    estimated_pos = available_data_stand[:, [0, 1]]
    real_pos = available_data_stand[:, [2, 3]]

    k_neighbors_regr.fit(estimated_pos, real_pos - estimated_pos)

    all_x = (torch.rand(500) - 0.5) * 4
    all_y = (torch.rand(500) - 0.5) * 4

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
    def __init__(self, fake):
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

            if fake:
                fake_examples = generate_fake_examples(self.tensor_dataset)

                self.tensor_dataset = torch.cat((tensor_dataset, fake_examples), dim=0)

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


class TestDataset(Dataset):
    def __init__(self):
        if "test_tensor_dataset.pt" not in os.listdir():
            self.dataset_table = create_table(test=True)

            print(f"Appended {len(self.dataset_table)} rows in total.")

            self.dataset_table = self.dataset_table.reset_index(drop=True)
            self.dataset_table = self.dataset_table.dropna(axis="rows")

            print(f"\tAfter deleting NaNs: {len(self.dataset_table)}.")

            self.dataset_table = self.dataset_table.applymap(float)
            self.tensor_dataset = torch.tensor(self.dataset_table.values)
            print("Standardizing dataset:")

            self.tensor_dataset = (self.tensor_dataset - M) / STD
            list_fromtensor = self.tensor_dataset.tolist()

            torch.save(self.tensor_dataset, "test_tensor_dataset.pt")

        else:
            self.tensor_dataset = torch.load("test_tensor_dataset.pt")

        mae = torch.nn.functional.l1_loss(
            self.tensor_dataset[:, [0, 1]], self.tensor_dataset[:, [2, 3]]
        )

        print(f"Loaded test dataset. MAE(X, Y) =", mae.item())

    def __len__(self):
        return len(self.tensor_dataset)

    def __getitem__(self, idx):
        real_position = self.tensor_dataset[idx, [2, 3]]
        est_position = self.tensor_dataset[idx, [0, 1]]
        difference = real_position - est_position

        return {"diff": difference, "est": est_position}
