import os
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class BaseDataset(Dataset):
    def __init__(self, csv_path, split, label_dict, label_col='type', ignore=[]):
        """
        Args:
            csv_path (str): Path to the csv file with annotations.
            split (pd.DataFrame): Train/val/test split. 
            label_dict (dict): Dictionary with key, value pairs for converting str labels to int. 
            label_col (str, optional): Label column. Defaults to 'type'.
            ignore (list, optional): Ignored labels. Defaults to [].
        """        
        slide_data = pd.read_csv(csv_path)
        slide_data = self._df_prep(slide_data, label_dict, ignore, label_col)
        assert len(split) > 0, "Split should not be empty!"
        mask = slide_data['slide_id'].isin(split.tolist())
        self.slide_data = slide_data[mask].reset_index(drop=True)
        self.n_cls = len(set(label_dict.values()))
        self.slide_cls_ids = self._cls_ids_prep()
        self._print_info()

    def __len__(self):
        return len(self.slide_data)

    def __getitem__(self, idx):
        return None

    def _print_info(self):
        print("Number of classes: {}".format(self.n_cls))
        print("Slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))

    def _cls_ids_prep(self):
        slide_cls_ids = [[] for i in range(self.n_cls)]
        for i in range(self.n_cls):
            slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]
        return slide_cls_ids

    def get_label(self, ids):
        return self.slide_data['label'][ids]

    @staticmethod
    def _df_prep(data, label_dict, ignore, label_col):
        if label_col != 'label':
            data['label'] = data[label_col].copy()

        mask = data['label'].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        for i in data.index:
            key = data.loc[i, 'label']
            data.at[i, 'label'] = label_dict[key]
        return data


class PatchFeatureDataset(BaseDataset):
    def __init__(self, data_path, low_mag, mid_mag, high_mag, **kwargs):
        """
        Args:
            data_path (str): Path to the data. 
            low_mag (str): Low magnifications. 
            mid_mag (str): Middle magnifications.
            high_mag (str): High magnifications.
        """        
        super(PatchFeatureDataset, self).__init__(**kwargs)
        self.data_path = data_path
        self.low_mag = low_mag
        self.mid_mag = mid_mag
        self.high_mag = high_mag
        
    def __len__(self):
        return len(self.slide_data)
    
    def __getitem__(self, idx):
        slide_id = self.slide_data['slide_id'][idx]
        label = self.slide_data['label'][idx]
        
        with h5py.File(os.path.join(self.data_path, '{}.h5'.format(slide_id)),'r') as hdf5_file:
            low_mag_feats = hdf5_file[f'{self.low_mag}_patches'][:]
            mid_mag_feats = hdf5_file[f'{self.mid_mag}_patches'][:]
            high_mag_feats = hdf5_file[f'{self.high_mag}_patches'][:]
        
        return torch.from_numpy(low_mag_feats), torch.from_numpy(mid_mag_feats), torch.from_numpy(high_mag_feats), label


class NewPatchFeatureDataset(Dataset):
    # def __init__(self, csv_path, split, label_dict, label_col='type', ignore=[]):
    def __init__(self, csv_path, split, data_path, low_mag, mid_mag, high_mag, cache_all_patches: bool = True):
        """
        A modified dataset to work with our differing format of pre-processed slides, as well as survival prediction
        rather than subtype classification. It works via the same API, but with a different implementation. Mainly taken
        from our own dataset code.
        """
        frame = pd.read_csv(csv_path, compression="zip")
        # Prune invalid rows with no corresponding slide
        invalid_labels = []
        example_pow = float(low_mag[:-1])
        for i in range(len(frame)):
            slide_id = frame.iloc[i].slide_id
            x = ".".join(slide_id.split(".")[:-1])
            path = os.path.join(data_path, x + f"_{example_pow:.3f}.pt")

            if not os.path.isfile(path):
                invalid_labels.append(i)
        print(f"Ignoring {len(invalid_labels)} rows without files.")
        frame.drop(invalid_labels, inplace=True)

        # Extract one random slide per patient.
        # Note: this operation is deterministic
        frame = frame.drop_duplicates(subset='case_id')
        frame.reset_index(drop=True, inplace=True)

        # Filter to necessary columns
        nbins = 4
        frame = frame[["case_id", "slide_id", "survival_months", "censorship", "oncotree_code"]]
        _, bins = pd.qcut(frame.survival_months, nbins, labels=False, retbins=True)

        # Filter to our split
        match_on = 'case_id'
        frame = frame[frame[match_on].isin(split)]
        frame.reset_index(inplace=True, drop=True)
        self.frame = frame

        self.q_survival_months = pd.cut(frame.survival_months, bins, labels=False, include_lowest=True)
        self.survival_months = frame.survival_months
        self.censorship = torch.tensor(frame.censorship.to_numpy(np.int64), dtype=torch.long)
        self.slide_ids = frame.slide_id

        self.data_path = data_path
        self.low_mag = low_mag
        self.mid_mag = mid_mag
        self.high_mag = high_mag

        self.power_levels = [low_mag, mid_mag, high_mag]  # Unfortunately the rest of this codebase hardcodes 3 levels

        for i in range(len(self.power_levels)):
            if isinstance(self.power_levels[i], str):
                self.power_levels[i] = float(self.power_levels[i][:-1])

        if cache_all_patches:
            print("Loading all patches...")
            self.data = []
            for i in tqdm(range(len(self))):
                self.data.append(self._load_patches(i))
        else:
            self.data = None

    def __len__(self):
        return len(self.slide_ids)

    def _load_patches(self, idx: int):
        """
        Loads all patches for a given slide. Returns a tuple of Tensors of shape (N x B), (M^2 N x B), ...
        """
        slide_id = self.slide_ids[idx][:-4]  # (remove .svs from str)

        data = []
        for power in self.power_levels:
            path = os.path.join(self.data_path, slide_id + f"_{power:.3f}.pt")
            data.append(torch.load(path))

        d0 = data[0]
        mask = torch.sum(d0.abs(), dim=-1) > 0  # [H x W] bool
        xs, ys = mask.nonzero(as_tuple=True)

        output = []

        for i, (tensor, power) in enumerate(zip(data, self.power_levels)):
            scale = round(power / self.power_levels[0])
            h, w, _ = tensor.shape

            # Scale up coordinates. E.g. x -> [2x, 2x+1] on the second iter
            scaled_xs = (xs * scale).view(-1, 1) + torch.arange(scale).view(1, -1)
            scaled_ys = (ys * scale).view(-1, 1) + torch.arange(scale).view(1, -1)

            scaled_coords = torch.stack([torch.cartesian_prod(sx, sy) for sx, sy in zip(scaled_xs, scaled_ys)], dim=0)
            patch_coords = scaled_coords.view(-1, 2)

            # Clamp coordinates to be within bounds
            #  (coords are *almost always* within bounds. there's just the occasional off-by-one error
            #   at the edges due to slide dimensions not being perfect powers of two.)
            out_of_bounds = (patch_coords[:, 0] >= tensor.shape[0]) | (patch_coords[:, 1] >= tensor.shape[1])
            patch_coords[out_of_bounds] *= 0  # just query (0, 0), we will zero them anyway after
            gathered_patches = tensor[patch_coords[:, 0], patch_coords[:, 1]]
            gathered_patches[out_of_bounds] *= 0

            output.append(gathered_patches)

        return output

    # Return: patch fts (N x D, N' x D, N'' x D), survival info
    @torch.no_grad()
    def __getitem__(self, idx):
        if self.data is None:
            output = self._load_patches(idx)
        else:
            output = self.data[idx]

        survival_data = {
            "survival_bin": self.q_survival_months[idx],
            "survival": self.survival_months[idx],
            "censored": self.censorship[idx]
        }

        return output, survival_data


