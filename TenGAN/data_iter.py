
import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import rdBase
from mol_metrics import Tokenizer
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from multiprocessing import Pool
rdBase.DisableLog('rdApp.error')


# ============================================================================
# Helper function for parallel SMILES normalization
def normalize_smiles(smi):
    """RDKitでSMILESを正規化（並列処理用）"""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            return Chem.MolToSmiles(mol)
        return smi
    except:
        return smi


# ============================================================================
# Build a dataset, inherited the methods of Dataset, that returns tensors for Genenerator
class GenDataset(Dataset):
    
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        smile = self.data[idx]
        tensor = self.tokenizer.encode(smile)
        return tensor


# ============================================================================
# Definite the data loader for Generator
class GenDataLoader(LightningDataModule):

    def __init__(self, positive_file, train_size=4800, batch_size=64):
        super().__init__()
        self.tokenizer = Tokenizer()
        self.train_size = train_size
        self.val_size = 200
        self.batch_size = batch_size
        self.positive_file = positive_file
    
    # Randomize the same molecules to different SMILES representations for sufficently training (c1cccc([N+]([O-])=O)c1 -> c1ccccc1[N+](=O)[O-])
    def randomize_smiles_atom_order(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        atom_idxs = list(range(mol.GetNumAtoms()))
        np.random.shuffle(atom_idxs)
        mol = Chem.RenumberAtoms(mol,atom_idxs)
        return Chem.MolToSmiles(mol, canonical=False)
    
    def custom_collate_and_pad(self, batch):
        # Batch is a list of vectorized smiles
        tensors = [torch.tensor(l) for l in batch]
        # Pad the different lengths of tensors to the maximum length (each column is a sequence)
        tensors = torch.nn.utils.rnn.pad_sequence(tensors) # [maxlength, batch_size]
        return tensors
    
    def setup(self):
        # Load data
        self.data = pd.read_csv(self.positive_file, nrows = self.val_size + self.train_size, names = ['smiles'])
        # Atom order randomize SMILES
        self.data['smiles'] = self.data['smiles'].apply(self.randomize_smiles_atom_order)
        # Initialize Tokenizer
        self.tokenizer.build_vocab()
        # Create splits for train/val
        idxs = np.array(range(len(self.data['smiles'])))
        np.random.shuffle(idxs)
        val_idxs, train_idxs = idxs[:self.val_size], idxs[self.val_size:self.val_size + self.train_size]
        self.train_data = self.data['smiles'][train_idxs]
        self.train_data.reset_index(drop=True, inplace=True)
        self.val_data = self.data['smiles'][val_idxs]
        self.val_data.reset_index(drop=True, inplace=True)
        
    def train_dataloader(self):
        dataset = GenDataset(self.train_data, self.tokenizer)
        # pin_memory=True: speed the dataloading, num_workers: multithreading for dataloading
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            collate_fn=self.custom_collate_and_pad,
            num_workers=4,              # 40 → 4: Reduce CPU resource contention
            persistent_workers=True,    # Keep workers alive between epochs
            prefetch_factor=2           # Prefetch 2 batches per worker
        )

    def val_dataloader(self):
        dataset = GenDataset(self.val_data, self.tokenizer)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            collate_fn=self.custom_collate_and_pad,
            shuffle=False,
            num_workers=4,              # 40 → 4: Reduce CPU resource contention
            persistent_workers=True,    # Keep workers alive between epochs
            prefetch_factor=2           # Prefetch 2 batches per worker
        )


# ============================================================================
# Build a dataset, inherited the methods of Dataset, that returns tensors for Discriminator
class DisDataset(Dataset):
    
    def __init__(self, pairs, tokenizer):
        self.data, self.labels = pairs['smiles'], pairs['labels']
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        smile = self.data[idx]
        # Remove the start token and end token
        tensor = self.tokenizer.encode(smile)[1:-1] 
        label = self.labels[idx]
        return tensor, label


# ============================================================================
# Definite the data loader for the Discriminator
class DisDataLoader(LightningDataModule):

    def __init__(self, positive_file, negative_file, batch_size=2):
        super().__init__()
        self.tokenizer = Tokenizer()
        self.batch_size = batch_size
        self.positive_file = positive_file
        self.negative_file = negative_file
        self._normalize_pool = None  # Pool reuse for performance optimization
    
    def custom_collate_and_pad(self, batch):
        # Zip a batch of data
        smiles, labels = zip(*batch) 
        # Batch is a list of vectorized smiles
        tensors = [torch.LongTensor(smi) for smi in smiles]
        # Pad the different lengths of tensors to the maximum length (each column is a sequence)
        tensors = torch.nn.utils.rnn.pad_sequence(tensors).transpose(0, 1) # [batch_size, maxlength]
        labels = torch.LongTensor(labels)
        return tensors, labels
    
    def setup(self, force_reload=False, use_parallel=True):
        # キャッシュの初期化または強制リロード
        if not hasattr(self, '_positive_cached') or force_reload:
            # Positive dataは変わらないので一度だけ読み込み
            self.positive_data = pd.read_csv(self.positive_file, names = ['smiles'])
            self._positive_cached = True

        # Negative data (生成データ) のみ毎回更新
        self.negative_data = pd.read_csv(self.negative_file, names = ['smiles'])

        # Keep the unique order for the NEGATIVE dataset
        # 最適化: 並列処理でSMILES正規化
        if use_parallel and len(self.negative_data) > 100:
            # Pool再利用による高速化（D_STEP回のsetup呼び出しでプロセス起動オーバーヘッド削減）
            if self._normalize_pool is None:
                self._normalize_pool = Pool()
            valid_smiles = self._normalize_pool.map(normalize_smiles, self.negative_data['smiles'])
            self.negative_data = pd.DataFrame(valid_smiles, columns=['smiles'])
        else:
            # シーケンシャル処理（小データまたは並列無効時）
            valid_smiles = []
            for s in self.negative_data['smiles']:
                mol = Chem.MolFromSmiles(s)
                if mol is not None:
                    valid_smiles.append(Chem.MolToSmiles(mol))
                else:
                    valid_smiles.append(s)
            self.negative_data = pd.DataFrame(valid_smiles, columns=['smiles'])

        self.data = pd.concat([self.positive_data['smiles'], self.negative_data['smiles']])
        self.labels = pd.DataFrame([1 for _ in range(len(self.positive_data))] + [0 for _ in range(len(self.negative_data))], columns=['labels'])
        self.pairs = list(zip(self.data, self.labels['labels']))

        # Initialize Tokenizer (一度だけ)
        if not hasattr(self, '_tokenizer_built'):
            self.tokenizer.build_vocab()
            self._tokenizer_built = True

        # Create splits for train/val
        np.random.shuffle(self.pairs)
        self.pairs = pd.DataFrame(self.pairs, columns=['smiles', 'labels'])
        self.train_data = self.pairs[:int(len(self.pairs)*0.9)]
        self.train_data.reset_index(drop=True, inplace=True)
        self.val_data = self.pairs[int(len(self.pairs)*0.9):]
        self.val_data.reset_index(drop=True, inplace=True)
        
    def train_dataloader(self):
        dataset = DisDataset(self.train_data, self.tokenizer)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            collate_fn=self.custom_collate_and_pad,
            num_workers=4,              # 0 → 4: Enable parallel data loading
            persistent_workers=True,    # Keep workers alive between epochs
            prefetch_factor=2           # Prefetch 2 batches per worker
        )

    def val_dataloader(self):
        dataset = DisDataset(self.val_data, self.tokenizer)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            collate_fn=self.custom_collate_and_pad,
            num_workers=4,              # 0 → 4: Enable parallel data loading
            persistent_workers=True,    # Keep workers alive between epochs
            prefetch_factor=2           # Prefetch 2 batches per worker
        )

    def __del__(self):
        """Cleanup Pool when object is destroyed"""
        if self._normalize_pool is not None:
            self._normalize_pool.close()
            self._normalize_pool.join()


