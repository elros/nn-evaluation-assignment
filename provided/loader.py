import time
import torch
from torch.utils.data import Dataset
import pandas as pd
from multiprocessing import Pool, cpu_count


class SingleProcessDataset(Dataset):
    def __init__(self, csv_file):
        start_time = time.time()
        print("Loading data using single process...")
        
        self.data = pd.read_csv(csv_file)
        self.features = torch.FloatTensor(self.data[['x1', 'x2', 'x3']].values)
        self.labels = torch.LongTensor(self.data['label'].values)
        
        # Calculate total load time
        self.load_time = time.time() - start_time
        print(f"Dataset loading completed in {self.load_time:.2f} seconds")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class MultiProcessDataset(SingleProcessDataset):
    def __init__(self, csv_file):
        start_time = time.time()
        print("Loading data using multi process...")

        ########### YOUR CODE HERE ############
        # Get total rows (subtract 1 for header)
        total_rows = sum(1 for _ in open(csv_file)) - 1

        # I'm using cpu_count as the number of chunks, but this could also be an import parameter.
        num_chunks = cpu_count()

        chunk_size = total_rows // num_chunks
        chunks = [(csv_file, i * chunk_size, min((i + 1) * chunk_size, total_rows)) 
                 for i in range(num_chunks)]

        with Pool(num_chunks) as pool:
            results = pool.map(self._process_chunk, chunks)

        # Combine results
        features_list, labels_list = zip(*results)
        self.features = torch.cat(features_list)
        self.labels = torch.cat(labels_list)
        ########### END YOUR CODE  ############
        
        # Calculate total load time
        self.load_time = time.time() - start_time
        print(f"Dataset loading completed in {self.load_time:.2f} seconds")

    @staticmethod
    def _process_chunk(args):
        csv_file, start, end = args  # Changed to unpack three values
        # Read only the required chunk using skiprows and nrows
        chunk_data = pd.read_csv(
            csv_file, 
            skiprows=range(1, start + 1),  # +1 to account for header
            nrows=end - start
        )
        return (
            torch.FloatTensor(chunk_data[['x1', 'x2', 'x3']].values),
            torch.LongTensor(chunk_data['label'].values)
        )
