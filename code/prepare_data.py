import os
import _pickle as cPickle
from tqdm import tqdm
import numpy as np

def parse_bpseq_file(filepath):
    # Parses a single .bpseq file to extract the sequence and secondary structure (base pairs).
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        if not lines:
            print(f"Warning: File is empty, skipping {filepath}")
            return None, None

        sequence = []
        pairing_dict = {}

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) < 3:
                continue

            # BPSEQ is 1-indexed, we need to convert it to 0-indexed
            index = int(parts[0]) - 1
            base = parts[1]
            pair_index = int(parts[2]) - 1

            sequence.append(base)
            if pair_index != -1:
                pairing_dict[index] = pair_index
        
        structure = [[i, j] for i, j in pairing_dict.items()]
        
        return "".join(sequence), structure

    except Exception as e:
        print(f"Error parsing file {filepath}: {e}")
        return None, None

def process_bpseq_folder(bpseq_folder_path, mask_matrix_output_dir):
    # Processes a folder containing multiple .bpseq files...
    all_sequences = []
    all_structures = []

    bpseq_files = [f for f in os.listdir(bpseq_folder_path) if f.endswith('.bpseq')]
    
    print(f"Found {len(bpseq_files)} .bpseq files, starting processing...")

    for index, filename in enumerate(tqdm(bpseq_files, desc="Processing BPSEQ files")):
        filepath = os.path.join(bpseq_folder_path, filename)
        sequence, structure = parse_bpseq_file(filepath)

        if sequence and structure is not None:
            all_sequences.append(sequence.upper())
            all_structures.append(structure)
        
        mask_mat = create_mask_matrix(sequence.upper())
        mask_output_path = os.path.join(mask_matrix_output_dir, f'{index}.pickle')
        
        with open(mask_output_path, 'wb') as f:
            cPickle.dump(mask_mat, f)
        
    mask_matrix_idx = list(range(len(all_sequences)))

    data_dict = {
        'seq': all_sequences,
        'ss': all_structures,
        'mask_matrix_idx': mask_matrix_idx
    }

    return data_dict

def create_mask_matrix(sequence, min_loop_length=3):
    # Generates a list of allowed base pair coordinates based on the RNA sequence.
    
    sequence = sequence.upper().replace('T', 'U')
    L = len(sequence)
    if L == 0:
        return np.empty((0, 2), dtype=int)

    seq_array = np.array(list(sequence))
    seq_row = seq_array.reshape(L, 1)
    seq_col = seq_array.reshape(1, L)

    mask_AU = (seq_row == 'A') & (seq_col == 'U')
    mask_GC = (seq_row == 'G') & (seq_col == 'C')
    mask_GU = (seq_row == 'G') & (seq_col == 'U')
    
    pairing_mask = mask_AU | mask_GC | mask_GU
    full_pairing_mask = pairing_mask | pairing_mask.T

    indices = np.arange(L)
    dist_matrix = indices.reshape(1, L) - indices.reshape(L, 1)
    
    loop_constraint_mask = dist_matrix > min_loop_length
    
    final_mask = full_pairing_mask & loop_constraint_mask

    indices_i, indices_j = np.where(final_mask)
    
    pairing_coordinates = np.stack([indices_i, indices_j], axis=1)

    return pairing_coordinates.astype(int)

def main():    
    data_folder = 'data/bpseq_test/'
    bpseq_dir = f'{data_folder}/bpseq/'
    mask_matrix_dir = f'{data_folder}/mask_matrix/'
    output_filename = 'all.pickle'
    output_path = os.path.join(data_folder, output_filename)

    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(bpseq_dir, exist_ok=True)
    os.makedirs(mask_matrix_dir, exist_ok=True)

    processed_data = process_bpseq_folder(bpseq_dir, mask_matrix_dir)

    if processed_data:
        with open(output_path, 'wb') as f:
            cPickle.dump(processed_data, f)
        print("Saved successfully!")
        
    print(f"\nProcessing complete. Contains {len(processed_data['seq'])} sequences in total.")

if __name__ == '__main__':
    main()