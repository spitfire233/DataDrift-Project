from collections import defaultdict

def load_sysnet_mapping(path):
    """
    Loads the sysnet mapping from a file and returns three dictionaries:
    one that maps wnids to indices, one that maps indices to wnids and one that
    maps indices to string descriptions
    """
    wnid_to_idx = {}
    idx_to_wnid = {}
    idx_to_desc = {}
    
    with open(path, "r") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            wnid, description = line.split(' ', 1)
            wnid_to_idx[wnid] = idx
            idx_to_wnid[idx] = wnid
            idx_to_desc[idx] = description
    return wnid_to_idx, idx_to_wnid, idx_to_desc


def cap_examples_per_class(dataset, label_column = "label", max_per_class = 50):
    """
    Return a new dataset with a maximum of 50 examples per class
    """
    indices_per_class = defaultdict(list)

    for i, label in enumerate(dataset[label_column]):
        if len(indices_per_class[label]) < max_per_class:
            indices_per_class[label].append(i)

    # Get the selected indices
    selected_indices = [i for indices in indices_per_class.values() for i in indices]
    selected_indices.sort()
    # Return a view of the dataset with exactly 50 examples per class
    return dataset.select(selected_indices)

