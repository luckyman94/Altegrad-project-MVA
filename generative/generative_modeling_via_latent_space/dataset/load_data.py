import pickle
from pathlib import Path

DATA_DIR = "NEED TO SET DATA DIRECTORY HERE"
def load_graphs(data_dir=DATA_DIR):
    data_dir = Path(data_dir)

    train_pkl = data_dir / "train_graphs.pkl"
    val_pkl = data_dir / "validation_graphs.pkl"

    if not train_pkl.exists() or not val_pkl.exists():
        raise FileNotFoundError(
            f"Missing dataset files in {data_dir.resolve()}"
        )

    with open(train_pkl, "rb") as f:
        train_graphs = pickle.load(f)

    with open(val_pkl, "rb") as f:
        val_graphs = pickle.load(f)

    train_id2text = {g.id: g.description for g in train_graphs}
    val_id2text = {g.id: g.description for g in val_graphs}

    return train_graphs, val_graphs, train_id2text, val_id2text
