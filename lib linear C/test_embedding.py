import torch
import numpy as np
from Embedding import Embedding

def test_embedding():
    num_embeddings = 100
    embedding_dim = 32
    batch_size = 2
    seq_len = 5

    # PyTorch
    pt_emb = torch.nn.Embedding(num_embeddings, embedding_dim)
    pt_emb.eval()

    # NumPy
    np_emb = Embedding(num_embeddings, embedding_dim)

    # Копируем веса
    np_emb.weight = pt_emb.weight.detach().numpy()

    # Одинаковые входные данные
    np.random.seed(42)
    input_np = np.random.randint(0, num_embeddings, (batch_size, seq_len))
    input_pt = torch.tensor(input_np)

    # Forward
    with torch.no_grad():
        out_pt = pt_emb(input_pt)
    out_np = np_emb(input_np)

    diff = np.max(np.abs(out_pt.numpy() - out_np))
    print(f"Embedding Max Diff: {diff:.2e}")
    print(f"Input shape:  {input_np.shape}")
    print(f"Output shape: {out_np.shape}")
    
    if diff < 1e-6:
        print("✅ EMBEDDING OK")
    else:
        print("❌ EMBEDDING IS BROKEN")

if __name__ == "__main__":
    test_embedding()