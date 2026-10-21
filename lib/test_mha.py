import torch
import numpy as np
from MultiheadAttention import MultiheadAttention

def test_mha():
    d_model = 64
    nhead = 4
    seq_len = 5
    batch_size = 2
    
    # PyTorch Layer
    pt_mha = torch.nn.MultiheadAttention(d_model, nhead, dropout=0.0, batch_first=False)
    pt_mha.eval()
    
    # Our Layer
    np_mha = MultiheadAttention(d_model, nhead, dropout=0.0, batch_first=False)
    np_mha.eval()
    
    # CRITICAL: Check weight shapes and copying logic!
    # In PyTorch MHA, in_proj_weight is a single packed matrix (3*d_model, d_model)
    # Make sure your copy function handles this correctly.
    
    # For this test, let's just verify shapes first
    print("PT in_proj_weight shape:", pt_mha.in_proj_weight.shape)
    print("NP in_proj_weight shape:", np_mha.in_proj_weight.shape)
    
    # If shapes match, try to copy
    # Note: You might need to adjust how you copy these specific weights 
    # depending on whether your internal _in_projection_packed expects them split or packed.
    
    # Let's assume standard packing for now
    np_mha.in_proj_weight = pt_mha.in_proj_weight.detach().numpy()
    np_mha.in_proj_bias = pt_mha.in_proj_bias.detach().numpy()
    np_mha.out_proj_weight = pt_mha.out_proj.weight.detach().numpy()
    np_mha.out_proj_bias = pt_mha.out_proj.bias.detach().numpy()
    
    # Input
    tgt_np = np.random.randn(seq_len, batch_size, d_model).astype(np.float32)
    tgt_pt = torch.tensor(tgt_np)
    
    # Forward
    with torch.no_grad():
        out_pt, _ = pt_mha(tgt_pt, tgt_pt, tgt_pt, need_weights=False)
    
    out_np, _ = np_mha(tgt_np, tgt_np, tgt_np, need_weights=False)
    
    diff = np.max(np.abs(out_pt.numpy() - out_np))
    print(f"MHA Max Diff: {diff:.2e}")
    if diff > 1e-4: # Attention can have slightly higher floating point noise due to softmax/exp
        print("MHA IS BROKEN")
        print("PyTorch:", out_pt.numpy()[0][0][:5])
        print("NumPy:  ", out_np[0][0][:5])
    else:
        print("MHA OK")

if __name__ == "__main__":
    test_mha()