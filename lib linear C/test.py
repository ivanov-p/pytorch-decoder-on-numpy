import numpy as np

'''
from iv import Linear


lin = Linear(2, 3)

lin.weights[0][0] = 1
lin.weights[0][1] = 4
lin.weights[1][0] = 2
lin.weights[1][1] = 5
lin.weights[2][0] = 3
lin.weights[2][1] = 6

lin.bias[0] = 10
lin.bias[1] = 20
lin.bias[2] = 30

X = np.array([[2, 3]], dtype=np.float32)

print('X:')
print(X)

print('\nlin.weights:')
print(lin.weights)

print('\nlin.bias:')
print(lin.bias)

print('\nResult:')
print(lin(X))
'''


'''
from iv import LayerNorm

ln = LayerNorm((3,))
X = np.array([1., 2., 3.])

Y = ln(X)
print(Y)
'''




'''
from iv import functional
sdpa = functional.scaled_dot_product_attention

q = np.array([[[[ 0.2196],          [-0.1540]],         [[-0.5722],          [-0.1047]]]])
print('q:', q)

k = np.array([[[[-1.0221],          [-0.1118]],         [[ 0.2020],          [-0.0294]]]])
print('\nk:', k)

v = np.array([[[[-0.0921],          [ 0.0489]],         [[ 0.5773],          [ 0.2252]]]])
print('\nv:', v)

attn_mask = None
print('\nattn_mask:', attn_mask)

dropout_p = 0.1
print('\ndropout_p:', dropout_p)

is_causal = True
print('\nis_causal:', is_causal)

attn_output = sdpa(q, k, v, attn_mask, dropout_p, is_causal)
print('\nattn_output = scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)')

print('attn_output:', attn_output)
'''




'''
from iv import functional
sdpa = functional.scaled_dot_product_attention

q = np.array([[[[-0.5732],          [-0.5731]],         [[ 0.4115],          [ 0.4114]]]])
print('q:', q)

temp = np.random.randn(1, 2, 5, 1)
k = temp[:, :, 0:0, :]
print('\nk:', k)

temp2 = np.random.randn(1, 2, 1, 1)
v = temp2[:, :, 0:0, :]
print('\nv:', v)

attn_mask = None
print('\nattn_mask:', attn_mask)

dropout_p = 0.1
print('\ndropout_p:', dropout_p)

is_causal = False
print('\nis_causal:', is_causal)

attn_output = sdpa(q, k, v, attn_mask, dropout_p, is_causal)
print('\nattn_output = scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)')

print('attn_output:', attn_output)
'''


'''
from iv import functional
sdpa = functional.scaled_dot_product_attention

q = np.array([[[[ 0.7419],          [ 0.7419]],         [[-0.5963],          [-0.5963]]]])
print('q:', q)

k = np.array([[[[-1.2640],          [-1.2640]],         [[ 0.3857],          [ 0.3857]]]])
print('\nk:', k)

v = np.array([[[[-0.2696],          [-0.2696]],         [[ 0.2859],          [ 0.2859]]]])
print('\nv:', v)

attn_mask = None
print('\nattn_mask:', attn_mask)

dropout_p = 0.1
print('\ndropout_p:', dropout_p)

is_causal = True
print('\nis_causal:', is_causal)

attn_output = sdpa(q, k, v, attn_mask, dropout_p, is_causal)
print('\nattn_output = scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)')

print('attn_output:', attn_output)
'''




'''
from iv import functional
sdpa = functional.scaled_dot_product_attention

q = np.array([[[[-0.5732],          [-0.5732]],         [[ 0.4115],          [ 0.4115]]]])
print('q:', q)

temp = np.random.randn(1, 2, 1, 1)
k = temp[:, :, 0:0, :]
print('\nk:', k)


v = k.copy()
print('\nv:', v)

attn_mask = None
print('\nattn_mask:', attn_mask)

dropout_p = 0.1
print('\ndropout_p:', dropout_p)

is_causal = False
print('\nis_causal:', is_causal)

attn_output = sdpa(q, k, v, attn_mask, dropout_p, is_causal)
print('\nattn_output = scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)')

print('attn_output:', attn_output)
'''



from iv import functional
multi_head_attention_forward = functional.multi_head_attention_forward

out = multi_head_attention_forward(
        query = np.array([[[-1.0958,  0.1919]],

        [[-0.2543, -0.2678]]]),

        key = np.array([[[-1.0958,  0.1919]],

        [[-0.2543, -0.2678]]]),

        value = np.array([[[-1.0958,  0.1919]],

        [[-0.2543, -0.2678]]]),

        embed_dim_to_check = 2,

        num_heads = 2,
        
        in_proj_weight = np.array([[-0.0855,  0.6564],
        [ 0.5065, -0.0898],
        [ 0.8625, -0.4016],
        [-0.1415,  0.2442],
        [ 0.0446, -0.2250],
        [-0.5780, -0.2921]]), # !!!!!!!!!!!!!! NOT SET None

        in_proj_bias = np.array([0., 0., 0., 0., 0., 0.]),

        bias_k = None,

        bias_v = None,

        add_zero_attn = False,

        dropout_p = 0.1,
        
        out_proj_weight = np.array([[-0.5371, -0.1290],
        [-0.2588,  0.4265]]),

        out_proj_bias = np.array([0., 0.]),

        training = True,

        key_padding_mask = None,

        need_weights = False, # !!!!!!!!!!!!!!!!!!  SET False

        attn_mask = np.array([[0., -np.inf],
        [0., 0.]]),

        use_separate_proj_weight = False, # !!!!! SET False

        q_proj_weight = None,

        k_proj_weight = None,

        v_proj_weight = None,

        static_k = None, # !!!!!!!!!!!!!! SET None

        static_v = None, # !!!!!!!!!!!!!! SET None

        average_attn_weights = True,

        is_causal = True,
)

print(out)