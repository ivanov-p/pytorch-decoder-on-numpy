import torch

import torch.nn as nn

import math

  
 
  

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5): # max_len = 5000

        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

  

    def forward(self, x):

        x = x + self.pe[:, :x.size(1)]

        return x

  
  

class BuiltInDecoderLM(nn.Module):

    def __init__(self, vocab_size, d_model=2, n_layers=2, n_heads=2,

                 d_ff=2, dropout=0.1, max_len=5): # self, vocab_size, d_model=512, n_layers=2, n_heads=8, d_ff=2048, dropout=0.1, max_len=512

        super().__init__()

        self.d_model = d_model

  

        # Эмбеддинги

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        self.pos_encoding = PositionalEncoding(d_model, max_len)

  

        # Встроенный слой декодера

        decoder_layer = nn.TransformerDecoderLayer(

            d_model=d_model,

            nhead=n_heads,

            dim_feedforward=d_ff,

            dropout=dropout,

            batch_first=True  # Удобно: (batch, seq_len, dim)

        )

  

        # Стек декодеров

        self.decoder = nn.TransformerDecoder(

            decoder_layer,

            num_layers=n_layers

        )

  

        # Выходной слой

        self.output_projection = nn.Linear(d_model, vocab_size)

  

    def generate_mask(self, sz):

        """Создаёт маску для маскированного самовнимания"""

        mask = torch.triu(torch.ones(sz, sz), diagonal=1)

        mask = mask.masked_fill(mask == 1, float('-inf'))

        return mask

  

    def forward(self, src):

        batch_size, seq_len = src.size()

  

        # Токенные эмбеддинги + масштабирование

        x = self.token_embedding(src) * math.sqrt(self.d_model)

  

        # Позиционное кодирование

        x = self.pos_encoding(x)

  

        # Маска для маскированного внимания

        tgt_mask = self.generate_mask(seq_len).to(src.device)

  

        # Проход через декодер (memory = None для авторегрессионной модели)

        output = self.decoder(

            tgt=x,

            memory=torch.zeros(batch_size, 0, self.d_model).to(x.device),  # Пустой memory

            tgt_mask=tgt_mask

        )

  

        # Проекция на пространство словаря

        logits = self.output_projection(output)

        return logits

  
  
  

# Инициализация модели

model = BuiltInDecoderLM(vocab_size=3, d_model=2, n_layers=2)
# BuiltInDecoderLM(vocab_size=10000, d_model=256, n_layers=4)

  

# Пример данных

input_ids = torch.randint(0, 3, (1, 2))  # batch=1, seq_len=2
# torch.randint(0, 10000, (1, 10))  # batch=1, seq_len=10
  

# Прямой проход

logits = model(input_ids)

print(f"Выходная форма: {logits.shape}")  # [1, 2, 3]
# [1, 10, 10000]