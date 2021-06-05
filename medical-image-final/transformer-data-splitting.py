#!/usr/bin/env python
# coding: utf-8

# In[1]:


import collections
import logging
import os
import pathlib
import re
import string
import sys
import time

import numpy as np
import matplotlib.pyplot as plt


import tensorflow as tf


# ## Download the Dataset

# In[2]:


import numpy as np
import cv2
import xml.etree.ElementTree as ET
import argparse
import os
import re
import nltk
nltk.download('punkt')


# In[3]:


file_list = os.listdir('./IUXR_report')
file_list.sort()


# In[4]:


image_path = []
caption_list = []
for file in file_list:
    caption_string = ''
    tree = ET.parse('./IUXR_report/' + file)
    root = tree.getroot()
    if root.find('parentImage') == None:
        continue;
    else:
        image_path.append('IUXR_png/'+root.find('parentImage').attrib['id']+'.png')
    for data in root.iter('AbstractText'):
        label = data.attrib['Label']
        if label == 'FINDINGS' and data.text != None:
            caption_string+=data.text
        if label == 'IMPRESSION' and data.text != None:
            caption_string+=data.text
    caption_list.append(caption_string)


# In[5]:


new_report_list = []
new_image_name_list = []

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

for idx, report in enumerate(caption_list):
    new_report = report.lower().replace("..", ".")
    new_report = new_report.replace("'", "")
    new_sentences = []
    # print(nltk.tokenize.sent_tokenize(new_report))
    for sentence in nltk.tokenize.sent_tokenize(new_report):
        new_sentence = sentence.replace("/", " / ")
        if "xxxx" not in sentence and not hasNumbers(sentence):
            new_sentences.append(sentence)
    new_report = '<start> ' + " ".join(new_sentences) + ' <end>'
    if len(new_report) > 0:
        new_report_list.append(new_report)
        new_image_name_list.append(image_path[idx])
#         print(image_path[idx])


# In[6]:


top_k = 1000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+-/:;=?@[\]^_`{|}~')
tokenizer.fit_on_texts(new_report_list)

tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'
train_seqs = tokenizer.texts_to_sequences(new_report_list)

cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')


# In[7]:


BUFFER_SIZE = 500
BATCH_SIZE = 32

def mapping(image_path, train_captions):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (128, 128))
    img = tf.image.flip_up_down(img)
    img = tf.image.flip_left_right(img)
    img = img / 255.0 - 0.5
    train_captions = tf.dtypes.cast(train_captions, tf.float32)
    return img, train_captions

def mapping_origin(image_path, train_captions):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (128, 128))
    img = img / 255.0 - 0.5
    train_captions = tf.dtypes.cast(train_captions, tf.float32)
    return img, train_captions

original_image_name_list = new_image_name_list
original_cap_vector = cap_vector

new_image_name_list = new_image_name_list[0:3500] 
cap_vector = cap_vector[0:3500] 

print(original_image_name_list[3501:3700])

# val_img_name_list = original_image_name_list[3500:3600]
# cap_val_vector = original_cap_vector[3500:3600]

dataset_train = tf.data.Dataset.from_tensor_slices((new_image_name_list, cap_vector)).map(mapping_origin)
dataset_train_flip_picture = tf.data.Dataset.from_tensor_slices((new_image_name_list, cap_vector)).map(mapping)

dataset_train = dataset_train.concatenate(dataset_train_flip_picture)
dataset_train = dataset_train.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset_train = dataset_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

dataset_val = tf.data.Dataset.from_tensor_slices((original_image_name_list[3501:3700], original_cap_vector[3501:3700])).map(mapping_origin)
dataset_val = dataset_val.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset_val = dataset_val.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
print(dataset_train)


# In[8]:


print(dataset_train)


# In[9]:


get_ipython().system('nvidia-smi')


# ## Positional encoding
# 

# In[13]:


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


# In[14]:


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


# In[15]:


n, d = 2048, 512
pos_encoding = positional_encoding(n, d)
print(pos_encoding.shape)
pos_encoding = pos_encoding[0]

# Juggle the dimensions for the plot
pos_encoding = tf.reshape(pos_encoding, (n, d//2, 2))
pos_encoding = tf.transpose(pos_encoding, (2, 1, 0))
pos_encoding = tf.reshape(pos_encoding, (d, n))

plt.pcolormesh(pos_encoding, cmap='RdBu')
plt.ylabel('Depth')
plt.xlabel('Position')
plt.colorbar()
plt.show()


# ## Masking

# In[16]:


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


# In[17]:


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


# ## Scaled dot product attention

# In[18]:


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


# ## Multi-head attention

# In[19]:


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        


        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)
        


        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)


        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)


        return output, attention_weights


# In[20]:


temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
out, attn = temp_mha(y, k=y, q=y, mask=None)
out.shape, attn.shape


# ## Point wise feed forward network

# In[21]:


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


# ## Encoder and decoder

# <img src="https://www.tensorflow.org/images/tutorials/transformer/transformer.png" width="600" alt="transformer">

# In[22]:


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
           look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        
        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


# In[23]:


class Residual(tf.keras.Model):
    """The Residual block of ResNet."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(num_channels, padding='same',
                                            kernel_size=3, strides=strides)
        self.conv2 = tf.keras.layers.Conv2D(num_channels, kernel_size=3,
                                            padding='same')
        self.conv3 = None
        if use_1x1conv:
            self.conv3 = tf.keras.layers.Conv2D(num_channels, kernel_size=1,
                                                strides=strides)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, X):
        Y = tf.keras.activations.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3 is not None:
            X = self.conv3(X)
        Y += X
        return tf.keras.activations.relu(Y)


# In[24]:


class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, num_residuals, first_block=False,
                 **kwargs):
        super(ResnetBlock, self).__init__(**kwargs)
        self.residual_layers = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                self.residual_layers.append(
                    Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                self.residual_layers.append(Residual(num_channels))

    def call(self, X):
        for layer in self.residual_layers.layers:
            X = layer(X)
        return X


# In[25]:


class Encoder(tf.keras.layers.Layer): # ResNet
    def __init__(self):
        super(Encoder, self).__init__()
        self.cov1 = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same')
        self.batch_norm =  tf.keras.layers.BatchNormalization()
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')
        self.res_1 = ResnetBlock(64, 2, first_block=True)
        self.res_2 = ResnetBlock(128, 2)
        self.res_3 = ResnetBlock(256, 2)
        self.res_4 = ResnetBlock(512, 2)
        self.glob_pool =  tf.keras.layers.GlobalAvgPool2D()
        self.fc = tf.keras.layers.Dense(1024)
        
    def call(self, x):
        x = self.cov1(x)
        x = self.batch_norm(x)
        x = tf.nn.relu(x)
        x = self.max_pool(x)
        x = self.res_1(x)
        x = self.res_2(x)
        x = self.res_4(x)
        x = self.glob_pool(x)
        x = self.fc(x)
        
        return x  # (batch_size, input_seq_len, d_model)


# In[26]:


temp_embedding = tf.random.uniform((64, 7), dtype=tf.float64, minval=0, maxval=28)
temp_img = tf.random.uniform((64, 128), dtype=tf.float64, minval=-1, maxval=1)


sample_encoder = Encoder()


temp_input = tf.random.uniform((64, 128, 128, 3), dtype=tf.float64, minval=-1, maxval=1)

sample_encoder_output = sample_encoder(temp_input)

sample_encoder_output.shape


# ### Decoder

# In[27]:


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
               maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
               look_ahead_mask, padding_mask):

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                 look_ahead_mask, padding_mask)
            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2

        return x, attention_weights


# In[28]:


sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8,
                         dff=2048, target_vocab_size=29,
                         maximum_position_encoding=141)

temp_input = tf.random.uniform((1, 7), dtype=tf.float64, minval=0, maxval=28)

sample_encoder_output = tf.random.uniform((1, 128), dtype=tf.float64, minval=0, maxval=28)

output, attn = sample_decoder(temp_input,
                              enc_output=sample_encoder_output,
                              training=False,
                              look_ahead_mask=None,
                              padding_mask=None)


# ## Create the Transformer

# Transformer consists of the encoder, decoder and a final linear layer. The output of the decoder is the input to the linear layer and its output is returned.

# In[29]:


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder()

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, x, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):

        x = self.encoder(x)
    
        dec_output, attention_weights = self.decoder(
            tar, x, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights


# In[30]:


sample_transformer = Transformer(
    num_layers=2, d_model=1024, num_heads=64, dff=2048,
    input_vocab_size=1000, target_vocab_size=1000,
    pe_input=10000, pe_target=6000)

temp_input = tf.random.uniform((64, 128, 128, 3), dtype=tf.float64, minval=-1, maxval=1)

temp_target = tf.random.uniform((64, 7), dtype=tf.int32, minval=0, maxval=28)

fn_out, _ = sample_transformer(temp_input, temp_target, training=False,enc_padding_mask=None,look_ahead_mask=None,dec_padding_mask=None)

fn_out.shape  # (batch_size, tar_seq_len, target_vocab_size)


# ## Set hyperparameters

# In[31]:


num_layers = 8
d_model = 512
dff = 512
num_heads = 8
dropout_rate = 0.3


# ## Optimizer

# Use the Adam optimizer with a custom learning rate scheduler according to the formula in the [paper](https://arxiv.org/abs/1706.03762).
# 
# $$\Large{lrate = d_{model}^{-0.5} * \min(step{\_}num^{-0.5}, step{\_}num \cdot warmup{\_}steps^{-1.5})}$$
# 

# In[32]:


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


# In[33]:


learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)


# ## Loss and metrics

# Since the target sequences are padded, it is important to apply a padding mask when calculating the loss.

# In[34]:


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


# In[35]:


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def accuracy_function(real, pred):
    not_paddings = (real != 0)
    logics = (real == tf.dtypes.cast(pred, tf.float32))
    logics = logics & not_paddings
    summation = tf.reduce_sum(tf.cast(logics, tf.float32))
    total = tf.reduce_sum(tf.cast(not_paddings, tf.float32))
    return summation/total


# In[36]:


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.Mean(name='val_accuracy')


# ## Training and checkpointing

# In[37]:


transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=1000,
    target_vocab_size=1000,
    pe_input=1000,
    pe_target=1000,
    rate=dropout_rate)


# In[38]:


def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


temp_embedding = tf.random.uniform((64, 142), dtype=tf.float64, minval=0, maxval=28)
temp_img = tf.random.uniform((64, 128), dtype=tf.float64, minval=-1, maxval=1)

enc_padding_mask, combined_mask, dec_padding_mask = create_masks(temp_img, temp_embedding)


# In[39]:


EPOCHS = 100


# In[40]:


def train_step(inp, tar):
    tar_inp = tar[:, :-1]
#     print(tar_inp)
    tar_real = tar[:, 1:]
#     print(tar_real)

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp,
                                     True,
                                     None,
                                     combined_mask,
                                     None)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    prediction = tf.argmax(predictions, axis=-1)

    train_loss(loss)
    train_accuracy(accuracy_function(tar_real, prediction))


# In[41]:


def val_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    predictions, _ = transformer(inp, tar_inp,
                                     True,
                                     None,
                                     combined_mask,
                                     None)
    loss = loss_function(tar_real, predictions)

    prediction = tf.argmax(predictions, axis=-1)

    val_loss(loss)
    val_accuracy(accuracy_function(tar_real, prediction))


# In[42]:


checkpoint_path = "./checkpoints/term_project_jinghao_transformer"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')


# In[43]:


get_ipython().system('nvidia-smi')


# In[60]:


early_stop = np.zeros(EPOCHS)
count = 1
maximum = 0.0
prev_maximum = 0.0
break_flag = 0
check_counter = 0
for epoch in range(EPOCHS):
    start = time.time()
    train_loss.reset_states()
    train_accuracy.reset_states()
    val_loss.reset_states()
    val_accuracy.reset_states()
    ckpt_save_path = ckpt_manager.save()
    print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

  # inp -> img, tar -> captioning
    for (batch, (inp, tar)) in enumerate(dataset_train):
        train_step(inp, tar)

        if batch % 50 == 0:
            print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
            
    for (batch, (inp, tar)) in enumerate(dataset_val):
        val_step(inp, tar)

        if batch % 50 == 0:
            print(f'Epoch {epoch + 1} Batch {batch} Loss {val_loss.result():.4f} Accuracy {val_accuracy.result():.4f}')

    print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f} VAL_Loss {val_loss.result():.4f} VAL_Accuracy {val_accuracy.result():.4f}')
    print(tf.keras.backend.get_value(val_accuracy.result()))
    early_stop[count] = tf.keras.backend.get_value(val_accuracy.result())
    ## if the val_accuracy can't improve in 10 epoochs, break the loop
    maximum = np.max(early_stop)
    
    if(maximum > prev_maximum):
        prev_maximum = maximum
        check_counter = 0
    else:
        check_counter+=1
    if(check_counter == 7):
        break_flag = 1
    print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')
    if(break_flag == 1):
        break


# In[44]:


def evaluate(img):
    output = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []
    
    result.append('<start>')

    for i in range(141):
        temp_img = tf.random.uniform((1, 128), dtype=tf.float64, minval=-1, maxval=1)

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(temp_img, output)
        
        predictions, attention_weights = transformer(img,
                                                 output,
                                                 False,
                                                 None,
                                                 combined_mask,
                                                 None)
        
        predictions = predictions[:, -1:, :] 
        predicted_id = tf.argmax(predictions.numpy(), axis=-1)
        result.append(tokenizer.index_word[predicted_id.numpy()[0][0]])
        to_append = tf.cast(predicted_id, tf.int32)
        # as its input.
        output = tf.concat([output, to_append], axis=-1)
        

        if tokenizer.index_word[predicted_id.numpy()[0][0]] == '<end>':
            return result


    return result


# In[45]:


def calc_max_length(tensor):
    return max(len(t) for t in tensor)




for x, y in dataset_train:
    result = []
    token = y.numpy()[2]
    img_ = x[2]
    img_ = tf.expand_dims(img_, axis=0)
    for j in token:
        if j != 0:
            result.append(tokenizer.index_word[j])
    predict = evaluate(img_)
    print('Predict Caption:', ' '.join(predict))
    plt.imshow(x[2].numpy()+ 0.5)
    break



print('Real Caption:', ' '.join(result))


# In[46]:


from nltk.translate.bleu_score import sentence_bleu

idx = 3771

for idx in range(3701,3740):
    image_path = original_image_name_list[idx]

    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (128, 128))
    img = img / 255.0 - 0.5

    img_ = tf.expand_dims(img, axis=0)
    predict = evaluate(img_)
#     print(predict)
    predict_bleu = predict.remove('<start>')
    predict_bleu = predict.remove('<end>')
    print('Predict Caption:', ' '.join(predict))
    predict = ' '.join(predict)

    result = []
    for j in original_cap_vector[idx]:
            if j != 0:
                result.append(tokenizer.index_word[j])
#     print(result)
    result_bleu = result.remove('<start>')
    result_bleu = result.remove('<end>')
    print('   Real Caption:', ' '.join(result))
    score = sentence_bleu(predict, result, weights=(1, 0, 0, 0))
    print(score)
    print('--------------------------------------------------------------------------------------------------------')
    tar_real = ' '.join(result)
    plt.imshow(img + 0.5)


# In[49]:


idx = 3771
BLEU_1 = 0
BLEU_2 = 0
BLEU_3 = 0
BLEU_4 = 0
for idx in range(3701,3825):
    image_path = original_image_name_list[idx]

    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = img / 255.0 - 0.5

    img_ = tf.expand_dims(img, axis=0)
    predict = evaluate(img_)
    print('Predict Caption:', ' '.join(predict))

    result = []
    for j in original_cap_vector[idx]:
            if j != 0:
                result.append(tokenizer.index_word[j])
    print('Real    Caption:', ' '.join(result))
    print('<------------------------------------------------>')
    BLEUscore_1 = nltk.translate.bleu_score.sentence_bleu([result], predict, weights=(1, 0, 0, 0))
    BLEUscore_2 = nltk.translate.bleu_score.sentence_bleu([result], predict, weights=(0, 1, 0, 0))
    BLEUscore_3 = nltk.translate.bleu_score.sentence_bleu([result], predict, weights=(0, 0, 1, 0))
    BLEUscore_4 = nltk.translate.bleu_score.sentence_bleu([result], predict, weights=(0, 0, 0, 1))
    print('1:', BLEUscore_1)
    print('2:', BLEUscore_2)
    print('3:', BLEUscore_3)
    print('4:', BLEUscore_4)
    BLEU_1 += BLEUscore_1
    BLEU_2 += BLEUscore_2
    BLEU_3 += BLEUscore_3
    BLEU_4 += BLEUscore_4
#     result = ' '.join(result)
#     predict = ' '.join(predict)
#     print(result)
#     print(predict)
#     print(accuracy_function(result, predict))
print('average 1-gram', BLEU_1/125)
print('average 2-gram', BLEU_2/125)
print('average 3-gram', BLEU_3/125)
print('average 4-gram', BLEU_4/125)


# In[ ]:




