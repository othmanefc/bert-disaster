import io
import random
import os

import pandas as pd
import numpy as np

import mxnet as mx
import gluonnlp as nlp
from bert import data, model

DATA_PATH = "data"

train = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
test = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))

np.random.seed(100)
random.seed(100)
mx.random.seed(10000)
# change `ctx` to `mx.cpu()` if no GPU is available.
ctx = mx.cpu()

bert_base, vocabulary = nlp.model.get_model(
    'bert_12_768_12',
    dataset_name='book_corpus_wiki_en_uncased',
    pretrained=True,
    ctx=ctx,
    use_pooler=True,
    use_decoder=False,
    use_classifier=False)
print(bert_base)

bert_classifier = nlp.model.BERTClassifier(bert_base,
                                           num_classes=2,
                                           dropout=0.1)
# only need to initialize the classifier layer.
print("Initializing classifier...")
bert_classifier.classifier.initialize(init=mx.init.Normal(0.02), ctx=ctx)
bert_classifier.hybridize(static_alloc=True)

# softmax cross entropy loss for classification
loss_function = mx.gluon.loss.SoftmaxCELoss()
loss_function.hybridize(static_alloc=True)

metric = mx.metric.Accuracy()

## imputation
print("Imputing...")
train = train.fillna("missing")
test = test.fillna("missing")

bert_tokenizer = nlp.data.BERTTokenizer(vocabulary, lower=True)

# The maximum length of an input sequence
max_len = 256

# The labels for the two classes [(0 = not similar) or  (1 = similar)]
all_labels = [0, 1]

# whether to transform the data as sentence pairs.
# for single sentence classification, set pair=False
# for regression task, set class_labels=None
# for inference without label available, set has_label=False
pair = True
print("Transformin...")
transform = data.transform.BERTDatasetTransform(bert_tokenizer,
                                                max_len,
                                                class_labels=all_labels,
                                                has_label=True,
                                                pad=True,
                                                pair=False)

train_df = train.drop(columns="id")
train_df["full_text"] = train_df.apply(
    lambda x: f"{x['text']} at {x['location']} for {x['keyword']}", axis=1)
data_list = []
for _, line in train_df.iterrows():
    line_trans = transform(line[["full_text", "target"]])
    data_list.append(line_trans)


##Fine Tuning

# The hyperparameters
print("starting tunning...")
batch_size = 32
lr = 5e-6

# The FixedBucketSampler and the DataLoader for making the mini-batches
train_sampler = nlp.data.FixedBucketSampler(lengths=[int(item[1]) for item in data_list],
                                            batch_size=batch_size,
                                            shuffle=True)
bert_dataloader = mx.gluon.data.DataLoader(data_list, batch_sampler=train_sampler)

trainer = mx.gluon.Trainer(bert_classifier.collect_params(), 'adam',
                           {'learning_rate': lr, 'epsilon': 1e-9})

# Collect all differentiable parameters
# `grad_req == 'null'` indicates no gradients are calculated (e.g. constant parameters)
# The gradients for these params are clipped later
params = [p for p in bert_classifier.collect_params().values() if p.grad_req != 'null']
grad_clip = 1

# Training the model with only three epochs
log_interval = 4
num_epochs = 3
for epoch_id in range(num_epochs):
    metric.reset()
    step_loss = 0
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(bert_dataloader):
        with mx.autograd.record():
            # Load the data to the GPU
            token_ids = token_ids.as_in_context(ctx)
            valid_length = valid_length.as_in_context(ctx)
            segment_ids = segment_ids.as_in_context(ctx)
            label = label.as_in_context(ctx)
            # Forward computation
            out = bert_classifier(token_ids, segment_ids, valid_length.astype('float32'))
            ls = loss_function(out, label).mean()

        # And backwards computation
        ls.backward()

        # Gradient clipping
        trainer.allreduce_grads()
        nlp.utils.clip_grad_global_norm(params, 1)
        trainer.update(1)
        print("gradient clipped")
        step_loss += ls.asscalar()
        metric.update([label], [out])

        # Printing vital information
        print('[Epoch {} Batch {}/{}] loss={:.4f}, lr={:.7f}, acc={:.3f}'
                        .format(epoch_id, batch_id + 1, len(bert_dataloader),
                                step_loss / log_interval,
                                trainer.learning_rate, metric.get()[1]))
        step_loss = 0

