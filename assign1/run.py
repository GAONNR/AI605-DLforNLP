import random
import time

import torch

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

from dataset import SkipgramDataset
from word2vec import NegSamplingWord2Vec, NaiveWord2Vec

### Select device; if you coded your script with many 'for' iteration, cpu would be faster.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu') # Uncomment if you want to test with CPU

### Select loss
loss_select = 'naive'
# loss_select = 'neg'

### Reduce batch_size if you suffer performance issue
batch_size = 256

dim_vector = 50
context_size = 5

random.seed(1234)
torch.manual_seed(1234)

start_time = time.time()
dataset = SkipgramDataset(window_size=context_size, device=device, min_freq=2)

print ('The number of tokens: %d' % dataset.n_tokens)

if loss_select == 'naive':
    model = NaiveWord2Vec(n_tokens=dataset.n_tokens, word_dimension=dim_vector).to(device)
else:
    model = NegSamplingWord2Vec(n_tokens=dataset.n_tokens, word_dimension=dim_vector, negative_sampler=dataset.negative_sampler).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=.5)
data_iterator = iter(torch.utils.data.dataloader.DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=False))

exp_loss = None
loss_log = tqdm(total=0, bar_format='{desc}', position=1)
for step in trange(20000, desc="Step", position=0):
    center_word_index, outside_word_indices = next(data_iterator)
    center_word_index, outside_word_indices = center_word_index.to(device), outside_word_indices.to(device)

    optimizer.zero_grad()
    loss = model(center_word_index, outside_word_indices).mean()
    loss.backward()
    optimizer.step()
    scheduler.step()

    exp_loss = .95 * exp_loss + .05 * loss.cpu() if exp_loss is not None else loss.cpu()
    des = 'Loss: {:06.4f}'.format(loss.cpu())
    loss_log.set_description_str(des)


print("training took %d seconds" % (time.time() - start_time))
print("Training takes less then 3 minutes with RTX TITAN GPU and less then 1 hour with Xeon Silver CPU in case of naive softmax loss")

# concatenate the input and output word vectors
word_vectors = torch.cat([model.center_vectors, model.outside_vectors], dim=1)

visualize_words = [
    "man", "woman", "large", "little", "young", "old",
    "boy", "girl", "black", "white", "dog", "cat",
    "a", "is", "small", "big", "blue", "red",
    "baseball", "basketball", "hot", "cold", "snow", "rain"]

visualize_idx = list(map(dataset.word2idx, visualize_words))
visualize_vecs = torch.index_select(word_vectors, dim=0, index=word_vectors.new_tensor(visualize_idx, dtype=torch.long))
temp = (visualize_vecs - visualize_vecs.mean(dim=0))
covariance = 1.0 / len(visualize_idx) * temp.T @ temp
U, S, V = covariance.svd()
coord = temp @ U[:, 0:2]

coord = coord.cpu().detach()
for i in range(len(visualize_words)):
    plt.text(coord[i, 0], coord[i, 1], visualize_words[i],
            bbox=dict(facecolor='green', alpha=.1))

plt.xlim((coord[:,0].min(), coord[:,0].max()))
plt.ylim((coord[:,1].min(), coord[:,1].max()))

plt.savefig('word_vectors_' + loss_select +'.png')

torch.save(model.state_dict(), "word2vec_" + loss_select + ".pth")