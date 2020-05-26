import os.path as osp

import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid
from node2vec_impl_dp import Node2Vec
import time
try:
    import torch_cluster  # noqa
    random_walk = torch.ops.torch_cluster.random_walk
except ImportError:
    random_walk = None

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset)
data = dataset[0]

class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        return getattr(self.module, name)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
torch.cuda.set_device(0)
model = Node2Vec(data.edge_index, embedding_dim=128, walk_length=20,
                 context_size=10, walks_per_node=10, num_negative_samples=1,
                 sparse=True)

optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)
model = torch.nn.DataParallel(model)
model.to(device)

loader = model.module.loader(batch_size=128, shuffle=True, num_workers=0)
num_gpus = 2
def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        # concatenate along last dimension
        batch = torch.cat((pos_rw, neg_rw), -1)
        # for calling data parallel, call model.forward
        # for calling forward without dataparallel, call model.module
        loss = model(batch.to(device))
        # because now loss will consist of two elements, one for each GPU
        loss = loss.sum()/num_gpus
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def test():
    model.eval()
    z = model.module() # use non data-parallel version of forward
    acc = model.module.test(z[data.train_mask], data.y[data.train_mask],
                     z[data.test_mask], data.y[data.test_mask], max_iter=150)
    return acc

start = time.time()
losses = []
accs = []
for epoch in range(1, 50):
    loss = train()
    acc = test()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}')
    losses.append(loss)
    accs.append(acc)
print('time to run 50 epochs: {0}'.format(time.time() - start))

plt.pyplot.plot(accs)
plt.pyplot.plot(losses)
plt.pyplot.legend(['DataParallel: Accuracy against epoch count', 'DataParallel: Loss progression against epoch count'])

@torch.no_grad()
def plot_points(colors):
    model.eval()
    z = model(torch.arange(data.num_nodes, device=device))
    z = TSNE(n_components=2).fit_transform(z.cpu().numpy())
    y = data.y.cpu().numpy()

    plt.figure(figsize=(8, 8))
    for i in range(dataset.num_classes):
        plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=colors[i])
    plt.axis('off')
    plt.show()


colors = [
    '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535', '#ffd700'
]
plot_points(colors)

