import dgl
import torch
import torch.nn.functional as F
import torch.nn as nn


def msg_func(edges):
    return {'k': edges.src['k'], 'v': edges.src['v']}


def reduce_func(nodes):
    # nodes.data['q'] has the shape
    #     (number_of_nodes, feature_dims)
    # nodes.data['k'] and nodes.data['v'] have the shape
    #     (number_of_nodes, number_of_incoming_messages, feature_dims)
    # You only need to deal with the case where all nodes have the same number
    # of incoming messages.
    q = nodes.data['q'][:, None]
    k = nodes.mailbox['k']
    v = nodes.mailbox['v']
    s = F.softmax((q * k).sum(-1), 1)[:, :, None]
    return {'v': torch.sum(s * v, 1)}


class GATLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GATLayer, self).__init__()
        self.Q = nn.Linear(in_feats, out_feats)
        self.K = nn.Linear(in_feats, out_feats)
        self.V = nn.Linear(in_feats, out_feats)

    def apply(self, nodes):
        return {'v': F.relu(self.linear(nodes.data['v']))}

    def forward(self, g, feature):
        g.ndata['v'] = self.V(feature)
        g.ndata['q'] = self.Q(feature)
        g.ndata['k'] = self.K(feature)
        g.update_all(msg_func, reduce_func)
        g.apply_nodes(func=self.apply)
        return g.ndata['v']


#####################################################################
# Multi-head Attention
# ^^^^^^^^^^^^^^^^^^^^
#
# Analogous to multiple channels in ConvNet, GAT introduces **multi-head
# attention** to enrich the model capacity and to stabilize the learning
# process. Each attention head has its own parameters and their outputs can be
# merged in two ways:
#
# .. math:: \text{concatenation}: h^{(l+1)}_{i} =||_{k=1}^{K}\sigma\left(\sum_{j\in \mathcal{N}(i)}\alpha_{ij}^{k}W^{k}h^{(l)}_{j}\right)
#
# or
#
# .. math:: \text{average}: h_{i}^{(l+1)}=\sigma\left(\frac{1}{K}\sum_{k=1}^{K}\sum_{j\in\mathcal{N}(i)}\alpha_{ij}^{k}W^{k}h^{(l)}_{j}\right)
#
# where :math:`K` is the number of heads. The authors suggest using
# concatenation for intermediary layers and average for the final layer.
#
# We can use the above defined single-head ``GATLayer`` as the building block
# for the ``MultiHeadGATLayer`` below:


class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))

###########################################################################
# Put everything together
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# Now, we can define a two-layer GAT model:

class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1)

    def forward(self, h):
        h = self.layer1(h)
        h = F.elu(h)
        h = self.layer2(h)
        return h


if __name__ == "__main__":

    g = dgl.DGLGraph()
    g.add_nodes(5)                          # add 5 nodes
    g.add_edges([0, 0, 0, 0], [1, 2, 3, 4]) # add 4 edges 0->1, 0->2, 0->3, 0->4
    g.ndata['h'] = torch.randn(5, 3)           # assign one 3D vector to each node
    g.edata['h'] = torch.randn(4, 4)           # assign one 4D vector to each edge