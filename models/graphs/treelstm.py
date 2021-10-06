"""https://github.com/aykutfirat/pyTorchTree"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class TreeLSTM(nn.Module):
    def __init__(self, vocabSize, hdim=100, numClasses=5):
        super(TreeLSTM, self).__init__()
        self.embedding = nn.Embedding(int(vocabSize), hdim)
        self.Wi = nn.Linear(hdim, hdim, bias=True)
        self.Wo = nn.Linear(hdim, hdim, bias=True)
        self.Wu = nn.Linear(hdim, hdim, bias=True)
        self.Ui = nn.Linear(2 * hdim, hdim, bias=True)
        self.Uo = nn.Linear(2 * hdim, hdim, bias=True)
        self.Uu = nn.Linear(2 * hdim, hdim, bias=True)
        self.Uf1 = nn.Linear(hdim, hdim, bias=True)
        self.Uf2 = nn.Linear(hdim, hdim, bias=True)
        self.projection = nn.Linear(hdim, numClasses, bias=True)
        self.activation = F.relu
        self.nodeProbList = []
        self.labelList = []

    def traverse(self, node):
        if node.isLeaf():
            e = self.embedding(Variable(torch.LongTensor([node.getLeafWord()])))
            i = F.sigmoid(self.Wi(e))
            o = F.sigmoid(self.Wo(e))
            u = self.activation(self.Wu(e))
            c = i * u
        else:
            leftH,leftC = self.traverse(node.left())
            rightH,rightC = self.traverse(node.right())
            e = torch.cat((leftH, rightH), 1)
            i = F.sigmoid(self.Ui(e))
            o = F.sigmoid(self.Uo(e))
            u = self.activation(self.Uu(e))
            c = i * u + F.sigmoid(self.Uf1(leftH)) * leftC + F.sigmoid(self.Uf2(rightH)) * rightC
        h = o * self.activation(c)
        self.nodeProbList.append(self.projection(h))
        self.labelList.append(torch.LongTensor([node.label()]))
        return h,c

    def forward(self, x):
        self.nodeProbList = []
        self.labelList = []
        self.traverse(x)
        self.labelList = Variable(torch.cat(self.labelList))
        return torch.cat(self.nodeProbList)

    def getLoss(self, tree):
        nodes = self.forward(tree)
        predictions = nodes.max(dim=1)[1]
        loss = F.cross_entropy(input=nodes, target=self.labelList)
        return predictions,loss

    def evaluate(self, trees):
        n = nAll = correctRoot = correctAll = 0.0
        for j, tree in enumerate(trees):
            predictions, loss = self.getLoss(tree)
            correct = (predictions.data == self.labelList.data)
            correctAll += correct.sum()
            nAll += correct.squeeze().size()[0]
            correctRoot += correct.squeeze()[-1]
            n += 1
        return correctRoot / n, correctAll/nAll
