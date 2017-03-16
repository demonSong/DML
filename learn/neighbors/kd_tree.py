import numpy as np
import sys


class KDTree(object):
    def __init__(self, data, leafsize=10):
        # 初始化数据基本属性
        self.data = np.asarray(data)
        self.n, self.m = np.shape(self.data)
        self.leafsize = int(leafsize)

        if self.leafsize < 1:
            raise ValueError("leafsize must be at least 1")

        self.maxes = np.amax(self.data, axis=0)
        self.mins = np.amax(self.data, axis=0)

        # 构建每个样例数据的下标
        self.tree = self.__build(np.arange(self.n),self.maxes,self.mins)

    # 在判断语句内 重写函数 是什么意思？ id是什么？
    class node(object):
        if sys.version_info[0] >= 3:
            def __lt__(self, other):
                return id(self) < id(other)

            def __gt__(self, other):
                return id(self) > id(other)

            def __le__(self, other):
                return id(self) <= id(other)

            def __ge__(self, other):
                return id(self) >= id(other)

            def __eq__(self, other):
                return id(self) == id(other)

    class leafnode(node):
        def __init__(self,idx):
            self.idx = idx
            self.children = len(idx)


    # idx 属于一个集合对象啊
    def __build(self,idx,maxes,mins):
        if len(idx) <= self.leafsize:
            return KDTree.leafnode(idx)
        else:

            data = self.data[idx]

            # 求的是某个维度的最大值下标
            d = np.argmax(maxes-mins)
            maxval = maxes[d]
            minval = mins[d]

            if maxval == minval:
                return KDTree.leafnode(idx)


    def query(self,x,k=1,eps=0,p=2,distance_upper_boud=np.inf):
        print("进行查询操作")






