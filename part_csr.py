import torch as th

# 0 1 1 1 1 0 0
# 0 0 1 1 1 1 0
# 1 1 0 0 0 0 0

# 0, 4, 8, 10
# 1, 2, 3, 4, 2, 3, 4, 5, 0, 1
# 0, 1, 2, 3, 4, 5, 6
#


def partition_csr(indptr, indices, eids, chunk_size=32):
    indptr = indptr.cpu()
    row = []
    indptr_ = [0]
    for i in range(indptr):
        for j in range(indptr[i], indptr[i + 1] + chunk_size - 1, chunk_size):
            row.append(i)
            indptr_.append(min(j, indptr[i + 1])) 

    row = th.tensor(row)
    indptr_ = th.tensor(indptr_)
    return row, indptr_, indices, eids

if __name__ == '__main__':
    indptr = th.tensor([0, 4, 8, 10])
    indices = th.tensor([1, 2, 3, 4, 2, 3, 4, 5, 0, 1])
    eids = th.tensor([0, 1, 2, 3, 4, 5, 6])
    row, indptr_ = partition_csr(indptr, indices, eids, chunk_size=2) 
    print(row, indptr_)
