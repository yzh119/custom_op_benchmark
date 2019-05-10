import torch as th

# 0 1 1 1 1 0 0
# 0 0 1 1 1 1 0
# 1 1 0 0 0 0 0

# 0, 4, 8, 10
# 1, 2, 3, 4, 2, 3, 4, 5, 0, 1
# 0, 1, 2, 3, 4, 5, 6
#


def partition_csr(indptr, indices, eids, chunk_size=32):
    device = indptr.device
    indptr = indptr.cpu()
    row = []
    indptr_ = []
    for i in range(len(indptr) - 1):
        for j in range(indptr[i], indptr[i + 1], chunk_size):
            row.append(i)
            indptr_.append(j)

    indptr_.append(indptr[-1])

    row = th.tensor(row, device=device)
    indptr_ = th.tensor(indptr_, device=device)
    return row, indptr_, indices, eids

if __name__ == '__main__':
    indptr = th.tensor([0, 4, 8, 10]).cuda()
    indices = th.tensor([1, 2, 3, 4, 2, 3, 4, 5, 0, 1]).cuda()
    eids = th.tensor([0, 1, 2, 3, 4, 5, 6]).cuda()
    row, indptr_, _, _ = partition_csr(indptr, indices, eids, chunk_size=2) 
    print(row, indptr_)
