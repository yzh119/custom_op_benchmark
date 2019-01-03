import torch as th
from graphop import *
from torch.autograd import Function

class MaskedMM(Function):
    @staticmethod
    def forward(ctx, adj, A, B):
        row, col = adj._indices()
        ctx.save_for_backward(row, col, A, B)
        return maskedmm_forward(row.int(), col.int(), A, B)

    @staticmethod
    def backward(ctx, grad_o):
        row, col, A, B = ctx.saved_tensors
        dA, dB = maskedmm_backward(row.int(), col.int(), A, B, grad_o)
        return None, dA, dB

class MaskedMMSimple(Function):
    @staticmethod
    def forward(ctx, inc_x, inc_y, A, B):
        with th.no_grad():
            A_e = th.sparse.mm(inc_x.float(), A) # shape: (e, d)
            B_e = th.sparse.mm(inc_y.float(), B) # shape: (e, d)
            ctx.save_for_backward(A_e, B_e, inc_x, inc_y)
            o = (A_e * B_e).sum(-1) # shape: (e)
        assert o.requires_grad==False
        return o

    @staticmethod
    def backward(ctx, grad_o): # shape: (e)
        A_e, B_e, inc_x, inc_y = ctx.saved_tensors
        dAe = grad_o.unsqueeze(-1) * B_e
        dBe = grad_o.unsqueeze(-1) * A_e
        dA = th.sparse.mm(inc_x.t().float(), dAe)
        dB = th.sparse.mm(inc_y.t().float(), dBe)
        return None, None, dA, dB

if __name__ == '__main__':
    import os
    batch_size = 20 
    l = 25
    n = batch_size * l
    e = batch_size * (l ** 2)
    v = th.ones(e, dtype=th.uint8)
    if not os.path.exists('i.pt'):
        i = th.zeros(2, e, dtype=th.long)
        cnt = 0
        for b in range(batch_size):
            for x in range(b * l, (b + 1) * l):
                for y in range(b * l, (b + 1) * l):
                    i[0, cnt] = x
                    i[1, cnt] = y
                    cnt += 1
        th.save(i, 'i.pt')
    else:
        i = th.load('i.pt')

    adj = th.sparse.ByteTensor(i, v, th.Size([n, n]))

    v = th.ones(e, dtype=th.uint8)
    if not os.path.exists('ix.pt'):
        i_x = th.zeros(2, e, dtype=th.long)
        i_y = th.zeros(2, e, dtype=th.long)
        cnt = 0
        for b in range(batch_size):
            for x in range(b * l, (b + 1) * l):
                for y in range(b * l, (b + 1) * l):
                    i_x[0, cnt] = cnt 
                    i_x[1, cnt] = x
                    i_y[0, cnt] = cnt 
                    i_y[1, cnt] = y
                    cnt += 1
        th.save(i_x, 'ix.pt')
        th.save(i_y, 'iy.pt')
    else:
        i_x = th.load('ix.pt')
        i_y = th.load('iy.pt')

    inc_x = th.sparse.ByteTensor(i_x, v, th.Size([e, n]))
    inc_y = th.sparse.ByteTensor(i_y, v, th.Size([e, n])) 

    import time

    print('simple implementation')
    dim = 1024 
    A = th.rand(n, dim, requires_grad=True)
    B = th.rand(n, dim, requires_grad=True)
    grad = th.rand(e)
    tic = time.time()
    A_e = th.sparse.mm(inc_x.float(), A)
    B_e = th.sparse.mm(inc_y.float(), B)
    O = (A_e * B_e).sum(-1)
    O_ori = O
    print('forward elapse time: {}'.format(time.time() - tic))
    tic = time.time()
    O.backward(grad)
    print('backward elapse time: {}'.format(time.time() - tic))
    A_grad_ori, B_grad_ori = A.grad, B.grad
    A.grad.zero_()
    B.grad.zero_()

    print('simple implementation, hand-crafted autograd')
    tic = time.time()
    O = MaskedMMSimple.apply(inc_x, inc_y, A, B)
    print('forward elapse time: {}'.format(time.time() - tic))
    assert th.allclose(O, O_ori)
    tic = time.time()
    O.backward(grad)
    print('backward elapse time: {}'.format(time.time() - tic))
    assert th.allclose(A.grad, A_grad_ori) and th.allclose(B.grad, B_grad_ori)
    A.grad.zero_()
    B.grad.zero_()
   
    print('custom kernel')
    tic = time.time()
    O = MaskedMM.apply(adj.cuda(), A.cuda(), B.cuda())
    print('forward elapse time: {}'.format(time.time() - tic))
    assert th.allclose(O.cpu(), O_ori)
    tic = time.time()
    O.backward(grad.cuda())
    print('backward elapse time: {}'.format(time.time() - tic))
    assert th.allclose(A.grad.cpu(), A_grad_ori)
