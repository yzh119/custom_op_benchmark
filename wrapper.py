import torch as th
from graphop import *
from torch.autograd import Function

class SparseSoftmax(Function):
    @staticmethod
    def forward(ctx, ptr, eid, x):
        y = sparse_softmax_forward(ptr, eid, x);
        ctx.save_for_backward(ptr, eid, y)
        return y

    @staticmethod
    def backward(ctx, dy):
        ptr, eid, y = ctx.saved_tensors
        return None, None, sparse_softmax_backward(ptr, eid, y, dy);

class MaskedMM(Function):
    @staticmethod
    def forward(ctx, adj, A, B):
        row, col = adj._indices()
        ctx.save_for_backward(row, col, A, B)
        return maskedmm_forward(row, col, A, B)

    @staticmethod
    def backward(ctx, grad):
        row, col, A, B = ctx.saved_tensors
        dA, dB = maskedmm_backward(row, col, A, B, grad)
        return None, dA, dB

class MaskedMMCSR(Function):
    @staticmethod
    def forward(ctx, ptr_r, eid_r, nid_r, ptr_c, eid_c, nid_c, A, B):
        ctx.save_for_backward(ptr_r, eid_r, nid_r, ptr_c, eid_c, nid_c, A, B)
        return maskedmm_csr_forward(ptr_r, eid_r, nid_r, A, B)

    @staticmethod
    def backward(ctx, grad):
        ptr_r, eid_r, nid_r, ptr_c, eid_c, nid_c, A, B = ctx.saved_tensors
        dA, dB = maskedmm_csr_backward(ptr_r, eid_r, nid_r, ptr_c, eid_c, nid_c, A, B, grad)
        return None, None, None, None, None, None, dA, dB

class MaskedMMSimple(Function):
    @staticmethod
    def forward(ctx, inc_x, inc_y, A, B):
        with th.no_grad():
            A_e = th.sparse.mm(inc_x.float(), A) # shape: (e, d)
            B_e = th.sparse.mm(inc_y.float(), B) # shape: (e, d)
            ctx.save_for_backward(A_e, B_e, inc_x, inc_y)
            y = (A_e * B_e).sum(-1) # shape: (e)
        assert y.requires_grad==False
        return y

    @staticmethod
    def backward(ctx, grad): # shape: (e)
        A_e, B_e, inc_x, inc_y = ctx.saved_tensors
        dAe = grad.unsqueeze(-1) * B_e
        dBe = grad.unsqueeze(-1) * A_e
        dA = th.sparse.mm(inc_x.t().float(), dAe)
        dB = th.sparse.mm(inc_y.t().float(), dBe)
        return None, None, dA, dB

if __name__ == '__main__':
    import os
    batch_size = 512
    l = 30
    n = batch_size * l
    e = batch_size * (l ** 2)
    v = th.ones(e, dtype=th.uint8)
    if not os.path.exists('i.pt'):
        i = th.zeros(2, e, dtype=th.long)
        eid_r = th.zeros(e, dtype=th.long)
        eid_c = th.zeros(e, dtype=th.long)
        ptr_r = th.zeros(n + 1, dtype=th.long)
        ptr_c = th.zeros(n + 1, dtype=th.long)
        nid_r = th.zeros(e, dtype=th.long)
        nid_c = th.zeros(e, dtype=th.long)    
        cnt = 0
        for b in range(batch_size):
            for x in range(b * l, (b + 1) * l):
                ptr_r[x] = cnt
                for y in range(b * l, (b + 1) * l):
                    i[0, cnt] = x
                    i[1, cnt] = y
                    nid_r[cnt] = y
                    eid_r[cnt] = cnt
                    cnt += 1
        ptr_r[n] = cnt

        cnt = 0
        for b in range(batch_size):
            for y in range(b * l, (b + 1) * l):
                ptr_c[y] = cnt
                for x in range(b * l, (b + 1) * l):
                    nid_c[cnt] = x
                    eid_c[cnt] = b * l * l + (x % l) * l + (y % l)
                    cnt += 1
        ptr_c[n] = cnt

        th.save((i, eid_r, eid_c, ptr_r, ptr_c, nid_r, nid_c), 'i.pt')
    else:
        i, eid_r, eid_c, ptr_r, ptr_c, nid_r, nid_c = th.load('i.pt')

    adj = th.sparse.ByteTensor(i, v, th.Size([n, n]))

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
        th.save((i_x, i_y), 'ixy.pt')
    else:
        i_x, i_y = th.load('ixy.pt')

    inc_x = th.sparse.ByteTensor(i_x, v, th.Size([e, n]))
    inc_y = th.sparse.ByteTensor(i_y, v, th.Size([e, n])) 

    import time

    inc_x = inc_x.cuda()
    inc_y = inc_y.cuda()
    adj = adj.cuda()
    eid_r, eid_c, ptr_r, ptr_c, nid_r, nid_c = eid_r.cuda(), eid_c.cuda(), ptr_r.cuda(), ptr_c.cuda(), nid_r.cuda(), nid_c.cuda()

    print('simple implementation')
    dim = 1024 
    A = th.rand(n, dim, requires_grad=True, device='cuda:0')
    B = th.rand(n, dim, requires_grad=True, device='cuda:0')
    grad = th.rand(e, device='cuda:0')
    tic = time.time()
    A_e = th.sparse.mm(inc_x.float(), A)
    B_e = th.sparse.mm(inc_y.float(), B)
    y = (A_e * B_e).sum(-1)
    y_ori = y.clone()
    th.cuda.synchronize()
    print('forward elapse time: {}'.format(time.time() - tic))
    tic = time.time()
    y.backward(grad)
    th.cuda.synchronize()
    print('backward elapse time: {}'.format(time.time() - tic))
    A_grad_ori, B_grad_ori = A.grad.clone(), B.grad.clone()
    A.grad.zero_()
    B.grad.zero_()

    print('simple implementation, hand-crafted autograd')
    tic = time.time()
    y = MaskedMMSimple.apply(inc_x, inc_y, A, B)
    th.cuda.synchronize()
    print('forward elapse time: {}'.format(time.time() - tic))
    assert th.allclose(y, y_ori)
    tic = time.time()
    y.backward(grad)
    th.cuda.synchronize()
    print('backward elapse time: {}'.format(time.time() - tic))
    assert th.allclose(A.grad, A_grad_ori) and th.allclose(B.grad, B_grad_ori)
    A.grad.zero_()
    B.grad.zero_()
   
    print('custom kernel')
    tic = time.time()
    y = MaskedMM.apply(adj, A, B)
    th.cuda.synchronize()
    print('forward elapse time: {}'.format(time.time() - tic))
    assert th.allclose(y, y_ori)
    tic = time.time()
    y.backward(grad)
    th.cuda.synchronize()
    print('backward elapse time: {}'.format(time.time() - tic))
    assert th.allclose(A.grad, A_grad_ori) and th.allclose(B.grad, B_grad_ori)
    A.grad.zero_()
    B.grad.zero_()

    print('vanilla bmm')
    tic = time.time()
    y = (A.view(batch_size, l, dim) @ B.view(batch_size, l, dim).transpose(-1, -2)).view(-1)
    th.cuda.synchronize()
    print('forward elapse time: {}'.format(time.time() - tic))
    assert th.allclose(y, y_ori)
    tic = time.time()
    y.backward(grad)
    th.cuda.synchronize()
    print('backward elapse time: {}'.format(time.time() - tic))
    assert th.allclose(A.grad, A_grad_ori) and th.allclose(B.grad, B_grad_ori)
    A.grad.zero_()
    B.grad.zero_()


    print('custom kernel(csr)')
    tic = time.time()
    y = MaskedMMCSR.apply(ptr_r, eid_r, nid_r, ptr_c, eid_c, nid_c, A, B)
    th.cuda.synchronize()
    print('forward elapse time: {}'.format(time.time() - tic))
    assert th.allclose(y, y_ori)
    tic = time.time()
    y.backward(grad)
    th.cuda.synchronize()
    print('backward elapse time: {}'.format(time.time() - tic))
    assert th.allclose(A.grad, A_grad_ori) and th.allclose(B.grad, B_grad_ori)

    # ------------------------------------------------------------------------
    # Test sparse softmax
    # ------------------------------------------------------------------------
    print('------------------------------------')
    print('vanilla softmax(reduce)')
    tic = time.time()
    x = th.rand(e, requires_grad=True, device='cuda:0')
    y = th.softmax(x.view(batch_size, l, l), -1).view(-1)
    th.cuda.synchronize()
    print('forward elapse time: {}'.format(time.time() - tic))
    tic = time.time()
    y_ori = y.clone() 
    y.backward(grad)
    th.cuda.synchronize()
    print('backward elapse time: {}'.format(time.time() - tic))
    x_grad_ori = x.grad.clone()
    x.grad.zero_()
    
    print('custom softmax(reduce)')
    tic = time.time()
    y = SparseSoftmax.apply(ptr_r, eid_r, x)
    th.cuda.synchronize()
    print('forward elapse time: {}'.format(time.time() - tic))
    assert th.allclose(y_ori, y) 
    tic = time.time() 
    y.backward(grad)
    th.cuda.synchronize()
    print('backward elapse time: {}'.format(time.time() - tic))
    assert th.allclose(x_grad_ori, x.grad, rtol=1e-3, atol=1e-6)
    x.grad.zero_()

    print('vanilla softmax(scatter)')
    tic = time.time()
    x = th.rand(e, requires_grad=True, device='cuda:0')
    y = th.softmax(x.view(batch_size, l, l), -2).view(-1)
    th.cuda.synchronize()
    print('forward elapse time: {}'.format(time.time() - tic))
    tic = time.time()
    y_ori = y.clone() 
    y.backward(grad)
    th.cuda.synchronize()
    print('backward elapse time: {}'.format(time.time() - tic))
    x_grad_ori = x.grad.clone()
    x.grad.zero_()
    
    print('custom softmax(scatter)')
    tic = time.time()
    y = SparseSoftmax.apply(ptr_c, eid_c, x)
    th.cuda.synchronize()
    print('forward elapse time: {}'.format(time.time() - tic))
    assert th.allclose(y_ori, y) 
    tic = time.time() 
    y.backward(grad)
    th.cuda.synchronize()
    print('backward elapse time: {}'.format(time.time() - tic))
    assert th.allclose(x_grad_ori, x.grad, rtol=1e-3, atol=1e-6)
    x.grad.zero_()


