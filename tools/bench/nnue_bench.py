import numpy as np, time
from numba import njit, int16, int32, int64
import numba
print("numba", numba.__version__)

def make(H):
    W1 = np.random.randint(-127,128,size=(768,H)).astype(np.int16)
    b1 = np.random.randint(-127,128,size=H).astype(np.int16)
    W2 = np.random.randint(-64,65,size=2*H).astype(np.int16)
    return W1,b1,W2

@njit(cache=False, boundscheck=False, error_model='numpy', nogil=True)
def acc_update(acc, W1, added, removed, n_add, n_rem):
    # acc shape (2,H): two perspectives; added/removed shape (2,4) feature indices per perspective
    H = acc.shape[1]
    for p in range(2):
        for k in range(n_add):
            f = added[p,k]
            for i in range(H):
                acc[p,i] += W1[f,i]
        for k in range(n_rem):
            f = removed[p,k]
            for i in range(H):
                acc[p,i] -= W1[f,i]

@njit(cache=False, boundscheck=False, error_model='numpy', nogil=True)
def out_screlu(acc, W2, stm):
    H = acc.shape[1]
    s = int64(0)
    us = stm; them = 1-stm
    for i in range(H):
        v = int32(acc[us,i])
        if v < 0: v = 0
        elif v > 255: v = 255
        s += int64(v*v) * int64(W2[i])
    for i in range(H):
        v = int32(acc[them,i])
        if v < 0: v = 0
        elif v > 255: v = 255
        s += int64(v*v) * int64(W2[H+i])
    return s

@njit(cache=False, boundscheck=False, error_model='numpy', nogil=True)
def out_crelu(acc, W2, stm):
    H = acc.shape[1]
    s = int32(0)
    us = stm; them = 1-stm
    for i in range(H):
        v = int32(acc[us,i])
        if v < 0: v = 0
        elif v > 255: v = 255
        s += v * int32(W2[i])
    for i in range(H):
        v = int32(acc[them,i])
        if v < 0: v = 0
        elif v > 255: v = 255
        s += v * int32(W2[H+i])
    return s

@njit(cache=False, boundscheck=False, error_model='numpy', nogil=True)
def refresh(acc, W1, b1, feats, n):
    H = acc.shape[1]
    for p in range(2):
        for i in range(H): acc[p,i] = b1[i]
        for k in range(n):
            f = feats[p,k]
            for i in range(H): acc[p,i] += W1[f,i]

@njit(cache=False, boundscheck=False, error_model='numpy', nogil=True)
def node_loop(acc, W1, b1, W2, added, removed, N, stack):
    # simulate copy-make: copy acc into stack slot, update, evaluate
    H = acc.shape[1]
    tot = int64(0)
    for n in range(N):
        d = n & 63
        # copy-make: copy parent accumulator into child slot
        for p in range(2):
            for i in range(H): stack[d,p,i] = acc[p,i]
        acc_update(stack[d], W1, added, removed, 2, 2)
        tot += out_screlu(stack[d], W2, n & 1)
    return tot

for H in (64, 128, 256, 512):
    W1,b1,W2 = make(H)
    acc = np.zeros((2,H), np.int16)
    stack = np.zeros((64,2,H), np.int16)
    added = np.random.randint(0,768,size=(2,4)).astype(np.int64)
    removed = np.random.randint(0,768,size=(2,4)).astype(np.int64)
    feats = np.random.randint(0,768,size=(2,32)).astype(np.int64)
    # warmup
    node_loop(acc,W1,b1,W2,added,removed,1000,stack); refresh(acc,W1,b1,feats,32); out_crelu(acc,W2,0)
    N=2_000_000
    t=time.perf_counter(); node_loop(acc,W1,b1,W2,added,removed,N,stack); dt=time.perf_counter()-t
    per_node_ns = dt/N*1e9
    # refresh cost
    R=200_000
    t=time.perf_counter()
    for _ in range(R//1000):
        pass
    @njit(cache=False, boundscheck=False, error_model='numpy', nogil=True)
    def refresh_loop(acc,W1,b1,feats,R):
        for _ in range(R): refresh(acc,W1,b1,feats,32)
    refresh_loop(acc,W1,b1,feats,10)
    t=time.perf_counter(); refresh_loop(acc,W1,b1,feats,R); dr=(time.perf_counter()-t)/R*1e9
    @njit(cache=False, boundscheck=False, error_model='numpy', nogil=True)
    def crelu_loop(acc,W2,R):
        s=int32(0)
        for r in range(R): s+=out_crelu(acc,W2,r&1)
        return s
    crelu_loop(acc,W2,10)
    t=time.perf_counter(); crelu_loop(acc,W2,N); dc=(time.perf_counter()-t)/N*1e9
    print(f"H={H:4d}: copy+update(2add,2rem)+SCReLU-out per node = {per_node_ns:7.1f} ns ; CReLU-out only = {dc:6.1f} ns ; full refresh (32 feats) = {dr:7.1f} ns ; implied max nps if search costs 500ns/node: {1e9/(500+per_node_ns)/1e6:.2f} M")
