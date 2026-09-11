import torch, time, torch.nn as nn
torch.manual_seed(0)
class Net(nn.Module):
    def __init__(s,H):
        super().__init__(); s.ft=nn.EmbeddingBag(768,H,mode='sum'); s.b=nn.Parameter(torch.zeros(H)); s.out=nn.Linear(2*H,1)
    def forward(s, idx_w, off_w, idx_b, off_b):
        a=torch.clamp(s.ft(idx_w,off_w)+s.b,0,1)**2; b=torch.clamp(s.ft(idx_b,off_b)+s.b,0,1)**2
        return s.out(torch.cat([a,b],1))
def bench(dev,H,B=16384,steps=20,npieces=30):
    net=Net(H).to(dev); opt=torch.optim.Adam(net.parameters(),1e-3)
    idx=torch.randint(0,768,(B*npieces,),device=dev); off=torch.arange(0,B*npieces,npieces,device=dev)
    y=torch.rand(B,1,device=dev)
    for i in range(steps+3):
        if i==3:
            if dev=='mps': torch.mps.synchronize()
            t=time.perf_counter()
        opt.zero_grad(); loss=((torch.sigmoid(net(idx,off,idx,off))-y)**2).mean(); loss.backward(); opt.step()
    if dev=='mps': torch.mps.synchronize()
    dt=time.perf_counter()-t
    return B*steps/dt
torch.set_num_threads(8)
for dev in ['cpu','mps']:
    for H in (128,256,512):
        try: print(f"{dev} H={H}: {bench(dev,H)/1e6:.2f} M positions/s (fwd+bwd+Adam, batch 16384, 30 active features/side)")
        except Exception as e: print(dev,H,'ERR',e)
