from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler
import torch.nn as nn
import torch.optim as optim
from zstd_store import load_tensor
from Imports import device
from tqdm import trange
import torch 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(56, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        return self.net(x).squeeze(-1)  # Ensures output shape is (N,)

X = load_tensor("../DATA/nns/NNS_1.pt.zst").to(device)/128
Y = load_tensor("../DATA/SGM/SGM_1.pt.zst").to(device)


@profile
def my_code():
    N = 1

    print('data loaded')
    batch_size_tr = max(X.shape[0] // 1000, 1)
    batch_size_vl = max(X.shape[0] // 1000, 1)

    model = Net().to(device=device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    scheduler = CosineAnnealingLR(optimizer, T_max=100)

    best_loss = torch.inf
    best_model_path = "../MODEL/model.pt"

    epoch_bar = trange(1000, desc="Epochs")
    for epoch in epoch_bar:
        torch.cuda.empty_cache()
        perm = torch.randperm(X.shape[0], device=X.device)

        model.train()
        for i in range(0, X.shape[0], batch_size_tr):
            idx = perm[i:i+batch_size_tr]
            xb, yb = X[idx], Y[idx]
            xb = xb.to(dtype=torch.float32)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        # ⬇️ Step the scheduler after each epoch
        scheduler.step()

        val_loss = 0.0
        with torch.no_grad():
            model.eval()
            for i in range(0, X.shape[0], batch_size_vl):
                xb, yb = X[i:i+batch_size_vl].contiguous(), Y[i:i+batch_size_vl].contiguous()
                xb = xb.to(dtype=torch.float32)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.detach() * xb.size(0)

            val_loss /= X.shape[0]

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), best_model_path)

            if best_loss < 1e-4:
                break

        epoch_bar.set_postfix(loss=val_loss, best_loss=best_loss)

    

    
if __name__ == "__main__":

    my_code()
    model = Net().to(device)
    best_model_path = "../MODEL/model.pt"
    model.load_state_dict(torch.load(best_model_path))
    model.eval()  # Set to evaluation mode if you're going to use it for inference
    with torch.no_grad():
        y = model(X[0,:])
        import pdb; pdb.set_trace()
        
   