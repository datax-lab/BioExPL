import torch
import torch.nn as nn
import math

# BaseLine model (MLP)
class Baseline_MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, math.ceil(input_dim**0.5))
        self.fc3 = nn.Linear(math.ceil(input_dim**0.5), math.ceil((input_dim**0.5)**0.5))
        self.out = nn.Linear(math.ceil((input_dim**0.5)**0.5), 1)

    def forward(self, x, return_hidden=False):
        z1 = self.fc1(x)
        h1 = torch.relu(z1)

        h2 = torch.relu(self.fc2(h1))
        h3 = torch.relu(self.fc3(h2))

        out = self.out(h3).squeeze(-1)

        if return_hidden:
            return out, h1
        return out

# feature importance
def integrated_gradients_feature_importance(
    model,
    X,
    baseline=None,
    steps=50,
    batch_size=128,
    internal_batch_size=None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    X = X.to(device)

    if baseline is None:
        baseline = torch.zeros_like(X)
    else:
        baseline = baseline.to(device)
        if baseline.dim() == 1:
            baseline = baseline.unsqueeze(0).expand_as(X)
        elif baseline.shape != X.shape:
            raise ValueError("baseline must have shape (D,) or (N, D)")

    all_attr = []

    for start in range(0, X.size(0), batch_size):
        end = min(start + batch_size, X.size(0))

        xb = X[start:end]
        bb = baseline[start:end]

        diff = xb - bb          

        alphas = torch.linspace(0.0, 1.0, steps + 1, device=device)[1:].view(-1, 1, 1)
        scaled = bb.unsqueeze(0) + alphas * diff.unsqueeze(0)

        if internal_batch_size is None:
            internal_batch_size = steps

        grad_list = []

        for s_start in range(0, scaled.size(0), internal_batch_size):
            s_end = min(s_start + internal_batch_size, scaled.size(0))

            scaled_chunk = scaled[s_start:s_end]
            S, B, D = scaled_chunk.shape

            flat_input = scaled_chunk.reshape(S * B, D).clone().detach()
            flat_input.requires_grad_(True)

            output = model(flat_input)
            output = output.squeeze(-1)
            total_output = output.sum()

            grads = torch.autograd.grad(
                total_output,
                flat_input,
                create_graph=False,
                retain_graph=False,
            )[0]                   

            grads = grads.reshape(S, B, D) 
            grad_list.append(grads)

        grads_all = torch.cat(grad_list, dim=0)

        avg_grads = grads_all.mean(dim=0)       

        # IG = (x - baseline) * average gradient
        attr = diff * avg_grads                 

        all_attr.append(attr.detach().cpu())

    attributions = torch.cat(all_attr, dim=0)   
    return attributions