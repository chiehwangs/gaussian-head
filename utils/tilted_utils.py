import torch

# randm to quaternion
def sample_uniform(key):
    # Uniformly sample over S^3.
    # Reference: http://planning.cs.uiuc.edu/node198.html
    key = key.detach().cpu().numpy()

    u1, u2, u3 = torch.rand(3, generator=torch.Generator().manual_seed(int(key)))

    a = torch.sqrt(1.0 - u1)
    b = torch.sqrt(u1)

    return [a * torch.sin(u2),a * torch.cos(u2),b * torch.sin(u3),b * torch.cos(u3),]

def as_matrix(x) -> torch.Tensor:
        norm = torch.norm(x, dim=-1)
        q = x * torch.sqrt(2.0 / norm.unsqueeze(-1))
        q = torch.bmm(q.unsqueeze(-1), q.unsqueeze(1)) 
        matrix = torch.cat(
            [
                1.0 - q[:, 2, 2] - q[:, 3, 3], q[:, 1, 2] - q[:, 3, 0], q[:, 1, 3] + q[:, 2, 0],
                q[:, 1, 2] + q[:, 3, 0], 1.0 - q[:, 1, 1] - q[:, 3, 3], q[:, 2, 3] - q[:, 1, 0],
                q[:, 1, 3] - q[:, 2, 0], q[:, 2, 3] + q[:, 1, 0], 1.0 - q[:, 1, 1] - q[:, 2, 2],
            ], dim=-1
            ).view(-1,3,3)
        return matrix.to(dtype=torch.float32, device='cuda')

class So3(torch.nn.Module):
    def __init__(self, num_points):
        super(So3, self).__init__()
        samples = []
        self.trans_num = 4
        for _ in range(self.trans_num):
             key = torch.randint(0, 2**32, (1,))
             sample = sample_uniform(key=key)
             samples.append(sample)

        samples = torch.tensor(samples)
        samples = samples.unsqueeze(0).expand(num_points, -1, -1) 
        self.tau = torch.nn.Parameter(samples, requires_grad=True)  
        
    def forward(self, coords, iters): 
        R =[]
        for i in range(self.trans_num):
                Ri = as_matrix(x=self.tau[:, i, :])
                R.append(Ri)
        R = torch.stack(R, dim=1)
        # "n transform i j, n j -> transform n j"
        transformed_coords = torch.einsum("nTij, nj -> Tnj", R, coords)  
        return transformed_coords 