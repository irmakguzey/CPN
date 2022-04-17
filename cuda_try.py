import torch
import torch.nn as nn
import timeit

print("Beginning..")

t0 = timeit.default_timer()
if torch.cuda.is_available():
    torch.cuda.manual_seed(2809)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda:1')
    ngpus = torch.cuda.device_count()
    print("Using {} GPU(s)...".format(ngpus))
print("Setup takes {:.2f}".format(timeit.default_timer()-t0))

t1 = timeit.default_timer()
model = nn.Sequential(
    nn.Conv2d(3, 6, 3, 1, 1),
    nn.ReLU(),
    nn.Conv2d(6, 1, 3, 1, 1)
)
print("Model init takes {:.2f}".format(timeit.default_timer()-t1))


if torch.cuda.is_available():
    t2 = timeit.default_timer()
    model = model.to(device)
print("Model to device takes {:.2f}".format(timeit.default_timer()-t2))

t3 = timeit.default_timer()
torch.cuda.synchronize()
print("Cuda Synch takes {:.2f}".format(timeit.default_timer()-t3))

print('done')