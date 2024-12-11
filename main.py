import numpy as np
import torch
import torch.optim as optim
import scipy.io as scio
from func.utils import trace_mask_generator, model_train, metric
import matplotlib.pyplot as plt

from func.model import CAE

# 1. load data
missing_ratio = 0.5
survey = 'bs' # ca: central alaska, bs: beaufort sea
Line = 'WB901m'

data = scio.loadmat('./data/' + survey + '_mat/' + Line +'.mat')['data']
data = torch.tensor(data).unsqueeze(0).unsqueeze(0).float()
missing_mask = trace_mask_generator(data, missing_ratio)
masked_data = data * missing_mask

# 2. train model
# 2.1 parameters
device = 'cuda'
max_epoch = 8000
lr = 0.001
mode = 'zs-scl'

data = data.to(device)
missing_mask = missing_mask.to(device)
masked_data = masked_data.to(device)

model = CAE(1, 8).to(device)
print('The number of learnable parameters: ', sum([param.nelement() for param in model.parameters()]))
_, _, h, w = data.shape
print('The number of sampling points (h*w): ', h*w)
loss_func = torch.nn.MSELoss()
optimizer = optim.Adam (model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(max_epoch / 4), gamma = 0.9)

# 2.2 training
model, out, loss1, loss2, loss3 = model_train(data=masked_data, model=model, optimizer=optimizer, scheduler=scheduler,  loss_func=loss_func, epoch=max_epoch,
                    mode=mode, missing_mask=missing_mask)

# 3 visualization
out = out.detach().cpu().squeeze(0).squeeze(0).numpy()
missing_mask = missing_mask.cpu().squeeze(0).squeeze(0).numpy()
masked_data = masked_data.cpu().squeeze(0).squeeze(0).numpy()
data = data.cpu().squeeze(0).squeeze(0).numpy()

snr_in, ssim_in, r2_in = metric(masked_data, data)
merged_snr_out, merged_ssim_out, merged_r2_out = metric(out * (1-missing_mask) + masked_data, data)

print('SNR of irregularly sampled data: ', snr_in)
print('SSIM of irregularly sampled data: ', ssim_in)
print('R2 of irregularly sampled data: ', r2_in)
print('    SNR of reconstructed data: ', merged_snr_out)
print('    SSIM of reconstructed data: ', merged_ssim_out)
print('    R2 of reconstructed data: ', merged_r2_out)

plt.subplot(2,4,1)
plt.imshow(out,'RdGy')
plt.clim(vmin=np.min(data), vmax=np.max(data))
plt.subplot(2,4,2)
plt.imshow(masked_data,'RdGy')
plt.clim(vmin=np.min(data), vmax=np.max(data))
plt.subplot(2,4,3)
plt.imshow(data,'RdGy')
plt.clim(vmin=np.min(data), vmax=np.max(data))
plt.subplot(2,4,4)
plt.imshow(data-out,'RdGy')
plt.clim(vmin=np.min(data), vmax=np.max(data))

plt.subplot(2,4,7)
plt.plot(loss1)
plt.plot(loss2)
plt.plot(loss3)
plt.legend(['loss1', 'loss2', 'loss3'])

plt.show()