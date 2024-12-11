import matplotlib.pyplot as plt
import numpy as np

line = 'WB919m'
data_dict = np.load('./data/examples/' + line + '.npy',allow_pickle=True).item()
complete_data = data_dict['data']
missing_mask = data_dict['mask']
missed_data = complete_data * missing_mask

data_dict = np.load('./data/examples/' + line + '_zs-scl.npy',allow_pickle=True).item()
ours = data_dict['data']

data_dict = np.load('./data/examples/' + line + '_traditional.npy',allow_pickle=True).item()
traditional = data_dict['data']

cmin = -4000
cmax = 4000
fig = plt.figure(figsize=[20,12])

plt.subplot(3,4,1)
plt.imshow(complete_data,cmap='RdGy',vmin=cmin, vmax=cmax,aspect='auto')
plt.yticks(ticks=np.arange(0, 896, 250), labels=[round(i) for i in np.arange(0, 3.584, 1)],fontsize=12,fontweight='bold')
plt.xticks(ticks=np.arange(0, 640, 143), labels=[round(i) for i in np.arange(0, 44.8, 10)],fontsize=12,fontweight='bold')
plt.ylabel('Time(s)',fontsize=12,fontweight='bold')
plt.xlabel('Distance(km)',fontsize=12,fontweight='bold')
plt.title('Complete field seismic data',fontsize=12,fontweight='bold')
ax = plt.gca()
ax.add_patch(plt.Rectangle((180, 100), 100, 100, color="blue", fill=False, linewidth=1))
ax.add_patch(plt.Rectangle((450, 620), 100, 100, color="red", fill=False, linewidth=1))
#
plt.subplot(3,4,2)
plt.imshow(missed_data,cmap='RdGy',vmin=cmin, vmax=cmax,aspect='auto')
plt.yticks(ticks=np.arange(0, 896, 250), labels=[round(i) for i in np.arange(0, 3.584, 1)],fontsize=12,fontweight='bold')
plt.xticks(ticks=np.arange(0, 640, 143), labels=[round(i) for i in np.arange(0, 44.8, 10)],fontsize=12,fontweight='bold')
plt.ylabel('Time(s)',fontsize=12,fontweight='bold')
plt.xlabel('Distance(km)',fontsize=12,fontweight='bold')
plt.title('Irregularly sampled field seismic data',fontsize=12,fontweight='bold')
ax = plt.gca()
ax.add_patch(plt.Rectangle((180, 100), 100, 100, color="blue", fill=False, linewidth=1))
ax.add_patch(plt.Rectangle((450, 620), 100, 100, color="red", fill=False, linewidth=1))
#
plt.subplot(3,4,3)
plt.imshow(ours,cmap='RdGy',vmin=cmin, vmax=cmax,aspect='auto')
plt.yticks(ticks=np.arange(0, 896, 250), labels=[round(i) for i in np.arange(0, 3.584, 1)],fontsize=12,fontweight='bold')
plt.xticks(ticks=np.arange(0, 640, 143), labels=[round(i) for i in np.arange(0, 44.8, 10)],fontsize=12,fontweight='bold')
plt.ylabel('Time(s)',fontsize=12,fontweight='bold')
plt.xlabel('Distance(km)',fontsize=12,fontweight='bold')
plt.title('Reconstruction by ZS-SCL',fontsize=12,fontweight='bold')
ax = plt.gca()
ax.add_patch(plt.Rectangle((180, 100), 100, 100, color="blue", fill=False, linewidth=1))
ax.add_patch(plt.Rectangle((450, 620), 100, 100, color="red", fill=False, linewidth=1))
#
plt.subplot(3,4,4)
plt.imshow(traditional,cmap='RdGy',vmin=cmin, vmax=cmax,aspect='auto')
plt.yticks(ticks=np.arange(0, 896, 250), labels=[round(i) for i in np.arange(0, 3.584, 1)],fontsize=12,fontweight='bold')
plt.xticks(ticks=np.arange(0, 640, 143), labels=[round(i) for i in np.arange(0, 44.8, 10)],fontsize=12,fontweight='bold')
plt.ylabel('Time(s)',fontsize=12,fontweight='bold')
plt.xlabel('Distance(km)',fontsize=12,fontweight='bold')
plt.title('Reconstruction by traditional learning',fontsize=12,fontweight='bold')
ax = plt.gca()
ax.add_patch(plt.Rectangle((180, 100), 100, 100, color="blue", fill=False, linewidth=1))
ax.add_patch(plt.Rectangle((450, 620), 100, 100, color="red", fill=False, linewidth=1))
#
plt.subplot(3,4,5)
plt.imshow(complete_data[100:200, 180:280],cmap='RdGy',vmin=cmin, vmax=cmax,aspect='auto')
plt.yticks(ticks=np.arange(0, 100, 30), labels=[round(i,2) for i in np.arange(0.4, 0.8, 0.12)],fontsize=12,fontweight='bold')
plt.xticks(ticks=np.arange(0, 100, 29), labels=[round(i) for i in np.arange(12.6, 19.6, 2)],fontsize=12,fontweight='bold')
#
plt.subplot(3,4,6)
plt.imshow(missed_data[100:200, 180:280],cmap='RdGy',vmin=cmin, vmax=cmax,aspect='auto')
plt.yticks(ticks=np.arange(0, 100, 30), labels=[round(i,2) for i in np.arange(0.4, 0.8, 0.12)],fontsize=12,fontweight='bold')
plt.xticks(ticks=np.arange(0, 100, 29), labels=[round(i) for i in np.arange(12.6, 19.6, 2)],fontsize=12,fontweight='bold')
#
plt.subplot(3,4,7)
plt.imshow(ours[100:200, 180:280],cmap='RdGy',vmin=cmin, vmax=cmax,aspect='auto')
plt.yticks(ticks=np.arange(0, 100, 30), labels=[round(i,2) for i in np.arange(0.4, 0.8, 0.12)],fontsize=12,fontweight='bold')
plt.xticks(ticks=np.arange(0, 100, 29), labels=[round(i) for i in np.arange(12.6, 19.6, 2)],fontsize=12,fontweight='bold')
#
plt.subplot(3,4,8)
cb_loc = plt.imshow(traditional[100:200, 180:280],cmap='RdGy',vmin=cmin, vmax=cmax,aspect='auto')
plt.yticks(ticks=np.arange(0, 100, 30), labels=[round(i,2) for i in np.arange(0.4, 0.8, 0.12)],fontsize=12,fontweight='bold')
plt.xticks(ticks=np.arange(0, 100, 29), labels=[round(i) for i in np.arange(12.6, 19.6, 2)],fontsize=12,fontweight='bold')
#
plt.subplot(3,4,9)
plt.imshow(complete_data[620:720, 450:550],cmap='RdGy',vmin=cmin, vmax=cmax,aspect='auto')
plt.yticks(ticks=np.arange(0, 100, 30), labels=[round(i,2) for i in np.arange(2.48, 2.88, 0.12)],fontsize=12,fontweight='bold')
plt.xticks(ticks=np.arange(0, 100, 29), labels=[round(i) for i in np.arange(31.5, 38.5, 2)],fontsize=12,fontweight='bold')
#
plt.subplot(3,4,10)
plt.imshow(missed_data[620:720, 450:550],cmap='RdGy',vmin=cmin, vmax=cmax,aspect='auto')
plt.yticks(ticks=np.arange(0, 100, 30), labels=[round(i,2) for i in np.arange(2.48, 2.88, 0.12)],fontsize=12,fontweight='bold')
plt.xticks(ticks=np.arange(0, 100, 29), labels=[round(i) for i in np.arange(31.5, 38.5, 2)],fontsize=12,fontweight='bold')
#
plt.subplot(3,4,11)
plt.imshow(ours[620:720, 450:550],cmap='RdGy',vmin=cmin, vmax=cmax,aspect='auto')
plt.yticks(ticks=np.arange(0, 100, 30), labels=[round(i,2) for i in np.arange(2.48, 2.88, 0.12)],fontsize=12,fontweight='bold')
plt.xticks(ticks=np.arange(0, 100, 29), labels=[round(i) for i in np.arange(31.5, 38.5, 2)],fontsize=12,fontweight='bold')
#
plt.subplot(3,4,12)
plt.imshow(traditional[620:720, 450:550],cmap='RdGy',vmin=cmin, vmax=cmax,aspect='auto')
plt.yticks(ticks=np.arange(0, 100, 30), labels=[round(i,2) for i in np.arange(2.48, 2.88, 0.12)],fontsize=12,fontweight='bold')
plt.xticks(ticks=np.arange(0, 100, 29), labels=[round(i) for i in np.arange(31.5, 38.5, 2)],fontsize=12,fontweight='bold')
#
fig.subplots_adjust(right=0.9)
l = 0.92
b = 0.12
w = 0.01
h = 1 - 2.1 * b

rect = [l, b, w, h]
cbar_ax = fig.add_axes(rect)
cb = plt.colorbar(cb_loc, cax=cbar_ax)
cb.ax.tick_params(labelsize=12,width=1)
plt.show()
