import csv
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.font_manager as fm


plt.rcParams.update({'font.size': 18,'font.family':'serif'})
with open("acc_summary.csv", "r", encoding='utf-8') as file:
    reader = csv.reader(file)
    data = list(reader)

data=np.array(data,dtype='float')
# print(data.shape)

scenario_names=['Scenario 1','Scenario 2', 'Scenario 3']
data_scene=[]
width=0.8

for i in range(3):
    data_scene.append(data[i*10:i*10+10,:])

data_scene=np.array(data_scene)
print(data_scene.shape)

means = np.mean(data_scene, axis=1)
stds = np.std(data_scene, axis=1)
names = ['Voting', 'AB-DT', 'AB-SVM', 'AB-GNB', 'AB-LR', 'RF', 'MLP']
print(means.shape,stds.shape)

means_all=np.mean(means,axis=0)
stds_all=np.std(data_scene,axis=(0,1))
print(means_all.shape,stds_all.shape)

sorted_indices = np.argsort(means_all)  # [::-1]

sorted_datas = data_scene[:,:, sorted_indices]
sorted_means = means[:, sorted_indices]
sorted_stds = stds[:, sorted_indices]
sorted_means_all=means_all[sorted_indices]
sorted_stds_all=stds_all[sorted_indices]
sorted_names = [names[i] for i in sorted_indices]
print(sorted_names)
# fig = plt.figure(figsize=(5,15),dpi=300)
fig = plt.figure(figsize=(6,9),dpi=300)
x = np.arange(0,14,2)
width=0.5
cmap = plt.get_cmap('coolwarm')
colors=[cmap(0.0625+0.25),cmap(0.5),cmap(0.8125-0.125)]
for i in range(7):
    for j in range(3):
        if i == 0:
            plt.barh(x[i]+width-j*width, sorted_means[j,i], width, xerr=sorted_stds[j,i], capsize=5,
                 color=colors[j], edgecolor='k',label=scenario_names[j]) #alpha=0.3
        else:
            plt.barh(x[i]+width-j*width, sorted_means[j,i], width, xerr=sorted_stds[j,i], capsize=5,
                 color=colors[j], edgecolor='k')

        plt.text(0.95, x[i]+width-j*width-0.15, f'{sorted_means[j,i]:.2f}±{sorted_stds[j,i]:.2f}',
                 ha='center', rotation=0,fontsize=15)
    plt.text(1.02, x[i]-0.4, '}',
             ha='center', rotation=0, fontsize=40)
    # plt.barh(x[i] , sorted_means_all[i], width, xerr=sorted_stds_all[i], capsize=3,
    #                       color=cmap((i+0.5)/8), edgecolor='k')
    plt.text(1.09, x[i]-0.15, f'{sorted_means_all[i]:.2f}±{sorted_stds_all[i]:.2f}',
             ha='center', rotation=0, fontsize=15)
# plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.32), ncol=5,handletextpad=0)
plt.grid()
plt.yticks(x, sorted_names, rotation=0)
plt.xlabel('Balanced Accuracy')
plt.xlim(0.45, 0.90)
plt.ylabel('Classifiers')
plt.legend(loc='lower right', bbox_to_anchor=(1.35, -0.015), bbox_transform=plt.gcf().transFigure)

ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plt.savefig('vertical_plot.pdf', format='pdf', bbox_inches='tight')
plt.show()


sorted_datas_aeb=sorted_datas[0,:,:]
sorted_datas_cutin=sorted_datas[1,:,:]
sorted_datas_ped=sorted_datas[2,:,:]
def colormap_plot(data_colormap,mode):
    plt.rcParams.update({'font.size': 15, 'font.family': 'serif'})
    fig = plt.figure(figsize=(5, 5), dpi=300)
    subjects=["S1","S2","S3","S4","S5","S6","S7","S8","S9","S10"]

    plt.xticks(np.arange(len(subjects)), labels=subjects, rotation=45, rotation_mode="anchor", ha="right")
    plt.yticks(np.arange(len(sorted_names)), labels=sorted_names)

    im=plt.imshow(data_colormap.T,cmap=cmap, origin='lower',vmax=0.89)
    cbar = plt.colorbar(im, orientation='horizontal')
    cbar.set_label('Balanced Accuracy')
    cbar.ax.set_position([cbar.ax.get_position().x0, cbar.ax.get_position().y0 - 0.03,
                          cbar.ax.get_position().width, cbar.ax.get_position().height])
    for i in range(len(subjects)):
        for j in range(len(sorted_names)):
            relative_value=(data_colormap.T[j, i]-np.min(data_colormap))/(np.max(data_colormap)-np.min(data_colormap))
            if relative_value>0.9 or relative_value<0.1:
                color_text='white'
            else:
                color_text='black'
            plt.text(i, j, f'{data_colormap.T[j, i]:.2f}', ha='center', va='center', color=color_text,fontsize=11)
    plt.text(-2.5,-2.5,'Subjects',rotation=45)
    plt.text(-3,6.75,'Classifiers')


    # plt.savefig('classifier_{}.pdf'.format(mode), format='pdf', bbox_inches='tight')
    plt.show()

colormap_plot(sorted_datas_aeb,'aeb')
colormap_plot(sorted_datas_cutin,'cutin')
colormap_plot(sorted_datas_ped,'ped')