import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score
import random

plt.rcParams.update({'font.size': 15,'font.family':'serif'})
cmap = plt.get_cmap('coolwarm')


scene='cutin' # 'aeb' for Scenario 1; 'cutin' for Scenario 2; 'ped' for Scenario 3
RL_agent=[]
RL_list=[]
for i in [1,2,3,4,5,6,7,8,9,10]:
    detect=np.load(r'.\Human-in-loop Data\{}\total_rewards_list_test{}.npy'.format(scene,i),allow_pickle=True)
    detect = np.hstack(detect)
    detect = detect.reshape((30, 2))
    RL_agent.append(detect[:, 1])

    RL = np.load(r'.\Pure RL Data\{}\total_rewards_list_test{}.npy'.format(scene,i), allow_pickle=True)
    RL = np.hstack(RL[:30])
    RL_list.append(RL)

RL_agent=np.array(RL_agent)
RL_list=np.array(RL_list)

RL_agent_mean=np.mean(RL_agent,axis=0)
RL_list_mean=np.mean(RL_list,axis=0)

x=np.linspace(1,30,30)
# Calculate Standard Error
RL_agent_sde = np.std(RL_agent, axis=0)/np.sqrt(10)
RL_list_sde = np.std(RL_list, axis=0)/np.sqrt(10)


RL_agent_upper = RL_agent_mean + RL_agent_sde
RL_agent_lower = RL_agent_mean - RL_agent_sde
RL_list_upper = RL_list_mean + RL_list_sde
RL_list_lower = RL_list_mean - RL_list_sde

fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=300)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.plot(x, RL_list_mean, label='w/o fNIRS', color=cmap(0.0625))
plt.fill_between(x, RL_list_lower, RL_list_upper, alpha=0.8, facecolor=cmap(0.0625+0.25))
plt.plot(x, RL_agent_mean, label='w/ fNIRS', color=cmap(0.8125+0.125))
plt.fill_between(x, RL_agent_lower, RL_agent_upper, alpha=0.8, facecolor=cmap(0.8125+0.125-0.25))

plt.xlabel('Episodes')
plt.ylabel('Rewards')

plt.xlim(1,30)
plt.grid()
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.45), ncol=5,handletextpad=0)
plt.show()

def calculate_ttc(obs):
    obs[:, 0] = obs[:, 0] / 3.6
    obs[:, 4] = obs[:, 4] / 3.6
    ttc_list=[]
    for i in range(25,len(obs)):
        if abs(obs[i, 0]) > abs(obs[i, 4]):
            distance = max(abs(obs[i, 2] - obs[i, 6])-4,0)
            velocity = abs(obs[i, 0] - obs[i, 4])
            ttc=min(distance/velocity,10)
        else:
            ttc=float('inf')
        # if ttc<10:
        ttc_list.append(ttc)
    ttc_list=np.array(ttc_list)
    return ttc_list

def calculate_jerk(obs,time):
    velocity=obs[25:, 0]/3.6
    time=time[25:]
    # 计算速度的一阶导数
    first_derivative = np.gradient(velocity, time)

    # 计算速度的二阶导数
    second_derivative = np.gradient(first_derivative, time)
    positive_values = second_derivative[second_derivative > 0]
    negative_values = second_derivative[second_derivative < 0]

    return np.mean(positive_values),np.mean(negative_values)


def get_risk_field(obs):
    # risk_field = 0
    G = 0.001
    # k1 = 1
    k2 = 0.05
    M = 1705  # 920
    risk_fields=[]
    for i in range(len(obs)):
        ego_x = obs[i,2]
        ego_y = obs[i,3]


        actor_x = obs[i,6]
        actor_y = obs[i,7]
        actor_vx = obs[i,4]
        actor_vy = obs[i,5]
        distance = np.sqrt((ego_x - actor_x) ** 2 + (ego_y - actor_y) ** 2)
        actor_v = np.sqrt(actor_vx ** 2 + actor_vy ** 2)
        if actor_v!=0:
            cos_theta = ((actor_vx * (ego_x - actor_x) + actor_vy * (ego_y - actor_y)) / (distance * actor_v))
            risk_field = G * M / distance * np.exp(k2 * cos_theta * actor_v/3.6)
        else:
            risk_field =G*M/distance
        risk_fields.append(risk_field)

    return risk_fields
ttc_min_RL=[]
ttc_min_fNIRS=[]
ttc_danger_RL=[]
ttc_danger_fNIRS=[]
jerkpos_RL=[]
jerkpos_fNIRS=[]
jerkneg_RL=[]
jerkneg_fNIRS=[]
risk_RL=[]
risk_fNIRS=[]
for i in [1,2,3,4,5,6,7,8,9,10]:
    print(i)
    obs1 = np.load(r'.\Pure RL Data\{}\observation_test_total{}.npy'.format(scene,i), allow_pickle=True)
    time1 = np.load(r'.\Pure RL Data\{}\time_test_total{}.npy'.format(scene,i), allow_pickle=True)
    for j in range(0,30):
        obs1j = abs(obs1[j])
        if j in range(25,30):
            ttc1=calculate_ttc(obs1j)
            ttc_min_RL.append(min(ttc1))
            posjerk1,negjerk1=calculate_jerk(obs1j,time1[j])
            jerkpos_RL.append(posjerk1)
            jerkneg_RL.append(negjerk1)

        risk1 = get_risk_field(obs1j)
        risk_RL.append(max(risk1))
    ttc_danger_RL_j=[]

for i in [1,2,3,4,5,6,7,8,9,10]:
    print(i)
    obs2 = np.load(r'.\Human-in-loop Data\{}\observation_test_total{}.npy'.format(scene,i), allow_pickle=True)
    time2 = np.load(r'.\Human-in-loop Data\{}\time_test_total{}.npy'.format(scene,i), allow_pickle=True)

    for j in range(0,30):
        obs2j = abs(obs2[2 * j])
        if j in range(25,30):
            ttc2=calculate_ttc(obs2j)
            ttc_min_fNIRS.append(min(ttc2))
            posjerk2,negjerk2=calculate_jerk(obs2j,time2[2*j])
            jerkpos_fNIRS.append(posjerk2)
            jerkneg_fNIRS.append(negjerk2)

        risk2 = get_risk_field(obs2j)
        risk_fNIRS.append(max(risk2))

    ttc_danger_fNIRS_j=[]

import scipy.stats as stats

data1=np.array(risk_RL)
data2=np.array(risk_fNIRS)
data1=data1.reshape(10,30)
data2=data2.reshape(10,30)

fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=300)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

means1 = np.mean(data1, axis=0)
stds1 = np.std(data1, axis=0)

means2 = np.mean(data2, axis=0)
stds2 = np.std(data2, axis=0)

data1_higher = means1 + stds1
data1_lower = means1 - stds1
data2_higher = means2 + stds2
data2_lower = means2 - stds2

x = np.arange(1,31)


plt.plot(x, means1, label='w/o fNIRS', color=cmap(0.0625))
plt.fill_between(x, data1_lower, data1_higher, alpha=0.8, facecolor=cmap(0.0625+0.25))  # 填充方差范围
plt.plot(x, means2, label='w/ fNIRS', color=cmap(0.8125+0.125))
plt.fill_between(x, data2_lower, data2_higher, alpha=0.8, facecolor=cmap(0.8125+0.125-0.25))  # 填充方差范围
plt.grid()
plt.xlim((1,30))
plt.xlabel('Episodes')
plt.ylabel('Maximum Risk Field')
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.45), ncol=5,handletextpad=0)
plt.show()

def create_violin_plot(data1, data2, y_label, test_side):
    fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=300)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    box = plt.boxplot([data1, data2], positions=[1, 2], widths=0.25)

    for median in box['medians']:
        median.set(color='red')
    violin_parts = plt.violinplot([data1, data2], bw_method="scott", showmedians=False, showextrema=False)

    violin_parts['bodies'][0].set_facecolor(cmap(0.0625 + 0.25))
    violin_parts['bodies'][0].set_alpha(1)
    violin_parts['bodies'][1].set_facecolor(cmap(0.8125 - 0.125))
    violin_parts['bodies'][1].set_alpha(1)
    plt.grid()
    print(violin_parts)

    outliers = [flier.get_ydata() for flier in box['fliers']]
    colors = [cmap(0.0625), cmap(0.8125 + 0.125)]

    for i, data in enumerate([data1, data2]):
        for idx, val in enumerate(data):
            if val in outliers[i]:
                plt.scatter(i + 1, val, edgecolors='none', facecolors=colors[i], alpha=0.7, marker='o', s=20)
            else:
                plt.scatter(np.random.normal(i + 1, 0.02, 1), val, edgecolors='none', facecolors=colors[i], alpha=0.7,
                            marker='o', s=20)

    plt.xticks([1, 2], ['w/o fNIRS', 'w/ fNIRS'])

    U_stat, p_val = stats.wilcoxon(data1, data2, alternative=test_side)
    print(U_stat, p_val)


    y_pos = max(max(data1), max(data2)) + 0.1 * max(max(data1) - min(data1), max(data2) - min(data2))
    size = max(max(data1) - min(data1), max(data2) - min(data2))
    if p_val < 0.001:
        plt.text(1.5, max(max(data1), max(data2)) + 0.1 * max(max(data1) - min(data1), max(data2) - min(data2)),
                 '***', ha='center', va='bottom', color='k', fontsize=15, weight='bold')
        plt.text(1.5, max(max(data1), max(data2)) + 0.25 * max(max(data1) - min(data1), max(data2) - min(data2)),
                 f'p = {p_val:.2e}', ha='center', va='bottom', color='k', fontsize=15)
        plt.plot([1, 2], [y_pos, y_pos], linewidth=1, color='k')
        plt.plot([2, 2], [y_pos, y_pos - 0.04 * size], linewidth=1, color='k')
        plt.plot([1, 1], [y_pos, y_pos - 0.04 * size], linewidth=1, color='k')
    elif p_val < 0.01:
        plt.text(1.5, max(max(data1), max(data2)) + 0.1 * max(max(data1) - min(data1), max(data2) - min(data2)),
                 '**', ha='center', va='bottom', color='k', fontsize=15, weight='bold')
        plt.text(1.5, max(max(data1), max(data2)) + 0.25 * max(max(data1) - min(data1), max(data2) - min(data2)),
                 f'p = {p_val:.3f}', ha='center', va='bottom', color='k', fontsize=15)
        plt.plot([1, 2], [y_pos, y_pos], linewidth=1, color='k')
        plt.plot([2, 2], [y_pos, y_pos - 0.04 * size], linewidth=1, color='k')
        plt.plot([1, 1], [y_pos, y_pos - 0.04 * size], linewidth=1, color='k')
    elif p_val < 0.05:
        plt.text(1.5, max(max(data1), max(data2)) + 0.1 * max(max(data1) - min(data1), max(data2) - min(data2)),
                 '*', ha='center', va='bottom', color='k', fontsize=15, weight='bold')
        plt.text(1.5, max(max(data1), max(data2)) + 0.25 * max(max(data1) - min(data1), max(data2) - min(data2)),
                 f'p = {p_val:.3f}', ha='center', va='bottom', color='k', fontsize=15)
        plt.plot([1, 2], [y_pos, y_pos], linewidth=1, color='k')
        plt.plot([2, 2], [y_pos, y_pos - 0.04 * size], linewidth=1, color='k')
        plt.plot([1, 1], [y_pos, y_pos - 0.04 * size], linewidth=1, color='k')

    plt.ylabel(y_label)
    plt.xlim((0.5, 2.5))
    plt.show()

create_violin_plot(ttc_min_RL,ttc_min_fNIRS,'Minimum TTC(s)','less')
create_violin_plot(jerkneg_RL,jerkneg_fNIRS,'Average Negative Jerk (m/s$^3$)','less')
create_violin_plot(jerkpos_RL,jerkpos_fNIRS,'Average Positive Jerk (m/s$^3$)','greater')

