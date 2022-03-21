import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
plt.rcdefaults()

# 数据集

path="C:\\Users\\cc\\Desktop\\2020年，TCI,LCM\\所有的结果，表格在此\\数值仿真结果\\20200410绘图总表格.xlsx"
data_consider=pd.read_excel(path,sheet_name="考虑分心")


plt.figure(figsize=(10,8),dpi=200)
plt.figure(1)

# ------------------------------------------------TD,TS------------------------------------------------
ax1 = plt.subplot(321)
plt.plot(data_consider["Time(s)"],data_consider["follower_1_TD_following"],label="TD_following",color="b")
plt.plot(data_consider["Time(s)"],data_consider["follower_1_TD_total"],label="TD_total",color="black")
plt.plot(data_consider["Time(s)"],data_consider["follower_1_TD_telephone"],label="TD_telephone",color="y")
plt.grid(False)
plt.ylim(0,2,0.4)
plt.yticks([0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2])
plt.ylabel('TD', fontsize=12,color="b")
plt.legend(loc=4)
ax2=ax1.twinx()
plt.plot(data_consider["Time(s)"],data_consider["follower_1_SA"],label="SA",color="g")
plt.ylim(0,1,0.1)
plt.yticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.ylabel('SA', fontsize=12,color="g")
plt.xlim(20,40,0.1)
plt.xticks([20,25,30,35,40])
plt.xlabel("Time(s)",fontsize=12)
sns.set_context("notebook")
sns.set_style('whitegrid',{"xtick.major.size": 10, "ytick.major.size": 8})
ax1.grid(True, linestyle=':')
ax2.grid(True, linestyle=':')
plt.title("Follower_1")
ax1.set_xlabel("( a )")
plt.legend(loc=1)

ax3 = plt.subplot(322)
plt.plot(data_consider["Time(s)"],data_consider["follower_2_TD_following"],label="TD_following",color="b")
plt.plot(data_consider["Time(s)"],data_consider["follower_2_TC"],label="TC",color="r",linewidth=2,linestyle="--")
plt.ylim(0,2,0.4)
plt.yticks([0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2])
plt.ylabel('TD', fontsize=12,color="b")
plt.legend(loc=4)
ax4=ax3.twinx()
plt.plot(data_consider["Time(s)"],data_consider["follower_2_SA"],label="SA",color="g")
plt.ylim(0,1,0.1)
plt.yticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.ylabel('SA', fontsize=12,color="g")
plt.xlim(20,40,0.1)
plt.xticks([20,25,30,35,40])
plt.xlabel("Time(s)",fontsize=12)
sns.set_context("notebook")
sns.set_style('whitegrid',{"xtick.major.size": 10, "ytick.major.size": 8})
ax3.grid(True, linestyle=':')
ax4.grid(True, linestyle=':')
ax3.set_xlabel("( b )")

plt.title("Follower_2")
plt.legend(loc=1)
# ------------------------------------------------TD,TS------------------------------------------------



# ------------------------------------------------reaction time,sped------------------------------------------------

ax5 = plt.subplot(323)
plt.plot(data_consider["Time(s)"],data_consider["follower_1_Reaction_Time"],label="Reaction_Time",color="b")
plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.ylabel('Reaction_Time(s)', fontsize=12,color="b")
ax6=ax5.twinx()
plt.plot(data_consider["Time(s)"],data_consider["follower_1_vel"],label="Velocity",color="r")
plt.ylim(10,35,1)
plt.yticks([10,15,20,25,30,35])
plt.ylabel('Velocity(m/s)', fontsize=12,color="r")
plt.xlim(20,40,0.1)
plt.xticks([20,25,30,35,40])
plt.xlabel("Time(s)",fontsize=12)
sns.set_context("notebook")
sns.set_style('whitegrid',{"xtick.major.size": 10, "ytick.major.size": 8})
ax5.grid(True, linestyle=':')
ax6.grid(True, linestyle=':')
plt.title("Follower_1")
ax5.set_xlabel("( c )")


ax7 = plt.subplot(324)
plt.plot(data_consider["Time(s)"],data_consider["follower_2_Reaction_Time"],label="Reaction_Time",color="b")
plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.ylabel('Reaction_Time(s)', fontsize=12,color="b")
ax8=ax7.twinx()
plt.plot(data_consider["Time(s)"],data_consider["follower_2_vel"],label="Velocity",color="r")
plt.ylim(10,35,1)
plt.yticks([10,15,20,25,30,35])
plt.ylabel('Velocity(m/s)', fontsize=12,color="r")
plt.xlim(20,40,0.1)
plt.xticks([20,25,30,35,40])
plt.xlabel("Time(s)",fontsize=12)
sns.set_context("notebook")
sns.set_style('whitegrid',{"xtick.major.size": 10, "ytick.major.size": 8})
ax7.grid(True, linestyle=':')
ax8.grid(True, linestyle=':')
plt.title("Follower_2")
ax7.set_xlabel("( d )")



# ------------------------------------------------safety indicator------------------------------------------------

ax9 = plt.subplot(325)
plt.plot(data_consider["Time(s)"],data_consider["follower_1_FRI"],label="FRI",color="b")
plt.ylim(0,3,0.5)
plt.yticks([0,0.5,1,1.5,2,2.5,3])
plt.ylabel('FRI', fontsize=12,color="b")
ax10=ax9.twinx()
plt.plot(data_consider["Time(s)"],data_consider["follower_1_TTC"],label="TTC",color="r")
plt.ylim(0,30,1)
plt.yticks([0,5,10,15,20,25,30])
plt.ylabel('TTC(s)', fontsize=12,color="r")
plt.xlim(20,40,0.1)
plt.xticks([20,25,30,35,40])
plt.xlabel("Time(s)",fontsize=12)
sns.set_context("notebook")
sns.set_style('whitegrid',{"xtick.major.size": 10, "ytick.major.size": 8})
ax9.grid(True, linestyle=':')
ax10.grid(True, linestyle=':')
plt.title("Follower_1")
ax9.set_xlabel("( e )")

# plt.show()

ax11 = plt.subplot(326)
plt.plot(data_consider["Time(s)"],data_consider["follower_2_FRI"],label="FRI",color="b")
plt.ylim(0,3,0.5)
plt.yticks([0,0.5,1,1.5,2,2.5,3])
plt.ylabel('FRI', fontsize=12,color="b")
ax12=ax11.twinx()
plt.plot(data_consider["Time(s)"],data_consider["follower_2_TTC"],label="TTC",color="r")
plt.ylim(0,30,1)
plt.yticks([0,5,10,15,20,25,30])
plt.ylabel('TTC(s)', fontsize=12,color="r")
plt.xlim(20,40,0.1)
plt.xticks([20,25,30,35,40])
plt.xlabel("Time(s)")
sns.set_context("notebook")
sns.set_style('whitegrid',{"xtick.major.size": 10, "ytick.major.size": 8})
ax11.grid(True, linestyle=':')
ax12.grid(True, linestyle=':')
plt.title("Follower_2")
ax11.set_xlabel("( f )")





plt.tight_layout()

plt.savefig("TD,TS,with.png",bbox_inches='tight',dpi=600)
plt.show()

# ------------------------------------------------ACC,SPE------------------------------------------------




