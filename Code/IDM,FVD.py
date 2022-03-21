import numpy as np
from sko.GA import GA, GA_TSP
import pandas as pd
import  pandas  as pd
from  pandas import  DataFrame as df
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import math

import pandas as pd
import matplotlib.pyplot as plt

def get_mse(records_real, records_predict):
    """
    均方误差 估计值与真值 偏差
    """
    if len(records_real) == len(records_predict):
        return sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / len(records_real)
    else:
        return None


def get_rmse(records_real, records_predict):
    """
    均方根误差：是均方误差的算术平方根
    """
    mse = get_mse(records_real, records_predict)
    if mse:
        return math.sqrt(mse)
    else:
        return None

# 读取数据
total_data=pd.read_excel("C:\\Users\\cc\\Desktop\\2020年，TCI,LCM\\所有的结果，表格在此\\轨迹数据，第二版.xlsx",sheet_name="Sheet1")
data=df(total_data)

simulate_data=pd.DataFrame()
final_parameter_idm=pd.DataFrame()
final_loss_idm=pd.DataFrame()
final_parameter_fvd=pd.DataFrame()
final_loss_fvd=pd.DataFrame()

# ga的边界

lb_x_idm = [4,4,4,10,0.1,0.1,0.1]
ub_x_idm = [6,10,10,30,10,10,10]

lb_x_fvd = [10,0.1,0.1,0.1,0.1,5,4]
ub_x_fvd = [30,10,10,10,10,50,6]

# 开始便利数据中的每一段跟驰数据
for id,group in data.groupby("Pair_Number"):
    forehead_position = group["Leader_Pos"].values
    follower_position = group["Follower_Pos"].values
    forhead_speed = group["Leader_Spe"].values
    space_headway = group["Space_Headway"].values
    follower_speed = group["Follower_Spe"].values
    follower_acc = group["Follower_Acc"].values


    try:
        # IDM遗传算法
        # 遗传算法
        def function_idm(p):

            x1,x2,x3,x4,x5,x6,x7 = p
            Length_of_Car=x1
            Maximum_Acc = x2  # 最大加速度
            Comfortable_Dec =x3  # 舒适减速度
            Desire_Spe=x4  # 期望速度
            Desire_Spa_Tim =x5  # 期望车头时距
            Minimum_Spa = x6  # 最短车头间距
            Para_Beta = x7  # 加速度系数

            simulated_spe = []
            cur_sim_acc = []
            simulate_headway = []
            cur_sim_hea = space_headway[0]
            cur_sim_spe = follower_speed[0]

            for i in range(len(group)):

                cur_forhead_spe = forhead_speed[i]
                cur_head_space = space_headway[i]
                cur_follower_speed = follower_speed[i]

                try:
                    cur_des_spa = Minimum_Spa + cur_follower_speed * Desire_Spa_Tim + 0 * math.sqrt(cur_follower_speed / Desire_Spe) + \
                                  (cur_follower_speed * (cur_follower_speed - cur_forhead_spe)) / (
                                          2 * math.sqrt(Maximum_Acc * Comfortable_Dec))
                except:
                    cur_des_spa=5

                cur_follow_1_acc = Maximum_Acc * (1 - math.pow((cur_follower_speed / Desire_Spe), Para_Beta) -
                                         math.pow(cur_des_spa / (cur_head_space - Length_of_Car),
                                                  2))

                cur_sim_spe = cur_sim_spe + cur_follow_1_acc
                cur_sim_hea = cur_sim_hea + cur_sim_spe

                simulated_spe.append(cur_sim_spe)
                simulate_headway.append(cur_sim_hea)
                cur_sim_acc.append(cur_follow_1_acc)

            abs_sum_1 = 0
            for i in range(len(cur_sim_acc)):
                abs_sum_1 += abs(cur_sim_acc[i])

            abs_sum_2 = 0
            for i in range(len(cur_sim_acc)):
                abs_sum_2 += (cur_sim_acc[i] - follower_acc[i]) ** 2 / (abs(follower_acc[i]))

            youhua = abs_sum_2 / abs_sum_1

            return get_rmse(follower_acc,cur_sim_acc)

        ga_idm= GA(func=function_idm, n_dim=7, max_iter=20, size_pop=20, lb=lb_x_idm, ub=ub_x_idm)

        #得到最优的x和y
        best_x_idm, best_y_idm = ga_idm.run()
        cur_id_idm="%s,idm"%id

        def draw_picture_idm(p):

            x1,x2,x3,x4,x5,x6,x7 = p
            Length_of_Car=x1
            Maximum_Acc = x2  # 最大加速度
            Comfortable_Dec =x3  # 舒适减速度
            Desire_Spe=x4  # 期望速度
            Desire_Spa_Tim =x5  # 期望车头时距
            Minimum_Spa = x6  # 最短车头间距
            Para_Beta = x7  # 加速度系数

            simulated_spe = []
            cur_sim_acc = []
            simulate_position = []
            cur_sim_spe = follower_speed[0]
            cur_simu_position=follower_position[0]

            for i in range(len(group)):

                cur_forhead_spe = forhead_speed[i]
                cur_head_space = space_headway[i]
                cur_follower_speed = follower_speed[i]

                cur_des_spa = Minimum_Spa + cur_follower_speed * Desire_Spa_Tim + 0 * math.sqrt(cur_follower_speed / Desire_Spe) + \
                              (cur_follower_speed * (cur_follower_speed - cur_forhead_spe)) / (
                                      2 * math.sqrt(Maximum_Acc * Comfortable_Dec))

                cur_follow_1_acc = Maximum_Acc * (1 - math.pow((cur_follower_speed / Desire_Spe), Para_Beta) -
                                         math.pow(cur_des_spa / (cur_head_space - Length_of_Car),
                                                  2))

                simulated_spe.append(cur_sim_spe)
                simulate_position.append(cur_simu_position)

                cur_sim_spe = cur_sim_spe + cur_follow_1_acc*0.1
                cur_simu_position = cur_simu_position + cur_sim_spe*0.1

                cur_sim_acc.append(cur_follow_1_acc)


            return simulated_spe,cur_sim_acc,simulate_position

        simulate_spe_idm,simulated_acc_idm, simulate_postion_idm= draw_picture_idm(best_x_idm)



        # FVD遗传算法
        # 遗传算法
        def function_fvd(p):

            x1,x2,x3,x4,x5,x6,x7 = p
            # 期望速度
            Desire_Spe = x1
                # 常敏感性系数
            Constant_Sensitivity_Coe = x2
            # 相对速度差敏感性系数
            Rel_Spe_Sen_Coe = x3
            # beta，b系数
            Para_Beta=x4
            Para_b =x5
            # 自由流与跟驰策略的间距阈值
            Max_Fol_Dis = x6
            # 有效的车长
            Eff_veh_len = x7

            simulated_spe = []
            cur_sim_acc = []
            simulate_headway = []
            cur_sim_hea = space_headway[0]
            cur_sim_spe = follower_speed[0]

            for i in range(len(group)):

                cur_forhead_spe = forhead_speed[i]
                cur_head_space = space_headway[i]
                cur_follower_speed = follower_speed[i]

                # 后车参数的更新
                if cur_head_space <= Max_Fol_Dis:
                    cur_Rel_Spe_Sen_Coe = Rel_Spe_Sen_Coe
                else:
                    cur_Rel_Spe_Sen_Coe = 0

                cur_bef_des_spe = (0.5 * Desire_Spe) * (
                        math.tanh(((cur_head_space - Eff_veh_len) / Para_b - Para_Beta))
                        - math.tanh(-Para_Beta))
                cur_follow_1_acc = Constant_Sensitivity_Coe * (cur_bef_des_spe - cur_follower_speed) + cur_Rel_Spe_Sen_Coe * (cur_forhead_spe - cur_follower_speed)



                cur_sim_spe = cur_sim_spe + cur_follow_1_acc
                cur_sim_hea = cur_sim_hea + cur_sim_spe

                simulated_spe.append(cur_sim_spe)
                simulate_headway.append(cur_sim_hea)
                cur_sim_acc.append(cur_follow_1_acc)

            return get_rmse(follower_acc,cur_sim_acc)

        ga_fvd= GA(func=function_fvd, n_dim=7, max_iter=20, size_pop=20, lb=lb_x_fvd, ub=ub_x_fvd)

        #得到最优的x和y
        best_x_fvd, best_y_fvd = ga_fvd.run()
        cur_id_fvd="%s,fvd"%id

        def draw_picture_fvd(p):

            x1,x2,x3,x4,x5,x6,x7 = p
            # 期望速度
            Desire_Spe = x1
            # 常敏感性系数
            Constant_Sensitivity_Coe = x2
            # 相对速度差敏感性系数
            Rel_Spe_Sen_Coe = x3
            # beta，b系数
            Para_Beta=x4
            Para_b =x5
            # 自由流与跟驰策略的间距阈值
            Max_Fol_Dis = x6
            # 有效的车长
            Eff_veh_len = x7

            simulated_spe = []
            cur_sim_acc = []
            simulate_position = []
            cur_sim_spe = follower_speed[0]
            cur_simu_position=follower_position[0]

            for i in range(len(group)):

                cur_forhead_spe = forhead_speed[i]
                cur_head_space = space_headway[i]
                cur_follower_speed = follower_speed[i]

                # 后车参数的更新
                if cur_head_space <= Max_Fol_Dis:
                    cur_Rel_Spe_Sen_Coe = Rel_Spe_Sen_Coe
                else:
                    cur_Rel_Spe_Sen_Coe = 0

                cur_bef_des_spe = (0.5 * Desire_Spe) * (
                        math.tanh(((cur_head_space - Eff_veh_len) / Para_b - Para_Beta))
                        - math.tanh(-Para_Beta))
                cur_follow_1_acc = Constant_Sensitivity_Coe * (cur_bef_des_spe - cur_follower_speed) + cur_Rel_Spe_Sen_Coe * (cur_forhead_spe - cur_follower_speed)



                simulated_spe.append(cur_sim_spe)
                simulate_position.append(cur_simu_position)

                cur_sim_spe = cur_sim_spe + cur_follow_1_acc*0.1
                cur_simu_position = cur_simu_position + cur_sim_spe*0.1

                cur_sim_acc.append(cur_follow_1_acc)

            return simulated_spe,cur_sim_acc,simulate_position

        simulate_spe_fvd,simulated_acc_fvd, simulate_postion_fvd= draw_picture_fvd(best_x_fvd)

        # ________________________________________________________________________________输出

        cur_id_sim_acc_fvd = "%s,simulate_acc" % cur_id_fvd
        cur_id_sim_spe_fvd = "%s,simulate_spe" % cur_id_fvd
        cur_id_sim_follower_position_fvd = "%s,sim_follower_poi" % cur_id_fvd

        cur_id_sim_acc_idm="%s,simulate_acc"%cur_id_idm
        cur_id_tru_acc_idm="%s,true_acc"%cur_id_idm
        cur_id_tru_spe_idm="%s,true_spe"%cur_id_idm
        cur_id_sim_spe_idm="%s,simulate_spe"%cur_id_idm
        cur_id_tru_follower_position_idm="%s,true_follower_poi"%cur_id_idm
        cur_id_sim_follower_position_idm="%s,sim_follower_poi"%cur_id_idm
        cur_id_tru_forehead_position_idm="%s,leader_poi"%cur_id_idm

        # 导出
        cur_pd_id=pd.DataFrame()
        # 加速度
        cur_pd_id[cur_id_tru_acc_idm]=follower_acc
        cur_pd_id[cur_id_sim_acc_idm]=simulated_acc_idm
        cur_pd_id[cur_id_sim_acc_fvd]=simulated_acc_fvd

        # 速度
        cur_pd_id[cur_id_tru_spe_idm]=follower_speed
        cur_pd_id[cur_id_sim_spe_idm]=simulate_spe_idm
        cur_pd_id[cur_id_sim_spe_fvd]=simulate_spe_fvd

        cur_pd_id[cur_id_tru_forehead_position_idm]=forehead_position
        cur_pd_id[cur_id_tru_follower_position_idm]=follower_position
        cur_pd_id[cur_id_sim_follower_position_idm]=simulate_postion_idm
        cur_pd_id[cur_id_sim_follower_position_fvd]=simulate_postion_fvd


        final_parameter_idm[cur_id_idm]=best_x_idm
        final_loss_idm[cur_id_idm]=best_y_idm
        final_parameter_fvd[cur_id_fvd]=best_x_fvd
        final_loss_fvd[cur_id_fvd]=best_y_fvd

        simulate_data = pd.concat([cur_pd_id,simulate_data],axis=1)

    except:
        ""


writer = pd.ExcelWriter('C:\\Users\\cc\\Desktop\\2020年，TCI,LCM\\所有的结果，表格在此\\GA结果\IDM和FVD\\IDM,FVD.xlsx')
simulate_data.to_excel(writer,sheet_name="simulate_data")
final_parameter_idm.to_excel(writer,sheet_name="final_parameter_idm")
final_parameter_fvd.to_excel(writer,sheet_name="final_parameter_fvd")
final_loss_fvd.to_excel(writer,sheet_name="final_loss_fvd")
final_loss_idm.to_excel(writer,sheet_name="final_loss_idm")

writer.save()
