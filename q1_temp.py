import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties
from scipy.optimize import curve_fit
from math import *
from scipy.optimize import fsolve
import mpmath
import sys
import time

# 修改浮点精度
mpmath.mp.dps = 50  # 设置精度为50位小数

font1 = FontProperties(fname="F:/临时/字体/simsun.ttc")
plt.rcParams.update({'font.size': 14})


def Prepare_for_processing():
    ############################ 压力(MPa)与弹性模量取自然对数(ln(MPa))拟合 ############
    df = pd.read_excel("附件3-弹性模量与压力.xlsx", usecols=['压力(MPa)', '弹性模量(ln)'])
    x = df.loc[:, '压力(MPa)'].tolist()
    y = df.loc[:, '弹性模量(ln)'].tolist()

    # x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # y = np.array([2.5, 4.5, 4.8, 5.5, 6.0, 7.0, 7.8, 8.0, 9.0, 9.5])

    # # 计算回归系数
    # slope, intercept = np.polyfit(x, y, 1) # 斜率，截距
    # print("斜率：", slope, " 截距：", intercept)
    #
    # # 绘制拟合曲线
    # plt.scatter(x, y)
    # plt.plot(x, slope * np.array(x, dtype=np.float64) + intercept, color='red')
    #
    # plt.show()

    # 二次函数拟合 (degree=2)
    coefficients = np.polyfit(x, y, 2)
    print("拟合系数 (a, b, c):", coefficients)

    # 创建拟合函数
    poly_func = np.poly1d(coefficients)
    print("拟合函数:", poly_func)

    # 计算拟合值
    y_fit = poly_func(x)

    # 计算R²值
    residuals = y - y_fit
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"R²值: {r_squared:.4f}")

    # 绘制结果
    plt.rcParams['font.family'] = 'SimHei'
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='原始数据', color='blue')
    plt.plot(x, y_fit, label='二次拟合', color='red', linewidth=2)
    plt.xlabel('X：压力(MPa)', fontproperties=font1, fontsize=18)
    plt.ylabel('Y：弹性模量取自然对数(ln(MPa))', fontproperties=font1, fontsize=18)
    plt.title('二次函数拟合', fontproperties=font1, fontsize=18)
    plt.legend()
    plt.grid(True)
    plt.show()

    ############################ 解微分方程得密度与压强关系 并且拟合 ############
    ############################ 画出 ρ=k*f(p),k=0.85/f(100) ############
    """
    # 参数
    a = 5.039e-06
    b = 0.002891
    c = 7.343

    # 定义函数 f(p, rho) = drho/dp
    def f(p, rho):
        E = np.exp(a * p ** 2 + b * p + c)
        return rho / E

    # RK4方法
    def rk4(p0, rho0, p_end, h):
        n = int((p_end - p0) / h) + 1
        ps = np.linspace(p0, p_end, n)
        rhos = np.zeros(n)
        rhos[0] = rho0

        for i in range(n - 1):
            p_i = ps[i]
            rho_i = rhos[i]

            k1 = h * f(p_i, rho_i)
            k2 = h * f(p_i + h / 2, rho_i + k1 / 2)
            k3 = h * f(p_i + h / 2, rho_i + k2 / 2)
            k4 = h * f(p_i + h, rho_i + k3)

            rhos[i + 1] = rho_i + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return ps, rhos

    # 初始条件
    p0 = 100.0
    rho0 = 0.850

    # 正向积分（p > 100）
    p_end_forward = 200.0  # 积分到p=200
    h = 0.1  # 步长

    # 反向积分（p < 100）
    p_end_backward = 0.0  # 积分到p=0

    print("正向积分（p > 100）:")
    ps_forward, rhos_forward = rk4(p0, rho0, p_end_forward, h)
    print("反向积分（p < 100）:")
    ps_backward, rhos_backward = rk4(p0, rho0, p_end_backward, -h)  # 负步长表示反向积分

    # 合并结果
    ps = np.concatenate((ps_backward[::-1], ps_forward))
    print("ps的--为：", int(len(ps)))
    rhos = np.concatenate((rhos_backward[::-1], rhos_forward))
    for i in range(len(rhos)):
        # rhos[i] = exp(rhos[i])
        rhos[i] = rhos[i] * 0.5

    # 绘图
    plt.rcParams['font.family'] = 'SimHei'
    plt.figure(figsize=(12, 8))

    # p vs ρ 曲线
    plt.subplot(1, 1, 1)
    plt.plot(ps, rhos, 'b-', linewidth=2)
    plt.xlabel('压强 p(MPa)')
    plt.ylabel('密度 ρ(mg/mm³)')
    plt.title('密度与压强的关系: ρ vs p')
    plt.grid(True)
    plt.axvline(x=100, color='r', linestyle='--', alpha=0.7, label='初始点 (p=100)')
    plt.axhline(y=0.850, color='r', linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 输出结果表格
    # print("\n数值解结果:")
    # print("p\t\tρ\t\tln(ρ)")
    # print("-" * 40)
    # for i in range(0, len(ps), 50):  # 每50个点输出一次
    #     print(f"{ps[i]:8.2f}\t{rhos[i]:10.6f}\t{np.log(rhos[i]):10.6f}")

    # 验证初始条件
    initial_idx = np.where(np.isclose(ps, 100))[0][0]
    print(f"\n验证初始条件: p = {ps[initial_idx]:.2f}, ρ = {rhos[initial_idx]:.6f}")
    """

    # 定义函数
    def rho(p):
        return 0.000392 * np.exp(0.000005039 * p ** 2 + 0.00289 * p + 7.343)

    # 创建p值的范围（选择合适的范围很重要）
    p_values = np.linspace(0, 200, 1000)  # 调整范围以显示重要特征
    rho_values = rho(p_values)

    # 打印一些关键信息
    print("函数在p=0时的值:", rho(0))
    print("函数在p=-1000时的值:", rho(-1000))
    print("函数在p=500时的值:", rho(500))
    print("函数最小值大约在p =", p_values[np.argmin(rho_values)])
    print("函数最大值大约在p =", p_values[np.argmax(rho_values)])

    # 可选：绘制线性坐标的图像进行比较
    plt.figure(figsize=(12, 7))
    plt.plot(p_values, rho_values, 'r-', linewidth=2)
    plt.title(r'函数图像（线性坐标）: $\rho = 0.000392 \times e^{(0.000005039p^2 + 0.00289p + 7.343)}$', fontsize=18)
    plt.xlabel('p(MPa)', fontsize=18)
    plt.ylabel('ρ(mg/mm3)', fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


##################################################################################

def q1_1():
    """模拟在1s内的压强变化"""
    # 初始变量
    P_0 = 100  # 单位是MPa
    d = 10
    l = 500
    V_0 = pi * ((d / 2) ** 2) * l  # 油桶体积
    d_a = 1.4
    P_a = 160  # 单位MPa

    run_t_0 = 0
    run_t_max = 1000
    run_t_deta = 0.1

    t_deta_open = 0.2  # 阀门开启时长
    t_deta_open_max = 3  # 阀门最大开启时长
    t_deta_close = 10  # 阀门关闭时长10ms
    t_deta_flag = 0
    t_deta_open_add = 0.1  # 步进

    m_out = []  # 存储1s内每0.1ms内步长的出油量
    # list_p = [] # 存储某个时长的1s内的气压变化
    list_p_all = []  # 存储所有时长

    average_all = []

    ci = 0

    # 计算油输出
    for i in range(0, 10000, 10000):
        for ii in range(10):
            tmp_0 = 0
            for j in range(2):
                m_out.append(tmp_0)
                tmp_0 += 2 / 2
            for j in range(20):
                m_out.append(tmp_0)
            for j in range(2):
                m_out.append(tmp_0)
                tmp_0 -= 2 / 2
            for j in range(976):
                m_out.append(tmp_0)
    # print(m_out[0 : 10])
    # sys.exit(0)

    # 计算
    flag_out = False
    flag_out_time = 0
    range_ci = 0
    while not flag_out:
        range_ci += 1
        t_deta_flag = t_deta_open
        t_temp_open = True
        list_p = []  # 存储某个时长的1s内的气压变化
        average = 0
        while run_t_0 <= run_t_max:
            rho_1 = function_rho(P_0)
            Q = function_Q(pi * (d_a / 2) ** 2, abs(P_a - P_0), 1.094)
            if t_temp_open:
                m_in = Q / 10 * rho_1  # 每0.1ms加入的燃油质量
                t_deta_flag -= t_deta_open_add
                if t_deta_flag <= 0:
                    t_temp_open = False
                    t_deta_flag = t_deta_close
            else:
                m_in = 0
                t_deta_flag -= t_deta_open_add
                if t_deta_flag <= 0:
                    t_temp_open = True
                    t_deta_flag = t_deta_open
            m_1 = V_0 * rho_1
            m_out_tmp = m_out[ci] * rho_1
            m_2 = m_1 + m_in - m_out_tmp  # 剩下的质量
            # m_2 = m_1 + m_in # 剩下的质量
            rho_2 = m_2 / V_0
            # print(f"rho_2 is {rho_2}")
            P_2 = function_rho_pi(rho_2)
            # print(P_2)
            # 储存
            list_p.append(P_2)
            # 更新变量
            ci += 1
            run_t_0 += run_t_deta
            P_0 = P_2
            average += P_2  # 存储总和
        print(f"ci is {ci}")
        print(f"本次总和为{average}")

        # 重置/更新数值
        run_t_0 = 0
        average_all.append(average / ci)
        print(average / ci, "此时，开闸时长是:", t_deta_open, f"第{range_ci}次模拟")
        t_deta_open += t_deta_open_add
        P_0 = 100  # 单位是MPa
        ci = 0

        list_p_all.append(list_p)
        flag_out_time += 1
        print(f"flag_out_time is {flag_out}")
        if flag_out_time == 30:
            flag_out = True
    #
    # print(list_p_all)

def function_rho(p):
    """计算ρ=k*f(p)"""
    # k = 0.000392
    # E = np.exp(0.00000503 * p ** 2 + 0.00289 * p + 7.343)
    # rho = k * E
    # return rho
    return 0.000392 * np.exp(0.000005039 * p ** 2 + 0.00289 * p + 7.343)


def function_Q(A, deta_p, rho_1):
    """计算进出高压油管的流量Q, A是孔面"""
    part1 = 2 * deta_p / rho_1
    Q = 0.85 * A * sqrt(part1)
    return Q


def function_rho_pi(rho_rho):
    """计算ρ=k*f(p)的反解"""

    def rho(p):
        return 0.000392 * np.exp(0.000005039 * p ** 2 + 0.00289 * p + 7.343)

    def find_all_p_solutions(target_rho):
        """
        找到所有可能的p值解，并返回
        """
        # 计算函数的最小值点
        min_p = -0.00289 / (2 * 0.000005039)
        min_rho = rho(min_p)

        # print(f"函数最小值点: p = {min_p:.6f}, ρ_min = {min_rho:.6f}")

        if target_rho < min_rho:
            raise ValueError(f"目标ρ值 {target_rho} 小于最小值 {min_rho}，无实数解")

        # 定义方程
        def equation(p):
            return rho(p) - target_rho

        # 可能的两个解分别在最小值点左右两侧
        solutions = []

        # 尝试左侧的解（p < min_p）
        try:
            p_left = fsolve(equation, min_p - 1000, xtol=1e-12, full_output=True)
            if p_left[2] == 1:  # 成功收敛
                solutions.append(p_left[0][0])
        except:
            pass

        # 尝试右侧的解（p > min_p）
        try:
            p_right = fsolve(equation, min_p + 1000, xtol=1e-12, full_output=True)
            if p_right[2] == 1:  # 成功收敛
                solutions.append(p_right[0][0])
        except:
            pass

        return solutions

    def find_correct_p(target_rho, expected_p=None):
        """
        找到正确的p值解

        参数:
        target_rho: 目标ρ值
        expected_p: 期望的p值（如果知道的话）
        """
        solutions = find_all_p_solutions(target_rho)

        # print(f"\n找到 {len(solutions)} 个解:")
        for i, p_val in enumerate(solutions):
            # print(f"解{i + 1}: p = {p_val:.12f}, ρ验证 = {rho(p_val):.12f}")
            pass

        if expected_p is not None:
            # 选择最接近期望值的解
            closest_solution = min(solutions, key=lambda x: abs(x - expected_p))
            # print(f"\n选择最接近期望值 {expected_p} 的解: p = {closest_solution:.12f}")
            return closest_solution
        elif len(solutions) == 1:
            return solutions[0]
        else:
            # 默认返回较大的解（通常是物理上更有意义的解）
            return max(solutions)

    found_p = find_correct_p(rho_rho, expected_p=None)
    return found_p


##################################################################################

def q1_2():
    """模拟在1s内的压强变化"""
    # 初始变量
    P_0 = 100  # 单位是MPa
    # P_0 = 150  # 单位是MPa
    d = 10
    l = 500
    V_0 = pi * ((d / 2) ** 2) * l  # 油桶体积
    d_a = 1.4
    P_a = 160  # 单位MPa

    run_t_0 = 0
    run_t_max = 2000  # 最大单位运行时长
    run_t_deta = 0.1

    t_deta_open = 0.2  # 阀门开启时长
    t_deta_open_max = 1  # 阀门最大开启时长
    t_deta_close = 10  # 阀门关闭时长10ms
    t_deta_flag = 0
    t_deta_open_add = 0.1  # 步进

    m_out = []  # 存储1s内每0.1ms内步长的出油量
    list_p_all = []  # 存储所有时长

    average_all = []

    ci = 0

    # 计算油输出
    for i in range(0, 3, 1):
        for ii in range(10):
            tmp_0 = 0
            for j in range(2):
                m_out.append(tmp_0)
                tmp_0 += 2 / 2
            for j in range(20):
                m_out.append(tmp_0)
            for j in range(2):
                m_out.append(tmp_0)
                tmp_0 -= 2 / 2
            for j in range(976):
                m_out.append(tmp_0)
    print(len(m_out))
    # time.sleep(4)

    # 计算
    flag_out = False
    flag_out_time = 0
    range_ci = 0
    while not flag_out:
        range_ci += 1
        t_deta_flag = t_deta_open
        t_temp_open = True
        list_p = []  # 存储某个时长的1s内的气压变化
        average = 0
        while run_t_0 <= run_t_max:
            rho_1 = function_rho(P_0)
            Q = function_Q(pi * (d_a / 2) ** 2, abs(P_a - P_0), 1.094)
            if t_temp_open:
                m_in = Q / 10 * rho_1  # 每0.1ms加入的燃油质量
                t_deta_flag -= t_deta_open_add
                if t_deta_flag <= 0:
                    t_temp_open = False
                    t_deta_flag = t_deta_close
            else:
                m_in = 0
                t_deta_flag -= t_deta_open_add
                if t_deta_flag <= 0:
                    t_temp_open = True
                    t_deta_flag = t_deta_open
            m_1 = V_0 * rho_1
            try:
                m_out_tmp = m_out[ci] * rho_1
            except:
                print(f"ci is {ci}")
                time.sleep(10)
                sys.exit(0)
            m_2 = m_1 + m_in - m_out_tmp  # 剩下的质量
            # m_2 = m_1 + m_in # 剩下的质量
            rho_2 = m_2 / V_0
            # print(f"rho_2 is {rho_2}")
            P_2 = function_rho_pi(rho_2)
            # print(P_2)
            # 储存
            list_p.append(P_2)
            # 更新变量
            ci += 1
            run_t_0 += run_t_deta
            P_0 = P_2
            average += P_2  # 存储总和
            print(f"P_2为{P_2}")
            if P_2 >= 150:
                print(f"P_2为{P_2}")
                print(f"t_deta_open is {t_deta_open}")
                time.sleep(6)
        print(f"ci is {ci}")
        #  print(f"本次总和为{average}")

        # 重置/更新数值
        run_t_0 = 0
        average_all.append(average / ci)
        print(average / ci, "此时，开闸时长是:", t_deta_open, f"第{range_ci}次模拟")
        t_deta_open += t_deta_open_add
        P_0 = 100  # 单位是MPa
        ci = 0

        list_p_all.append(list_p)
        flag_out_time += 1
        print(f"flag_out_time is {flag_out}")
        if flag_out_time == 98:  # 98为最大10ms开启时长
            flag_out = True


if __name__ == "__main__":
    # Prepare_for_processing()
    q1_1()
    # q1_2()