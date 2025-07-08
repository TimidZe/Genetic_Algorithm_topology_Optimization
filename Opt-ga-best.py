import numpy as np
import random
from multiprocessing import Pool
import pandas as pd

from numba import njit
from scipy.sparse import lil_matrix
# import pandas as pd
from collections import deque
from itertools import product
import numpy as np
import random

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('Results_ga_github_2/optimization_output.txt', mode='w'),
        logging.StreamHandler()  # 同时打印到控制台
    ]
)


# —— 先插入你之前的温度场求解器 —— #
# from your_solver import calculate_temperature_matrix_helper

@njit
def compute_directional_averages(matrix):
    rows, cols = matrix.shape
    averages = np.zeros((rows, cols, 4), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            # 左
            averages[i, j, 0] = matrix[i, j] / 2 if i == 0 else (matrix[i, j] + matrix[i-1, j]) / 2
            # 右
            averages[i, j, 1] = matrix[i, j] / 2 if i == rows - 1 else (matrix[i, j] + matrix[i+1, j]) / 2
            # 下
            averages[i, j, 2] = matrix[i, j] / 2 if j == 0 else (matrix[i, j] + matrix[i, j-1]) / 2
            # 上
            averages[i, j, 3] = matrix[i, j] / 2 if j == cols - 1 else (matrix[i, j] + matrix[i, j+1]) / 2
    return averages

# === 索引变换函数 ===
@njit
def indexer(i, j, mesh_x):
    return i * mesh_x + j

# === 温度场求解函数 ===
def calculate_temperature_matrix_helper(
    L=1.0, a=0.1, q=100, T0=300, k0=1, k1=500,
    mesh_x=51, mesh_y=51, given_distribution=None, epsilon=1.0e-5
):
    """
    求解温度场，返回维度为(mesh_x, mesh_y)的温度分布（已做标准化）。
    given_distribution为热导率矩阵。
    """
    if given_distribution is None:
        raise ValueError("given_distribution 必须给定！")
    A = lil_matrix((mesh_x * mesh_y, mesh_x * mesh_y))
    b = np.zeros((mesh_x * mesh_y)).reshape(-1, 1)
    T = np.full((mesh_x * mesh_y, 1), T0)

    delta_x = L / mesh_x
    delta_y = L / mesh_y
    delta_x_squre = delta_x**2
    delta_y_squre = delta_y**2
    start_index = int((L - a) / 2 / delta_x)
    end_index = int((L + a) / 2 / delta_x)

    k_averages = compute_directional_averages(given_distribution)
    for i in range(mesh_x):
        for j in range(mesh_y):
            idx = indexer(i, j, mesh_x)
            if i == 0:
                A[idx, indexer(i + 1, j, mesh_x)] = 1
            if i == mesh_x - 1:
                A[idx, indexer(i - 1, j, mesh_x)] = 1
            if i > 0 and i < mesh_x - 1 and j == 0:
                A[idx, indexer(i, j + 1, mesh_x)] = 1
            if i > 0 and i < mesh_x - 1 and j == mesh_y - 1:
                A[idx, indexer(i, j - 1, mesh_x)] = 1
            if i > 0 and i < mesh_x - 1 and j > 0 and j < mesh_y - 1:
                S = (
                    (k_averages[i, j, 1] + k_averages[i, j, 0]) / delta_x_squre +
                    (k_averages[i, j, 2] + k_averages[i, j, 3]) / delta_y_squre
                )
                A[idx, indexer(i - 1, j, mesh_x)] = k_averages[i, j, 0] / S / delta_x_squre
                A[idx, indexer(i + 1, j, mesh_x)] = k_averages[i, j, 1] / S / delta_x_squre
                A[idx, indexer(i, j - 1, mesh_x)] = k_averages[i, j, 2] / S / delta_y_squre
                A[idx, indexer(i, j + 1, mesh_x)] = k_averages[i, j, 3] / S / delta_y_squre
                b[idx, 0] = q / S

    A = A.tocsr()
    converged = False
    while not converged:
        T_old = T.copy()
        T = A @ T_old + b
        for i in range(start_index, end_index + 1):
            T[indexer(mesh_x - 1, i, mesh_x)] = T0
        max_abs_error = np.max(np.abs(T - T_old))
        if max_abs_error < epsilon:
            converged = True

    # 返回标准化后二维温度场
    result = k0 * (T.reshape(mesh_x, mesh_y)[::-1] - T0) / (q * L**2)
    return result

# Problem settings
nx, ny = 51, 51
N = nx * ny
n_high = 390         # 高导体积分数
pop_size = 100       # 种群规模
max_gen  = 10000       # 最大代数
elite_num = 5        # 精英保留数量

# GA hyper‑parameters
pc = 0.7   # 交叉概率
pm = 0.2   # 变异概率

SAVE_INTERVAL = 50

def repair_connectivity(mask):
    """
    保证 mask（长度 N=nx*ny 的 0/1 向量）对应的高导点恰好 n_high 个且 8-连通。
    步骤：
      1) 在原 mask 中寻找所有 1 的最大 8-连通分量 best_comp；
      2) 丢弃其他 1，只保留 best_comp；
      3) 以 best_comp 为种子，向周围零点“生长”填充，直到补齐到 n_high 个。
    返回修复后的 0/1 向量。
    """
    # 假定全局已定义：nx, ny, n_high
    # mask: 1D numpy array, length nx*ny
    mask_2d = mask.reshape((nx, ny))

    # 1) 找最大连通分量
    visited = np.zeros((nx, ny), dtype=bool)
    best_comp = set()
    for i in range(nx):
        for j in range(ny):
            if mask_2d[i, j] == 1 and not visited[i, j]:
                comp = set()
                q = deque([(i, j)])
                visited[i, j] = True
                comp.add((i, j))
                while q:
                    ci, cj = q.popleft()
                    for di, dj in product([-1, 0, 1], repeat=2):
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = ci + di, cj + dj
                        if 0 <= ni < nx and 0 <= nj < ny \
                           and not visited[ni, nj] \
                           and mask_2d[ni, nj] == 1:
                            visited[ni, nj] = True
                            comp.add((ni, nj))
                            q.append((ni, nj))
                if len(comp) > len(best_comp):
                    best_comp = comp

    # 2) 用 best_comp 初始化新 mask_comp
    comp = set(best_comp)
    mask_comp = np.zeros((nx, ny), dtype=int)
    for (ci, cj) in comp:
        mask_comp[ci, cj] = 1

    # 3) 逐步向周围“生长”直到有 n_high 个点
    while len(comp) < n_high:
        # 收集所有与 comp 相邻的零点，并统计每个候选点邻接 comp 的次数
        cand = {}
        for (ci, cj) in comp:
            for di, dj in product([-1, 0, 1], repeat=2):
                if di == 0 and dj == 0:
                    continue
                ni, nj = ci + di, cj + dj
                if 0 <= ni < nx and 0 <= nj < ny and mask_comp[ni, nj] == 0:
                    idx = ni * ny + nj
                    cand[idx] = cand.get(idx, 0) + 1

        if not cand:
            # 极端情况：四周没零点，就挑整个区域最近的零点
            zeros = [(i, j) for i in range(nx) for j in range(ny) if mask_comp[i, j] == 0]
            comp_list = list(comp)
            best_zero = min(
                zeros,
                key=lambda z: min((z[0] - ci)**2 + (z[1] - cj)**2 for ci, cj in comp_list)
            )
            comp.add(best_zero)
            mask_comp[best_zero[0], best_zero[1]] = 1
            continue

        # 从候选点中选一个最少与 comp 邻接的，保证“枝状”生长
        min_count = min(cand.values())
        min_cands = [idx for idx, cnt in cand.items() if cnt == min_count]
        chosen = random.choice(min_cands)
        ci, cj = divmod(chosen, ny)
        comp.add((ci, cj))
        mask_comp[ci, cj] = 1

    # 4) 构造并返回修复后的 1D mask
    repaired_mask = np.zeros_like(mask)
    for (ci, cj) in comp:
        repaired_mask[ci * ny + cj] = 1

    return repaired_mask

# —— 个体解码为 k 分布，并评估适应度 —— #
def fitness(mask):
    # 修复连通，保证恰好 n_high 个 1 并连通
    m = repair_connectivity(mask)
    k_dist = m.reshape((nx,ny)) * 500 + (1-m.reshape((nx,ny))) * 1
    T = calculate_temperature_matrix_helper(
        L=1.0, a=0.1, q=100, T0=300,
        k0=1, k1=500, mesh_x=nx, mesh_y=ny,
        given_distribution=k_dist
    )
    return np.mean(T)

# —— 初始化种群 —— #
def init_population():
    pop = []
    for _ in range(pop_size):
        mask = np.zeros(N, dtype=int)
        ones = np.random.choice(N, n_high, replace=False)
        mask[ones] = 1
        pop.append(mask)
    return pop

# —— 选择：锦标赛 —— #
def tournament_selection(pop, fits, k=3):
    idx = random.sample(range(pop_size), k)
    winner = min(idx, key=lambda i: fits[i])
    return pop[winner].copy()

# —— 交叉：二维块交叉 —— #
def crossover(p1, p2):
    if random.random() > pc:
        return p1.copy(), p2.copy()
    # 随机选若干行/列作为“块”进行交换
    child1, child2 = p1.copy(), p2.copy()
    # 例如：随机选 5 行，整行交换
    rows = random.sample(range(nx), k=random.randint(1,5))
    for r in rows:
        idxs = slice(r*ny, (r+1)*ny)
        child1[idxs], child2[idxs] = child2[idxs].copy(), child1[idxs].copy()
    # 修正体积分数
    for child in (child1, child2):
        diff = child.sum() - n_high
        if diff>0:
            # 多余 1，随机置 0
            ones = np.where(child==1)[0]
            to_zero = np.random.choice(ones, diff, replace=False)
            child[to_zero] = 0
        elif diff<0:
            # 不足 1，随机置 1
            zeros = np.where(child==0)[0]
            to_one = np.random.choice(zeros, -diff, replace=False)
            child[to_one] = 1
    return child1, child2

# —— 变异：swap 并修复连通 —— #
def mutate(mask):
    if random.random()>pm:
        return mask
    m = mask.copy()
    # 随机选 2 对点 swap
    for _ in range(3):
        i,j = np.random.choice(N,2,replace=False)
        if m[i]!=m[j]:
            m[i], m[j] = m[j], m[i]
    # 修复连通
    return repair_connectivity(m)

# —— 主优化流程 —— #
def ga_optimize():
    pop = init_population()
    # 并行池
    pool = Pool()
    fits = pool.map(fitness, pop)
    best_idx = int(np.argmin(fits))
    best_mask = pop[best_idx].copy()
    best_fit  = fits[best_idx]

    # 评估初代
    fits = pool.map(fitness, pop)
    best_idx = int(np.argmin(fits))
    best_mask = pop[best_idx].copy()
    best_fit  = fits[best_idx]

    for gen in range(1, max_gen+1):
        new_pop = []
        # 精英保留
        elites = [pop[i] for i in np.argsort(fits)[:elite_num]]
        new_pop.extend(elites)

        # 生成剩余个体
        while len(new_pop) < pop_size:
            p1 = tournament_selection(pop, fits)
            p2 = tournament_selection(pop, fits)
            c1, c2 = crossover(p1, p2)
            new_pop.append(mutate(c1))
            if len(new_pop)<pop_size:
                new_pop.append(mutate(c2))

        pop = new_pop
        fits = pool.map(fitness, pop)

        # 更新全局最优
        cur_best = np.min(fits)
        if cur_best < best_fit:
            best_fit = cur_best
            best_mask = pop[int(np.argmin(fits))].copy()

        # print(f'Gen {gen}: best_avgT = {best_fit:.6f}')
        logging.info(f'Gen {gen}: best_avgT = {best_fit:.6f}')
        # —— 每隔 SAVE_INTERVAL 代保存当前最优 k 分布 —— #
        if gen % SAVE_INTERVAL == 0:
            k_current = best_mask.reshape((nx,ny)) * 500 + (1 - best_mask.reshape((nx,ny))) * 1
            pd.DataFrame(k_current).to_excel(
                f'Results_ga_github_2/k_ga_best_gen{gen}.xlsx',
                header=False, index=False
            )
            print(f'  ↳ saved k distribution at generation {gen}')

    # —— 结束循环后也别忘了保存一次最终结果 —— #
    k_final = best_mask.reshape((nx,ny)) * 500 + (1 - best_mask.reshape((nx,ny))) * 1
    pd.DataFrame(k_final).to_excel('Results_ga_github_2/k_ga_result.xlsx', header=False, index=False)
    print('GA Completed. Best avg T:', best_fit)
    pool.close()
    pool.join()

    # # 保存结果
    # k_final = best_mask.reshape((nx,ny))*500 + (1-best_mask.reshape((nx,ny)))*1
    # pd.DataFrame(k_final).to_excel('k_ga_result.xlsx',header=False,index=False)
    # print('GA Completed. Best avg T:', best_fit)

if __name__=='__main__':
    ga_optimize()
