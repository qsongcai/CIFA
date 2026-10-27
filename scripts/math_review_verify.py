"""
落稿前独立数学复核：定理1 / 定理2 / 假设2 修正方案的关键论断
所有结论用独立数值计算检验，不引用修订记录的结论。
"""
import numpy as np
from scipy.special import logsumexp

rng = np.random.default_rng(0)

def tv(p, q, axis=-1):
    """离散分布 TV = 0.5 * sum|p-q|。"""
    p = np.asarray(p, float); q = np.asarray(q, float)
    return 0.5 * np.abs(p - q).sum(axis=axis)

print("=" * 70)
print("【定理1 / T3】有限维参数 != 有限维仿射包：一维高斯平移族")
print("=" * 70)
# P_theta = N(theta,1)。任意有限个高斯密度在函数空间线性无关。
# 特征函数 phi_theta(t)=exp(-t^2/2) exp(i t theta)。
# 若 sum c_j p_{theta_j}=0，Fourier 得 sum c_j exp(i t theta_j)=0 对所有 t。
# 在 t 网格上采样得到 Vandermonde 型矩阵 V[k,j]=exp(i t_k theta_j)，检验满秩。
for n in [2, 3, 5, 8]:
    thetas = np.linspace(-2, 2, n)
    tgrid = np.linspace(0, 6, 400)
    V = np.exp(1j * np.outer(tgrid, thetas))      # 特征函数矩阵
    # 最小奇异值衡量线性无关程度（>0 即满秩）
    s = np.linalg.svd(V, compute_uv=False)
    # 密度网格上同样检验
    xgrid = np.linspace(-6, 6, 2000)
    G = np.stack([np.exp(-0.5*(xgrid-th)**2) for th in thetas], axis=1)
    sg = np.linalg.svd(G, compute_uv=False)
    print(f"  n={n}: 特征函数矩阵最小奇异值={s[-1]:.3e} (秩={np.linalg.matrix_rank(V)}), "
          f"密度矩阵最小奇异值={sg[-1]:.3e} -> {'线性无关' if s[-1] > 1e-8 and sg[-1] > 1e-8 else '相关!'}")
print("  结论：一维参数族的任意有限个测度线性无关，仿射包无限维。删 d+1 正确。")

print()
print("=" * 70)
print("【定理1 / 界】d_TV(P_te, U_A^(s)) <= eta + h_s  的三角推导（随机验证）")
print("=" * 70)
# 在概率单纯形上随机构造：全集 U_A（多面体）、有限凸包 U_A^(s)（s 个点）、P_te
def random_simplex(n, k):
    M = rng.random((k, n)); M = M / M.sum(1, keepdims=True)
    return M

def dist_to_hull(p, V):
    """p 到 conv(V) 的 TV 距离，精确 LP 求解：
    min 0.5*1'u  s.t. |p-wV|<=u, w>=0, 1'w=1。变量 x=[w(k),u(n)]。"""
    from scipy.optimize import linprog
    k, n = V.shape
    c = np.concatenate([np.zeros(k), 0.5*np.ones(n)])
    Aeq = np.concatenate([np.ones(k), np.zeros(n)])[None, :]
    beq = np.array([1.0])
    rows, b = [], []
    for i in range(n):
        r1 = np.zeros(k+n); r1[:k] = V[:, i]; r1[k+i] = -1.0   # (wV)_i - u_i <= p_i
        r2 = np.zeros(k+n); r2[:k] = -V[:, i]; r2[k+i] = -1.0  # -(wV)_i - u_i <= -p_i
        rows += [r1, r2]; b += [p[i], -p[i]]
    res = linprog(c, A_ub=np.array(rows), b_ub=np.array(b),
                  A_eq=Aeq, b_eq=beq, bounds=[(0, None)]*(k+n), method='highs')
    return res.fun

n_state = 5
fails = 0
for trial in range(200):
    full = random_simplex(n_state, 12)          # 完整族生成点（近似 U_A）
    s_idx = rng.choice(12, 4, replace=False)
    finite = full[s_idx]                          # U_A^(s)
    pte = rng.random(n_state); pte /= pte.sum()
    eta = dist_to_hull(pte, full)                 # 到全集距离
    # h_s = sup_{Q in U_A} d(Q, U_A^(s))：用 full 各点距离的最大（采样近似）
    h_s = max(dist_to_hull(q, finite) for q in full)
    d_finite = dist_to_hull(pte, finite)
    if d_finite > eta + h_s + 1e-6:
        fails += 1
print(f"  200 次随机：d_finite <= eta+h_s 违反次数 = {fails}  (应为 0)")
print("  结论：eta+h_s 三角不等式成立；删 d+1 后核心界不受影响。")

print()
print("=" * 70)
print("【定理2 / T6】积分 Taylor + kink 项；单点 Lagrange 失效")
print("=" * 70)

def piecewise_decompose(g, gp, kinks, npts=200001):
    """对 g(s),s∈[0,1]（分段 C2，kink 点处 gp 跳），用积分形式分解：
       g(1)-g(0) = gp(0) + int_0^1 (1-u) g''_reg(u) du + sum (1-s_i) J_i
    返回 (g(1)-g(0), gp(0), hessian_int, kink_sum)。"""
    u = np.linspace(0, 1, npts)
    du = u[1]-u[0]
    # 区域内二阶导（数值），在 kink 邻域置零避免污染
    gpv = gp(u)
    gpp = np.gradient(gpv, u)
    mask = np.ones_like(u, bool)
    for sk in kinks:
        mask &= np.abs(u-sk) > 3*du
    gpp = np.where(mask, gpp, 0.0)
    hess_int = np.trapz((1-u)*gpp, u)
    # kink 跳变 J = gp(si+) - gp(si-)
    kink_sum = 0.0
    for sk in kinks:
        jp = gp(sk + 1e-6); jm = gp(sk - 1e-6)
        kink_sum += (1-sk)*(jp-jm)
    return g(1)-g(0), gp(0), hess_int, kink_sum

print("-- 例1（审稿人反例 / 修订记录）：h(theta)=ReLU(theta)，θ:-1→+1 --")
h  = lambda th: np.maximum(th, 0)
hp = lambda th: (np.asarray(th,float) > 0).astype(float)
g  = lambda s: h(-1+2*s); gp = lambda s: hp(-1+2*s)*2
kink = [0.5]
diff, g0, hi, ks = piecewise_decompose(g, gp, kink)
lhs_lagrange = g0 + 0.0      # 区域内 Hessian=0 -> 单点 Lagrange 预测
print(f"  真实 g(1)-g(0) = {diff:.6f} (= ReLU(1)=1)")
print(f"  一阶项 gp(0)   = {g0:.6f}")
print(f"  Hessian 积分项 = {hi:.6f} (区域内 H=0)")
print(f"  kink 项 Σ(1-s)J = {ks:.6f}  (s*=0.5,J=2 -> 0.5*2=1)")
print(f"  积分式重建 = {g0+hi+ks:.6f}  {'✓ 吻合' if abs(g0+hi+ks-diff)<1e-3 else '✗'}")
print(f"  单点 Lagrange 预测 = {lhs_lagrange:.6f}  -> 漏掉 kink，误差 {diff-lhs_lagrange:.3f}（失效）")

print("-- 例2：二次段 + 一个 kink（验证 Hessian 项与 kink 项可同时成立）--")
# θ<0: h=0.3 θ^2（H=0.6）；θ>0: h=θ + 0.3 θ^2（导数 1+0.6θ）
def h2(th):
    return np.where(th<0, 0.3*th**2, th + 0.3*th**2)
def hp2(th):
    return np.where(th<0, 0.6*th, 1+0.6*th)
g  = lambda s: h2(-1+2*s); gp = lambda s: hp2(-1+2*s)*2
diff, g0, hi, ks = piecewise_decompose(g, gp, [0.5])
# 解析：s<0.5 段 γ=-1+2s, g''= (2)^2 *0.6=2.4；s>0.5 段 g''=2.4
# int(1-u)2.4 du 全段 = 2.4*1/2=1.2；kink: gp 跳 = 2（链式），(1-0.5)*2=1
print(f"  真实差={diff:.6f}  一阶={g0:.6f}  Hessian积分={hi:.6f}(期望1.2)  kink={ks:.6f}(期望1.0)")
print(f"  重建={g0+hi+ks:.6f}  {'✓' if abs(g0+hi+ks-diff)<2e-3 else '✗'}")

print("-- 例3：两个 kink 的分段线性函数（多 kink 求和）--")
# h(θ)=ReLU(θ)+ReLU(θ-0.5)，θ̄=-1→θ_t=1.5
def h3(th): return np.maximum(th,0)+np.maximum(th-0.5,0)
def hp3(th):
    th=np.asarray(th,float)
    return (th>0).astype(float)+(th>0.5).astype(float)
g  = lambda s: h3(-1+2.5*s); gp=lambda s: hp3(-1+2.5*s)*2.5
# kink 位置：-1+2.5s=0 -> .4；=0.5 -> .6
diff, g0, hi, ks = piecewise_decompose(g, gp, [0.4,0.6])
# J1=2.5*(1-0.4)=1.5; J2=2.5*(1-0.6)=1.0 -> 2.5
print(f"  真实差={diff:.6f}  kink项={ks:.6f}(期望2.5)  重建={g0+hi+ks:.6f} "
      f"{'✓' if abs(g0+hi+ks-diff)<2e-3 else '✗'}")

print("-- 例4（SWAD 重心几何）：等权 θ1=-1,θ2=+1，θ̄=0，h=ReLU(θ) --")
# θ̄=0 恰在 kink；臂 θ2=+1 段 g(s)=ReLU(s)，臂 θ1=-1 段全 0
# 集成-平均差 = 0.5*h(1)+0.5*h(-1)-h(0) = 0.5
hens = 0.5*h(1)+0.5*h(-1); hbar = h(0)
print(f"  h_ens-h̄ = {hens-hbar:.3f}（=0.5）；边界 kink 落于重心，是真实几何退化点，落稿需说明")

print()
print("=" * 70)
print("【假设2 / T7】τ_Y ≤ τ_C（TV 凸性/Jensen），且 τ_Y 小不保证 τ_C 小")
print("=" * 70)
# 离散模型：C 细类 -> Y 粗类(Y=m(C))；d 依赖 C（内容泄漏）。
# 构造：同 y 内两个 c 的 d 偏向相反 -> E[d|Y]=0 但 E[d|C]≠0。
# d ∈ {0,1,2}（离散风格码）
dvals = np.array([0,1,2])
def model(pc, p_d_given_c, c2y):
    nC = len(pc); nY = len(set(c2y))
    pd = sum(pc[c]*p_d_given_c[c] for c in range(nC))
    # P(d|Y=y)
    pd_y = np.zeros((nY,3))
    for y in range(nY):
        cy = [c for c in range(nC) if c2y[c]==y]
        w = np.array([pc[c] for c in cy]); w=w/w.sum()
        pd_y[y] = sum(w[k]*p_d_given_c[c] for k,c in enumerate(cy))
    tauC = max(tv(p_d_given_c[c], pd) for c in range(nC))
    tauY = max(tv(pd_y[y], pd) for y in range(nY))
    # 校验类条件零均值 E[d|Y]=0（用中心化 d 表示风格偏差；这里报告均值差）
    mean_given_y = [(pd_y[y]*dvals).sum() for y in range(nY)]
    return pd, tauC, tauY, mean_given_y

# 4 个细类，2 个粗类 y1={0,1}, y2={2,3}，等权
pc = np.ones(4)/4; c2y=[0,0,1,1]
# 同 y 内 c 偏向相反：c0 偏 d=0、c1 偏 d=2；c2 偏 d=0、c3 偏 d=2
eps=0.35
pdc=[
    np.array([0.5+eps,0.5-2*eps,0.0+eps]),
    np.array([0.5-eps,0.5-2*eps,0.0+eps+0.0]),
]
# 重新手工设（确保归一）
pdc = [
    np.array([0.85,0.10,0.05]),   # c0 偏向 d=0
    np.array([0.05,0.10,0.85]),   # c1 偏向 d=2（与 c0 相反，同 y1）
    np.array([0.85,0.10,0.05]),   # c2
    np.array([0.05,0.10,0.85]),   # c3（同 y2）
]
pdc=[p/p.sum() for p in pdc]
pd, tauC, tauY, my = model(pc,pdc,c2y)
print(f"  类泄漏模型：E[d|Y=y] 的均值（中心化后应为0，此处 d 非对称，仅展示）")
print(f"  τ_C = {tauC:.4f}   τ_Y = {tauY:.4f}   τ_Y≤τ_C : {tauY<=tauC+1e-9}")
# 显式做类中心化：把 d 替换为 d - E[d|Y]，构造风格偏差，再看条件分布仍可分
# 极端例：Y 与“决定 d 的 C 成分”独立（标签不携带泄漏信息）
# 令 Y 为独立粗化，d 的偏向依赖一个 Y 完全无法区分的 C 细分，且在 y 内对称抵消
pdc2=[
    np.array([0.9,0.05,0.05]),
    np.array([0.05,0.05,0.9]),
    np.array([0.9,0.05,0.05]),
    np.array([0.05,0.05,0.9]),
]
pdc2=[p/p.sum() for p in pdc2]
# 令 Y 与 c 配对方式：y 内总是同时含一个“偏0”和一个“偏2”的 c，且 P(d|Y) 完全对称
pc2=np.ones(4)/4
pd2, tauC2, tauY2, _ = model(pc2,pdc2,c2y)
print(f"  极端例：τ_C = {tauC2:.4f}   τ_Y = {tauY2:.4f}   (τ_Y=0 而 τ_C>0: {tauY2<1e-9 and tauC2>1e-9})")
print("  结论：τ_Y≤τ_C 成立；τ_Y 是可计算下界（监控量），τ_Y=0 不蕴含 τ_C=0。")

# 随机大规模验证 τ_Y≤τ_C（上百个随机联合分布）
viol=0
for _ in range(3000):
    nC=rng.integers(2,7); nY=rng.integers(2,nC+1)
    c2y_r=rng.integers(0,nY,nC)
    if len(set(c2y_r))<nY: continue
    pcr=rng.random(nC); pcr/=pcr.sum()
    PdC=rng.random((nC,3)); PdC/=PdC.sum(1,keepdims=True)
    Pdr=pcr@PdC
    Pdy=np.zeros((nY,3))
    for y in range(nY):
        idx=np.where(c2y_r==y)[0]; w=pcr[idx]; w=w/w.sum()
        Pdy[y]=w@PdC[idx]
    tC=max(tv(PdC[c],Pdr) for c in range(nC))
    tY=max(tv(Pdy[y],Pdr) for y in range(nY))
    if tY>tC+1e-9: viol+=1
print(f"  3000 个随机联合分布：τ_Y>τ_C 违反次数 = {viol}（应为 0）")

print()
print("="*70)
print("【命题1 已落稿，顺手复核 reverse-Pinsker】D_KL(P||Q) <= TV/q_min")
print("=" * 70)
def kl(p,q,axis=-1):
    p=np.asarray(p,float);q=np.asarray(q,float)
    with np.errstate(divide='ignore',invalid='ignore'):
        return np.sum(np.where(p>0,p*np.log(np.where(p>0,p/np.where(q>0,q,1),1)),0),axis=axis)
# 审稿人精确反例（K=2，Q 满支撑 q_min=0.5）：旧界（带 log）违反，新界（TV/qmin）成立
P=np.array([1.,0.]); Q=np.array([.5,.5])
print(f"  反例 P=(1,0),Q=(1/2,1/2): D_KL={kl(P,Q):.4f}")
print(f"    旧界 TV*log(1/qmin)={tv(P,Q)*np.log(1/Q.min()):.4f} -> "
      f"{'★错误(违反)' if kl(P,Q)>tv(P,Q)*np.log(1/Q.min())+1e-12 else '成立'}")
print(f"    新界 TV/qmin     ={tv(P,Q)/Q.min():.4f} -> "
      f"{'错误' if kl(P,Q)>tv(P,Q)/Q.min()+1e-12 else '成立'}")
# 低维大规模：带 log 旧界在低维高比例失败；TV/qmin 0 失败（高维随机会掩盖旧界错误）
for K in [2,3,4,8]:
    fails_old=fails_new=0; N=40000
    for _ in range(N):
        p=rng.random(K); p/=p.sum()
        q=rng.random(K)+1e-3; q/=q.sum()          # 满支撑
        T=tv(p,q); qmin=q.min()
        if kl(p,q)>T*np.log(1/qmin)+1e-9: fails_old+=1
        if kl(p,q)>T/qmin+1e-9: fails_new+=1
    print(f"  K={K}: 旧界[TV log(1/qmin)]失败 {fails_old:6d}/{N}   新界[TV/qmin]失败 {fails_new}/{N}")
