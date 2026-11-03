#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from ase.io import read
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import os
from matplotlib.ticker import MultipleLocator, NullLocator

# ============================
# 1. 从 POSCAR 计算倒格矢
# ============================

def get_reciprocal_lattice_from_poscar(poscar_file="POSCAR", use_2pi=True):
    """
    从 POSCAR 读取实空间晶格向量 a1,a2,a3，计算倒格矢 b1,b2,b3。
    """
    mol = read(poscar_file, format="vasp")
    cell = mol.cell  # 3x3 matrix, rows are a1,a2,a3
    a1 = np.array(cell[0], dtype=float)
    a2 = np.array(cell[1], dtype=float)
    a3 = np.array(cell[2], dtype=float)

    volume = np.dot(a1, np.cross(a2, a3))

    factor = 2.0 * np.pi if use_2pi else 1.0
    b1 = factor * np.cross(a2, a3) / volume
    b2 = factor * np.cross(a3, a1) / volume
    b3 = factor * np.cross(a1, a2) / volume

    return b1, b2, b3


# ============================
# 2. 用 Voronoi 得到 BZ 顶点
# ============================

def compute_bz_vertices(b1, b2, b3, nmax=2, round_decimals=8):
    """
    用 Voronoi + Wigner–Seitz 方法计算第一布里渊区的顶点坐标。

    返回:
      verts: (N_vertices, 3) 的 numpy 数组，每一行是一个 BZ 顶点 (kx, ky, kz)
    """
    b1 = np.array(b1, dtype=float)
    b2 = np.array(b2, dtype=float)
    b3 = np.array(b3, dtype=float)

    pts = []
    origin_index = None

    # 生成一块倒格子点: G = n1*b1 + n2*b2 + n3*b3
    for n1 in range(-nmax, nmax + 1):
        for n2 in range(-nmax, nmax + 1):
            for n3 in range(-nmax, nmax + 1):
                g = n1 * b1 + n2 * b2 + n3 * b3
                pts.append(g)
                if n1 == 0 and n2 == 0 and n3 == 0:
                    origin_index = len(pts) - 1

    pts = np.array(pts)
    if origin_index is None:
        raise RuntimeError("没有找到 Γ 点对应的倒格子点 (0,0,0)。")

    # Voronoi 分解
    vor = Voronoi(pts)

    # 对应 Γ 点的 Voronoi 区域 → 第一布里渊区
    region_index = vor.point_region[origin_index]
    region = vor.regions[region_index]

    if -1 in region:
        raise RuntimeError(
            "Voronoi 区域包含 -1 顶点（无穷远），nmax 可能太小或晶格太扭曲，"
            "请尝试增大 nmax。"
        )

    bz_vertices = vor.vertices[region]

    # 数值去重 + 排序
    bz_rounded = np.round(bz_vertices, round_decimals)
    unique_vertices = np.unique(bz_rounded, axis=0)

    # 按离 Γ 的距离排序（为了输出好看）
    radii = np.linalg.norm(unique_vertices, axis=1)
    order = np.argsort(radii)
    unique_vertices = unique_vertices[order]

    return unique_vertices


# ============================
# 3. 手动连线用的函数
# ============================

def draw_edge(ax, verts, i, j, **kwargs):
    """
    在 3D 图上连一条边：从顶点 Vi 到 Vj。

    参数:
      ax    : matplotlib 3D 坐标轴
      verts : (N,3) 顶点数组，顺序对应 V1, V2, ...
      i, j  : 顶点序号（1-based，和打印出的 V1,V2... 对应）
    """
    p1 = verts[i - 1]
    p2 = verts[j - 1]
    xs = [p1[0], p2[0]]
    ys = [p1[1], p2[1]]
    zs = [p1[2], p2[2]]
    #ax.plot(xs, ys, zs, **kwargs)
    ax.plot(xs, ys, zs, **kwargs,zorder=1)

def draw_edges(ax, verts, edge_list, **kwargs):
    """
    批量连线。
    edge_list 是 [(i1,j1), (i2,j2), ...] 这样的列表，i/j 都是 1-based 序号。
    """
    for (i, j) in edge_list:
        draw_edge(ax, verts, i, j, **kwargs)


# ============================
# 4. 读 Nodes.dat → Weyl 点坐标
# ============================

def load_weyl_points(nodes_file, b1, b2, b3):
    """
    从 Nodes.dat 中读取 Weyl 点，并转换到倒空间直角坐标。

    假定 Nodes.dat 每行格式类似：
    kx ky kz gap E k1 k2 k3
    这里使用 k1,k2,k3 作为 (b1,b2,b3) 基底下的 fractional 坐标。

    返回:
      weyl_cart : (N,3) Weyl 点在倒空间的直角坐标
      weyl_E    : (N,) 对应能量
      k_frac    : (N,3) fractional 坐标 (k1,k2,k3)
    """
    data = np.loadtxt(nodes_file, comments="#")

    # fractional coordinates in reciprocal basis (b1,b2,b3)
    k_frac = data[:, 5:8]   # (k1,k2,k3)
    weyl_E = data[:, 4]     # energy

    b1 = np.array(b1, dtype=float)
    b2 = np.array(b2, dtype=float)
    b3 = np.array(b3, dtype=float)
    B = np.vstack([b1, b2, b3])  # 3x3

    # cart = k1*b1 + k2*b2 + k3*b3
    weyl_cart = k_frac @ B

    # ===== 只保留 kz > 0 的 Weyl 点 =====
    mask = weyl_cart[:, 2] > 0.0
    weyl_cart = weyl_cart[mask]
    weyl_E    = weyl_E[mask]
    k_frac    = k_frac[mask]
    return weyl_cart, weyl_E, k_frac

def load_weyl_points_with_chiralities(nodes_file, b1, b2, b3):
    data = np.loadtxt(nodes_file, comments="#")

    # fractional coordinates in reciprocal basis (b1,b2,b3)
    k_frac = data[:, :3]   # (k1,k2,k3)
    weyl_chirality = data[:, 6]     # energy

    b1 = np.array(b1, dtype=float)
    b2 = np.array(b2, dtype=float)
    b3 = np.array(b3, dtype=float)
    B = np.vstack([b1, b2, b3])  # 3x3

    # cart = k1*b1 + k2*b2 + k3*b3
    weyl_cart = k_frac @ B

    # ===== 只保留 kz > 0 的 Weyl 点 =====
    mask = weyl_cart[:, 2] > 0.0
    weyl_cart = weyl_cart[mask]
    weyl_chirality    = weyl_chirality[mask]
    k_frac    = k_frac[mask]
    return weyl_cart, weyl_chirality, k_frac




# ============================
# 5. 画点 + 标签 + Weyl 点
# ============================

def plot_bz_vertices_and_weyl(verts, weyl_cart=None, weyl_labels=None):
    """
    画 BZ 顶点 (黑色实心点 + V_i 标签)，并可选叠加 Weyl 点 (红点 + W_i 标签)。

    weyl_cart  : (M,3) Weyl 直角坐标
    weyl_labels: 长度为 M 的字符串列表，如 ["W1","W2",...]
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    # 关键：关闭透视，改用正交投影
    ax.set_proj_type('ortho')
    #ax.set_box_aspect((1, 1, 1))
    # 关掉网格
    ax.grid(False)
    # ---- BZ 顶点：黑色实心点 + 标签 V1,V2,...
    for i, v in enumerate(verts, start=1):
        ax.scatter(
            v[0], v[1], v[2],
            c="k",
            #c='g', 
            s=0,  # 黑色实心点
            edgecolor='none',
            zorder=2,
        )
        ax.text(
            v[0]*1.05, 
            v[1]*1.05, 
            v[2]*1.05,
            #v[0],v[1],v[2],
            f"V{i}",
            #f"",
            fontsize=8,
            ha="center", va="center",
            zorder=3,
        )

    # ---- Weyl 点：如果有的话，用红点 + W_i 标签
    if weyl_cart is not None and len(weyl_cart) > 0:
        #weyl_chi = np.asarray(weyl_chirality)
        #colors = np.where(weyl_chi > 0, "r", "b")
        colors = ['r' if lab == '+' else 'b' for lab in weyl_labels]
        ax.scatter(
            weyl_cart[:, 0], 
            weyl_cart[:, 1], 
            weyl_cart[:, 2],
            #c="r",
            c=colors, 
            s=35,
            edgecolors="k", linewidths=1.0,
            zorder=3,
            depthshade=False,   # 关键：不要让 3D 深度再去管它的明暗和排序
        )

        # 加标签 W1, W2, ...
        if weyl_labels is None:
            weyl_labels = [f"W{i+1}" for i in range(len(weyl_cart))]

        for (x, y, z), lab in zip(weyl_cart, weyl_labels):
            ax.text(
                x, 
                y, 
                z,
                #lab,
                '',
                fontsize=20,
                #ha="left", 
                ha="center",
                va="center",
                #va="bottom",
                zorder=5,
            )

    # 坐标轴范围：含 BZ 顶点 + Weyl 点
    all_pts = verts
    if weyl_cart is not None and len(weyl_cart) > 0:
        all_pts = np.vstack([verts, weyl_cart])

    max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2.0
    mid = all_pts.mean(axis=0)
    #ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    #ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    #ax.set_zlim(mid[2] - max_range, mid[2] + max_range)


    ax.set_xlabel(r"$k_x \,(2\pi/a)$",labelpad=12)
    ax.set_ylabel(r"$k_y \,(2\pi/b)$",labelpad=12)
    #ax.set_zlabel(r"$k_z \,(2\pi/c)$")

    # 在设置完 all_pts / max_range 之后，改成固定范围
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.set_zlim(-0.6, 0.6)  # 俯视图时 z 其实无所谓

    # 自己设刻度（可选）
    #ax.set_xticks(np.linspace(-0.5, 0.5, 5))  # -0.8, -0.4, 0, 0.4, 0.8
    ax.set_xticks([-0.5,-0.25,0,0.25,0.5])
    #ax.set_yticks(np.linspace(-0.5, 0.5, 5))
    ax.set_yticks([-0.5,-0.25,0,0.25,0.5])
    # 如果是 top view，还可以不设 z 的 ticks，或者设少一点

    # 3) 关掉 minor ticks，避免多出来一排“小刻度”
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.zaxis.set_minor_locator(NullLocator())

    # 往外挪一点刻度数字（单位大概是像素）
    ax.tick_params(axis='x', which='major', pad=8)
    ax.tick_params(axis='y', which='major', pad=8)
    ax.tick_params(axis='z', which='major', pad=8)

    # 去掉 pane 背景和刻度
    ax.xaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.yaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.zaxis.pane.set_facecolor((1, 1, 1, 0))
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_edgecolor((1, 1, 1, 0))
    #ax.set_xticks([])
    #ax.set_yticks([])
    ax.set_zticks([])

    # --- 初始视角改成“俯视图”：从 +k_z 方向往下看 ---
    #ax.view_init(elev=90, azim=0)
    ax.view_init(elev=90, azim=270)
    #ax.view_init(elev=20, azim=35)

    return fig, ax


# ============================
# 6. 主程序
# ============================

def main():
    poscar_file = "POSCAR"
    #nodes_file = "Nodes.dat"
    nodes_file = "filtered_Nodes_with_Chiralities.dat"
    # 1) 倒格矢
    b1, b2, b3 = get_reciprocal_lattice_from_poscar(poscar_file, use_2pi=True)
    print("Reciprocal lattice vectors (with 2π):")
    print("b1 =", b1)
    print("b2 =", b2)
    print("b3 =", b3)
    print()

    # 2) 计算 BZ 顶点
    verts = compute_bz_vertices(b1, b2, b3, nmax=2, round_decimals=8)

    print(f"Number of BZ vertices: {len(verts)}")
    print("BZ corner coordinates (kx, ky, kz):")
    for i, v in enumerate(verts, start=1):
        print(f"  V{i:2d}:  {v[0]: .8f}  {v[1]: .8f}  {v[2]: .8f}")

    # 3) 读取 Weyl 点（如果有 Nodes.dat）
    weyl_cart = None
    weyl_E = None
    k_frac = None
    weyl_labels = None

    if os.path.exists(nodes_file):
        #weyl_cart, weyl_E, k_frac = load_weyl_points(nodes_file, b1, b2, b3)
        weyl_cart, weyl_chirality, k_frac = load_weyl_points_with_chiralities(nodes_file, b1, b2, b3)
        weyl_sign = np.where(weyl_chirality > 0, '+', '-')
        print()
        print(f"Loaded {len(weyl_cart)} Weyl points from {nodes_file}")

        # ---- ✅ 打印 W1...WX 的信息（想关掉就整块注释掉） ----
        #print("\nWeyl points (fractional k1,k2,k3 and energy):")
        #for i, (kf, E) in enumerate(zip(k_frac, weyl_E), start=1):
        #    print(
        #        f"  W{i:2d}:  "
        #        f"k_frac = ({kf[0]: .6f}, {kf[1]: .6f}, {kf[2]: .6f}),  "
        #        f"E = {E: .6f} eV"
        #    )
        # ---------------------------------------------------

        # 生成标签 W1, W2, ...
        weyl_labels = [f"W{i+1}" for i in range(len(weyl_cart))]

    # 4) 画 BZ 顶点 + Weyl 点（带 W 标签）
    fig, ax = plot_bz_vertices_and_weyl(
        verts,
        weyl_cart=weyl_cart,
        weyl_labels=weyl_sign,
        #weyl_labels='',
    )

    # 5) ✅ 这里你可以自己手动连线
    # 例如：
    # edges = [(1, 2), (2, 3), (3, 1)]
    # draw_edges(ax, verts, edges, color="k", linewidth=1.0)
    #
    # 或者只连 V1–V4:
    # draw_edge(ax, verts, 1, 4, color="k", linewidth=1.2)

    edges = [
            (24,10),(10,19),(19,11),(11,24),
            (24,1),(10,4),(19,8),(11,6),
            (1,13),(1,14),(6,21),(6,17),(8,15),(8,16),(4,22),(4,18),
            (13,18),(14,21),(17,16),(15,22),
            (13,7),(14,7),(21,3),(17,3),(16,2),(15,2),(22,5),(18,5),
            (5,9),(7,20),(3,12),(2,23),
            (12,23),(23,9),(9,20),(20,12),

            #(20,12),(12,17),(17,13),(13,20),
            
            #(20,1),(12,5),(17,8),(13,3),
            #(1,15),(1,9),(5,23),(5,22),(8,10),(8,16),(3,24),(3,21),
            
             #3(10,22),(23,15),(9,21),(24,16),
            #(10,2),(16,2),(24,6),(21,6),(9,7),(15,7),(23,4),(22,4),
            
            #3(2,19),(4,11),(7,18),(6,14),
            #(19,14),(14,18),(18,11),(11,19)

            # (23,13),(13, 18),(15,18),(18,7),(7,9),(9,20),(20,5),(5,15),(15,23),
            # (7,11),(11,21),(21,1),(1,13),
            # (1,19),(19,12),(12,4),(4,23),
            # (20,2),(2,14),(14,24),(24,3),(3,9),
            # (11,3),(24,16),(16,6),(6,21),
            # (6,19),(16,17),(17,8),(8,12),
            # (22,10),(10,4),(10,8),
            # (22,5),(22,2),(14,17)
            ]
    draw_edges(ax, verts, edges, color="k", linewidth=1.0)

    #plt.tight_layout()
    plt.savefig("BZ_vertices_with_Weyl_labeled.png", dpi=600)
    plt.show()


if __name__ == "__main__":
    main()

