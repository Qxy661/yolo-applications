"""生成风电叶片缺陷检测PPT配图 v2 — 中文标注 + 专业风格"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# ============================================================
# 全局设置
# ============================================================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 200

BASE = os.path.dirname(__file__)
IMG = os.path.join(BASE, 'images')
os.makedirs(IMG, exist_ok=True)

# 配色
C_BLUE = '#2E86AB'
C_GREEN = '#28A745'
C_ORANGE = '#FD7E14'
C_RED = '#DC3545'
C_PURPLE = '#6F42C1'
C_TEAL = '#17A2B8'
C_DARK = '#1A5676'
C_GRAY = '#6C757D'
C_LIGHT_BLUE = '#D6EAF8'
C_LIGHT_GREEN = '#D5F5E3'
C_LIGHT_ORANGE = '#FDEBD0'
C_LIGHT_RED = '#FADBD8'
C_LIGHT_PURPLE = '#E8DAEF'
C_LIGHT_TEAL = '#D1F2EB'
C_BG = '#FAFBFC'


def save(fig, name, bbox_inches='tight', pad=0.1):
    path = os.path.join(IMG, name)
    fig.savefig(path, dpi=200, bbox_inches=bbox_inches, pad_inches=pad,
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f'  Saved: {name}')


def add_rounded_box(ax, xy, w, h, text, fc, ec='none', fontsize=12,
                    fontweight='bold', fontcolor='black', alpha=0.9, radius=0.02):
    """添加圆角矩形框"""
    box = FancyBboxPatch(xy, w, h, boxstyle=f"round,pad=0.01,rounding_size={radius}",
                         facecolor=fc, edgecolor=ec, linewidth=1.5, alpha=alpha,
                         transform=ax.transAxes, clip_on=False)
    ax.add_patch(box)
    cx, cy = xy[0] + w/2, xy[1] + h/2
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize,
            fontweight=fontweight, color=fontcolor, transform=ax.transAxes)


def add_arrow(ax, start, end, color=C_GRAY, lw=2, style='->', mutation_scale=15):
    """添加箭头"""
    ax.annotate('', xy=end, xytext=start, xycoords='axes fraction',
                textcoords='axes fraction',
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                mutation_scale=mutation_scale))


# ============================================================
# fig02 — 风电场全景 + 缺陷类型
# ============================================================
def gen_fig02():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
    fig.subplots_adjust(wspace=0.3)

    # 左：风电场场景
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('风电场巡检场景', fontsize=16, fontweight='bold', pad=10)

    # 天空渐变
    for i in range(100):
        ax1.axhspan(i*0.06, (i+1)*0.06, color=plt.cm.Blues(0.2 + 0.3*i/100), alpha=0.5)

    # 地面
    ax1.fill_between([0, 10], [0, 0], [2, 2], color='#90EE90', alpha=0.6)

    # 风力发电机
    for x in [2, 5, 8]:
        # 塔架
        ax1.plot([x, x], [2, 7.5], color='#555', lw=3, solid_capstyle='round')
        # 机舱
        ax1.add_patch(FancyBboxPatch((x-0.3, 7.2), 0.6, 0.5, boxstyle="round,pad=0.05",
                                      facecolor='#888', edgecolor='#555'))
        # 叶片
        for angle in [0, 120, 240]:
            rad = np.radians(angle + 30)
            dx = 2.0 * np.cos(rad)
            dy = 2.0 * np.sin(rad)
            ax1.plot([x, x+dx], [7.5, 7.5+dy], color='#666', lw=2.5, solid_capstyle='round')

    # 标注
    ax1.text(5, 9.2, '全球风电装机 >1000GW', ha='center', fontsize=11,
             fontweight='bold', color=C_DARK,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax1.text(5, 0.5, '无人机巡检: 15-30分钟/台', ha='center', fontsize=10,
             color=C_GREEN, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # 右：缺陷类型
    ax2.axis('off')
    ax2.set_title('典型叶片缺陷类型', fontsize=16, fontweight='bold', pad=10)

    defects = [
        ('裂纹 (Crack)', '细长线状，宽<5px', C_RED, 0.90, '检测难度: 极高'),
        ('侵蚀 (Erosion)', '面状剥落，边界不规则', C_ORANGE, 0.70, '检测难度: 中等'),
        ('雷击 (Lightning)', '焦黑放射状', C_PURPLE, 0.50, '检测难度: 中等'),
        ('涂层脱落 (Peeling)', '片状剥落', C_TEAL, 0.40, '检测难度: 中等'),
        ('孔洞 (Hole)', '圆形/不规则空洞', C_GREEN, 0.30, '检测难度: 中等'),
    ]

    for i, (name, desc, color, pct, diff) in enumerate(defects):
        y = 0.88 - i * 0.17
        # 背景条
        ax2.barh(y, pct, height=0.10, left=0.35, color=color, alpha=0.25, edgecolor=color, lw=1.5)
        # 标签
        ax2.text(0.33, y+0.02, name, ha='right', va='center', fontsize=12,
                 fontweight='bold', color=color)
        ax2.text(0.35, y-0.04, desc, ha='left', va='center', fontsize=9, color=C_GRAY)
        # 百分比
        ax2.text(0.35 + pct + 0.02, y+0.02, f'{int(pct*100)}%', ha='left', va='center',
                 fontsize=10, fontweight='bold', color=color)
        # 难度
        ax2.text(0.98, y+0.02, diff, ha='right', va='center', fontsize=8, color=C_GRAY)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    save(fig, 'fig02_wind_farm.png')


# ============================================================
# fig03 — 五大技术挑战
# ============================================================
def gen_fig03():
    fig, ax = plt.subplots(figsize=(14, 6), facecolor='white')
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('五大技术挑战', fontsize=20, fontweight='bold', pad=15, color=C_DARK)

    challenges = [
        ('尺度差异极大\n>100×', '微小裂纹2-5px\nvs 大面积侵蚀数百px', '★★★★★', C_RED, C_LIGHT_RED),
        ('背景高度复杂', '天空/云层/塔架干扰\n缺陷对比度低', '★★★★☆', C_ORANGE, C_LIGHT_ORANGE),
        ('部署环境受限', '无人机载荷有限\n模型需<10MB', '★★★★☆', C_BLUE, C_LIGHT_BLUE),
        ('标注数据稀缺', '专业标注成本高\n样本不平衡', '★★★☆☆', C_GREEN, C_LIGHT_GREEN),
        ('缺陷形态多样', '裂纹(线)/侵蚀(面)\n孔洞(点)/雷击(放射)', '★★★☆☆', C_PURPLE, C_LIGHT_PURPLE),
    ]

    n = len(challenges)
    w = 0.16
    gap = 0.02
    start_x = (1 - n*w - (n-1)*gap) / 2

    for i, (title, desc, stars, color, bg) in enumerate(challenges):
        x = start_x + i * (w + gap)
        # 卡片背景
        box = FancyBboxPatch((x, 0.08), w, 0.78, boxstyle="round,pad=0.01,rounding_size=0.02",
                             facecolor=bg, edgecolor=color, linewidth=2, alpha=0.9,
                             transform=ax.transAxes, clip_on=False)
        ax.add_patch(box)
        # 标题
        ax.text(x + w/2, 0.72, title, ha='center', va='center', fontsize=11,
                fontweight='bold', color=color, transform=ax.transAxes)
        # 描述
        ax.text(x + w/2, 0.42, desc, ha='center', va='center', fontsize=9,
                color='#333', transform=ax.transAxes, linespacing=1.4)
        # 星级
        ax.text(x + w/2, 0.16, stars, ha='center', va='center', fontsize=12,
                color=color, transform=ax.transAxes)

    save(fig, 'fig03_challenges.png')


# ============================================================
# fig04 — GCB-YOLO 架构
# ============================================================
def gen_fig04():
    fig, ax = plt.subplots(figsize=(14, 5), facecolor='white')
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('GCB-YOLO 架构 (Wind Energy 2025)', fontsize=18, fontweight='bold',
                 pad=15, color=C_DARK)

    # 模块
    modules = [
        ('输入图像', C_GRAY, '#F0F0F0'),
        ('GhostNet\nBackbone', C_GREEN, C_LIGHT_GREEN),
        ('CA\n坐标注意力', C_ORANGE, C_LIGHT_ORANGE),
        ('BiFPN\n加权融合', C_BLUE, C_LIGHT_BLUE),
        ('检测头', C_PURPLE, C_LIGHT_PURPLE),
        ('输出', C_GRAY, '#F0F0F0'),
    ]

    n = len(modules)
    w = 0.12
    gap = 0.025
    start_x = (1 - n*w - (n-1)*gap) / 2

    for i, (name, ec, fc) in enumerate(modules):
        x = start_x + i * (w + gap)
        box = FancyBboxPatch((x, 0.3), w, 0.4, boxstyle="round,pad=0.01,rounding_size=0.015",
                             facecolor=fc, edgecolor=ec, linewidth=2.5, alpha=0.95,
                             transform=ax.transAxes, clip_on=False)
        ax.add_patch(box)
        ax.text(x + w/2, 0.5, name, ha='center', va='center', fontsize=11,
                fontweight='bold', color=ec, transform=ax.transAxes)
        # 箭头
        if i < n-1:
            ax.annotate('', xy=(x+w+gap*0.8, 0.5), xytext=(x+w+gap*0.2, 0.5),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', color=C_GRAY, lw=2))

    # 底部指标
    metrics = [
        ('7.5MB', '模型大小', C_GREEN),
        ('94.72% mAP', '检测精度', C_BLUE),
        ('115.3 FPS', '推理速度', C_ORANGE),
    ]
    for i, (val, label, color) in enumerate(metrics):
        x = 0.2 + i * 0.3
        ax.text(x, 0.15, val, ha='center', va='center', fontsize=14,
                fontweight='bold', color=color, transform=ax.transAxes)
        ax.text(x, 0.07, label, ha='center', va='center', fontsize=9,
                color=C_GRAY, transform=ax.transAxes)

    save(fig, 'fig04_gcb_yolo.png')


# ============================================================
# fig05 — WTBD-YOLOv8 四大模块
# ============================================================
def gen_fig05():
    fig, ax = plt.subplots(figsize=(14, 5), facecolor='white')
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('WTBD-YOLOv8 四大创新模块 (Sustainability 2024)', fontsize=16,
                 fontweight='bold', pad=15, color=C_DARK)

    modules = [
        ('GhostCBS', '替换标准卷积\n参数 -38.2%', C_GREEN, C_LIGHT_GREEN),
        ('DFSB-C3k2', '密集特征\n尺度平衡', C_BLUE, C_LIGHT_BLUE),
        ('MHSA-C3k2', '多头自注意力\n全局上下文', C_ORANGE, C_LIGHT_ORANGE),
        ('Mini-BiFPN', '2层加权融合\n参数 -50%', C_PURPLE, C_LIGHT_PURPLE),
    ]

    n = len(modules)
    w = 0.18
    gap = 0.03
    start_x = (1 - n*w - (n-1)*gap) / 2

    for i, (name, desc, ec, fc) in enumerate(modules):
        x = start_x + i * (w + gap)
        box = FancyBboxPatch((x, 0.25), w, 0.5, boxstyle="round,pad=0.01,rounding_size=0.02",
                             facecolor=fc, edgecolor=ec, linewidth=2.5, alpha=0.95,
                             transform=ax.transAxes, clip_on=False)
        ax.add_patch(box)
        ax.text(x + w/2, 0.57, name, ha='center', va='center', fontsize=13,
                fontweight='bold', color=ec, transform=ax.transAxes)
        ax.text(x + w/2, 0.38, desc, ha='center', va='center', fontsize=9,
                color='#333', transform=ax.transAxes, linespacing=1.4)
        if i < n-1:
            ax.annotate('', xy=(x+w+gap*0.8, 0.5), xytext=(x+w+gap*0.2, 0.5),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', color=C_GRAY, lw=2))

    # 底部结果
    results = [
        ('AP: 98.3%', '(+2.2%)', C_GREEN),
        ('小目标AP: 97.9%', '(+4.8%)', C_BLUE),
        ('参数: 1.99M', '(-38.2%)', C_ORANGE),
    ]
    for i, (val, delta, color) in enumerate(results):
        x = 0.2 + i * 0.3
        ax.text(x, 0.12, f'{val} {delta}', ha='center', va='center', fontsize=11,
                fontweight='bold', color=color, transform=ax.transAxes)

    save(fig, 'fig05_wtbd_yolov8.png')


# ============================================================
# fig06 — SimAM 零参数注意力 (替换旧版)
# ============================================================
def gen_fig06():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor='white',
                                    gridspec_kw={'width_ratios': [1.2, 1]})
    fig.subplots_adjust(wspace=0.3)

    # 左：SimAM 能量函数
    ax1.set_title('SimAM 能量函数', fontsize=14, fontweight='bold', color=C_DARK)
    t = np.linspace(-3, 3, 200)
    energy = 4 * (t - 0)**2 / (2 * 1**2 + 0.0001)  # simplified
    attention = 1 / (1 + np.exp(energy - 8))

    ax1_twin = ax1.twinx()
    l1, = ax1.plot(t, energy, color=C_BLUE, lw=2.5, label='能量值 e_t')
    l2, = ax1_twin.plot(t, attention, color=C_RED, lw=2.5, ls='--', label='注意力权重')
    ax1.set_xlabel('像素值 t', fontsize=11)
    ax1.set_ylabel('能量值', fontsize=11, color=C_BLUE)
    ax1_twin.set_ylabel('注意力权重', fontsize=11, color=C_RED)
    ax1.legend([l1, l2], ['能量值 e_t', '注意力权重'], loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 右：参数量对比
    ax2.set_title('注意力机制参数量对比', fontsize=14, fontweight='bold', color=C_DARK)
    methods = ['SE', 'CBAM', 'CA', 'ECA', 'SimAM']
    params = [0.05, 0.08, 0.06, 0.01, 0.0]
    colors = [C_GRAY, C_RED, C_BLUE, C_GREEN, C_ORANGE]
    bars = ax2.bar(methods, params, color=colors, alpha=0.8, edgecolor='white', lw=1.5)
    ax2.set_ylabel('参数量 (M)', fontsize=11)
    ax2.set_ylim(0, 0.12)
    for bar, p in zip(bars, params):
        label = 'ZERO!' if p == 0 else f'{p:.2f}M'
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 label, ha='center', fontsize=10, fontweight='bold',
                 color=C_ORANGE if p == 0 else '#333')
    ax2.grid(axis='y', alpha=0.3)

    # 添加大标题
    fig.suptitle('SimAM: 零参数注意力机制 (LE-YOLO, IEEE Access 2024)',
                 fontsize=16, fontweight='bold', y=1.02, color=C_DARK)

    save(fig, 'fig06_le_yolo.png')


# ============================================================
# fig07a — 三条技术路线
# ============================================================
def gen_fig07a():
    fig, ax = plt.subplots(figsize=(14, 6), facecolor='white')
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('三条技术路线 + 跨论文共同配方', fontsize=20, fontweight='bold',
                 pad=15, color=C_DARK)

    routes = [
        ('路线一: 轻量化卷积', 'GhostNet / GSConv / GhostCBS', '参数量 -38~44%', C_GREEN, C_LIGHT_GREEN),
        ('路线二: 注意力机制', 'CA / SimAM / MHSA', '+2~3% mAP\n避免CBAM!', C_BLUE, C_LIGHT_BLUE),
        ('路线三: 特征融合', 'BiFPN 加权双向融合', '+4.6% mAP\n小目标 +4.8%', C_ORANGE, C_LIGHT_ORANGE),
    ]

    n = len(routes)
    w = 0.26
    gap = 0.03
    start_x = (1 - n*w - (n-1)*gap) / 2

    for i, (title, tech, effect, ec, fc) in enumerate(routes):
        x = start_x + i * (w + gap)
        box = FancyBboxPatch((x, 0.25), w, 0.55, boxstyle="round,pad=0.01,rounding_size=0.02",
                             facecolor=fc, edgecolor=ec, linewidth=2.5, alpha=0.95,
                             transform=ax.transAxes, clip_on=False)
        ax.add_patch(box)
        ax.text(x + w/2, 0.67, title, ha='center', va='center', fontsize=13,
                fontweight='bold', color=ec, transform=ax.transAxes)
        ax.text(x + w/2, 0.50, tech, ha='center', va='center', fontsize=10,
                color='#333', transform=ax.transAxes)
        ax.text(x + w/2, 0.35, effect, ha='center', va='center', fontsize=11,
                fontweight='bold', color=ec, transform=ax.transAxes, linespacing=1.4)

    # 底部共同配方
    box = FancyBboxPatch((0.1, 0.03), 0.8, 0.14, boxstyle="round,pad=0.01,rounding_size=0.015",
                         facecolor=C_LIGHT_ORANGE, edgecolor=C_ORANGE, linewidth=2,
                         transform=ax.transAxes, clip_on=False)
    ax.add_patch(box)
    ax.text(0.5, 0.10, '共同配方: YOLO + 轻量化 + 注意力 + BiFPN + EIoU/DFL = 更小、更快、更准',
            ha='center', va='center', fontsize=13, fontweight='bold', color=C_ORANGE,
            transform=ax.transAxes)

    save(fig, 'fig07_three_routes.png')


# ============================================================
# fig07b — 风电 vs 桥梁跨领域对比
# ============================================================
def gen_fig07b():
    fig, ax = plt.subplots(figsize=(14, 7), facecolor='white')
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('风电叶片 vs 桥梁结构: 跨领域技术对比', fontsize=18, fontweight='bold',
                 pad=15, color=C_DARK)

    # 左卡片：风电
    box1 = FancyBboxPatch((0.03, 0.12), 0.43, 0.75, boxstyle="round,pad=0.01,rounding_size=0.02",
                          facecolor=C_LIGHT_BLUE, edgecolor=C_BLUE, linewidth=2.5,
                          transform=ax.transAxes, clip_on=False)
    ax.add_patch(box1)
    ax.text(0.245, 0.80, '风电叶片', ha='center', fontsize=16, fontweight='bold',
            color=C_BLUE, transform=ax.transAxes)

    items_left = [
        ('数据资源', '稀缺 (~9,900张)'),
        ('技术基线', 'YOLOv11'),
        ('缺陷类型', '裂纹/侵蚀/雷击'),
        ('核心挑战', '小目标 + 数据稀缺'),
        ('技术成熟度', '相对成熟 (4篇验证)'),
    ]
    for i, (k, v) in enumerate(items_left):
        y = 0.68 - i * 0.12
        ax.text(0.08, y, f'● {k}:', fontsize=10, fontweight='bold', color=C_BLUE,
                transform=ax.transAxes)
        ax.text(0.25, y, v, fontsize=10, color='#333', transform=ax.transAxes)

    # 右卡片：桥梁
    box2 = FancyBboxPatch((0.54, 0.12), 0.43, 0.75, boxstyle="round,pad=0.01,rounding_size=0.02",
                          facecolor=C_LIGHT_GREEN, edgecolor=C_GREEN, linewidth=2.5,
                          transform=ax.transAxes, clip_on=False)
    ax.add_patch(box2)
    ax.text(0.755, 0.80, '桥梁结构', ha='center', fontsize=16, fontweight='bold',
            color=C_GREEN, transform=ax.transAxes)

    items_right = [
        ('数据资源', '丰富 (56K+张)'),
        ('技术基线', 'YOLOv5/v7/v8'),
        ('缺陷类型', '裂纹/剥落/锈蚀'),
        ('核心挑战', '细长裂缝 + 多场景'),
        ('技术成熟度', '快速追赶 (500+/年)'),
    ]
    for i, (k, v) in enumerate(items_right):
        y = 0.68 - i * 0.12
        ax.text(0.59, y, f'● {k}:', fontsize=10, fontweight='bold', color=C_GREEN,
                transform=ax.transAxes)
        ax.text(0.76, y, v, fontsize=10, color='#333', transform=ax.transAxes)

    # 底部共性
    box3 = FancyBboxPatch((0.03, 0.01), 0.94, 0.08, boxstyle="round,pad=0.01,rounding_size=0.01",
                          facecolor=C_LIGHT_ORANGE, edgecolor=C_ORANGE, linewidth=2,
                          transform=ax.transAxes, clip_on=False)
    ax.add_patch(box3)
    ax.text(0.5, 0.05, '共性: 裂纹检测是核心需求 | 轻量化+注意力+BiFPN是共同最佳实践 | 迁移学习可行',
            ha='center', fontsize=11, fontweight='bold', color=C_ORANGE, transform=ax.transAxes)

    save(fig, 'fig07_cross_domain.png')


# ============================================================
# fig08 — 数据集对比柱状图
# ============================================================
def gen_fig08():
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')

    datasets = ['Blade30\n(真实无人机)', '合成数据\n(zhaowenhai)', 'Fan Defect\n(mxy021120)', '9类无人机\n(share2code)']
    counts = [1302, 3800, 4802, 9900]
    colors = [C_GREEN, C_BLUE, C_ORANGE, C_PURPLE]
    stars = ['★★★★★', '★★★☆☆', '★★★★☆', '★★★★☆']

    bars = ax.bar(datasets, counts, color=colors, alpha=0.8, edgecolor='white', lw=2, width=0.6)

    for bar, count, star in zip(bars, counts, stars):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 150,
                f'{count:,}', ha='center', fontsize=13, fontweight='bold', color='#333')
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                star, ha='center', fontsize=11, color='white', fontweight='bold')

    ax.set_ylabel('图像数量', fontsize=13, fontweight='bold')
    ax.set_title('风电叶片缺陷检测数据集对比', fontsize=16, fontweight='bold', pad=15, color=C_DARK)
    ax.set_ylim(0, 11500)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save(fig, 'fig08_datasets.png')


# ============================================================
# fig09 — 损失函数演进
# ============================================================
def gen_fig09():
    fig, ax = plt.subplots(figsize=(14, 5), facecolor='white')
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('损失函数演进', fontsize=20, fontweight='bold', pad=15, color=C_DARK)

    losses = [
        ('MSE', '坐标差', C_GRAY, '#F0F0F0'),
        ('IoU', '重叠度量', C_GRAY, '#F0F0F0'),
        ('GIoU', '+包围框', C_BLUE, C_LIGHT_BLUE),
        ('DIoU', '+中心距离', C_BLUE, C_LIGHT_BLUE),
        ('CIoU', '+宽高比', C_BLUE, C_LIGHT_BLUE),
        ('EIoU', '分解三组件', C_GREEN, C_LIGHT_GREEN),
        ('DFL', '分布建模', C_ORANGE, C_LIGHT_ORANGE),
    ]

    n = len(losses)
    w = 0.095
    gap = 0.015
    start_x = (1 - n*w - (n-1)*gap) / 2

    for i, (name, desc, ec, fc) in enumerate(losses):
        x = start_x + i * (w + gap)
        # 圆形
        circle = plt.Circle((x + w/2, 0.55), w/2, facecolor=fc, edgecolor=ec,
                            linewidth=2.5, alpha=0.95, transform=ax.transAxes, clip_on=False)
        ax.add_patch(circle)
        ax.text(x + w/2, 0.58, name, ha='center', va='center', fontsize=11,
                fontweight='bold', color=ec, transform=ax.transAxes)
        ax.text(x + w/2, 0.46, desc, ha='center', va='center', fontsize=8,
                color=C_GRAY, transform=ax.transAxes)
        # 箭头
        if i < n-1:
            ax.annotate('', xy=(x+w+gap*0.7, 0.55), xytext=(x+w+gap*0.3, 0.55),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', color=C_GRAY, lw=1.5))

    # 推荐框
    box = FancyBboxPatch((0.35, 0.08), 0.3, 0.15, boxstyle="round,pad=0.01,rounding_size=0.015",
                         facecolor=C_LIGHT_ORANGE, edgecolor=C_ORANGE, linewidth=2,
                         transform=ax.transAxes, clip_on=False)
    ax.add_patch(box)
    ax.text(0.5, 0.155, '风电推荐', ha='center', fontsize=12, fontweight='bold',
            color=C_ORANGE, transform=ax.transAxes)
    ax.text(0.5, 0.105, 'EIoU + DFL', ha='center', fontsize=14, fontweight='bold',
            color=C_ORANGE, transform=ax.transAxes)

    save(fig, 'fig09_loss.png')


# ============================================================
# fig10 — 推理链
# ============================================================
def gen_fig10():
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('从调研到方案: 推理链', fontsize=20, fontweight='bold', pad=15, color=C_DARK)

    # 三层
    layers = [
        (0.65, '文献调研结论', 'GCB-YOLO: 7.5MB, 94.72% | WTBD: 1.99M, 98.3% | LE-YOLO: 2.1M, 78.7%',
         C_LIGHT_BLUE, C_BLUE),
        (0.40, '为什么选 YOLOv11?', 'C3k2 (精细特征) + C2PSA (位置注意力) + 解耦头 + DFL',
         C_LIGHT_GREEN, C_GREEN),
        (0.15, '我们的方案', 'YOLOv11 + GhostNet + CA + BiFPN + EIoU/DFL + Mosaic/MixUp + SAHI\n预期: mAP>90% | 模型<10MB | 速度>60FPS',
         C_LIGHT_ORANGE, C_ORANGE),
    ]

    for y, title, desc, fc, ec in layers:
        box = FancyBboxPatch((0.08, y), 0.84, 0.20, boxstyle="round,pad=0.01,rounding_size=0.02",
                             facecolor=fc, edgecolor=ec, linewidth=2.5, alpha=0.95,
                             transform=ax.transAxes, clip_on=False)
        ax.add_patch(box)
        ax.text(0.5, y + 0.14, title, ha='center', fontsize=14, fontweight='bold',
                color=ec, transform=ax.transAxes)
        ax.text(0.5, y + 0.06, desc, ha='center', fontsize=10, color='#333',
                transform=ax.transAxes, linespacing=1.5)

    # 箭头
    ax.annotate('', xy=(0.5, 0.62), xytext=(0.5, 0.58),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color=C_GRAY, lw=2.5))
    ax.annotate('', xy=(0.5, 0.37), xytext=(0.5, 0.33),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color=C_GRAY, lw=2.5))

    save(fig, 'fig10_reasoning.png')


# ============================================================
# fig11 — 方案总体架构
# ============================================================
def gen_fig11():
    fig, ax = plt.subplots(figsize=(14, 7), facecolor='white')
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('拟采用方案总体架构', fontsize=20, fontweight='bold', pad=15, color=C_DARK)

    # 上排：主流程
    main_modules = [
        ('数据增强', 'Mosaic+MixUp\n+CopyPaste', C_LIGHT_BLUE, C_BLUE),
        ('Backbone', 'GhostNet + C3k2\n(轻量化)', C_LIGHT_GREEN, C_GREEN),
        ('★ Neck', 'BiFPN + CA\n(双向融合+注意力)', C_LIGHT_ORANGE, C_ORANGE),
        ('Head', '解耦AF + DFL\n(精确回归)', C_LIGHT_PURPLE, C_PURPLE),
    ]

    w = 0.18
    gap = 0.03
    start_x = (1 - len(main_modules)*w - (len(main_modules)-1)*gap) / 2

    for i, (name, desc, fc, ec) in enumerate(main_modules):
        x = start_x + i * (w + gap)
        box = FancyBboxPatch((x, 0.58), w, 0.28, boxstyle="round,pad=0.01,rounding_size=0.02",
                             facecolor=fc, edgecolor=ec, linewidth=2.5, alpha=0.95,
                             transform=ax.transAxes, clip_on=False)
        ax.add_patch(box)
        ax.text(x + w/2, 0.76, name, ha='center', fontsize=12, fontweight='bold',
                color=ec, transform=ax.transAxes)
        ax.text(x + w/2, 0.66, desc, ha='center', fontsize=9, color='#333',
                transform=ax.transAxes, linespacing=1.4)
        if i < len(main_modules)-1:
            ax.annotate('', xy=(x+w+gap*0.8, 0.72), xytext=(x+w+gap*0.2, 0.72),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', color=C_GRAY, lw=2))

    # 下排：损失 + SAHI
    # 损失函数
    box_loss = FancyBboxPatch((0.05, 0.25), 0.25, 0.20,
                              boxstyle="round,pad=0.01,rounding_size=0.02",
                              facecolor=C_LIGHT_GREEN, edgecolor=C_GREEN, linewidth=2.5,
                              transform=ax.transAxes, clip_on=False)
    ax.add_patch(box_loss)
    ax.text(0.175, 0.39, '损失函数', ha='center', fontsize=12, fontweight='bold',
            color=C_GREEN, transform=ax.transAxes)
    ax.text(0.175, 0.31, 'EIoU + DFL', ha='center', fontsize=11, fontweight='bold',
            color=C_GREEN, transform=ax.transAxes)

    # SAHI
    box_sahi = FancyBboxPatch((0.35, 0.25), 0.60, 0.20,
                              boxstyle="round,pad=0.01,rounding_size=0.02",
                              facecolor=C_LIGHT_RED, edgecolor=C_RED, linewidth=2.5,
                              transform=ax.transAxes, clip_on=False)
    ax.add_patch(box_sahi)
    ax.text(0.65, 0.39, '★ SAHI 切片推理', ha='center', fontsize=12, fontweight='bold',
            color=C_RED, transform=ax.transAxes)
    ax.text(0.65, 0.31, '640×640 切片 | overlap=0.2 | conf=0.15 | NMS融合', ha='center',
            fontsize=9, color='#333', transform=ax.transAxes)

    # 底部指标
    targets = [
        ('mAP >90%', C_BLUE),
        ('模型 <10MB', C_GREEN),
        ('速度 >60 FPS', C_ORANGE),
    ]
    for i, (text, color) in enumerate(targets):
        x = 0.2 + i * 0.3
        ax.text(x, 0.10, text, ha='center', fontsize=14, fontweight='bold',
                color=color, transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, lw=2))

    save(fig, 'fig11_architecture.png')


# ============================================================
# 主函数
# ============================================================
if __name__ == '__main__':
    print('Generating figures v2 (中文标注)...')
    gen_fig02()
    gen_fig03()
    gen_fig04()
    gen_fig05()
    gen_fig06()
    gen_fig07a()
    gen_fig07b()
    gen_fig08()
    gen_fig09()
    gen_fig10()
    gen_fig11()
    print('Done! All figures generated.')
