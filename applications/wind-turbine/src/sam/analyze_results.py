import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 读取CSV
df = pd.read_csv('detection_results.csv')
print(f"Total detections: {len(df)}")
print(df.head())

# 设置中文字体（如果系统有SimHei，否则注释掉）
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 1. 各类别检测数量
plt.figure(figsize=(10,6))
counts = df['class_name'].value_counts()
sns.barplot(x=counts.values, y=counts.index, palette='viridis')
plt.xlabel('Detection Count')
plt.ylabel('Class')
plt.title('Number of Detections per Class')
plt.tight_layout()
plt.savefig('analysis_counts.png', dpi=300)
plt.show()

# 2. 置信度箱线图
plt.figure(figsize=(10,6))
sns.boxplot(data=df, x='class_name', y='confidence', palette='Set2')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Class')
plt.ylabel('Confidence')
plt.title('Confidence Distribution by Class')
plt.tight_layout()
plt.savefig('analysis_confidence_box.png', dpi=300)
plt.show()

# 3. 掩码面积分布（对数坐标）
plt.figure(figsize=(12,6))
df['log_area'] = np.log10(df['area_px'] + 1)
sns.boxplot(data=df, x='class_name', y='log_area', palette='coolwarm')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Class')
plt.ylabel('Log10(Area in pixels)')
plt.title('Mask Area Distribution by Class (log scale)')
plt.tight_layout()
plt.savefig('analysis_area_box.png', dpi=300)
plt.show()

# 4. 掩码得分直方图
plt.figure(figsize=(10,6))
sns.histplot(df['mask_score'], bins=30, kde=True, color='green')
plt.xlabel('Mask Quality Score')
plt.ylabel('Frequency')
plt.title('Distribution of SAM Mask Scores')
plt.tight_layout()
plt.savefig('analysis_mask_scores.png', dpi=300)
plt.show()

# 5. 置信度 vs 掩码得分散点图
plt.figure(figsize=(10,8))
sns.scatterplot(data=df, x='confidence', y='mask_score', hue='class_name', alpha=0.7, s=30)
plt.xlabel('YOLO Confidence')
plt.ylabel('SAM Mask Score')
plt.title('Confidence vs Mask Score by Class')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('analysis_conf_vs_mask.png', dpi=300)
plt.show()

# 输出各类别面积统计
area_stats = df.groupby('class_name')['area_px'].describe()
print(area_stats)
area_stats.to_csv('area_statistics.csv')