import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,calinski_harabasz_score
import matplotlib.pyplot as plt

#数据读取
raw_data = pd.read_csv('C:\\Users\\ASUS\\Desktop\\附件\\chapter4\\cluster.txt')
numeric_features = raw_data.iloc[:,1:3] #获取数值型特征

#对数值型数据进行归一化 (AVG_ORDER和AVG_MONEY有量纲差异)
scaler = MinMaxScaler()
scaled_numeric_features = scaler.fit_transform(numeric_features)
'''print(scaled_numeric_features[:,:])'''

#训练聚类模型
n_cluster = 3 #设置聚类数量
model_kmeans = KMeans(n_cluster, random_state=0) #建立聚类模型对象
model_kmeans.fit(scaled_numeric_features) #训练聚类模型

#模型效果指标评估(非监督式评估方法）
silhouette_s = silhouette_score(scaled_numeric_features, model_kmeans.labels_, metric='euclidean') #平均轮廓系数
calinski_harabasz_s = calinski_harabasz_score(scaled_numeric_features, model_kmeans.labels_) #C&h得分
unsupervised_data = {'silh':[silhouette_s],'c&h':[calinski_harabasz_s]}
unsupervised_score = pd.DataFrame.from_dict(unsupervised_data)
'''print('\n','unsupervised score','\n','-'*60)
print(unsupervised_score)'''

#将聚类标签合并到原始数据
kmeans_labels = pd.DataFrame(model_kmeans.labels_, columns=['labels']) #组合原始数据与标签
kmeans_data = pd.concat((raw_data,kmeans_labels),axis=1)

#计算不同聚类类别的样本量和占比
label_count = kmeans_data.groupby(['labels'])['SEX'].count() #指定任意列对label做汇总统计
label_count_rate = label_count/kmeans_data.shape[0]

#将样本量和样本量占比合并为一个数据框
kmeans_record_count = pd.concat((label_count,label_count_rate),axis=1)
kmeans_record_count.columns = ['record_count','record_rate'] #给数据库列命名
'''print(kmeans_record_count.head()) #head() 将第一行看做列名，并默认输出之后的五行'''

#计算不同聚类类别数值型特征
kmeans_numerical_features = kmeans_data.groupby(['labels'])['AVG_ORDERS','AVG_MONEY'].mean()
'''print(kmeans_numerical_features.head())'''

#计算不同聚类类别分类型特征-用count计算频数并转换为频数占比
active_list = []
sex_list = []
unique_labels = np.unique(model_kmeans.labels_) #获取label的唯一值标签
for each_label in unique_labels:
    each_data = kmeans_data[kmeans_data['labels']==each_label]
    active_list.append(each_data.groupby(['IS_ACTIVE'])['USER_ID'].count()/each_data.shape[0])
    sex_list.append(each_data.groupby(['SEX'])['USER_ID'].count()/each_data.shape[0])

#将不同聚类类别分类型特征合并到数据框
kmeans_active_pd = pd.DataFrame(active_list)
kmeans_sex_pd = pd.DataFrame(sex_list)
kmeans_classified_features = pd.concat((kmeans_active_pd,kmeans_sex_pd),axis=1)
kmeans_classified_features.index = unique_labels #刷新index,便于后续与其他数据concat
'''print(kmeans_classified_features.head())'''

#合并所有类别的分析解雇
features_all = pd.concat((kmeans_record_count,kmeans_numerical_features,kmeans_classified_features),axis=1)
'''print(features_all.head())'''

#导出数据
'''features_all.to_csv('C:\\Users\\ASUS\\Desktop\\cluster_result.csv',encoding='utf_8_sig')'''

# 可视化图形展示
# part 1 全局配置
fig = plt.figure(figsize=(10, 7))
titles = ['RECORD_RATE', 'AVG_ORDERS', 'AVG_MONEY', 'IS_ACTIVE', 'SEX']  # 共用标题
line_index, col_index = 3, 5  # 定义网格数
ax_ids = np.arange(1, 16).reshape(line_index, col_index)  # 生成子网格索引值
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

# part 2 画出三个类别的占比
pie_fracs = features_all['record_rate'].tolist()
for ind in range(len(pie_fracs)):
    ax = fig.add_subplot(line_index, col_index, ax_ids[:, 0][ind])
    init_labels = ['', '', '']  # 初始化空label标签
    init_labels[ind] = 'cluster_{0}'.format(ind)  # 设置标签
    init_colors = ['lightgray', 'lightgray', 'lightgray']
    init_colors[ind] = 'g'  # 设置目标面积区别颜色
    ax.pie(x=pie_fracs, autopct='%3.0f %%', labels=init_labels, colors=init_colors)
    ax.set_aspect('equal')  # 设置饼图为圆形
    if ind == 0:
        ax.set_title(titles[0])

# part 3  画出AVG_ORDERS均值
avg_orders_label = 'AVG_ORDERS'
avg_orders_fraces = features_all[avg_orders_label]
for ind, frace in enumerate(avg_orders_fraces):
    ax = fig.add_subplot(line_index, col_index, ax_ids[:, 1][ind])
    ax.bar(x=unique_labels, height=[0, avg_orders_fraces[ind], 0])  # 画出柱形图
    ax.set_ylim((0, max(avg_orders_fraces) * 1.2))
    ax.set_xticks([])
    ax.set_yticks([])
    if ind == 0:  # 设置总标题
        ax.set_title(titles[1])
    # 设置每个柱形图的数值标签和x轴label
    ax.text(unique_labels[1], frace + 0.4, s='{:.2f}'.format(frace), ha='center', va='top')
    ax.text(unique_labels[1], -0.4, s=avg_orders_label, ha='center', va='bottom')

# part 4  画出AVG_MONEY均值
avg_money_label = 'AVG_MONEY'
avg_money_fraces = features_all[avg_money_label]
for ind, frace in enumerate(avg_money_fraces):
    ax = fig.add_subplot(line_index, col_index, ax_ids[:, 2][ind])
    ax.bar(x=unique_labels, height=[0, avg_money_fraces[ind], 0])  # 画出柱形图
    ax.set_ylim((0, max(avg_money_fraces) * 1.2))
    ax.set_xticks([])
    ax.set_yticks([])
    if ind == 0:  # 设置总标题
        ax.set_title(titles[2])
    # 设置每个柱形图的数值标签和x轴label
    ax.text(unique_labels[1], frace + 4, s='{:.0f}'.format(frace), ha='center', va='top')
    ax.text(unique_labels[1], -4, s=avg_money_label, ha='center', va='bottom')

# part 5  画出是否活跃
axtivity_labels = ['不活跃', '活跃']
x_ticket = [i for i in range(len(axtivity_labels))]
activity_data = features_all[axtivity_labels]
ylim_max = np.max(np.max(activity_data))
for ind, each_data in enumerate(activity_data.values):
    ax = fig.add_subplot(line_index, col_index, ax_ids[:, 3][ind])
    ax.bar(x=x_ticket, height=each_data)  # 画出柱形图
    ax.set_ylim((0, ylim_max * 1.2))
    ax.set_xticks([])
    ax.set_yticks([])
    if ind == 0:  # 设置总标题
        ax.set_title(titles[3])
    # 设置每个柱形图的数值标签和x轴label
    activity_values = ['{:.1%}'.format(i) for i in each_data]
    for i in range(len(x_ticket)):
        ax.text(x_ticket[i], each_data[i] + 0.05, s=activity_values[i], ha='center', va='top')
        ax.text(x_ticket[i], -0.05, s=axtivity_labels[i], ha='center', va='bottom')

# part 6  画出性别分布
sex_data = features_all.iloc[:, -3:]
x_ticket = [i for i in range(len(sex_data))]
sex_labels = ['SEX_{}'.format(i) for i in range(3)]
ylim_max = np.max(np.max(sex_data))
for ind, each_data in enumerate(sex_data.values):
    ax = fig.add_subplot(line_index, col_index, ax_ids[:, 4][ind])
    ax.bar(x=x_ticket, height=each_data)  # 画柱形图
    ax.set_ylim((0, ylim_max * 1.2))
    ax.set_xticks([])
    ax.set_yticks([])
    if ind == 0:  # 设置标题
        ax.set_title(titles[4])
        # 设置每个柱形图的数值标签和x轴label
    sex_values = ['{:.1%}'.format(i) for i in each_data]
    for i in range(len(x_ticket)):
        ax.text(x_ticket[i], each_data[i] + 0.1, s=sex_values[i], ha='center', va='top')
        ax.text(x_ticket[i], -0.1, s=sex_labels[i], ha='center', va='bottom')

plt.tight_layout(pad=0.8)  # 设置默认的间距
plt.savefig('C:\\Users\\ASUS\\Desktop\\cluster_result.png')









