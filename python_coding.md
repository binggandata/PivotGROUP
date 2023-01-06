## 换源_python

```python
!pip insall package_name -i https://pypi.douban.com/simple/ 
#从指定镜像下载安装工具包，镜像URL可自行修改
```
# 模型服务

## keras 部署服务

```python
import tensorflow as tf
## freeze traiing session
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    ## get session graph
    graph = session.graph
    with graph.as_default():
        ## remove training related nodes
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        ## remove device info if trained on gpu
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph
        
from keras import backend as K
frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, "./", "model.pb", as_text=False)
```

## 可部署为服务的 Python 脚本

```python
__name__ = 'model'

def func(input_object):
    ## your code here
    return output_object
```

# SQL操作

## 连接 PostgreSQL

```python
import psycopg2
import pandas as pd

connection = psycopg2.connect(user = "username",
                                  password = "password",
                                  host = "127.0.0.1",
                                  port = "5432",
                                  database = "dn_name")
cursor = connection.cursor()

test_query = """SELECT subject_id, hadm_id, admittime, dischtime, admission_type, diagnosis
FROM admissions
"""

test = pd.read_sql_query(test_query, connection)
```

## 单表导出数据

```python
### 从单表中获取数据
### 使用 WHERE 和 HAVING 语句进行过滤
### 用 order_by_list 字段进行降序排序
### conn 为 postgreSQL 的数据库连接对象
query_string = """SELECT [DISTINCT] <select_column_list> [AS <alias_name>]
FROM <table_name>
WHERE <where_condition_1> AND <where_condition_2> OR <where_condition_3>
HAVING <having_condition>
ORDER BY <order_by_list> [DESC]
"""

df = pd.read_sql_query(query_string, conn)


```

## 使用 WHERE 语句过滤

```python
### 从单表中获取数据
### 使用 WHERE 语句进行过滤，包含了 IN 包含关系，和使用 LIKE 做模式匹配
### conn 为 postgreSQL 的数据库连接对象
query_string = """SELECT [DISTINCT] <select_column_list> [AS <alias_name>]
FROM <table_name>
WHERE column_name [NOT] IN ( value_1, value_2, ...,value_n)
AND column_name BETWEEN value_1 AND value_2
OR column_name LIKE 'string'
"""

df = pd.read_sql_query(query_string, conn)

```

## 使用 HAVING 语句过滤

```python
### 从单表中获取数据
### 使用 HAVING 语句作过滤，包含聚合函数
### conn 为 postgreSQL 的数据库连接对象
query_string = """SELECT [DISTINCT] <select_column_list> [AS <alias_name>]
FROM <table_name>
HAVING [aggregation function] = value_1
AND [aggregation_function] = value_2
"""

df = pd.read_sql_query(query_string, conn)
```

## 取出前N条数据

```python
### 从单表中获取数据
### 使用 WHERE 和 HAVING 语句进行过滤
### 用 order_by_list 字段进行降序排序
### 取出头部的N条数据
### conn 为 postgreSQL 的数据库连接对象
query_string = """SELECT [DISTINCT] <select_column_list> [AS <alias_name>]
FROM <table_name>
WHERE <where_condition_1> AND <where_condition_2> OR <where_condition_3>
HAVING <having_condition>
ORDER BY <order_by_list> [DESC]
LIMIT <selected N>
"""

df = pd.read_sql_query(query_string, conn)
```

## 多表导出数据

```python
### 从多表中获取数据
### 使用 INNER JOIN 从两表中获取数据
### 使用 WHERE 和 HAVING 语句进行过滤
### conn 为 postgreSQL 的数据库连接对象
query_string = """SELECT [DISTINCT] <select_column_list>
FROM <left_table>
<join_type> JOIN <right_table>
ON <join_condition>  
WHERE <where_condition>
HAVING <having_condition>
ORDER BY <order_by_list> DESC
LIMIT <limit_number>
"""

df = pd.read_sql_query(query_string, conn)
```

## 使用聚合函数

```python
### 从多表中获取数据
### 使用 INNER JOIN 从两表中获取数据
### 使用 WHERE 和 HAVING 语句进行过滤
### conn 为 postgreSQL 的数据库连接对象
query_string = """SELECT [aggregation function] (<column_name>)
FROM <left_table>
<join_type> JOIN <right_table>
ON <join_condition>
WHERE <where_condition>
GROUP BY <group_by_all_the_left_columns>
[HAVING <having_condition>]
[ORDER BY <order_by_list>]
[LIMIT <limit_number>]
"""

df = pd.read_sql_query(query_string, conn)
```

## 使用 Subquery

```python
### 如果SQL Query过长过复杂，可把Subquery的结果进行调用
### conn 为 postgreSQL 的数据库连接对象

query_string = """--subquery作为一张表
SELECT <column_list> FROM ( 
  SELECT <column_list> FROM table_name
) AS alias_table_name
"""
query_string_1 = """--subquery作为一系列值
SELECT <column_list> FROM <table_name>
WHERE <column_name> IN (
  SELECT <column_name> FROM <table_name> WHERE  <where_condition>
)
"""

df = pd.read_sql_query(query_string, conn)
```

## 使用 WITH 语句

```python
### 如果SQL Query过长过复杂，可以将子查询的结果定义为表变量在后期复用
### conn 为 postgreSQL 的数据库连接对象
query_string = """WITH table_variable_1 AS (
  <SELECT query>
),

table_variable_2 AS (
  SELECT * FROM table_variable_1;
)

SELECT * FROM table_variable_2；
"""

df = pd.read_sql_query(query_string, conn)
```

## 通用条件表达式 (类似if-else)

```python
### 条件语句
### conn 为 postgreSQL 的数据库连接对象
query_string = """-- 通用条件表达式 (类似if-else)
CASE  
WHEN condition1 THEN result1 -- i.e WHEN count > 5 THEN 1
WHEN condition2 THEN result2 -- i.e. WHEN name = 'elle' THEN 'ELLE'
[...]
[ELSE result_n]
END 
"""

df = pd.read_sql_query(query_string, conn)


```

## 查看数据库中所有表名

```python
### 查看数据库中所有表名
### conn 为 postgreSQL 的数据库连接对象
query_string = """
SELECT * FROM pg_tables
WHERE schemaname <> 'pg_catalog' AND schemaname <> 'information_schema'; 
"""

df = pd.read_sql_query(query_string, conn)
```

## 窗口函数

```python
### 窗口函数
### conn 为 postgreSQL 的数据库连接对象
query_string = """SELECT <<column_name>,
       window_func() OVER ( [PARTITION BY xx] [ORDER BY xx] )
FROM <table_name>
"""

df = pd.read_sql_query(query_string, conn)
```

## 查看表内字段类型

```python
### 查看表内字段类型
### conn 为 postgreSQL 的数据库连接对象
query_string = """
select column_name, data_type 
from information_schema.columns 
where table_name = <table_name>
"""

df = pd.read_sql_query(query_string, conn)
```

# 基本操作

## 装包换源-Python

```python
!pip install package_name -i https://pypi.douban.com/simple/ #从指定镜像下载安装工具包，镜像URL可自行修改
```

## 静默安装-Python

```python
!pip install package_name -q
```

#描述性统计信息

## 皮尔森相关系数

```python
import numpy as np 

matrix = np.transpose(np.array(X))
np.corrcoef(matrix[0], matrix[1])[0, 1]

## X: array-like
## https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.corrcoef.html


```

## 移动平均数

```python
import numpy as np

ret = np.cumsum(np.array(X), dtype=float)
ret[w:] = ret[w:] - ret[:-w]
result = ret[w - 1:] / w

## X: array-like
## window: int

```

## 列的分位数

```python
import pandas as pd
## set columns type
my_df['col'] = my_df['col'].astype(np.float64)

## computations for 4 quantiles : quartiles
bins_col = pd.qcut(my_df['col'], 4)
bins_col_label = pd.qcut(my_df['col'], 4).labels
```

## 多重聚合(组数据)

```python
## columns settings
grouped_on = 'col_0'  ## ['col_0', 'col_2'] for multiple columns
aggregated_column = 'col_1'

#### Choice of aggregate functions
### On non-NA values in the group
### - numeric choice :: mean, median, sum, std, var, min, max, prod
### - group choice :: first, last, count
## list of functions to compute
agg_funcs = ['mean', 'max']


## compute aggregate values
aggregated_values = my_df.groupby(grouped_on)[aggregated_columns].agg(agg_funcs)

## get the aggregate of group
aggregated_values.ix[group]
```

## 以列值排序

```python
import pandas as pd
## my_df 是pandas dataframe
my_df['col_0'].value_counts().sort_index()
```

## 用户定义方程(组数据)

```python
## columns settings
grouped_on = ['col_0']
aggregated_columns = ['col_1']

def my_func(my_group_array):
    return my_group_array.min() * my_group_array.count()

### list of functions to compute
agg_funcs = [my_func] ## could be many

## compute aggregate values
aggregated_values = my_df.groupby(grouped_on)[aggregated_columns].agg(agg_funcs)

```

## 标准差

```python
import numpy as np
np.std(np.array(X))

## X: array-like
## https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.std.html
```

## 在聚合的dataframe上使用用户定义方程

```python
## top n in aggregate dataframe
def top_n(group_df, col, n=2):
    bests = group_df[col].value_counts()[:n]
    return bests

## columns settings
grouped_on = 'col_0'
aggregated_column = 'col'

grouped = my_df.groupby(grouped_on)
groups_top_n = grouped.apply(top_n, aggregated_column, n=3)
```

## 所有列(列的数据类型为数值型)

```python
my_df.describe()
```

## 以频率降序排序

```python
import pandas as pd
## my_df 是pandas dataframe
my_df['col_0'].value_counts()
```

## 一些列的单一属性(如最大值， 列的类型为数值型)

```python
my_df["col"].max() ## [["col_0", "col_1"]] 多字段

```

## 最大互信息数

```python
import numpy as np

matrix = np.transpose(np.array(X)).astype(float)
mine = MINE(alpha=0.6, c=15, est="mic_approx")
mic_result = []
for i in matrix[1:]:
    mine.compute_score(t_matrix[0], i)
    mic_result.append(mine.mic())
return mic_result


```

## 平均数

```python
import numpy as np
np.average(np.array(X))

## X: array-like
## https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.average.html
```

## 组数据的基本信息

```python
## columns settings
grouped_on = 'col_0'  ## ['col_0', 'col_1'] for multiple columns
aggregated_column = 'col_1'

#### Choice of aggregate functions
### On non-NA values in the group
### - numeric choice : mean, median, sum, std, var, min, max, prod
### - group choice : first, last, count
### On the group lines
### - size of the group : size
aggregated_values = my_df.groupby(grouped_on)[aggregated_column].mean()
aggregated_values.name = 'mean'

## get the aggregate of group
aggregated_values.ix[group]
```

## 行列数

```python
## 获取当前dataframe的形状
nb_rows = my_df.shape[0]
nb_cols = my_df.shape[1]
```

## 相关性

```python
my_df.corr()
```

## 中位数

```python
import numpy as np
np.median(np.array(X))

## X: array-like
## https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.median.html
```

## 数据组的遍历

```python
## columns settings
grouped_on = 'col_0'  ## ['col_0', 'col_1'] for multiple columns

grouped = my_df.groupby(grouped_on)

i = 0
for group_name, group_dataframe in grouped:
    if i > 10:
        break
    i += 1
    print(i, group_name, group_dataframe.mean())  ### mean on all numerical columns

```

## 协方差

```python
my_df.cov()

```

## 方差

```python
import numpy as np

np.var(np.array(X))

## X: array-like
## https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.var.html
```

## K平均数算法

```python
import numpy as np
from sklearn.cluster import KMeans

k_means = KMeans(k).fit(np.array(X))
result = k_means.labels_
label = result.tolist()
return label, k, k_means.cluster_centers_.tolist(), k_means.inertia_

## k: int, k>=2
## X: array-like
## http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
```

## 所有列的单一属性(如最大值， 列的类型为数值型)

```python
my_df.max()
```

## 获得类别字段的频数

```python
my_df['category'].value_counts()
```

## 查看缺失情况

```python
for col in my_df.columns:
    print("column {} 包含 {} 个缺失值".format(col, my_df[col].isnull().sum()))
```

## 查看字段包含unique值的程度

```python
for col in my_df.columns:
    print("column {} 中共有 {} 个unique value".format(col, my_df[col].nunique()))
```

# 新建列

## 使用复杂的方程设值

```python
import numpy as np
def complex_formula(col0_value, col1_value):
    return "%s (%s)" % (col0_value, col1_value)

my_df['new_col'] = np.vectorize(complex_formula)(my_df['col_0'], my_df['col_1'])
```

## 标准聚合(使用groupby)

```python
## columns settings
grouped_on = 'col_1'
aggregated_column = 'col_0'

#### Choice of aggregate functions
### On non-NA values in the group
### - numeric choice : mean, median, sum, std, var, min, max, prod
### - group choice : first, last, count
my_df['aggregate_values_on_col'] = my_df.groupby(grouped_on)[aggregated_column].transform(lambda v: v.mean())

```

## 使用自定义方程设值

```python
def to_log(v):
    try:
        return log(v)
    except:
        return np.nan
my_df['new_col'] = my_df['col_0'].map(to_log)
```

## 用户定义的聚合方程(使用groupby)

```python
def zscore(x):
    return (x - x.mean()) / x.std()
   
my_df['zscore_col'] = my_df.groupby(grouped_on)[aggregated_column].transform(zscore)
```

# 缺失值处理

## 用一个值填补多列的缺失值

```python
my_df[['col_0', 'col_1']] = my_df[['col_0', 'col_1']].fillna(value)
```

## 用一个值填补一列的缺失值

```python
my_df['col'] = my_df['col'].fillna(value)
```

## 去除在指定列中带有缺失值的数据条目

```python
cols = ['col_0', 'col_1']
records_without_nas_in_cols = my_df.dropna(subset=cols)
```

## 用最后一个或缺失值的下一个数据填补缺失值

```python
## - ffill : propagate last valid observation forward to next valid
## - backfill : use NEXT valid observation to fill gap
my_df['col'] = my_df['col'].fillna(method='ffill')
```

## 用一个由聚合得出的值填补缺失值

```python
grouped_on = 'col_0' ## ['col_1', 'col_1'] ## for multiple columns

#### Choice of aggregate functions
### On non-NA values in the group
### - numeric choice : mean, median, sum, std, var, min, max, prod
### - group choice : first, last, count
def filling_function(v):
    return v.fillna(v.mean())
                    
my_df['col'] = my_df.groupby(grouped_on)['col'].transform(filling_function)
```

## 以列为单位审查数据集

```python
my_df.info()
```

## 去除所有任何缺失值的数据条目

```python
records_without_nas = my_df.dropna()
```

# Pandas基本操作

## 计算频率来转置表

```python
freqs = my_df.pivot_table(
    rows=["make"],
    cols=["fuel_type", "aspiration"],
    margins=True     ## add subtotals on rows and cols
)
```

## 水平连接数据

```python
import pandas as pd
two_dfs_hconcat = pd.concat([my_df, my_df2], axis=1)
```

## 将dataframe的index重置成标准值

```python
my_df_no_index = my_df.reset_index()
## my_df_no_index.index is now [0, 1, 2, ...]
```

## 以元组(tuple)形式按行写入数据

```python
import pandas as pd
py_recipe_output = pd.read_csv("data.csv")
writer = py_recipe_output.get_writer()

## t is of the form :
##   (value0, value1, ...)

for t in data_to_write:
    writer.write_tuple(t)
```

## 数据重构

```python
#Pivot
df3 = df2.pivot( index = 'Date', 
columns = 'Type', values = 'Value') #行变列

#Pivot Table
df4 = pd.pivot_table( df2,
values='Value',
index = 'Date', 
columns='Type'] #行变列

#Stack/Unstack
stacked = df5.stack( ) 
stacked.unstacked( )

#Melt
pd.melt(df2, id_vars=["Date"], 
value_vars=["Type","Value"], 
value_name="Observations") 
#将列变行

```

## 用数据的位置获取数据

```python
my_record = my_df.iloc[1]  ## !! get the second records, positions start at 0
```

## 从列新建虚数据

```python
import pandas as pd
dummified_cols = pd.get_dummies(my_df['col']
    ## dummy_na=True ## to include NaN values
    )
```

## 合并有相同名称的列的两个dataframe

```python
import pandas as pd
## my_df 是一个pandas dataframe
merged = my_df.merge(my_df2,
on='col', ## ['col_0', 'col_1'] for many
how="inner",
## suffixes=("_from_my_df", "_from_my_df2"))
```

## 合并两个dataframe没有相同名称的列

```python
## 每次只合并两个dataframe
## 合并方式 : 'left', 'right', 'outer', 'inner'
import pandas as pd
## my_df 是一个pandas dataframe
merged = my_df.merge(my_df2,
left_on=['col_0', 'col_1'],
right_on=['col_2', 'col_3'],
## suffixes=("_from_my_df", "_from_my_df2")
how="inner")
```

## 分组数据

```python
#aggregation
df2.groupby(
by=['Data', 'Type']).mean()
df4.groupby(level=0).sum()
df4.groupby(level=0).agg
({'a':lama x:sum(x)/len(x), 
'b':np.sum})

#Transformation
customSum=lamda x:(x+x%2)
df4.groupby
(level=0).transform(customSum)
```

## 使用目标值的位置定位替换值

```python
import pandas as pd
## my_df 是一个pandas dataframe
my_df[]'col'].iloc[0] = new_value  ## replacement for first record
```

## 分簇

```python
import pandas as pd
## Set columns type
my_df['col'] = my_df['col'].astype(np.float64)

## Computations
bins = [0, 100, 1000, 10000, 100000] ## 5 binned, labeled 0,1,2,3,4
bins_col = pd.cut(my_df['col'], bins)
bins_col_label = pd.cut(my_df['col'], bins).labels
```

## 获取元组(tuple)的迭代器

```python
import pandas as pd
my_dataset = pd.read_csv("path_to_dataset")

i = 0
for my_row_as_tuple in my_dataset.iter_tuples():
    if i > 10:
        break
    i += 1
    print (my_row_as_tuple)
```

## 获取文档

```python
## Everywhere
print(my_df.__doc__)
print(my_df.sort.__doc__)

## When using notebook : append a '?' to get help
my_df?
my_df.sort?
```

## 复制数据

```python
s3.unique() 
#返回唯一值
df2.duplicated('Tyepe') 
#检查重复值
df2.drop_duplicates(
'Type', keep=last'') 
#丢弃重复值
df.index.duplicated() 
#检查索引重复
```

## 重命名所有列

```python
my_df.columns = ['new_col_0', 'new_col_1']  ## needs the right number of columns
```

## 设置聚合方程进行交叉制表

```python
## 交叉制表是统计学中一个将分类数据聚合成列联表的操作
import pandas as pd
## my_df 是一个pandas dataframe
stats =  pd.crosstab(
    rows=my_df["make"],
    cols=[my_df["fuel_type"], my_df["aspiration"]],
    values=my_df["horsepower"],
    aggfunc='max',   ## aggregation function
    margins=True     ## add subtotals on rows and cols
)
```

## 高级索引

```python
#basic
df3.loc[:, (df3>1).any()] 
#选列 其中任意元素>1
df3.loc[:, (df3>1).all()] 
#选列 其中所有>1
df3.loc[:, df3.isnull().any()] 
#选列 其中含空
df3.loc[:,df3.notnull().all()] 
#选列 其中不含空

df[(df.Country.isin(df2.Type))] 
#寻找相同元素
df3.filter(items=["a","b"]) 
#根据值筛选
df.select(lamda x: not x%5) 
#选特定元素

s.where(s>0) 
#数据分子集

df6.query('second >first') 
#query 数据结构


#selecting
df.set_index('Country') 
#设置索引
df4 = df.reset_index() 
#重置索引
df =df.rename(index = str, 
columns={"Country":"cntry", 
"Capital":"cptl","Population":"ppltn"}) 
#重命名数据结构

#Reindexing
s2 = s.reindex(['a', 'c', 'd', 'e', 'b'])
#Forward Filling
df.reindex(range(4), 
method='ffill')
#Backward Filling
s3 = s.reindex(range(5), 
method='bfill')

#MultiIndexing
arrays = [np.array([1,2,3]), 
np.array([5,4,3])]
df5 = pd.DataFrame(
np.random.rand(3,2), index = arrays)
tuples = list(zip( *arrays))
index = pd.MultiIndex.from_tuples(
tuples, names=['first', 'second'])
df6 = pd.DataFrame(
np.random.rand(3,2), index = index)
df2.set_index(["Data","Type"])
```

## 使用某种法则替换列中的值

```python
import numpy as np
rules = {
    value: value1,
    value2: value3,
    'Invalid': np.nan  ## replace by an true invalid value
}

my_df['col'] = my_df['col'].map(rules)
```

## 遍历数据字典

```python
import pandas as pd
my_dataset = pd.read_csv("path_to_dataset")

i = 0
for my_row_as_dict in my_dataset.iter_rows():
    if i > 10:
        break
    i += 1
    print my_row_as_dict
```

## 垂直连接数据

```python
two_dfs_vconcat = my_df1.append(my_df2)
```

## 计算频率进行交叉制表

```python
## 交叉制表是统计学中一个将分类数据聚合成列联表的操作
import pandas as pd
## my_df 是一个pandas dataframe
freqs = pd.crosstab(
    rows=my_df["make"],
    cols=[my_df["fuel_type"], my_df["aspiration"]]
)
```

## 迭代

```python
df.iteritems( ) #列索引 
df.iterrows( ) #行索引

```

## 结合数据

```python
#Merge
pd.merge(data1, data2, 
how='left', on='X1')
pd.merge(data1, data2,
how='right', on='X1')
pd.merge(data1, data2, 
how='inner', on='X1')
pd.merge(data1, data2, 
how='outer', on='X1')

#join
data1.join(data2, how='right')

#concatenate
#vertical
s.append(s2)
#horizontal/vertical
pd.concat([s,s2], axis=1,
 keys=['One','Two'])
pd.concat([data1, data2], 
axis=1, join='inner')
```

## 用index获取数据

```python
my_record = my_df.loc[label] ## !! If the label in the index defines a unique record
```

## 使用目标值的label定位替换值

```python
import pandas as pd
## my_df 是一个pandas dataframe
my_df['col'].loc[my_label] = new_value
```

## 获取多列数据

```python
cols_names = ['col_0', 'col_1']
my_cols_df = my_df[cols_names]
```

## 使用函数新增列

```python
def remove_minus_sign(v):
    return str.replace('-', ' ', max=2)

my_df['col'] = my_df['col'].map(remove_minus_sign)
```

## 基本的数据排序

```python
my_df = my_df.sort('col', ascending=False)
```

## 使用条件定位替换值

```python
import pandas as pd
## my_df_orig 是一个pandas dataframe
cond = (my_df_orig['col'] != value)
my_df_orig['col'][cond] = "other_value"
```

## 从dataframe随机抽取N行数据

```python
import random
n = 10
sample_rows_index = random.sample(range(len(my_df)), 10)
my_sample = my_df.take(rows)
my_sample_complementary = my_df.drop(rows)
```

## 按列里的值过滤或去除数据

```python
cond = (my_df['col'] == value)

## multiple values
## cond = my_df['col'].isin([value1, value2])

## null value
## cond = my_df['col'].isnull()

## exclude (negate condition)
## cond = ~cond

my_records = my_df[cond]
```

## 对多列数据排序

```python
my_df = my_df.sort(['col_0', 'col_1'], ascending=[True, False])
```

## 将dataframe的index设为列名

```python
my_df_with_col_index = my_df.set_index("col")
## my_df_col_index.index is [record0_col_val, record1_col_val, ...]
```

## 将数据集加载成多个dataframe

```python
import pandas as pd
my_dataset = pd.read_csv("data.csv")

for partial_dataframe in my_dataset.iter_dataframes(chunksize=2000):
    ## Insert here applicative logic on each partial dataframe.
    pass
```

## 以字典(dict)形式按行写入数据

```python
import pandas as pd
py_recipe_output = pd.read_csv("data.csv")
writer = py_recipe_output.get_writer()

## d is of the form :
##   {'col_0': value0, 'col_1': value1, ...}

for d in data_to_write:
    writer.write_row_dict(r)
```

## 使用dataframe的index新建列

```python
my_df['new_col'] = my_df.index
```

## 可视化

```python
import Matplotlib.pyplot as plt
s.plot()
plt.show()

df2.plot()
plt.show()
```

## 设置聚合方程转置表

```python
stats = my_df.pivot_table(
    rows=["make"],
    cols=["fuel_type", "aspiration"],
    values=["horsepower"],
    aggfunc='max',   ## aggregation function
    margins=True     ## add subtotals on rows and cols
)
```

## 使用boolean formula过滤或去除数据

```python
import pandas as pd
## single value
cond1 = (my_df['col_0'] == value)
cond2 = (my_df['col_1'].isin([value1, value2]))
## boolean operators :
## - negation : ~  (tilde)
## - or : |
## - and : &
cond = (cond1 | cond2)
my_records = my_df[cond]
```

## 日期

```python
df2['Date'] = pd.to_datetime
(df2['Data'])
df2['Date'] = pd.date_range
('2000-1-1', periods=6, freq ='M')
dates = [datetime(2012,5,1), 
datetime(2012,5,2)]
index = pd.DatetimeIndex(dates)
index = pd.date_range
(datetime(2012,2,1)), end, freq='BM'
```

## 重命名指定列

```python
my_df = my_df.rename(columns = {'col_1':'new_name_1', 'col_2':'new_name_2'})

```

# Numpy基本操作

## array处理

```python
import numpy as np

#Transposing Array
I = np.transpose(b) #转置矩阵
i.T #转置矩阵

#Changing Array Shape
b.ravel() #降为一维数组
g.reshape(3,-2) #重组

#Adding/Removing Elements
h.resize((2,6)) #返回shape(2,6)
np.append(h,g) #添加
np.insert(a,1,5) #插入
np.delete(a,[1]) #删除

#Combining Arrays
np.concatenate((a,d), axis=0) #连结
np.vstack((a,b)) #垂直堆叠
np.r_[e,f] #垂直堆叠
np.hstack((e,f)) #水平堆叠
np.column_stack((a,d)) #创建水平堆叠
np.c_[a,d] ##创建水平堆叠

#splitting arrays
np.hsplit(a,3) #水平分离
np.vsplit(c,2) #垂直分离

```

## 创建array

```python
import numpy as np
a = np.array([1,2,3])   #创建数组
b = np.array([(1.5,2,3), (4,5,6)], 
dtype=float)
c = np.array([(1.5,2,3), (4,5,6)], 
[(3,2,1), (4,5,6) ] ], dtype=float)

np.zeros((3,4))  #创建0数组
np.ones((2,3,4), dtype=np.int16)  #创建1数组
d = np.arrange(10,25,5)  #创建相同步数数组
np.linspace(0,2,9)  #创建等差数组

e = np.full((2,2), 7) #创建常数数组
f = np.eye(2) #创建2x2矩阵
np.random.random((2,2)) #创建随机数组
np.empty((3,2)) #创建空数组

```

## 排序array

```python
#对数组进行排序
a.sort()
c.sort(axis=0)
```

## 索引

```python
import numpy as np
#subsetting
a[2] #选取数组第三个元素
b[1,2] #选取2行3列元素

#slicing
a[0:2] #选1到3元素
b[0:2,1] #选1到2行的2列元素
b[:1] #选所有1行的元素
c[1,...] #c[1,:,:]
a[ : :-1]  #反转数组

#Boolean Indexing
a[a<2] #选取数组中元素<2的

#Fancy Indexing
b[[1,0,1,0], [0,1,2,0]]
#选取[1,0],[0,1],[1,2],[0,0]
b[[1,0,1,0][:, [0,1,2,0]]] 
#选取矩阵的一部分


```

## 基本运算

```python
import numpy as np

#arithmetic operation算术运算
g = a - b
np.subtract(a,b) #减法
b+a
np.add(b,a) #加法
a / b
np.divide(a,b) #除法
a * b
np.multiply(a,b) #乘法
np.exp(b) #指数
np.sqrt(b) #开方
np.sin(a) #sin函数
np.cos(b) #cos函数
np.log(a) #log函数
e.dot(f) #内积

#Comparison比较
a == b #元素
a < 2 #元素
np.array_equal(a,b) #数组 

#Aggregate Functions 函数
a.sum() #求和
b.min() #最小值
b.max(axis=0) #最大值数组列
b.cumsum(axis=1) #元素累加和
a.mean() #平均值
b.median() #中位数
a.corrcoef() #相关系数
np.std(b) #标准差



```

## 检查array

```python
a.shape #数组维度
len(a) #数组长度
b.ndim #数组维度数量
e.size #数组元素数量
b.dtype #元素数据类型
b.dtype.name #数据类型名
b.astype(int) #改变数组类型

#asking for help更多信息
np.info(np.ndarray.dtype)
```

## 输出array

```python
import numpy as np
print(my_array) #打印数组

#saving &Loading on disk保存到磁盘
np.save('my_array', a) 
np.savez('array.npz', a, b)
np.load('my_array.npy')

#saving &Loading Text files保存到文件
np.loadtxt("my file.txt")
np.genfromtxt("my_file.csv", delimiter=',')
np.savetxt("marry.txt", a, delimiter=" ")

```

## 数据类型

```python
np.int64 #64位整数
np.float32 #标准双精度浮点
np.complex #复杂树已浮点128为代表
np.bool #true&false
np.object #python object
np.string_ #固定长度字符串
np.unicode_ #固定长度统一码

```

## 复制array

```python
#复制数组
h = a.view() 
np.copy(a)
h = a.copy()
```

# Matplotlib基本操作

## 数据录入

```python
import numpy as np
#1D data
x = np.linspace(0, 10, 100)
y = np.cos(x)
z = np.sin(x)

#2D data or images
data = 2* np.random.random((10,10))
data2 = 3 * np.random.random((10,10))
Y, X = np.mgrid[-3:3:100j, -3:3:100j]
U = -1-X**2+Y
V = 1+X-Y**2
from Matplotlib.cbook 
import get_sample_data
img = np.load(get_sample_data
('axes_grid/bivariate_normal.npy'))
```

## 显示与保存

```python
#save plot
plt.savefig('foo.png')
plt.savefig('foo.png', transparent=True)

#show plot
plt.show()

#close &clear
plt.cla() #clear an axis
plt.clf() #clear the entire figure
plt.close() #close a window

```

## 图像设置

```python
import matplotli.pyplot as plt

#figure
fig = plt.figure( )
fig2 = plt.figure( figsize = 
plt.figaspect(2,0) )

#axes
fig.add_axes( )
ax1 = fig.add_subplot( 221 ) 
#row-col-num
ax3 = fig.add_subplot( 212 )
fig3, axes = plt.subplots(
nrows=2, ncols=2)
fig4, axes2 = plt.subplots( ncols=3 )
```

## 自定义

```python
#colors
plt.plot(x, x, x, x**2, x, x**3)
ax.plot(x, y, alpha = 0.4)
ax.plot(x, y, c='k')
fig.colorbar(im, orientation='horizontal')
im = ax.imshow(img, cmap='seismic')

#markers
fig, ax = plt.subplots()
ax.scatter(x,y,marker='.')
ax.plot(x,y,marker='o')

#linestyles
plt.plot(x,y,linewidth=4.0)
plt.plot(x,y,ls='solid')
plt.plot(x,y,ls='--')
plt.plot(x,y,'--',x**2,y**2,'-.')
plt.setp(lines, color='r',linewidth=4.0)

#Text & Annotations
ax.text(1,-2.1, 
'Example Graph', style='italic')
ax.annotate("Sine", xy=(8,0),
xycoords='data',xytext=(10.5,0), 
textcoords='data', arrowprops=
dict(arrowstyle="->",connectionstyle="arc3"),)

#mathtext
plt.title(r'$sigma_i=15$',
 fontsize=20)

#Limits,Legends&Layouts
#Limites&Autoscaling
ax.margins(x=0.0, y=0.1) 
#Add padding to a plot
ax.axis('equal') 
#Set the aspect ratio of the plot to 1
ax.set(xlim=[0,10.5],ylim=[-1.5,1.5]) 
#Set limits for x- and y-axis
ax.set_xlim(0,10.5) 
#Set limits for x-axis

#Legends
ax.set(title='An Example Axes',
ylabel='Y-Axis',xlabel='X-Axis') 
#Set a title and x- and y-axis labels
ax.legend(loc='best') 
#No overlapping plot elements

#Ticks
ax.xaxis.set(ticks=range(1,5),
ticklabels=[3,100,-12,"foo"]) 
#Manually set x-ticks
ax.tick_params(axis='y',
direction ='inout',length=10) 
#Make y-ticks longer and go in and out

#Subplot Spacing
fig3.subplots_adjust(wspace=0.5, 
hspace=0.3,left=0.125,right=0.9,
top=0.9,bottom=0.1) 
#Adjust the spacing between subplots
fig.tight_layout() 
#Fit subplots in to the figure area

#Axis Spines
ax1.spines['top'].set_visible(False) 
#Make the top axis line for a plot invisible
ax1.spines['bottom'].set_position(('outward',10)) 
#Move the bottom axis line outward
```

## 图像绘制

```python
#1D data
fig, ax = plt.subplots()
lines = ax.plot(x,y) 
#Draw points with lines or makers connecting them
ax.scatter(x,y)
#Draw unconnected points, scaled or colored
axes[0,0].bar([1,2,3],[3,4,5]) 
#Plot vertical rectangles
axes[1,0].barh([0.5,1,2.5], [0,1,2]) 
#Plot horizontal rectangles
axes[1,1].axhline(0.45) 
#Draw a horizontal line across axes
axes[0,1].axvline(0.65) 
#Draw a vertical line across axes
ax.fill(x,y,color='blue') 
#Draw filled polygons
ax.fill_between(x,y,color='yellow') 
#Fill between y-values and o

#2D data or images
fig, ax = plt.subplots( ) 
#Colormapped or RGB arrays
im = ax. imshow(img, cmap='gist_earth', 
interpolation='nearest',vmin=-2,vmax=2)
axes2[0].pcolor(data2) 
#Pseudocolor plot of 2D array
axes2[0].pcolormesh(data) 
#Pseudocolor plot of 2D array
CS = plt.contour(Y,X,U)
#Plot contours
axes2[2].contourf(data1) 
#Plot filled contours
axes2[2] = ax.clabel(CS)
#Label a contour plot

#Vector Field
axes[0,1].arrow(0,0,0.5,0.5) 
#Add an arrow to the axes
axes[1,1].quiver(y,z) 
#Plot a 2D field of arrows
axes[0,1].streamplot(X,Y,U,V) 
#plot a 2D field of arrows

#Data Distributions
ax1.hist(y) #Plot a histogram
ax3.boxplot(y) 
#make a box and whisker plot
ax3.violinplot(z) 
#make a violin plot

```

# 以pandas dataframe制图

## 所有列(分隔)

```python
my_df.plot(kind='line', subplots=True, figsize=(8,8))

```

## 所有列(交叉)

```python
my_df.plot(kind='line', alpha=0.3)
```

## 散点图

```python
my_df.plot(kind='scatter', x='col1', y='col2',
    ## c='col_for_color', s=my_df['col_for_size']*10
    );
```

## 单一列

```python
my_df['col_name'].plot(kind='hist', bins=100)
```

# Seaborn基本操作

## 数据导入

```python
import pandas as pd
import numpy as np
uniform_data = np.random.rand(10,12)
data = pd.DataFrame( ( 'x': np.array(1, 101), 
'y': np.random.normal( 0,4,100 ) ) )

#内部数据集
titanic = sns.load_dataset( "titanic" ) 
iris = sns.load_dataset( "iris" )
```

## 图像绘制

```python
#axis grids
g = sns.FacetGrid(titanic, 
col = "survived", row = "sex" ) 
#subplot grid
g = g.map( pot.hist, "age")

sns.factorplot( x = "pclass",
y = "survived", hue = "sex", 
data = titanic) ## categorical plot
sns. lmplot( x = "sepal_width",
 y = "sepal_length", hue = "species", 
data = iris) 
#plot data and regression model

h = sns.PairGrid( iris ) 
#subplot grid for plotting pairwise relationships
h = h.map( pot.scatter) 
sns.pairplot( iris ) 
#plot pairwise bivariate distributions
i = sns.JointGrid( x = "x",
 y = "y", data= data )
 #grid for bivariate plot with marginal
i = i.plot( sns.regplot, sns.distplot )
sns.jointplot( "sepal_length", 
"sepal_width", data = iris, kind = 'kde')
#plot bivariate distribution


#Categorical Plots

#Scatterplot
sns.stripplot( x = "species", 
y = "petal_length", data = iris)
sns.swarmplot( x = "species", 
y = "petal_length", data = iris)

#bar chart
sns.barplot( x = "sex", y = "survived",
 hue = "class", data = titanic)

#count plot
sns.countplot( x = "deck", 
data = titanic, palette = "Greens_d")

#Point Plot
sns.pointplot( x = "class", y = "survived",
hue = "sex", data = titanic, 
palette ={ "male": "g", "female": "m"}, 
markers = [ "^","o"], linestyles = ["-","--"])

#Boxplot
sns.boxplot( x = "alive", y = "age",
hue = "adult_male", data = titanic)
sns.boxplot( data = iris, orient = "h")

#Violinplot
sns.violinplot( x = "age", y = "sex",
 hue = "survived", data = titanic ) 


#regression plots
sns.regplot(x="sepal_width", 
y ="sepal_length", data=iris, ax=ax)

#distribution plots
plot = sns.displot(data.y, 
kde=False, color="b")

#Matrix plots
sns.heatmap(uniform_data, 
vmin=0, vmax=1)

```

## 自定义

```python
#Axisgrid Objects
g.despine( left = True) 
#remove left spine
g.set_ylabels( "Survived") 
#set the labels of the y-axis
g.set_xticklabels( rotation = 45) 
#set the tick labels for x
g.set_axis_labels( "Survived", "Sex")
 #set the axis labels
h.set( xlim = (0,5), ylim = (0,5), 
xticks = [0,2.5,5], yticks =[0,2.5,5] )
#set the limit and ticks of the x- and y-axis

#Plot
plt.title( "A title" ) 
#add plot title
plt.ylabel( "Survived" ) 
#adjust the label of the y-axis
plt.xlabel( "Sex" ) 
#adjust the label of the x-axis
plt.ylim(0,100)
 #adjust the limits of the y-axis
plt.xlim(0,10) 
#adjust the limits of the x-axis
pot.setp(ax, yticks =[0,5]) 
#adjust a plot property
plt.tight_layout() 
#adjust subplot params

```

## 显示与保存

```python
plt.show( ) #显示图像
plt.saveflg( "foo.png" ) #存储
plt.saveflg( "foo.png", transparent = True) 
#存储透明图像

#关闭与清除
plt.cla( ) #清除axis
plt.clf( ) #清除整个图像
plt.close( ) #关闭窗口

```

## 图像设置

```python
#创建图像绘制窗口
f, ax = ply.subplots( figsize = ( 5,6 ) )

#seaborn style
sis.set( ) #重置
sns.set_style( "whitegrid" )
#设置matplotlib parameters
sns.set_style( "ticks",
( "xtick.major.size": 8, 
"ytick.major.size": 8 ) )
sns.axes_style( "whitegrid" ) 
#返回一个参数指引

#context functions
sns.set_context( "talk" ) 
#设置内容为“talk”
sns.set_context( "notebook",
 font_scale = 1.5, 
rc = ( "lines.linewidth": 2.5 ) ) 
#设置内容为“notebook”

#color palette
sns.set_palette( "husl",3 )
 #定义color palette
sns.color_palette( "husl" )
flatui = ( “#9b59b6”,"#3498db",
"#95a5a6","#e74c3c",
"#34495e","#2ecc71")
sns.set_palette( flatui )
#定义自己的color palette

```

# Bokeh基本操作

## 输出

```python
#Notebook: 
from bokeh.io 
import output_notebook, show
output_notebook( )

#HTML:

#Standalone HTML 
from bokeh.embed 
import file_html; 
from bokeh.resources 
import CDN; 
html=file_html(p,CDN,"my_plot");
from bokeh.io 
import output_file,show; 
output_file('my_bar_chart.html',
mode='cdn')

#Components:
from bokeh.embed 
import components; 
script, div=components(p)

#PNG: 
from bokeh.io import export_png;
export_png(p,filename="plot.png")

#SVG:
from bokeh.io import export_svgs;
p.output_backend="svg";
export_svgs(p,filename="plot.svg")
```

## 数据导入

```python
import numpy as np
import pandas as pd
df=pd.DataFrame(
np.array([[ 33.9, 4, 65, 'US'], 
[32.4,4, 66, 'Asia'], 
[21.4, 4,109, 'Europe'] ] ), 
columns = [ 'mpg', 'cyl', 'hp', 'origin'],
index= [ 'Toyota', 'Fiat', 'Volvo'] )

from bokeh.models
import ColumnDataSource 
cds_df=ColumDataSource(df)
```

## 图像绘制

```python
from bokeh.plotting import figure 
p1 = figure( plot_width = 300, 
tools = 'pan, box_zoom' )  
p2 = figure( plot_width = 300, 
plot_height = 300, 
x_range = (0,8), y_range = (0,8) )
p3 = figure( )
```

## 可视化

```python
#Glyphs

#Scatter markers 
p1.circle( np.array( [1,2,3]), 
np.array([3,2,1]), fill_color = 'white')
p2.square(np.array([1.5,3.5,5.5]), 
[1,4,3], color = 'blue', size = 1)

#Line Glyphs: 
p1.line( np.array( [1,2,3,4], 
[3,4,5,6], ), line_width = 2)
p2.multi_line(pd.DataFrame(),
pd.DataFrame(),color='blue')



#Customized Glyphs

#Selection and Non-Selection Glyphs
p = figure( tools = 'box_select')
p.circle( 'mpg', 'cyl', source = cds_df, 
selection_color = 'red', 
nonselection_alpha = 0.1)

#Hover Glyphs: 
from bokeh.models import HoverTool 
hover = HoverTool( tooltips = None, 
mode = 'vline' )

#Colormapping:
from bokeh.models 
import CategoricalColorMapper
color_mapper = CategoricalColorMapper
( factors = ['US', 'Asia', 'Europe'], 
palette = ['blue', 'red', 'green'])
p3.circle( 'mpg', 'cyl', source = cds_df, 
color = dict( field = 'orign',
transform = color_mapper), legend = 'Orign')



#Legend Location

#Inside Plot Area: 
p.legend.location = 'bottom_left'

#Outside Plot Area: 
from bokeh.models import Legend
r1 = p2.asterisk( 
np.array( [1,2,3]), np.array([3,2,1]))
r2 = p2.line([1,2,3,4], [3,4,5,6])
Legend = Legend( items = [("One",
 [p1,r1]]), ("Two", [r2])], location = (0,-30))
p.add_layout( legend,'right')

#Legend Orientation
p.legend.orientation = "horizontal"
p.legend.orientation = "vertical"

#Legend Background &Border
p.legend.border_line_color = "navy"
p.legend.background_fill_color = "white"



#Rows & Columns Layout

#Rows:
from bokeh.layouts import row
layout = row( p1,p2,p3)

#Columns 
from bokeh.layouts import columns
layout = row( column ( p1,p2), p3)

#Nesting Rows&Columns 
layout = row( column (p1,p2), p3)

#Grid Layout:
from bokeh.layouts import gridplot
row1 = [p1, p2]
row2 = [p3]
layout = gridplot ( [ [p1,p2], [p3]])

#Tabbed Layout
from bokeh.models.widgets import Panel, Tabs
tab1 = Panel ( child = p1, title = "tab1" )
tab2 = Panel ( child = p1, title = "tab2" )
layout = Tabs( tabs = [tab1,tab2])



#Linked Plots

#linked Axes
p2.x_range = p1.x_range
p2.y_range = p1.y_range

#Linked Brushing
p4 = figure( plot_width = 100, 
tools = 'box_select, lasso_select')
p4.circle( 'mpg', 'cyl', source = cds_df)
p5 = figure( plot_width = 200, 
tools = 'box_select, lasso_select')
p5.circle( 'mpg', 'hp', source = cds_df)
layout = row( p4, p5)
```

## 显示与保存

```python
show(p1)
save(p1)
show(layout)
save(layout)
```

# seaborn可视化

## swarmplot

```python
import seaborn as sns
%matplotlib inline
sns.swarmplot(x='species',y='petal_length',data=my_data) ## 以iris数据集为例
```

## pairplot

```python
import seaborn as sns
sns.pairplot(iris,hue='Species')
```

## countplot

```python
import seaborn as sns;titanic= sns.load_dataset('titanic')
sns.countplot(y="deck", hue="class", data=titanic, palette="Greens_d") ## 以titanic数据集为例
```

## violinplot

```python
import seaborn as sns;iris = sns.load_dataset('iris')
sns.violinplot(x='species',y='sepal_length',data=iris) ## 以iris数据集为例
```

## pointplot

```python
import seaborn as sns; iris = sns.load_dataset('iris')
sns.pointplot(x='species',y='petal_width',data=iris) ## 以iris数据集为例
```

## boxplot

```python
import seaborn as sns
sns.boxplot(x='species',y='Petal.Length',data=iris) ## 以iris数据集为例
```

## stripplot

```python
import seaborn as sns; iris = sns.load_dataset('iris')
sns.stripplot(x='species',y='sepal_length',data=iris)
```

## jointplot

```python
import seaborn as sns; sns.set(style="ticks", color_codes=True);tips = sns.load_dataset("tips")
g = sns.JointGrid(x="total_bill", y="tip", data=tips) ## 以tips数据集为例
g = sns.JointGrid(x="total_bill", y="tip", data=tips)
g.plot(sns.regplot, sns.distplot)
```

## heatmap

```python
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()
uniform_data = np.random.rand(10, 12)
sns.heatmap(uniform_data)
```

## tsplot

```python
import numpy as np; np.random.seed(22)
import seaborn as sns; sns.set(color_codes=True)
x = np.linspace(0, 15, 31);data = np.sin(x) + np.random.rand(10, 31) + np.random.randn(10, 1)
sns.tsplot(data=data)
```

## distplot

```python
import seaborn as sns, numpy as np,matplotlib.pyplot as plt
%matplotlib inline
sns.set(); np.random.seed(0)
x = np.random.randn(100)
sns.distplot(x)
```

## kdeplot

```python
import numpy as np; np.random.seed(10)
import seaborn as sns; sns.set(color_codes=True)
mean, cov = [0, 2], [(1, .5), (.5, 1)]
x, y = np.random.multivariate_normal(mean, cov, size=50).T
sns.kdeplot(x,y)
```

## barplot

```python
import seaborn as sns; titanic=sns.load_dataset('titanic')
sns.barplot(x="sex", y="survived", hue="class", data=titanic)
```

# 数据清洗

## 去除所有重复数据

```python
import pandas as pd
## my_df 是一个pandas dataframe
unique_records = my_df.drop_duplicates()
```

## 归一化

```python
import numpy as np
from sklearn import preprocessing

raw_result = preprocessing.Normalizer().fit_transform(np.array(X))
result = raw_result.tolist()

## X: array-like
## http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html
```

## 多项式数据变换

```python
import numpy as np
from sklearn import preprocessing

raw_result = preprocessing.PolynomialFeatures().fit_transform(np.array(X))
result = raw_result.tolist()

## X: array-like
## http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
```

## 去除特定列中的重复数据

```python
import pandas as pd
## my_df 是一个pandas dataframe
unique_records_for_cols = my_df.drop_duplicates(cols=['col_1', 'col_0'])
```

## 二值化

```python
import numpy as np
from sklearn import preprocessing

raw_result = preprocessing.Binarizer(threshold=t).fit_transform(np.array(X))
result = raw_result.tolist()

## X: array-like
## t: float
## http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer.html
```

## 类别转数字

```python
## Label1 和 Label2 表示 类别
target = pd.Series(map(lambda x: dict(Label1=1, Label2=0)[x], my_df.target_col.tolist()), my_df.index)
my_df.target_col = target
```

## 特征缩放

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

# 无纲量化

## 正态分布

```python
import numpy as np
from sklearn import preprocessing

raw_result = preprocessing.StandardScaler().fit_transform(np.array(X))
result = raw_result.tolist()

## X: array-like
## http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
```

## 伯努利分布

```python
import numpy as np
from sklearn import preprocessing

raw_result = preprocessing.MinMaxScaler().fit_transform(np.array(X))
result = raw_result.tolist()

## X: array-like
## http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
```

# 基础维度变化

## 用拆堆栈进行变化(long format -> wide format)

```python
pivoted = base.unstack(level=1)
```

## 用栈变化(wide format -> long format)

```python
stacked = pivoted.stack(dropna=True)
```

## 使用melt降维(wide format -> long format)

```python
## melt是一种降维的方法
import pandas as pd
rows = pd.melt(pivoted, id_vars=["type"])
```

## 转置

```python
pivoted = rows.pivot(index="type", columns="make", values="qty")
```

# 特征选取

## 递归特征消除法

```python
import numpy as np
from sklearn import feature_selection
from sklearn.linear_model import LogisticRegression

matrix = np.array(X)
target = np.array(target)
temp = feature_selection.RFE(estimator=LogisticRegression(), n_features_to_select=n_features).fit(matrix, target)
scores = temp.ranking_.tolist()
indx = temp.support_.tolist()
result = temp.transform(matrix).tolist()
return scores, indx, result

## X: array-like
## target: array-like
## n-features: int
## http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
```

## 相关系数选择法

```python
import numpy as np
from sklearn import feature_selection
from sklearn.feature_selection import chi2

matrix = np.array(X)
target = np.array(target)
temp = feature_selection.SelectKBest(lambda X, Y: np.array(list(map(lambda x: abs(pearsonr(x, Y)[0]), X.T))), k=k).fit(matrix, target)
scores = temp.scores_.tolist()
indx = temp.get_support().tolist()
result = temp.transform(matrix).tolist()
return scores, indx, result

## X: array-like
## target: array-like
## k: int
## http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
```

## 方差选择

```python
import numpy as np
from sklearn import feature_selection

matrix = np.array(X)
temp = feature_selection.VarianceThreshold(threshold=t).fit(matrix)
scores = [np.var(el) for el in matrix.T]
indx = temp.get_support().tolist()
result = temp.transform(matrix).tolist()
return scores, indx, result

## X: array-like
## t: float
## http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html
```

## 方差选择

```python
import numpy as np
from sklearn import feature_selection

matrix = np.array(X)
temp = feature_selection.VarianceThreshold(threshold=t).fit(matrix)
scores = [np.var(el) for el in matrix.T]
indx = temp.get_support().tolist()
result = temp.transform(matrix).tolist()
return scores, indx, result

## X: array-like
## t: float
## http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html
```

## 互信息选择法

```python
from minepy import MINE
import numpy as np
from sklearn import feature_selection

matrix = np.array(X)
target = np.array(target)
def mic(x, y):
    m = MINE()
    m.compute_score(x, y)
    return (m.mic(), 0.5)
temp = feature_selection.SelectKBest(lambda X, Y: np.array(list(map(lambda x: mic(x, Y), X.T))).T[0], k=k).fit(matrix, target)
scores = temp.scores_.tolist()
indx = temp.get_support().tolist()
result = temp.transform(matrix).tolist()
return scores, indx, result

## X: array-like
## target: array-like
## k: int
## http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
```

## 基于树模型

```python
import numpy as np
from sklearn import feature_selection
from sklearn.ensemble import GradientBoostingClassifier

matrix = np.array(X)
target = np.array(target)
temp = feature_selection.SelectFromModel(GradientBoostingClassifier()).fit(matrix, target)
indx = temp._get_support_mask().tolist()
scores = get_importance(temp.estimator_).tolist()
result = temp.transform(matrix).tolist()
return scores, indx, result

## X: array-like
## target: array-like
## http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html
```

## 卡方检验法

```python
import numpy as np
from sklearn import feature_selection
from sklearn.feature_selection import chi2

matrix = np.array(X)
target = np.array(target)
temp = feature_selection.SelectKBest(chi2, k=k).fit(matrix, target)
scores = temp.scores_.tolist()
indx = temp.get_support().tolist()
result = temp.transform(matrix).tolist()
return scores, indx, result

## X: array-like
## target: array-like
## k: int
## http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html

```

## 基于惩罚值

```python
import numpy as np
from sklearn import feature_selection
from sklearn.linear_model import LogisticRegression

matrix = np.array(arr0)
target = np.array(target)
temp = feature_selection.SelectFromModel(LogisticRegression(penalty="l1", C=0.1)).fit(matrix, target)
indx = temp._get_support_mask().tolist()
scores = get_importance(temp.estimator_).tolist()
result = temp.transform(matrix).tolist()
return scores, indx, result

## X: array-like
## target: array-like
## http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html
```

# 降维

## PCA主成分分析算法

```python
import numpy as np
from sklearn.decomposition import PCA

matrix = np.array(X)
pca = PCA(n_components='mle', svd_solver='auto').fit(matrix)
result = pca.transform(matrix)
label = result.tolist()
return label, pca.components_.tolist(), pca.explained_variance_.tolist(), pca.explained_variance_ratio_.tolist(), pca.mean_.tolist(), pca.noise_variance_

```

## 哑变量

```python
import numpy as np
from sklearn import preprocessing

raw_result = preprocessing.OneHotEncoder().fit_transform(np.array(X))
result = raw_result.tolist()

## X: array-like
## http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

```


## TSNE-t 分布邻域嵌入算法

```python
import numpy as np
from sklearn.manifold import TSNE

matrix = np.array(X)
t_sne = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
result = t_sne.fit(matrix)
kl_divergence = result.kl_divergence_
label = t_sne.fit_transform(matrix).tolist()

return label, kl_divergence
## X: array-like
## http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
```

## 线性判别分析法(LDA)

```python
from sklearn.lda import LDA
import numpy as np
from sklearn import preprocessing

matrix = np.array(X)
target = np.array(target)
temp = LDA(n_components=n_components).fit(matrix, target)
coef = temp.coef_
mean = temp.means_
priors = temp.priors_
scalings = temp.scalings_
xbar = temp.xbar_
label = temp.transform(matrix).tolist()
return label, coef.tolist(), mean.tolist(), priors.tolist(), scalings.tolist(), xbar.tolist()

## X: array-like
## target: array-like
## n_components: int
## http://scikit-learn.org/0.15/modules/generated/sklearn.lda.LDA.html
```

# 分类器

## 随机梯度下降(SGD)

```python
## http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier
from sklearn.linear_model import SGDClassifier

clf = SGDClassifier(loss="hinge", penalty="l2")
clf.fit(X, y)

## 数组 X shape:[n_samples, n_features]
## 数组 y shape:[n_samples]
```

## 逻辑回归(LR)

```python
## http://scikit-learn.org/stable/modules/linear_model.html
from sklearn import linear_model

lr =linear_model.LogisticRegression()  ## penalty : str, ‘l1’ or ‘l2’, default: ‘l2’
lr.fit(X, y)


```

## 分类树(Tree)

```python
#http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
from sklearn.tree import DecisionTreeClassifier

## criterion = "gini" (CART) or "entropy" (ID3)
clf = DecisionTreeClassifier(criterion = 'entropy' ,random_state = 0)
clf.fit(X,y)
```

## 随机森林(Random Forest)

```python
## http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
from sklearn.ensemble import RandomForestClassifier
import numpy as np

## criterion: str, 'gini' for Gini impurity (Default) and “entropy” for the information gain.
clf = RandomForestClassifier(criterion='gini')  
## X: array-like, shape = [n_samples, n_features]
## y: array-like, shape = [n_samples] or [n_samples, n_outputs]
clf.fit(np.array(X), np.array(y))
```

## 支持向量机(SVM)

```python
## http://scikit-learn.org/dev/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
from sklearn import svm
import numpy as np

## Classifier Option 1: SVC()
clf = svm.SVC()       ## kernel = 'linear' or 'rbf' (default) or 'poly' or custom kernels; penalty C = 1.0 (default)
## Option 2: NuSVC()
## clf = svm.NuSVC() 
## Option 3: LinearSVC()
## clf = svm.LinearSVC()     ## penalty : str, ‘l1’ or ‘l2’ (default=’l2’)
clf.fit(X, y)                ## X shape = [n_samples, n_features], y shape = [n_samples] or [n_samples, n_outputs]

## print(clf.support_vectors_) ## get support vectors
## print(clf.support_)         ## get indeices of support vectors
## print(clf.n_support_)       ## get number of support vectors for each class

mean_accuracy = clf.score(X,y)
print("Accuracy: %.3f"%(mean_accuracy))
```

## 朴素贝叶斯(Naive Bayesian)

```python
## http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
## predict(test_set或者train_set) 值为你需要预测的数据集
y_pred = gnb.fit(X, y).predict(X)
print("Number of mislabeled points out of a total %d points : %d" % (X.shape[0],(y != y_pred).sum()))
```

## AdaBoostClassifier

```python
## 决策树集成
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

adaboost_dt_clf = AdaBoostClassifier(
                                    DecisionTreeClassifier(
                                        max_depth=2,   ## 决策树最大深度，默认可不输入即不限制子树深度
                                        min_samples_split=20, ## 内部结点再划分所需最小样本数，默认值为2，若样本量不大，无需更改，反之增大
                                        min_samples_leaf=5    ## 叶子节点最少样本数,默认值为1，若样本量不大，无需更改，反之增大
                                        ),
                                    algorithm="SAMME", ## boosting 算法 {‘SAMME’, ‘SAMME.R’}, 默认为后者
                                    n_estimators=200,  ## 最多200个弱分类器，默认值为50
                                    learning_rate=0.8  ## 学习率，默认值为1
                                     )
adaboost_dt_clf.fit(X,y)
```

## GBDT(梯度加速决策树)

```python
## http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier

gbdt = GradientBoostingClassifier(max_depth=4,   ## 决策树最大深度，默认可不输入，即不限制子树深度
                                max_features="auto",  ## 寻找最优分割的特征数量，可为int,float,"auto","sqrt","log2",None:
                                n_estimators=100 ## Boosting阶段的数量，默认值为100。
                                )
gbdt.fit(X,y)
```

## K-Means

```python
## 聚类
## http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
from sklearn.cluster import KMeans

kmeans = KMeans(
                n_clusters = 2, ## 簇的个数，默认值为8
                random_state=0  
                ).fit(X)

print(kmeans.labels_)
print("K Clusters Centroids:\n", kmeans.cluster_centers_)
```

## 使用 keras 做分类

```python
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.utils import np_utils

training_epochs = 200 #训练次数，总体数据需要循环多少次
batch_size = 10  

model = Sequential()
input = X.shape[1]
## 隐藏层128
model.add(Dense(128, input_shape=(input,)))
model.add(Activation('relu'))
## Dropout层用于防止过拟合
model.add(Dropout(0.2))
## 隐藏层128
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.2))
## 没有激活函数用于输出层，二分类问题，用sigmoid激活函数进行变换，多分类用softmax。
model.add(Dense(1))
model.add(Activation('sigmoid'))
## 使用高效的 ADAM 优化算法以，二分类损失函数binary_crossentropy，多分类的损失函数categorical_crossentropy
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=training_epochs, batch_size=32)

```

# 线性回归

## 使用sklearn

```python
import sklearn
sklearn.linear_model.LinearRegression()

## http://scikit-learn.org/stable/modules/linear_model.html
```

## 高度专门化的线性回归函数(scipy.stats)

```python
import scipy
from scipy import stats

stats.linregress(x, y=none)

## x, y : array_like
## https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html

```

## 使用Statsmodels的模型

```python
import statsmodels.api as sm
results = sm.OLS(y, X).fit()

## y: matrix
## X: constant
## http://www.statsmodels.org/dev/index.html
```

## 一般的最小二乘多项式拟合(numpy)

```python
import numpy as np
np.polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False)
## x : array_like, shape (M,)
## y : array_like, shape (M,) or (M, K)
## deg : int
## https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.polyfit.html
```

## 用矩阵因式分解计算最小二乘(numpy.linalg.lstsq)

```python
import numpy
from numpy import linalg

linalg.lstsq(a, b, rcond=-1)

## a : (M, N) array_like
## b : {(M,), (M, K)} array_like
## https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.lstsq.html
```

## 更通用的最小二乘极小化(scipy.optimize)

```python
import scipy
from scipy import optimize

optimize.curve_fit(f, xdata, ydata, p0=None, sigma=None, absolute_sigma=False, check_finite=True, bounds=(-inf, inf), method=None, jac=None, **kwargs)

## f : callable
## xdata : An M-length sequence or an (k,M)-shaped array for functions with k predictors
## ydata : M-length sequence
## https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
```

## 使用 keras 做线性回归

```python
from keras.models import Sequential #顺序模型
from keras.layers import Dense

n_hidden_1 = 64 #隐藏层1的神经元个数
n_hidden_2 = 64 #隐藏层2的神经元个数
n_input = 13 #输入层的个数，也就是特征数
n_classes = 1 #输出层的个数
training_epochs = 200 #训练次数，总体数据需要循环多少次
batch_size = 10  

model = Sequential()#先建立一个顺序模型
#向顺序模型里加入第一个隐藏层，第一层一定要有一个输入数据的大小，需要有input_shape参数
#model.add(Dense(n_hidden_1, activation='relu', input_shape=(n_input,)))
model.add(Dense(n_hidden_1, activation='relu', input_dim=n_input))
model.add(Dense(n_hidden_2, activation='relu'))
model.add(Dense(n_classes))

model.compile(loss='mse', optimizer='rmsprop', metrics=['mae',r2])

history = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=training_epochs, batch_size=batch_size)
```

