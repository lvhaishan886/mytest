#导入库
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pickle
#设置输出右对齐，防止中文不对齐
pd.set_option('display.unicode.east_asian_width',True)
#读取数据集，并将字符编码指定为gbk，防止中文报错
insurance_df=pd.read_csv(r'insurance-chinese.csv',encoding='gbk')


#将医疗费用定义为目标输出变量
output=insurance_df['医疗费用']

#使用年龄、性别、BMI、子女数量、是否吸烟、区域作为特征
features=insurance_df[['年龄','性别','BMI','子女数量','是否吸烟','区域']]
#对特征列进行独热编码
features=pd.get_dummies(features)
print('前五行数据：')
print(features.head())
#换行分割
print()

print("前五行目标数据:")
print(output.head())

#从features和output这两个数组中将数据集划分为训练集和测试集
#训练集为80%，测试集为20 ( 1- 80% )
#返回的xtrain和ytrain为划分得到的训练集特征和标签
# x _ test和y _ test为划分得到的测试集特征和标签
#这里标签和目标输出变量是一个意思

x_train,x_test, y_train , y_test = train_test_split ( features , output , train_size = 0.8 ) 

#构建一个随机森林回归模型的实例
rfr = RandomForestRegressor () 

#使用训练集数据x_train和y_train来拟（训练）模型
rfr .fit ( x_train , y_train )
#用训练好的模型rfr对测试集数据x_test进行预测，将预测结果存储在y_pred中
y_pred = rfr.predict(x_test)

#计算模型的可决系数(R_squared)
#- R_squared的值界定在0-1
#- R_squared接近0，表示模型仅能做出与平均值相当的预测
#- R_squared接近1，表示模型对数据的变异有很好的解释能力
#- 一般来讲，当R_squared值超过0.5以上时才被认为模型有良好的预测能力

r2 = r2_score(y_test,y_pred)

#使用with语句，简化文件操作
#open()函数和'wb'参数用于创建并写入字节流
#pickle.dump()方法将模型对象转换成字节流
with open('rfr_model.pkl','wb') as f:
    pickle.dump(rfr,f)


print('保存成功，已生成相关文件。')

def introduce_page () :
    """当选择简介页面时，将呈现该函数的内容"""
    st.write("# 欢迎使用！")
    st.sidebar.success("单击👈预测医疗费用")
    st.markdown(
    """
    # 医疗费用预测应用💰
    这个应用利用机器学习模型来预测医疗费用，为保险公司的保险定价提供参考。

    ## 背景介绍
    - 开发目标：帮助保险公司合理定价保险产品，控制风险；
    - 模型算法：利用随机森林回归算法训练医疗费用预测模型。

    ## 使用指南
    - 输入准确完整的被保险人信息，可以得到更准确的费用预测。
    - 预测结果可以作为保险定价的重要参考，但需审慎决策。
    - 有任何问题欢迎联系我们的技术支持。

    技术支持✉：support@example.com
    """
)
    
def predict_page():
    """当选择预测费用页面时，将呈现该函数的内容！"""

    
    st.markdown(
    """
    ## 使用说明
    这个应用利用机器学习模型来预测医疗费用，为保险公司的保险定价提供参考。
    
    - **👉输入信息**：在下面输入被保险人的个人信息、疾病信息等。
    - **👉费用预测**：应用会预测被保险人的未来医疗费用支出。
    """
)
    #运用表单和表单提交按钮
    with st.form ('user_inputs'):
        age = st. number_input('年龄',min_value=0)
        sex = st.radio('性别',options=['男性','女性'])
        bmi = st. number_input ('BMI', min_value=0.0)

        children = st. number_input("子女数量：",step=1,min_value=0)
        smoke = st.radio("是否吸烟",("是","否"))
        region = st.selectbox('区域' ,('东南部','西南部','东北部','西北部'))
        submitted = st.form_submit_button('预测费用!')

    if submitted:
        format_data = [age, sex, bmi, children, smoke, region]
        #初始化数据预处理格式中与岛屿相关的变量
        sex_female,sex_male=0,0
        #根据用户输入的性别数据更改对应的值
        if sex=='女性':
            sex_female=1
        elif sex =='男性':
            sex_male = 1
        smoke_yes,smoke_no=0,0
        #根据用户输入的吸烟数据更改对应的值
        if smoke=='是':
            smoke_yes = 1
        elif smoke =='否':
            smoke_no=1

        region_northeast,region_southeast,region_northwest,region_southwest=0,0,0,0
        #根据用户输入的岛屿数据更改对应的值
        if region =='东北部！':
            region_northeast = 1
        elif region =='东南部':
            region_southeast = 1
        elif region =='西北部':
            region_northwest = 1
        elif region =='西南部':
            region_southwest = 1

        format_data =[age,bmi,children,sex_female,sex_male,
                      smoke_no,smoke_yes,
                      region_northeast,region_southeast,region_northwest,region_southwest]

    #使用pickle 的1oad 方法从磁盘文件反序列化加载一个之前保存的随机森林回归模型
    with open (r'rfr_model.pkl','rb')as f:
        rfr_model = pickle.load(f)

    if submitted:
        format_data_df = pd.DataFrame(data=[format_data],columns=rfr_model.feature_names_in_)

        #使用模型对格式化后的数据 format_data 进行预测，返回预测的医疗费用
        predict_result=rfr_model.predict(format_data_df)[0]
        
        st.write('根据您输入的数据,预测该客户的医疗费用是：',round(predict_result,2))
        st.write("技术支持:email：:support@example.com")

#设置页面的标题、图标
st.set_page_config(
    page_title="医疗费用预测",
    page_icon="💰",
    )

#在左侧添加侧边栏并设置单选按钮
nav =st.sidebar.radio("导航",["简介","预测医疗费用"])
#根据选择的结果，展示不同的页面
if nav =="简介":
    introduce_page()
else:
    predict_page()
