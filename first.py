import streamlit as st

import pandas as pd

from datetime import datetime



# 主标题

st.title("🕶️ 学生 小陆 - 数字档案")



# 基础信息章节

st.header("🔑 基础信息")

st.text("学生ID: NEO-2023-001")

st.markdown("**注册时间**: `2023-10-01 08:30:17` | **精神状态**: ✅ 正常")

st.markdown("**当前教室**: `实训楼301` | **安全等级**: `绝密`")



# 技能矩阵章节

st.header("📊 技能矩阵")

col1, col2, col3 = st.columns(3)

col1.metric("C语言", "95%", "2%", help="近期训练提升") 

col2.metric("Python", "87%", "-1%")

col3.metric("Java", "68%", "-10%", help="用则进废则退")



# 进度条展示

st.subheader("Streamlit课程进度")

st.progress(28, text="Streamlit课程进度")



# 任务日志章节

st.header("📝 任务日志")

mission_data = {

    "日期": ["2023-10-01", "2023-10-05", "2023-10-12"],

    "任务": ["学生数字档案", "课程管理系统", "数据图表展示"],

    "状态": ["✅ 完成", "🕒 进行中", "❌ 未完成"],

    "难度": ["★☆☆☆☆", "★★☆☆☆", "★★★☆☆"]

}

mission_df = pd.DataFrame(mission_data)

st.table(mission_df.style.applymap(

    lambda x: 'color: #0f0' if x == "✅ 完成" else 'color: #ff0')

)



# 代码块展示

st.subheader("🔐 最新代码成果")

st.code('''def matrix_breach():

    while True:

        if detect_vulnerability():

            exploit()

            return "ACCESS GRANTED"

        else:

            stealth_evade()''', language='python')



# 动态信息流

st.write("---")

st.write("`>> SYSTEM MESSAGE:` 下一个任务目标已解锁...")

st.write("`>> TARGET:` 课程管理系统")

st.write("`>> COUNTDOWN:`", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))



# 终端效果

st.markdown("""

系统状态: 在线

连接状态: 已加密

""")