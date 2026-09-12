import re
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Customer Feedback Intelligence", page_icon="🤖", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("feedback_dataset_v2.csv")

@st.cache_resource
def train(df):
    v=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True)
    X=v.fit_transform(df.feedback)
    s=LogisticRegression(max_iter=3000,C=2).fit(X,df.sentiment)
    c=LogisticRegression(max_iter=3000,C=2).fit(X,df.category)
    return v,s,c

def keyword_category(t):
    t=t.lower()
    rules={
      "Feature Request":["please add","add ","feature","option","would like","allow ","support "],
      "Billing":["charged","payment","billing","deducted","invoice","price"],
      "Delivery":["delivery","delivered","late","shipping","order arrived","tracking"],
      "Customer Support":["support","agent","respond","reply","help"],
      "Product Quality":["product quality","damaged","broken","quality","works as expected"],
      "Packaging":["packaging","package","box"],
      "App/Technical":["app","website","crash","checkout","login","slow","unresponsive"]
    }
    for cat,ks in rules.items():
        if any(k in t for k in ks): return cat
    return None

df=load_data()
v,smodel,cmodel=train(df)

st.title("🤖 AI-Powered Customer Feedback Intelligence")
st.caption("NLP + Machine Learning • Sentiment • Issue Classification • Business Prioritization")

a,b,c,d=st.columns(4)
a.metric("Total Feedback",len(df))
b.metric("Negative",int((df.sentiment=="Negative").sum()))
c.metric("Positive",int((df.sentiment=="Positive").sum()))
d.metric("High Priority",int((df.priority=="High").sum()))

t1,t2,t3=st.tabs(["🔮 Live AI Prediction","📊 Analytics","📈 Model Evaluation"])

with t1:
    text=st.text_area("Customer feedback","Please add UPI and more payment options to checkout.",height=120)
    if st.button("Analyze with AI",type="primary"):
        X=v.transform([text])
        sentiment=smodel.predict(X)[0]
        category=keyword_category(text) or cmodel.predict(X)[0]
        sc=max(smodel.predict_proba(X)[0])
        cc=max(cmodel.predict_proba(X)[0])
        if category=="Feature Request": priority="Low"
        elif sentiment=="Negative": priority="High"
        else: priority="Medium"
        x1,x2,x3,x4=st.columns(4)
        x1.metric("Sentiment",sentiment)
        x2.metric("Issue Category",category)
        x3.metric("Priority",priority)
        x4.metric("Confidence",f"{max(sc,cc)*100:.1f}%")
        if priority=="High":
            msg="Escalate this complaint and contact the customer quickly."
        elif category=="Feature Request":
            msg="Log this request for product-team review and roadmap prioritization."
        else:
            msg="Monitor this feedback and include it in service improvement analysis."
        st.success("Recommended action: "+msg)

with t2:
    left,right=st.columns(2)
    with left:
        st.plotly_chart(px.pie(df,names="sentiment",title="Sentiment Distribution"),use_container_width=True)
    with right:
        q=df.category.value_counts().reset_index()
        q.columns=["category","count"]
        st.plotly_chart(px.bar(q,x="category",y="count",title="Feedback by Issue Category"),use_container_width=True)
    st.subheader("Feedback Data")
    st.dataframe(df,use_container_width=True,hide_index=True)

with t3:
    st.subheader("Model Performance")
    Xtr,Xte,ytr,yte=train_test_split(df.feedback,df.sentiment,test_size=.25,random_state=42,stratify=df.sentiment)
    vv=TfidfVectorizer(ngram_range=(1,2),lowercase=True)
    A=vv.fit_transform(Xtr); B=vv.transform(Xte)
    mm=LogisticRegression(max_iter=3000,C=2).fit(A,ytr)
    pred=mm.predict(B)
    acc=accuracy_score(yte,pred)
    pr,re,f1,_=precision_recall_fscore_support(yte,pred,average="weighted",zero_division=0)
    z1,z2,z3,z4=st.columns(4)
    z1.metric("Accuracy",f"{acc*100:.1f}%");z2.metric("Precision",f"{pr*100:.1f}%");z3.metric("Recall",f"{re*100:.1f}%");z4.metric("F1 Score",f"{f1*100:.1f}%")
    labels=sorted(df.sentiment.unique())
    cm=confusion_matrix(yte,pred,labels=labels)
    fig=px.imshow(cm,x=labels,y=labels,text_auto=True,labels=dict(x="Predicted",y="Actual"),title="Sentiment Confusion Matrix")
    st.plotly_chart(fig,use_container_width=True)
