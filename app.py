# import libery
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')


# set the page confiygure , page title , page icon , layout
st.set_page_config(page_title="Online Payments Fraud Detection" , page_icon='💳',layout='wide')

# load model
with open("model.pkl" , "rb") as file:
    model = pickle.load(file)

# load encoder
with open("encoder.pkl" , "rb") as file:
    encoder = pickle.load(file)

# load scaler
with open("scaler.pkl" , "rb") as file:
    scaler = pickle.load(file)


## sidebar header
st.sidebar.header("💳 Online Payments Fraud Detection")

# Sidebar Menu Section 
st.sidebar.markdown("---")

# Sidebar menu
menu = st.sidebar.radio("Select Option", ["Prediction", "Model Performance"])

if menu == "Prediction":
    st.title("🔮 User Prediction")

    ## create a form for user input
    with st.form(key="prediction_form"):
        # create input fields for user input
        unique_user = st.slider("Unique User ID", min_value=0,max_value=20,value=1)
        sex = st.number_input("Sex (Male = 0, Female = 1)", min_value=0, max_value=1)
        age = st.slider("Age", min_value=18, max_value=80, value=33)
        device_id_count = st.slider("Device ID Count", min_value=1, max_value=30, value=5)
        ip_address_count = st.slider("IP Address Count", min_value=1, max_value=30, value=3)
        source = st.selectbox("Source", options=["SEO", "Ads" , "Direct"])
        browser = st.selectbox("Browser", options=["Chrome", "Firefox", "Safari", "IE", "OPERA"])
        signup_month = st.slider("Account Signup Month", min_value=1, max_value=12, value=6)
        signup_hour= st.slider("Account Signup Hour", min_value=0, max_value=23, value=14)
        purchase_month = st.slider("Purchase Month", min_value=1, max_value=12, value=7)
        purchase_dayofweek = st.slider("Purchase Day of Week", min_value=0, max_value=6, value=2)
        # create a submit button
        submit_button = st.form_submit_button(label="Predict")

    ## created dataframe for user input
    input_data = pd.DataFrame({
        "unique_user": [unique_user], 
        "sex": [sex],
        "age": [age],
        "device_id_count": [device_id_count],
        "ip_address_count": [ip_address_count],
        "source": [source],
        "browser": [browser],
        "signup_month": [signup_month],
        "signup_hour": [signup_hour],
        "purchase_month": [purchase_month],
        "purchase_dayofweek": [purchase_dayofweek]
    })

    ## preprocess the input data
    encoded = encoder.transform(input_data[["source", "browser"]]) ##encode categorical data
    encoded_df = pd.DataFrame(encoded,columns=encoder.get_feature_names_out(["source", "browser"])) ## create a dataframe for encoded data
    encoded_df.index = input_data.index
    input_data = input_data.drop(["source", "browser"], axis=1) ## drop original categorical columns
    input_data = pd.concat([input_data, encoded_df], axis=1) ## concatenate encoded data with original data

    num_features = ['signup_month', 'signup_hour', 'purchase_month', 'purchase_dayofweek', 'device_id_count', 'ip_address_count', 'unique_user',"age"]
    input_data[num_features] = scaler.transform(input_data[num_features]) ## scale data

    ## make prediction
    if submit_button:
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.error("⚠️ The transaction is predicted to be fraudulent.")
        else:
            st.success("✅ The transaction is predicted to be legitimate.")

elif menu == "Model Performance":
    st.title("📊 Model Performance")

    ## open X_test, y_test and y_pred datasets
    X_test = pickle.load(open("X_test.pkl", "rb"))
    y_test = pickle.load(open("y_test.pkl", "rb"))
    y_pred = pickle.load(open("y_pred.pkl", "rb"))

    from sklearn.metrics import confusion_matrix
    import plotly.figure_factory as ff
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    st.markdown("""<h4 style='text-align: center;'>Confusion Matrix Heatmap</h4>""",unsafe_allow_html=True)
    fig = ff.create_annotated_heatmap(z=cm, x=["Predicted 0", "Predicted 1"], y=["Actual 0", "Actual 1"],colorscale="Blues",showscale=True)
    st.plotly_chart(fig)

    import plotly.express as px
    # class distribution
    y= pickle.load(open("y.pkl", "rb")) ## load y dataset for class distribution plot
    class_counts = y.value_counts().reset_index()
    class_counts.columns = ["Class", "Count"]
    # plotly bar chart
    st.markdown("""<h4 style='text-align: center;'>Class Distribution</h4>""",unsafe_allow_html=True)
    fig = px.bar(class_counts,x="Class",y="Count",text="Count",title="Class Distribution",color="Class")
    st.plotly_chart(fig, use_container_width=True)

    import plotly.graph_objects as go
    from sklearn.metrics import classification_report
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.markdown("""<h4 style='text-align: center;'>Classification Report</h4>""",unsafe_allow_html=True)
    fig = go.Figure(data=[go.Table(
    header=dict( values=list(df_report.columns), fill_color='royalblue', font=dict(color='black', size=14), align='center' ),
    cells=dict(values=[df_report[col] for col in df_report.columns],fill_color='black',align='center' ))])
    st.plotly_chart(fig)