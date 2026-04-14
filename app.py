import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# -------------------------------
# PAGE CONFIG (IMPORTANT)
# -------------------------------
st.set_page_config(layout="wide")


# -------------------------------
# LOAD DATA + TRAIN MODEL
# -------------------------------
@st.cache_resource
def load_model():

    df = pd.read_csv(r"C:\Users\Varsha\OneDrive\Desktop\sleep\Sleep_health_and_lifestyle_dataset.csv")

    def convert_quality(value):
        if value <= 4:
            return "Poor"
        elif value <= 7:
            return "Average"
        else:
            return "Good"

    df["Sleep Category"] = df["Quality of Sleep"].apply(convert_quality)

    X = df[[
        "Sleep Duration",
        "Stress Level",
        "Physical Activity Level",
        "Heart Rate"
    ]]

    y = df["Sleep Category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    return model, df, X


model, df, X = load_model()


# -------------------------------
# TITLE
# -------------------------------
st.title("Sleep Quality Predictor")


# -------------------------------
# INPUT SECTION
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    sleep = st.slider("Sleep Duration (hours)", 0.0, 12.0, 7.0)
    stress = st.slider("Stress Level", 0, 10, 5)

with col2:
    activity = st.slider("Physical Activity Level", 0, 100, 30)
    heart = st.slider("Heart Rate", 50, 120, 70)


# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict"):

    sample = pd.DataFrame([{
        "Sleep Duration": sleep,
        "Stress Level": stress,
        "Physical Activity Level": activity,
        "Heart Rate": heart
    }])

    result = model.predict(sample)[0]

    # COLOR OUTPUT
    if result == "Good":
        st.markdown("<h2 style='color:green;'>Good Sleep Quality</h2>", unsafe_allow_html=True)

    elif result == "Average":
        st.markdown("<h2 style='color:orange;'>Average Sleep Quality</h2>", unsafe_allow_html=True)

    else:
        st.markdown("<h2 style='color:red;'>Poor Sleep Quality</h2>", unsafe_allow_html=True)


    # SUGGESTIONS
    st.write("### Suggestions")

    if stress > 7:
        st.warning("Reduce stress")

    if activity < 30:
        st.info("Increase physical activity")

    if sleep < 6:
        st.warning("Sleep at least 7 hours")

    if heart > 90:
        st.warning("Maintain heart health")

    if result == "Good":
        st.success("Keep it up!")


# -------------------------------
# GRAPHS SIDE BY SIDE
# -------------------------------
st.write("## Analysis")

col3, col4 = st.columns(2)

# Graph 1
with col3:
    counts = df["Sleep Category"].value_counts()

    fig1, ax1 = plt.subplots()
    ax1.bar(counts.index, counts.values)
    ax1.set_title("Sleep Category Distribution")

    st.pyplot(fig1)


# Graph 2
with col4:
    importance = model.feature_importances_

    fig2, ax2 = plt.subplots()
    ax2.bar(X.columns, importance)
    ax2.set_title("Feature Importance")
    plt.xticks(rotation=45)

    st.pyplot(fig2)


# -------------------------------
# GENDER GRAPH (FULL WIDTH)
# -------------------------------
st.write("## Gender Analysis")

gender_data = df.groupby(["Gender", "Sleep Category"]).size().unstack()

fig3, ax3 = plt.subplots()
gender_data.plot(kind="bar", ax=ax3)

ax3.set_title("Sleep Quality by Gender")

st.pyplot(fig3)