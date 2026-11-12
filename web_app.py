import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="Student Performance Predictor")
st.title("🎓 Student Performance Predictor")
st.write("Enter the student's details and click Predict.")


@st.cache_resource
def train_model():
    df = pd.read_csv("students.csv")
    X = df.drop(columns=["final_score"])
    y = df["final_score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    return df, model, X.columns, r2, mae


df, model, columns, r2, mae = train_model()

# Inputs
study = st.slider("Study hours per day", 0.0, 10.0, 5.0, 0.5)
attendance = st.slider("Attendance (%)", 50, 100, 80)
previous = st.slider("Previous exam score", 0, 100, 65)
assignments = st.slider("Assignments done (out of 10)", 0, 10, 7)

# Predict
if st.button("Predict"):
    student = pd.DataFrame(
        [[study, attendance, previous, assignments]], columns=columns
    )
    score = float(model.predict(student)[0])
    score = min(max(score, 0), 100)
    st.success(f"Predicted final score: {score:.1f} / 100")

    if score >= 75:
        st.write("Result: Excellent 🌟")
    elif score >= 60:
        st.write("Result: Good 👍")
    elif score >= 40:
        st.write("Result: Average, needs more effort")
    else:
        st.write("Result: At risk, needs extra support")

# Model quality
st.subheader("Model accuracy")
col1, col2 = st.columns(2)
col1.metric("R2 score", f"{r2:.2f}")
col2.metric("Average error (marks)", f"{mae:.2f}")

# Graph
st.subheader("What affects marks the most?")
st.write("Marks gained for each +1 increase in a factor:")
effect = pd.Series(model.coef_, index=columns)
st.bar_chart(effect)

# Data
with st.expander("See the student data"):
    st.dataframe(df)