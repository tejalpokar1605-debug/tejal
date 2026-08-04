
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="EduPro Predictive Analytics", page_icon="📊", layout="wide")

st.title("📊 EduPro Predictive Modeling Dashboard")
st.markdown("### Course Demand and Revenue Forecasting")
st.caption("Interactive Streamlit Web Application")

uploaded_file = st.file_uploader(
    "Upload EduPro_Online_Platform Excel file",
    type=["xlsx", "xls"]
)

if uploaded_file is None:
    st.info("Upload the EduPro Excel dataset to start the dashboard.")
    st.markdown("""
    **Expected sheets:** Courses, Teachers, Transactions, Users

    **Dashboard modules**
    - Dataset overview
    - Course demand analysis
    - Revenue analysis
    - Teacher analysis
    - Feature engineering
    - Enrollment prediction
    - Model performance
    """)
    st.stop()

@st.cache_data
def load_data(file):
    courses = pd.read_excel(file, sheet_name="Courses")
    teachers = pd.read_excel(file, sheet_name="Teachers")
    transactions = pd.read_excel(file, sheet_name="Transactions")
    users = pd.read_excel(file, sheet_name="Users")
    return courses, teachers, transactions, users

courses, teachers, transactions, users = load_data(uploaded_file)

transactions["TransactionDate"] = pd.to_datetime(
    transactions["TransactionDate"], errors="coerce"
)

course_metrics = (
    transactions.groupby("CourseID")
    .agg(
        EnrollmentCount=("CourseID", "count"),
        CourseRevenue=("Amount", "sum")
    )
    .reset_index()
)

master = courses.merge(course_metrics, on="CourseID", how="left")
master["EnrollmentCount"] = master["EnrollmentCount"].fillna(0).astype(int)
master["CourseRevenue"] = master["CourseRevenue"].fillna(0)
category_revenue = (
    master.groupby("CourseCategory")["CourseRevenue"]
    .sum()
    .reset_index(name="CategoryRevenue")
)
master = master.merge(category_revenue, on="CourseCategory", how="left")

# Feature engineering
def price_band(x):
    if x == 0: return "Free"
    if x <= 150: return "Low"
    if x <= 350: return "Medium"
    return "High"

def duration_bucket(x):
    if x <= 10: return "Short"
    if x <= 25: return "Medium"
    if x <= 40: return "Long"
    return "Very Long"

def rating_tier(x):
    if x < 2: return "Low"
    if x < 3.5: return "Medium"
    if x < 4.5: return "High"
    return "Top"

master["PriceBand"] = master["CoursePrice"].apply(price_band)
master["DurationBucket"] = master["CourseDuration"].apply(duration_bucket)
master["RatingTier"] = master["CourseRating"].apply(rating_tier)
master["IsFree"] = (master["CourseType"] == "Free").astype(int)

st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Select Module",
    [
        "Dashboard Overview",
        "Course Demand Analysis",
        "Revenue Analysis",
        "Teacher Analysis",
        "Enrollment Prediction",
        "Model Performance"
    ]
)

if page == "Dashboard Overview":
    st.subheader("EduPro Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Courses", len(courses))
    c2.metric("Teachers", len(teachers))
    c3.metric("Transactions", f"{len(transactions):,}")
    c4.metric("Users", f"{len(users):,}")

    c5, c6, c7 = st.columns(3)
    c5.metric("Total Revenue", f"₹{master['CourseRevenue'].sum():,.2f}")
    c6.metric("Avg Enrollment", f"{master['EnrollmentCount'].mean():.2f}")
    c7.metric("Highest Enrollment", int(master["EnrollmentCount"].max()))

    st.subheader("Course Dataset")
    st.dataframe(master, use_container_width=True)

elif page == "Course Demand Analysis":
    st.subheader("Course Demand Analysis")

    col1, col2 = st.columns(2)

    with col1:
        demand = (
            master.groupby("CourseCategory")["EnrollmentCount"]
            .sum().sort_values(ascending=False)
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        demand.plot(kind="bar", ax=ax)
        ax.set_title("Enrollment by Course Category")
        ax.set_xlabel("Course Category")
        ax.set_ylabel("Total Enrollments")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

    with col2:
        top = master.nlargest(10, "EnrollmentCount").sort_values("EnrollmentCount")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(top["CourseName"], top["EnrollmentCount"])
        ax.set_title("Top 10 Most Popular Courses")
        ax.set_xlabel("Enrollments")
        st.pyplot(fig)

    st.dataframe(
        master[["CourseName", "CourseCategory", "CourseLevel", "EnrollmentCount"]]
        .sort_values("EnrollmentCount", ascending=False),
        use_container_width=True
    )

elif page == "Revenue Analysis":
    st.subheader("Revenue Analysis")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Revenue", f"₹{master['CourseRevenue'].sum():,.2f}")
    c2.metric("Average Revenue/Course", f"₹{master['CourseRevenue'].mean():,.2f}")
    c3.metric("Highest Course Revenue", f"₹{master['CourseRevenue'].max():,.2f}")

    col1, col2 = st.columns(2)

    with col1:
        rev = (
            master.groupby("CourseCategory")["CourseRevenue"]
            .sum().sort_values(ascending=False)
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        rev.plot(kind="bar", ax=ax)
        ax.set_title("Revenue by Course Category")
        ax.set_ylabel("Revenue (₹)")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

    with col2:
        monthly = (
            transactions.dropna(subset=["TransactionDate"])
            .set_index("TransactionDate")["Amount"]
            .resample("ME").sum()
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(monthly.index, monthly.values, marker="o")
        ax.set_title("Monthly Transaction Revenue")
        ax.set_ylabel("Revenue (₹)")
        plt.xticks(rotation=45)
        st.pyplot(fig)

elif page == "Teacher Analysis":
    st.subheader("Teacher Analysis")

    c1, c2, c3 = st.columns(3)
    c1.metric("Teachers", len(teachers))
    c2.metric("Avg Experience", f"{teachers['YearsOfExperience'].mean():.2f} years")
    c3.metric("Avg Teacher Rating", f"{teachers['TeacherRating'].mean():.2f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(teachers["YearsOfExperience"], teachers["TeacherRating"])
    ax.set_title("Teacher Experience vs Rating")
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Teacher Rating")
    st.pyplot(fig)

elif page == "Enrollment Prediction":
    st.subheader("🎯 Enrollment Prediction")

    # A practical prediction model using course-level information.
    features = [
        "CoursePrice",
        "CourseDuration",
        "CourseRating",
        "IsFree"
    ]
    X = master[features].copy()
    y = master["EnrollmentCount"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train, y_train)

    st.markdown("#### Enter Course Details")
    col1, col2 = st.columns(2)

    with col1:
        price = st.number_input("Course Price (₹)", min_value=0.0, value=100.0)
        duration = st.number_input("Course Duration (hours)", min_value=0.1, value=20.0)

    with col2:
        rating = st.slider("Course Rating", 1.0, 5.0, 4.0, 0.01)
        course_type = st.selectbox("Course Type", ["Paid", "Free"])

    is_free = 1 if course_type == "Free" else 0

    if st.button("Predict Enrollment", type="primary"):
        input_df = pd.DataFrame([{
            "CoursePrice": price,
            "CourseDuration": duration,
            "CourseRating": rating,
            "IsFree": is_free
        }])
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Enrollment: **{prediction:.0f} students**")

elif page == "Model Performance":
    st.subheader("Model Performance")

    features = ["CoursePrice", "CourseDuration", "CourseRating", "IsFree"]
    X = master[features]
    y = master["EnrollmentCount"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=5, min_samples_leaf=2, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, random_state=42
        )
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R²": r2_score(y_test, pred)
        })

    result_df = pd.DataFrame(results)
    st.dataframe(result_df.round(3), use_container_width=True)

    best = result_df.loc[result_df["R²"].idxmax()]
    st.success(
        f"Best model on this uploaded dataset: **{best['Model']}** "
        f"(R² = {best['R²']:.3f})"
    )

st.sidebar.markdown("---")
st.sidebar.caption("EduPro | Predictive Modeling for Course Demand and Revenue Forecasting")
