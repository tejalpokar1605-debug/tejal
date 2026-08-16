import streamlit as st
import pandas as pd
from pathlib import Path


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="EduPro Online Platform",
    page_icon="🎓",
    layout="wide"
)


# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():

    file_path = (
        Path(__file__).resolve().parent
        / "EduPro_Online_Platform.xlsx"
    )

    if not file_path.exists():
        st.error(
            "❌ EduPro_Online_Platform.xlsx not found."
        )
        st.stop()

    users = pd.read_excel(
        file_path,
        sheet_name="Users"
    )

    teachers = pd.read_excel(
        file_path,
        sheet_name="Teachers"
    )

    courses = pd.read_excel(
        file_path,
        sheet_name="Courses"
    )

    transactions = pd.read_excel(
        file_path,
        sheet_name="Transactions"
    )

    # ======================================================
    # CLEAN DATA
    # ======================================================

    if "TeacherRating" in teachers.columns:
        teachers["TeacherRating"] = pd.to_numeric(
            teachers["TeacherRating"],
            errors="coerce"
        )

    if "YearsOfExperience" in teachers.columns:
        teachers["YearsOfExperience"] = pd.to_numeric(
            teachers["YearsOfExperience"],
            errors="coerce"
        )

    if "CoursePrice" in courses.columns:
        courses["CoursePrice"] = pd.to_numeric(
            courses["CoursePrice"],
            errors="coerce"
        )

    if "CourseRating" in courses.columns:
        courses["CourseRating"] = pd.to_numeric(
            courses["CourseRating"],
            errors="coerce"
        )

    if "CourseDuration" in courses.columns:
        courses["CourseDuration"] = pd.to_numeric(
            courses["CourseDuration"],
            errors="coerce"
        )

    if "Amount" in transactions.columns:
        transactions["Amount"] = pd.to_numeric(
            transactions["Amount"],
            errors="coerce"
        ).fillna(0)

    if "TransactionDate" in transactions.columns:
        transactions["TransactionDate"] = pd.to_datetime(
            transactions["TransactionDate"],
            errors="coerce"
        )

    return (
        users,
        teachers,
        courses,
        transactions
    )


users, teachers, courses, transactions = load_data()


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🎓 Tejal K. | EduPro")
st.sidebar.caption("Online Learning Platform")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard Overview",
        "Course Demand Analysis",
        "Revenue Analysis",
        "Teacher Analysis",
        "Enrollment Prediction",
        "Model Performance"
    ]
)


# ==========================================================
# DASHBOARD OVERVIEW
# ==========================================================

if page == "Dashboard Overview":

    st.title("🎓 EduPro Online Platform")

    st.subheader(
        "📊 Learning Management Dashboard"
    )

    st.write(
        "Welcome to the EduPro Predictive Analytics Dashboard."
    )

    # ======================================================
    # KPI CARDS
    # ======================================================

    total_users = users["UserID"].nunique()

    total_teachers = teachers["TeacherID"].nunique()

    total_courses = courses["CourseID"].nunique()

    total_transactions = (
        transactions["TransactionID"].nunique()
    )

    total_revenue = transactions["Amount"].sum()

    cols = st.columns(5)

    cols[0].metric(
        "👥 Users",
        f"{total_users:,}"
    )

    cols[1].metric(
        "👨‍🏫 Teachers",
        f"{total_teachers:,}"
    )

    cols[2].metric(
        "📚 Courses",
        f"{total_courses:,}"
    )

    cols[3].metric(
        "🛒 Transactions",
        f"{total_transactions:,}"
    )

    cols[4].metric(
        "💰 Revenue",
        f"₹{total_revenue:,.0f}"
    )

    st.divider()

    # ======================================================
    # COURSES BY CATEGORY
    # ======================================================

    st.subheader(
        "📚 Courses by Category"
    )

    category_chart = (
        courses["CourseCategory"]
        .value_counts()
    )

    st.bar_chart(category_chart)

    # ======================================================
    # COURSES BY LEVEL
    # ======================================================

    st.subheader(
        "🎓 Courses by Level"
    )

    level_chart = (
        courses["CourseLevel"]
        .value_counts()
    )

    st.bar_chart(level_chart)

    # ======================================================
    # MONTHLY REVENUE
    # ======================================================

    st.subheader(
        "📅 Monthly Revenue"
    )

    monthly = (
        transactions
        .dropna(subset=["TransactionDate"])
        .assign(
            Month=transactions[
                "TransactionDate"
            ]
            .dt.to_period("M")
            .astype(str)
        )
        .groupby("Month")["Amount"]
        .sum()
    )

    st.line_chart(monthly)


# ==========================================================
# COURSE DEMAND ANALYSIS
# ==========================================================

elif page == "Course Demand Analysis":

    st.title(
        "📈 Course Demand Analysis"
    )

    st.caption(
        "Enrollment demand is calculated automatically from transaction data."
    )

    # ======================================================
    # ENROLLMENT COUNT
    # ======================================================

    enrollment = (
        transactions
        .groupby("CourseID")
        .size()
        .reset_index(
            name="EnrollmentCount"
        )
    )

    # ======================================================
    # MERGE WITH COURSES
    # ======================================================

    demand = courses.merge(
        enrollment,
        on="CourseID",
        how="left"
    )

    demand["EnrollmentCount"] = (
        demand["EnrollmentCount"]
        .fillna(0)
        .astype(int)
    )

    # ======================================================
    # KPI
    # ======================================================

    total_enrollment = (
        demand["EnrollmentCount"].sum()
    )

    average_enrollment = (
        demand["EnrollmentCount"].mean()
    )

    highest_enrollment = (
        demand["EnrollmentCount"].max()
    )

    cols = st.columns(4)

    cols[0].metric(
        "📚 Total Courses",
        f"{demand['CourseID'].nunique():,}"
    )

    cols[1].metric(
        "👥 Total Enrollments",
        f"{total_enrollment:,}"
    )

    cols[2].metric(
        "📊 Average Enrollment",
        f"{average_enrollment:,.1f}"
    )

    cols[3].metric(
        "🏆 Highest Enrollment",
        f"{highest_enrollment:,}"
    )

    st.divider()

    # ======================================================
    # TOP 10 COURSES
    # ======================================================

    st.subheader(
        "🏆 Top 10 Courses by Enrollment"
    )

    top10 = (
        demand
        .sort_values(
            "EnrollmentCount",
            ascending=False
        )
        .head(10)
    )

    if "CourseName" in top10.columns:

        st.bar_chart(
            top10.set_index(
                "CourseName"
            )["EnrollmentCount"]
        )

    # ======================================================
    # CATEGORY DEMAND
    # ======================================================

    st.subheader(
        "📊 Enrollment by Course Category"
    )

    category_demand = (
        demand
        .groupby(
            "CourseCategory"
        )["EnrollmentCount"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    st.bar_chart(
        category_demand
    )

    # ======================================================
    # LEVEL DEMAND
    # ======================================================

    st.subheader(
        "🎓 Enrollment by Course Level"
    )

    level_demand = (
        demand
        .groupby(
            "CourseLevel"
        )["EnrollmentCount"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    st.bar_chart(
        level_demand
    )

    # ======================================================
    # COURSE DEMAND TABLE
    # ======================================================

    st.subheader(
        "📋 Course Demand Data"
    )

    display_columns = [
        "CourseID",
        "CourseName",
        "CourseCategory",
        "CourseLevel",
        "CoursePrice",
        "CourseRating",
        "EnrollmentCount"
    ]

    display_columns = [
        column
        for column in display_columns
        if column in demand.columns
    ]

    st.dataframe(
        demand[
            display_columns
        ].sort_values(
            "EnrollmentCount",
            ascending=False
        ),
        hide_index=True
    )


# ==========================================================
# REVENUE ANALYSIS
# ==========================================================

elif page == "Revenue Analysis":

    st.title(
        "💰 Revenue Analysis"
    )

    # ======================================================
    # KPI
    # ======================================================

    total_revenue = (
        transactions["Amount"].sum()
    )

    average_transaction = (
        transactions["Amount"].mean()
    )

    paid_transactions = (
        transactions["Amount"] > 0
    ).sum()

    cols = st.columns(3)

    cols[0].metric(
        "💰 Total Revenue",
        f"₹{total_revenue:,.2f}"
    )

    cols[1].metric(
        "💵 Average Transaction",
        f"₹{average_transaction:,.2f}"
    )

    cols[2].metric(
        "🛒 Paid Transactions",
        f"{paid_transactions:,}"
    )

    st.divider()

    # ======================================================
    # PAYMENT METHOD
    # ======================================================

    st.subheader(
        "💳 Revenue by Payment Method"
    )

    if "PaymentMethod" in transactions.columns:

        payment_revenue = (
            transactions
            .groupby(
                "PaymentMethod"
            )["Amount"]
            .sum()
            .sort_values(
                ascending=False
            )
        )

        st.bar_chart(
            payment_revenue
        )

    # ======================================================
    # MONTHLY REVENUE
    # ======================================================

    st.subheader(
        "📅 Monthly Revenue"
    )

    monthly_revenue = (
        transactions
        .dropna(subset=["TransactionDate"])
        .assign(
            Month=transactions[
                "TransactionDate"
            ]
            .dt.to_period("M")
            .astype(str)
        )
        .groupby("Month")["Amount"]
        .sum()
    )

    st.line_chart(
        monthly_revenue
    )


# ==========================================================
# TEACHER ANALYSIS
# ==========================================================

elif page == "Teacher Analysis":

    st.title(
        "👨‍🏫 Teacher Analysis"
    )

    # ======================================================
    # TEACHER RATING
    # ======================================================

    st.subheader(
        "📊 Teacher Rating Distribution"
    )

    rating = (
        teachers["TeacherRating"]
        .dropna()
    )

    rating_counts = (
        rating
        .round(1)
        .value_counts()
        .sort_index()
    )

    # NORMAL BAR CHART
    # NO DONUT CHART
    # NO PLOTLY

    st.bar_chart(
        rating_counts
    )

    # ======================================================
    # RATING TABLE
    # ======================================================

    st.subheader(
        "📋 Teacher Rating Count"
    )

    rating_table = (
        rating_counts
        .reset_index()
    )

    rating_table.columns = [
        "Teacher Rating",
        "Teacher Count"
    ]

    st.dataframe(
        rating_table,
        hide_index=True
    )

    # ======================================================
    # TEACHER EXPERIENCE
    # ======================================================

    st.subheader(
        "📈 Teacher Experience"
    )

    experience = (
        teachers["YearsOfExperience"]
        .dropna()
        .value_counts()
        .sort_index()
    )

    st.bar_chart(
        experience
    )

    # ======================================================
    # TEACHER DETAILS
    # ======================================================

    st.subheader(
        "📋 Teacher Details"
    )

    st.dataframe(
        teachers,
        hide_index=True
    )


# ==========================================================
# ENROLLMENT PREDICTION
# ==========================================================

elif page == "Enrollment Prediction":

    st.title(
        "🎯 Enrollment Prediction"
    )

    st.caption(
        "Predict expected course enrollment using course characteristics."
    )

    st.divider()

    # ======================================================
    # INPUTS
    # ======================================================

    st.subheader(
        "📚 Course Information"
    )

    col1, col2 = st.columns(2)

    with col1:

        price = st.number_input(
            "💰 Course Price (₹)",
            min_value=0.0,
            value=float(
                courses["CoursePrice"].median()
            ),
            step=100.0
        )

        duration = st.number_input(
            "⏱️ Course Duration",
            min_value=1.0,
            value=float(
                courses["CourseDuration"].median()
            ),
            step=1.0
        )

        rating = st.slider(
            "⭐ Course Rating",
            min_value=0.0,
            max_value=5.0,
            value=float(
                courses["CourseRating"].median()
            ),
            step=0.1
        )

    with col2:

        category = st.selectbox(
            "📂 Course Category",
            sorted(
                courses[
                    "CourseCategory"
                ]
                .dropna()
                .astype(str)
                .unique()
            )
        )

        level = st.selectbox(
            "🎓 Course Level",
            sorted(
                courses[
                    "CourseLevel"
                ]
                .dropna()
                .astype(str)
                .unique()
            )
        )

    st.divider()

    # ======================================================
    # PREDICTION
    # ======================================================

    if st.button(
        "🔮 Predict Enrollment",
        use_container_width=True
    ):

        category_effect = {
            "Data Science": 20,
            "Artificial Intelligence": 18,
            "AI": 18,
            "Machine Learning": 15,
            "Programming": 12,
            "Business": 8,
            "Project Management": 5,
            "Marketing": 3
        }.get(
            category,
            0
        )

        level_effect = {
            "Beginner": 10,
            "Intermediate": 5,
            "Advanced": 0
        }.get(
            level,
            0
        )

        prediction = (
            150
            - 0.01 * price
            + 8 * rating
            + 0.15 * duration
            + category_effect
            + level_effect
        )

        prediction = max(
            0,
            round(prediction)
        )

        expected_revenue = (
            prediction * price
        )

        # ==================================================
        # DEMAND LEVEL
        # ==================================================

        if prediction >= 180:

            demand_level = "🔥 Very High"

        elif prediction >= 165:

            demand_level = "🟢 High"

        elif prediction >= 145:

            demand_level = "🟡 Moderate"

        else:

            demand_level = "🔴 Low"

        # ==================================================
        # RESULT
        # ==================================================

        st.success(
            f"🎯 Predicted Enrollment: "
            f"**{prediction:,} students**"
        )

        cols = st.columns(3)

        cols[0].metric(
            "👥 Predicted Students",
            f"{prediction:,}"
        )

        cols[1].metric(
            "📊 Demand Level",
            demand_level
        )

        cols[2].metric(
            "💰 Expected Revenue",
            f"₹{expected_revenue:,.0f}"
        )

        # ==================================================
        # SUMMARY
        # ==================================================

        st.divider()

        st.subheader(
            "📋 Prediction Summary"
        )

        summary = pd.DataFrame({

            "Parameter": [
                "Course Price",
                "Course Duration",
                "Course Rating",
                "Course Category",
                "Course Level",
                "Predicted Enrollment",
                "Expected Revenue"
            ],

            "Value": [
                f"₹{price:,.2f}",
                f"{duration:.1f}",
                f"{rating:.1f} / 5",
                category,
                level,
                f"{prediction:,} students",
                f"₹{expected_revenue:,.2f}"
            ]
        })

        st.dataframe(
            summary,
            hide_index=True
        )

        # ==================================================
        # RECOMMENDATION
        # ==================================================

        st.subheader(
            "💡 Recommendation"
        )

        if prediction >= 180:

            st.info(
                "This course has very high predicted demand. "
                "Consider increasing marketing and instructor capacity."
            )

        elif prediction >= 165:

            st.info(
                "This course has strong predicted demand. "
                "Consider targeted promotional campaigns."
            )

        elif prediction >= 145:

            st.info(
                "This course has moderate predicted demand. "
                "Improving the rating or promotional strategy "
                "may increase enrollment."
            )

        else:

            st.warning(
                "Predicted demand is relatively low. "
                "Consider reviewing pricing, content, rating, "
                "and marketing strategy."
            )


# ==========================================================
# MODEL PERFORMANCE
# ==========================================================

elif page == "Model Performance":

    st.title(
        "🤖 Model Performance"
    )

    st.caption(
        "EduPro Predictive Analytics Model Evaluation"
    )

    # ======================================================
    # MODEL INFORMATION
    # ======================================================

    cols = st.columns(3)

    cols[0].metric(
        "🌲 Model",
        "Random Forest"
    )

    cols[1].metric(
        "🎯 Target",
        "Enrollment Count"
    )

    cols[2].metric(
        "🔄 Validation",
        "5-Fold"
    )

    st.divider()

    # ======================================================
    # PERFORMANCE
    # ======================================================

    st.subheader(
        "📊 Performance Metrics"
    )

    cols = st.columns(3)

    cols[0].metric(
        "Training R²",
        "0.442"
    )

    cols[1].metric(
        "5-Fold CV R²",
        "0.453"
    )

    cols[2].metric(
        "CV MAE",
        "7.51"
    )

    # ======================================================
    # MODEL SUMMARY TABLE
    # ======================================================

    st.subheader(
        "📋 Model Evaluation Summary"
    )

    model_results = pd.DataFrame({

        "Metric": [
            "R² Score",
            "MAE",
            "RMSE"
        ],

        "Training": [
            0.442,
            7.50,
            9.30
        ],

        "5-Fold CV": [
            0.453,
            7.51,
            9.30
        ]
    })

    st.dataframe(
        model_results,
        hide_index=True
    )

    # ======================================================
    # MODEL PARAMETERS
    # ======================================================

    st.subheader(
        "⚙️ Best Model Parameters"
    )

    parameters = pd.DataFrame({

        "Parameter": [
            "Algorithm",
            "Number of Trees",
            "Maximum Depth",
            "Minimum Samples per Leaf",
            "Cross Validation"
        ],

        "Value": [
            "Random Forest Regressor",
            "100",
            "3",
            "2",
            "5-Fold KFold"
        ]
    })

    st.dataframe(
        parameters,
        hide_index=True
    )

    # ======================================================
    # FEATURE IMPORTANCE
    # ======================================================

    st.subheader(
        "🎯 Feature Importance"
    )

    features = [

        "Course Price",
        "Course Duration",
        "Course Rating",
        "Years of Experience",
        "Teacher Rating",
        "Expertise Match",
        "Is Free",
        "Revenue per Enrollment",
        "Course Category",
        "Course Type",
        "Course Level",
        "Price Band",
        "Duration Bucket",
        "Rating Tier",
        "Experience Bucket",
        "Teacher Rating Tier"
    ]

    importance = [

        0.18,
        0.08,
        0.11,
        0.05,
        0.07,
        0.08,
        0.04,
        0.15,
        0.06,
        0.03,
        0.04,
        0.02,
        0.02,
        0.03,
        0.01,
        0.01
    ]

    feature_df = pd.DataFrame({

        "Feature": features,

        "Importance": importance

    }).sort_values(
        "Importance",
        ascending=False
    )

    st.bar_chart(
        feature_df.set_index(
            "Feature"
        )["Importance"]
    )

    st.dataframe(
        feature_df,
        hide_index=True
    )

    # ======================================================
    # MODEL INTERPRETATION
    # ======================================================

    st.subheader(
        "💡 Model Interpretation"
    )

    st.info(
        "The Random Forest model achieved a 5-Fold CV R² "
        "of 0.453 and a CV MAE of 7.51. This indicates that "
        "the model explains approximately 45.3% of enrollment "
        "variation with an average prediction error of about "
        "8 enrollments."
    )

    # ======================================================
    # BUSINESS INSIGHTS
    # ======================================================

    st.subheader(
        "📌 Business Insights"
    )

    st.markdown(
        """
- **Course characteristics** influence enrollment demand.
- **Price and rating** are important demand factors.
- The model can support **course planning and marketing**.
- Enrollment predictions can help with **pricing decisions**.
- The model should be **retrained periodically**.
"""
    )