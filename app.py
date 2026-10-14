"""
PlaceReady - AI Placement Readiness Predictor
Run:  pip install streamlit scikit-learn pandas numpy
      streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------
# 1. Role -> required skills: (skill, importance 1-5, what to learn)
# ---------------------------------------------------------------
ROLES = {
    "Data Science / ML": [
        ("Python", 5, "Finish a Python for Data Analysis course; practice daily"),
        ("SQL", 5, "Practice joins, group by, window functions on LeetCode/HackerRank"),
        ("Statistics", 5, "Learn probability, hypothesis testing, distributions"),
        ("Machine Learning", 5, "Andrew Ng's ML course + build 2 end-to-end models"),
        ("Pandas & NumPy", 4, "Clean and analyze 2 Kaggle datasets"),
        ("Data Visualization", 3, "Matplotlib/Seaborn; build one dashboard"),
        ("Power BI / Tableau", 3, "Make a business dashboard from a public dataset"),
        ("Deep Learning", 3, "Basics of neural nets with TensorFlow/PyTorch"),
        ("Git & GitHub", 2, "Push all projects with a clear README"),
    ],
    "Web Development": [
        ("HTML & CSS", 4, "Clone 2 real websites responsively"),
        ("JavaScript", 5, "Learn ES6, DOM, async/await; build small apps"),
        ("React", 5, "Build a full React app with routing and state"),
        ("Node.js", 4, "Create a REST API with Express"),
        ("REST APIs", 4, "Learn HTTP methods, auth (JWT), Postman"),
        ("SQL", 3, "Basic queries and schema design"),
        ("MongoDB", 3, "CRUD with Mongoose in a MERN project"),
        ("Git & GitHub", 3, "Branching, pull requests, deploy via Vercel/Netlify"),
    ],
    "Software Engineer": [
        ("DSA", 5, "Solve 150+ problems: arrays, trees, graphs, DP"),
        ("OOP", 4, "Master 4 pillars in Java/C++/Python with small designs"),
        ("Core Language (Java/C++/Python)", 5, "Pick one language and go deep"),
        ("DBMS", 4, "Normalization, transactions, SQL queries"),
        ("Operating Systems", 3, "Processes, threads, scheduling, memory"),
        ("Computer Networks", 3, "OSI/TCP-IP, HTTP, DNS basics"),
        ("System Design Basics", 3, "Learn scalability, caching, load balancing"),
        ("Git & GitHub", 3, "Version control and collaboration workflow"),
    ],
}


# ---------------------------------------------------------------
# 2. Synthetic dataset + model
#    (Be upfront in the demo: training data is synthetic.)
# ---------------------------------------------------------------
FEATURES = ["cgpa", "projects", "dsa_solved", "internships", "certifications", "skill_match"]


def generate_data(n=600, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "cgpa": np.clip(rng.normal(7.8, 0.9, n), 5.5, 10),
        "projects": rng.integers(0, 7, n),
        "dsa_solved": rng.integers(0, 400, n),
        "internships": rng.choice([0, 1, 2, 3], n, p=[0.45, 0.35, 0.15, 0.05]),
        "certifications": rng.integers(0, 6, n),
        "skill_match": rng.beta(2, 2, n),  # 0-1 fraction of required skills known
    })
    score = (
        0.20 * (df.cgpa - 5.5) / 4.5
        + 0.15 * df.projects / 6
        + 0.20 * df.dsa_solved / 400
        + 0.15 * df.internships / 3
        + 0.05 * df.certifications / 5
        + 0.25 * df.skill_match
    )
    score += rng.normal(0, 0.05, n)  # noise
    df["placed"] = (score > score.quantile(0.5)).astype(int)
    return df


@st.cache_resource
def train_model():
    df = generate_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df["placed"], test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    return model, model.score(X_test, y_test)


# ---------------------------------------------------------------
# 3. Helpers
# ---------------------------------------------------------------
def skill_match_and_gaps(role, known):
    skills = ROLES[role]
    total = sum(w for _, w, _ in skills)
    have = sum(w for s, w, _ in skills if s in known)
    gaps = sorted(
        [(s, w, tip) for s, w, tip in skills if s not in known],
        key=lambda x: -x[1],
    )
    return have / total, gaps


def profile_tips(cgpa, projects, dsa, internships, certs):
    tips = []
    if cgpa < 8.0:
        tips.append("Raise CGPA toward 8+ (many companies use it as a cutoff).")
    if projects < 3:
        tips.append("Build at least 3 solid projects with GitHub links.")
    if dsa < 150:
        tips.append("Solve more DSA problems (target 150+).")
    if internships < 1:
        tips.append("Get an internship or a virtual internship.")
    if certs < 2:
        tips.append("Add 2 relevant certifications (NPTEL, Coursera, etc.).")
    return tips


def label_for(score):
    if score >= 75:
        return "Placement Ready"
    if score >= 50:
        return "Almost There"
    return "Needs Work"


# ---------------------------------------------------------------
# 4. UI
# ---------------------------------------------------------------
st.set_page_config(page_title="PlaceReady", page_icon="🎯", layout="wide")
st.title("🎯 PlaceReady")
st.caption("AI Placement Readiness Predictor | Team Byte Blazers | INNOVATE X 2026")

model, acc = train_model()

with st.sidebar:
    st.header("Your Profile")
    role = st.selectbox("Target role", list(ROLES.keys()))
    cgpa = st.slider("CGPA", 5.0, 10.0, 8.0, 0.1)
    projects = st.number_input("Projects completed", 0, 15, 2)
    dsa = st.number_input("DSA problems solved", 0, 1000, 50)
    internships = st.number_input("Internships", 0, 5, 0)
    certs = st.number_input("Certifications", 0, 15, 1)
    known = st.multiselect(
        "Skills you already know",
        [s for s, _, _ in ROLES[role]],
    )
    go = st.button("Predict my readiness", type="primary", use_container_width=True)

if not go:
    st.info("Fill in your profile on the left and click **Predict my readiness**.")
    st.caption(f"Model: Random Forest | Test accuracy on synthetic data: {acc:.0%}")
    st.stop()

match, gaps = skill_match_and_gaps(role, known)
row = pd.DataFrame([[cgpa, projects, dsa, internships, certs, match]], columns=FEATURES)
score = float(model.predict_proba(row)[0][1] * 100)

# --- Score ---
c1, c2 = st.columns([1, 2])
with c1:
    st.metric("Readiness Score", f"{score:.0f} / 100")
    st.subheader(label_for(score))
with c2:
    st.progress(min(int(score), 100))
    st.write(f"**Skill match for {role}:** {match:.0%}")
    st.progress(int(match * 100))

st.divider()

# --- Gaps + skills to learn ---
left, right = st.columns(2)
with left:
    st.subheader("🔍 Top Skill Gaps")
    if gaps:
        gap_df = pd.DataFrame(gaps, columns=["Skill", "Importance", "Tip"]).set_index("Skill")
        st.bar_chart(gap_df["Importance"])
    else:
        st.success("You cover every key skill for this role!")

with right:
    st.subheader("📚 Skills to Learn Next")
    if gaps:
        for s, w, tip in gaps[:5]:
            st.markdown(f"**{s}** (priority {w}/5)  \n{tip}")
    else:
        st.write("Move on to advanced projects and mock interviews.")

st.divider()

# --- 30-day roadmap ---
st.subheader("🗓️ Your 30-Day Plan")
if gaps:
    for i, (s, _, tip) in enumerate(gaps[:3]):
        st.markdown(f"**Days {i*10+1}-{i*10+10}: {s}**: {tip}")
else:
    st.write("Do 2 mock interviews a week and polish your resume.")

# --- Profile improvements ---
tips = profile_tips(cgpa, projects, dsa, internships, certs)
if tips:
    st.subheader("⚡ Profile Boosters")
    for t in tips:
        st.write("• " + t)

st.caption(f"Model: Random Forest | Test accuracy on synthetic data: {acc:.0%}")
