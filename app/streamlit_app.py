import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clean, professional CSS - NO LINKS
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
    }
    
    /* Main background - clean white */
    [data-testid="stAppViewContainer"] {
        background-color: #f5f7fa !important;
    }
    
    .main {
        background-color: #f5f7fa !important;
    }
    
    /* Remove all link styling */
    a {
        text-decoration: none !important;
        color: inherit !important;
    }
    
    a:hover {
        color: inherit !important;
        text-decoration: none !important;
    }
    
    /* Header */
    .header-title {
        font-size: 3em;
        font-weight: 800;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 0.3em;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        font-size: 1.05em;
        color: #475569;
        text-align: center;
        margin-bottom: 2em;
        font-weight: 500;
    }
    
    /* Input card - NO LINK STYLING */
    .input-card {
        background: white;
        border-radius: 12px;
        padding: 1.8em;
        margin: 1em 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
    }
    
    .input-card h3 {
        color: #1e3a8a !important;
        font-size: 1.2em;
        margin-bottom: 1.2em;
        font-weight: 700;
        text-decoration: none !important;
    }
    
    .input-card h3 a {
        color: #1e3a8a !important;
        text-decoration: none !important;
        background: none !important;
    }
    
    /* Section header - NO UNDERLINE OR LINKS */
    .section-header {
        font-size: 1.6em;
        font-weight: 700;
        color: #1e3a8a;
        margin: 2em 0 1.2em 0;
        padding-bottom: 0em;
        border-bottom: none;
        text-decoration: none !important;
    }
    
    .section-header a {
        color: #1e3a8a !important;
        text-decoration: none !important;
    }
    
    /* Prediction result cards */
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 2em;
        margin: 1em 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-top: 5px solid #3b82f6;
        text-align: center;
    }
    
    .result-label {
        font-size: 0.9em;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.6em;
    }
    
    .result-value {
        font-size: 2.5em;
        font-weight: 800;
        color: #1e3a8a;
        margin: 0.3em 0;
    }
    
    .result-range {
        font-size: 1.1em;
        color: #475569;
        margin-top: 0.5em;
        font-weight: 600;
    }
    
    /* Grade A/B/C/D colors */
    .grade-a {
        border-top-color: #10b981 !important;
        background: linear-gradient(135deg, rgba(16,185,129,0.05) 0%, transparent 100%);
    }
    
    .grade-b {
        border-top-color: #f59e0b !important;
        background: linear-gradient(135deg, rgba(245,158,11,0.05) 0%, transparent 100%);
    }
    
    .grade-c {
        border-top-color: #f97316 !important;
        background: linear-gradient(135deg, rgba(249,115,22,0.05) 0%, transparent 100%);
    }
    
    .grade-d {
        border-top-color: #ef4444 !important;
        background: linear-gradient(135deg, rgba(239,68,68,0.05) 0%, transparent 100%);
    }
    
    .grade-text-a { color: #10b981 !important; }
    .grade-text-b { color: #f59e0b !important; }
    .grade-text-c { color: #f97316 !important; }
    .grade-text-d { color: #ef4444 !important; }
    
    /* Recommendation boxes */
    .recommendation {
        background: white;
        border-left: 5px solid #3b82f6;
        padding: 1.6em;
        margin: 1.2em 0;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    
    .recommendation-critical {
        border-left-color: #ef4444;
        background: #fef2f2;
    }
    
    .recommendation-warning {
        border-left-color: #f59e0b;
        background: #fffbf0;
    }
    
    .recommendation-good {
        border-left-color: #10b981;
        background: #f0fdf4;
    }
    
    .recommendation strong {
        color: #1e3a8a;
        display: block;
        margin-bottom: 0.6em;
        font-size: 1.08em;
    }
    
    .recommendation p {
        color: #475569;
        line-height: 1.7;
        margin: 0.5em 0;
    }
    
    .score-highlight {
        display: inline-block;
        background: #dbeafe;
        color: #1e40af;
        padding: 0.3em 0.6em;
        border-radius: 4px;
        font-weight: 600;
        margin: 0 0.2em;
    }
    
    .psych-tip {
        background: #f0f9ff;
        border-left: 4px solid #0284c7;
        padding: 1em;
        margin-top: 0.8em;
        border-radius: 4px;
        font-size: 0.95em;
        color: #0c4a6e;
        font-style: italic;
    }
    
    /* Activity box */
    .activity-box {
        background: #f0f9ff;
        border-left: 4px solid #0ea5e9;
        padding: 0.8em;
        margin: 0.6em 0;
        border-radius: 4px;
        font-size: 0.95em;
        color: #0369a1;
    }
    
    .activity-box strong {
        color: #0c4a6e;
        display: block;
        margin-bottom: 0.3em;
    }
    
    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 1em 2.5em !important;
        font-size: 1.1em !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.4) !important;
    }
    
    /* Chart container */
    .chart-container {
        background: white;
        border-radius: 12px;
        padding: 1.8em;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin: 1em 0;
        border: 1px solid #e2e8f0;
    }
    
    /* Summary container */
    .summary-container {
        background: white;
        border-radius: 12px;
        padding: 2em;
        margin: 1.5em 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        margin-top: 3em;
        padding: 2em;
        border-top: 1px solid #e2e8f0;
        font-size: 0.95em;
    }
    
    /* Info box */
    .info-box {
        background: #dbeafe;
        border-left: 4px solid #3b82f6;
        padding: 1.2em;
        border-radius: 8px;
        margin: 1em 0;
        color: #1e40af;
    }
</style>
""", unsafe_allow_html=True)

# Load models
MODEL_CLF = 'app/trained_classifier.pkl'
MODEL_REG = 'app/trained_regressor.pkl'
SCALER = 'app/scaler.pkl'

if not os.path.exists(MODEL_CLF) or not os.path.exists(MODEL_REG) or not os.path.exists(SCALER):
    st.error("❌ Model files not found. Please run the training cell first.")
    st.stop()

clf = joblib.load(MODEL_CLF)
reg = joblib.load(MODEL_REG)
with open(SCALER, 'rb') as f:
    scaler = pickle.load(f)

# Header
st.markdown('<h1 class="header-title">🎓 Student Performance Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="header-subtitle">Discover your potential and get personalized learning tips based on psychology</p>', unsafe_allow_html=True)

# Input section
st.markdown('<h2 class="section-header">📝 Tell Us About Yourself</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="input-card"><h3>📚 Academic Habits</h3>', unsafe_allow_html=True)
    study_hours = st.slider("Study Hours per Day", 0, 44, 20)
    attendance = st.slider("Attendance %", 0, 100, 80)
    assign_comp = st.slider("Assignment Completion %", 0, 100, 80)
    online_courses = st.slider("Online Courses Enrolled", 0, 20, 5)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="input-card"><h3>👤 Personal Info</h3>', unsafe_allow_html=True)
    age = st.slider("Age", 15, 60, 23)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    resources = st.selectbox("Learning Materials Access", ["Low", "Medium", "High"], index=1)
    internet = st.selectbox("Internet Access", ["No", "Yes"], index=1)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<h2 class="section-header">💪 Engagement & Well-being</h2>', unsafe_allow_html=True)

col3, col4, col5, col6 = st.columns(4)

with col3:
    motivation = st.selectbox("Motivation", ["Low", "Medium", "High"], index=1)

with col4:
    stress_level = st.selectbox("Stress Level", ["Low", "Medium", "High"], index=1)

with col5:
    discussions = st.selectbox("Class Participation", ["Low", "Medium", "High"], index=1)

with col6:
    extracurricular = st.selectbox("Extracurricular", ["No", "Yes"], index=0)

col7, col8, col9 = st.columns(3)

with col7:
    edutech = st.selectbox("EduTech Tools Usage", ["No", "Yes"], index=0)

with col8:
    learning_style = st.selectbox("Learning Style", ["Visual", "Auditory", "Kinesthetic", "Mixed"], index=3)

with col9:
    sleep_hours = st.slider("Sleep Hours per Night", 4, 12, 7, help="Critical for memory and focus!")

# Convert inputs
gender_map = {"Male": 0, "Female": 1, "Other": 2}
resource_map = {"Low": 0, "Medium": 1, "High": 2}
motivation_map = {"Low": 0, "Medium": 1, "High": 2}
stress_map = {"Low": 0, "Medium": 1, "High": 2}
discussions_map = {"Low": 0, "Medium": 1, "High": 2}
internet_map = {"No": 0, "Yes": 1}
extracurricular_map = {"No": 0, "Yes": 1}
edutech_map = {"No": 0, "Yes": 1}
learning_map = {"Visual": 0, "Auditory": 1, "Kinesthetic": 2, "Mixed": 3}

input_data = {
    "StudyHours": study_hours,
    "Attendance": attendance,
    "Resources": resource_map[resources],
    "Extracurricular": extracurricular_map[extracurricular],
    "Motivation": motivation_map[motivation],
    "Internet": internet_map[internet],
    "Gender": gender_map[gender],
    "Age": age,
    "LearningStyle": learning_map[learning_style],
    "OnlineCourses": online_courses,
    "Discussions": discussions_map[discussions],
    "AssignmentCompletion": assign_comp,
    "EduTech": edutech_map[edutech],
    "StressLevel": stress_map[stress_level]
}

input_df = pd.DataFrame([input_data])
numeric_cols = ["StudyHours", "Attendance", "AssignmentCompletion", "OnlineCourses", "Age"]
input_df_scaled = input_df.copy()
input_df_scaled[numeric_cols] = scaler.transform(input_df[numeric_cols])

# Predict button
col_btn = st.columns([1, 3, 1])
with col_btn[1]:
    predict_clicked = st.button("🚀 Get My Predictions & Recommendations", use_container_width=True)

if predict_clicked:
    try:
        # Get predictions
        grade_pred = clf.predict(input_df_scaled)[0]
        exam_pred = reg.predict(input_df_scaled)[0]
        exam_pred = float(np.clip(exam_pred, 0, 100))
        
        # Map grade to letter grade and range
        grade_labels = {0: "D", 1: "C", 2: "B", 3: "A"}
        grade_letter = grade_labels.get(int(grade_pred), "D")
        
        # Define grade ranges and score ranges
        grade_ranges = {
            "A": ("Excellent", 85, 100),
            "B": ("Good", 70, 84),
            "C": ("Average", 55, 69),
            "D": ("Needs Improvement", 0, 54)
        }
        
        grade_name, score_min, score_max = grade_ranges[grade_letter]
        
        # Calculate predicted score range
        score_range_min = max(int(exam_pred) - 8, 0)
        score_range_max = min(int(exam_pred) + 8, 100)
        
        # Get grade styling
        grade_styles = {
            "A": "grade-a grade-text-a",
            "B": "grade-b grade-text-b",
            "C": "grade-c grade-text-c",
            "D": "grade-d grade-text-d"
        }
        
        st.markdown('<h2 class="section-header">✨ Your Predictions</h2>', unsafe_allow_html=True)
        
        col_pred1, col_pred2 = st.columns(2)
        
        with col_pred1:
            st.markdown(f'''
            <div class="result-card {grade_styles[grade_letter]}">
                <div class="result-label">Expected Final Grade</div>
                <div class="result-value">{grade_letter}</div>
                <div class="result-range">{grade_name}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col_pred2:
            st.markdown(f'''
            <div class="result-card">
                <div class="result-label">Expected Exam Score</div>
                <div class="result-value">{score_range_min} - {score_range_max}</div>
                <div class="result-range">out of 100</div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown('<div class="info-box">📌 These ranges are based on psychology-backed research. Better sleep, breaks, and mental health = better learning!</div>', unsafe_allow_html=True)
        
        # Psychology-based recommendations
        st.markdown('<h2 class="section-header">💡 Your Personalized Learning Path (Psychology-Based)</h2>', unsafe_allow_html=True)
        
        # Function to calculate score impact
        def calculate_potential_score(base_score, improvement_factor):
            return min(base_score + improvement_factor, 100)
        
        # Study Hours with sleep consideration
        if study_hours >= 18:
            rec_title = "📚 Excellent Study Routine!"
            rec_text = f"You're studying {study_hours} hours daily - fantastic dedication! But remember: quality > quantity."
            psych_note = "🧠 Psychology says: After 50-90 min of focused study, take a 10-15 min break. This boosts retention by 30% and prevents burnout."
            activities = """
            <div class="activity-box"><strong>🎯 Focus-Boosting Activities (50-90 min study blocks):</strong>
            • Pomodoro Technique: 25 min study + 5 min break (repeats of 4 = 1 long break)<br>
            • Active Recall: Test yourself without looking at notes<br>
            • Chunking: Break topics into small, digestible pieces<br>
            • Mind Mapping: Draw connections between concepts visually</div>
            """
            potential = min(score_range_max + 5, 100)
            st.markdown(f'''
            <div class="recommendation recommendation-good">
                <strong>{rec_title}</strong>
                <p>{rec_text}</p>
                <p style="margin-top: 0.8em;"><strong>Current Score Range:</strong> <span class="score-highlight">{score_range_min}-{score_range_max}</span></p>
                <p><strong>With Strategic Breaks:</strong> <span class="score-highlight">{int(score_range_min + 3)}-{int(potential)}</span></p>
                <div class="psych-tip">{psych_note}</div>
                {activities}
            </div>
            ''', unsafe_allow_html=True)
        elif study_hours >= 15:
            rec_title = "📚 Good Study Foundation"
            rec_text = f"You're studying {study_hours} hours daily - solid commitment! Adding just 3-5 more focused hours can boost grades significantly."
            psych_note = "🧠 Psychology says: Your brain needs spaced repetition (review material over days/weeks). One long study session is less effective than studying the same topic 3 times across different days."
            activities = """
            <div class="activity-box"><strong>🎯 Focus-Improvement Activities:</strong>
            • Spaced Repetition: Review material after 1 day, 3 days, 1 week<br>
            • Teach Others: Explain concepts to friends (Feynman Technique)<br>
            • Practice Problems: Apply concepts to new questions<br>
            • Interleaved Practice: Mix different topics while studying</div>
            """
            potential = calculate_potential_score(score_range_max, 8)
            st.markdown(f'''
            <div class="recommendation recommendation-good">
                <strong>{rec_title}</strong>
                <p>{rec_text}</p>
                <p style="margin-top: 0.8em;"><strong>Current Score Range:</strong> <span class="score-highlight">{score_range_min}-{score_range_max}</span></p>
                <p><strong>With +5 Hours & Spaced Repetition:</strong> <span class="score-highlight">{int(score_range_min + 5)}-{int(potential)}</span></p>
                <div class="psych-tip">{psych_note}</div>
                {activities}
            </div>
            ''', unsafe_allow_html=True)
        else:
            rec_title = "📚 Level Up Your Study Time"
            rec_text = f"Currently studying {study_hours} hours daily. Research shows students who study 18-22 hours daily with proper breaks perform best."
            psych_note = f"🧠 Psychology says: Your brain can only focus intensely for 50-90 minutes. Instead of studying {study_hours} straight, split it: 6-7 sessions of 90 min with 15-20 min breaks. You'll absorb more!"
            activities = """
            <div class="activity-box"><strong>🎯 High-Impact Focus Activities (Try These First):</strong>
            • Power Hour: 60 min intense focus → 10 min break (repeat 3x)<br>
            • Feynman Technique: Explain topics simply - if you can't, you don't understand<br>
            • Dual Coding: Combine notes with diagrams/videos (35% better retention)<br>
            • The Leitner System: Use flashcards, move mastered ones aside<br>
            • Forest App: Gamified focus timer (prevents phone distractions)</div>
            """
            potential = calculate_potential_score(score_range_max, 15)
            st.markdown(f'''
            <div class="recommendation recommendation-warning">
                <strong>{rec_title}</strong>
                <p>{rec_text}</p>
                <p style="margin-top: 0.8em;"><strong>Current Score Range:</strong> <span class="score-highlight">{score_range_min}-{score_range_max}</span></p>
                <p><strong>With 18-22 Hours (Smart Breaks):</strong> <span class="score-highlight">{int(score_range_min + 12)}-{int(potential)}</span></p>
                <div class="psych-tip">{psych_note}</div>
                {activities}
            </div>
            ''', unsafe_allow_html=True)
        
        # Sleep recommendations (critical for learning)
        if sleep_hours >= 8:
            rec_title = "😴 Perfect Sleep Pattern!"
            rec_text = f"You're sleeping {sleep_hours} hours - excellent! Your brain consolidates memories during sleep."
            psych_note = "🧠 Psychology says: Sleep is when your brain transfers information from short-term to long-term memory. Never skip sleep for studying!"
            activities = """
            <div class="activity-box"><strong>🌙 Sleep-Quality Tips to Boost Focus:</strong>
            • Sleep Cycle: Go to bed same time daily (helps memory consolidation)<br>
            • No Screens 1h Before Sleep (blue light delays melatonin)<br>
            • Morning Sunlight: 15 min sunlight exposure resets circadian rhythm<br>
            • Sleep = Study: 1 hour of sleep = 1 hour of study for memory!</div>
            """
            potential = calculate_potential_score(score_range_max, 5)
            st.markdown(f'''
            <div class="recommendation recommendation-good">
                <strong>{rec_title}</strong>
                <p>{rec_text}</p>
                <p style="margin-top: 0.8em;"><strong>Score Boost from Good Sleep:</strong> <span class="score-highlight">+5-7 points</span></p>
                <div class="psych-tip">{psych_note}</div>
                {activities}
            </div>
            ''', unsafe_allow_html=True)
        elif sleep_hours >= 7:
            rec_title = "😴 Good Sleep Routine"
            rec_text = f"You're getting {sleep_hours} hours - decent! Aim for 8 hours to maximize learning."
            psych_note = "🧠 Psychology says: Even 1 hour less sleep reduces attention by 30%. Try sleeping just 1 more hour - it's worth it!"
            activities = """
            <div class="activity-box"><strong>🌙 Sleep-Focus Improvements:</strong>
            • Wind-Down Ritual: 30 min of relaxing before bed (reading, music)<br>
            • Temperature: Sleep in 65-68°F room (improves focus next day)<br>
            • Post-Sleep Study: Most effective 30-60 min AFTER waking (memory peak)<br>
            • Avoid Caffeine After 3 PM (interferes with sleep quality)</div>
            """
            potential = calculate_potential_score(score_range_max, 3)
            st.markdown(f'''
            <div class="recommendation recommendation-good">
                <strong>{rec_title}</strong>
                <p>{rec_text}</p>
                <p style="margin-top: 0.8em;"><strong>Current Score Range:</strong> <span class="score-highlight">{score_range_min}-{score_range_max}</span></p>
                <p><strong>With 8 Hours Sleep:</strong> <span class="score-highlight">{int(score_range_min + 2)}-{int(potential)}</span></p>
                <div class="psych-tip">{psych_note}</div>
                {activities}
            </div>
            ''', unsafe_allow_html=True)
        else:
            rec_title = "😴 Prioritize Sleep - It's Study Fuel!"
            rec_text = f"Only {sleep_hours} hours of sleep? Your brain can't learn effectively without rest. Aim for 8 hours minimum."
            psych_note = f"🧠 Psychology says: Sleep deprivation kills grades. Students who sleep 8+ hours score 15-20% higher. You're sabotaging yourself with only {sleep_hours} hours!"
            activities = """
            <div class="activity-box"><strong>🌙 Critical Sleep-Focus Recovery:</strong>
            • Start Tonight: Commit to 8h sleep (watch grades improve in 1 week)<br>
            • Dark Room: Complete darkness triggers melatonin (10x better focus)<br>
            • No Phone in Bed: Light from phone = brain thinks it's daytime<br>
            • Sleep is Learning: Memory encoding happens ONLY during sleep<br>
            • Recovery: After 3 nights of 8h sleep, focus improves by 40%!</div>
            """
            potential = calculate_potential_score(score_range_max, 20)
            st.markdown(f'''
            <div class="recommendation recommendation-critical">
                <strong>{rec_title}</strong>
                <p>{rec_text}</p>
                <p style="margin-top: 0.8em;"><strong>Current Score Range:</strong> <span class="score-highlight">{score_range_min}-{score_range_max}</span></p>
                <p><strong>With 8 Hours Sleep + Study:</strong> <span class="score-highlight">{int(score_range_min + 15)}-{int(potential)}</span></p>
                <div class="psych-tip">{psych_note}</div>
                {activities}
            </div>
            ''', unsafe_allow_html=True)
        
        # Stress management with focus activities
        if stress_level == "Low":
            rec_title = "😌 Your Mind is Ready to Learn!"
            rec_text = "Low stress = optimal brain performance. Your prefrontal cortex (learning center) is fully active."
            psych_note = "🧠 Psychology says: Chronic stress reduces hippocampus (memory) function by 30%. You're in the perfect mental state for learning!"
            activities = """
            <div class="activity-box"><strong>🎯 Maintain Peak Focus (Low Stress):</strong>
            • Gratitude Practice: 5 min daily (keeps dopamine high for focus)<br>
            • Nature Walks: 20 min in nature = 20% focus boost<br>
            • Meditation: Even 5 min daily improves attention span<br>
            • Social Connection: Study groups boost focus & understanding</div>
            """
            st.markdown(f'''
            <div class="recommendation recommendation-good">
                <strong>{rec_title}</strong>
                <p>{rec_text}</p>
                <p style="margin-top: 0.8em;"><strong>Score Boost from Low Stress:</strong> <span class="score-highlight">+8-12 points</span></p>
                <div class="psych-tip">{psych_note}</div>
                {activities}
            </div>
            ''', unsafe_allow_html=True)
        elif stress_level == "Medium":
            rec_title = "😌 Manage Your Stress Better"
            rec_text = "Moderate stress can hurt focus. Try exercise, meditation, or time in nature."
            psych_note = "🧠 Psychology says: Just 20 minutes of exercise releases endorphins, reduces cortisol (stress hormone), and boosts focus by 40%. Walk or workout before studying!"
            activities = """
            <div class="activity-box"><strong>🎯 Stress-Relief Focus Hacks:</strong>
            • Box Breathing: 4s in, hold 4s, out 4s, hold 4s (calms nervous system)<br>
            • HIIT Exercise: 10 min intense workout before studying (+ 35% focus)<br>
            • Cold Shower: 2 min cold water = alertness boost for 2 hours<br>
            • Progressive Muscle Relaxation: Tense & release muscles (stress drops)</div>
            """
            potential = calculate_potential_score(score_range_max, 5)
            st.markdown(f'''
            <div class="recommendation recommendation-warning">
                <strong>{rec_title}</strong>
                <p>{rec_text}</p>
                <p style="margin-top: 0.8em;"><strong>Current Score Range:</strong> <span class="score-highlight">{score_range_min}-{score_range_max}</span></p>
                <p><strong>With Stress Management:</strong> <span class="score-highlight">{int(score_range_min + 4)}-{int(potential)}</span></p>
                <div class="psych-tip">{psych_note}</div>
                {activities}
            </div>
            ''', unsafe_allow_html=True)
        else:
            rec_title = "😌 Your Stress is Blocking Learning"
            rec_text = "High stress shuts down your brain's learning center. Get help NOW - this is critical."
            psych_note = "🧠 Psychology says: Under extreme stress, your amygdala (fear center) takes over and blocks the prefrontal cortex (learning). Talk to a counselor, exercise, or meditate daily!"
            activities = """
            <div class="activity-box"><strong>🎯 Emergency Stress-Relief for Focus:</strong>
            • Wim Hof Breathing: 30 deep breaths = instant calm (cortisol drops)<br>
            • 30 Min Exercise: Walking, running, or swimming (releases stress completely)<br>
            • Talk to Someone: Counselor/friend (sharing stress cuts its power by 50%)<br>
            • Yoga: 20 min yoga = mental clarity + focus boost<br>
            • Digital Detox: 2h phone-free time daily (reduces anxiety signals)</div>
            """
            potential = calculate_potential_score(score_range_max, 15)
            st.markdown(f'''
            <div class="recommendation recommendation-critical">
                <strong>{rec_title}</strong>
                <p>{rec_text}</p>
                <p style="margin-top: 0.8em;"><strong>Current Score Range:</strong> <span class="score-highlight">{score_range_min}-{score_range_max}</span></p>
                <p><strong>With Stress Relief:</strong> <span class="score-highlight">{int(score_range_min + 12)}-{int(potential)}</span></p>
                <div class="psych-tip">{psych_note}</div>
                {activities}
            </div>
            ''', unsafe_allow_html=True)
        
        # Assignment completion
        if assign_comp >= 95:
            rec_title = "✅ Perfect Assignment Game!"
            rec_text = f"{assign_comp}% completion - exceptional! You're reinforcing concepts through practice."
            psych_note = "🧠 Psychology says: Assignments trigger 'active recall' - retrieving information you've learned. This strengthens memories 3x better than just reading!"
            activities = """
            <div class="activity-box"><strong>🎯 Maximize Assignment Benefits:</strong>
            • Write Explanations: Don't just solve - explain WHY each step<br>
            • Review Errors: Spend 2x time on mistakes than correct answers<br>
            • Teach the Assignment: Explain your solution to someone else<br>
            • Connect to Real World: Ask "Where is this used in real life?"</div>
            """
            st.markdown(f'''
            <div class="recommendation recommendation-good">
                <strong>{rec_title}</strong>
                <p>{rec_text}</p>
                <p style="margin-top: 0.8em;"><strong>Score Boost from Assignments:</strong> <span class="score-highlight">+10-15 points</span></p>
                <div class="psych-tip">{psych_note}</div>
                {activities}
            </div>
            ''', unsafe_allow_html=True)
        elif assign_comp >= 80:
            rec_title = "✅ Strong Assignment Work"
            rec_text = f"Nice! {assign_comp}% completion. Complete the remaining assignments for maximum learning."
            psych_note = "🧠 Psychology says: Every assignment you skip means missing an opportunity to consolidate learning. Push to 100%!"
            activities = """
            <div class="activity-box"><strong>🎯 Finish Strong (Assignment Tips):</strong>
            • Set Deadlines: Complete 1 day BEFORE due date (pressure hurts focus)<br>
            • Study Partners: Work with classmates on assignments<br>
            • Ask Questions: If stuck > 15 min, ask teacher (faster learning)<br>
            • Review After: Spend time understanding the solution</div>
            """
            potential = calculate_potential_score(score_range_max, 5)
            st.markdown(f'''
            <div class="recommendation recommendation-good">
                <strong>{rec_title}</strong>
                <p>{rec_text}</p>
                <p style="margin-top: 0.8em;"><strong>Current Score Range:</strong> <span class="score-highlight">{score_range_min}-{score_range_max}</span></p>
                <p><strong>At 100% Completion:</strong> <span class="score-highlight">{int(score_range_min + 4)}-{int(potential)}</span></p>
                <div class="psych-tip">{psych_note}</div>
                {activities}
            </div>
            ''', unsafe_allow_html=True)
        else:
            rec_title = "✅ Complete Every Assignment"
            rec_text = f"You're at {assign_comp}% - assignments are your brain's practice ground. Missing them costs you big."
            psych_note = f"🧠 Psychology says: Skipping assignments means your brain never gets to 'practice' applying concepts. This reduces memory retention by 50%. Don't skip!"
            activities = """
            <div class="activity-box"><strong>🎯 Assignment Focus Strategies:</strong>
            • Bite-Sized Tasks: Break assignments into 15-20 min micro-tasks<br>
            • Time Blocking: Schedule specific times for each assignment<br>
            • Environment: Work in quiet place (no phone, TV, distractions)<br>
            • Reward System: Small reward after completing each assignment<br>
            • Buddy System: Find accountability partner for assignments</div>
            """
            potential = calculate_potential_score(score_range_max, 10)
            st.markdown(f'''
            <div class="recommendation recommendation-warning">
                <strong>{rec_title}</strong>
                <p>{rec_text}</p>
                <p style="margin-top: 0.8em;"><strong>Current Score Range:</strong> <span class="score-highlight">{score_range_min}-{score_range_max}</span></p>
                <p><strong>At 100% Completion:</strong> <span class="score-highlight">{int(score_range_min + 8)}-{int(potential)}</span></p>
                <div class="psych-tip">{psych_note}</div>
                {activities}
            </div>
            ''', unsafe_allow_html=True)
        
        # Feature importance
        st.markdown('<h2 class="section-header">📊 What Drives Your Success Most</h2>', unsafe_allow_html=True)
        
        feature_importance = clf.feature_importances_
        feature_names = input_df_scaled.columns
        importance_df = pd.DataFrame({
            'Factor': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False).head(5)
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.bar_chart(importance_df.set_index('Factor')['Importance'])
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Summary
        st.markdown('<h2 class="section-header">📈 Your Results Summary</h2>', unsafe_allow_html=True)
        st.markdown('<div class="summary-container">', unsafe_allow_html=True)
        
        col_sum1, col_sum2 = st.columns(2)
        with col_sum1:
            st.markdown(f'''
            <div class="result-card {grade_styles[grade_letter]}">
                <div class="result-label">Expected Grade</div>
                <div class="result-value">{grade_letter}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col_sum2:
            st.markdown(f'''
            <div class="result-card">
                <div class="result-label">Score Range</div>
                <div class="result-value">{score_range_min}-{score_range_max}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Error: {e}")

# Footer
st.markdown('''
<div class="footer">
    <p>💪 <strong>Remember:</strong> Your brain is incredible! With the right habits (sleep, breaks, consistency), you can achieve anything!</p>
    <p style="margin-top: 0.8em; color: #94a3b8;">Built on psychology-backed research to help students succeed 🎓</p>
</div>
''', unsafe_allow_html=True)