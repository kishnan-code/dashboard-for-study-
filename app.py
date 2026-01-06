import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for, redirect
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

app = Flask(__name__)

# Constants
MODEL_PATH = 'model.pkl'
DATA_PATH = 'UG_Student_CGPA_Prediction.xlsx'
IQ_HISTORY_FILE = 'iq_history.json'

import json
from iq_data import get_questions_by_level

def get_default_comprehensive_plan():
    return {
        "category": "Average Performer",
        "cat_class": "average",
        "difficulty": "Moderate",
        "allocations": {},
        "adjustments": [],
        "daily_total": 0,
        "risk_score": 0,
        "focus_score": 0,
        "focus_level": "Normal",
        "break_strategy": "Generic",
        "learning_speed": "Normal",
        "learning_tip": "Stay consistent",
        "predicted_next_gpa": 0.0,
        "ninja_rank": "Genin",
        "potential_xp": 0,
        "day_theme": "Planning Phase",
        "energy_peak": "Evening",
        "timetable": []
    }

# Global model variable
model = None

def train_model():
    global model
    if not os.path.exists(DATA_PATH):
        print("Dataset not found. Cannot train model.")
        return

    df = pd.read_excel(DATA_PATH)
    
    target_col = None
    for col in df.columns:
        if 'cgpa' in col.lower() and 'predict' not in col.lower(): # e.g. 'CGPA' or 'Final CGPA'
            target_col = col
            break
    
    if target_col:
        y = df[target_col]
    else:
        # Fallback: Create synthetic target if missing (to ensure app functionality)
        print("Target CGPA column not found. Generating synthetic target for training...")
        # Synthetic formula: Avg(Sem1, Sem2) + slight boost from habits
        s1 = df.get('Sem_1_GPA', df.get('Sem1', 0))
        s2 = df.get('Sem_2_GPA', df.get('Sem2', 0))
        attendance = df.get('Attendance (%)', df.get('Attendance', 0))
        study = df.get('Study_Hours_Per_Week', df.get('Study Hours', 0))
        
        # Synthetic CGPA
        df['synthetic_cgpa'] = (s1 + s2) / 2 + (study * 0.02) + (attendance * 0.005)
        # Clamp to 10
        df['synthetic_cgpa'] = df['synthetic_cgpa'].clip(upper=10)
        y = df['synthetic_cgpa']

    # Encode Gender (Male=1, Female=0) to match form
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].apply(lambda x: 1 if str(x).lower().strip() in ['male', 'm', '1'] else 0)
    
    # Select features strictly matching the prediction form order
    # features list order: [Gender, Age, Attendance, HSC, Internal, Participation, Projects, Study, Backlogs, Sem1, Sem2]
    
    # Map dataset columns to these expected features
    # Dataset cols: ['Student_ID', 'Gender', 'Age', 'Attendance (%)', 'HighSchool_GPA', 'Internal_Assessments (%)', 'Participation_Score', 'Projects_Completed', 'Study_Hours_Per_Week', 'Backlogs', 'Sem_1_GPA', 'Sem_2_GPA']
    
    feature_mapping = {
        'Gender': 'Gender',
        'Age': 'Age',
        'Attendance': 'Attendance (%)',
        'HighSchool_GPA': 'HighSchool_GPA',
        'Internal_Assessments': 'Internal_Assessments (%)',
        'Participation_Score': 'Participation_Score',
        'Projects_Completed': 'Projects_Completed',
        'Study_Hours': 'Study_Hours_Per_Week',
        'Backlogs': 'Backlogs',
        'Sem1_GPA': 'Sem_1_GPA', # or Sem_1_GPA
        'Sem2_GPA': 'Sem_2_GPA'
    }
    
    # Create a clean X dataframe with ordered columns
    X_clean = pd.DataFrame()
    
    # 0: Gender
    X_clean['Gender'] = df.get('Gender', 0)
    # 1: Age
    X_clean['Age'] = df.get('Age', 20)
    # 2: Attendance
    X_clean['Attendance'] = df.get('Attendance (%)', df.get('Attendance', 0))
    # 3: HighSchool_GPA
    X_clean['HighSchool_GPA'] = df.get('HighSchool_GPA', 0)
    # 4: Internal_Assessments
    X_clean['Internal_Assessments'] = df.get('Internal_Assessments (%)', 0)
    # 5: Participation_Score
    X_clean['Participation_Score'] = df.get('Participation_Score', 0)
    # 6: Projects_Completed
    X_clean['Projects_Completed'] = df.get('Projects_Completed', 0)
    # 7: Study_Hours
    X_clean['Study_Hours'] = df.get('Study_Hours_Per_Week', df.get('Study Hours', 0))
    # 8: Backlogs
    X_clean['Backlogs'] = df.get('Backlogs', 0)
    # 9: Sem1_GPA
    X_clean['Sem1_GPA'] = df.get('Sem_1_GPA', df.get('Sem1', 0))
    # 10: Sem2_GPA
    X_clean['Sem2_GPA'] = df.get('Sem_2_GPA', df.get('Sem2', 0))

    X = X_clean
    
    # Train
    regr = LinearRegression()
    # Handle NaN
    X = X.fillna(0)
    regr.fit(X, y)
    model = regr
    
    # Save
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print("Model trained and saved.")

def load_or_train_model():
    global model
    try:
        with open(MODEL_PATH, 'rb') as f:
            loaded_content = pickle.load(f)
            
        # Verify it's a valid estimator with predict method
        if hasattr(loaded_content, 'predict'):
            model = loaded_content
            # Check feature count if available (sklearn models usually have this)
            if hasattr(model, 'n_features_in_') and model.n_features_in_ != 11:
                print(f"Model validation failed: Expected 11 features, found {model.n_features_in_}. Retraining...")
                train_model()
            else:
                print("Model loaded successfully.")
        else:
            print("Loaded model is not a valid estimator. Retraining...")
            train_model()
            
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Retraining model from dataset...")
        train_model()

# Global context to store last prediction for navigation
last_prediction_context = {}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result')
def result_page():
    global last_prediction_context
    if not last_prediction_context:
        # If no context, redirect to home
        return render_template('index.html')
    return render_template('result.html', **last_prediction_context)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global last_prediction_context
    
    if request.method == 'GET':
        return redirect(url_for('home'))

    if model is None:
        return render_template('error.html', message="Model not available.")

    # Extract features from form
    try:
        features = [
            int(request.form.get('gender', 0)),
            int(request.form.get('age', 20)),
            float(request.form.get('attendance', 0)),
            float(request.form.get('highschool_gpa', 0)),
            float(request.form.get('internal_assessments', 0)),
            float(request.form.get('participation_score', 0)),
            int(request.form.get('projects_completed', 0)),
            float(request.form.get('study_hours', 0)),
            int(request.form.get('backlogs', 0)),
            float(request.form.get('sem1_gpa', 0)),
            float(request.form.get('sem2_gpa', 0))
        ]
        
        # Prediction
        raw_prediction = model.predict([features])
        
        # Robust scalar extraction
        try:
            if isinstance(raw_prediction, (np.ndarray, list)):
                 flat_pred = np.ravel(raw_prediction)
                 if len(flat_pred) > 0:
                     prediction = flat_pred[0]
                 else:
                     prediction = 0.0
            else:
                 prediction = raw_prediction

            if hasattr(prediction, 'item'): 
                prediction = prediction.item()
            else:
                prediction = float(prediction)
        except Exception as e:
            print(f"Error converting prediction: {e}")
            prediction = 0.0

        prediction = round(prediction, 2)
        
        # Store context for /result GET route
        last_prediction_context = {
            'student_name': request.form.get('student_name'),
            'predicted_cgpa': prediction,
            'attendance': features[2],
            'study_hours': features[7],
            'projects_completed': features[6],
            'backlogs': features[8],
            'sem1_gpa': features[9],
            'sem2_gpa': features[10],
            'highschool_gpa': features[3],
            'internal_assessments': features[4],
            'participation_score': features[5],
            'comprehensive_plan': get_default_comprehensive_plan(),
            'plan': None,
            'preserved_data': None,
            'daily_hours': features[7]
        }
        
        return redirect(url_for('result_page'))
                               
    except Exception as e:
        print(f"Prediction error: {e}")
        try:
            s1 = float(request.form.get('sem1_gpa', 0))
            s2 = float(request.form.get('sem2_gpa', 0))
            fallback_cgpa = (s1 + s2) / 2
            
            att = float(request.form.get('attendance', 0))
            sh = float(request.form.get('study_hours', 0))
            pc = int(request.form.get('projects_completed', 0))
            bl = int(request.form.get('backlogs', 0))
            hsc = float(request.form.get('highschool_gpa', 0))
            ia = float(request.form.get('internal_assessments', 0))
            ps = float(request.form.get('participation_score', 0))
        except ValueError:
            fallback_cgpa = 0.0
            att = 0.0
            sh = 0.0
            pc = 0
            bl = 0
            hsc = 0.0
            ia = 0.0
            ps = 0.0
            s1 = 0.0
            s2 = 0.0

        last_prediction_context = {
            'student_name': request.form.get('student_name'),
            'predicted_cgpa': round(fallback_cgpa, 2),
            'attendance': att,
            'study_hours': sh,
            'projects_completed': pc,
            'backlogs': bl,
            'sem1_gpa': s1,
            'sem2_gpa': s2,
            'highschool_gpa': hsc,
            'internal_assessments': ia,
            'participation_score': ps,
            'comprehensive_plan': get_default_comprehensive_plan(),
            'plan': None,
            'preserved_data': None,
            'daily_hours': sh
        }

        return redirect(url_for('result_page'))

@app.route('/planner', methods=['GET', 'POST'])
def planner():
    if request.method == 'POST':
        daily_hours = float(request.form.get('daily_hours', 0))
        subjects = request.form.getlist('subject_name')
        difficulties = request.form.getlist('difficulty')
        tasks = request.form.getlist('tasks')
        exam_dates = request.form.getlist('exam_date')
        
        plan = []
        total_weight = 0
        
        # Advanced Ninja Planning Logic
        try:
            # Extract inputs (defaults to safe values if missing)
            s1 = float(request.form.get('sem1_gpa', 0))
            s2 = float(request.form.get('sem2_gpa', 0))
            gpa = (s1 + s2) / 2 if s1 and s2 else s1 or s2 or 0
            attendance = float(request.form.get('attendance', 0))
            backlogs = int(request.form.get('backlogs', 0))
            internal = float(request.form.get('internal_assessments', 0))
            projects = int(request.form.get('projects_completed', 0))
        except ValueError:
            gpa = 0.0
            attendance = 0.0
            backlogs = 0
            internal = 0.0
            projects = 0

        # 2. Student Classification
        category = "Average Performer"
        cat_class = "average" # for styling
        if gpa >= 8.0 and backlogs == 0:
            category = "High Performer"
            cat_class = "high"
        elif gpa < 6.0 or backlogs > 2 or attendance < 75:
            category = "At-Risk Student"
            cat_class = "risk"
        
        # 3. Difficulty Level
        difficulty_level = "Moderate"
        if gpa >= 8.0:
            difficulty_level = "Advanced"
        elif gpa < 6.0:
            difficulty_level = "Basic + Recovery"

        # 4. Time Distribution Logic
        # Ratios (Concept, Practice, Revision, Backlog, Projects)
        if category == "High Performer":
            ratios = [0.30, 0.30, 0.20, 0.00, 0.20]
        elif category == "Average Performer":
            ratios = [0.40, 0.30, 0.20, 0.10, 0.10]
        else: # At-Risk
            ratios = [0.50, 0.20, 0.20, 0.30, 0.00] # Normalized below if sum > 1
        
        daily_mins = daily_hours * 60
        
        # --- NEW ADVANCED METRICS & LOGIC ---
        
        # 1. Academic Risk Score
        gpa_gap = max(0, 6.0 - gpa)
        risk_score = (backlogs * 20) + (100 - attendance) + (gpa_gap * 10)
        risk_score = min(100, max(0, risk_score)) # Clamp 0-100
        
        # 2. Focus Consistency Score & Break Strategy
        # Focus = (Study Hours * Attendance) / 100. Typical: 20 * 80 / 100 = 16.
        focus_score = (daily_hours * 7 * attendance) / 100 
        if focus_score < 12:
            break_strategy = "Pomodoro (25m Study / 5m Break)"
            focus_level = "Low (Needs Momentum)"
        else:
            break_strategy = "Deep Work (50m Focus / 10m Break)"
            focus_level = "High (Flow State)"
            
        # 3. Learning Speed Index
        # Efficiency = Internal Marks / Study Hours.
        # Max Internal=20. Study=20/wk. Ratio ~ 1.0.
        safe_study = daily_hours * 7 if daily_hours > 0 else 1
        learning_ratio = internal / (safe_study / 5) # Normalize roughly
        if learning_ratio > 1.2:
            learning_speed = "Fast Learner"
            learning_tip = "Increase difficulty to avoid boredom."
        elif learning_ratio > 0.8:
            learning_speed = "Balanced"
            learning_tip = "Maintain current pacing."
        else:
            learning_speed = "Needs Repetition"
            learning_tip = "Use more examples and visual aids."
            
        # 4. Predicted GPA Improvement
        # Simple heuristic: +0.2 for every 5 hrs study above 10, +0.1 for high attendance
        base_improvement = 0
        if daily_hours * 7 > 15:
            base_improvement += ((daily_hours * 7) - 15) * 0.05
        if attendance > 85:
            base_improvement += 0.2
        if backlogs > 0:
            base_improvement -= (backlogs * 0.1)
            
        predicted_next_gpa = min(10.0, gpa + base_improvement)
        
        # 5. Dynamic Time Allocation Adjustments (Exam Mode)
        # We need to look ahead at exams.
        is_exam_near = False
        
        # 5. Dynamic Time Allocation Adjustments (Exam Mode)
        # We need to look ahead at exams.
        adjustments = []
        is_exam_near = False
        
        # Ratios (Concept, Practice, Revision, Backlog, Projects)
        if category == "High Performer":
            ratios = [0.30, 0.30, 0.20, 0.00, 0.20]
        elif category == "Average Performer":
            ratios = [0.40, 0.30, 0.20, 0.10, 0.10]
        else: # At-Risk
            ratios = [0.40, 0.20, 0.20, 0.20, 0.00] 
            
        # Check for Exam Mode (Need to peek at dates first)
        min_days = 999
        for i in range(len(subjects)):
             if i < len(exam_dates) and exam_dates[i]:
                 try:
                     ed = datetime.strptime(exam_dates[i], '%Y-%m-%d').date()
                     d = (ed - datetime.now().date()).days
                     if d < min_days: min_days = d
                 except: pass
        
        if min_days <= 7:
            is_exam_near = True
            ratios = [0.10, 0.40, 0.50, 0.00, 0.00] # 90% Practice/Rev
            adjustments.append("🔥 EXAM MODE ACTIVE: Switched to Revision/Practice focus.")
        
        allocations = {
            "Concept Study": int(daily_mins * ratios[0]),
            "Practice": int(daily_mins * ratios[1]),
            "Revision": int(daily_mins * ratios[2]),
            "Backlog Recovery": int(daily_mins * ratios[3]),
            "Projects/Creative": int(daily_mins * ratios[4])
        }
        
        # Refine adjustments
        if attendance < 75 and not is_exam_near:
            allocations["Revision"] += 30
            allocations["Concept Study"] = max(0, allocations["Concept Study"] - 30)
            adjustments.append("Attendance Alert: Added catch-up revision.")

        if learning_speed == "Fast Learner" and not is_exam_near:
            allocations["Projects/Creative"] += 15
            allocations["Concept Study"] = max(0, allocations["Concept Study"] - 15)
            adjustments.append("Fast Learner: Added challenge tasks.")

        # 6. Gamification / Rank
        ninja_rank = "Genin"
        xp_gain = 0
        if risk_score < 20 and gpa >= 8.5:
            ninja_rank = "Kage (Elite)"
            xp_gain = 500
        elif risk_score < 40:
            ninja_rank = "Jonin (Expert)"
            xp_gain = 300
        elif risk_score < 70:
            ninja_rank = "Chunin (Intermediate)"
            xp_gain = 150
        else:
            ninja_rank = "Genin (Beginner)"
            xp_gain = 50
            
        # 7. Generate Daily Timetable
        timetable = []
        
        # Chronotype Logic
        energy_peak = request.form.get('energy_peak', 'Evening')
        if energy_peak == 'Morning':
            start_str = "06:00 AM"
        elif energy_peak == 'Night':
            start_str = "09:00 PM"
        else:
            start_str = "06:00 PM"
        study_start = datetime.strptime(start_str, "%I:%M %p")
        
        # Day & Subject Rotation
        current_day = datetime.now().strftime('%A')
        is_weekend = current_day in ['Saturday', 'Sunday']
        
        # Rotation Map
        rotation_map = {
            'Monday': {'Major': 'Math / Logic', 'Minor': 'Revision'},
            'Tuesday': {'Major': 'Programming', 'Minor': 'Practice'},
            'Wednesday': {'Major': 'Core Subject', 'Minor': 'Backlog'},
            'Thursday': {'Major': 'Programming', 'Minor': 'Project'},
            'Friday': {'Major': 'Core Subject', 'Minor': 'Mock Test'},
            'Saturday': {'Major': 'Weak Areas', 'Minor': 'Revision'},
            'Sunday': {'Major': 'Backlogs', 'Minor': 'Planning'}
        }
        todays_focus = rotation_map.get(current_day, {'Major': 'General', 'Minor': 'General'})
        
        # --- TIMETABLE SLOT LOGIC ---
        
        # A. Exam-Week Mode (Intense Hour-Wise)
        if is_exam_near:
             slots = [
                 {"activity": "Quick Revision", "ratio": 0.2, "focus": "Formulas & Notes", "color": "#28a745"}, 
                 {"activity": "MCQ Practice", "ratio": 0.2, "focus": "Speed Test", "color": "#ffc107"}, 
                 {"activity": "Long Answers", "ratio": 0.2, "focus": "Detailed Writing", "color": "#ffc107"},
                 {"activity": "Weak Topic Drill", "ratio": 0.2, "focus": "Targeted Repair", "color": "#dc3545"}, 
                 {"activity": "Mock Test", "ratio": 0.2, "focus": "Full Length", "color": "#17a2b8"} 
             ]
             
        # B. Micro-Block / Pomodoro (Low Focus)
        elif focus_score < 12:
             # ... (Pomodoro Logic same as before)
             activities = [
                 {"activity": f"{todays_focus['Major']} Concept", "focus": "Small Topic", "color": "#007bff"},
                 {"activity": f"{todays_focus['Minor']} Drill", "focus": "Solve 5 Qs", "color": "#ffc107"},
                 {"activity": "Rapid Review", "focus": "Flashcards", "color": "#28a745"}
             ]
             # Generate 30m blocks (25+5) until time fills
             current_time = study_start
             start_mins = daily_hours * 60
             
             count = 0
             while start_mins >= 30:
                 act = activities[count % 3]
                 # Study Block
                 end_study = current_time + timedelta(minutes=25)
                 timetable.append({
                     "time": f"{current_time.strftime('%I:%M %p')} - {end_study.strftime('%I:%M %p')}",
                     "activity": act['activity'],
                     "focus": act['focus'],
                     "type": "study",
                     "color": act['color']
                 })
                 # Break
                 end_break = end_study + timedelta(minutes=5)
                 timetable.append({
                     "time": f"{end_study.strftime('%I:%M %p')} - {end_break.strftime('%I:%M %p')}",
                     "activity": "Micro-Break",
                     "focus": "Stretch / Breath",
                     "type": "break",
                     "color": "#f8f9fa"
                 })
                 current_time = end_break
                 start_mins -= 30
                 count += 1
             
             # Skip standard slot logic for Pomodoro users
             slots = [] 

        # C. Weekend Intensive Mode
        elif is_weekend:
             slots = [
                 {"activity": f"{todays_focus['Major']} Deep Dive", "ratio": 0.4, "focus": "Hard Concepts", "color": "#17a2b8"},
                 {"activity": "Practice Marathon", "ratio": 0.3, "focus": "50+ Questions", "color": "#ffc107"},
                 {"activity": f"{todays_focus['Minor']} Review", "ratio": 0.3, "focus": "Weekly Recap", "color": "#28a745"}
             ]

        # D. Backlog Recovery (At-Risk)
        elif category == "At-Risk Student":
             slots = [
                 {"activity": "Backlog Clearing", "ratio": 0.5, "focus": "Old Modules", "color": "#dc3545"}, 
                 {"activity": "Current Syllabus", "ratio": 0.25, "focus": "Keep Up", "color": "#007bff"},
                 {"activity": "Revision", "ratio": 0.15, "focus": "Basics", "color": "#28a745"},
                 {"activity": "Practice", "ratio": 0.10, "focus": "Easy Problems", "color": "#ffc107"}
             ]
        
        # E. Standard Day (Difficulty Gradient based on Chronotype)
        else:
             # Define standard blocks
             block_easy = {"activity": "Warm-up Revision", "ratio": 0.2, "focus": "Yesterday's Topics", "color": "#28a745"}
             block_med = {"activity": f"{todays_focus['Major']} Concepts", "ratio": 0.4, "focus": "New Material", "color": "#007bff"}
             block_hard = {"activity": "Deep Practice", "ratio": 0.4, "focus": "Application / Projects", "color": "#6610f2"}
             
             if energy_peak == 'Morning':
                 # Morning Peak: Hard -> Med -> Easy
                 slots = [block_hard, block_med, block_easy]
             elif energy_peak == 'Night':
                 # Night Peak: Easy -> Med -> Hard (Ramp up)
                 slots = [block_easy, block_med, block_hard]
             else:
                 # Evening/Standard: Easy -> Med -> Hard
                 slots = [block_easy, block_med, block_hard]
        
        # Update Comprehensive Plan
        comprehensive_plan = {
            "category": category,
            "cat_class": cat_class,
            "difficulty": difficulty_level,
            "allocations": allocations,
            "adjustments": adjustments,
            "daily_total": daily_hours,
            
            # New Metrics
            "risk_score": round(risk_score, 1),
            "focus_score": round(focus_score, 1),
            "focus_level": focus_level,
            "break_strategy": break_strategy,
            "learning_speed": learning_speed,
            "learning_tip": learning_tip,
            "predicted_next_gpa": round(predicted_next_gpa, 2),
            "ninja_rank": ninja_rank,
            "potential_xp": xp_gain,
            "day_theme": f"{current_day}: {todays_focus['Major']} + {todays_focus['Minor']}",
            "energy_peak": energy_peak
        }

        # --- GENERATE TIMETABLE FROM SLOTS (If not Pomodoro) ---
        if slots:
            current_time = study_start
            total_mins_available = daily_hours * 60
            
            for i, slot in enumerate(slots):
                duration = int(total_mins_available * slot["ratio"])
                if duration < 15: continue
                
                # Add study block
                end_time = current_time + timedelta(minutes=duration)
                timetable.append({
                    "time": f"{current_time.strftime('%I:%M %p')} - {end_time.strftime('%I:%M %p')}",
                    "activity": slot["activity"],
                    "focus": slot["focus"],
                    "type": "study",
                    "duration": duration,
                    "color": slot.get("color", "#fff")
                })
                current_time = end_time
                
                # Add break if not last slot
                if i < len(slots) - 1:
                    break_duration = 10 
                    end_break = current_time + timedelta(minutes=break_duration)
                    timetable.append({
                        "time": f"{current_time.strftime('%I:%M %p')} - {end_break.strftime('%I:%M %p')}",
                        "activity": "Break",
                        "focus": "Refresh / Hydrate",
                        "type": "break",
                        "color": "#f8f9fa"
                    })
                    current_time = end_break
            
            comprehensive_plan["timetable"] = timetable

        # Calculate weights for Subject Plan (Existing Logic preserved)
        temp_data = []
        today = datetime.now().date()
        
        for i in range(len(subjects)):
            if not subjects[i]: continue
            
            try:
                diff = int(difficulties[i])
                tsk = int(tasks[i])
                
                exam_date_str = exam_dates[i]
                if exam_date_str:
                    exam_date = datetime.strptime(exam_date_str, '%Y-%m-%d').date()
                    days_left = (exam_date - today).days
                else:
                    days_left = 30 
                
                if days_left <= 0: days_left = 1 
                
                weight = (diff * tsk) / days_left
                total_weight += weight
                
                temp_data.append({
                    'subject': subjects[i],
                    'difficulty': diff,
                    'tasks': tsk,
                    'days_left': days_left,
                    'weight': weight
                })
            except ValueError:
                continue

        for item in temp_data:
            if total_weight > 0:
                hours = (item['weight'] / total_weight) * daily_hours
            else:
                hours = 0
            
            plan.append({
                'subject': item['subject'],
                'hours': round(hours, 1),
                'tasks': item['tasks'],
                'days_left': item['days_left']
            })

        # Prepare preserved data for re-populating form
        preserved_data = []
        for i in range(len(subjects)):
             if i < len(subjects) and subjects[i]: # Ensure valid index and non-empty subject
                preserved_data.append({
                    'subject': subjects[i],
                    'difficulty': difficulties[i] if i < len(difficulties) else 5,
                    'tasks': tasks[i] if i < len(tasks) else 1,
                    'exam_date': exam_dates[i] if i < len(exam_dates) else ''
                })
            
        return render_template('planner.html', plan=plan, daily_hours=daily_hours,
                               comprehensive_plan=comprehensive_plan, # New Advanced Plan
                               preserved_data=preserved_data, # Pass back for form repopulation
                               # Pass defaults logic
                               student_name=request.form.get('student_name', "Ninja Student"), 
                               predicted_cgpa=round(gpa, 2), # Use calculated GPA
                               attendance=attendance,
                               study_hours=daily_hours, # usage context
                               projects_completed=projects,
                               backlogs=backlogs,
                               sem1_gpa=s1,
                               sem2_gpa=s2,
                               highschool_gpa=float(request.form.get('highschool_gpa', 0) or 0),
                               internal_assessments=internal,
                               participation_score=float(request.form.get('participation_score', 0) or 0))

    return render_template('planner.html',
                           student_name="Ninja Student",
                           predicted_cgpa=0.0,
                           attendance=0,
                           study_hours=4,
                           projects_completed=0,
                           backlogs=0,
                           sem1_gpa=0,
                           sem2_gpa=0,
                           highschool_gpa=0,
                           internal_assessments=0,
                           participation_score=0,
                           comprehensive_plan=get_default_comprehensive_plan(),
                           plan=None,
                           preserved_data=None,
                           daily_hours=4)

@app.route('/history')
def history():
    return "History feature coming soon"

# --- IQ TEST MODULE ---

def load_iq_history():
    if os.path.exists(IQ_HISTORY_FILE):
        try:
            with open(IQ_HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_iq_result(result):
    history = load_iq_history()
    history.append(result)
    with open(IQ_HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)

@app.route('/iq')
def iq_dashboard():
    global last_prediction_context
    # Pass initial stats
    history = load_iq_history()
    total_tests = len(history)
    avg_score = sum([h['score'] for h in history]) / total_tests if total_tests > 0 else 0
    
    student_name = last_prediction_context.get('student_name', 'Student')
    
    return render_template('iq_dashboard.html', total_tests=total_tests, avg_score=round(avg_score, 1), student_name=student_name)

@app.route('/iq/test/<level>')
def iq_test(level):
    # Questions logic
    questions = get_questions_by_level(level)
    if not questions:
        return "Level not found", 404
    return render_template('iq_test.html', level=level, questions=questions)

@app.route('/api/iq/submit', methods=['POST'])
def submit_iq_test():
    data = request.json
    # data structure: { 'level': '1', 'answers': { '101': '32', ... }, 'time_spent': 120 }
    
    level = data.get('level')
    user_answers = data.get('answers', {})
    time_spent = data.get('time_spent', 0)
    
    questions = get_questions_by_level(level)
    correct_count = 0
    total_questions = len(questions)
    
    category_performance = {} # e.g. {'Math': {'correct': 2, 'total': 3}}
    
    for q in questions:
        qid = str(q['id'])
        q_type = q.get('type', 'General')
        
        if q_type not in category_performance:
            category_performance[q_type] = {'correct': 0, 'total': 0}
        category_performance[q_type]['total'] += 1
            
        if qid in user_answers and user_answers[qid] == q['answer']:
            correct_count += 1
            category_performance[q_type]['correct'] += 1
            
    score = (correct_count / total_questions) * 100 if total_questions > 0 else 0
    
    # Add Difficulty Bonus
    bonus = 0
    if level == '2': bonus = 5
    if level == '3': bonus = 10
    if level == '4': bonus = 20
    
    final_score = min(100, score + bonus) if score > 0 else 0 
    
    result = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'level': level,
        'raw_score': score,
        'bonus': bonus,
        'score': final_score,
        'correct': correct_count,
        'total': total_questions,
        'time_spent': time_spent,
        'category_performance': category_performance
    }
    
    save_iq_result(result)
    
    return jsonify({
        'success': True,
        'result': result,
        'redirect': '/iq'
    })

@app.route('/api/iq/history')
def get_iq_history():
    return jsonify(load_iq_history())




# Initialize model on startup
load_or_train_model()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
