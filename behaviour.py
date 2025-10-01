import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
import threading
import cv2
import numpy as np
import os
from PIL import Image, ImageTk
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization
import mediapipe as mp
from collections import deque
import time
import math
import json
from datetime import datetime


class ASDEmotionDetectorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SpectrAble - ASD Behavioral Analysis System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        self.stop_flag = False
        self.model = None
        self.face_cascade = None
        self.labels = {
            0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 
            4: 'Neutral', 5: 'Sad', 6: 'Surprise'
        }
        
        # Initialize MediaPipe for advanced facial analysis
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # ASD-specific behavioral tracking variables
        self.init_asd_tracking()
        
        self.setup_ui()
        self.load_model_and_cascade()
    
    def init_asd_tracking(self):
        """Initialize ASD-specific behavioral tracking variables"""
        # Focused Attention Tracking
        self.attention_history = deque(maxlen=300)  # 10 seconds at 30fps
        self.focus_start_time = None
        self.focus_duration_threshold = 3.0  # seconds
        self.attention_scores = deque(maxlen=100)
        self.distraction_events = []
        
        # Eye Contact Tracking
        self.eye_contact_history = deque(maxlen=600)  # 20 seconds
        self.direct_gaze_count = 0
        self.total_gaze_samples = 0
        self.eye_contact_sessions = []
        self.avoidance_patterns = []
        
        # Repetitive Behavior Tracking
        self.movement_history = deque(maxlen=200)
        self.head_position_history = deque(maxlen=150)
        self.repetitive_patterns = []
        self.pattern_detection_window = 50
        self.movement_threshold = 15.0
        
        # Behavioral State Variables
        self.current_attention_level = "Unknown"
        self.current_eye_contact_status = "No Contact"
        self.current_repetitive_behavior = "None Detected"
        
        # Session Analytics
        self.session_start_time = time.time()
        self.behavioral_events = []
        self.asd_indicators = {
            'poor_attention': 0,
            'limited_eye_contact': 0,
            'repetitive_behaviors': 0,
            'social_engagement': 0
        }
    
    def create_emotion_model(self):
        """Create CNN model for emotion detection"""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(7, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def load_model_and_cascade(self):
        """Load or create model and face cascade"""
        try:
            haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(haar_file)
            
            model_path = "facialemotionmodel.h5"
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                self.update_status("✅ Model loaded successfully!")
            else:
                self.model = self.create_emotion_model()
                self.model.save(model_path)
                self.update_status("✅ New model created!")
        except Exception as e:
            self.update_status(f"❌ Error: {e}")
    
    def setup_ui(self):
        """Setup enhanced user interface with ASD analysis panel"""
        title_label = tk.Label(self.root, text="SpectrAble ASD Behavioral Analysis", 
                              font=('Arial', 20, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(pady=15)
        
        # Main container
        main_container = tk.Frame(self.root, bg='#2c3e50')
        main_container.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Left panel for controls
        left_panel = tk.Frame(main_container, bg='#34495e', relief='raised', bd=2)
        left_panel.pack(side='left', fill='y', padx=(0, 10))
        
        # Input controls
        input_frame = tk.LabelFrame(left_panel, text="Input Source", 
                                   font=('Arial', 12, 'bold'), fg='white', bg='#34495e')
        input_frame.pack(padx=15, pady=15, fill='x')
        
        button_style = {'font': ('Arial', 10, 'bold'), 'width': 18, 'height': 2}
        
        tk.Button(input_frame, text="📹 Webcam", command=self.use_webcam,
                 bg='#3498db', fg='white', **button_style).pack(pady=5)
        
        tk.Button(input_frame, text="🎬 Video", command=self.upload_video,
                 bg='#e74c3c', fg='white', **button_style).pack(pady=5)
        
        tk.Button(input_frame, text="🖼️ Image", command=self.upload_image,
                 bg='#2ecc71', fg='white', **button_style).pack(pady=5)
        
        tk.Button(input_frame, text="📊 Generate Report", command=self.generate_asd_report,
                 bg='#9b59b6', fg='white', **button_style).pack(pady=5)
        
        # ASD Behavioral Indicators Panel
        asd_frame = tk.LabelFrame(left_panel, text="ASD Behavioral Indicators", 
                                 font=('Arial', 12, 'bold'), fg='white', bg='#34495e')
        asd_frame.pack(padx=15, pady=15, fill='both', expand=True)
        
        # Attention Level Display
        tk.Label(asd_frame, text="Focused Attention:", font=('Arial', 10, 'bold'),
                fg='white', bg='#34495e').pack(anchor='w', pady=2)
        self.attention_label = tk.Label(asd_frame, text="Monitoring...", 
                                       font=('Arial', 9), fg='#f39c12', bg='#34495e')
        self.attention_label.pack(anchor='w', padx=10)
        
        # Eye Contact Display
        tk.Label(asd_frame, text="Eye Contact:", font=('Arial', 10, 'bold'),
                fg='white', bg='#34495e').pack(anchor='w', pady=2)
        self.eye_contact_label = tk.Label(asd_frame, text="Assessing...", 
                                         font=('Arial', 9), fg='#e74c3c', bg='#34495e')
        self.eye_contact_label.pack(anchor='w', padx=10)
        
        # Repetitive Behavior Display
        tk.Label(asd_frame, text="Repetitive Behavior:", font=('Arial', 10, 'bold'),
                fg='white', bg='#34495e').pack(anchor='w', pady=2)
        self.repetitive_label = tk.Label(asd_frame, text="Observing...", 
                                        font=('Arial', 9), fg='#27ae60', bg='#34495e')
        self.repetitive_label.pack(anchor='w', padx=10)
        
        # Metrics Display
        metrics_frame = tk.Frame(asd_frame, bg='#34495e')
        metrics_frame.pack(fill='x', pady=10)
        
        tk.Label(metrics_frame, text="Session Metrics:", font=('Arial', 10, 'bold'),
                fg='white', bg='#34495e').pack()
        
        self.metrics_text = tk.Text(metrics_frame, height=8, width=25, font=('Courier', 8),
                                   bg='#2c3e50', fg='#ecf0f1')
        self.metrics_text.pack(fill='both', expand=True, pady=5)
        
        # Right panel for analysis display
        right_panel = tk.Frame(main_container, bg='#34495e', relief='raised', bd=2)
        right_panel.pack(side='right', fill='both', expand=True)
        
        # Analysis Report Area
        analysis_label = tk.Label(right_panel, text="ASD Behavioral Analysis Report", 
                                 font=('Arial', 14, 'bold'), fg='white', bg='#34495e')
        analysis_label.pack(pady=10)
        
        self.analysis_text = scrolledtext.ScrolledText(right_panel, height=35, width=80,
                                                      font=('Courier', 9), bg='#2c3e50', 
                                                      fg='#ecf0f1', wrap=tk.WORD)
        self.analysis_text.pack(padx=15, pady=10, fill='both', expand=True)
        
        # Status label
        self.status_label = tk.Label(left_panel, text="Ready to analyze ASD behavioral patterns...",
                                    font=('Arial', 9), fg='#bdc3c7', bg='#34495e',
                                    wraplength=200)
        self.status_label.pack(padx=15, pady=15)
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def update_status(self, message):
        """Update status label"""
        self.status_label.config(text=message)
        self.root.update_idletasks()
    
    def extract_features(self, image):
        """Extract features from image"""
        feature = np.array(image).reshape(1, 48, 48, 1)
        return feature / 255.0
    
    def analyze_focused_attention(self, face_landmarks, frame_shape):
        """Analyze focused attention patterns"""
        h, w, _ = frame_shape
        current_time = time.time()
        
        # Calculate gaze stability and focus metrics
        gaze_stability = self.calculate_gaze_stability(face_landmarks, w, h)
        head_stability = self.calculate_head_stability(face_landmarks, w, h)
        
        # Combined attention score
        attention_score = (gaze_stability * 0.6) + (head_stability * 0.4)
        self.attention_scores.append(attention_score)
        
        # Determine attention level
        if attention_score > 0.7:
            attention_level = "High Focus"
            if not self.focus_start_time:
                self.focus_start_time = current_time
        elif attention_score > 0.4:
            attention_level = "Moderate Focus"
        else:
            attention_level = "Distracted"
            if self.focus_start_time:
                focus_duration = current_time - self.focus_start_time
                if focus_duration > self.focus_duration_threshold:
                    self.behavioral_events.append({
                        'type': 'sustained_attention',
                        'duration': focus_duration,
                        'timestamp': current_time
                    })
                self.focus_start_time = None
            
            # Record distraction event
            self.distraction_events.append({
                'timestamp': current_time,
                'attention_score': attention_score
            })
            
            if len(self.distraction_events) > 10:  # Frequent distractions
                self.asd_indicators['poor_attention'] += 1
        
        self.current_attention_level = attention_level
        self.attention_history.append({
            'timestamp': current_time,
            'score': attention_score,
            'level': attention_level
        })
        
        return attention_level, attention_score
    
    def analyze_eye_contact(self, face_landmarks, frame_shape):
        """Analyze eye contact patterns"""
        h, w, _ = frame_shape
        current_time = time.time()
        
        # Calculate eye contact indicators
        eye_direction = self.estimate_eye_direction(face_landmarks, w, h)
        contact_probability = self.calculate_eye_contact_probability(face_landmarks, w, h)
        
        self.total_gaze_samples += 1
        
        # Determine eye contact status
        if contact_probability > 0.6 and eye_direction == 'center':
            eye_contact_status = "Direct Contact"
            self.direct_gaze_count += 1
            
            # Record positive eye contact event
            self.eye_contact_sessions.append({
                'timestamp': current_time,
                'duration': 1/30,  # Approximate frame duration
                'quality': contact_probability
            })
            
        elif contact_probability > 0.3:
            eye_contact_status = "Peripheral Gaze"
        else:
            eye_contact_status = "Avoidance/No Contact"
            
            # Record avoidance pattern
            self.avoidance_patterns.append({
                'timestamp': current_time,
                'reason': 'low_contact_probability',
                'direction': eye_direction
            })
        
        # Calculate eye contact percentage
        eye_contact_percentage = (self.direct_gaze_count / self.total_gaze_samples) * 100 if self.total_gaze_samples > 0 else 0
        
        # ASD indicator: Limited eye contact
        if eye_contact_percentage < 30 and self.total_gaze_samples > 100:
            self.asd_indicators['limited_eye_contact'] += 1
        
        self.current_eye_contact_status = f"{eye_contact_status} ({eye_contact_percentage:.1f}%)"
        
        self.eye_contact_history.append({
            'timestamp': current_time,
            'status': eye_contact_status,
            'probability': contact_probability,
            'direction': eye_direction
        })
        
        return eye_contact_status, eye_contact_percentage
    
    def analyze_repetitive_behavior(self, face_landmarks, frame_shape):
        """Analyze repetitive behavior patterns"""
        h, w, _ = frame_shape
        current_time = time.time()
        
        # Track head movements and positions
        head_position = self.extract_head_position(face_landmarks, w, h)
        self.head_position_history.append({
            'timestamp': current_time,
            'position': head_position,
            'movement': self.calculate_movement_magnitude(head_position)
        })
        
        # Detect repetitive patterns
        repetitive_behavior = "None Detected"
        
        if len(self.head_position_history) >= self.pattern_detection_window:
            patterns = self.detect_movement_patterns()
            
            if patterns['repetitive_head_movement']:
                repetitive_behavior = f"Repetitive Head Movement (Pattern: {patterns['pattern_type']})"
                
                self.behavioral_events.append({
                    'type': 'repetitive_movement',
                    'pattern': patterns['pattern_type'],
                    'frequency': patterns['frequency'],
                    'timestamp': current_time
                })
                
                self.asd_indicators['repetitive_behaviors'] += 1
            
            elif patterns['self_stimming']:
                repetitive_behavior = "Self-Stimulatory Behavior Detected"
                
                self.behavioral_events.append({
                    'type': 'self_stimming',
                    'intensity': patterns['intensity'],
                    'timestamp': current_time
                })
        
        self.current_repetitive_behavior = repetitive_behavior
        
        return repetitive_behavior
    
    def calculate_gaze_stability(self, face_landmarks, w, h):
        """Calculate gaze stability for attention analysis"""
        if not face_landmarks.multi_face_landmarks:
            return 0.0
        
        # Extract eye landmarks
        landmarks = face_landmarks.multi_face_landmarks[0].landmark
        
        # Left and right eye centers (approximate)
        left_eye_center = np.array([landmarks[33].x * w, landmarks[33].y * h])
        right_eye_center = np.array([landmarks[362].x * w, landmarks[362].y * h])
        
        # Calculate eye center
        eye_center = (left_eye_center + right_eye_center) / 2
        
        # Compare with recent history for stability
        if len(self.attention_history) > 5:
            recent_positions = [pos.get('eye_center', eye_center) for pos in list(self.attention_history)[-5:]]
            if recent_positions:
                position_variance = np.var([pos[0] for pos in recent_positions if hasattr(pos, '__len__')])
                stability = max(0, 1 - (position_variance / 1000))  # Normalize
                return stability
        
        return 0.5  # Default moderate stability
    
    def calculate_head_stability(self, face_landmarks, w, h):
        """Calculate head pose stability"""
        if not face_landmarks.multi_face_landmarks:
            return 0.0
        
        landmarks = face_landmarks.multi_face_landmarks[0].landmark
        
        # Key facial points for head pose
        nose_tip = np.array([landmarks[1].x * w, landmarks[1].y * h])
        
        if len(self.head_position_history) > 3:
            recent_positions = [entry['position'] for entry in list(self.head_position_history)[-3:]]
            position_variance = np.var([pos[1] for pos in recent_positions])  # Y-axis variance
            stability = max(0, 1 - (position_variance / 500))
            return stability
        
        return 0.5
    
    def estimate_eye_direction(self, face_landmarks, w, h):
        """Estimate eye gaze direction"""
        if not face_landmarks.multi_face_landmarks:
            return 'unknown'
        
        landmarks = face_landmarks.multi_face_landmarks[0].landmark
        
        # Simplified gaze estimation based on eye position relative to face center
        left_eye = np.array([landmarks[33].x, landmarks[33].y])
        right_eye = np.array([landmarks[362].x, landmarks[362].y])
        nose = np.array([landmarks[1].x, landmarks[1].y])
        
        eye_center = (left_eye + right_eye) / 2
        
        # Calculate relative position
        horizontal_diff = eye_center[0] - nose[0]
        
        if abs(horizontal_diff) < 0.02:
            return 'center'
        elif horizontal_diff < -0.02:
            return 'left'
        else:
            return 'right'
    
    def calculate_eye_contact_probability(self, face_landmarks, w, h):
        """Calculate probability of direct eye contact"""
        if not face_landmarks.multi_face_landmarks:
            return 0.0
        
        landmarks = face_landmarks.multi_face_landmarks[0].landmark
        
        # Eye aspect ratio calculation (simplified)
        left_eye_landmarks = [landmarks[i] for i in [33, 7, 163, 144, 145, 153]]
        right_eye_landmarks = [landmarks[i] for i in [362, 382, 381, 380, 374, 373]]
        
        # Calculate eye openness
        left_ear = self.calculate_eye_aspect_ratio(left_eye_landmarks)
        right_ear = self.calculate_eye_aspect_ratio(right_eye_landmarks)
        
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Higher EAR suggests more alert/engaged state
        if avg_ear > 0.2:
            return min(1.0, avg_ear * 2)  # Scale appropriately
        
        return 0.1
    
    def calculate_eye_aspect_ratio(self, eye_landmarks):
        """Calculate Eye Aspect Ratio"""
        if len(eye_landmarks) < 6:
            return 0.2
        
        # Vertical distances
        A = np.linalg.norm(np.array([eye_landmarks[1].x, eye_landmarks[1].y]) - 
                          np.array([eye_landmarks[5].x, eye_landmarks[5].y]))
        B = np.linalg.norm(np.array([eye_landmarks[2].x, eye_landmarks[2].y]) - 
                          np.array([eye_landmarks[4].x, eye_landmarks[4].y]))
        
        # Horizontal distance
        C = np.linalg.norm(np.array([eye_landmarks[0].x, eye_landmarks[0].y]) - 
                          np.array([eye_landmarks[3].x, eye_landmarks[3].y]))
        
        if C > 0:
            ear = (A + B) / (2.0 * C)
            return ear
        return 0.2
    
    def extract_head_position(self, face_landmarks, w, h):
        """Extract head position for movement tracking"""
        if not face_landmarks.multi_face_landmarks:
            return (0, 0)
        
        landmarks = face_landmarks.multi_face_landmarks[0].landmark
        
        # Use nose tip as reference point
        nose_x = landmarks[1].x * w
        nose_y = landmarks[1].y * h
        
        return (nose_x, nose_y)
    
    def calculate_movement_magnitude(self, current_position):
        """Calculate movement magnitude from previous position"""
        if len(self.head_position_history) == 0:
            return 0.0
        
        prev_position = self.head_position_history[-1]['position']
        distance = np.sqrt((current_position[0] - prev_position[0])**2 + 
                          (current_position[1] - prev_position[1])**2)
        
        return distance
    
    def detect_movement_patterns(self):
        """Detect repetitive movement patterns"""
        patterns = {
            'repetitive_head_movement': False,
            'self_stimming': False,
            'pattern_type': 'none',
            'frequency': 0,
            'intensity': 0
        }
        
        if len(self.head_position_history) < 20:
            return patterns
        
        # Extract recent movements
        recent_movements = [entry['movement'] for entry in list(self.head_position_history)[-20:]]
        positions = [entry['position'] for entry in list(self.head_position_history)[-20:]]
        
        # Check for repetitive patterns
        movement_threshold = self.movement_threshold
        high_movement_count = sum(1 for m in recent_movements if m > movement_threshold)
        
        if high_movement_count > 10:  # More than 50% high movement
            patterns['repetitive_head_movement'] = True
            patterns['pattern_type'] = 'high_frequency'
            patterns['frequency'] = high_movement_count / 20
            patterns['intensity'] = np.mean(recent_movements)
            
            # Check for specific patterns
            x_positions = [pos[0] for pos in positions]
            y_positions = [pos[1] for pos in positions]
            
            x_variance = np.var(x_positions)
            y_variance = np.var(y_positions)
            
            if x_variance > y_variance * 2:
                patterns['pattern_type'] = 'horizontal_movement'
            elif y_variance > x_variance * 2:
                patterns['pattern_type'] = 'vertical_movement'
            else:
                patterns['pattern_type'] = 'circular_movement'
        
        # Self-stimming detection (rapid, repetitive movements)
        if high_movement_count > 15 and np.mean(recent_movements) > movement_threshold * 1.5:
            patterns['self_stimming'] = True
            patterns['intensity'] = np.mean(recent_movements) / movement_threshold
        
        return patterns
    
    def detect_emotions_and_behaviors(self, frame):
        """Enhanced emotion detection with ASD behavioral analysis"""
        if self.model is None or self.face_cascade is None:
            return frame, ""
        
        # Convert frame for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_mesh_results = self.face_mesh.process(rgb_frame)
        
        # Traditional emotion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        behavioral_analysis = ""
        
        for (x, y, w, h) in faces:
            # Emotion detection
            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (48, 48))
            face_processed = self.extract_features(face_resized)
            
            prediction = self.model.predict(face_processed, verbose=0)
            emotion_index = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            emotion_label = self.labels[emotion_index]
            
            # ASD Behavioral Analysis
            if face_mesh_results.multi_face_landmarks:
                # Focused Attention Analysis
                attention_level, attention_score = self.analyze_focused_attention(face_mesh_results, frame.shape)
                
                # Eye Contact Analysis
                eye_contact_status, eye_contact_percentage = self.analyze_eye_contact(face_mesh_results, frame.shape)
                
                # Repetitive Behavior Analysis
                repetitive_behavior = self.analyze_repetitive_behavior(face_mesh_results, frame.shape)
                
                # Update UI labels
                self.root.after(0, lambda: self.update_behavioral_display())
                
                # Generate behavioral analysis text
                behavioral_analysis = self.generate_behavioral_analysis_report(
                    emotion_label, confidence, attention_level, attention_score,
                    eye_contact_status, eye_contact_percentage, repetitive_behavior
                )
            
            # Draw emotion detection results
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f'{emotion_label}: {confidence:.1f}%', 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw behavioral indicators
            if hasattr(self, 'current_attention_level'):
                cv2.putText(frame, f'Attention: {self.current_attention_level}', 
                           (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            if hasattr(self, 'current_eye_contact_status'):
                cv2.putText(frame, f'Eye Contact: {self.current_eye_contact_status.split("(")[0]}', 
                           (x, y+h+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        return frame, behavioral_analysis
    
    def generate_behavioral_analysis_report(self, emotion, confidence, attention_level, 
                                           attention_score, eye_contact_status, 
                                           eye_contact_percentage, repetitive_behavior):
        """Generate comprehensive behavioral analysis report"""
        current_time = time.time()
        session_duration = current_time - self.session_start_time
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║               ASD BEHAVIORAL ANALYSIS REPORT                 ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║ Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S"):<25}                     ║
║ Session Duration: {session_duration/60:.1f} minutes                         ║
║                                                              ║
║ CURRENT EMOTIONAL STATE:                                     ║
║ • Primary Emotion: {emotion:<15} (Confidence: {confidence:.1f}%)      ║
║                                                              ║
║ ASD-SPECIFIC BEHAVIORAL INDICATORS:                          ║
║                                                              ║
║ 1. FOCUSED ATTENTION ANALYSIS:                               ║
║    • Current Level: {attention_level:<25}                    ║
║    • Attention Score: {attention_score:.3f}/1.000                        ║
║    • Focus Stability: {"High" if attention_score > 0.7 else "Low" if attention_score < 0.4 else "Moderate":<15}                          ║"""

        if len(self.distraction_events) > 5:
            report += f"""
║    • Distraction Events: {len(self.distraction_events)} (Concerning)              ║
║    ⚠️  HIGH DISTRACTIBILITY DETECTED                        ║"""
        else:
            report += f"""
║    • Distraction Events: {len(self.distraction_events)} (Normal range)            ║"""
        
        report += f"""
║                                                              ║
║ 2. EYE CONTACT ANALYSIS:                                     ║
║    • Current Status: {eye_contact_status:<20}                  ║
║    • Eye Contact Rate: {eye_contact_percentage:.1f}% (Sample: {self.total_gaze_samples})         ║"""
        
        if eye_contact_percentage < 30:
            report += """
║    ⚠️  LIMITED EYE CONTACT - ASD INDICATOR                  ║"""
        elif eye_contact_percentage < 50:
            report += """
║    ⚠️  REDUCED EYE CONTACT - MONITOR                        ║"""
        else:
            report += """
║    ✅ APPROPRIATE EYE CONTACT LEVELS                        ║"""
        
        report += f"""
║                                                              ║
║ 3. REPETITIVE BEHAVIOR ANALYSIS:                             ║
║    • Current Pattern: {repetitive_behavior[:30]}{"..." if len(repetitive_behavior) > 30 else ""}║"""
        
        if "Repetitive" in repetitive_behavior:
            report += """
║    ⚠️  REPETITIVE BEHAVIORS DETECTED - ASD INDICATOR        ║"""
        elif "Self-Stimulatory" in repetitive_behavior:
            report += """
║    ⚠️  SELF-STIMULATING BEHAVIORS - MONITOR                 ║"""
        else:
            report += """
║    ✅ NO CONCERNING REPETITIVE PATTERNS                     ║"""
        
        # ASD Risk Assessment
        total_indicators = sum(self.asd_indicators.values())
        
        report += f"""
║                                                              ║
║ ASD BEHAVIORAL RISK ASSESSMENT:                              ║
║ • Poor Attention Indicators: {self.asd_indicators['poor_attention']:<5}                    ║
║ • Limited Eye Contact Events: {self.asd_indicators['limited_eye_contact']:<5}                 ║
║ • Repetitive Behavior Events: {self.asd_indicators['repetitive_behaviors']:<5}                 ║
║ • Total Risk Indicators: {total_indicators:<5}                           ║
║                                                              ║"""
        
        if total_indicators >= 15:
            report += """║ 🔴 HIGH RISK - Multiple ASD indicators present             ║
║    Recommendation: Professional evaluation suggested         ║"""
        elif total_indicators >= 8:
            report += """║ 🟡 MODERATE RISK - Some ASD indicators present            ║
║    Recommendation: Continue monitoring and assessment        ║"""
        else:
            report += """║ 🟢 LOW RISK - Few or no significant ASD indicators        ║
║    Recommendation: Continue regular developmental monitoring  ║"""
        
        report += f"""
║                                                              ║
║ BEHAVIORAL RECOMMENDATIONS:                                  ║"""
        
        # Generate specific recommendations
        recommendations = self.generate_behavioral_recommendations(
            attention_level, eye_contact_percentage, repetitive_behavior, total_indicators
        )
        
        for rec in recommendations:
            report += f"""
║ • {rec:<58} ║"""
        
        report += """
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """
        
        return report.strip()
    
    def generate_behavioral_recommendations(self, attention_level, eye_contact_percentage, 
                                           repetitive_behavior, total_indicators):
        """Generate specific behavioral intervention recommendations"""
        recommendations = []
        
        # Attention-based recommendations
        if "Distracted" in attention_level:
            recommendations.append("Use visual schedules and structured activities")
            recommendations.append("Minimize environmental distractions")
            recommendations.append("Break tasks into smaller, manageable segments")
        
        # Eye contact recommendations
        if eye_contact_percentage < 30:
            recommendations.append("Practice indirect gaze activities (looking at objects)")
            recommendations.append("Use social stories about eye contact")
            recommendations.append("Reward approximations of eye contact behavior")
        
        # Repetitive behavior recommendations
        if "Repetitive" in repetitive_behavior:
            recommendations.append("Provide sensory breaks and alternatives")
            recommendations.append("Introduce calming activities and fidget tools")
            recommendations.append("Create predictable routines to reduce anxiety")
        
        # General ASD support recommendations
        if total_indicators >= 8:
            recommendations.append("Consider comprehensive developmental evaluation")
            recommendations.append("Implement structured teaching methods")
            recommendations.append("Focus on communication and social skills development")
        
        # Default recommendations
        if not recommendations:
            recommendations.append("Continue regular developmental monitoring")
            recommendations.append("Maintain consistent routines and expectations")
            recommendations.append("Encourage social interaction opportunities")
        
        return recommendations
    
    def update_behavioral_display(self):
        """Update behavioral indicators display"""
        self.attention_label.config(text=self.current_attention_level)
        self.eye_contact_label.config(text=self.current_eye_contact_status)
        self.repetitive_label.config(text=self.current_repetitive_behavior[:30] + 
                                   ("..." if len(self.current_repetitive_behavior) > 30 else ""))
        
        # Update metrics
        session_duration = time.time() - self.session_start_time
        eye_contact_rate = (self.direct_gaze_count / self.total_gaze_samples * 100) if self.total_gaze_samples > 0 else 0
        
        metrics_text = f"""Session: {session_duration/60:.1f}m
Samples: {self.total_gaze_samples}
Eye Contact: {eye_contact_rate:.1f}%
Attention Events: {len(self.behavioral_events)}
Distractions: {len(self.distraction_events)}
Risk Indicators: {sum(self.asd_indicators.values())}"""
        
        self.metrics_text.delete('1.0', tk.END)
        self.metrics_text.insert('1.0', metrics_text)
    
    def update_analysis_display(self, analysis_text):
        """Update analysis display"""
        self.analysis_text.delete('1.0', tk.END)
        self.analysis_text.insert('1.0', analysis_text)
        self.root.update_idletasks()
    
    def generate_asd_report(self):
        """Generate comprehensive ASD assessment report"""
        if not self.behavioral_events and self.total_gaze_samples == 0:
            messagebox.showwarning("Warning", "No behavioral data collected yet!")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ASD_Assessment_Report_{timestamp}.txt"
        
        # Generate comprehensive report
        report = self.generate_comprehensive_asd_report()
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report)
            
            messagebox.showinfo("Success", f"ASD Assessment Report saved as:\n{filename}")
            self.update_status(f"✅ Report generated: {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save report: {e}")
    
    def generate_comprehensive_asd_report(self):
        """Generate detailed ASD assessment report"""
        session_duration = time.time() - self.session_start_time
        
        report = f"""
════════════════════════════════════════════════════════════════
              SPECTRABLEPROASD BEHAVIORAL ASSESSMENT REPORT
════════════════════════════════════════════════════════════════

ASSESSMENT INFORMATION:
• Date: {datetime.now().strftime("%Y-%m-%d")}
• Time: {datetime.now().strftime("%H:%M:%S")}
• Session Duration: {session_duration/60:.2f} minutes
• Total Behavioral Events: {len(self.behavioral_events)}
• Analysis Framework: ASD-Specific Behavioral Indicators

════════════════════════════════════════════════════════════════
                        EXECUTIVE SUMMARY
════════════════════════════════════════════════════════════════

This assessment analyzed three key behavioral domains associated with 
Autism Spectrum Disorder (ASD): Focused Attention, Eye Contact, and 
Repetitive Behaviors. The analysis was conducted using computer vision 
and behavioral pattern recognition technologies.

OVERALL ASSESSMENT:
• Risk Level: {"HIGH" if sum(self.asd_indicators.values()) >= 15 else "MODERATE" if sum(self.asd_indicators.values()) >= 8 else "LOW"}
• Total ASD Indicators: {sum(self.asd_indicators.values())}
• Behavioral Concerns: {len([k for k, v in self.asd_indicators.items() if v > 5])}

════════════════════════════════════════════════════════════════
                    DETAILED BEHAVIORAL ANALYSIS
════════════════════════════════════════════════════════════════

1. FOCUSED ATTENTION ANALYSIS:

Definition: Refers to a child's ability to concentrate on a specific 
task or activity for an extended period. Children with ASD might 
display challenges in maintaining focused attention on tasks of 
interest or importance.

FINDINGS:
• Total Attention Samples: {len(self.attention_history)}
• Average Attention Score: {np.mean([a['score'] for a in self.attention_history]) if self.attention_history else 0:.3f}
• Distraction Events: {len(self.distraction_events)}
• Sustained Focus Episodes: {len([e for e in self.behavioral_events if e['type'] == 'sustained_attention'])}

ATTENTION LEVEL DISTRIBUTION:"""
        
        if self.attention_history:
            attention_levels = {}
            for entry in self.attention_history:
                level = entry['level']
                attention_levels[level] = attention_levels.get(level, 0) + 1
            
            for level, count in attention_levels.items():
                percentage = (count / len(self.attention_history)) * 100
                report += f"\n• {level}: {count} instances ({percentage:.1f}%)"
        
        report += f"""

CLINICAL INTERPRETATION:
"""
        
        if len(self.distraction_events) > 10:
            report += """• HIGH DISTRACTIBILITY: Frequent attention shifting observed.
  This pattern may indicate attention regulation difficulties 
  commonly seen in ASD individuals.
"""
        
        avg_attention = np.mean([a['score'] for a in self.attention_history]) if self.attention_history else 0
        if avg_attention < 0.4:
            report += """• POOR SUSTAINED ATTENTION: Low attention scores suggest 
  difficulties maintaining focus on activities, which is 
  characteristic of ASD attention profiles.
"""
        
        report += f"""
════════════════════════════════════════════════════════════════

2. EYE CONTACT ANALYSIS:

Definition: Indicates the ability or frequency with which a child 
makes eye contact during social interactions. Limited or atypical 
eye contact is a common trait observed in ASD, affecting social 
communication.

FINDINGS:
• Total Gaze Samples: {self.total_gaze_samples}
• Direct Eye Contact Instances: {self.direct_gaze_count}
• Eye Contact Percentage: {(self.direct_gaze_count/self.total_gaze_samples*100) if self.total_gaze_samples > 0 else 0:.1f}%
• Avoidance Patterns: {len(self.avoidance_patterns)}
• Eye Contact Sessions: {len(self.eye_contact_sessions)}

EYE CONTACT PATTERN ANALYSIS:"""
        
        if self.eye_contact_history:
            contact_patterns = {}
            for entry in self.eye_contact_history:
                status = entry['status']
                contact_patterns[status] = contact_patterns.get(status, 0) + 1
            
            for pattern, count in contact_patterns.items():
                percentage = (count / len(self.eye_contact_history)) * 100
                report += f"\n• {pattern}: {count} instances ({percentage:.1f}%)"
        
        eye_contact_rate = (self.direct_gaze_count/self.total_gaze_samples*100) if self.total_gaze_samples > 0 else 0
        
        report += f"""

CLINICAL INTERPRETATION:
"""
        
        if eye_contact_rate < 20:
            report += """• SIGNIFICANTLY LIMITED EYE CONTACT: Very low eye contact rates 
  observed, which is a strong indicator of ASD-related social 
  communication differences.
"""
        elif eye_contact_rate < 40:
            report += """• REDUCED EYE CONTACT: Below-typical eye contact patterns observed, 
  suggesting possible social communication challenges associated with ASD.
"""
        
        if len(self.avoidance_patterns) > 20:
            report += """• ACTIVE GAZE AVOIDANCE: Frequent gaze avoidance patterns detected, 
  indicating possible social anxiety or sensory sensitivities 
  common in ASD.
"""
        
        report += f"""
════════════════════════════════════════════════════════════════

3. REPETITIVE BEHAVIOR ANALYSIS:

Definition: Encompasses a range of repetitive actions, behaviors, 
or interests that are persistent and often resist change. These 
behaviors might include repetitive movements, insistence on sameness, 
or rigid adherence to routines.

FINDINGS:
• Movement Samples Analyzed: {len(self.head_position_history)}
• Repetitive Movement Events: {len([e for e in self.behavioral_events if e['type'] == 'repetitive_movement'])}
• Self-Stimulatory Behaviors: {len([e for e in self.behavioral_events if e['type'] == 'self_stimming'])}
• Average Movement Intensity: {np.mean([h['movement'] for h in self.head_position_history]) if self.head_position_history else 0:.2f}

REPETITIVE PATTERN ANALYSIS:"""
        
        repetitive_events = [e for e in self.behavioral_events if e['type'] in ['repetitive_movement', 'self_stimming']]
        if repetitive_events:
            pattern_types = {}
            for event in repetitive_events:
                if 'pattern' in event:
                    pattern = event['pattern']
                    pattern_types[pattern] = pattern_types.get(pattern, 0) + 1
            
            for pattern, count in pattern_types.items():
                report += f"\n• {pattern.replace('_', ' ').title()}: {count} episodes"
        else:
            report += "\n• No significant repetitive patterns detected"
        
        report += f"""

CLINICAL INTERPRETATION:
"""
        
        repetitive_count = len([e for e in self.behavioral_events if e['type'] == 'repetitive_movement'])
        if repetitive_count > 5:
            report += """• FREQUENT REPETITIVE MOVEMENTS: Multiple episodes of repetitive 
  head movements detected, which may indicate self-regulatory 
  behaviors or stereotypies associated with ASD.
"""
        
        stimming_count = len([e for e in self.behavioral_events if e['type'] == 'self_stimming'])
        if stimming_count > 3:
            report += """• SELF-STIMULATORY BEHAVIORS: Self-stimulating behaviors observed, 
  which are common regulatory mechanisms in individuals with ASD 
  for managing sensory input or emotional states.
"""
        
        report += f"""
════════════════════════════════════════════════════════════════
                        CLINICAL RECOMMENDATIONS
════════════════════════════════════════════════════════════════

Based on the behavioral analysis results, the following recommendations 
are suggested:

IMMEDIATE INTERVENTIONS:
"""
        
        total_indicators = sum(self.asd_indicators.values())
        
        if total_indicators >= 15:
            report += """
• URGENT: Comprehensive developmental evaluation recommended
• Contact pediatric developmental specialist or autism diagnostic team
• Implement immediate environmental modifications
• Begin structured behavioral support strategies
"""
        elif total_indicators >= 8:
            report += """
• Schedule comprehensive developmental assessment
• Implement targeted behavioral interventions
• Monitor progress with regular assessments
• Consider early intervention services consultation
"""
        else:
            report += """
• Continue regular developmental monitoring
• Implement preventive developmental support strategies
• Schedule follow-up assessment in 6 months
"""
        
        report += f"""
SPECIFIC BEHAVIORAL STRATEGIES:

Focused Attention Support:
• Use visual schedules and structured activity sequences
• Implement attention-building games and activities  
• Create distraction-free learning environments
• Use timers and visual cues for task transitions

Eye Contact Development:
• Practice indirect gaze activities (looking at objects together)
• Use social stories about eye contact and social interaction
• Reward approximations and attempts at eye contact
• Avoid forcing direct eye contact, respect comfort levels

Repetitive Behavior Management:
• Provide appropriate sensory breaks and alternatives
• Introduce calming strategies and self-regulation tools
• Create predictable routines to reduce anxiety-driven behaviors
• Use positive behavioral supports for adaptive alternatives

════════════════════════════════════════════════════════════════
                            DISCLAIMER
════════════════════════════════════════════════════════════════

This automated behavioral analysis is a screening tool designed to 
identify potential ASD-related behavioral patterns. It is NOT a 
diagnostic instrument and should not be used as the sole basis for 
clinical decisions.

For comprehensive ASD assessment, please consult qualified healthcare 
professionals including:
• Developmental Pediatricians
• Child Psychologists
• Speech-Language Pathologists  
• Occupational Therapists
• ASD Diagnostic Specialists

This report should be shared with healthcare providers as part of a 
comprehensive developmental evaluation process.

════════════════════════════════════════════════════════════════
Report Generated by SpectrAble ASD Behavioral Analysis System
Version 2.0 | Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
════════════════════════════════════════════════════════════════
        """
        
        return report.strip()
    
    def use_webcam(self):
        """Start webcam detection with ASD behavioral analysis"""
        self.stop_flag = False
        threading.Thread(target=self.process_webcam, daemon=True).start()
    
    def process_webcam(self):
        """Process webcam feed with comprehensive ASD analysis"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot access webcam!")
            return
        
        self.update_status("✅ ASD behavioral analysis active. Press 'Q' to stop.")
        
        while not self.stop_flag:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame, behavioral_analysis = self.detect_emotions_and_behaviors(frame)
            
            if behavioral_analysis:
                self.update_analysis_display(behavioral_analysis)
            
            cv2.imshow("SpectrAble - ASD Behavioral Analysis", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.update_status("⏹️ Analysis stopped")
    
    def upload_video(self):
        """Upload and process video with ASD analysis"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
        )
        
        if file_path:
            self.stop_flag = False
            threading.Thread(target=self.process_video, args=(file_path,), daemon=True).start()
    
    def process_video(self, video_path):
        """Process video file with ASD behavioral analysis"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot open video file!")
            return
        
        self.update_status(f"✅ Analyzing: {os.path.basename(video_path)}")
        
        while not self.stop_flag:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame, behavioral_analysis = self.detect_emotions_and_behaviors(frame)
            
            if behavioral_analysis:
                self.update_analysis_display(behavioral_analysis)
            
            cv2.imshow("SpectrAble - Video ASD Analysis", frame)
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.update_status("✅ Video analysis completed")
    
    def upload_image(self):
        """Upload and process image with ASD analysis"""
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )
        
        if file_path:
            threading.Thread(target=self.process_image, args=(file_path,), daemon=True).start()
    
    def process_image(self, image_path):
        """Process single image with ASD behavioral analysis"""
        frame = cv2.imread(image_path)
        if frame is None:
            messagebox.showerror("Error", "Cannot open image!")
            return
        
        frame, behavioral_analysis = self.detect_emotions_and_behaviors(frame)
        
        if behavioral_analysis:
            self.update_analysis_display(behavioral_analysis)
        
        cv2.imshow("SpectrAble - Image ASD Analysis", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def on_closing(self):
        """Handle window closing"""
        self.stop_flag = True
        cv2.destroyAllWindows()
        self.root.destroy()
    
    def run(self):
        """Start the enhanced ASD behavioral analysis application"""
        self.root.mainloop()


if __name__ == "__main__":
    app = ASDEmotionDetectorGUI()
    app.run()
