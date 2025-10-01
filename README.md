# face_and_behaviour_analysis
Features
 Real-time emotion detection using facial expression CNN model
 Behavioral analysis tracking specific to ASD indicators
 Advanced facial landmark detection with MediaPipe Face Mesh
 Support for webcam, video, and image input processing
 Graphical User Interface built with Tkinter
 Comprehensive ASD behavioral report generation
 Installation
 Requirements
 Python 3.x
 TensorFlow
 Keras
 OpenCV
 MediaPipe
 NumPy
 Pillow
 Tkinter (usually included with Python)
 Install the required packages:
 pip install tensorflow keras opencv-python mediapipe numpy pillow
Usage
 Run the 
behaviour.py
 script:
 python behaviour.py
 Use the GUI buttons to:
 Start webcam analysis
 Upload and analyze video files
 Upload and analyze single images
 Generate ASD behavioral reports based on captured analysis
 Behavioral Analysis
 The system tracks and evaluates:
 Focused Attention: Measures gaze and head stability to assess attention levels
 Eye Contact: Analyzes gaze direction and probability for eye contact assessment
 Repetitive Behavior: Detects head movement patterns indicating repetitive behaviors or
 stimming
 Output
 Real-time overlay of detected emotions and behavioral states on video feed
 Session metrics displayed in the GUI
 Detailed behavioral analysis reports saved as text files
 Disclaimer
 This system is a screening tool designed to identify potential ASD-related behavioral patterns. It
 is not a diagnostic instrument and should not replace clinical evaluation by healthcare
 professionals.
 For clinical diagnosis, consult qualified specialists
