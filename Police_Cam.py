import flet as ft
import cv2
import numpy as np
import threading
import time
import os
import json
from datetime import datetime
import base64
import io
from PIL import Image
import face_recognition
import pygame
import winsound  # For Windows alert sounds
import platform

class PoliceAISystem:
    def __init__(self):
        self.known_faces = []
        self.known_names = []
        self.known_details = []
        self.surveillance_active = False
        self.camera_cap = None
        self.current_frame = None
        self.detection_thread = None
        self.alert_active = False
        self.confidence_threshold = 0.6
        self.alert_log = []
        
        # Initialize pygame for audio alerts
        try:
            pygame.mixer.init()
        except:
            print("Audio system not available")
        
        # Create directories
        self.create_directories()
    
    def create_directories(self):
        """Create necessary directories"""
        directories = ["suspects", "alerts", "logs"]
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
    
    def add_suspect(self, image_path, name, details):
        """Add a suspect to the database"""
        try:
            # Load and encode the face
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            
            if len(face_encodings) > 0:
                self.known_faces.append(face_encodings[0])
                self.known_names.append(name)
                self.known_details.append(details)
                return True
            else:
                return False
        except Exception as e:
            print(f"Error adding suspect: {e}")
            return False
    
    def add_suspect_from_array(self, image_array, name, details):
        """Add suspect from numpy array (uploaded image)"""
        try:
            face_encodings = face_recognition.face_encodings(image_array)
            
            if len(face_encodings) > 0:
                self.known_faces.append(face_encodings[0])
                self.known_names.append(name)
                self.known_details.append(details)
                return True
            else:
                return False
        except Exception as e:
            print(f"Error adding suspect: {e}")
            return False
    
    def start_surveillance(self, camera_index=0):
        """Start live surveillance"""
        if self.surveillance_active:
            return False
        
        try:
            self.camera_cap = cv2.VideoCapture(camera_index)
            if not self.camera_cap.isOpened():
                return False
            
            self.surveillance_active = True
            self.detection_thread = threading.Thread(target=self._surveillance_loop)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            return True
        except Exception as e:
            print(f"Error starting surveillance: {e}")
            return False
    
    def stop_surveillance(self):
        """Stop surveillance"""
        self.surveillance_active = False
        
        if self.detection_thread:
            self.detection_thread.join()
        
        if self.camera_cap:
            self.camera_cap.release()
            self.camera_cap = None
    
    def _surveillance_loop(self):
        """Main surveillance loop with face detection"""
        while self.surveillance_active and self.camera_cap:
            ret, frame = self.camera_cap.read()
            if not ret:
                continue
            
            self.current_frame = frame.copy()
            
            # Detect faces in the frame
            if len(self.known_faces) > 0:
                self._detect_faces(frame)
            
            time.sleep(0.1)  # Reduce CPU usage
    
    def _detect_faces(self, frame):
        """Detect and match faces in the frame"""
        try:
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]
            
            # Find faces
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Compare with known faces
                matches = face_recognition.compare_faces(self.known_faces, face_encoding)
                face_distances = face_recognition.face_distance(self.known_faces, face_encoding)
                
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index] and face_distances[best_match_index] < self.confidence_threshold:
                    # Suspect detected!
                    suspect_name = self.known_names[best_match_index]
                    suspect_details = self.known_details[best_match_index]
                    confidence = 1 - face_distances[best_match_index]
                    
                    self._trigger_alert(suspect_name, suspect_details, confidence, frame)
        
        except Exception as e:
            print(f"Detection error: {e}")
    
    def _trigger_alert(self, name, details, confidence, frame):
        """Trigger alert when suspect is detected"""
        if self.alert_active:
            return  # Prevent multiple simultaneous alerts
        
        self.alert_active = True
        
        # Log the alert
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'suspect_name': name,
            'suspect_details': details,
            'confidence': float(confidence),
            'location': 'Camera Feed'
        }
        
        self.alert_log.append(alert_data)
        
        # Save alert frame
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        alert_image_path = f"alerts/alert_{name}_{timestamp}.jpg"
        cv2.imwrite(alert_image_path, frame)
        
        # Play alert sound
        self._play_alert_sound()
        
        # Reset alert after 5 seconds
        threading.Timer(5.0, self._reset_alert).start()
    
    def _play_alert_sound(self):
        """Play alert sound"""
        try:
            if platform.system() == "Windows":
                # Windows system sound
                winsound.Beep(1000, 1000)  # 1000Hz for 1 second
            else:
                # Cross-platform beep (requires pygame)
                frequency = 1000
                duration = 1000
                sample_rate = 22050
                frames = int(duration * sample_rate / 1000)
                
                arr = np.zeros(frames)
                for i in range(frames):
                    arr[i] = np.sin(2 * np.pi * frequency * i / sample_rate)
                
                arr = (arr * 32767).astype(np.int16)
                sound = pygame.sndarray.make_sound(arr)
                sound.play()
                time.sleep(1)
        except Exception as e:
            print(f"Audio alert error: {e}")
    
    def _reset_alert(self):
        """Reset alert status"""
        self.alert_active = False
    
    def get_current_frame_base64(self):
        """Get current frame as base64 for display"""
        if self.current_frame is not None:
            try:
                # Convert frame to RGB
                rgb_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                
                # Convert to base64
                buffer = io.BytesIO()
                pil_image.save(buffer, format='JPEG')
                img_bytes = buffer.getvalue()
                return base64.b64encode(img_bytes).decode()
            except Exception as e:
                print(f"Frame conversion error: {e}")
        return None
    
    def save_logs(self):
        """Save alert logs to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/alert_log_{timestamp}.json"
        
        with open(log_file, 'w') as f:
            json.dump(self.alert_log, f, indent=2)
        
        return log_file

def main(page: ft.Page):
    page.title = "AI Police Surveillance System"
    page.theme_mode = ft.ThemeMode.DARK
    page.window.width = 1200
    page.window.height = 660
    page.padding = 0
    page.window.center()
    
    # Initialize AI system
    ai_system = PoliceAISystem()
    
    # Status indicators
    surveillance_status = ft.Text("Surveillance: OFF", color=ft.colors.RED, size=16, weight=ft.FontWeight.BOLD)
    alert_status = ft.Text("No Alerts", color=ft.colors.GREEN, size=14)
    suspects_count = ft.Text("Suspects in Database: 0", size=14)
    
    # Live feed display
    live_feed = ft.Image(
        width=640,
        height=480,
        fit=ft.ImageFit.CONTAIN,
        border_radius=10
    )
    
    # Suspect details form
    suspect_name_field = ft.TextField(label="Suspect Name", width=300)
    suspect_details_field = ft.TextField(
        label="Details (Crime, Description, etc.)",
        width=300,
        multiline=True,
        max_lines=3
    )
    
    # File picker for suspect images
    file_picker = ft.FilePicker()
    page.overlay.append(file_picker)
    
    selected_image_path = ft.Text("No image selected", size=12, color=ft.colors.GREY)
    
    # Alert log list
    alert_log_list = ft.Column(height=300, scroll=ft.ScrollMode.AUTO)
    
    def update_live_feed():
        """Update live feed display"""
        while True:
            if ai_system.surveillance_active:
                frame_base64 = ai_system.get_current_frame_base64()
                if frame_base64:
                    live_feed.src_base64 = frame_base64
                    
                    # Update alert status
                    if ai_system.alert_active:
                        alert_status.value = "🚨 SUSPECT DETECTED! 🚨"
                        alert_status.color = ft.colors.RED
                    else:
                        alert_status.value = "Monitoring..."
                        alert_status.color = ft.colors.GREEN
                    
                    page.update()
            
            time.sleep(0.1)
    
    def start_surveillance(e):
        if ai_system.start_surveillance():
            surveillance_status.value = "Surveillance: ON"
            surveillance_status.color = ft.colors.GREEN
            start_btn.disabled = True
            stop_btn.disabled = False
            
            # Start live feed update thread
            feed_thread = threading.Thread(target=update_live_feed)
            feed_thread.daemon = True
            feed_thread.start()
        else:
            show_error("Failed to start surveillance. Check camera connection.")
        page.update()
    
    def stop_surveillance(e):
        ai_system.stop_surveillance()
        surveillance_status.value = "Surveillance: OFF"
        surveillance_status.color = ft.colors.RED
        start_btn.disabled = False
        stop_btn.disabled = True
        alert_status.value = "No Alerts"
        alert_status.color = ft.colors.GREEN
        page.update()
    
    def pick_image(e):
        file_picker.pick_files(
            dialog_title="Select Suspect Image",
            file_type=ft.FilePickerFileType.IMAGE
        )
    
    def on_file_picked(e: ft.FilePickerResultEvent):
        if e.files:
            file_path = e.files[0].path
            selected_image_path.value = f"Selected: {os.path.basename(file_path)}"
            selected_image_path.data = file_path  # Store path for later use
            page.update()
    
    file_picker.on_result = on_file_picked
    
    def add_suspect(e):
        name = suspect_name_field.value
        details = suspect_details_field.value
        image_path = getattr(selected_image_path, 'data', None)
        
        if not name or not image_path:
            show_error("Please provide suspect name and image.")
            return
        
        if ai_system.add_suspect(image_path, name, details):
            show_success(f"Suspect '{name}' added successfully!")
            update_suspects_count()
            # Clear form
            suspect_name_field.value = ""
            suspect_details_field.value = ""
            selected_image_path.value = "No image selected"
            selected_image_path.data = None
        else:
            show_error("Failed to add suspect. Please ensure the image contains a clear face.")
        
        page.update()
    
    def update_suspects_count():
        suspects_count.value = f"Suspects in Database: {len(ai_system.known_names)}"
        page.update()
    
    def update_alert_log():
        """Update alert log display"""
        alert_log_list.controls.clear()
        
        for alert in reversed(ai_system.alert_log[-10:]):  # Show last 10 alerts
            timestamp = datetime.fromisoformat(alert['timestamp']).strftime("%H:%M:%S")
            confidence_percent = int(alert['confidence'] * 100)
            
            alert_tile = ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Row([
                            ft.Text(f"🚨 {alert['suspect_name']}", 
                                   weight=ft.FontWeight.BOLD, 
                                   color=ft.colors.RED),
                            ft.Text(f"{timestamp}", size=12, color=ft.colors.GREY)
                        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                        ft.Text(f"Confidence: {confidence_percent}%", size=12),
                        ft.Text(f"Details: {alert['suspect_details'][:50]}...", 
                               size=11, color=ft.colors.GREY)
                    ]),
                    padding=10
                )
            )
            
            alert_log_list.controls.append(alert_tile)
        
        page.update()
    
    def save_logs(e):
        try:
            log_file = ai_system.save_logs()
            show_success(f"Logs saved to {log_file}")
        except Exception as ex:
            show_error(f"Failed to save logs: {str(ex)}")
    
    def show_success(message):
        snack_bar = ft.SnackBar(ft.Text(message), bgcolor=ft.colors.GREEN)
        page.overlay.append(snack_bar)
        snack_bar.open = True
        page.update()
    
    def show_error(message):
        snack_bar = ft.SnackBar(ft.Text(message), bgcolor=ft.colors.RED)
        page.overlay.append(snack_bar)
        snack_bar.open = True
        page.update()
    
    # Periodic alert log update
    def periodic_update():
        while True:
            if len(ai_system.alert_log) > len(alert_log_list.controls):
                update_alert_log()
            time.sleep(2)
    
    update_thread = threading.Thread(target=periodic_update)
    update_thread.daemon = True
    update_thread.start()
    
    # Control buttons
    start_btn = ft.ElevatedButton(
        "Start Surveillance",
        icon=ft.icons.PLAY_ARROW,
        on_click=start_surveillance,
        color=ft.colors.GREEN,
        width=200
    )
    
    stop_btn = ft.ElevatedButton(
        "Stop Surveillance",
        icon=ft.icons.STOP,
        on_click=stop_surveillance,
        color=ft.colors.RED,
        width=200,
        disabled=True
    )
    
    # Create main layout
    page.add(
        ft.Row([
            # Left Panel - Suspect Management
            ft.Container(
                width=350,
                height=900,
                bgcolor=ft.colors.SURFACE_VARIANT,
                padding=20,
                content=ft.Column([
                    ft.Text("Suspect Database", size=20, weight=ft.FontWeight.BOLD),
                    ft.Divider(),
                    
                    suspects_count,
                    
                    ft.Text("Add New Suspect", size=16, weight=ft.FontWeight.BOLD),
                    suspect_name_field,
                    suspect_details_field,
                    
                    ft.ElevatedButton(
                        "Select Image",
                        icon=ft.icons.IMAGE,
                        on_click=pick_image,
                        width=300
                    ),
                    selected_image_path,
                    
                    ft.ElevatedButton(
                        "Add Suspect",
                        icon=ft.icons.PERSON_ADD,
                        on_click=add_suspect,
                        color=ft.colors.BLUE,
                        width=300
                    ),
                    
                    ft.Divider(),
                    
                    ft.Text("System Settings", size=16, weight=ft.FontWeight.BOLD),
                    ft.Text(f"Confidence Threshold: {ai_system.confidence_threshold}", size=12),
                    ft.Slider(
                        min=0.3,
                        max=0.9,
                        value=0.6,
                        divisions=6,
                        label="Confidence",
                        on_change=lambda e: setattr(ai_system, 'confidence_threshold', e.control.value)
                    ),
                    
                ], scroll=ft.ScrollMode.AUTO)
            ),
            
            # Center Panel - Live Feed
            ft.Container(
                width=700,
                padding=20,
                content=ft.Column([
                    ft.Text("Live Surveillance Feed", size=20, weight=ft.FontWeight.BOLD),
                    
                    # Status indicators
                    ft.Row([
                        surveillance_status,
                        alert_status
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    
                    # Live feed
                    ft.Container(
                        content=live_feed,
                        bgcolor=ft.colors.BLACK,
                        border_radius=10,
                        padding=5
                    ),
                    
                    # Control buttons
                    ft.Row([
                        start_btn,
                        stop_btn
                    ], alignment=ft.MainAxisAlignment.SPACE_EVENLY),
                    
                ])
            ),
            
            # Right Panel - Alert Log
            ft.Container(
                width=350,
                height=900,
                bgcolor=ft.colors.SURFACE_VARIANT,
                padding=20,
                content=ft.Column([
                    ft.Text("Alert Log", size=20, weight=ft.FontWeight.BOLD),
                    ft.Divider(),
                    
                    alert_log_list,
                    
                    ft.ElevatedButton(
                        "Save Logs",
                        icon=ft.icons.SAVE,
                        on_click=save_logs,
                        width=300
                    ),
                    
                    ft.Text("Recent Alerts", size=14, color=ft.colors.GREY),
                    ft.Text("• Alerts auto-save screenshots", size=12),
                    ft.Text("• Audio alerts play automatically", size=12),
                    ft.Text("• Logs saved in JSON format", size=12),
                    
                ], scroll=ft.ScrollMode.AUTO)
            ),
        ])
    )
    
    # Cleanup on close
    def on_window_event(e):
        if e.data == "close":
            ai_system.stop_surveillance()
            page.window.destroy()
    
    page.window.prevent_close = True
    page.on_window_event = on_window_event

if __name__ == "__main__":
    # Install requirements reminder
    print("Required packages: flet, opencv-python, face-recognition, pillow, numpy, pygame")
    print("Install with: pip install flet opencv-python face-recognition pillow numpy pygame")
    
    ft.app(target=main)