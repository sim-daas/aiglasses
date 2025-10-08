import os
import cv2
import json
import base64
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
import google.generativeai as genai
from dotenv import load_dotenv
import threading
import queue

class VisionAPI:
    def __init__(self, env_file_path=".env"):
        """
        Initialize VisionAPI with Gemini configuration
        
        Args:
            env_file_path (str): Path to .env file containing GEMINI_API_KEY
        """
        # Load environment variables
        load_dotenv(env_file_path)
        
        # Configure Gemini API
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Initialize webcam
        self.cap = None
        self.current_frame = None
        
    def initialize_camera(self, camera_index=0):
        """Initialize webcam capture"""
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")
        return True
    
    def capture_frame(self):
        """Capture current frame from webcam"""
        if self.cap is None:
            raise RuntimeError("Camera not initialized")
        
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            return frame
        return None
    
    def frame_to_base64(self, frame):
        """Convert OpenCV frame to base64 string"""
        _, buffer = cv2.imencode('.jpg', frame)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64
    
    def analyze_image(self, frame, user_query):
        """
        Send image and query to Gemini API and return structured JSON response
        
        Args:
            frame: OpenCV frame/image
            user_query (str): User's text query about the image
            
        Returns:
            dict: JSON with 'answer' and 'label' keys
        """
        try:
            # Convert frame to PIL Image for Gemini
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Craft prompt for structured output
            prompt = f"""
            Analyze this image and answer the user's query. You must respond in valid JSON format with exactly these two keys:
            
            User Query: {user_query}
            
            Required JSON format:
            {{
                "answer": "Your detailed response to the user's query about what you see in the image",
                "label": "Primary object name (single word or short phrase)"
            }}
            
            Rules:
            - The "answer" should directly address the user's query about the image
            - The "label" should contain only the main object/thing visible in the image (e.g., "person", "car", "dog", "book")
            - Always output valid JSON format
            - Do not include any text outside the JSON structure
            """
            
            response = self.model.generate_content([prompt, pil_image])
            
            # Parse JSON response
            response_text = response.text.strip()
            
            # Clean response if it has markdown formatting
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            result = json.loads(response_text)
            
            # Validate required keys
            if 'answer' not in result or 'label' not in result:
                raise ValueError("Invalid response format from API")
            
            return result
            
        except json.JSONDecodeError as e:
            return {
                "answer": f"Error parsing API response: {str(e)}",
                "label": "error"
            }
        except Exception as e:
            return {
                "answer": f"Error analyzing image: {str(e)}",
                "label": "error"
            }
    
    def cleanup(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

class VisionGUI:
    def __init__(self):
        """Initialize GUI interface for VisionAPI"""
        self.vision_api = VisionAPI()
        self.setup_gui()
        self.camera_running = False
        self.frame_queue = queue.Queue()
        
    def setup_gui(self):
        """Setup tkinter GUI"""
        self.root = tk.Tk()
        self.root.title("AI Vision API")
        self.root.geometry("800x600")
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Camera preview
        self.camera_label = ttk.Label(main_frame, text="Camera Preview")
        self.camera_label.grid(row=0, column=0, columnspan=2, pady=5)
        
        self.camera_frame = ttk.Label(main_frame, text="No camera feed", 
                                     background="black", foreground="white")
        self.camera_frame.grid(row=1, column=0, columnspan=2, pady=5)
        
        # Query input
        ttk.Label(main_frame, text="Enter your query:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.query_entry = ttk.Entry(main_frame, width=50)
        self.query_entry.grid(row=2, column=1, pady=5, padx=5)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        self.start_camera_btn = ttk.Button(button_frame, text="Start Camera", 
                                          command=self.start_camera)
        self.start_camera_btn.grid(row=0, column=0, padx=5)
        
        self.capture_btn = ttk.Button(button_frame, text="Capture & Analyze", 
                                     command=self.capture_and_analyze, state="disabled")
        self.capture_btn.grid(row=0, column=1, padx=5)
        
        self.stop_camera_btn = ttk.Button(button_frame, text="Stop Camera", 
                                         command=self.stop_camera, state="disabled")
        self.stop_camera_btn.grid(row=0, column=2, padx=5)
        
        # Results display
        ttk.Label(main_frame, text="Results:").grid(row=4, column=0, sticky=tk.W, pady=(20,5))
        
        self.results_text = scrolledtext.ScrolledText(main_frame, height=10, width=70)
        self.results_text.grid(row=5, column=0, columnspan=2, pady=5)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
    def start_camera(self):
        """Start camera preview"""
        try:
            self.vision_api.initialize_camera()
            self.camera_running = True
            self.start_camera_btn.config(state="disabled")
            self.capture_btn.config(state="normal")
            self.stop_camera_btn.config(state="normal")
            
            # Start camera thread
            self.camera_thread = threading.Thread(target=self.camera_loop)
            self.camera_thread.daemon = True
            self.camera_thread.start()
            
            # Start GUI update
            self.update_camera_display()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {str(e)}")
    
    def camera_loop(self):
        """Camera capture loop running in separate thread"""
        while self.camera_running:
            frame = self.vision_api.capture_frame()
            if frame is not None:
                self.frame_queue.put(frame)
    
    def update_camera_display(self):
        """Update camera display in GUI"""
        if self.camera_running:
            try:
                frame = self.frame_queue.get_nowait()
                
                # Resize frame for display
                display_frame = cv2.resize(frame, (400, 300))
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PhotoImage
                pil_image = Image.fromarray(rgb_frame)
                photo = ImageTk.PhotoImage(pil_image)
                
                # Update display
                self.camera_frame.configure(image=photo, text="")
                self.camera_frame.image = photo  # Keep a reference
                
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Display update error: {e}")
            
            # Schedule next update
            self.root.after(50, self.update_camera_display)
    
    def capture_and_analyze(self):
        """Capture image and analyze with user query"""
        query = self.query_entry.get().strip()
        if not query:
            messagebox.showwarning("Warning", "Please enter a query")
            return
        
        try:
            # Capture current frame
            frame = self.vision_api.capture_frame()
            if frame is None:
                messagebox.showerror("Error", "Failed to capture image")
                return
            
            # Show loading message
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Analyzing image... Please wait.\n")
            self.root.update()
            
            # Analyze image
            result = self.vision_api.analyze_image(frame, query)
            
            # Display results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Query: {query}\n\n")
            self.results_text.insert(tk.END, f"Answer: {result['answer']}\n\n")
            self.results_text.insert(tk.END, f"Object Label: {result['label']}\n\n")
            self.results_text.insert(tk.END, f"Full JSON Response:\n{json.dumps(result, indent=2)}\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
    
    def stop_camera(self):
        """Stop camera preview"""
        self.camera_running = False
        self.vision_api.cleanup()
        
        self.start_camera_btn.config(state="normal")
        self.capture_btn.config(state="disabled")
        self.stop_camera_btn.config(state="disabled")
        
        self.camera_frame.configure(image="", text="Camera stopped")
        self.camera_frame.image = None
    
    def run(self):
        """Run the GUI application"""
        try:
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.root.mainloop()
        except KeyboardInterrupt:
            self.on_closing()
    
    def on_closing(self):
        """Handle application closing"""
        self.camera_running = False
        self.vision_api.cleanup()
        self.root.destroy()

# Example usage and CLI interface
def main():
    """Main function to run the application"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Vision API with Gemini")
    parser.add_argument("--mode", choices=["gui", "cli"], default="gui",
                       help="Run in GUI or CLI mode")
    
    args = parser.parse_args()
    
    if args.mode == "gui":
        # Run GUI version
        app = VisionGUI()
        app.run()
    else:
        # Run CLI version
        try:
            vision = VisionAPI()
            vision.initialize_camera()
            
            print("Vision API CLI Mode")
            print("Press 'c' to capture image, 'q' to quit")
            
            while True:
                frame = vision.capture_frame()
                if frame is not None:
                    cv2.imshow("Camera Feed", frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('c'):
                    query = input("\nEnter your query about the image: ")
                    if query.strip():
                        print("Analyzing image...")
                        result = vision.analyze_image(frame, query)
                        print(f"\nResults:")
                        print(f"Answer: {result['answer']}")
                        print(f"Label: {result['label']}")
                        print(f"JSON: {json.dumps(result, indent=2)}\n")
                
                elif key == ord('q'):
                    break
            
            vision.cleanup()
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()