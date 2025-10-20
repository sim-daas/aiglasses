import pyaudio
import wave
import tempfile
import threading
import numpy as np
from config import Config

class AudioManager:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.recording = False
        self.frames = []
        self.stream = None
        
    def list_devices(self):
        """List all audio devices"""
        print("🎤 Available audio devices:")
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"  [{i}] {info['name']} (Input)")
    
    def start_recording(self, device_index=None):
        """Start recording audio"""
        try:
            self.frames = []
            self.recording = True
            
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=Config.AUDIO_CHANNELS,
                rate=Config.AUDIO_RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=Config.AUDIO_CHUNK
            )
            
            # Start recording thread
            self.record_thread = threading.Thread(target=self._record_loop, daemon=True)
            self.record_thread.start()
            
            print("🔴 Recording started...")
            return True
            
        except Exception as e:
            print(f"❌ Recording start error: {e}")
            return False
    
    def _record_loop(self):
        """Recording loop"""
        while self.recording:
            try:
                data = self.stream.read(Config.AUDIO_CHUNK, exception_on_overflow=False)
                self.frames.append(data)
            except Exception as e:
                print(f"⚠️  Recording error: {e}")
                break
    
    def stop_recording(self):
        """Stop recording and return audio file path"""
        try:
            self.recording = False
            
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            
            if not self.frames:
                print("⚠️  No audio recorded")
                return None
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            
            with wave.open(temp_file.name, 'wb') as wf:
                wf.setnchannels(Config.AUDIO_CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
                wf.setframerate(Config.AUDIO_RATE)
                wf.writeframes(b''.join(self.frames))
            
            print(f"✅ Audio saved: {temp_file.name}")
            return temp_file.name
            
        except Exception as e:
            print(f"❌ Recording stop error: {e}")
            return None
    
    def cleanup(self):
        """Cleanup audio resources"""
        if self.stream:
            self.stream.close()
        self.audio.terminate()
