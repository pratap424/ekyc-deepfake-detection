"""
Standalone CV2 Liveness Detection Window
Fast real-time performance - no Gradio overhead
"""
import os
import sys

# Add project root to path so imports work
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import time

def main():
    """Main function with proper error handling"""
    print("="*60)
    print("🚀 FAST LIVENESS DETECTION - CV2 Mode")
    print("="*60)
    
    try:
        # Import model (might fail if config is wrong)
        print("Loading liveness detector...")
        from models.liveness_model import LivenessDetector
        from config import LIVENESS_MODEL
        
        # Check if model file exists
        if not os.path.exists(LIVENESS_MODEL):
            print(f"❌ ERROR: Model file not found: {LIVENESS_MODEL}")
            print(f"   Current directory: {os.getcwd()}")
            print(f"   Looking for: {os.path.abspath(LIVENESS_MODEL)}")
            input("\nPress Enter to exit...")
            return 1
        
        detector = LivenessDetector(LIVENESS_MODEL)
        print("✅ Liveness detector loaded!")
        
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
        return 1
    
    # Try to open camera
    try:
        print("\nOpening camera...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ ERROR: Cannot open camera!")
            print("   Make sure no other application is using the camera")
            input("\nPress Enter to exit...")
            return 1
        
        # Configure camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print("✅ Camera opened successfully!")
        
    except Exception as e:
        print(f"❌ ERROR opening camera: {e}")
        input("\nPress Enter to exit...")
        return 1
    
    # Main loop
    print("\n" + "="*60)
    print("CONTROLS:")
    print("  Q or ESC - Quit")
    print("  S        - Save screenshot")
    print("="*60 + "\n")
    
    frame_count = 0
    start_time = time.time()
    last_fps_update = start_time
    fps = 0
    screenshot_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Failed to read frame")
                break
            
            frame_count += 1
            
            # Update FPS every 0.5 seconds
            current_time = time.time()
            if current_time - last_fps_update >= 0.5:
                elapsed = current_time - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                last_fps_update = current_time
            
            # Run liveness detection
            try:
                result = detector.predict(frame)
            except Exception as e:
                print(f"⚠️ Detection error: {e}")
                result = {'face_detected': False, 'is_live': False, 'score': 0.0, 'bbox': None}
            
            # Annotate frame
            if result['face_detected'] and result['bbox'] is not None:
                bbox = result['bbox']
                color = (0, 255, 0) if result['is_live'] else (0, 0, 255)
                
                # Draw bounding box
                cv2.rectangle(frame, tuple(bbox[0]), tuple(bbox[1]), color, 3)
                
                # Status text
                status = "LIVE ✓" if result['is_live'] else "SPOOF ✗"
                cv2.putText(frame, status, (bbox[0][0], bbox[0][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                # Score
                cv2.putText(frame, f"Score: {result['score']:.4f}",
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Confidence: {result.get('confidence', 0):.1f}%",
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                # No face detected
                cv2.putText(frame, "No face detected", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # FPS counter
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Instructions
            cv2.putText(frame, "Q/ESC=Quit | S=Screenshot", (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Show frame
            cv2.imshow("Fast Liveness Detection (CV2)", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                print("\n✅ Quitting...")
                break
            elif key == ord('s'):  # S for screenshot
                screenshot_count += 1
                filename = f"liveness_screenshot_{screenshot_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved: {filename}")
        
    except KeyboardInterrupt:
        print("\n✅ Interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
    
    # Print summary
    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed if elapsed > 0 else 0
    
    print("\n" + "="*60)
    print("📊 SESSION SUMMARY")
    print("="*60)
    print(f"Total frames processed: {frame_count}")
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Screenshots saved: {screenshot_count}")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)
