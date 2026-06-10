import cv2
import threading
import queue
import time
import os
from functools import partial

class CameraProcessor:
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=1)
        self.running = False
        self.cap = None
        self.processing_thread = None
        self.on_capture_callbacks = []  # List of callback functions
        self.crop_x = 0                # Default crop region
        self.crop_y = 0
        self.crop_width = 640          # Default crop width
        self.crop_height = 480         # Default crop height
        self.image_counter = 0         # Counter for image filenames
        self.save_directory = "captures"  # Default save directory

    def register_capture_callback(self, callback):
        """Register a callback function to be called when a frame is captured."""
        self.on_capture_callbacks.append(callback)

    def set_crop_region(self, width, height, x=None, y=None):
        """
        Set the crop region for all captured frames.

        Args:
            width: The width of the crop region.
            height: The height of the crop region.
            x: The x-coordinate of the top-left corner of the crop region (optional).
            y: The y-coordinate of the top-left corner of the crop region (optional).

        If x and y are not provided, the crop region is centered.
        """
        self.crop_width = width
        self.crop_height = height

        # If x and y are not provided, center the crop region
        if x is None or y is None:
            # Get a test frame to determine the camera resolution
            if not self.running:
                cap = cv2.VideoCapture(0)
                ret, test_frame = cap.read()
                cap.release()
                if ret:
                    frame_height, frame_width = test_frame.shape[:2]
                    self.crop_x = (frame_width - width) // 2
                    self.crop_y = (frame_height - height) // 2
                else:
                    self.crop_x, self.crop_y = 0, 0
            else:
                # If the camera is already running, use the last frame's dimensions
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    frame_height, frame_width = frame.shape[:2]
                    self.crop_x = (frame_width - width) // 2
                    self.crop_y = (frame_height - height) // 2
                    self.frame_queue.put(frame)  # Put the frame back in the queue
                else:
                    self.crop_x, self.crop_y = 0, 0
        else:
            self.crop_x = x
            self.crop_y = y

    def get_crop_region(self):
        """Return the current crop region as a tuple (x, y, width, height)."""
        return (self.crop_x, self.crop_y, self.crop_width, self.crop_height)

    def _crop_frame(self, frame):
        """Internal method to crop a frame to the specified region."""
        return frame[
            self.crop_y : self.crop_y + self.crop_height,
            self.crop_x : self.crop_x + self.crop_width
        ]

    def _background_processing(self):
        """Background thread for frame capture."""
        self.cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.frame_queue.full():
                # Crop the frame before putting it in the queue
                cropped_frame = self._crop_frame(frame)
                self.frame_queue.put(cropped_frame)
        self.cap.release()

    def start(self):
        """Start the camera and background processing thread."""
        self.running = True
        self.processing_thread = threading.Thread(
            target=self._background_processing,
            daemon=True
        )
        self.processing_thread.start()

    def stop(self):
        """Stop the camera and background processing thread."""
        self.running = False
        if self.processing_thread is not None:
            self.processing_thread.join()
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

    def capture_frame(self, frame):
        """
        Capture a frame and return the results from all callbacks.

        Returns:
            A list of results from all registered callbacks.
        """
        results = []
        for callback in self.on_capture_callbacks:
            result = callback(frame)
            results.append(result)
        return results


    def save_image(self, frame):
        """
        Save the captured frame to the specified directory with ascending, zero-padded filenames.
        Overwrites existing files if they exist.

        Args:
            frame: The frame to save.

        Returns:
            The path where the image was saved.
        """
        # Create the directory if it doesn't exist
        os.makedirs(self.save_directory, exist_ok=True)

        # Create the filename with zero-padding
        filename = f"{self.image_counter:04d}.png"
        save_path = os.path.join(self.save_directory, filename)

        # Save the image
        cv2.imwrite(save_path, frame)
        print(f"Image saved to: {save_path}")

        # Increment the counter for the next image
        self.image_counter += 1

        return save_path

    def set_save_directory(self, directory):
        """Set the directory where captured images will be saved."""
        self.save_directory = directory

    def reset_image_counter(self):
        """Reset the image counter to 0."""
        self.image_counter = 0

    def display_frames(self):
        """Main thread: Display frames and handle 'c' key press."""
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                cv2.imshow("Live Camera Feed", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    # Call all registered callbacks with the captured frame
                    results = self.capture_frame(frame)
                    print("Callback results:", results)
                elif key == ord('q'):
                    self.stop()
                    break

def main():
    # Example callback functions that return values
    def show_grayscale(frame):
        """Show the captured frame in grayscale and return the grayscale frame."""
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Captured (Grayscale)", processed_frame)
        return processed_frame

    def calculate_brightness(frame):
        """Calculate and return the average brightness of the frame."""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = gray_frame.mean()
        return brightness

    # Create the processor
    processor = CameraProcessor()

    # Set the crop region (width, height) - will be centered automatically
    processor.set_crop_region(400, 300)

    # Set the save directory
    processor.set_save_directory("captures")

    # Register the save_image method as a callback
    processor.register_capture_callback(processor.save_image)

    # Register other callbacks
    processor.register_capture_callback(show_grayscale)
    processor.register_capture_callback(calculate_brightness)

    # Start the processor
    processor.start()
    processor.display_frames()

if __name__ == "__main__":
    main()
