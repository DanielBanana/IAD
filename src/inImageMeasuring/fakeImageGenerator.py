import cv2
import numpy as np

def generate_synthetic_image(image_size, objects, noise_intensity=5):
    """
    Generate a synthetic grayscale image with rectangles, circles, and mild random noise.

    Args:
        image_size (tuple): (width, height) of the image in pixels.
        objects (list): List of objects to draw. Each object is a dictionary with:
            - 'type': 'rectangle' or 'circle'
            - For rectangles: 'bottom_left' (x, y), 'top_right' (x, y), 'color' (grayscale intensity, 0-255)
            - For circles: 'origin' (x, y), 'radius' (int), 'color' (grayscale intensity, 0-255)
        noise_intensity (int): Intensity of the random noise (default: 5).

    Returns:
        numpy.ndarray: The generated grayscale image as a NumPy array.
    """
    # Create a blank white grayscale image
    image = np.ones((image_size[1], image_size[0]), dtype=np.uint8) * 255

    for obj in objects:
        if obj['type'] == 'rectangle':
            # Draw rectangle
            bottom_left = obj['bottom_left']
            top_right = obj['top_right']
            color = obj['color']
            cv2.rectangle(image, bottom_left, top_right, color, -1)
        elif obj['type'] == 'circle':
            # Draw circle
            origin = obj['origin']
            radius = obj['radius']
            color = obj['color']
            cv2.circle(image, origin, radius, color, -1)

    # Add mild random noise
    noise = np.random.normal(0, noise_intensity, image.shape).astype(np.int16)
    noisy_image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return noisy_image


# Example usage
if __name__ == "__main__":
    # Define image size (width, height)
    image_size = (2000,1000)

    # Define objects to include in the image (grayscale intensity: 0-255)
    objects = [
        {
            'type': 'rectangle',
            'bottom_left': (500, 200),
            'top_right': (800, 250),
            'color': 100  # Grayscale intensity
        },
    ]

    # Generate the image with mild noise
    image = generate_synthetic_image(image_size, objects, noise_intensity=100)
    # Save the image
    cv2.imwrite('images/object.png', image)

    # Display the image
    # cv2.imshow('Synthetic Grayscale Noisy Image', image)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()