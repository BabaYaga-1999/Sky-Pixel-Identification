import gradio as gr
import cv2
import numpy as np
from PIL import Image

def detect_sky_pil(pil_image):
    try:
        print("Converting PIL image to OpenCV format...")
        # Convert PIL Image to OpenCV format
        image = np.array(pil_image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Convert to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Threshold for blue sky and sunset colors
        lower_blue = np.array([10, 50, 50])  # Adjusted for yellow/orange
        upper_blue = np.array([170, 255, 255])  # Adjusted for deep blue
        lower_orange = np.array([0, 50, 50])  # For sunset
        upper_orange = np.array([30, 255, 255])
        lower_white = np.array([0, 0, 210])  # For whitish sky
        upper_white = np.array([180, 30, 255])

        blue_sky_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
        sunset_mask = cv2.inRange(hsv_image, lower_orange, upper_orange)
        whitish_sky_mask = cv2.inRange(hsv_image, lower_white, upper_white)

        # Combine masks
        mask = cv2.bitwise_or(blue_sky_mask, sunset_mask)
        mask = cv2.bitwise_or(mask, whitish_sky_mask)

        # Remove green areas like mountains
        lower_green = np.array([30, 40, 40])
        upper_green = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(green_mask))

        # Calculate gradient magnitude
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = cv2.magnitude(grad_x, grad_y)

        # # Save the edge detection result
        # edge_output_path = os.path.join(output_folder, "edges_" + os.path.basename(image_path))
        # cv2.imwrite(edge_output_path, gradient_magnitude)
        # print(f"Edge detection result saved to {edge_output_path}")

        # Gradient threshold
        grad_threshold = 10
        _, gradient_mask = cv2.threshold(gradient_magnitude, grad_threshold, 255, cv2.THRESH_BINARY_INV)

        # Combine color and gradient masks
        combined_mask = cv2.bitwise_and(mask, mask, mask=gradient_mask.astype(np.uint8))

        # Morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        # Find continuous regions (contours)
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sky_mask = np.zeros_like(combined_mask)

        # Filter the regions based on size and position
        for contour in contours:
            if cv2.contourArea(contour) > 1000:  # Size threshold
                x, y, w, h = cv2.boundingRect(contour)
                if y < image.shape[0] / 2:  # Position threshold (upper half of the image)
                    cv2.drawContours(sky_mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Apply the mask to the original image
        sky = cv2.bitwise_and(image, image, mask=sky_mask)

        # Optional: Create a red color layer for sky region
        red_layer = np.zeros_like(image)
        red_layer[:, :] = [255, 255, 255]  # BGR for red
        sky_red = cv2.bitwise_and(red_layer, red_layer, mask=sky_mask)
        result = cv2.addWeighted(image, 1, sky_red, 1, 0)

        print("Converting processed image back to RGB format for display...")
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return result
    except Exception as e:
        print(f"Error in detect_sky_pil: {e}")
        return None

def gradio_sky_detection_demo(pil_image):
    print("Running sky detection demo...")
    try:
        result = detect_sky_pil(pil_image)
        if result is None:
            print("No result from detect_sky_pil, returning original image")
            return pil_image
        print("Returning processed image...")
        return result
    except Exception as e:
        print(f"An error occurred in gradio_sky_detection_demo: {e}")
        return pil_image

iface = gr.Interface(
    fn=gradio_sky_detection_demo,
    inputs="image",
    outputs="image",
    title="Sky Detection Demo",
    description="Upload an image to see the sky detection result."
)

iface.launch(share=False)

