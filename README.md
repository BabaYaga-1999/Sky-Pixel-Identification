Documentation and Reflection

Chosen Techniques and Rationale
HSV Color Space: I utilized the HSV (Hue, Saturation, Value) color space for sky detection because it provides a more intuitive way to specify color ranges compared to RGB. This approach is effective in differentiating sky colors from other elements in the image.

Gradient Magnitude Calculation: To identify the smooth regions characteristic of the sky, I calculated the gradient magnitude. The sky, typically being a smooth region, would have lower gradient values.

Morphological Operations: To refine the mask and remove small noise, morphological operations like closing and opening were applied.

Contours Detection: I employed contour detection to identify continuous sky regions, further refining the detection process by considering the size and position of these regions.

Combined Color and Gradient Masks: This method allowed us to leverage both color and texture information, improving the robustness of the detection, especially in challenging lighting conditions like sunsets.

Implementation Process and Code Snippets

HSV Thresholding:
lower_blue = np.array([10, 50, 50])
upper_blue = np.array([170, 255, 255])
mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

Gradient Magnitude and Thresholding:
grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
gradient_magnitude = cv2.magnitude(grad_x, grad_y)
_, gradient_mask = cv2.threshold(gradient_magnitude, 10, 255, cv2.THRESH_BINARY_INV)

Contours Detection:
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Size threshold
            x, y, w, h = cv2.boundingRect(contour)
            if y < image.shape[0] / 2:  # Position threshold (upper half of the image)
                cv2.drawContours(sky_mask, [contour], -1, 255, thickness=cv2.FILLED)

Challenges and Solutions

Diverse Sky Conditions: Different lighting conditions, like sunsets, posed a challenge. I addressed this by including multiple color ranges (e.g., for blue sky and sunset).
False Positives: Elements like mountains were initially misidentified as the sky. I mitigated this by using gradient information and refining HSV thresholds.

Reflection

Effectiveness of the Approach

Our method shows excellent performance under clear weather conditions when the sky is a light blue, accurately identifying the sky's edges with minimal false detections.
However, the approach struggles in cloudy or foggy conditions and during sunrises or sunsets when the sky turns yellowish. These conditions bring the sky's color closer to other elements like mountains, leading to difficulties in sky detection and potential misidentification of non-sky areas as sky.

Limitations and Potential Improvements

Limitations: The current method relies on color and gradient thresholds, which may not be accurate enough under complex weather conditions.
Potential Improvements: Further optimization of the HSV color threshold settings, especially under varying lighting and weather conditions, is needed. Additionally, considering machine learning methods, such as training a classifier to recognize the sky, could enhance accuracy and robustness.

Learning Outcomes

This project provided a practical application of image processing, deepening our understanding of concepts like color spaces, gradient computation, morphological operations, and contour detection.
I learned how to adjust and test different image processing techniques based on practical scenarios and how to combine these techniques to address specific problems.
Through hands-on experience, I realized the limitations of algorithms under different conditions and the necessity to continue learning and exploring, especially in the field of machine learning.
