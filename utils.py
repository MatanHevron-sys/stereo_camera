# utils.py
'''
 This optional module can house shared utility functions. For this refactored code,
it's not strictly necessary, but it's included for potential future expansions.
'''

import cv2

def annotate_image(image, text, position, color=(0, 255, 0), font_scale=0.5, thickness=2):
    """
    Annotates the image with the given text at the specified position.

    Parameters:
        image (numpy.ndarray): The image to annotate.
        text (str): The text to display.
        position (tuple): (x, y) coordinates for the text position.
        color (tuple): BGR color for the text.
        font_scale (float): Font scale factor.
        thickness (int): Thickness of the text.
    
    Returns:
        Annotated image.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return image
