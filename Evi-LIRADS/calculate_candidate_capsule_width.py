import cv2
import numpy as np
import matplotlib.pyplot as plt


# Function to find the innermost and outermost intersections in one direction
def find_edge_intersections(centroid, contour, angle, img_shape, max_distance=5000):
    innermost_intersection = None
    outermost_intersection = None
    x, y = centroid
    step_size = 1

    for dist in range(0, max_distance, step_size):
        new_x = int(x + dist * np.cos(angle))
        new_y = int(y + dist * np.sin(angle))

        if new_x < 0 or new_y < 0 or new_x >= img_shape[1] or new_y >= img_shape[0]:
            break

        inside = cv2.pointPolygonTest(contour, (new_x, new_y), False) >= 0   # 检查该点是否在轮廓上或轮廓内部。如果是，返回值大于等于0

        if inside and innermost_intersection is None:
            innermost_intersection = (new_x, new_y)
        if not inside and innermost_intersection is not None:
            outermost_intersection = (new_x, new_y)
            break

    return innermost_intersection, outermost_intersection



# Function to calculate widths for intersections
def calculate_widths(centroid, contour, angles, img_shape, img_color):
    widths = []
    for angle in angles:
        inner, outer = find_edge_intersections(centroid, contour, angle, img_shape)
        # inner_negative, outer_negative = find_edge_intersections(centroid, contour, angle + np.pi, img_shape)

        # Draw the intersections if they exist
        if inner:
            cv2.circle(img_color, inner, 1, (255, 0, 0), -1)
        if outer:
            cv2.circle(img_color, outer, 1, (255, 0, 0), -1)


        if inner and outer:
            widths.append(np.linalg.norm(np.subtract(inner, outer)))

    # Only include widths where both intersections were found
    return widths

# Updated visualization function to include rays

# Function to draw lines from centroid in given angles
def draw_lines(image, centroid, angles):
    for angle in angles:
        x, y = centroid
        # Set a large enough length for the line
        line_length = 5000
        x_end = int(x + line_length * np.cos(angle))
        y_end = int(y + line_length * np.sin(angle))
        cv2.line(image, (x, y), (x_end, y_end), (0, 0, 255), 1)
    return image

def draw_intersections_and_rays(image, centroid, contour, angles, img_shape):
    for angle in angles:
        # Draw the ray from the centroid
        max_ray_length = max(img_shape[0], img_shape[1])
        end_point_x = int(centroid[0] + max_ray_length * np.cos(angle))
        end_point_y = int(centroid[1] + max_ray_length * np.sin(angle))
        cv2.line(image, centroid, (end_point_x, end_point_y), (255, 255, 0), 1)

        # Get the innermost and outermost intersections in one direction
        inner, outer = find_edge_intersections(centroid, contour, angle, img_shape)
        if inner:
            cv2.circle(image, inner, 2, (0, 255, 0), -1)  # Green for inner intersection
        if outer:
            cv2.circle(image, outer, 2, (255, 0, 0), -1)  # Red for outer intersection

        # Check the opposite direction as well
        inner_opposite, outer_opposite = find_edge_intersections(centroid, contour, angle + np.pi, img_shape)
        if inner_opposite:
            cv2.circle(image, inner_opposite, 2, (0, 255, 0), -1)  # Green for inner intersection
        if outer_opposite:
            cv2.circle(image, outer_opposite, 2, (255, 0, 0), -1)  # Red for outer intersection

    return image


def calculate_capsule_width(tumor_label, capsule_label, plotfigure_tag):
    # Find contours
    tumor_label_ = np.uint8(tumor_label)
    ret, binary_tumor_label_ = cv2.threshold(tumor_label_, 0, 1, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary_tumor_label_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnt = max(contours, key=cv2.contourArea)  # Assume we are interested in the largest contour
    cnt = contours[0]

    capsule_label_ = np.uint8(capsule_label)
    ret, binary_capsule_label_ = cv2.threshold(capsule_label_, 0, 1, cv2.THRESH_BINARY)

    capsule_contours, _ = cv2.findContours(binary_capsule_label_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # Calculate the centroid of the contour
    M = cv2.moments(cnt)
    cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
    centroid = (cx, cy)

    # Define angles for rays
    angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)

    # Convert to a color image to draw colored annotations
    img_color = cv2.cvtColor(capsule_label_*255, cv2.COLOR_GRAY2BGR)


    # # Draw the contours
    # for contour in capsule_contours:
    #     cv2.drawContours(img_color, [contour], -1, (0, 255, 0), 2)

    # Draw the centroid
    cv2.circle(img_color, centroid, 1, (0, 0, 255), -1)

    # Draw the rays from the centroid
    img_color = draw_lines(img_color, (cx, cy), angles)


    Widths = []
    for contour in capsule_contours:
        widths = calculate_widths(centroid, contour, angles, tumor_label.shape, img_color)
        Widths += widths

    # Calculate the average width, excluding rays that didn't intersect the contour
    average_width = np.mean(Widths) if Widths else None
    # print(f"Average Width: {average_width}")

    if plotfigure_tag:
        if average_width is not None:
            # Display the result
            plt.figure(figsize=(10, 10))
            plt.imshow(img_color[..., ::-1])  # Convert BGR to RGB for displaying
            plt.title(f"Average Width: {average_width:.2f} pixels")
            plt.axis('off')  # Hide the axis
            plt.show()


    return average_width


# calculate_capsule_width(tumor_label=img, capsule_label=img)