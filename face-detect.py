import cv2
import face_recognition

# Path to the image file
img_path = "pic1.jpeg"

# Load the image
img = cv2.imread(img_path)

# Load a pre-trained face recognition model
encodings = []
names = []

# Set the screen size for display
screen_width = 800
screen_height = 600

# Convert the image to RGB for face recognition
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Find faces in the image using the face_recognition library
face_locations = face_recognition.face_locations(rgb_img)
face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

# Iterate over each detected face
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # Draw a rectangle around the face
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

    # Check if the face matches any known faces
    matches = face_recognition.compare_faces(encodings, face_encoding)
    face_name = "peoples"

    # If a match is found, use the name of the known face
    if True in matches:
        match_index = matches.index(True)
        face_name = names[match_index]

    # Display the name near the face
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img, face_name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

# Display the processed image
cv2.imshow('Face Detection and Recognition System', img)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()