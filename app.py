from flask import Flask, render_template, session, redirect, url_for , request , send_file , Response
import os
import random
import csv
import cv2
import pytesseract
from PIL import Image
import uuid
import cv2
import numpy as np
from skimage import measure, morphology
import pytesseract
from PIL import Image
import re
from docx import Document
import io
import matplotlib.pyplot as plt
import base64
import os


app = Flask(__name__ ,static_folder='static/assets')
app.secret_key = 'Recaptcha'


def extract_low_confidence_words(image_path):
    pytesseract.pytesseract.tesseract_cmd =r'/usr/bin/tessecart'
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to load image at path: {image_path}")

    saved_image_paths = []
    # Use Pytesseract to do OCR on the image
    ocr_result = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    low_confidence_words = []
    low_confidence_words = [word for word in ocr_result['text'] if word.strip() and word not in [',', '']]
    for i, word in enumerate(low_confidence_words):
        # Filter out words where confidence is below a threshold (e.g., 50)
        # Extract bounding box coordinates
            (x, y, w, h) = (ocr_result['left'][i], ocr_result['top'][i], 
                            ocr_result['width'][i], ocr_result['height'][i])
            
            # Crop the image to this bounding box
            cropped_image = image[y:y+h, x:x+w]

            # Save the cropped image
            cropped_image_path = f"cropped_words/{word}_{i}.jpg"
            cv2.imwrite(cropped_image_path, cropped_image)
            saved_image_paths.append(cropped_image_path)

    print(saved_image_paths)

    return saved_image_paths

def get_random_image_and_text_from_folder(folder_path):
    images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    if not images:
        return None, None

    random_image = random.choice(images)
    random_image_path = os.path.join(folder_path, random_image)

    # Replace backslashes with forward slashes in the file path
    random_image_path = random_image_path.replace("\\", "/")
    random_image_path= random_image_path.replace("static/", "")
    # Assuming the text is the filename without extension
    text = os.path.splitext(random_image)[0]

    return random_image_path, text


# Get a random known image and its text
known_image_path, known_text = get_random_image_and_text_from_folder('static/assets/known_words_images')
# Get a random unknown image and its text
unknown_image_path, unknown_text = get_random_image_and_text_from_folder('static/assets/unknown_words_images')
print(f"Known Image Path: {known_image_path}")
print(f"Unknown Image Path: {unknown_image_path}")

@app.route('/')
def index():
    return render_template('index.html',
                           known_image_path=known_image_path,
                           unknown_image_path=unknown_image_path,
                           known_text=known_text)


#Function signature
def signature(img):
    # Convert PIL image to grayscale numpy array if necessary
    if not isinstance(img, np.ndarray):
        img = np.array(img.convert('L'))

    img_binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]

    # Connected component analysis by scikit-learn framework
    blobs = img_binary > img_binary.mean()
    blobs_labels = measure.label(blobs, background=1)

    # Calculate properties of connected components
    properties = measure.regionprops(blobs_labels)

    # Filter components based on area
    valid_components = [prop for prop in properties if prop.area >= 10]

    # Calculate average area
    total_area = sum(prop.area for prop in valid_components)
    average_area = total_area / len(valid_components) if valid_components else 0.0

    # Define constants for filtering
    a4_small_size_outliar_constant = ((average_area / 84) * 250) + 100
    a4_big_size_outliar_constant = a4_small_size_outliar_constant * 18

    # Remove small and big connected components
    pre_version = morphology.remove_small_objects(blobs_labels, a4_small_size_outliar_constant)

    component_sizes = np.bincount(pre_version.ravel())
    too_small = component_sizes > a4_big_size_outliar_constant
    too_small_mask = too_small[pre_version]
    pre_version[too_small_mask] = 0

    # Save the pre-version
    plt.imsave('pre_version.png', pre_version)

    # Read the pre-version
    img_pre_version = cv2.imread('pre_version.png', 0)

    
    # Ensure binary
    img_result = cv2.threshold(img_pre_version, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    return img_result

@app.route('/convert' , methods=['GET', 'POST'])
def challenge():
    extracted_text = None
    img_data = None
    docx_filename = None  # Initialize to None

    pytesseract.pytesseract.tesseract_cmd =r'C:\Program Files\Tesseract-OCR\tesseract.exe'


    # Check if the temp directory exists, create if not
    temp_dir = 'temp'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)


    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file and uploaded_file.filename.endswith(('png', 'jpg', 'jpeg')):
            # Read the image into a byte stream
            in_memory_file = io.BytesIO()
            uploaded_file.save(in_memory_file)
            in_memory_file.seek(0)
            file_bytes = np.asarray(bytearray(in_memory_file.read()), dtype=np.uint8)

            # Convert to an OpenCV image
            cv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Convert OpenCV image to base64 for HTML display
            _, buffer = cv2.imencode('.png', cv_image)
            img_data = base64.b64encode(buffer).decode('utf-8')

            # OCR and text extraction
            extracted_text = pytesseract.image_to_string(cv_image)

            # Save extracted text to a temporary DOCX file
            if extracted_text:  # Ensure there is text to save
                doc = Document()
                doc.add_paragraph(extracted_text)
                docx_filename = f"{uuid.uuid4()}.docx"
                doc.save(f"temp/{docx_filename}")

    return render_template('challenge.html', extracted_text=extracted_text, img_data=img_data, docx_filename=docx_filename)




@app.route('/download/<filename>')
def download_file(filename):
    path = os.path.join(app.root_path, 'temp', filename)
    with open(path, 'rb') as file:
        
        data = file.read()

    response = Response(data, mimetype="application/octet-stream", headers={"Content-Disposition": f"attachment;filename={filename}"})
    return response



def select_newspaper_image():
    images = os.listdir(os.path.join('static', 'newspaper_images'))
    selected_image = random.choice(images)
    return os.path.join('static', 'newspaper_images', selected_image)

@app.route('/verify_recaptcha', methods=['POST'])
def verify_recaptcha():
    # Retrieve the stored challenge words
    challenge_words = session.get('challenge_words', [])

    # Process each word in the challenge
    for i, word in enumerate(challenge_words):
        user_input = request.form.get(f'word_{i+1}', '').strip()
        
        # Compare user input with the challenge word
        if user_input.lower() != word.lower():
            # If any word doesn't match, re-attempt the challenge
            return redirect(url_for('index'))

    # If all words match, redirect to the main content page
    session['verified'] = True
    return redirect(url_for('main_content'))

@app.route('/main_content')
def main_content():
    # Ensure the user has passed the reCAPTCHA
    if not session.get('verified'):
        return redirect(url_for('index'))

    # Render the main content page
    return render_template('main_content.html')
#Store in csv
def store_user_input(unknown_image_path, user_input):
    # Open the CSV file in append mode and write the user input
    file_path = 'user_inputs.csv'
    fieldnames = ['unknown_image_path', 'user_input']
    

    # Check if the file already exists
    file_exists = os.path.isfile(file_path)
    # Extract just the name from the unknown_image_path
    image_name = os.path.basename(unknown_image_path)
    print(image_name)

    # Open the CSV file in append mode and write the user input
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        # Write headers only if the file is being created
        if not file_exists:
            writer.writerow(fieldnames)

        # Write a row containing the values of unknown_image_path and user_input
        writer.writerow([image_name, user_input])

@app.route('/submit', methods=['POST'])
def submit():
    input_text = request.form['input_text']

    # Split the input text into two variables separated by space
    text_values = input_text.split(' ')

    # Assuming you want the two values in separate variables
    if len(text_values) >= 2:
        known_user_input = text_values[0]
        unknown_user_input = text_values[1]
    else:
        # Handle the case when there is only one word
        known_user_input = input_text
        unknown_user_input = ""
    # Do something with the user inputs, for example, print them
    print(f"Known User Input: {known_user_input}")
    print(f"Unknown User Input: {unknown_user_input}")
    # Check if the condition is true for the known input
    if known_user_input == known_text:
        # Store the user's input for the unknown image
        store_user_input(unknown_image_path,unknown_user_input)
        # Pass a variable to the template indicating whether to show reCAPTCHA
        return render_template('challenge.html')
    else:
        # Redirect to the home page with the reCAPTCHA section visible
        return redirect(url_for('index'))
    


@app.route('/verification' , methods=['GET', 'POST'])
def verification():
    return render_template('verification.html',
                           known_image_path=known_image_path,
                           unknown_image_path=unknown_image_path,
                           known_text=known_text
                           )
if __name__ == '__main__':
    app.run(debug=True)
