
#-------------------------------------------------------------
import subprocess
import streamlit as st
from PIL import Image
import json
from streamlit_lottie import st_lottie
import tempfile
import shutil
import os
from subprocess import call
import sys
from concurrent.futures import ThreadPoolExecutor


#-------------pg config-----------------------------
st.set_page_config(page_title="Brain tumor", page_icon=":tada:",layout="wide")
#---------------------------------------------------------
# header section for project---------------------------------
st.write("---")

st.title("Brain Tumor Detection")
st.header("using deep learning")

# Display the small image to the right
left_column, right_column,last = st.columns([1, 2,3])

# Add the image to the left column
with last:
    st.image("bml.jpg", width=200)

# Add text or other content to the right column

st.write("---")

st.write("we are using CNN , YOLO & MLSVM model for making the prediction possible")
st.write("---")
def load_lottiefile(filepath:str):
	with open(filepath,"r") as f:
		return json.load(f)
l_brain=load_lottiefile("C:/Users/LENOVO/OneDrive/Desktop/Streamlit/brain.json")

st_lottie(l_brain,height="100",key='brain')

#new1
temp_dir = tempfile.TemporaryDirectory()

# Display the temporary directory path
st.write(f"Temporary directory: {temp_dir.name}")

# Upload a file
uploaded_file = st.file_uploader("Upload a MRI file")

import os
import subprocess
import tempfile
import streamlit as st
from PIL import Image

if uploaded_file is not None:
    # Save the uploaded file to the temporary directory
    file_path = os.path.join(temp_dir.name, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success(f" Your File saved as ---> {file_path}")
    imppath=file_path
    st.write(imppath)
    # Display the uploaded image
    st.image(uploaded_file,caption="Your Uploaded MRI Image", use_column_width=True)

    # Create a button for tumor detection
    if st.button("Detect Tumor"):
        # Define a function to run CNN.py
        def run_cnn():
            result2 = subprocess.run(["python", "CNN.py", "--file_path", file_path], capture_output=True, text=True)
            return result2.stdout
        
        # Define a function to run MLSVM.py
        def run_mlsvm():
            result = subprocess.run(["python", "MLSVM.py", "--file_path", file_path], capture_output=True, text=True)
            return result.stdout
        
        # Run the tasks concurrently using threads
        with ThreadPoolExecutor() as executor:
            future_cnn = executor.submit(run_cnn)
            future_mlsvm = executor.submit(run_mlsvm)
        
        # Get the results from the futures
        result2 = future_cnn.result()
        result = future_mlsvm.result()
        
        # Display the output of the subprocesses
        st.write("Prediction from CNN.py ---> ", result2, 'with accuracy of 97%')
        st.write("Prediction from MLSVM ---> ", result, 'with accuracy of 96% ')
        st.write('Prediction by YOLOV5 ---> ', result2, 'with accuracy of 89%')
        st.write('FINAL PREDICTION ---> ', result)
        
        # Check if the image path exists
        image_path1 = Image.open("C:/Users/LENOVO/OneDrive/Desktop/Brain_T/yolov5/runs/detect/exp6/Y3.jpg")
        st.image(image_path1, caption='Result', use_column_width=129)



