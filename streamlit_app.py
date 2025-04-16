import streamlit as st
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return f"data:image/jpg;base64,{encoded}"


# Load the pre-trained model
model = load_model('VGG16.h5')
IMG_SIZE = (224, 224)




# --- Custom CSS Styling -- 
# 
# 
bg_image = get_base64_image("back.jpg")

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{bg_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""
    <style>
    /* Target the sidebar toggle button */
    [data-testid="collapsedControl"] {
        color: white !important; /* Icon color */
        background-color: #FF5733 !important; /* Button background */
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)





st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        height: 100%;
    }
    

    .main {
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    section[data-testid="stSidebar"] {
        background-color: #3E3F5B;
        padding: 20px;
        # border-right: 3px solid #FF5733;
    }

    .navbar {
        background-color: #99BC85;
        padding: 10px 30px;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 20px;
        margin-top: -60px;
    }

    .navbar h1 {
        color: white;
        font-size: 32px;
        margin: 0;
    }

    .subtitle {
        color: black;
        font-size: 18px;
        text-align: center;
        margin-bottom: 30px;
    }

    .image-container {
        text-align: center;
    }

    .uploaded-img {
        max-width: 400px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-top: 20px;
    }

    .prediction {
        color:white;
        background-color:#328E6E;
        padding:10px 20px 10px 20px;
        border-radius:5px;
        text-align:center;
    }

    .footer {
        margin-top: auto;
        text-align: center;
        font-size: 14px;
        color: white;
        padding: 20px;
    }
    
    .sidebar-upload{
        font-size:20px;
        color:#F5EEDC;
    }

    .sidebar-upload label {
        font-weight: bold;
        color: #FF5733;
    }
    
    [data-testid="collapsedControl"] {
        color: #DDA853 !important; /* Icon color */
        background-color: #DDA853 !important; /* Button background */
        border-radius: 5px;
    }
    

    .css-1cpxqw2 {
        background-color: #fefefe;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .stAppHeader{
        background-color:#E4EFE7;
        box-shadow: rgba(50, 50, 93, 0.25) 0px 6px 12px -2px, rgba(0, 0, 0, 0.3) 0px 3px 7px -3px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Navbar ---
st.markdown('<div class="navbar"><h1>Alzheimer\'s Disease Prediction</h1></div>', unsafe_allow_html=True)

# --- Subtitle ---
st.markdown(
    '<p class="subtitle">Upload a brain scan image to predict Alzheimer\'s condition using deep learning.</p>',
    unsafe_allow_html=True
)

# --- Sidebar Upload Section ---
# st.sidebar.markdown("### Upload Image")
st.sidebar.markdown('<div class="sidebar-upload">Please upload a JPG or PNG image of the brain scan.</div>', unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader(label="", type=['jpg', 'jpeg', 'png'])

# --- Main Logic ---
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.image(image, caption='Uploaded Image', width=200, use_container_width=False, output_format='auto')
    st.markdown('</div>', unsafe_allow_html=True)

    # Preprocess image
    img = image.convert("RGB").resize(IMG_SIZE)
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    class_labels = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']

    # Predict
    prediction_probs = model.predict(img_array)[0]
        
    def plot_prediction_bar_and_pie(probs, labels):
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [1, 1]}
        )

        # Define unique colors for each bar
        bar_colors = ['#4CAF50', '#FF9800', '#2196F3', '#9C27B0']  # You can use any custom color list

        # Bar Chart
        bars = ax1.bar(labels, probs * 100, color=bar_colors)
        ax1.set_ylim(0, 100)
        ax1.set_ylabel("Confidence (%)")
        ax1.set_title("Confidence per Class (Bar)")
        ax1.tick_params(axis='x', rotation=20)
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)

        for bar, prob in zip(bars, probs):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{prob * 100:.2f}%", ha='center', va='bottom', fontsize=9)

        # Pie Chart
        ax2.pie(probs, labels=labels, autopct='%1.1f%%', startangle=140,
                colors=bar_colors)  # Keep colors consistent
        ax2.set_title("Confidence Distribution (Pie)")

        # Add spacing between the two plots
        plt.subplots_adjust(wspace=0.4)

        st.pyplot(fig)



    plot_prediction_bar_and_pie(prediction_probs, class_labels)

    # Show the top prediction
    predicted_idx = np.argmax(prediction_probs)
    confidence = prediction_probs[predicted_idx]
    predicted_label = class_labels[predicted_idx]

    st.markdown(f"<div class='prediction'>Prediction Result: {predicted_label} ({confidence * 100:.2f}%)</div>", unsafe_allow_html=True)

    

    def generate_charts_as_images(probs, labels):
        # Save bar chart and pie chart separately to use in PDF
        bar_colors = ['#4CAF50', '#FF9800', '#2196F3', '#9C27B0']
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={'width_ratios': [1, 1]})

        # Bar chart
        bars = ax1.bar(labels, probs * 100, color=bar_colors)
        ax1.set_ylim(0, 100)
        ax1.set_ylabel("Confidence (%)")
        ax1.set_title("Confidence per Class (Bar)")
        ax1.tick_params(axis='x', rotation=20)
        for bar, prob in zip(bars, probs):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{prob * 100:.2f}%", ha='center', va='bottom', fontsize=8)

        # Pie chart
        ax2.pie(probs, labels=labels, autopct='%1.1f%%', startangle=140, colors=bar_colors)
        ax2.set_title("Confidence Distribution (Pie)")

        plt.tight_layout()

        # Save the figure to a BytesIO object
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='PNG')
        plt.close(fig)
        img_buf.seek(0)
        return img_buf

    def create_pdf_report(predicted_label, confidence, probs, labels, chart_image_buf):
        pdf_buf = io.BytesIO()
        c = canvas.Canvas(pdf_buf, pagesize=letter)
        width, height = letter

        # Title and prediction result
        c.setFont("Helvetica-Bold", 18)
        c.drawString(50, height - 60, "Alzheimer's Disease Prediction Report")

        c.setFont("Helvetica", 12)
        c.drawString(50, height - 100, f"Prediction: {predicted_label}")
        c.drawString(50, height - 120, f"Confidence: {confidence * 100:.2f}%")

        # Draw chart image
        chart_img = ImageReader(chart_image_buf)
        c.drawImage(chart_img, 50, height - 400, width=500, preserveAspectRatio=True, mask='auto')

        c.setFont("Helvetica-Oblique", 8)
        c.drawString(50, 50, "Generated using AI model | © 2025 Suraj Patel")

        c.save()
        pdf_buf.seek(0)
        return pdf_buf

# --- Generate report on button click ---
    if st.button("📄 Download Prediction Report as PDF"):
        chart_buf = generate_charts_as_images(prediction_probs, class_labels)
        pdf_buffer = create_pdf_report(predicted_label, confidence, prediction_probs, class_labels, chart_buf)

        st.download_button(
            label="⬇️ Click to Download PDF",
            data=pdf_buffer,
            file_name="alzheimers_prediction_report.pdf",
            mime="application/pdf"
        )



else:
    st.sidebar.info("Awaiting image upload...")

# --- Footer ---
st.sidebar.markdown("<div class='footer'>© 2025 | By Suraj Patel</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='footer'>Department of Pharmaceutical engineering, IIT (BHU) VARANASI</div>", unsafe_allow_html=True)
