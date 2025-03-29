import os
import streamlit as st
from predict import predict_parkinson
from PIL import Image
import tempfile

# Configuration
TEMP_DIR = tempfile.mkdtemp()
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
SUPPORTED_TYPES = ["jpg", "jpeg", "png"]

def set_custom_style():
    st.markdown("""
    <style>
    /* Main app background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #2c3e50;
    }
    
    /* Sidebar text - ensure visibility */
    .sidebar .sidebar-content .stMarkdown,
    .sidebar .sidebar-content .stMarkdown p,
    .sidebar .sidebar-content .stMarkdown h1,
    .sidebar .sidebar-content .stMarkdown h2,
    .sidebar .sidebar-content .stMarkdown h3,
    .sidebar .sidebar-content .stMarkdown h4,
    .sidebar .sidebar-content .stMarkdown li {
        color: #ecf0f1 !important;
    }
    
    /* Sidebar divider */
    .sidebar .sidebar-content hr {
        border-color: rgba(255,255,255,0.1) !important;
        margin: 1.5rem 0;
    }
    
    /* Title styling */
    .title-text {
        color: #2c3e50;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    /* Subtitle styling */
    .subtitle-text {
        color: #7f8c8d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* File uploader area */
    .stFileUploader > div {
        border: 2px dashed #bdc3c7;
        border-radius: 10px;
        padding: 2rem;
        background-color: white;
    }
    
    /* Result boxes */
    .result-box {
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    .parkinson-result {
        border-left: 5px solid #e74c3c;
    }
    
    .healthy-result {
        border-left: 5px solid #2ecc71;
    }
    
    /* Disclaimer box */
    .disclaimer-box {
        background-color: #fff8e1;
        border-left: 5px solid #ffc107;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 2rem;
    }
    
    /* Example image container */
    .example-container {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        height: 100%;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)

def validate_image(uploaded_file):
    """Validate the uploaded image file"""
    if uploaded_file.size > MAX_FILE_SIZE:
        raise ValueError(f"File too large. Max size is {MAX_FILE_SIZE//1024//1024}MB")
    
    ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    if ext not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported file type. Supported types: {', '.join(SUPPORTED_TYPES)}")
    
    try:
        Image.open(uploaded_file).verify()
    except Exception as e:
        raise ValueError(f"Invalid image file: {str(e)}")

def display_result(result):
    """Display the prediction result with appropriate styling"""
    if result['likelihood'] == "Parkinson's":
        st.markdown(f"""
        <div class="result-box parkinson-result">
            <h3 style="color: #e74c3c;">⚠️ Potential Parkinson's Detected</h3>
            <p style="color: #333;">Our analysis suggests characteristics consistent with Parkinson's disease in the spiral drawing.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-box healthy-result">
            <h3 style="color: #2ecc71;">✅ No Parkinson's Detected</h3>
            <p style="color: #333;">Our analysis didn't detect characteristics typically associated with Parkinson's disease.</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    # App configuration
    st.set_page_config(
        page_title="NeuroScan: Parkinson's Detection", 
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Set custom style
    set_custom_style()
    
    # Sidebar with improved text visibility
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="color: #ffffff !important;">🧠 NeuroScan</h1>
            <p style="color: #ecf0f1 !important;">Parkinson's Detection System</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
        <div>
            <h3 style="color: #ffffff !important;">About This Tool</h3>
            <p>This AI-based screening tool analyzes spiral drawings for potential signs of Parkinson's disease.</p>
            <p><strong>Important:</strong> This is not a diagnostic tool. Always consult a medical professional for proper evaluation.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
        <div style="font-size: 0.9rem; text-align: center;">
            <p style="color: #bdc3c7 !important;">For Testing Purpose only</p>
            <p style="color: #bdc3c7 !important;">© 2025 PCCOE Students</p>
        </div>
        """, unsafe_allow_html=True)

    # Main content
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<p class="title-text">Parkinson\'s Detection</p>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle-text">Upload a spiral drawing image for analysis</p>', unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=SUPPORTED_TYPES,
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            try:
                # Validate image
                validate_image(uploaded_file)
                
                # Create temp file
                temp_image_path = os.path.join(TEMP_DIR, f"temp_{uploaded_file.name}")
                with open(temp_image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Make prediction
                with st.spinner('Analyzing drawing patterns...'):
                    result = predict_parkinson(temp_image_path)
                
                # Display results
                st.markdown("---")
                display_result(result)
                
                # Show disclaimer
                st.markdown("""
                <div class="disclaimer-box">
                    <h4 style="color: #ff9800;">Medical Disclaimer</h4>
                    <p style="color: #333;">This tool provides preliminary screening only and cannot diagnose Parkinson's disease. The results should not be used as a substitute for professional medical advice.</p>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
            finally:
                # Clean up
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
    
    with col2:
        if uploaded_file is not None:
            # Display uploaded image
            st.markdown("""
            <div class="example-container">
                <h3 style="color: #2c3e50; text-align: center;">Your Spiral Drawing</h3>
            </div>
            """, unsafe_allow_html=True)
            st.image(uploaded_file, use_column_width=True)
        else:
            # Placeholder with instructions
            st.markdown("""
            <div class="example-container">
                <h3 style="color: #2c3e50; text-align: center;">How To Use</h3>
                <ol style="color: #34495e;">
                    <li style="margin-bottom: 0.5rem;">Draw a spiral on blank paper</li>
                    <li style="margin-bottom: 0.5rem;">Take a clear, well-lit photo</li>
                    <li style="margin-bottom: 0.5rem;">Upload the image for analysis</li>
                </ol>
                <div style="text-align: center; margin-top: 2rem;">
                    <p style="color: #7f8c8d;">Example of a proper spiral:</p>
                    <img src="https://i.imgur.com/JqYeZoL.png" width="70%" style="border-radius: 8px; margin-top: 1rem;">
                </div>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()