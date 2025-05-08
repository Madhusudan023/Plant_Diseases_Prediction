import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import datetime
import os

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) # return index of max element

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition", "Product Recommendation", "Crop Info"])

# Main Page
if app_mode == "Home":
    st.header("IDENTIFY AND SOLVE DISEASE IN PLANT")
    image_path = "home_page.jpeg"
    st.image(image_path, use_container_width=False)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
                
    Our goal is to help farmers identify plant diseases efficiently. Simply upload an image of the affected plant, and our AI-powered system will analyze it to detect any signs of disease. Together, let's protect our crops and ensure a healthier, more sustainable harvest! Our platform is an AI-powered digital solution designed to help farmers identify, manage, and prevent crop diseases efficiently. By leveraging computer vision, deep learning, and IoT technologies, we provide real-time disease detection and expert treatment recommendations to ensure healthier crops and better yields.
                
    ### How It Works 
    -1️⃣ Submit an Image:Go to the **Disease Detection** page and upload a clear photo of the affected plant.
    -2️⃣ AI Processing:Our advanced deep learning model will analyze the image to detect potential diseases.
    -3️⃣ Diagnosis & Solutions:Receive instant results with disease identification and expert-recommended treatments.

    ### Why Choose Us?
    - **High Precision:** Our AI-driven model ensures reliable and accurate plant disease detection.  
    - **Simple & Intuitive:** User-friendly interface designed for easy navigation and seamless experience.  
    - **Quick & Effective:** Get instant disease analysis and recommendations, enabling fast decision-making.  

    ### Get Started  
    Click on the **🌿 Disease Recognition** page in the sidebar to **upload an image** and experience the power of our **AI-Powered Plant Disease Detection System!** 

    ### About Us
    Learn more about our **project, team, and mission** on the ** About** page. Let's work together for a healthier, more sustainable future in agriculture!   
    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
                ### Our Platform  
                Our platform is an **AI-powered digital solution** designed to assist farmers in **detecting, managing, and preventing crop diseases** effectively.  

                #### How It Helps Farmers 
                *"Our intelligent system enables farmers to diagnose crop diseases using image analysis and environmental data, delivering fast and precise treatment recommendations."*  

                ---
                By integrating **computer vision, deep learning, and IoT technologies**, our platform ensures **real-time disease detection** and provides expert guidance for **healthier crops and increased yields.** 🌾🚀
            
                #### About Dataset 
                This dataset is an **enhanced version** created using IEEE and Kagle augmentation from the original dataset, which can be found on the **GitHub repository respectively**.  

                It consists of **87,000+ RGB images** of **healthy and diseased crop leaves**, categorized into **38 different classes**. The dataset is split into an **80/20 ratio** for training and validation, maintaining the original directory structure. Additionally, a separate directory containing **33 test images** has been created for prediction purposes.  

                #### Dataset Structure 📂
                1. **Train:** 70,295 images  
                2. **Test:** 33 images  
                3. **Validation:** 17,572 images  
                """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, width=4, use_column_width=True)
        # Predict button
        if st.button("Predict"):
            st.snow()
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            # Reading Labels
            class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                        'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                        'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                        'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                        'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                        'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                        'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                        'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                        'Tomato___healthy']
            st.success("Model is Predicting it's a {}".format(class_name[result_index]))

# Product Recommendation Page
elif app_mode == "Product Recommendation":
    st.header("Product Recommendation")
    image_path = "assets/Pesticides_recomdation_image.png"
    if os.path.exists(image_path):
        img = Image.open(image_path)
        img = img.resize((300, int(400 * img.height / img.width)))
        st.image(image_path, caption="Pesticides Recommendation", use_container_width=False)
    else:
        st.error(f"Image not found: {image_path}")
    
    st.markdown("""
                Select a plant disease from the dropdown menu to see the recommended products.
                """)

    data = {
        "Plant Disease": [
            "Apple - Apple Scab", "Apple - Black Rot", "Apple - Cedar Apple Rust", "Apple - Healthy",
            "Blueberry - Healthy", "Cherry - Powdery Mildew", "Cherry - Healthy", "Corn - Cercospora Leaf Spot",
            "Corn - Common Rust", "Corn - Northern Leaf Blight", "Corn - Healthy", "Grape - Black Rot",
            "Grape - Esca (Black Measles)", "Grape - Leaf Blight", "Grape - Healthy", "Orange - Citrus Greening",
            "Peach - Bacterial Spot", "Peach - Healthy", "Pepper - Bacterial Spot", "Pepper - Healthy",
            "Potato - Early Blight", "Potato - Late Blight", "Potato - Healthy", "Raspberry - Healthy",
            "Soybean - Healthy", "Squash - Powdery Mildew", "Strawberry - Leaf Scorch", "Strawberry - Healthy",
            "Tomato - Bacterial Spot", "Tomato - Early Blight", "Tomato - Late Blight", "Tomato - Leaf Mold",
            "Tomato - Septoria Leaf Spot", "Tomato - Spider Mites", "Tomato - Target Spot", "Tomato - Yellow Leaf Curl Virus",
            "Tomato - Mosaic Virus", "Tomato - Healthy"
        ],
        "Pesticide/Insecticide": [
            "Mancozeb", "Captan", "Copper-based Fungicide", "None",
            "None", "Sulfur Spray", "None", "Azoxystrobin",
            "Chlorothalonil", "Propiconazole", "None", "Mancozeb",
            "Myclobutanil", "Copper Hydroxide", "None", "Imidacloprid",
            "Copper Oxychloride", "None", "Copper Hydroxide", "None",
            "Chlorothalonil", "Metalaxyl", "None", "None",
            "None", "Sulfur Fungicide", "Captan", "None",
            "Copper Hydroxide", "Chlorothalonil", "Metalaxyl", "Chlorothalonil",
            "Propiconazole", "Abamectin", "Difenoconazole", "Thiamethoxam",
            "Copper Hydroxide", "None"
        ],
        "Fertilizer/Manure": [
            "Organic Compost", "NPK 10-10-10", "Potassium Sulfate", "Compost Tea",
            "Mulch", "Calcium Nitrate", "Organic Compost", "Urea",
            "NPK 20-20-20", "Potassium Nitrate", "Organic Mulch", "Bone Meal",
            "Fish Emulsion", "Seaweed Extract", "Mulch", "Micronutrient Fertilizer",
            "Potassium Phosphate", "Organic Mulch", "Calcium Carbonate", "Mulch",
            "Ammonium Sulfate", "Calcium Nitrate", "Organic Compost", "Compost",
            "Soybean Meal", "Neem Cake", "Bone Meal", "Organic Compost",
            "Calcium Nitrate", "Fish Emulsion", "Potassium Sulfate", "Compost Tea",
            "Organic Manure", "Sulfur Dust", "Phosphoric Acid", "Bio-fertilizer",
            "Seaweed Extract", "Organic Compost"
        ],
        "Product Image": [
            "Mancozeb.jpg", "Captan_fungicide.webp", "Copper_Hydroxide.jpeg", "",
            "", "Sulfur_Spray.jpeg", "", "Azoxystrobin_fungicide.png",
            "Chlorothalonil-720-1.png", "Propiconazole.jpeg", "", "Mancozeb.jpg",
            "Myclobutanil.jpeg", "Copper_Hydroxide.jpeg", "", "Imidacloprid.jpeg",
            "Copper_Oxychloride.jpeg", "", "Copper_Hydroxide.jpeg", "",
            "Chlorothalonil-720-1.png", "Metalaxyl.jpeg", "", "",
            "", "Sulfur_Spray.jpeg", "Captan_fungicide.webp", "",
            "Copper_Hydroxide.jpeg", "Chlorothalonil-720-1.png", "Metalaxyl.jpeg", "Chlorothalonil-720-1.png",
            "Propiconazole.jpeg", "Abamectin_insecticide.jpeg", "Difenoconazole.jpeg", "Thiamethoxam_insecticide.jpeg",
            "Copper_Hydroxide.jpeg", "mosaic_virus_tomato.jpg"
        ],
            "Purchase Link": [
        "https://www.amazon.com/s?k=mancozeb",  # Mancozeb
        "https://www.domyown.com/captan-fungicide-s-5.html",  # Captan
        "https://www.amazon.com/s?k=copper+fungicide",  # Copper Fungicide
        "",  # None
        "",  # None
        "https://www.amazon.com/s?k=sulfur+fungicide",  # Sulfur Spray
        "",  # None
        "https://www.syngenta-us.com/fungicides/amistar",  # Azoxystrobin
        "https://www.syngenta-us.com/fungicides/bravo-weather-stik",  # Chlorothalonil
        "https://www.syngenta-us.com/fungicides/tilt",  # Propiconazole
        "",  # None
        "https://www.amazon.com/s?k=mancozeb",  # Mancozeb
        "https://www.amazon.com/s?k=myclobutanil",  # Myclobutanil
        "https://www.amazon.com/s?k=copper+hydroxide",  # Copper Hydroxide
        "",  # None
        "https://www.amazon.com/s?k=imidacloprid",  # Imidacloprid
        "https://www.amazon.com/s?k=copper+oxychloride",  # Copper Oxychloride
        "",  # None
        "https://www.amazon.com/s?k=copper+hydroxide",  # Copper Hydroxide
        "",  # None
        "https://www.syngenta-us.com/fungicides/bravo-weather-stik",  # Chlorothalonil
        "https://www.syngenta-us.com/fungicides/ridomil-gold-sl",  # Metalaxyl
        "",  # None
        "",  # None
        "",  # None
        "https://www.amazon.com/s?k=sulfur+fungicide",  # Sulfur Fungicide
        "https://www.domyown.com/captan-fungicide-s-5.html",  # Captan
        "",  # None
        "https://www.amazon.com/s?k=copper+hydroxide",  # Copper Hydroxide
        "https://www.syngenta-us.com/fungicides/bravo-weather-stik",  # Chlorothalonil
        "https://www.syngenta-us.com/fungicides/ridomil-gold-sl",  # Metalaxyl
        "https://www.syngenta-us.com/fungicides/bravo-weather-stik",  # Chlorothalonil
        "https://www.syngenta-us.com/fungicides/tilt",  # Propiconazole
        "https://www.amazon.com/s?k=abamectin+insecticide",  # Abamectin
        "https://www.amazon.com/s?k=difenoconazole",  # Difenoconazole
        "https://www.amazon.com/s?k=thiamethoxam",  # Thiamethoxam
        "https://www.amazon.com/s?k=copper+hydroxide",  # Copper Hydroxide
        ""  # None
    ]
    }

    df = pd.DataFrame(data)
    disease_selected = st.selectbox("Select a plant disease", df["Plant Disease"].unique())
    filtered_df = df[df["Plant Disease"] == disease_selected]
    
    st.write("### Recommended Products")
    for index, row in filtered_df.iterrows():
        if row["Product Image"]:  # Only show if there's an image
            image_path = os.path.join("assets", row["Product Image"])
            
            if os.path.exists(image_path):
                st.image(image_path, caption=row["Pesticide/Insecticide"], use_column_width=True)
            else:
                st.error(f"Image not found: {image_path}")
        
        st.write(f"**Pesticide/Insecticide:** {row['Pesticide/Insecticide']}")
        st.write(f"**Fertilizer/Manure:** {row['Fertilizer/Manure']}")
        if row["Purchase Link"]:
         st.markdown(f"[Purchase this product]({row['Purchase Link']})", unsafe_allow_html=True)
        st.write("---")

# Crop Info Page
elif app_mode == "Crop Info":
    st.header("Crop Information")
    
    
    crop_data = {
        "Apple": {
            "sowing_start": "December", "sowing_end": "February", "harvest_days": 150,
            "irrigation": "Every 10-15 days", "required_irrigation": "800-1200 mm",
            "weeding": "Once every 30 days",
            "fertilizer": "NPK 70:40:70 kg/ha",
            "insecticides": "Chlorpyrifos, Malathion",
            "pesticides": "Copper Sulfate, Mancozeb",
            "manure": "Farmyard Manure, Compost"
        },
        "Bajra": {
            "sowing_start": "June", "sowing_end": "July", "harvest_days": 90,
            "irrigation": "Once every 10-12 days", "required_irrigation": "300-500 mm",
            "weeding": "1st at 20 days, 2nd at 40 days",
            "fertilizer": "NPK 60:30:30 kg/ha",
            "insecticides": "Thiamethoxam, Lambda Cyhalothrin",
            "pesticides": "Metalaxyl, Carbendazim",
            "manure": "Vermicompost, Poultry Manure"
        },
        "Cotton": {
            "sowing_start": "April", "sowing_end": "May", "harvest_days": 150,
            "irrigation": "Every 7-10 days", "required_irrigation": "700-900 mm",
            "weeding": "1st at 20 days, 2nd at 45 days",
            "fertilizer": "NPK 90:45:45 kg/ha",
            "insecticides": "Imidacloprid, Spinosad",
            "pesticides": "Chlorothalonil, Mancozeb",
            "manure": "Green Manure, Organic Compost"
        },
        "Cereals": {
            "sowing_start": "November", "sowing_end": "January", "harvest_days": 120,
            "irrigation": "Every 10-14 days", "required_irrigation": "500-800 mm",
            "weeding": "1st at 25 days, 2nd at 50 days",
            "fertilizer": "NPK 80:40:40 kg/ha",
            "insecticides": "Chlorantraniliprole, Lambda Cyhalothrin",
            "pesticides": "Azoxystrobin, Captan",
            "manure": "Farmyard Manure, Compost"
        },
        "Jowar": {
            "sowing_start": "June", "sowing_end": "July", "harvest_days": 100,
            "irrigation": "Once every 12-15 days", "required_irrigation": "300-600 mm",
            "weeding": "1st at 20 days, 2nd at 50 days",
            "fertilizer": "NPK 50:25:25 kg/ha",
            "insecticides": "Dimethoate, Spinosad",
            "pesticides": "Mancozeb, Zineb",
            "manure": "Vermicompost, Green Manure"
        },
        "Maize (Corn)": { 
            "sowing_start": "June", "sowing_end": "July", "harvest_days": 90,
            "irrigation": "Every 6-9 days", "required_irrigation": "500-700 mm",
            "weeding": "1st at 20 days, 2nd at 40 days", "fertilizer": "NPK 90:60:40 kg/ha",
            "insecticides": "Thiamethoxam, Chlorpyrifos", "pesticides": "Carbendazim, Propiconazole",
            "manure": "Compost, Cow Dung Manure"
        },
        "Onion": {
            "sowing_start": "October", "sowing_end": "December", "harvest_days": 120,
            "irrigation": "Every 5-7 days", "required_irrigation": "600-800 mm",
            "weeding": "1st at 15 days, 2nd at 45 days",
            "fertilizer": "NPK 70:30:30 kg/ha",
            "insecticides": "Deltamethrin, Malathion",
            "pesticides": "Copper Oxychloride, Mancozeb",
            "manure": "Compost, Poultry Manure"
        },
        "Pulses": {
            "sowing_start": "June", "sowing_end": "July", "harvest_days": 90,
            "irrigation": "Once every 12-15 days", "required_irrigation": "400-600 mm",
            "weeding": "1st at 25 days, 2nd at 45 days",
            "fertilizer": "NPK 40:20:20 kg/ha",
            "insecticides": "Lambda Cyhalothrin, Acephate",
            "pesticides": "Chlorothalonil, Carbendazim",
            "manure": "Green Manure, Organic Manure"
        },
        "Strawberry": {
            "sowing_start": "September", "sowing_end": "November", "harvest_days": 120,
            "irrigation": "Every 3-5 days", "required_irrigation": "500-700 mm",
            "weeding": "Regular hand weeding every 15 days",
            "fertilizer": "NPK 80:60:40 kg/ha",
            "insecticides": "Spinosad, Thiamethoxam",
            "pesticides": "Copper Hydroxide, Mancozeb",
            "manure": "Cow Dung Manure, Compost"
        },
        "Tomato": {
            "sowing_start": "June", "sowing_end": "August", "harvest_days": 90,
            "irrigation": "Every 5-7 days", "required_irrigation": "600-800 mm",
            "weeding": "1st at 15 days, 2nd at 40 days",
            "fertilizer": "NPK 100:50:50 kg/ha",
            "insecticides": "Imidacloprid, Spinosad",
            "pesticides": "Mancozeb, Copper Oxychloride",
            "manure": "Compost, Farmyard Manure"
        },
        "Vegetables": {
            "sowing_start": "March", "sowing_end": "May", "harvest_days": 70,
            "irrigation": "Every 4-6 days", "required_irrigation": "500-700 mm",
            "weeding": "1st at 10 days, 2nd at 35 days",
            "fertilizer": "NPK 60:30:30 kg/ha",
            "insecticides": "Chlorpyrifos, Acephate",
            "pesticides": "Captan, Azoxystrobin",
            "manure": "Organic Compost, Vermicompost"
        }
    }
    
    crop_selected = st.selectbox("Select a Crop", list(crop_data.keys()))

    #    crop_selected = st.selectbox("Select a Crop", list(crop_data.keys()))

    # Sowing Date Input
    sowing_date = st.date_input("Enter Crop Sowing Date", datetime.date.today())
    sowing_month = sowing_date.strftime("%B")  # e.g., "June"
    sowing_month_num = sowing_date.month       # e.g., 6 for June

    # Convert sowing start/end months to numbers
    month_name_to_num = {
        "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
        "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
    }
    
    start_month = crop_data[crop_selected]["sowing_start"]  # e.g., "June"
    end_month = crop_data[crop_selected]["sowing_end"]      # e.g., "July"
    start_num = month_name_to_num[start_month]              # e.g., 6
    end_num = month_name_to_num[end_month]                  # e.g., 7

    # Handle cross-year ranges (e.g., November–January)
    if start_num > end_num:
        # Case like November (11) to January (1)
        valid_sowing = (sowing_month_num >= start_num) or (sowing_month_num <= end_num)
    else:
        # Normal case (e.g., June–July)
        valid_sowing = (start_num <= sowing_month_num <= end_num)

    if not valid_sowing:
        st.error(f"⚠️ {crop_selected} should be sown between {crop_data[crop_selected]['sowing_start']} and {crop_data[crop_selected]['sowing_end']}.")
    else:
        # Calculate Harvest Date
        harvest_date = sowing_date + datetime.timedelta(days=crop_data[crop_selected]["harvest_days"])

        # Display Crop Details (same as before)
        st.subheader(f"🌱 {crop_selected} Farming Details")
        st.markdown(f"""
        - **Irrigation Frequency:** {crop_data[crop_selected]["irrigation"]}
        - **Total Water Requirement:** {crop_data[crop_selected]["required_irrigation"]}
        - **Weeding Schedule:** {crop_data[crop_selected]["weeding"]}
        - **Recommended Fertilizer:** {crop_data[crop_selected]["fertilizer"]}
        - **Common Insecticides:** {crop_data[crop_selected]["insecticides"]}
        - **Common Pesticides:** {crop_data[crop_selected]["pesticides"]}
        - **Suggested Manure:** {crop_data[crop_selected]["manure"]}
        - **Estimated Harvest Date:** 📅 {harvest_date.strftime("%d-%B-%Y")}
        """)
        st.success(f"✅ Your {crop_selected} crop is expected to be ready for harvest by {harvest_date.strftime('%d-%B-%Y')}!")