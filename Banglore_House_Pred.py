#Author - Abdul Faheem (AF011)
import numpy as np
import pandas as pd
import json
import pickle as pkl
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
   page_title="House Price Prediction",
   page_icon="🏠",
   layout="wide",
   initial_sidebar_state="expanded",
)


# Loading column data and the model
with open('home_price_pred_cols.json') as f:
    df_cols = json.load(f)
X = np.array(df_cols['data_columns'])
with open('banglore_home_price_pred.pickle', "rb") as f:
    LR_Model = pkl.load(f)

# Prediction function
def predict(sqft, bath, balcony, bhk, location, area_type):
    try:
        loc_index = np.where(X == location)[0]
        area_index = np.where(X == area_type.lower())[0]

        x = np.zeros(len(X))
        x[0] = sqft
        x[1] = bath
        x[2] = balcony
        x[3] = bhk
        
        if loc_index.size > 0:
            x[loc_index] = 1
        if area_index.size > 0:
            x[area_index] = 1

    except Exception as e:
        return f"An error occurred: {e}"
  
    return LR_Model.predict([x])[0]

#Scatter Plot Plotting Function
def plot_scatter(df, location):
    st.divider()
    df1 = df[(df.location == location)]
    st.write(f"## Stat at a Glance - {location}")
    col1, col2, col3 = st.columns(3)
    stat = df1.price.describe()
    col1.metric(label = ":green[Min Home Price]", value = f"₹{stat[3]: .2f} L")
    col2.metric(label = ":green[Average Home Price]", value = f"₹{stat[1]: .2f} L")
    col3.metric(label = ":green[Max Home Price]", value = f"₹{stat[7]: .2f} L")
    
    df1.rename(columns = {'total_sqft' : 'Total Sqft', 'price' : 'Price', 'bhk' : 'BHK'}, inplace = True)
    st.caption(f'### :orange[Other Property Prices according to BHK in {location}]')
    st.scatter_chart(df1, x = 'Total Sqft', y = 'Price', color = 'BHK')

   
def footer():
    # Footer Section
    st.markdown('<style>div.block-container{padding-bottom: 100px;}</style>', unsafe_allow_html=True)
    st.markdown("""---""")
    st.markdown("""
        ### 🚀 Let's Connect!
        This Application is Developed by **Abdul Faheem** with 💡 and 🥤. If you have any questions or just want to connect, feel free to reach out!
        
        <p align="left">
          <a href="https://www.linkedin.com/in/abdulfaheem011/" target="_blank">
            <img src="https://img.icons8.com/fluent/48/000000/linkedin.png" alt="LinkedIn" style="width:40px;"/>
          </a>
          <a href="https://github.com/abdulfaheemaf" target="_blank">
            <img src="https://img.icons8.com/fluent/48/000000/github.png" alt="GitHub" style="width:40px;"/>
          </a>
          <a href="mailto:abdulfaheemaf11@gmail.com">
            <img src="https://img.icons8.com/fluent/48/000000/mail.png" alt="Email" style="width:40px;"/>
          </a>
        </p>
        
        Cheers,
        
        -AF011
    """, unsafe_allow_html=True)

    
def main():
    col1, col2 = st.columns(2)
    col1.title('Future Foundations')
    col1.caption('### Projecting Real Estate Values with AI  (Banglore)')
    col2.image('House_Price_Pred.jpg', width = 200)
    st.markdown('''*Future Foundations uses AI to change how we look at and guess
                  property prices in Bangalore. We look at a lot of data to find important patterns.
                  This helps house owners, investors, and real estate folks make smart choices. With
                  our tool, you get to explore how tech reshapes buying and selling houses in one of
                  India's lively cities, all based on one detailed dataset.*''')
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        avail_loc = [val for val in X[4:-3]]
        avail_loc.append('Other')
        location = st.selectbox("Location", list(avail_loc), 
                            placeholder = "Select Location Here...!")

        avail_area = [val.title() for val in X[-3:]]
        avail_area.append('Built-up Area')
        area_type = st.selectbox("Area Type", list(avail_area),
                                 placeholder = "Select the Area Type Here...!")

        sqft = st.number_input("Area (Sqft)", max_value = 30000, min_value = 300,
                                       value = 1000, step = 50)
    with col2:
        bhk = st.slider("No. of Bedrooms", max_value = 16, min_value = 1, value = 3)
        bath = st.slider("No. of Bathrooms", max_value = 16, min_value = 1, value = 2)
        balcony = st.slider("No. of Balconies", max_value = 3, min_value = 1, value = 1)
    
    result = predict(sqft, bath, balcony, bhk, location, area_type)
    submit_button = st.button("Predict")
    df = pd.read_csv('banglore_house_price.csv')
        
    if submit_button:
        larger_text = f'''**<h4>Voilà! The future home of your dreams in :green[{location}] is valued at a splendid :rainbow[₹{result:.2f}] Lakhs 🌟</h4>**'''
        st.markdown(larger_text, unsafe_allow_html=True)

        plot_scatter(df, location)

    footer()

    
if __name__ == '__main__':
    main()
