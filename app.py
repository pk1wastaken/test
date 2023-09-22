import streamlit as st
import numpy as np
import string
import xgboost
import pickle
import os
from pathlib import Path
current_directory = Path(__file__).parent #Get current directory
file = open(os.path.join(current_directory, 'food_price.pkl'), 'rb')#rb = read bytes because we are reading the file
model = pickle.load(file) #We give it the file and it loads in everything from the file. That data is the loaded as the variable model
#filename = '../food_price.pkl'
#model = pickle.load(open(os.path.join(filename, 'rb')))
# st.set_option('deprecation.showfileUploaderEncoding',False)
#model = pickle.load(open('food_price.pkl','rb'))


def main():
    st.markdown("<h1 style='text-align: center; color: White;background-color:#e84343'>Food Price Predictor</h1>",
                unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: Black;'>Drop in The required Inputs and we will do  the rest.</h3>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: Black;'>Submission for XGBoost Hamoye Internship Project</h4>", unsafe_allow_html=True)
    st.sidebar.header("What is this Project about?")
    st.sidebar.text(
        "It  is a Web app that would help the user in determining whether he/she will get the right food price in a territory.")
    st.sidebar.header("What tools where used to make this?")
    st.sidebar.text("The Model was made using a dataset supplied by hamoye along with using Google Colab notebooks to train the model. We made use of XGBboost in order to make our Xtreme Gradient Boosting Model.")

    year_recorded = st.slider("Input Your Year", 0.0, 10.0)
    country_id = st.slider("Input your country Score", 0, 340)
    locality_id = st.slider("Input your locality_id Score", 0, 120)
    quantity = st.slider("What Quantity?", 0, 100)
    uni_rating = st.slider(
        "Rating of the market type you wish to buy in on a Scale 1-5", 1, 5)

    #passenger_data = {'country_id':1,'country_name':2,'locality_id':5,'market_id':10,'market_name':21,'commodity_purchase_id':8,'commodity_purchased':4,'currency_id':67,'name_of_currency':0.8,'market_type_id':2,'market_type':8,'measurement_id':12,'unit_of_goods_measurement':55,'month_recorded':4,'year_recorded':7,'price_paid':0.12,'quantity':83,'price_paid_usd':19,'measurement_unit':1}
    # a is defined user input
    #a = np.arange(1,18)
    #a = a.reshape(1,-1)
    a = np.random.randint(0, 100, size=(1, 17))
    a = a.reshape(1, -1)

    # country_id	country_name	locality_id	market_id	market_name	commodity_purchase_id	commodity_purchased	currency_id	name_of_currency	market_type_id	market_type	measurement_id	unit_of_goods_measurement	month_recorded	year_recorded	price_paid	quantity	price_paid_usd	measurement_unit
    #inputs = [[year_recorded,country_id,locality_id,quantity,uni_rating]]
    #inputs = np.array(inputs)

    if st.button('Predict'):
        

       #prediction = model.predict(df)

       #prediction_proba = model.predict_proba(df)

       # if st.butt

       # And write the output:

       # = np.array(['No','Yes'])

       # st.write(churn_labels[prediction])

       #st.subheader('Prediction Probability')

       # st.write(prediction_proba)


if __name__ == '__main__':
    main()
