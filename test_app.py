import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import pickle

with open('models.sav', 'rb') as file:
    model = pickle.load(file)
with open('labels.sav', 'rb') as file:
    labels = pickle.load(file)
with open('train_df.sav', 'rb') as file:
    X_train = pickle.load(file)
    
# Destination, passenger,weather , time , coupon_type , expiration, age, maritalStatus, has_children, education, income , Bar, CoffeeHouse, CarryAway, RestaurantLessThan20, Restaurant20To50, toCoupon_GEQ
# destination_lables, passanger_lables, weather_lables, time_lables, coupon_type_lables , expiration_lables, maritalStatus_lables, has_children_lables, education_lables, income_labels , Bar_lables,CoffeeHouse_labels, CarryAway_lables, RestaurantLessThan20_lables, Restaurant20To50_labels,

def make_input(Values,Labels):
    value_labelled = []
    for i in range(len(Labels)):
        if Labels[i] != '':
            # print(Labels[i],Values[i])
            k = int(Labels[i][Values[i]])
            value_labelled.append(k)
        else:
            value_labelled.append(int(Values[i]))
    input_df = pd.DataFrame([value_labelled],dtype=int, columns=['destination', 'passanger', 'weather', 'time', 'coupon_type',
       'expiration', 'age', 'maritalStatus', 'has_children', 'education',
       'income', 'Bar', 'CoffeeHouse', 'CarryAway', 'RestaurantLessThan20',
       'Restaurant20To50', 'toCoupon_GEQ'])
    return input_df

page = st.sidebar.radio("Navigation", ["Home", "Result","Best Model Description"])

if page=='Home':

    st.title("Vehicle Coupon Recommendation")
    Destination = st.selectbox("Destination", ("No Urgent Place", "Home", "Work"))

    passenger = st.selectbox("Passenger", ("Alone", "Friend(s)",  "Kid(s)", "Partner"))

    weather = st.selectbox("Weather", ("Sunny", "Rainy", "Snowy"))

    time = st.selectbox("Time", ("2PM", "10AM", "6PM", "7AM", "10PM"))

    coupon_type = st.selectbox("Coupon type", ("Restaurant(<20)", "Coffee House", "Carry out & Take away", "Bar", "Restaurant(20-50)"))

    expiration = st.selectbox("Expiration", ("1d", "2h"))

    age= st.text_input("Age(Between 1 and 71)")

    maritalStatus = st.selectbox("Marital status", ("Unmarried partner", "Single", "Married partner", "Divorced", "Widowed"))

    has_children = st.selectbox("Has children", (1, 0))

    education = st.selectbox("Education", ("Some college - no degree", "Bachelors degree", "Associates degree", "High School Graduate", "Graduate degree (Masters or Doctorate)"))

    income = st.selectbox("Income", ("$100000 or More", "$12500 - $24999",  "$25000 - $37499", "$37500 - $49999", "$50000 - $62499", "$62500 - $74999", "$75000 - $87499", "$87500 - $99999", "Less than $12500"))

    Bar = st.selectbox("Bar", ('never','less_than_1','1_to_3','4_to_8', 'greater_than_8'))

    CoffeeHouse = st.selectbox("CoffeeHouse", ('never','less_than_1','1_to_3','4_to_8', 'greater_than_8'))

    CarryAway = st.selectbox("CarryAway", ('never','less_than_1','1_to_3','4_to_8', 'greater_than_8'))

    RestaurantLessThan20 = st.selectbox("RestaurantLessThan20", ('never','less_than_1','1_to_3','4_to_8', 'greater_than_8'))

    Restaurant20To50 = st.selectbox("Restaurant20To50", ('never','less_than_1','1_to_3','4_to_8', 'greater_than_8'))

    toCoupon_GEQ = st.selectbox("ToCoupon_GEQ", (1, 0))

    Model = st.selectbox("ML MODEL", ("XGBOOST", "LOGISTIC REGRESSION", "Random Forest", "Gradient Boosting Machine","KNN","Naive Bayes","SVM"))
    
    button_style = """
    <style>
    .stButton > button {
        background-color: #4CAF50;  /* Green background color */
        color: white;               /* Text color */
        padding: 10px 20px;         /* Padding (top/bottom, left/right) */
        border: none;               /* No border */
        border-radius: 5px;         /* Rounded corners */
        cursor: pointer;            /* Pointer cursor on hover */
        transition-duration: 0.4s;  /* Smooth transition */
    }
    .stButton > button:hover {
        background-color: #45a049;  /* Darker green on hover */
    }
    </style>
    """

# Apply custom CSS styling to the button
    st.markdown(button_style, unsafe_allow_html=True)
    Button = st.button("Submit")

    Values = [Destination, passenger,weather , time , coupon_type , expiration, age, maritalStatus, has_children, education, income , Bar, CoffeeHouse, CarryAway, RestaurantLessThan20, Restaurant20To50, toCoupon_GEQ]
    model_dict={"LOGISTIC REGRESSION":0,"Random Forest":1,"Gradient Boosting Machine":2,"KNN":3,"Naive Bayes":4,"SVM":5,"XGBOOST":6}
    if Button:
        if any(not value or value == "" for value in Values):  # Check if both dropdowns are selected
            st.error("Please select options from all dropdowns before submitting.")
        else: 
            input_df = make_input(Values,labels)
            predicted_val=model[model_dict[Model]].predict(input_df)
            best_predicted_val=model[6].predict(input_df)
            if best_predicted_val==1:
                best_predicted_val="The User will accept the coupon."
            else:
                best_predicted_val="The User will probably decline the coupon."
            st.session_state.predicted_val = predicted_val
            st.session_state.best_predicted_val = best_predicted_val
if page == "Result":
    st.title("Prediction Result")

    # Retrieve and display prediction result
    if "predicted_val" in st.session_state:
        predicted_val = st.session_state.predicted_val
        best_predicted_val=st.session_state.best_predicted_val 
        if predicted_val == 1:
            st.write("According to the model chosen:")
            st.markdown(f'<p style="font-size: 20px; background-color: lightgreen; padding: 10px;">The User will accept the coupon.</p>',
            unsafe_allow_html=True)
        else:
            st.write("According to the model chosen:")
            st.markdown(f'<p style="font-size: 20px; background-color: #ffcccc; padding: 10px;">The User will probably decline the coupon.</p>',
            unsafe_allow_html=True)
        st.write(f"According to the best model (XGBoost):")
        st.markdown(f'<p style="font-size: 20px; background-color: lightgrey; padding: 10px;">{best_predicted_val}</p>',
            unsafe_allow_html=True)      

    else:
        st.write("No prediction result available. Please go to 'Home' and submit the form.")

if page=='Best Model Description':
    xgb_classifier=model[6]
    st.title("XGboost Model")
    st.markdown("""The XGBClassifier is a component of the XGBoost (Extreme Gradient Boosting) library, known for its efficient implementation of the gradient boosting algorithm. 
                Gradient boosting, an ensemble learning technique, aggregates predictions from numerous decision trees to construct a robust predictive model. XGBoost employs an objective function comprising a loss function and regularization terms, which are optimized during training.
                Additionally, it integrates L1 (LASSO) and L2 (Ridge) regularization terms into the objective function, serving to mitigate overfitting and enhance the model's ability to generalize to unseen data.""")
    st.header("Key Factors")

    st.markdown("""
        **1. Predictive Accuracy:** XGBoost is particularly effective in achieving high predictive accuracy.""")

    st.markdown("""
        **2. Regularization:** XGBoost incorporates regularization techniques like L1 (LASSO) and L2 (Ridge) regularization to prevent overfitting.""")

    st.markdown("""
        **3. Speed and Performance:** XGBoost is highly computationally efficient and designed for parallel processing. 
        It can handle large datasets and complex patterns.""")
    st.header("Model performance")
    metrics_data = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC Score"],
        "Value": ["74.54%", "75.35%", "81.02%", "78.08%", "73.66%"]}
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.index.name = "S.No"
    metrics_df.index += 1
    st.table(metrics_df)
    st.header("Feature Impotance")
    

    importances = xgb_classifier.feature_importances_
    feature_names = X_train.columns
    sorted_indices = importances.argsort()

    # feature impotance
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance - XGBoost")
    plt.barh(range(X_train.shape[1]), importances[sorted_indices], align="center", color='skyblue')
    plt.yticks(range(X_train.shape[1]), [feature_names[i] for i in sorted_indices])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    for i, v in enumerate(importances[sorted_indices]):
        plt.text(v, i, f'{v:.4f}', ha='left', va='center', color='black')
    st.pyplot(plt) 