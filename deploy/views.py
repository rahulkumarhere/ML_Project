from django.shortcuts import render
import pickle
from pathlib import Path
import os
# Create your views here.


def home(request):
    return render(request, "index.html")


def predict(Tenure, PreferredLoginDevice, CityTier, WarehouseToHome, PreferredPaymentMode, Gender,
                  HourSpendOnApp, NumberOfDeviceRegistered, PreferedOrderCat, SatisfactionScore,
                  MaritalStatus, NumberOfAddress, Complain, OrderAmountHikeFromlastYear, OrderCount,
                  DaySinceLastOrder, CashbackAmount):
    # model_path = Path('./model_data/ml_model.sav').resolve()
    # model_path = os.path.abspath("./model_data/ml_model.sav")
    model_path = os.path.join(os.path.dirname(__file__), './model_data/ml_model.sav')
    print(model_path)
    # scaler_path = Path('./model_data/scalar.sav').resolve()
    # scaler_path = os.path.abspath("./model_data/scalar.sav")
    scaler_path = os.path.join(os.path.dirname(__file__), './model_data/scalar.sav')
    model = pickle.load(open(model_path, 'rb'))
    scaled = pickle.load(open(scaler_path, 'rb'))

    prediction = model.predict(scaled.transform([[Tenure, PreferredLoginDevice, CityTier, WarehouseToHome, PreferredPaymentMode, Gender,
                  HourSpendOnApp, NumberOfDeviceRegistered, PreferedOrderCat, SatisfactionScore,
                  MaritalStatus, NumberOfAddress, Complain, OrderAmountHikeFromlastYear, OrderCount,
                  DaySinceLastOrder, CashbackAmount]]))
    if prediction == 1:
        return "Yes"
    elif prediction == 0:
        return "No"
    else:
        return "error"


def result(request):
    Tenure = request.POST.get('Tenure')
    PreferredLoginDevice = request.POST.get('PreferredLoginDevice')
    CityTier = request.POST.get('CityTier')
    WarehouseToHome = request.POST.get('WarehouseToHome')
    PreferredPaymentMode = request.POST.get('PreferredPaymentMode')
    Gender = request.POST.get('Gender')
    HourSpendOnApp = request.POST.get('HourSpendOnApp')
    NumberOfDeviceRegistered = request.POST.get('NumberOfDeviceRegistered')
    PreferedOrderCat = request.POST.get('PreferedOrderCat')
    SatisfactionScore = request.POST.get('SatisfactionScore')
    MaritalStatus = request.POST.get('MaritalStatus')
    NumberOfAddress = request.POST.get('NumberOfAddress')
    Complain = request.POST.get('Complain')
    OrderAmountHikeFromlastYear = request.POST.get('OrderAmountHikeFromlastYear')
    OrderCount = request.POST.get('OrderCount')
    DaySinceLastOrder = request.POST.get('DaySinceLastOrder')
    CashbackAmount = request.POST.get('CashbackAmount')

    churn_result = predict(Tenure, PreferredLoginDevice, CityTier, WarehouseToHome, PreferredPaymentMode, Gender,
                  HourSpendOnApp, NumberOfDeviceRegistered, PreferedOrderCat, SatisfactionScore,
                  MaritalStatus, NumberOfAddress, Complain, OrderAmountHikeFromlastYear, OrderCount,
                  DaySinceLastOrder, CashbackAmount)

    return render(request, 'predict.html', {'result': churn_result})
