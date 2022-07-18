from django import forms
from deploy.models import Churn


class ChurnForm(forms.ModelForm):
    class Meta:
        model = Churn
        fields = ['Tenure', 'PreferredLoginDevice', 'CityTier', 'WarehouseToHome', 'PreferredPaymentMode', 'Gender',
                  'HourSpendOnApp', 'NumberOfDeviceRegistered', 'PreferedOrderCat', 'SatisfactionScore',
                  'MaritalStatus', 'NumberOfAddress', 'Complain', 'OrderAmountHikeFromlastYear', 'OrderCount',
                  'DaySinceLastOrder', 'CashbackAmount']
