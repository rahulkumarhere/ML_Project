from django.db import models

# Create your models here.
class Churn:
    Churn = models.IntegerField(default=0)
    Tenure = models.IntegerField(default=0)
    PreferredLoginDevice = models.IntegerField(default=0)
    CityTier = models.IntegerField(default=0)
    WarehouseToHome = models.IntegerField(default=0)
    PreferredPaymentMode = models.IntegerField(default=0)
    Gender = models.IntegerField(default=0)
    HourSpendOnApp = models.IntegerField(default=0)
    NumberOfDeviceRegistered = models.IntegerField(default=0)
    PreferedOrderCat = models.IntegerField(default=0)
    SatisfactionScore = models.IntegerField(default=0)
    MaritalStatus = models.IntegerField(default=0)
    NumberOfAddress = models.IntegerField(default=0)
    Complain = models.IntegerField(default=0)
    OrderAmountHikeFromlastYear = models.IntegerField(default=0)
    OrderCount = models.IntegerField(default=0)
    DaySinceLastOrder = models.IntegerField(default=0)
    CashbackAmount = models.IntegerField(default=0)