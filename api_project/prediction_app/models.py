from django.db import models

# Create your models here.


class Activity(models.Model):
    # field to predict
    label                           = models.CharField(max_length=20,null=True, blank=True)
    
    #561 features
    tBodyAcc_Mean_1                 = models.FloatField()
    tBodyAcc_Mean_2                 = models.FloatField()
    tBodyAcc_Mean_3                 = models.FloatField()
    tBodyAcc_STD_1                  = models.FloatField()
    tBodyAcc_STD_2                  = models.FloatField()
    tBodyAcc_STD_3                  = models.FloatField()
    tBodyAcc_Mad_1                  = models.FloatField()
    tBodyAcc_Mad_2                  = models.FloatField()
    tBodyAcc_Mad_3                  = models.FloatField()
    tBodyAcc_Max_1                  = models.FloatField()
    tBodyAcc_Max_2                  = models.FloatField()
    tBodyAcc_Max_3                  = models.FloatField()
    tBodyAcc_Min_1                  = models.FloatField()
    tBodyAcc_Min_2                  = models.FloatField()
    tBodyAcc_Min_3                  = models.FloatField()
    tBodyAcc_SMA_1                  = models.FloatField()
    tBodyAcc_Energy_1               = models.FloatField()
    tBodyAcc_Energy_2               = models.FloatField()
    tBodyAcc_Energy_3               = models.FloatField()
    tBodyAcc_IQR_1                  = models.FloatField()
    tBodyAcc_IQR_2                  = models.FloatField()
    tBodyAcc_IQR_3                  = models.FloatField()
    tBodyAcc_ropy_1                 = models.FloatField()
    tBodyAcc_ropy_1                 = models.FloatField()
    tBodyAcc_ropy_1                 = models.FloatField()
    tBodyAcc_ARCoeff_1              = models.FloatField()
    tBodyAcc_ARCoeff_2              = models.FloatField()
    tBodyAcc_ARCoeff_3              = models.FloatField()
    tBodyAcc_ARCoeff_4              = models.FloatField()
    tBodyAcc_ARCoeff_5              = models.FloatField()
    tBodyAcc_ARCoeff_6              = models.FloatField()
    tBodyAcc_ARCoeff_7              = models.FloatField()
    tBodyAcc_ARCoeff_8              = models.FloatField()
    tBodyAcc_ARCoeff_9              = models.FloatField()
    tBodyAcc_ARCoeff_10             = models.FloatField()
    tBodyAcc_ARCoeff_11             = models.FloatField()
    tBodyAcc_ARCoeff_12             = models.FloatField()
    tBodyAcc_Correlation_1          = models.FloatField()
    tBodyAcc_Correlation_2          = models.FloatField()
    tBodyAcc_Correlation_3          = models.FloatField()
    tGravityAcc_Mean_1              = models.FloatField()
    tGravityAcc_Mean_2              = models.FloatField()
    tGravityAcc_Mean_3              = models.FloatField()
    tGravityAcc_STD_1               = models.FloatField()
    tGravityAcc_STD_2               = models.FloatField()
    tGravityAcc_STD_3               = models.FloatField()
    tGravityAcc_Mad_1               = models.FloatField()
    tGravityAcc_Mad_2               = models.FloatField()
    tGravityAcc_Mad_3               = models.FloatField()
    tGravityAcc_Max_1               = models.FloatField()
    tGravityAcc_Max_2               = models.FloatField()
    tGravityAcc_Max_3               = models.FloatField()
    tGravityAcc_Min_1               = models.FloatField()
    tGravityAcc_Min_2               = models.FloatField()
    tGravityAcc_Min_3               = models.FloatField()
    tGravityAcc_SMA_1               = models.FloatField()
    tGravityAcc_Energy_1            = models.FloatField()
    tGravityAcc_Energy_2            = models.FloatField()
    tGravityAcc_Energy_3            = models.FloatField()
    tGravityAcc_IQR_1               = models.FloatField()
    tGravityAcc_IQR_2               = models.FloatField()
    tGravityAcc_IQR_3               = models.FloatField()
    tGravityAcc_ropy_1              = models.FloatField()
    tGravityAcc_ropy_1              = models.FloatField()
    tGravityAcc_ropy_1              = models.FloatField()
    tGravityAcc_ARCoeff_1           = models.FloatField()
    tGravityAcc_ARCoeff_2           = models.FloatField()
    tGravityAcc_ARCoeff_3           = models.FloatField()
    tGravityAcc_ARCoeff_4           = models.FloatField()
    tGravityAcc_ARCoeff_5           = models.FloatField()
    tGravityAcc_ARCoeff_6           = models.FloatField()
    tGravityAcc_ARCoeff_7           = models.FloatField()
    tGravityAcc_ARCoeff_8           = models.FloatField()
    tGravityAcc_ARCoeff_9           = models.FloatField()
    tGravityAcc_ARCoeff_10          = models.FloatField()
    tGravityAcc_ARCoeff_11          = models.FloatField()
    tGravityAcc_ARCoeff_12          = models.FloatField()
    tGravityAcc_Correlation_1       = models.FloatField()
    tGravityAcc_Correlation_2       = models.FloatField()
    tGravityAcc_Correlation_3       = models.FloatField()
    tBodyAccJerk_Mean_1             = models.FloatField()
    tBodyAccJerk_Mean_2             = models.FloatField()
    tBodyAccJerk_Mean_3             = models.FloatField()
    tBodyAccJerk_STD_1              = models.FloatField()
    tBodyAccJerk_STD_2              = models.FloatField()
    tBodyAccJerk_STD_3              = models.FloatField()
    tBodyAccJerk_Mad_1              = models.FloatField()
    tBodyAccJerk_Mad_2              = models.FloatField()
    tBodyAccJerk_Mad_3              = models.FloatField()
    tBodyAccJerk_Max_1              = models.FloatField()
    tBodyAccJerk_Max_2              = models.FloatField()
    tBodyAccJerk_Max_3              = models.FloatField()
    tBodyAccJerk_Min_1              = models.FloatField()
    tBodyAccJerk_Min_2              = models.FloatField()
    tBodyAccJerk_Min_3              = models.FloatField()
    tBodyAccJerk_SMA_1              = models.FloatField()
    tBodyAccJerk_Energy_1           = models.FloatField()
    tBodyAccJerk_Energy_2           = models.FloatField()
    tBodyAccJerk_Energy_3           = models.FloatField()
    tBodyAccJerk_IQR_1              = models.FloatField()
    tBodyAccJerk_IQR_2              = models.FloatField()
    tBodyAccJerk_IQR_3              = models.FloatField()
    tBodyAccJerk_ropy_1             = models.FloatField()
    tBodyAccJerk_ropy_1             = models.FloatField()
    tBodyAccJerk_ropy_1             = models.FloatField()
    tBodyAccJerk_ARCoeff_1          = models.FloatField()
    tBodyAccJerk_ARCoeff_2          = models.FloatField()
    tBodyAccJerk_ARCoeff_3          = models.FloatField()
    tBodyAccJerk_ARCoeff_4          = models.FloatField()
    tBodyAccJerk_ARCoeff_5          = models.FloatField()
    tBodyAccJerk_ARCoeff_6          = models.FloatField()
    tBodyAccJerk_ARCoeff_7          = models.FloatField()
    tBodyAccJerk_ARCoeff_8          = models.FloatField()
    tBodyAccJerk_ARCoeff_9          = models.FloatField()
    tBodyAccJerk_ARCoeff_10         = models.FloatField()
    tBodyAccJerk_ARCoeff_11         = models.FloatField()
    tBodyAccJerk_ARCoeff_12         = models.FloatField()
    tBodyAccJerk_Correlation_1      = models.FloatField()
    tBodyAccJerk_Correlation_2      = models.FloatField()
    tBodyAccJerk_Correlation_3      = models.FloatField()
    tBodyGyro_Mean_1                = models.FloatField()
    tBodyGyro_Mean_2                = models.FloatField()
    tBodyGyro_Mean_3                = models.FloatField()
    tBodyGyro_STD_1                 = models.FloatField()
    tBodyGyro_STD_2                 = models.FloatField()
    tBodyGyro_STD_3                 = models.FloatField()
    tBodyGyro_Mad_1                 = models.FloatField()
    tBodyGyro_Mad_2                 = models.FloatField()
    tBodyGyro_Mad_3                 = models.FloatField()
    tBodyGyro_Max_1                 = models.FloatField()
    tBodyGyro_Max_2                 = models.FloatField()
    tBodyGyro_Max_3                 = models.FloatField()
    tBodyGyro_Min_1                 = models.FloatField()
    tBodyGyro_Min_2                 = models.FloatField()
    tBodyGyro_Min_3                 = models.FloatField()
    tBodyGyro_SMA_1                 = models.FloatField()
    tBodyGyro_Energy_1              = models.FloatField()
    tBodyGyro_Energy_2              = models.FloatField()
    tBodyGyro_Energy_3              = models.FloatField()
    tBodyGyro_IQR_1                 = models.FloatField()
    tBodyGyro_IQR_2                 = models.FloatField()
    tBodyGyro_IQR_3                 = models.FloatField()
    tBodyGyro_ropy_1                = models.FloatField()
    tBodyGyro_ropy_1                = models.FloatField()
    tBodyGyro_ropy_1                = models.FloatField()
    tBodyGyro_ARCoeff_1             = models.FloatField()
    tBodyGyro_ARCoeff_2             = models.FloatField()
    tBodyGyro_ARCoeff_3             = models.FloatField()
    tBodyGyro_ARCoeff_4             = models.FloatField()
    tBodyGyro_ARCoeff_5             = models.FloatField()
    tBodyGyro_ARCoeff_6             = models.FloatField()
    tBodyGyro_ARCoeff_7             = models.FloatField()
    tBodyGyro_ARCoeff_8             = models.FloatField()
    tBodyGyro_ARCoeff_9             = models.FloatField()
    tBodyGyro_ARCoeff_10            = models.FloatField()
    tBodyGyro_ARCoeff_11            = models.FloatField()
    tBodyGyro_ARCoeff_12            = models.FloatField()
    tBodyGyro_Correlation_1         = models.FloatField()
    tBodyGyro_Correlation_2         = models.FloatField()
    tBodyGyro_Correlation_3         = models.FloatField()
    tBodyGyroJerk_Mean_1            = models.FloatField()
    tBodyGyroJerk_Mean_2            = models.FloatField()
    tBodyGyroJerk_Mean_3            = models.FloatField()
    tBodyGyroJerk_STD_1             = models.FloatField()
    tBodyGyroJerk_STD_2             = models.FloatField()
    tBodyGyroJerk_STD_3             = models.FloatField()
    tBodyGyroJerk_Mad_1             = models.FloatField()
    tBodyGyroJerk_Mad_2             = models.FloatField()
    tBodyGyroJerk_Mad_3             = models.FloatField()
    tBodyGyroJerk_Max_1             = models.FloatField()
    tBodyGyroJerk_Max_2             = models.FloatField()
    tBodyGyroJerk_Max_3             = models.FloatField()
    tBodyGyroJerk_Min_1             = models.FloatField()
    tBodyGyroJerk_Min_2             = models.FloatField()
    tBodyGyroJerk_Min_3             = models.FloatField()
    tBodyGyroJerk_SMA_1             = models.FloatField()
    tBodyGyroJerk_Energy_1          = models.FloatField()
    tBodyGyroJerk_Energy_2          = models.FloatField()
    tBodyGyroJerk_Energy_3          = models.FloatField()
    tBodyGyroJerk_IQR_1             = models.FloatField()
    tBodyGyroJerk_IQR_2             = models.FloatField()
    tBodyGyroJerk_IQR_3             = models.FloatField()
    tBodyGyroJerk_ropy_1            = models.FloatField()
    tBodyGyroJerk_ropy_1            = models.FloatField()
    tBodyGyroJerk_ropy_1            = models.FloatField()
    tBodyGyroJerk_ARCoeff_1         = models.FloatField()
    tBodyGyroJerk_ARCoeff_2         = models.FloatField()
    tBodyGyroJerk_ARCoeff_3         = models.FloatField()
    tBodyGyroJerk_ARCoeff_4         = models.FloatField()
    tBodyGyroJerk_ARCoeff_5         = models.FloatField()
    tBodyGyroJerk_ARCoeff_6         = models.FloatField()
    tBodyGyroJerk_ARCoeff_7         = models.FloatField()
    tBodyGyroJerk_ARCoeff_8         = models.FloatField()
    tBodyGyroJerk_ARCoeff_9         = models.FloatField()
    tBodyGyroJerk_ARCoeff_10        = models.FloatField()
    tBodyGyroJerk_ARCoeff_11        = models.FloatField()
    tBodyGyroJerk_ARCoeff_12        = models.FloatField()
    tBodyGyroJerk_Correlation_1     = models.FloatField()
    tBodyGyroJerk_Correlation_2     = models.FloatField()
    tBodyGyroJerk_Correlation_3     = models.FloatField()
    tBodyAccMag_Mean_1              = models.FloatField()
    tBodyAccMag_STD_1               = models.FloatField()
    tBodyAccMag_Mad_1               = models.FloatField()
    tBodyAccMag_Max_1               = models.FloatField()
    tBodyAccMag_Min_1               = models.FloatField()
    tBodyAccMag_SMA_1               = models.FloatField()
    tBodyAccMag_Energy_1            = models.FloatField()
    tBodyAccMag_IQR_1               = models.FloatField()
    tBodyAccMag_ropy_1              = models.FloatField()
    tBodyAccMag_ARCoeff_1           = models.FloatField()
    tBodyAccMag_ARCoeff_2           = models.FloatField()
    tBodyAccMag_ARCoeff_3           = models.FloatField()
    tBodyAccMag_ARCoeff_4           = models.FloatField()
    tGravityAccMag_Mean_1           = models.FloatField()
    tGravityAccMag_STD_1            = models.FloatField()
    tGravityAccMag_Mad_1            = models.FloatField()
    tGravityAccMag_Max_1            = models.FloatField()
    tGravityAccMag_Min_1            = models.FloatField()
    tGravityAccMag_SMA_1            = models.FloatField()
    tGravityAccMag_Energy_1         = models.FloatField()
    tGravityAccMag_IQR_1            = models.FloatField()
    tGravityAccMag_ropy_1           = models.FloatField()
    tGravityAccMag_ARCoeff_1        = models.FloatField()
    tGravityAccMag_ARCoeff_2        = models.FloatField()
    tGravityAccMag_ARCoeff_3        = models.FloatField()
    tGravityAccMag_ARCoeff_4        = models.FloatField()
    tBodyAccJerkMag_Mean_1          = models.FloatField()
    tBodyAccJerkMag_STD_1           = models.FloatField()
    tBodyAccJerkMag_Mad_1           = models.FloatField()
    tBodyAccJerkMag_Max_1           = models.FloatField()
    tBodyAccJerkMag_Min_1           = models.FloatField()
    tBodyAccJerkMag_SMA_1           = models.FloatField()
    tBodyAccJerkMag_Energy_1        = models.FloatField()
    tBodyAccJerkMag_IQR_1           = models.FloatField()
    tBodyAccJerkMag_ropy_1          = models.FloatField()
    tBodyAccJerkMag_ARCoeff_1       = models.FloatField()
    tBodyAccJerkMag_ARCoeff_2       = models.FloatField()
    tBodyAccJerkMag_ARCoeff_3       = models.FloatField()
    tBodyAccJerkMag_ARCoeff_4       = models.FloatField()
    tBodyGyroMag_Mean_1             = models.FloatField()
    tBodyGyroMag_STD_1              = models.FloatField()
    tBodyGyroMag_Mad_1              = models.FloatField()
    tBodyGyroMag_Max_1              = models.FloatField()
    tBodyGyroMag_Min_1              = models.FloatField()
    tBodyGyroMag_SMA_1              = models.FloatField()
    tBodyGyroMag_Energy_1           = models.FloatField()
    tBodyGyroMag_IQR_1              = models.FloatField()
    tBodyGyroMag_ropy_1             = models.FloatField()
    tBodyGyroMag_ARCoeff_1          = models.FloatField()
    tBodyGyroMag_ARCoeff_2          = models.FloatField()
    tBodyGyroMag_ARCoeff_3          = models.FloatField()
    tBodyGyroMag_ARCoeff_4          = models.FloatField()
    tBodyGyroJerkMag_Mean_1         = models.FloatField()
    tBodyGyroJerkMag_STD_1          = models.FloatField()
    tBodyGyroJerkMag_Mad_1          = models.FloatField()
    tBodyGyroJerkMag_Max_1          = models.FloatField()
    tBodyGyroJerkMag_Min_1          = models.FloatField()
    tBodyGyroJerkMag_SMA_1          = models.FloatField()
    tBodyGyroJerkMag_Energy_1       = models.FloatField()
    tBodyGyroJerkMag_IQR_1          = models.FloatField()
    tBodyGyroJerkMag_ropy_1         = models.FloatField()
    tBodyGyroJerkMag_ARCoeff_1      = models.FloatField()
    tBodyGyroJerkMag_ARCoeff_2      = models.FloatField()
    tBodyGyroJerkMag_ARCoeff_3      = models.FloatField()
    tBodyGyroJerkMag_ARCoeff_4      = models.FloatField()
    fBodyAcc_Mean_1                 = models.FloatField()
    fBodyAcc_Mean_2                 = models.FloatField()
    fBodyAcc_Mean_3                 = models.FloatField()
    fBodyAcc_STD_1                  = models.FloatField()
    fBodyAcc_STD_2                  = models.FloatField()
    fBodyAcc_STD_3                  = models.FloatField()
    fBodyAcc_Mad_1                  = models.FloatField()
    fBodyAcc_Mad_2                  = models.FloatField()
    fBodyAcc_Mad_3                  = models.FloatField()
    fBodyAcc_Max_1                  = models.FloatField()
    fBodyAcc_Max_2                  = models.FloatField()
    fBodyAcc_Max_3                  = models.FloatField()
    fBodyAcc_Min_1                  = models.FloatField()
    fBodyAcc_Min_2                  = models.FloatField()
    fBodyAcc_Min_3                  = models.FloatField()
    fBodyAcc_SMA_1                  = models.FloatField()
    fBodyAcc_Energy_1               = models.FloatField()
    fBodyAcc_Energy_2               = models.FloatField()
    fBodyAcc_Energy_3               = models.FloatField()
    fBodyAcc_IQR_1                  = models.FloatField()
    fBodyAcc_IQR_2                  = models.FloatField()
    fBodyAcc_IQR_3                  = models.FloatField()
    fBodyAcc_ropy_1                 = models.FloatField()
    fBodyAcc_ropy_1                 = models.FloatField()
    fBodyAcc_ropy_1                 = models.FloatField()
    fBodyAcc_MaxInds_1              = models.FloatField()
    fBodyAcc_MaxInds_2              = models.FloatField()
    fBodyAcc_MaxInds_3              = models.FloatField()
    fBodyAcc_MeanFreq_1             = models.FloatField()
    fBodyAcc_MeanFreq_2             = models.FloatField()
    fBodyAcc_MeanFreq_3             = models.FloatField()
    fBodyAcc_Skewness_1             = models.FloatField()
    fBodyAcc_Kurtosis_1             = models.FloatField()
    fBodyAcc_Skewness_1             = models.FloatField()
    fBodyAcc_Kurtosis_1             = models.FloatField()
    fBodyAcc_Skewness_1             = models.FloatField()
    fBodyAcc_Kurtosis_1             = models.FloatField()
    fBodyAcc_BandsEnergyOld_1       = models.FloatField()
    fBodyAcc_BandsEnergyOld_2       = models.FloatField()
    fBodyAcc_BandsEnergyOld_3       = models.FloatField()
    fBodyAcc_BandsEnergyOld_4       = models.FloatField()
    fBodyAcc_BandsEnergyOld_5       = models.FloatField()
    fBodyAcc_BandsEnergyOld_6       = models.FloatField()
    fBodyAcc_BandsEnergyOld_7       = models.FloatField()
    fBodyAcc_BandsEnergyOld_8       = models.FloatField()
    fBodyAcc_BandsEnergyOld_9       = models.FloatField()
    fBodyAcc_BandsEnergyOld_10      = models.FloatField()
    fBodyAcc_BandsEnergyOld_11      = models.FloatField()
    fBodyAcc_BandsEnergyOld_12      = models.FloatField()
    fBodyAcc_BandsEnergyOld_13      = models.FloatField()
    fBodyAcc_BandsEnergyOld_14      = models.FloatField()
    fBodyAcc_BandsEnergyOld_15      = models.FloatField()
    fBodyAcc_BandsEnergyOld_16      = models.FloatField()
    fBodyAcc_BandsEnergyOld_17      = models.FloatField()
    fBodyAcc_BandsEnergyOld_18      = models.FloatField()
    fBodyAcc_BandsEnergyOld_19      = models.FloatField()
    fBodyAcc_BandsEnergyOld_20      = models.FloatField()
    fBodyAcc_BandsEnergyOld_21      = models.FloatField()
    fBodyAcc_BandsEnergyOld_22      = models.FloatField()
    fBodyAcc_BandsEnergyOld_23      = models.FloatField()
    fBodyAcc_BandsEnergyOld_24      = models.FloatField()
    fBodyAcc_BandsEnergyOld_25      = models.FloatField()
    fBodyAcc_BandsEnergyOld_26      = models.FloatField()
    fBodyAcc_BandsEnergyOld_27      = models.FloatField()
    fBodyAcc_BandsEnergyOld_28      = models.FloatField()
    fBodyAcc_BandsEnergyOld_29      = models.FloatField()
    fBodyAcc_BandsEnergyOld_30      = models.FloatField()
    fBodyAcc_BandsEnergyOld_31      = models.FloatField()
    fBodyAcc_BandsEnergyOld_32      = models.FloatField()
    fBodyAcc_BandsEnergyOld_33      = models.FloatField()
    fBodyAcc_BandsEnergyOld_34      = models.FloatField()
    fBodyAcc_BandsEnergyOld_35      = models.FloatField()
    fBodyAcc_BandsEnergyOld_36      = models.FloatField()
    fBodyAcc_BandsEnergyOld_37      = models.FloatField()
    fBodyAcc_BandsEnergyOld_38      = models.FloatField()
    fBodyAcc_BandsEnergyOld_39      = models.FloatField()
    fBodyAcc_BandsEnergyOld_40      = models.FloatField()
    fBodyAcc_BandsEnergyOld_41      = models.FloatField()
    fBodyAcc_BandsEnergyOld_42      = models.FloatField()
    fBodyAccJerk_Mean_1             = models.FloatField()
    fBodyAccJerk_Mean_2             = models.FloatField()
    fBodyAccJerk_Mean_3             = models.FloatField()
    fBodyAccJerk_STD_1              = models.FloatField()
    fBodyAccJerk_STD_2              = models.FloatField()
    fBodyAccJerk_STD_3              = models.FloatField()
    fBodyAccJerk_Mad_1              = models.FloatField()
    fBodyAccJerk_Mad_2              = models.FloatField()
    fBodyAccJerk_Mad_3              = models.FloatField()
    fBodyAccJerk_Max_1              = models.FloatField()
    fBodyAccJerk_Max_2              = models.FloatField()
    fBodyAccJerk_Max_3              = models.FloatField()
    fBodyAccJerk_Min_1              = models.FloatField()
    fBodyAccJerk_Min_2              = models.FloatField()
    fBodyAccJerk_Min_3              = models.FloatField()
    fBodyAccJerk_SMA_1              = models.FloatField()
    fBodyAccJerk_Energy_1           = models.FloatField()
    fBodyAccJerk_Energy_2           = models.FloatField()
    fBodyAccJerk_Energy_3           = models.FloatField()
    fBodyAccJerk_IQR_1              = models.FloatField()
    fBodyAccJerk_IQR_2              = models.FloatField()
    fBodyAccJerk_IQR_3              = models.FloatField()
    fBodyAccJerk_ropy_1             = models.FloatField()
    fBodyAccJerk_ropy_1             = models.FloatField()
    fBodyAccJerk_ropy_1             = models.FloatField()
    fBodyAccJerk_MaxInds_1          = models.FloatField()
    fBodyAccJerk_MaxInds_2          = models.FloatField()
    fBodyAccJerk_MaxInds_3          = models.FloatField()
    fBodyAccJerk_MeanFreq_1         = models.FloatField()
    fBodyAccJerk_MeanFreq_2         = models.FloatField()
    fBodyAccJerk_MeanFreq_3         = models.FloatField()
    fBodyAccJerk_Skewness_1         = models.FloatField()
    fBodyAccJerk_Kurtosis_1         = models.FloatField()
    fBodyAccJerk_Skewness_1         = models.FloatField()
    fBodyAccJerk_Kurtosis_1         = models.FloatField()
    fBodyAccJerk_Skewness_1         = models.FloatField()
    fBodyAccJerk_Kurtosis_1         = models.FloatField()
    fBodyAccJerk_BandsEnergyOld_1   = models.FloatField()
    fBodyAccJerk_BandsEnergyOld_2   = models.FloatField()
    fBodyAccJerk_BandsEnergyOld_3   = models.FloatField()
    fBodyAccJerk_BandsEnergyOld_4   = models.FloatField()
    fBodyAccJerk_BandsEnergyOld_5   = models.FloatField()
    fBodyAccJerk_BandsEnergyOld_6   = models.FloatField()
    fBodyAccJerk_BandsEnergyOld_7   = models.FloatField()
    fBodyAccJerk_BandsEnergyOld_8   = models.FloatField()
    fBodyAccJerk_BandsEnergyOld_9   = models.FloatField()
    fBodyAccJerk_BandsEnergyOld_10  = models.FloatField()
    fBodyAccJerk_BandsEnergyOld_11  = models.FloatField()
    fBodyAccJerk_BandsEnergyOld_12  = models.FloatField()
    fBodyAccJerk_BandsEnergyOld_13  = models.FloatField()
    fBodyAccJerk_BandsEnergyOld_14  = models.FloatField()
    fBodyAccJerk_BandsEnergyOld_15  = models.FloatField()
    fBodyAccJerk_BandsEnergyOld_16  = models.FloatField()
    fBodyAccJerk_BandsEnergyOld_17  = models.FloatField()
    fBodyAccJerk_BandsEnergyOld_18  = models.FloatField()
    fBodyAccJerk_BandsEnergyOld_19  = models.FloatField()
    fBodyAccJerk_BandsEnergyOld_20  = models.FloatField()
    fBodyAccJerk_BandsEnergyOld_21  = models.FloatField()
    fBodyAccJerk_BandsEnergyOld_22  = models.FloatField()
    fBodyAccJerk_BandsEnergyOld_23  = models.FloatField()
    fBodyAccJerk_BandsEnergyOld_24  = models.FloatField()
    fBodyAccJerk_BandsEnergyOld_25  = models.FloatField()
    fBodyAccJerk_BandsEnergyOld_26  = models.FloatField()
    fBodyAccJerk_BandsEnergyOld_27  = models.FloatField()
    fBodyAccJerk_BandsEnergyOld_28  = models.FloatField()
    fBodyAccJerk_BandsEnergyOld_29  = models.FloatField()
    fBodyAccJerk_BandsEnergyOld_30  = models.FloatField()
    fBodyAccJerk_BandsEnergyOld_31  = models.FloatField()
    fBodyAccJerk_BandsEnergyOld_32  = models.FloatField()
    fBodyAccJerk_BandsEnergyOld_33  = models.FloatField()
    fBodyAccJerk_BandsEnergyOld_34  = models.FloatField()
    fBodyAccJerk_BandsEnergyOld_35  = models.FloatField()
    fBodyAccJerk_BandsEnergyOld_36  = models.FloatField()
    fBodyAccJerk_BandsEnergyOld_37  = models.FloatField()
    fBodyAccJerk_BandsEnergyOld_38  = models.FloatField()
    fBodyAccJerk_BandsEnergyOld_39  = models.FloatField()
    fBodyAccJerk_BandsEnergyOld_40  = models.FloatField()
    fBodyAccJerk_BandsEnergyOld_41  = models.FloatField()
    fBodyAccJerk_BandsEnergyOld_42  = models.FloatField()
    fBodyGyro_Mean_1                = models.FloatField()
    fBodyGyro_Mean_2                = models.FloatField()
    fBodyGyro_Mean_3                = models.FloatField()
    fBodyGyro_STD_1                 = models.FloatField()
    fBodyGyro_STD_2                 = models.FloatField()
    fBodyGyro_STD_3                 = models.FloatField()
    fBodyGyro_Mad_1                 = models.FloatField()
    fBodyGyro_Mad_2                 = models.FloatField()
    fBodyGyro_Mad_3                 = models.FloatField()
    fBodyGyro_Max_1                 = models.FloatField()
    fBodyGyro_Max_2                 = models.FloatField()
    fBodyGyro_Max_3                 = models.FloatField()
    fBodyGyro_Min_1                 = models.FloatField()
    fBodyGyro_Min_2                 = models.FloatField()
    fBodyGyro_Min_3                 = models.FloatField()
    fBodyGyro_SMA_1                 = models.FloatField()
    fBodyGyro_Energy_1              = models.FloatField()
    fBodyGyro_Energy_2              = models.FloatField()
    fBodyGyro_Energy_3              = models.FloatField()
    fBodyGyro_IQR_1                 = models.FloatField()
    fBodyGyro_IQR_2                 = models.FloatField()
    fBodyGyro_IQR_3                 = models.FloatField()
    fBodyGyro_ropy_1                = models.FloatField()
    fBodyGyro_ropy_1                = models.FloatField()
    fBodyGyro_ropy_1                = models.FloatField()
    fBodyGyro_MaxInds_1             = models.FloatField()
    fBodyGyro_MaxInds_2             = models.FloatField()
    fBodyGyro_MaxInds_3             = models.FloatField()
    fBodyGyro_MeanFreq_1            = models.FloatField()
    fBodyGyro_MeanFreq_2            = models.FloatField()
    fBodyGyro_MeanFreq_3            = models.FloatField()
    fBodyGyro_Skewness_1            = models.FloatField()
    fBodyGyro_Kurtosis_1            = models.FloatField()
    fBodyGyro_Skewness_1            = models.FloatField()
    fBodyGyro_Kurtosis_1            = models.FloatField()
    fBodyGyro_Skewness_1            = models.FloatField()
    fBodyGyro_Kurtosis_1            = models.FloatField()
    fBodyGyro_BandsEnergyOld_1      = models.FloatField()
    fBodyGyro_BandsEnergyOld_2      = models.FloatField()
    fBodyGyro_BandsEnergyOld_3      = models.FloatField()
    fBodyGyro_BandsEnergyOld_4      = models.FloatField()
    fBodyGyro_BandsEnergyOld_5      = models.FloatField()
    fBodyGyro_BandsEnergyOld_6      = models.FloatField()
    fBodyGyro_BandsEnergyOld_7      = models.FloatField()
    fBodyGyro_BandsEnergyOld_8      = models.FloatField()
    fBodyGyro_BandsEnergyOld_9      = models.FloatField()
    fBodyGyro_BandsEnergyOld_10     = models.FloatField()
    fBodyGyro_BandsEnergyOld_11     = models.FloatField()
    fBodyGyro_BandsEnergyOld_12     = models.FloatField()
    fBodyGyro_BandsEnergyOld_13     = models.FloatField()
    fBodyGyro_BandsEnergyOld_14     = models.FloatField()
    fBodyGyro_BandsEnergyOld_15     = models.FloatField()
    fBodyGyro_BandsEnergyOld_16     = models.FloatField()
    fBodyGyro_BandsEnergyOld_17     = models.FloatField()
    fBodyGyro_BandsEnergyOld_18     = models.FloatField()
    fBodyGyro_BandsEnergyOld_19     = models.FloatField()
    fBodyGyro_BandsEnergyOld_20     = models.FloatField()
    fBodyGyro_BandsEnergyOld_21     = models.FloatField()
    fBodyGyro_BandsEnergyOld_22     = models.FloatField()
    fBodyGyro_BandsEnergyOld_23     = models.FloatField()
    fBodyGyro_BandsEnergyOld_24     = models.FloatField()
    fBodyGyro_BandsEnergyOld_25     = models.FloatField()
    fBodyGyro_BandsEnergyOld_26     = models.FloatField()
    fBodyGyro_BandsEnergyOld_27     = models.FloatField()
    fBodyGyro_BandsEnergyOld_28     = models.FloatField()
    fBodyGyro_BandsEnergyOld_29     = models.FloatField()
    fBodyGyro_BandsEnergyOld_30     = models.FloatField()
    fBodyGyro_BandsEnergyOld_31     = models.FloatField()
    fBodyGyro_BandsEnergyOld_32     = models.FloatField()
    fBodyGyro_BandsEnergyOld_33     = models.FloatField()
    fBodyGyro_BandsEnergyOld_34     = models.FloatField()
    fBodyGyro_BandsEnergyOld_35     = models.FloatField()
    fBodyGyro_BandsEnergyOld_36     = models.FloatField()
    fBodyGyro_BandsEnergyOld_37     = models.FloatField()
    fBodyGyro_BandsEnergyOld_38     = models.FloatField()
    fBodyGyro_BandsEnergyOld_39     = models.FloatField()
    fBodyGyro_BandsEnergyOld_40     = models.FloatField()
    fBodyGyro_BandsEnergyOld_41     = models.FloatField()
    fBodyGyro_BandsEnergyOld_42     = models.FloatField()
    fBodyAccMag_Mean_1              = models.FloatField()
    fBodyAccMag_STD_1               = models.FloatField()
    fBodyAccMag_Mad_1               = models.FloatField()
    fBodyAccMag_Max_1               = models.FloatField()
    fBodyAccMag_Min_1               = models.FloatField()
    fBodyAccMag_SMA_1               = models.FloatField()
    fBodyAccMag_Energy_1            = models.FloatField()
    fBodyAccMag_IQR_1               = models.FloatField()
    fBodyAccMag_ropy_1              = models.FloatField()
    fBodyAccMag_MaxInds_1           = models.FloatField()
    fBodyAccMag_MeanFreq_1          = models.FloatField()
    fBodyAccMag_Skewness_1          = models.FloatField()
    fBodyAccMag_Kurtosis_1          = models.FloatField()
    fBodyAccJerkMag_Mean_1          = models.FloatField()
    fBodyAccJerkMag_STD_1           = models.FloatField()
    fBodyAccJerkMag_Mad_1           = models.FloatField()
    fBodyAccJerkMag_Max_1           = models.FloatField()
    fBodyAccJerkMag_Min_1           = models.FloatField()
    fBodyAccJerkMag_SMA_1           = models.FloatField()
    fBodyAccJerkMag_Energy_1        = models.FloatField()
    fBodyAccJerkMag_IQR_1           = models.FloatField()
    fBodyAccJerkMag_ropy_1          = models.FloatField()
    fBodyAccJerkMag_MaxInds_1       = models.FloatField()
    fBodyAccJerkMag_MeanFreq_1      = models.FloatField()
    fBodyAccJerkMag_Skewness_1      = models.FloatField()
    fBodyAccJerkMag_Kurtosis_1      = models.FloatField()
    fBodyGyroMag_Mean_1             = models.FloatField()
    fBodyGyroMag_STD_1              = models.FloatField()
    fBodyGyroMag_Mad_1              = models.FloatField()
    fBodyGyroMag_Max_1              = models.FloatField()
    fBodyGyroMag_Min_1              = models.FloatField()
    fBodyGyroMag_SMA_1              = models.FloatField()
    fBodyGyroMag_Energy_1           = models.FloatField()
    fBodyGyroMag_IQR_1              = models.FloatField()
    fBodyGyroMag_ropy_1             = models.FloatField()
    fBodyGyroMag_MaxInds_1          = models.FloatField()
    fBodyGyroMag_MeanFreq_1         = models.FloatField()
    fBodyGyroMag_Skewness_1         = models.FloatField()
    fBodyGyroMag_Kurtosis_1         = models.FloatField()
    fBodyGyroJerkMag_Mean_1         = models.FloatField()
    fBodyGyroJerkMag_STD_1          = models.FloatField()
    fBodyGyroJerkMag_Mad_1          = models.FloatField()
    fBodyGyroJerkMag_Max_1          = models.FloatField()
    fBodyGyroJerkMag_Min_1          = models.FloatField()
    fBodyGyroJerkMag_SMA_1          = models.FloatField()
    fBodyGyroJerkMag_Energy_1       = models.FloatField()
    fBodyGyroJerkMag_IQR_1          = models.FloatField()
    fBodyGyroJerkMag_ropy_1         = models.FloatField()
    fBodyGyroJerkMag_MaxInds_1      = models.FloatField()
    fBodyGyroJerkMag_MeanFreq_1     = models.FloatField()
    fBodyGyroJerkMag_Skewness_1     = models.FloatField()
    fBodyGyroJerkMag_Kurtosis_1     = models.FloatField()
    tBodyAcc_AngleWRTGravity_1      = models.FloatField()
    tBodyAccJerk_AngleWRTGravity_1  = models.FloatField()
    tBodyGyro_AngleWRTGravity_1     = models.FloatField()
    tBodyGyroJerk_AngleWRTGravity_1 = models.FloatField()
    tXAxisAcc_AngleWRTGravity_1     = models.FloatField()
    tYAxisAcc_AngleWRTGravity_1     = models.FloatField()
    tZAxisAcc_AngleWRTGravity_1     = models.FloatField()