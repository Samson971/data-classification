# %%
import pandas as pd
import sqlite3
from sqlite3 import Error


def create_connection(db_file):

    connection = None
    try:
        connection = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return connection

# %%


def insert_activity(connection, activity):
    sql = '''INSERT INTO prediction_app_activity(
        tBodyAcc_Mean_1,tBodyAcc_Mean_2,tBodyAcc_Mean_3,tBodyAcc_STD_1,tBodyAcc_STD_2,tBodyAcc_STD_3,tBodyAcc_Mad_1,tBodyAcc_Mad_2,tBodyAcc_Mad_3,tBodyAcc_Max_1,tBodyAcc_Max_2,tBodyAcc_Max_3,tBodyAcc_Min_1,tBodyAcc_Min_2,tBodyAcc_Min_3,tBodyAcc_SMA_1,tBodyAcc_Energy_1,tBodyAcc_Energy_2,tBodyAcc_Energy_3,tBodyAcc_IQR_1,tBodyAcc_IQR_2,tBodyAcc_IQR_3,tBodyAcc_ropy_1,tBodyAcc_ropy_1,tBodyAcc_ropy_1,tBodyAcc_ARCoeff_1,tBodyAcc_ARCoeff_2,tBodyAcc_ARCoeff_3,tBodyAcc_ARCoeff_4,tBodyAcc_ARCoeff_5,tBodyAcc_ARCoeff_6,tBodyAcc_ARCoeff_7,tBodyAcc_ARCoeff_8,tBodyAcc_ARCoeff_9,tBodyAcc_ARCoeff_10,tBodyAcc_ARCoeff_11,tBodyAcc_ARCoeff_12,tBodyAcc_Correlation_1,tBodyAcc_Correlation_2,tBodyAcc_Correlation_3,tGravityAcc_Mean_1,tGravityAcc_Mean_2,tGravityAcc_Mean_3,tGravityAcc_STD_1,tGravityAcc_STD_2,tGravityAcc_STD_3,tGravityAcc_Mad_1,tGravityAcc_Mad_2,tGravityAcc_Mad_3,tGravityAcc_Max_1,tGravityAcc_Max_2,tGravityAcc_Max_3,tGravityAcc_Min_1,tGravityAcc_Min_2,tGravityAcc_Min_3,tGravityAcc_SMA_1,tGravityAcc_Energy_1,tGravityAcc_Energy_2,tGravityAcc_Energy_3,tGravityAcc_IQR_1,tGravityAcc_IQR_2,tGravityAcc_IQR_3,tGravityAcc_ropy_1,tGravityAcc_ropy_1,tGravityAcc_ropy_1,tGravityAcc_ARCoeff_1,tGravityAcc_ARCoeff_2,tGravityAcc_ARCoeff_3,tGravityAcc_ARCoeff_4,tGravityAcc_ARCoeff_5,tGravityAcc_ARCoeff_6,tGravityAcc_ARCoeff_7,tGravityAcc_ARCoeff_8,tGravityAcc_ARCoeff_9,tGravityAcc_ARCoeff_10,tGravityAcc_ARCoeff_11,tGravityAcc_ARCoeff_12,tGravityAcc_Correlation_1,tGravityAcc_Correlation_2,tGravityAcc_Correlation_3,tBodyAccJerk_Mean_1,tBodyAccJerk_Mean_2,tBodyAccJerk_Mean_3,tBodyAccJerk_STD_1,tBodyAccJerk_STD_2,tBodyAccJerk_STD_3,tBodyAccJerk_Mad_1,tBodyAccJerk_Mad_2,tBodyAccJerk_Mad_3,tBodyAccJerk_Max_1,tBodyAccJerk_Max_2,tBodyAccJerk_Max_3,tBodyAccJerk_Min_1,tBodyAccJerk_Min_2,tBodyAccJerk_Min_3,tBodyAccJerk_SMA_1,tBodyAccJerk_Energy_1,tBodyAccJerk_Energy_2,tBodyAccJerk_Energy_3,tBodyAccJerk_IQR_1,tBodyAccJerk_IQR_2,tBodyAccJerk_IQR_3,tBodyAccJerk_ropy_1,tBodyAccJerk_ropy_1,tBodyAccJerk_ropy_1,tBodyAccJerk_ARCoeff_1,tBodyAccJerk_ARCoeff_2,tBodyAccJerk_ARCoeff_3,tBodyAccJerk_ARCoeff_4,tBodyAccJerk_ARCoeff_5,tBodyAccJerk_ARCoeff_6,tBodyAccJerk_ARCoeff_7,tBodyAccJerk_ARCoeff_8,tBodyAccJerk_ARCoeff_9,tBodyAccJerk_ARCoeff_10,tBodyAccJerk_ARCoeff_11,tBodyAccJerk_ARCoeff_12,tBodyAccJerk_Correlation_1,tBodyAccJerk_Correlation_2,tBodyAccJerk_Correlation_3,tBodyGyro_Mean_1,tBodyGyro_Mean_2,tBodyGyro_Mean_3,tBodyGyro_STD_1,tBodyGyro_STD_2,tBodyGyro_STD_3,tBodyGyro_Mad_1,tBodyGyro_Mad_2,tBodyGyro_Mad_3,tBodyGyro_Max_1,tBodyGyro_Max_2,tBodyGyro_Max_3,tBodyGyro_Min_1,tBodyGyro_Min_2,tBodyGyro_Min_3,tBodyGyro_SMA_1,tBodyGyro_Energy_1,tBodyGyro_Energy_2,tBodyGyro_Energy_3,tBodyGyro_IQR_1,tBodyGyro_IQR_2,tBodyGyro_IQR_3,tBodyGyro_ropy_1,tBodyGyro_ropy_1,tBodyGyro_ropy_1,tBodyGyro_ARCoeff_1,tBodyGyro_ARCoeff_2,tBodyGyro_ARCoeff_3,tBodyGyro_ARCoeff_4,tBodyGyro_ARCoeff_5,tBodyGyro_ARCoeff_6,tBodyGyro_ARCoeff_7,tBodyGyro_ARCoeff_8,tBodyGyro_ARCoeff_9,tBodyGyro_ARCoeff_10,tBodyGyro_ARCoeff_11,tBodyGyro_ARCoeff_12,tBodyGyro_Correlation_1,tBodyGyro_Correlation_2,tBodyGyro_Correlation_3,tBodyGyroJerk_Mean_1,tBodyGyroJerk_Mean_2,tBodyGyroJerk_Mean_3,tBodyGyroJerk_STD_1,tBodyGyroJerk_STD_2,tBodyGyroJerk_STD_3,tBodyGyroJerk_Mad_1,tBodyGyroJerk_Mad_2,tBodyGyroJerk_Mad_3,tBodyGyroJerk_Max_1,tBodyGyroJerk_Max_2,tBodyGyroJerk_Max_3,tBodyGyroJerk_Min_1,tBodyGyroJerk_Min_2,tBodyGyroJerk_Min_3,tBodyGyroJerk_SMA_1,tBodyGyroJerk_Energy_1,tBodyGyroJerk_Energy_2,tBodyGyroJerk_Energy_3,tBodyGyroJerk_IQR_1,tBodyGyroJerk_IQR_2,tBodyGyroJerk_IQR_3,tBodyGyroJerk_ropy_1,tBodyGyroJerk_ropy_1,tBodyGyroJerk_ropy_1,tBodyGyroJerk_ARCoeff_1,tBodyGyroJerk_ARCoeff_2,tBodyGyroJerk_ARCoeff_3,tBodyGyroJerk_ARCoeff_4,tBodyGyroJerk_ARCoeff_5,tBodyGyroJerk_ARCoeff_6,tBodyGyroJerk_ARCoeff_7,tBodyGyroJerk_ARCoeff_8,tBodyGyroJerk_ARCoeff_9,tBodyGyroJerk_ARCoeff_10,tBodyGyroJerk_ARCoeff_11,tBodyGyroJerk_ARCoeff_12,tBodyGyroJerk_Correlation_1,tBodyGyroJerk_Correlation_2,tBodyGyroJerk_Correlation_3,tBodyAccMag_Mean_1,tBodyAccMag_STD_1,tBodyAccMag_Mad_1,tBodyAccMag_Max_1,tBodyAccMag_Min_1,tBodyAccMag_SMA_1,tBodyAccMag_Energy_1,tBodyAccMag_IQR_1,tBodyAccMag_ropy_1,tBodyAccMag_ARCoeff_1,tBodyAccMag_ARCoeff_2,tBodyAccMag_ARCoeff_3,tBodyAccMag_ARCoeff_4,tGravityAccMag_Mean_1,tGravityAccMag_STD_1,tGravityAccMag_Mad_1,tGravityAccMag_Max_1,tGravityAccMag_Min_1,tGravityAccMag_SMA_1,tGravityAccMag_Energy_1,tGravityAccMag_IQR_1,tGravityAccMag_ropy_1,tGravityAccMag_ARCoeff_1,tGravityAccMag_ARCoeff_2,tGravityAccMag_ARCoeff_3,tGravityAccMag_ARCoeff_4,tBodyAccJerkMag_Mean_1,tBodyAccJerkMag_STD_1,tBodyAccJerkMag_Mad_1,tBodyAccJerkMag_Max_1,tBodyAccJerkMag_Min_1,tBodyAccJerkMag_SMA_1,tBodyAccJerkMag_Energy_1,tBodyAccJerkMag_IQR_1,tBodyAccJerkMag_ropy_1,tBodyAccJerkMag_ARCoeff_1,tBodyAccJerkMag_ARCoeff_2,tBodyAccJerkMag_ARCoeff_3,tBodyAccJerkMag_ARCoeff_4,tBodyGyroMag_Mean_1,tBodyGyroMag_STD_1,tBodyGyroMag_Mad_1,tBodyGyroMag_Max_1,tBodyGyroMag_Min_1,tBodyGyroMag_SMA_1,tBodyGyroMag_Energy_1,tBodyGyroMag_IQR_1,tBodyGyroMag_ropy_1,tBodyGyroMag_ARCoeff_1,tBodyGyroMag_ARCoeff_2,tBodyGyroMag_ARCoeff_3,tBodyGyroMag_ARCoeff_4,tBodyGyroJerkMag_Mean_1,tBodyGyroJerkMag_STD_1,tBodyGyroJerkMag_Mad_1,tBodyGyroJerkMag_Max_1,tBodyGyroJerkMag_Min_1,tBodyGyroJerkMag_SMA_1,tBodyGyroJerkMag_Energy_1,tBodyGyroJerkMag_IQR_1,tBodyGyroJerkMag_ropy_1,tBodyGyroJerkMag_ARCoeff_1,tBodyGyroJerkMag_ARCoeff_2,tBodyGyroJerkMag_ARCoeff_3,tBodyGyroJerkMag_ARCoeff_4,fBodyAcc_Mean_1,fBodyAcc_Mean_2,fBodyAcc_Mean_3,fBodyAcc_STD_1,fBodyAcc_STD_2,fBodyAcc_STD_3,fBodyAcc_Mad_1,fBodyAcc_Mad_2,fBodyAcc_Mad_3,fBodyAcc_Max_1,fBodyAcc_Max_2,fBodyAcc_Max_3,fBodyAcc_Min_1,fBodyAcc_Min_2,fBodyAcc_Min_3,fBodyAcc_SMA_1,fBodyAcc_Energy_1,fBodyAcc_Energy_2,fBodyAcc_Energy_3,fBodyAcc_IQR_1,fBodyAcc_IQR_2,fBodyAcc_IQR_3,fBodyAcc_ropy_1,fBodyAcc_ropy_1,fBodyAcc_ropy_1,fBodyAcc_MaxInds_1,fBodyAcc_MaxInds_2,fBodyAcc_MaxInds_3,fBodyAcc_MeanFreq_1,fBodyAcc_MeanFreq_2,fBodyAcc_MeanFreq_3,fBodyAcc_Skewness_1,fBodyAcc_Kurtosis_1,fBodyAcc_Skewness_1,fBodyAcc_Kurtosis_1,fBodyAcc_Skewness_1,fBodyAcc_Kurtosis_1,fBodyAcc_BandsEnergyOld_1,fBodyAcc_BandsEnergyOld_2,fBodyAcc_BandsEnergyOld_3,fBodyAcc_BandsEnergyOld_4,fBodyAcc_BandsEnergyOld_5,fBodyAcc_BandsEnergyOld_6,fBodyAcc_BandsEnergyOld_7,fBodyAcc_BandsEnergyOld_8,fBodyAcc_BandsEnergyOld_9,fBodyAcc_BandsEnergyOld_10,fBodyAcc_BandsEnergyOld_11,fBodyAcc_BandsEnergyOld_12,fBodyAcc_BandsEnergyOld_13,fBodyAcc_BandsEnergyOld_14,fBodyAcc_BandsEnergyOld_15,fBodyAcc_BandsEnergyOld_16,fBodyAcc_BandsEnergyOld_17,fBodyAcc_BandsEnergyOld_18,fBodyAcc_BandsEnergyOld_19,fBodyAcc_BandsEnergyOld_20,fBodyAcc_BandsEnergyOld_21,fBodyAcc_BandsEnergyOld_22,fBodyAcc_BandsEnergyOld_23,fBodyAcc_BandsEnergyOld_24,fBodyAcc_BandsEnergyOld_25,fBodyAcc_BandsEnergyOld_26,fBodyAcc_BandsEnergyOld_27,fBodyAcc_BandsEnergyOld_28,fBodyAcc_BandsEnergyOld_29,fBodyAcc_BandsEnergyOld_30,fBodyAcc_BandsEnergyOld_31,fBodyAcc_BandsEnergyOld_32,fBodyAcc_BandsEnergyOld_33,fBodyAcc_BandsEnergyOld_34,fBodyAcc_BandsEnergyOld_35,fBodyAcc_BandsEnergyOld_36,fBodyAcc_BandsEnergyOld_37,fBodyAcc_BandsEnergyOld_38,fBodyAcc_BandsEnergyOld_39,fBodyAcc_BandsEnergyOld_40,fBodyAcc_BandsEnergyOld_41,fBodyAcc_BandsEnergyOld_42,fBodyAccJerk_Mean_1,fBodyAccJerk_Mean_2,fBodyAccJerk_Mean_3,fBodyAccJerk_STD_1,fBodyAccJerk_STD_2,fBodyAccJerk_STD_3,fBodyAccJerk_Mad_1,fBodyAccJerk_Mad_2,fBodyAccJerk_Mad_3,fBodyAccJerk_Max_1,fBodyAccJerk_Max_2,fBodyAccJerk_Max_3,fBodyAccJerk_Min_1,fBodyAccJerk_Min_2,fBodyAccJerk_Min_3,fBodyAccJerk_SMA_1,fBodyAccJerk_Energy_1,fBodyAccJerk_Energy_2,fBodyAccJerk_Energy_3,fBodyAccJerk_IQR_1,fBodyAccJerk_IQR_2,fBodyAccJerk_IQR_3,fBodyAccJerk_ropy_1,fBodyAccJerk_ropy_1,fBodyAccJerk_ropy_1,fBodyAccJerk_MaxInds_1,fBodyAccJerk_MaxInds_2,fBodyAccJerk_MaxInds_3,fBodyAccJerk_MeanFreq_1,fBodyAccJerk_MeanFreq_2,fBodyAccJerk_MeanFreq_3,fBodyAccJerk_Skewness_1,fBodyAccJerk_Kurtosis_1,fBodyAccJerk_Skewness_1,fBodyAccJerk_Kurtosis_1,fBodyAccJerk_Skewness_1,fBodyAccJerk_Kurtosis_1,fBodyAccJerk_BandsEnergyOld_1,fBodyAccJerk_BandsEnergyOld_2,fBodyAccJerk_BandsEnergyOld_3,fBodyAccJerk_BandsEnergyOld_4,fBodyAccJerk_BandsEnergyOld_5,fBodyAccJerk_BandsEnergyOld_6,fBodyAccJerk_BandsEnergyOld_7,fBodyAccJerk_BandsEnergyOld_8,fBodyAccJerk_BandsEnergyOld_9,fBodyAccJerk_BandsEnergyOld_10,fBodyAccJerk_BandsEnergyOld_11,fBodyAccJerk_BandsEnergyOld_12,fBodyAccJerk_BandsEnergyOld_13,fBodyAccJerk_BandsEnergyOld_14,fBodyAccJerk_BandsEnergyOld_15,fBodyAccJerk_BandsEnergyOld_16,fBodyAccJerk_BandsEnergyOld_17,fBodyAccJerk_BandsEnergyOld_18,fBodyAccJerk_BandsEnergyOld_19,fBodyAccJerk_BandsEnergyOld_20,fBodyAccJerk_BandsEnergyOld_21,fBodyAccJerk_BandsEnergyOld_22,fBodyAccJerk_BandsEnergyOld_23,fBodyAccJerk_BandsEnergyOld_24,fBodyAccJerk_BandsEnergyOld_25,fBodyAccJerk_BandsEnergyOld_26,fBodyAccJerk_BandsEnergyOld_27,fBodyAccJerk_BandsEnergyOld_28,fBodyAccJerk_BandsEnergyOld_29,fBodyAccJerk_BandsEnergyOld_30,fBodyAccJerk_BandsEnergyOld_31,fBodyAccJerk_BandsEnergyOld_32,fBodyAccJerk_BandsEnergyOld_33,fBodyAccJerk_BandsEnergyOld_34,fBodyAccJerk_BandsEnergyOld_35,fBodyAccJerk_BandsEnergyOld_36,fBodyAccJerk_BandsEnergyOld_37,fBodyAccJerk_BandsEnergyOld_38,fBodyAccJerk_BandsEnergyOld_39,fBodyAccJerk_BandsEnergyOld_40,fBodyAccJerk_BandsEnergyOld_41,fBodyAccJerk_BandsEnergyOld_42,fBodyGyro_Mean_1,fBodyGyro_Mean_2,fBodyGyro_Mean_3,fBodyGyro_STD_1,fBodyGyro_STD_2,fBodyGyro_STD_3,fBodyGyro_Mad_1,fBodyGyro_Mad_2,fBodyGyro_Mad_3,fBodyGyro_Max_1,fBodyGyro_Max_2,fBodyGyro_Max_3,fBodyGyro_Min_1,fBodyGyro_Min_2,fBodyGyro_Min_3,fBodyGyro_SMA_1,fBodyGyro_Energy_1,fBodyGyro_Energy_2,fBodyGyro_Energy_3,fBodyGyro_IQR_1,fBodyGyro_IQR_2,fBodyGyro_IQR_3,fBodyGyro_ropy_1,fBodyGyro_ropy_1,fBodyGyro_ropy_1,fBodyGyro_MaxInds_1,fBodyGyro_MaxInds_2,fBodyGyro_MaxInds_3,fBodyGyro_MeanFreq_1,fBodyGyro_MeanFreq_2,fBodyGyro_MeanFreq_3,fBodyGyro_Skewness_1,fBodyGyro_Kurtosis_1,fBodyGyro_Skewness_1,fBodyGyro_Kurtosis_1,fBodyGyro_Skewness_1,fBodyGyro_Kurtosis_1,fBodyGyro_BandsEnergyOld_1,fBodyGyro_BandsEnergyOld_2,fBodyGyro_BandsEnergyOld_3,fBodyGyro_BandsEnergyOld_4,fBodyGyro_BandsEnergyOld_5,fBodyGyro_BandsEnergyOld_6,fBodyGyro_BandsEnergyOld_7,fBodyGyro_BandsEnergyOld_8,fBodyGyro_BandsEnergyOld_9,fBodyGyro_BandsEnergyOld_10,fBodyGyro_BandsEnergyOld_11,fBodyGyro_BandsEnergyOld_12,fBodyGyro_BandsEnergyOld_13,fBodyGyro_BandsEnergyOld_14,fBodyGyro_BandsEnergyOld_15,fBodyGyro_BandsEnergyOld_16,fBodyGyro_BandsEnergyOld_17,fBodyGyro_BandsEnergyOld_18,fBodyGyro_BandsEnergyOld_19,fBodyGyro_BandsEnergyOld_20,fBodyGyro_BandsEnergyOld_21,fBodyGyro_BandsEnergyOld_22,fBodyGyro_BandsEnergyOld_23,fBodyGyro_BandsEnergyOld_24,fBodyGyro_BandsEnergyOld_25,fBodyGyro_BandsEnergyOld_26,fBodyGyro_BandsEnergyOld_27,fBodyGyro_BandsEnergyOld_28,fBodyGyro_BandsEnergyOld_29,fBodyGyro_BandsEnergyOld_30,fBodyGyro_BandsEnergyOld_31,fBodyGyro_BandsEnergyOld_32,fBodyGyro_BandsEnergyOld_33,fBodyGyro_BandsEnergyOld_34,fBodyGyro_BandsEnergyOld_35,fBodyGyro_BandsEnergyOld_36,fBodyGyro_BandsEnergyOld_37,fBodyGyro_BandsEnergyOld_38,fBodyGyro_BandsEnergyOld_39,fBodyGyro_BandsEnergyOld_40,fBodyGyro_BandsEnergyOld_41,fBodyGyro_BandsEnergyOld_42,fBodyAccMag_Mean_1,fBodyAccMag_STD_1,fBodyAccMag_Mad_1,fBodyAccMag_Max_1,fBodyAccMag_Min_1,fBodyAccMag_SMA_1,fBodyAccMag_Energy_1,fBodyAccMag_IQR_1,fBodyAccMag_ropy_1,fBodyAccMag_MaxInds_1,fBodyAccMag_MeanFreq_1,fBodyAccMag_Skewness_1,fBodyAccMag_Kurtosis_1,fBodyAccJerkMag_Mean_1,fBodyAccJerkMag_STD_1,fBodyAccJerkMag_Mad_1,fBodyAccJerkMag_Max_1,fBodyAccJerkMag_Min_1,fBodyAccJerkMag_SMA_1,fBodyAccJerkMag_Energy_1,fBodyAccJerkMag_IQR_1,fBodyAccJerkMag_ropy_1,fBodyAccJerkMag_MaxInds_1,fBodyAccJerkMag_MeanFreq_1,fBodyAccJerkMag_Skewness_1,fBodyAccJerkMag_Kurtosis_1,fBodyGyroMag_Mean_1,fBodyGyroMag_STD_1,fBodyGyroMag_Mad_1,fBodyGyroMag_Max_1,fBodyGyroMag_Min_1,fBodyGyroMag_SMA_1,fBodyGyroMag_Energy_1,fBodyGyroMag_IQR_1,fBodyGyroMag_ropy_1,fBodyGyroMag_MaxInds_1,fBodyGyroMag_MeanFreq_1,fBodyGyroMag_Skewness_1,fBodyGyroMag_Kurtosis_1,fBodyGyroJerkMag_Mean_1,fBodyGyroJerkMag_STD_1,fBodyGyroJerkMag_Mad_1,fBodyGyroJerkMag_Max_1,fBodyGyroJerkMag_Min_1,fBodyGyroJerkMag_SMA_1,fBodyGyroJerkMag_Energy_1,fBodyGyroJerkMag_IQR_1,fBodyGyroJerkMag_ropy_1,fBodyGyroJerkMag_MaxInds_1,fBodyGyroJerkMag_MeanFreq_1,fBodyGyroJerkMag_Skewness_1,fBodyGyroJerkMag_Kurtosis_1,tBodyAcc_AngleWRTGravity_1,tBodyAccJerk_AngleWRTGravity_1,tBodyGyro_AngleWRTGravity_1,tBodyGyroJerk_AngleWRTGravity_1,tXAxisAcc_AngleWRTGravity_1,tYAxisAcc_AngleWRTGravity_1,tZAxisAcc_AngleWRTGravity_1
        )
    VALUES(
        ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
) '''
    cursor = connection.cursor()
    
    try:
        cursor.execute(sql, tuple(activity))
    except Error as e:
        print(e)
    return cursor.lastrowid


# %%


def load_data(filename):
    data = pd.read_csv(filename, header=None, delim_whitespace=True)
    return data.to_numpy()


# %%
X_train = load_data('/home/samson971/Documents/Python_For_Data/data-project/HAPT-Data/Train/X_train.txt')

# %%
connection = create_connection('/home/samson971/Documents/Python_For_Data/data-project/api_project/db.sqlite3')
# %%
for i in range(len(X_train)):
    #print(X_test[i])
    print(insert_activity(connection, X_train[i]))



# %%
def select_from_activity(connection):
    sql = 'select id,label from prediction_app_activity limit 10'
    cursor = connection.cursor()
    cursor.execute(sql)
    rows = cursor.fetchall()

    for row in rows:
        print(row)

# %%
select_from_activity(connection)

# %%
def select_count_from_activity(connection):
    sql = 'select count(*) from prediction_app_activity'
    cursor = connection.cursor()
    cursor.execute(sql)
    rows = cursor.fetchall()

    for row in rows:
        print(row)

# %%
select_count_from_activity(connection)

# %%
connection.commit()
connection.close()

# %%
print(X_train[0][0])

# %%
print(X_train[7766])


# %%
