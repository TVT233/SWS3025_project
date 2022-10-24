import hdd_test
age = input("input your age: (0~100)")
sex = input("input your sex: (1:male 0:female)")
cp = input("input your chest pain type: (0:no pain, 1:typical angina 2:atypical angina 3:non-anginal pain 4:asymptomatic)")
BO = input("input your blood oxygen saturation: (%)")
ECG = input("input your ECG result: (0:normal 1:abnormal)")
HR = input("input your heart rate: (bps)")
y = hdd_test.HDD.bpredict(20,1,1,0,1,1)
print(y)
if y == 0:
    print("Assessed that your heart is working normally")
else:
    print("Assessed that your heart is not working properly")