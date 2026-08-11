import pandas as p
import numpy as n
import matplotlib.pyplot as mat
import seaborn as sea

# reading the dataset

ds = p.read_csv('The Titanic dataset.csv', header = 1)

print(ds)

# Exploring phase
''' Shape of the dataset '''
print(f' the sape of dataset {ds.shape}')

# coumns names
print("the columns names",'\n', ds.columns)

# information related the data set 

print(' the information about the dataset:','\n', ds.info())\

# Statics description related dataset
print(' the descripyive statics about the dataset:-','\n',ds.describe())
# datatypes of each column
print(' the each column datatype is :','\n',ds.dtypes)

print(' the top five rows','\n',ds.head())
print(' the bottom five rows','\n',ds.tail())

"""
**Data cleaning and organizing**

"""

# Checking for inproper data

for c in ds.columns:
  print(c,'\n', ds[c].head())

ds["fare"] = p.to_numeric(ds["fare"], errors="coerce")
ds["age"] = p.to_numeric(ds["age"], errors="coerce")

# re-checking
print('\n','checkpoint')
for c in ds.columns:
  print(c,'\n', ds[c].head())
  print(c,'\n', ds[c].tail())

# handling the NaN data

print(ds.isnull().sum())

'''
gender          1
age           258
family          2
fare            3
embarked        6

'''
# for gender 
k = ds.groupby("gender")['sn'].count()
print(k)

"""
female    465 [1]
male      835  [0]

Mode means the value that appears most often.
"""
ds["gender"] = ds["gender"].fillna(ds["gender"].mode()[0])

# Rechecking the dataset
print(ds['gender'].head())

f = ds.groupby("family")['sn'].count()
print(f)
"""
family
0.0     781
1.0     234
2.0     159
3.0      43
4.0      22
5.0      25
6.0      16
7.0       8
10.0     11

"""

ds["family"] = ds["family"].fillna(ds["family"].mode()[0])
# re-check again
print(ds.isnull().sum())
'''
sn              0
pclass          0
survived        0
Unnamed: 3      0
gender          0
age           258
family          0
fare            3
embarked        6
date            0

'''
p = ds.groupby("pclass")['fare'].count()

print(p)

"""
1    323
2    277
3    698

"""
print(ds.loc[ds["fare"].isna()])

ds["fare"]=ds["fare"].fillna(
    ds.groupby("pclass")["fare"].transform("median")
)
print("re-check")
# re-check again
print(ds.isnull().sum())

"""

sn              0
pclass          0
survived        0
Unnamed: 3      0
gender          0
age           258
family          0
fare            0
embarked        6
date            0
dtype: int64

"""
print(ds[ds["age"].isna()])
'''
        sn  pclass  survived                              Unnamed: 3 gender  age  family     fare embarked       date
2        2       3         0                   Master. Eugene Joseph   male  NaN     2.0  20.2500        S  02-Jan-90
3        3       2         0             Abbott, Mr. Rossmore Edward   male  NaN     2.0  15.0458        S  03-Jan-90
98      98       1         0                     Baumann, Mr. John D   male  NaN     0.0  25.9250        S  08-Apr-90
119    119       3         0                   Betros, Master. Seman   male  NaN     0.0   7.2292        C  29-Apr-90
140    140       3         0                       Boulos, Mr. Hanna   male  NaN     0.0   7.2250        C  20-May-90
...    ...     ...       ...                                     ...    ...  ...     ...      ...      ...        ...
1247  1247       2         0              Watson, Mr. Ennis Hastings   male  NaN     0.0   0.0000        S  31-May-93
1264  1264       2         0          Wheeler, Mr. Edwin "Frederick"   male  NaN     0.0  12.8750        S  17-Jun-93
1281  1281       2         1            Williams, Mr. Charles Eugene   male  NaN     0.0  13.0000        S  04-Jul-93
1284  1284       1         0  Williams-Lambert, Mr. Fletcher Fellows   male  NaN     0.0  35.0000        S  07-Jul-93
1289  1289       1         1                       Woolner, Mr. Hugh   male  NaN     0.0  35.5000        S  12-Jul-93

[258 rows x 10 columns]

'''


ds["Title"] = ds["Unnamed: 3"].str.extract(r' ([A-Za-z]+)\.')
ds["age"] = ds["age"].fillna(
    ds.groupby("Title")["age"].transform("median")
)

print("re-check")
# re-check again
print(ds.isnull().sum())

ds['age'] = ds["age"].ffill()
# re-check again
print(ds.isnull().sum())

e =  ds.groupby('embarked')['sn'].count()
print(e)

e_t = ds.groupby(["embarked",'date'])['sn'].count()
print(e_t)


print(ds[ds["embarked"].isna()])

print(ds.groupby(["pclass", "Title"])["embarked"].value_counts().sort_values(ascending= False))

ds["embarked"] = ds["embarked"].fillna("S")
# re-check again
print(ds.isnull().sum()) 
ds.drop(columns=['Title'], inplace= True)
# re-check again
print(ds.isnull().sum()) 
# checking for duplicates
print(ds.duplicated())
print(ds.duplicated().sum())
# removing the duplicates
ds.drop(1,inplace= True)
# recheck
print(ds.duplicated())
print(ds.duplicated().sum())

print(" Final check point")

print(ds.isnull().sum())
print(ds.duplicated().sum(

))

"Analysis phase"
print("#"*40 + '| Analysis phase |' + "#"*40)

#Survival Rate
print(" the survival rate; 0 = No , 1 = Yes")

mapping = {0:"death", 1:"survive"}
ds['survived'] = ds['survived'].map(mapping)

survive = ds['survived'].value_counts()
print(survive)


'''
 the survival rate; 0 = No , 1 = Yes
survived
death      800
survive    500

How many survived?
 500
How many died? 
 800

survival rate of one person from enitre population on ship is 38.46%
 
'''
survival_rate = (500/1300)*100
print(f" survival rate of one person from enitre population on ship is { survival_rate:.2f}%" )

mat.pie( survive, labels= survive.index)
mat.title('survival')
mat.show()

Rev_mapping = {"death":0, "survive":1}
ds['survived'] = ds['survived'].map(Rev_mapping)

# Male and Female survival rate
q = ds.groupby(["gender",'survived'])['sn'].count()

print("the survival rate; 0 = No , 1 = Yes",'\n',q)
'''
the survival rate; 0 = No , 1 = Yes 
 gender  survived
female  0           126
        1           339
male    0           674
        1           161

        
Did women survive more?
Yes, 178 more woman survive  then mans. means more then twice woman survive then mans

 survival rate of females is 72.90%

Did men survive less?       
Yes total  548 more mans die then woman. means the death rate of man is more then 4 time of womans in that accident 
survival rate of males is 19.28%
'''
data = ds.loc[ ds["survived"]==1]
mat.figure(figsize=(8,9))
sea.countplot(data = data, x = data["gender"], hue= data["gender"])
mat.xlabel("Gender")
mat.ylabel("survival_count")
mat.title("MAle-Female survival difference")
mat.show()
gender_ratio = ds.groupby("gender")["survived"].mean()
mat.pie( gender_ratio, labels= gender_ratio.index, autopct='%1.2f%%')
mat.title('Male-Female Ratio')
mat.show()

survival_rate_of_man = (161/835)*100
print(f" survival rate of males is { survival_rate_of_man:.2f}%")
survival_rate_of_woman = (339/465)*100
print(f" survival rate of females is { survival_rate_of_woman:.2f}%")


# Passenger Class Analysis

s = q = ds.groupby(["pclass",'survived'])['sn'].count()
print(s)

'''
pclass  survived
1       0           123
        1           200
2       0           159
        1           119
3       0           518
        1           181

        

 Questions:

Which class had the highest survival?
1st class have highest survival with total 200 suvivals
Did rich passengers survive more?
Yes, expensive class 1st have higher amount of survivals then 2nd and 3rd class

'''
mat.figure(figsize=(8,9))
sea.countplot(data = data, x = data["pclass"], hue= data["pclass"])
mat.xlabel("Class")
mat.ylabel("survival_count")
mat.title("class survival rate")
mat.show()
survival_rate_of_1st = (200/500)*100
print(f" survival rate of 1st class is { survival_rate_of_1st:.2f}%")
survival_rate_of_2nd = (119/500)*100
print(f" survival rate of 2nd is class { survival_rate_of_2nd:.2f}%")
survival_rate_of_3rd = (181/500)*100
print(f" survival rate of 3rd class is { survival_rate_of_3rd:.2f}%")
# Age Analysis
 
print(ds["age"].describe())
'''
count    1300.000000
mean       29.493015
std        13.185078
min         0.170000
25%        22.000000
50%        29.000000
75%        36.000000
max        80.000000

Average age = 29.49
Youngest passenger = 0.17 (~ 2 month)
Oldest passenge = 80

''' 
# Give me all rows where the passenger's age is below 12.
Koo = ds[ds['age']<12]
print("##"*90)
print(Koo)
print("##"*90)

# Fare Analysis

print(ds["fare"].describe())

"""
count    1300.000000
mean       33.428852
std        51.887992
min         0.000000
25%         7.895800
50%        14.454200
75%        31.275000
max       512.329200

Maximum - 512.32
Minimum - 0.00
Avrage - 33.42

"""

# Family Size Analysis
F_s = ds.groupby(['family','survived'])['sn'].count()
print(F_s)


"""
family  survived
0.0     0           543
        1           239
1.0     0           108
        1           126
2.0     0            69
        1            90
3.0     0            13
        1            30
4.0     0            16
        1             6
5.0     0            20
        1             5
6.0     0            12
        1             4
7.0     0             8
10.0    0            11


Did passengers with family survive more?
people with family with 1.0 and 2.0 member survive but people with huge family like 4.0,5.0,6.0, and 7.0 very few survive.
people with 10,0 does not make it alive 

"""
print('avrage survival rate:' ,"\n", ds.groupby("family")["survived"].mean())

mat.figure(figsize=(8,9))
sea.countplot(data = data, x = data["family"], hue= data["family"])
mat.xlabel("families")
mat.ylabel("survival_count")
mat.title("family surive rate ")
mat.show()

fam_survi = ds["family"].value_counts()
mat.pie( fam_survi, labels= fam_survi.index, autopct="%1.2f%%")
mat.title('family survival ratio')
mat.show()

# Embarked Port Analysis

print(ds.groupby("embarked")["survived"].mean())
'''
Avrage survival ate

C    0.561798
Q    0.357724
S    0.336264

'''

print(ds.groupby("embarked")["survived"].count())
mat.figure(figsize=(8,9))
sea.countplot(data = data, x = data["embarked"], hue= data["embarked"])
mat.xlabel("Port")
mat.ylabel("survival_count")
mat.title("embarked surive rate ")
mat.show()
# survival rate
embark = ds.groupby("embarked")["survived"].mean()

mat.bar(embark.index, embark.values)
mat.title("Survival Rate by Embarkation Port")
mat.ylabel("Survival Rate")
mat.show()
"""
survival count

embarked
C    267
Q    123
S    910

"""

print(" Correlation between all numeric clumns")
corr = ds.corr(numeric_only=True)
print(corr)


"""

| Correlation | Meaning                          |
| ----------- | -------------------------------- |
| +1          | Perfect positive relationship 📈 |
| 0           | No relationship                  |
| -1          | Perfect negative relationship 📉 |


"""

'''
Name: survived, dtype: int64
                sn    pclass  survived       age    family      fare
sn        1.000000  0.036603 -0.027466  0.013402 -0.010396 -0.028979
pclass    0.036603  1.000000 -0.308653 -0.390403  0.052424 -0.557800
survived -0.027466 -0.308653  1.000000 -0.053005  0.024295  0.242458
age       0.013402 -0.390403 -0.053005  1.000000 -0.210211  0.179441
family   -0.010396  0.052424  0.024295 -0.210211  1.000000  0.225100
fare     -0.028979 -0.557800  0.242458  0.179441  0.225100  1.000000



'''
# Gender × Class Heatmap
sea.heatmap(corr, annot= True)
mat.title("correlation")
mat.show()

heatmap_data = ds.pivot_table( 
  values= "survived",
    index="gender",
    columns="pclass",
    aggfunc="mean")
sea.heatmap(heatmap_data, annot=True, cmap="Blues")
mat.title("Gender  X Class relation")
mat.show()


# Fare vs Survival Boxplot
fare = ds.groupby("fare")["survived"].mean()
sea.boxplot( fare )
mat.title("fare vs Survival Boxplot")
mat.show()

# Fare Distribution
sea.kdeplot(ds["fare"])
mat.title("Fare Distribution")
mat.show()
# Age Distribution
sea.histplot(ds["age"], kde= True)
mat.title("Age Distribution")
mat.show()
# Age vs Survival Boxplot
Age = ds.groupby("age")["survived"].mean()
sea.boxplot( Age )
mat.title("Age vs Survival Boxplot")
mat.show()
ds.to_csv("Modify_Titanic_Dataset")