import pandas as pd
 
testData = {
    'Name': ["League Of Legends", "Valorant", "Minecraft", "Valheim"],
    'Price_USD': [20.99,18.99,19.99,20.99],
    'Total_Reviews': [5000,6001,5602,7523]
}

df = pd.DataFrame(testData)
print(df)

