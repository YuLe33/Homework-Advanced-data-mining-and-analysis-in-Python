import pandas as pd
import numpy as np
import re
from datetime import datetime

def prepare_data(fname):
    # Read the DataFrame from Excel file
    df = pd.read_excel(fname)   
    
    # Function to clean the 'price' column
    def clean_price(value):
        # Remove non-numeric characters using regular expression
        cleaned_value = re.sub(r'[^\d.]', '', str(value))
        if cleaned_value:
            # Convert the cleaned value to float
            return float(cleaned_value)
        else:
            return None
    
    # Apply the clean_price function to the 'price' column
    df['price'] = df['price'].apply(clean_price)
    
    # Drop rows with null values in the 'price' column
    df.dropna(subset=['price'], inplace=True)
    
   # Clean the 'Area' column
    df['Area'] = df['Area'].apply(lambda x: float(re.search(r'\d+', str(x)).group()) if isinstance(x, str) and re.search(r'\d+', str(x)) else x)
    df['Area'] = pd.to_numeric(df['Area'], errors='coerce')
    
    # Clean the 'description' column
    df['description'] = df['description '].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)))
    df = df.drop(columns=['description '], axis=1)
    # Clean all others columns
    df['Street'] = df['Street'].str.replace('[^\w\s]', '')
    
    df['room_number'] = df['room_number'].apply(lambda x: float(re.search(r'\d+', str(x)).group()) if isinstance(x, str) and re.search(r'\d+', str(x)) else x)
    df['room_number'] = df['room_number'].apply(lambda x: None if isinstance(x, str) and '-' in x else x)

    df['city_area'] = df['city_area'].str.replace('[^\w\s]', '')
    df['number_in_street'] = df['number_in_street'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)) if isinstance(x, str) else x)
    df['floor'] = df['floor_out_of'].apply(lambda x: 0 if isinstance(x, str) and "קומת קרקע" in x else (-1 if isinstance(x, str) and "קומת מרתף" in x else (int(re.search(r'\d+', str(x)).group()) if isinstance(x, str) and re.search(r'\d+', str(x)) else None)))
    df['total_floors'] = df['floor_out_of'].str.extract(r'תוך\s*(\d+)').astype(float)
    df['total_floors'] = df['total_floors'].fillna(df['floor_out_of'].apply(lambda x: 0 if isinstance(x, str) and "קומת קרקע" in x else (-1 if isinstance(x, str) and "קומת מרתף" in x else None)))
    df['publishedDays '] = df['publishedDays '].apply(lambda x: None if isinstance(x, str) else x)
    
    #create boolians columns
    bool_fields = ['hasElevator ', 'hasParking ', 'hasBars ', 'hasStorage ', 'hasAirCondition ', 'hasBalcony ', 'hasMamad ', 'handicapFriendly ']
    df[bool_fields] = df[bool_fields].applymap(lambda x: 1 if isinstance(x, str) and ("כן" in x or "יש" in x or "yes" in x) else (0 if isinstance(x, str) and ("לא" in x or "אין" in x or "no" in x) else x))
    df[bool_fields] = df[bool_fields].apply(pd.to_numeric, errors='coerce')

    # Function to calculate the 'entrance_date' column
    def calculate_entrance_date(entrance):
        if type(entrance) == type('s'):
            entrance = entrance.strip()
            if pd.isnull(entrance) or entrance == 'לא צויין':
                return 'not_defined'
            elif entrance == 'מיידי':
                return 'less_than_6_months'
            elif entrance == 'גמיש':
                return 'flexible'
        else:
            date = entrance.date()
            today = datetime.now().date()
            if date > today:
                difference = (date - today).days
                if difference < 180:
                    return 'less_than_6_months'
                elif 180 <= difference <= 365:
                    return 'months_6_12'
                else:
                    return 'above_year'
            else:
                return 'not_defined'

    # Apply the calculate_entrance_date function to the 'entranceDate' column
    df['entrance_date'] = df['entranceDate '].apply(calculate_entrance_date)
    
    return df

# Clean the DataFrame using the clean_dataframe function
df = prepare_data('output_all_students_Train_v10.xlsx')
