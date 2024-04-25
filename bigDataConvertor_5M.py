import pandas as pd

# Зчитуємо дані з CSV-файлу
df = pd.read_csv("AI_Human.csv")

# Вибираємо 3000 рядків з generated, які рівні 0.0
df_0 = df[df['generated'] == 0.0].head(3000)

# Вибираємо 2000 рядків з generated, які рівні 1.0
df_1 = df[df['generated'] == 1.0].head(2000)

# Додаємо стовпець "TEXT" зі значенням рядка до нового DataFrame
df_0['TEXT'] = df_0['text']
df_1['TEXT'] = df_1['text']

# Встановлюємо значення 0.0 у стовпці "ID" для df_0
df_0['ID'] = 0.0

# Встановлюємо значення 1.0 у стовпці "ID" для df_1
df_1['ID'] = 1.0

# Об'єднуємо обидва DataFrames
df_result = pd.concat([df_0, df_1])

# Видаляємо непотрібні стовпці
df_result.drop(columns=['text', 'generated'], inplace=True)

# Встановлюємо значення 0 у стовпцях "REFERENCE" та "SOURCE" для всіх рядків
df_result['REFERENCE'] = 0
df_result['SOURCE'] = 0

# Записуємо новий DataFrame у CSV-файл
df_result.to_csv("TextBigData.csv", index=False)
