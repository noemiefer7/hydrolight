import pandas as pd
import csv

for i in range (1,63):
    with open('hydropt/data/water_mason016.csv','a',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        c=710+i*5
        c=str(c)
        writer.writerow([c,'0.827','0.000317557'])

