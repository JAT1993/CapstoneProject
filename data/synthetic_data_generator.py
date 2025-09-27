import pandas as pd
import numpy as np
from datetime import datetime, timedelta
np.random.seed(2025)
num_records=5200
start_time=datetime(2024,6,1,8,0)
machine_ids=['M001','M002','M003','M004']
product_ids=['P1001','P1002','P1003']
production_stages=['assembly','testing','packaging']
operator_ids=[f'O{str(i).zfill(3)}' for i in range (120,140)]
shifts=[1,2,3]

records=[]
for i in range(num_records):
    timestamp=start_time+timedelta(minutes=5*i)
    machine=np.random.choice(machine_ids)
    temp=np.random.normal(75,5)
    vibration=np.random.normal(0.004,0.001)
    product=np.random.choice(product_ids)
    defect=np.random.choice([0,1],p=[0.95,0.05])
    stage=np.random.choice(production_stages)
    operator=np.random.choice(operator_ids)
    shift=np.random.choice(shifts)
    output_qty=np.random.poisson(10)

    records.append([timestamp,machine,temp,vibration,product,defect,stage,operator,shift,output_qty])

# records
df=pd.DataFrame(records,columns=['timestamp','machine_id','sensor_1_temp','sensor_2_vibration','product_id','defect_flag','production_stage','operator_id','shift','output_qty'])
df.to_csv('manufacturing_production_data.csv',index=False)