
import gdown

base_url = 'http://drive.google.com/uc?'
data_id  = '1awF7pZ9Dz7X1jn1_QAiKN-_v56veCEKy'
data_url = f'{base_url}id={data_id}'
gdown.download(data_url, 'food-11.zip', quiet=False)
