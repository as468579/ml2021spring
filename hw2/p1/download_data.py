import gdown

base_url = 'http://drive.google.com/uc?'
data_id  = '1HPkcmQmFGu-3OknddKIa5dNDsR05lIQR'
data_url = f'{base_url}id={data_id}'
gdown.download(data_url, 'data.zip', quiet=False)
