import gdown

tr_path = 'covid.train.csv' # path to training data
te_path = 'covid.test.csv'  # path to testing  data

tr_id = '19CCyCgJrUxtvgZF53vnctJiOJ23T5mqF'
te_id = '1CE240jLm2npU-tdz81-oVKEF3T2yfT1O'

base_url = 'http://drive.google.com/uc?'
tr_url = f'{base_url}id={tr_id}'
te_url = f'{base_url}id={te_id}'

tr_output = tr_path
te_output = te_path
gdown.download(tr_url, tr_output, quiet=False)
gdown.download(te_url, te_output, quiet=False)
