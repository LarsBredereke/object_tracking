import os
from dotenv import dotenv_values

config = {
    **dotenv_values(".env"),  # load sensitive variables
    **os.environ,  # override loaded values with environment variables
}

key = config['LAKEFS_KEY']
secret = config['LAKEFS_SECRET']
endpoint = "http://10.101.1.140:9020/"

# Set the Repo
# likely always tsd1 if your'e working within csl and ease, refers to: http://10.101.1.140:9020/repositories/tsd1/
repo = "tsd1"

# this where we get the raw data from
download_branch = 'dev-raw'

# where our results are uploaded to
upload_branch = 'processed'

# you can find the commit-ids here: http://10.101.1.140:9020/repositories/tsd1/commits?ref=publish 
# just click on the "copy id to clipboard button"
# commit = "395bb9a2efb019d88cf1da4ef9ae0f31e036274c9486059f9792d8288d85cba8"

TIMEOUT = 30
# TIMEOUT = 60 * 2

STORAGE_OPTION = dict(key=key, secret=secret, client_kwargs=dict(endpoint_url=endpoint))

""" Note that if the environment vars are not read / set correctly, boto3 will search for an ~/.aws/credentials with the following content
[default]
aws_access_key_id = <yourKey>
aws_secret_access_key = <yourSecretKey>
"""