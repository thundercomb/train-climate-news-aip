import gpt_2_simple as gpt2
import shutil
import logging
from google.cloud import bigquery
from google.cloud import storage

import os
import argparse
import datetime

#
# Parse arguments
#

def parse_args():
    """Parses command-line arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--analytics-project',
        help='Project id where model will be run.',
        required=True,
    )

    parser.add_argument(
        '--training-iterations',
        help='Number of iterations to finetune the model.',
        required=True,
    )

    parser.add_argument(
        '--job-dir',
        help='Output directory for package data.',
        required=True,
    )

    return parser.parse_args()

#
# bucket interactions
#

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(f'File {source_file_name} uploaded to {destination_blob_name}.')

def rename_blob(bucket_name, blob_name, new_name):
    """Renames a blob."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)

    new_blob = bucket.rename_blob(blob, new_name)

    print(f'Blob {blob.name} has been renamed to {new_blob.name}')

def list_blobs(bucket_name, prefix):
    """Lists all the blobs in the bucket."""
    storage_client = storage.Client()

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)

    return blobs

#
# Get the Data
#

def get_data():
    client = bigquery.Client()

    query = (
        f"SELECT article FROM `{arguments.analytics_project}.clnn.news`"
    )
    query_job = client.query(
        query,
    )

    articles = ""
    for row in query_job:
        articles = articles + row.article

    textfile = open('clnn.txt', 'w')
    textfile.write(articles)
    textfile.close()

#
# Finetune the model on the new data
#

def finetune():
    model_name = "124M"
    gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/124M/

    session = gpt2.start_tf_sess()
    gpt2.finetune(session,
                  'clnn.txt',
                  model_name=model_name,
                  steps=arguments.training_iterations)   # steps is max number of training steps

    return session

#
# Archive and version the model and upload to storage
#

def archive_and_version():
    bucket_name = f"{arguments.analytics_project}-models"
    datetime_now = f"{datetime.datetime.now():%Y%m%d_%H%M%S}"
    model_name = f"clnn_news_{datetime_now}.zip.latest"
    bucket_prefix = 'clnn-news'

    print(f"Finding previous version in bucket {bucket_name} ...")
    blob_names = list_blobs(bucket_name, bucket_prefix)

    previous_model_old_name = ""
    previous_model_new_name = ""
    for blob in blob_names:
        if blob.name.find(".latest", -8) > 0:
            previous_model_old_name = blob.name
            previous_model_new_name = blob.name.replace(".latest","")
            print(f"Renaming previous version {previous_model_old_name} to {previous_model_new_name} ...")
            rename_blob(bucket_name, previous_model_old_name, previous_model_new_name)

    destination_blob_name = f"{bucket_prefix}/{model_name}"
    print(f"Uploading {model_name} to bucket {bucket_name} as {destination_blob_name} ...")
    # zip the model dir and upload to storage
    shutil.make_archive('model_archive', 'zip', 'checkpoint')
    upload_blob(bucket_name,'model_archive.zip',destination_blob_name)


arguments = parse_args()
get_data()
logging.basicConfig(level='INFO')
time_start = datetime.datetime.utcnow()
session = finetune()
time_end = datetime.datetime.utcnow()
time_trained = time_end - time_start
logging.info(f"Training time: {time_trained.total_seconds()} seconds")
archive_and_version()

# Generate some text for good measure
gpt2.generate(session)
