"""
This is the autoML interface.

[Modified to interact with pytorch and to use it as API]

Replace THISISYOURSERVICEACCOUNT.json in line 40 with your service account credentials.
"""



import os
import pickle
# from PIL import Image
# import numpy as np

from google.cloud import automl, storage

import google
from argparse import ArgumentParser
# from multiprocessing import Pool

import sys
from time import sleep
import datetime

import torch
import torchvision


def imagenet_to_gcloud(kettle, clean_uid, bucketname, format, dryrun=False):
    os.makedirs('csv/test', exist_ok=True)
    os.makedirs('img_temp', exist_ok=True)
    # Write train/test data
    csv_file = f'csv/{clean_uid}.csv'

    dm = torch.tensor(kettle.trainset.data_mean)[:, None, None]
    ds = torch.tensor(kettle.trainset.data_std)[:, None, None]


    storage_client = storage.Client.from_service_account_json('THISISYOURSERVICEACCOUNT.json')
    bucket = storage_client.bucket(bucketname)

    def _save_image(sample, index, filename, train=True):
        input_denormalized = sample * ds + dm
        torchvision.utils.save_image(input_denormalized, filename)

    classes = kettle.trainset.classes

    def label_name(label):
        candidate = classes[label]
        return ''.join(e for e in candidate if e.isalnum())

    with open(csv_file, 'w') as train_file:
        for split, dataset in zip(['TRAIN', 'TEST'], [kettle.trainset, kettle.validset]):
            for sample, label, index in dataset:
                # source_location = dataset.samples[iter]
                # file_format = os.path.splitext(path) # we could also prevent file re-creation,
                # but then the image dimensions are possibly too distinct for poisons vs non-poisons
                filename = f'img_temp/temp_image{format}'
                gcloud_blob = f'img/{clean_uid}/{split.lower()}/{index}{format}'

                # Save local copy
                _save_image(sample, index, filename, train=(split == 'TRAIN'))


                # Copy file into cloud
                blob = bucket.blob(gcloud_blob)
                blob.upload_from_filename(filename)

                # Add reference in .csv file
                train_file.write(f'{split},gs://{bucketname}/{gcloud_blob},{label_name(label)}\n')

                if index % 10_000 == 0:
                    print(f'Progress: {index} / {len(dataset)} of split {split} uploaded.')
                # if dryrun:
                #    break
            print(f'{split} fully uploaded to gcloud bucket.')

    # Upload csv files
    blob = bucket.blob(csv_file)
    blob.upload_from_filename(csv_file)

def poisons_to_gcloud(kettle, poison_delta, clean_uid, uid, bucketname, format, dryrun=False):
    os.makedirs('csv/test', exist_ok=True)
    os.makedirs(f'img/{uid}/train', exist_ok=True)
    os.makedirs(f'img/{uid}/target', exist_ok=True)
    # Write train/test data
    csv_file = f'csv/{uid}.csv'
    target_csv_file = f'csv/test/{uid}.csv'
    imagenet_csv = f'csv/{clean_uid}.csv'

    dm = torch.tensor(kettle.trainset.data_mean)[:, None, None]
    ds = torch.tensor(kettle.trainset.data_std)[:, None, None]


    storage_client = storage.Client.from_service_account_json('silent-venture-269920-0ad84136605a.json')
    bucket = storage_client.bucket(bucketname)

    def _save_image(sample, index, filename, train=True):
        """Save input image to given location, add poison_delta if necessary."""
        lookup = kettle.poison_lookup.get(index)
        if (lookup is not None) and train:
            sample += poison_delta[lookup, :, :, :]
        input_denormalized = sample * ds + dm
        torchvision.utils.save_image(input_denormalized, filename)

    classes = kettle.trainset.classes

    def label_name(label):
        candidate = classes[label]
        return ''.join(e for e in candidate if e.isalnum()) + str(label)


    # Write TRAIN / TEST
    validation_per_class = torch.zeros(len(classes))
    with open(csv_file, 'w') as train_file, open(target_csv_file, 'w') as test_file:
        for split, dataset in zip(['TRAIN', 'TEST'], [kettle.trainset, kettle.validset]):
            for idx in range(len(dataset)):  # prevent image reading in most cases
                label, index = dataset.get_target(idx)
                lookup = kettle.poison_lookup.get(index) if split == 'TRAIN' else None

                if lookup is not None:  # Image is poisoned, this only happens in the training set
                    filename = f'img/{uid}/{split.lower()}/{index}{format}'
                    # print(filename)
                    sample = dataset[idx][0]   # image reading happens here
                    _save_image(sample, index, filename, train=True)


                    # Copy file into cloud
                    blob = bucket.blob(filename)
                    blob.upload_from_filename(filename)
                    # Add reference in .csv file
                    if validation_per_class[label] < 50:  # do waste poisons for validation
                        validation_per_class[label] += 1
                        train_file.write(f'VALIDATION,gs://{bucketname}/{filename},{label_name(label)}\n')
                    else:
                        train_file.write(f'TRAIN,gs://{bucketname}/{filename},{label_name(label)}\n')

                else:  # Image is unchanged
                    # Add reference to base file location:
                    base_idx = dataset.full_imagenet_id[idx]
                    base_location = f'gs://{bucketname}/img/{clean_uid}/{split.lower()}/{base_idx}{format}'
                    if split == 'TRAIN':
                        if validation_per_class[label] < 50:
                            validation_per_class[label] += 1
                            train_file.write(f'VALIDATION,{base_location},{label_name(label)}\n')
                        else:
                            train_file.write(f'TRAIN,{base_location},{label_name(label)}\n')
                    else:
                        train_file.write(f'TEST,{base_location},{label_name(label)}\n')

                    if split == 'TEST':  # Ronny writes the test images into the /test/uid.csv file
                        test_file.write(f'{base_location}\n')

                if index % 10_000 == 0:
                    print(f'Progress: {index} / {len(dataset)} of split {split} considered for poisoned image upload.')
                # if dryrun:
                #    break
            # Batched upload into cloud
            # subprocess.run(f'gsutil -m cp -r img/{uid}/{split} gs://{bucketname}/img/{split}',
            #               shell=True)  # , stderr=subprocess.DEVNULL)
            print(f'{split} fully uploaded to gcloud bucket {bucketname} into ids img/{uid}.')

            if validation_per_class.min() < 50:
                raise ValueError(f'Class {label_name(validation_per_class.argmin())} has only {validation_per_class.min()} validation examples.')

    # WRITE TARGETS
    with open(target_csv_file, 'a') as test_file:
        for sample, label, index in kettle.targetset:

            # Save target temprarily
            filename = f'img/{uid}/target/{index}{format}'
            _save_image(sample, index, filename, train=False)

            # Copy file into cloud
            blob = bucket.blob(filename)
            blob.upload_from_filename(filename)

            # Add reference to csv
            test_file.write(f'gs://{bucketname}/{filename}\n')

        print(f'Targets fully uploaded to gcloud bucket {bucketname} into ids img/{uid}/target.')

    # Upload csv files
    blob = bucket.blob(csv_file)
    blob.upload_from_filename(csv_file)
    blob = bucket.blob(target_csv_file)
    blob.upload_from_filename(target_csv_file)

# create empty dataset in automl
def create_dataset_automl(uid, project_id, multilabel=False, wait_for_response=True):
    display_name = uid
    print(f'new automl dataset: {display_name}')

    client = automl.AutoMlClient()

    # A resource that represents Google Cloud Platform location.
    project_location = client.location_path(project_id, "us-central1")
    print(f'Project location determined to be {project_location}.')
    # Specify the classification type
    if multilabel:
        classificationtype = automl.enums.ClassificationType.MULTILABEL
    else:
        classificationtype = automl.enums.ClassificationType.MULTICLASS
    metadata = automl.types.ImageClassificationDatasetMetadata(
        classification_type=classificationtype
    )
    print(metadata)
    dataset = automl.types.Dataset(
        display_name=display_name,
        image_classification_dataset_metadata=metadata,
    )

    # Create a dataset with the dataset metadata in the region.
    response = client.create_dataset(project_location, dataset, timeout=60)

    if wait_for_response:
        created_dataset = response.result()

        # Display the dataset information
        print("Dataset name: {}".format(created_dataset.name))
        print("Dataset id: {}".format(created_dataset.name.split("/")[-1]))
        dataset_id = created_dataset.name.split("/")[-1]

        return dataset_id, display_name
    else:
        return None, None


# upload images to automl via csv
def upload_to_automl(dataset_id, project_id, csvpath, wait_for_response=True):
    # project_id = "YOUR_PROJECT_ID"
    # dataset_id = "YOUR_DATASET_ID"
    # path = "gs://YOUR_BUCKET_ID/path/to/data.csv"

    client = automl.AutoMlClient()
    # Get the full path of the dataset.
    dataset_full_id = client.dataset_path(
        project_id, "us-central1", dataset_id
    )
    # Get the multiple Google Cloud Storage URIs
    input_uris = csvpath.split(",")
    gcs_source = automl.types.GcsSource(input_uris=input_uris)
    input_config = automl.types.InputConfig(gcs_source=gcs_source)
    # Import data from the input URI
    print("Processing import...")
    while True:
        try:
            response = client.import_data(dataset_full_id, input_config)
            break
        except google.api_core.exceptions.ResourceExhausted:
            print('concurrent import-data-to-automl quota (4) exhausted, waiting 4 sec')
            sleep(4)
        except google.api_core.exceptions.ServiceUnavailable:
            print('service unavailable error... what?? waiting 4 sec')
            sleep(4)
        except google.api_core.exceptions.DeadlineExceeded:
            print('deadline exceeded, whatever, keep trying')
            sleep(4)
        except Exception as e:
            print(f'some other random error: {e}')

    if wait_for_response:
        print("Data imported. {}".format(response.result()))
        return
    else:
        return


# train
def train(dataset_id, display_name, project_id, dryrun=False, edge=True):
    # project_id = "YOUR_PROJECT_ID"
    # dataset_id = "YOUR_DATASET_ID"
    # display_name = "your_models_display_name"

    client = automl.AutoMlClient()

    # A resource that represents Google Cloud Platform location.
    project_location = client.location_path(project_id, "us-central1")
    # Leave model unset to use the default base model provided by Google
    # train_budget_milli_node_hours: The actual train_cost will be equal or
    # less than this value.
    # https://cloud.google.com/automl/docs/reference/rpc/google.cloud.automl.v1#imageclassificationmodelmetadata
    # also useful https: // googleapis.dev / python / automl / latest / gapic / v1 / types.html
    if edge:
        metadata = automl.types.ImageClassificationModelMetadata(
            train_budget_milli_node_hours=10_000,
            model_type='mobile-high-accuracy-1'
        )
    else:
        metadata = automl.types.ImageClassificationModelMetadata(
            train_budget_milli_node_hours=24_000
        )

    model = automl.types.Model(
        display_name=display_name,
        dataset_id=dataset_id,
        image_classification_model_metadata=metadata,
    )

    # Create a model with the model metadata in the region.
    print('Training beginning: creating model')
    while True:
        try:
            response = client.create_model(project_location, model, timeout=None)
            break
        except google.api_core.exceptions.ResourceExhausted:
            print('Concurrent model training quota exhausted, waiting 15 minutes.')
            sleep(900)
        except google.api_core.exceptions.FailedPrecondition:
            print('AutoML preparation is not yet finished, waiting for 15 minutes.')
            sleep(900)
        except Exception as e:
            print(f'some other random error: {e}')
            raise  # sleep(4)

    print("Training operation name: {}".format(response.operation.name))
    print("Training started...")

    def callback(operation_future):
        # Handle result.
        result = operation_future.result()

    response.add_done_callback(callback)
    while True:
        if response.done():
            break
        else:
            sleep(300)
            print('Model still training ...')

    # Catch model:
    try:
        model_id = response.result().name.split('/')[-1]
    except google.api_core.exceptions.GoogleAPICallError:
        print('Call to operation failed ...')
        print('Looking up if the model training finished in any case ...')
        while True:
            try:
                name = client.model_path(project_id, "us-central1", model.name)
                response = client.get_model(name)
                model_id = response.result().name.split('/')[-1]
                break
            except google.api_core.exceptions.GoogleAPICallError:
                print('API call failed ... waiting 60 seconds ...')
                sleep(60)

    print(f'model_id obtained after training {model_id}')
    return model_id


# batch inference
def predict(model_id, project_id, csvpath_test, resultpath_cloud):

    # project_id = "YOUR_PROJECT_ID"
    # model_id = "YOUR_MODEL_ID"
    input_uri = csvpath_test
    output_uri = resultpath_cloud

    prediction_client = automl.PredictionServiceClient()


    # Get the full path of the model.
    model_full_id = prediction_client.model_path(
        project_id, "us-central1", model_id
    )

    gcs_source = automl.types.GcsSource(input_uris=[input_uri])

    input_config = automl.types.BatchPredictInputConfig(gcs_source=gcs_source)
    gcs_destination = automl.types.GcsDestination(output_uri_prefix=output_uri)
    output_config = automl.types.BatchPredictOutputConfig(
        gcs_destination=gcs_destination
    )

    # Print model stats
    # client = automl.AutoMlClient()
    # print("List of model evaluations:")
    # for evaluation in client.list_model_evaluations(model_full_id, ""):
    #     print("Model evaluation name: {}".format(evaluation.name))
    #     print(
    #         "Model annotation spec id: {}".format(
    #             evaluation.annotation_spec_id
    #         )
    #     )
    #     print("Create Time:")
    #     print("\tseconds: {}".format(evaluation.create_time.seconds))
    #     print("\tnanos: {}".format(evaluation.create_time.nanos / 1e9))
    #     print(
    #         "Evaluation example count: {}".format(
    #             evaluation.evaluated_example_count
    #         )
    #     )
    #     print(
    #         "Classification model evaluation metrics: {}".format(
    #             evaluation.classification_evaluation_metrics
    #         )
    #     )

    print('batch prediction beginning...')
    while True:
        try:
            response = prediction_client.batch_predict(model_full_id, input_config, output_config, params={'score_threshold': '0'})
            break
        except google.api_core.exceptions.ResourceExhausted:
            print('Concurrent batch prediction quota exhausted. This is a common error. Waiting 4 sec...')
            sleep(4)
        except google.api_core.exceptions.DeadlineExceeded:
            print('Deadline exceeded, whatever, keep trying')
            sleep(4)
        except google.api_core.exceptions.ServiceUnavailable:
            print('Service unavailable error... what??')
            sleep(4)
        except google.api_core.exceptions.NotFound:
            print(f'Model {model_full_id} not found. probably training not finished. waiting 30 sec')
            sleep(30)
        except Exception as e:
            raise  # print(f'some other random error: {e}')

    print(f"Batch prediction operation id {response.operation.name.split('/')[-1]} has started \n\n\n")
    print(f"Waiting for batch prediction operation id {response.operation.name.split('/')[-1]} to complete...")

    # def callback(operation_future):
    #     # Handle result.
    #     print("Batch Prediction results saved to Cloud Storage bucket. {}".format(operation_future.result()))
    #
    # response.add_done_callback(callback)
    while True:
        if response.done():
            break
        else:
            sleep(300)
            print('Model still predicting ...')
    print("Batch Prediction results saved to Cloud Storage bucket. {}".format(response.result()))


def automl_interface(setup, kettle, poison_delta):
    print(f'python {" ".join(sys.argv)}')
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'silent-venture-269920-0ad84136605a.json'

    uid, project_id, multilabel, format, bucketname, display_name, \
        dataset_id, model_id, ntrial, mode, base_dataset, dryrun = setup.values()
    # uid is the folder where the current experiment is stored.

    csvpath = f'gs://{bucketname}/csv/{uid}.csv'
    csvpath_test = f'gs://{bucketname}/csv/test/{uid}.csv'
    clean_uid = 'baseline' + base_dataset  # this location stores the 200gb of clean ImageNet

    if mode in ['upload', 'imagenet-upload']:
        print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
        imagenet_to_gcloud(kettle, clean_uid, bucketname, format, dryrun=dryrun)

    if mode in ['all', 'upload', 'poison-upload']:
        print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
        poisons_to_gcloud(kettle, poison_delta, clean_uid, uid, bucketname, format, dryrun=dryrun)
        dataset_id, display_name = create_dataset_automl(uid, project_id, multilabel)
        if mode == 'all':  # wait for a dataset response:
            upload_to_automl(dataset_id, project_id, csvpath, wait_for_response=True)
        else:  # terminate after giving the API call
            upload_to_automl(dataset_id, project_id, csvpath, wait_for_response=False)
        print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))


    for trial in range(ntrial):

        if mode in ['all', 'train', 'pred', 'traintest']:
            print(f'trial {trial}')

        if mode in ['all', 'train', 'traintest']:
            print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
            model_id = train(dataset_id, display_name, project_id, dryrun=dryrun)  # Run just one millinode hour if dryrun

        if mode in ['all', 'train', 'pred', 'traintest']:
            print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
            resultpath_cloud = f'gs://{bucketname}/result/{uid}/{model_id}'
            print(f'batch predicting model {uid}/{model_id}')
            predict(model_id, project_id, csvpath_test, resultpath_cloud)
            # print(f'done batch predicting {uid}/{model_id}')
            #
            # resultpath_local = f'result/{uid}/{model_id}'
            # print(f'pulling batch prediction results to {resultpath_local}')
    print(f'Training    repeatable with setup uid {uid} -dataset_id {dataset_id}  -display_name {display_name}.')
    print(f'Predictions repeatable with setup uid {uid} -model_id {model_id}')


if __name__ == '__main__':
    parser = ArgumentParser()
    # basic args
    parser.add_argument('uid', default='debug', type=str)
    parser.add_argument('-file_name', default='kettle_ImageNetResNet18.pkl', type=str)
    parser.add_argument('-project_id', default='THIS IS YOUR PROJECT ID', type=str)
    parser.add_argument('-multilabel', action='store_true')
    parser.add_argument('-format', default='png', type=str)
    # parser.add_argument('-type', default='mobile-high-accuracy-1', type=str)
    parser.add_argument('-ntrial', default=1, type=int)  # number of models to train
    # mode options: start at different points in the process
    parser.add_argument('-mode', default='upload', type=str)
    parser.add_argument('-dataset_id', default=None, type=str)
    parser.add_argument('-display_name', default=None, type=str)
    parser.add_argument('-model_id', default=None, type=str)
    parser.add_argument('-base_dataset', default='CIFAR10', type=str)
    args = parser.parse_args()

    setup = dict(uid=args.uid,
                 project_id=args.project_id,
                 multilabel=args.multilabel,
                 format=args.format,
                 bucketname=f'THIS IS YOUR BUCKET',
                 display_name=args.display_name if args.display_name is not None else args.uid,
                 dataset_id=args.dataset_id,
                 model_id=args.model_id,
                 ntrial=args.ntrial,
                 mode=args.mode,
                 base_dataset=args.base_dataset,
                 dryrun=True)

    if args.mode in ['all', 'upload', 'imagenet-upload', 'poison_upload']:
        with open(file_name, 'rb') as file:
            kettle, poison_delta = pickle.load(file)
    else:
        kettle, poison_delta = None, None

    automl_interface(setup, kettle, poison_delta)
