"""
This Lambda function creates an Endpoint Configuration and deploys a model to an Endpoint. 
The name of the model to deploy is provided via the `event` argument
"""

import json
import boto3


def lambda_handler(event, context):
    """ """
    sm_client = boto3.client("sagemaker")

    # The name of the model created in the Pipeline CreateModelStep
    model_name = event["model_name"]
    model_package_arn = event["model_package_arn"]
    endpoint_config_name = event["endpoint_config_name"]
    endpoint_name = event["endpoint_name"]
    role = event["role"]
    bucket = event["bucket"]
    s3_prefix = event["s3_prefix"]

    container = {"ModelPackageName": model_package_arn}

    create_model_respose = sm_client.create_model(
        ModelName=model_name, ExecutionRoleArn=role, Containers=[container]
    )

    create_endpoint_config_response = sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "variant1",
                "ModelName": model_name,
                "InstanceType": "ml.m5.xlarge",
                "InitialInstanceCount": 1
            }
        ],
        AsyncInferenceConfig={
            "OutputConfig": {
                "S3OutputPath": f"s3://{s3_bucket}/{s3_prefix}/output",
            },
            "ClientConfig": {
                "MaxConcurrentInvocationsPerInstance": 4
            }
        }
    )

    create_endpoint_response = sm_client.create_endpoint(
        EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
    )

    return {
        "statusCode": 200,
        "body": json.dumps("Created Endpoint!"),
        "other_key": "example_value",
    }