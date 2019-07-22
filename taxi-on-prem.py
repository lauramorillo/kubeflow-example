#!/usr/bin/env python3
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import kfp
from kfp import dsl
from kfp import onprem

def dataflow_tf_data_validation_op(inference_data, validation_data,
                                   column_names, key_columns, project, mode,
                                   validation_output, volume,
                                   step_name='validation'):
    return dsl.ContainerOp(
        name=step_name,
        image='gcr.io/ml-pipeline/ml-pipeline-dataflow-tfdv:6ad2601ec7d04e842c212c50d5c78e548e12ddea',
        arguments=[
            '--csv-data-for-inference', inference_data,
            '--csv-data-to-validate', validation_data,
            '--column-names', column_names,
            '--key-columns', key_columns,
            '--project', project,
            '--mode', mode,
            '--output', '%s/{{workflow.name}}/validation' % validation_output,
        ],
        file_outputs={
            'schema': '/schema.txt',
            'validation': '/output_validation_result.txt',
        },
        pvolumes={validation_output: volume}
    )


def dataflow_tf_transform_op(train_data, evaluation_data, schema,
                             project, preprocess_mode, preprocess_module,
                             transform_output, volume,
                             step_name='preprocess'):
    return dsl.ContainerOp(
        name=step_name,
        image='gcr.io/ml-pipeline/ml-pipeline-dataflow-tft:6ad2601ec7d04e842c212c50d5c78e548e12ddea',
        arguments=[
            '--train', train_data,
            '--eval', evaluation_data,
            '--schema', schema,
            '--project', project,
            '--mode', preprocess_mode,
            '--preprocessing-module', preprocess_module,
            '--output', '%s/{{workflow.name}}/transformed' % transform_output,
        ],
        file_outputs={'transformed': '/output.txt'},
        pvolumes={transform_output: volume}
    )


def tf_train_op(transformed_data_dir, schema, learning_rate: float,
                hidden_layer_size: int, steps: int, target: str,
                preprocess_module, training_output, volume,
                step_name='training'):
    return dsl.ContainerOp(
        name=step_name,
        image='gcr.io/ml-pipeline/ml-pipeline-kubeflow-tf-trainer:5df2cdc1ed145320204e8bc73b59cdbd7b3da28f',
        arguments=[
            '--transformed-data-dir', transformed_data_dir,
            '--schema', schema,
            '--learning-rate', learning_rate,
            '--hidden-layer-size', hidden_layer_size,
            '--steps', steps,
            '--target', target,
            '--preprocessing-module', preprocess_module,
            '--job-dir', '%s/{{workflow.name}}/train' % training_output,
        ],
        file_outputs={'train': '/output.txt'},
        pvolumes={training_output: volume}
    )


def dataflow_tf_model_analyze_op(model: 'TensorFlow model', evaluation_data,
                                 schema, project, analyze_mode,
                                 analyze_slice_column, analysis_output,
                                 volume, step_name='analysis'):
    return dsl.ContainerOp(
        name=step_name,
        image='gcr.io/ml-pipeline/ml-pipeline-dataflow-tfma:6ad2601ec7d04e842c212c50d5c78e548e12ddea',
        arguments=[
            '--model', model,
            '--eval', evaluation_data,
            '--schema', schema,
            '--project', project,
            '--mode', analyze_mode,
            '--slice-columns', analyze_slice_column,
            '--output', '%s/{{workflow.name}}/analysis' % analysis_output,
        ],
        file_outputs={'analysis': '/output.txt'},
        pvolumes={analysis_output: volume}
    )


def dataflow_tf_predict_op(evaluation_data, schema, target: str,
                           model: 'TensorFlow model', predict_mode, project,
                           prediction_output, volume,
                           step_name='prediction'):
    return dsl.ContainerOp(
        name=step_name,
        image='gcr.io/ml-pipeline/ml-pipeline-dataflow-tf-predict:6ad2601ec7d04e842c212c50d5c78e548e12ddea',
        arguments=[
            '--data', evaluation_data,
            '--schema', schema,
            '--target', target,
            '--model', model,
            '--mode', predict_mode,
            '--project', project,
            '--output', '%s/{{workflow.name}}/predict' % prediction_output,
        ],
        file_outputs={'prediction': '/output.txt'},
        pvolumes={prediction_output: volume}
    )


def confusion_matrix_op(predictions, output, volume,
                        step_name='confusion_matrix'):
    return dsl.ContainerOp(
        name=step_name,
        image='gcr.io/ml-pipeline/ml-pipeline-local-confusion-matrix:5df2cdc1ed145320204e8bc73b59cdbd7b3da28f',
        arguments=[
            '--output', '%s/{{workflow.name}}/confusionmatrix' % output,
            '--predictions', predictions,
            '--target_lambda', """lambda x: (x['target'] > x['fare'] * 0.2)""",
        ],
        pvolumes={output: volume}
    )


def roc_op(predictions, output, volume, step_name='roc'):
    return dsl.ContainerOp(
        name=step_name,
        image='gcr.io/ml-pipeline/ml-pipeline-local-roc:5df2cdc1ed145320204e8bc73b59cdbd7b3da28f',
        arguments=[
            '--output', '%s/{{workflow.name}}/roc' % output,
            '--predictions', predictions,
            '--target_lambda', """lambda x: 1 if (x['target'] > x['fare'] * 0.2) else 0""",
        ],
        pvolumes={output: volume}
    )


def kubeflow_deploy_op(model: 'TensorFlow model', tf_server_name, pvc_name,
                       pvolumes, step_name='deploy'):
    return dsl.ContainerOp(
        name=step_name,
        image='gcr.io/ml-pipeline/ml-pipeline-kubeflow-deployer:727c48c690c081b505c1f0979d11930bf1ef07c0',
        arguments=[
            '--cluster-name', 'tfx-taxi-pipeline-on-prem',
            '--model-export-path', model,
            '--server-name', tf_server_name,
            '--pvc-name', pvc_name,
        ],
        pvolumes=pvolumes
    )


@dsl.pipeline(
    name='Taxi Cab on-prem',
    description='Example pipeline that does classification with model analysis based on a public BigQuery dataset for on-prem cluster.'
)
def taxi_cab_classification(
        pvc_size='1Gi',
        project='tfx-taxi-pipeline-on-prem',
        column_names='pipelines/samples/tfx/taxi-cab-classification/column-names.json',
        key_columns='trip_start_timestamp',
        train='pipelines/samples/tfx/taxi-cab-classification/train.csv',
        evaluation='pipelines/samples/tfx/taxi-cab-classification/eval.csv',
        mode='local',
        preprocess_module='pipelines/samples/tfx/taxi-cab-classification/preprocessing.py',
        learning_rate=0.1,
        hidden_layer_size=1500,
        steps=3000,
        analyze_slice_column='trip_start_hour'):

    tf_server_name = 'taxi-cab-classification-model-{{workflow.name}}'

    vop = dsl.VolumeOp(
        name='create-volume',
        resource_name='taxi-cab-data',
        modes=dsl.VOLUME_MODE_RWM,
        size=pvc_size
    )

    checkout = dsl.ContainerOp(
        name="checkout",
        image="alpine/git:latest",
        command=["git", "clone", "https://github.com/kubeflow/pipelines.git", "/mnt/pipelines"],
    ).apply(onprem.mount_pvc(vop.outputs["name"], 'local-storage', "/mnt"))
    checkout.after(vop)

    validation = dataflow_tf_data_validation_op(
        '/mnt/%s' % train,
        '/mnt/%s' % evaluation,
        '/mnt/%s' % column_names,
        key_columns,
        project,
        mode,
        '/mnt',
        vop.volume
    )
    validation.after(checkout)

    preprocess = dataflow_tf_transform_op(
        '/mnt/%s' % train,
        '/mnt/%s' % evaluation,
        validation.outputs['schema'],
        project, mode,
        '/mnt/%s' % preprocess_module,
        '/mnt',
        vop.volume
    )

    training = tf_train_op(
        preprocess.output,
        validation.outputs['schema'],
        learning_rate,
        hidden_layer_size,
        steps,
        'tips',
        '/mnt/%s' % preprocess_module,
        '/mnt',
        vop.volume
    )

    analysis = dataflow_tf_model_analyze_op(
        training.output,
        '/mnt/%s' % evaluation,
        validation.outputs['schema'],
        project,
        mode,
        analyze_slice_column,
        '/mnt',
        vop.volume
    )

    prediction = dataflow_tf_predict_op(
        '/mnt/%s' % evaluation,
        validation.outputs['schema'],
        'tips',
        training.output,
        mode,
        project,
        '/mnt',
        vop.volume
    )

    cm = confusion_matrix_op(
        prediction.output,
        '/mnt',
        vop.volume
    )

    roc = roc_op(
        prediction.output,
        '/mnt',
        vop.volume
    )

    deploy = kubeflow_deploy_op(
        training.output,
        tf_server_name,
        vop.output,
        {'/mnt': vop.volume}
    )


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(taxi_cab_classification, __file__ + '.tar.gz')