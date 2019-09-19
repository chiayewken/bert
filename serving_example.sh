# For use in google cloud shell

# https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-training-prediction
echo
echo "get data"
wget -q -nc https://github.com/GoogleCloudPlatform/cloudml-samples/archive/master.zip
unzip -qn master.zip
cd cloudml-samples-master/census/estimator

mkdir data
gsutil -m cp gs://cloud-samples-data/ml-engine/census/data/* data/

TRAIN_DATA=$(pwd)/data/adult.data.csv
EVAL_DATA=$(pwd)/data/adult.test.csv


echo
echo "make bucket"
BUCKET_NAME="ken_bucket_squad"
echo $BUCKET_NAME
REGION=us-central1
gsutil mb -l $REGION gs://$BUCKET_NAME

gsutil cp -r data gs://$BUCKET_NAME/data
TRAIN_DATA=gs://$BUCKET_NAME/data/adult.data.csv
EVAL_DATA=gs://$BUCKET_NAME/data/adult.test.csv
gsutil cp ../test.json gs://$BUCKET_NAME/data/test.json
TEST_JSON=gs://$BUCKET_NAME/data/test.json


echo
echo "training"
JOB_NAME=census_single_1
OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAME
gcloud ai-platform jobs submit training $JOB_NAME \
    --job-dir $OUTPUT_PATH \
    --runtime-version 1.10 \
    --module-name trainer.task \
    --package-path trainer/ \
    --region $REGION \
    -- \
    --train-files $TRAIN_DATA \
    --eval-files $EVAL_DATA \
    --train-steps 1000 \
    --eval-steps 100 \
    --verbosity DEBUG

MODEL_NAME=census
gcloud ai-platform models create $MODEL_NAME --regions=$REGION

echo
echo "binary"
echo $(gsutil ls $OUTPUT_PATH/export/$MODEL_NAME | tail -n 1)

gcloud ai-platform versions create v1 \
    --model $MODEL_NAME \
    --origin $(gsutil ls $OUTPUT_PATH/export/$MODEL_NAME | tail -n 1) \
    --runtime-version 1.10

echo
echo "list models"
gcloud ai-platform models list

echo
echo "doing predict"
gcloud ai-platform predict \
    --model $MODEL_NAME \
    --version v1 \
    --json-instances ../test.json