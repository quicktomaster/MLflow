import os
import spacy
import random
import json
from spacy.util import minibatch, compounding
import mlflow.spacy

#experiment_name = 'CONSOLE_RUN'
#remote_server_uri = 'http://si-desa.matrix-evolution.com.co:5001/'
remote_server_uri = 'http://127.0.0.1:5000/'

mlflow.set_tracking_uri(remote_server_uri)
#client = mlflow.tracking.MlflowClient()
#client.create_experiment('your_experiment_name')
#mlflow.start_run(run_id=None, experiment_id=3, run_name="Test", nested=False)
#mlflow.set_experiment('SI_Sent_ES_Political')
mlflow.source.type="CONSOLE"
mlflow.start_run(experiment_id=1)

tags = {
        'framework':'Spacy',
        'pipe':'texcat'
    }
mlflow.set_tags(tags)

with open("tweets6.json") as f:
    TRAINING_DATA = json.loads(f.read())
mlflow.log_artifact("tweets6.json", artifact_path=None)

# create a blank
nlp = spacy.blank("es")

#nlp = spacy.load("es_core_news_sm")

# create the texcat pipeline
if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe("textcat", config={"exclusive_classes": False})
        nlp.add_pipe(textcat, last=True)
        print("Added the texcat pipe to the pipeline")
        print(nlp.pipe_names)
# otherwise, get it, so we can add labels to it
else:
    textcat = nlp.get_pipe("textcat")
    print(nlp.pipe_names)
    
#mlflow.log_param("pipe",nlp.pipe_names)

textcat.add_label("POSITIVO")
textcat.add_label("NEGATIVO")

# Start the training
nlp.begin_training()

params = {
        'n_iter':20,
        'drop': 0.5
    }
mlflow.log_params(params)

# Loop for 10 iterations
for itn in range(params['n_iter']):
    # Shuffle the training data
    random.shuffle(TRAINING_DATA)
    losses = {}

    # Batch the examples and iterate over them
    batch_sizes = compounding(4.0, 32.0, 1.001)
    for batch in spacy.util.minibatch(TRAINING_DATA, size=1):
        texts = [text for text, entities in batch]
        annotations = [entities for text, entities in batch]
        # annotations = [{"cats": entities} for text, entities in batch]

        # Update the model
        nlp.update(texts, annotations, drop=params['drop'],losses=losses)
        print(losses)
        mlflow.log_metrics(losses)

# Save the model
nlp.to_disk("texcat")

mlflow.spacy.log_model(spacy_model=nlp, artifact_path='model')
model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=mlflow.active_run().info.run_id,artifact_path='model')

nlp2 = mlflow.spacy.load_model(model_uri=model_uri)
for text, _ in TRAINING_DATA:
    doc = nlp2(text)
    print(text, doc.cats)
    
#mlflow.end_run(status='FINISHED')
