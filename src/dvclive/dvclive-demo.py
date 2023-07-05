from dvclive import Live
from sklearn.ensemble import RandomForestClassifier
import os, pickle, sys
import numpy as np
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)
from src.evaluate import evaluate

NUM_EPOCHS=1

def train_model(input, output, seed, n_est, min_split):
    with open(os.path.join(input, "train.pkl"), "rb") as fd:
        matrix, _ = pickle.load(fd)

    labels = np.squeeze(matrix[:, 1].toarray())
    x = matrix[:, 2:]

    sys.stderr.write("Input matrix size {}\n".format(matrix.shape))
    sys.stderr.write("X matrix size {}\n".format(x.shape))
    sys.stderr.write("Y matrix size {}\n".format(labels.shape))

    clf = RandomForestClassifier(
        n_estimators=n_est, min_samples_split=min_split, n_jobs=2, random_state=seed
    )

    clf.fit(x, labels)

    with open(output, "wb") as fd:
        pickle.dump(clf, fd)

with Live(save_dvc_exp=True) as live:
    live.log_param("epochs", NUM_EPOCHS)
    with open("data/features/train.pkl", "rb") as fd:
        train, feature_names = pickle.load(fd)
    with open("model.pkl", "rb") as fd:
        model = pickle.load(fd)
    for epoch in range(NUM_EPOCHS):
        train_model("data/features", "model.pkl", 20170428, 100, 0.01)
        metrics = evaluate(model, train, "train", live)
        for metric_name, value in metrics.items():
            live.log_metric(metric_name, value)
        live.next_step()

    live.log_artifact("model.pkl", type="model")