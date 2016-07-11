from optomatic.worker import Worker
from sklearn.cross_validation import cross_val_score
import numpy as np
import user
import argparse
import logging
import yaml
logger = logging.getLogger(__name__)


def parse_cli():
    parser = argparse.ArgumentParser(
        description='Get new parameters from database and compute their corresponding score.')
    parser.add_argument('--configure',
                        # default='27017',
                        required=True,
                        help='project configuration file.')
    parser.add_argument('--batch-mode',
                        '-b',
                        action='store_true',
                        help="write in batch mode, i.e. exit when there's no jobs")

    args = parser.parse_args()
    args.loop = not args.batch_mode
    return args


def objective_func(clf_params, clf, X, y):
    clf.set_params(**clf_params)
    scores = cross_val_score(clf, X, y, scoring='log_loss', cv=4, n_jobs=-1)
    scores = -1 * np.array(scores)
    return list(scores)


def main():
    args = parse_cli()

    with open(args.configure, 'r') as f:
        config = yaml.load(f)

    for clf_name, db_collection in config['experiment_name'].items():
        clf = user.clfs[clf_name]
        w = Worker(config['project_name'], db_collection,
                   objective_func, host=config['MongoDB']['host'], port=config['MongoDB']['port'],
                   loop_forever=args.loop)
        w.start_worker(clf=clf, X=user.X, y=user.y)
main()
