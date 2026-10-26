from pathlib import Path
import argparse
from aicrowd.submission.create import create_submission


def submit(submission_file: Path, challenge: str='epfl-machine-learning-project-1'):
    """
    Submit a file to the specified AIcrowd challenge and rename associated files with the submission ID.

    Parameters
    ----------
    submission_file : Path
        Path to the submission CSV file.
    challenge : str
        The name of the challenge as required by `aicrowd.submission.create.create_submission`

    """
    submission_file = submission_file.resolve()
    metadata_file = submission_file.with_suffix('.json')
    result = create_submission(challenge=challenge, file_path=str(submission_file), description='', print_links=False)
    submission_id = result['submission_id']
    submission_file.rename(submission_file.with_name(f'submission_{submission_id}.csv'))
    metadata_file.rename(metadata_file.with_name(f'submission_{submission_id}.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=Path, required=True)
    args = parser.parse_args()

    submit(args.file)