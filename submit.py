from pathlib import Path
import argparse
from aicrowd.submission.create import submit_file
import os
from dotenv import load_dotenv


def submit(submission_file: Path, api_key: str, challenge: str='epfl-machine-learning-project-1'):
    """
    Submit a file to the specified AIcrowd challenge and rename associated files with the submission ID.

    Parameters
    ----------
    submission_file : Path
        Path to the submission CSV file.
    api_key : str
        The AIcrowd API key.
    challenge : str
        The name of the challenge as required by `aicrowd.submission.create.submit_file`

    """
    submission_file = submission_file.resolve()
    metadata_file = submission_file.with_suffix('.json')
    result = submit_file(challenge_slug=challenge, file_path=str(submission_file), description='', api_key=api_key, print_links=False)
    submission_id = result['submission_id']
    submission_file.rename(submission_file.with_name(f'submission_{submission_id}.csv'))
    metadata_file.rename(metadata_file.with_name(f'submission_{submission_id}.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=Path, required=True)
    args = parser.parse_args()

    load_dotenv(Path(__file__).resolve().parent / '.env')
    api_key = os.getenv('AICROWD_API_KEY')
    submit(args.file, api_key=api_key)