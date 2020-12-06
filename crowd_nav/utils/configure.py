import os
import sys
import git
import logging

def config_log(args):
    log_file = os.path.join(args.output_dir, 'output.log')
    file_handler = logging.FileHandler(log_file, mode='w')
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    repo = git.Repo(search_parent_directories=True)
    logging.info(' =========================== ')
    logging.info('Current git head hash code: %s', (repo.head.object.hexsha))

def config_path(args, suffix=None):
    if suffix: args.output_dir += suffix
    make_new_dir = True
    if os.path.exists(args.output_dir):
        make_new_dir = False
    if make_new_dir:
        os.makedirs(args.output_dir)
