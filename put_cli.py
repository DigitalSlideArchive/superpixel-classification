#!/usr/bin/env python

import argparse
import getpass
import sys
import time

import girder_client


def get_girder_client(opts):
    """
    Log in to Girder and return a reference to the client.

    :param opts: options that include the username, password, and girder api
        url.
    :returns: the girder client.
    """
    token = opts.get('token')
    username = opts.get('username')
    password = opts.get('password')
    if not username and not token:
        username = input('Admin login: ')
    if not password and not token:
        password = getpass.getpass('Password for %s: ' % (
            username if username else 'default admin user'))
    client = girder_client.GirderClient(apiUrl=opts['apiurl'])
    if token:
        client.setToken(token)
    else:
        client.authenticate(username, password)
    return client


def wait_for_job(client, job):
    """
    Wait for a job to complete.

    :param client: the girder client.
    :param job: a girder job.
    :return: the updated girder job.
    """
    lastdot = 0
    jobId = job['_id']
    while job['status'] not in (3, 4, 5):
        if time.time() - lastdot >= 3:
            sys.stdout.write('.')
            sys.stdout.flush()
            lastdot = time.time()
        time.sleep(0.25)
        job = client.get('job/%s' % jobId)
    if job['status'] == 3:
        print(' ready')
    else:
        print(' failed')
    return job


def install_cli(client, imageName):
    """
    Make sure the specified CLI is installed.

    :param client: girder client.
    :param imageName: name of the CLI docker image
    """
    client.put('slicer_cli_web/docker_image', data={'name': '["%s"]' % imageName})
    job = client.get('job/all', parameters={
        'sort': 'created', 'sortdir': -1,
        'types': '["slicer_cli_web_job"]',
        'limit': 1})[0]
    sys.stdout.write('Adding %s ' % imageName)
    wait_for_job(client, job)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Upload a CLI to Girder.')
    parser.add_argument(
        '--apiurl', '--api', '--url', '-a',
        default='http://127.0.0.1:8080/api/v1', help='The Girder api url.')
    parser.add_argument(
        '--password', '--pass', '--passwd', '--pw',
        help='The Girder admin password.  If not specified, a prompt is given.')
    parser.add_argument(
        '--username', '--user',
        help='The Girder admin username.  If not specified, a prompt is given.')
    parser.add_argument(
        '--token',
        help='A Girder admin authentication token.  If specified, username '
        'and password are ignored')

    parser.add_argument(
        'cli',
        help='A cli docker image name.  This is pulled and used in tests.')

    parser.add_argument('--verbose', '-v', action='count', default=0)

    args = parser.parse_args()
    if args.verbose >= 2:
        print('Parsed arguments: %r' % args)
    client = get_girder_client(vars(args))
    install_cli(client, args.cli)
