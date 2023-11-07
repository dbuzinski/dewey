import argparse

def main():
    parser = argparse.ArgumentParser(
        description='A tool to replace messy Dockerfiles with type-safe configuration'
        )
    parser.add_argument(
        '-f',
        '--file',
        metavar='string',
        type=str,
        help="Name of the Dhall builder file (Default is 'PATH/builder.dhall')")
    args = parser.parse_args()

    if args.file:
        pass


if __name__ == '__main__':
    main()
