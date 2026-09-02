import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target")
    args = parser.parse_args()
    print(args.target)


if __name__ == "__main__":
    main()
