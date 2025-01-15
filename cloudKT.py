import argparse

def main():
    args = parse_args()
    directory = args.directory
    config = args.config
    with open(directory + '/' + config, 'r') as f:
        config = f.readlines()
    print(config)
    

def parse_args():
    parser = argparse.ArgumentParser(description='nanoKT_v2')
    parser.add_argument('directory', type=str, help='Input directory')
    parser.add_argument('--config', type=str, help='Configuration file', default='CONFIG.txt')
    return parser.parse_args()

if __name__ == '__main__':
    main()
