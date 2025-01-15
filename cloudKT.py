import argparse

def main():
    args = parse_args()
    directory = args.directory
    config = args.config
    log = args.log

    with open(directory + '/' + config, 'r') as f:
        config = f.readlines()[0]
    print(config)
    
    with open(directory + '/' + 'log', 'w') as f:
        f.write('Hello, world!\n')
        f.write('New line\n')

    print('Welcome to this new and improved version of nanoKT!')
    
    # Load in dust data
    # FILE: Load in stellar data, apply selections to assign to sightlines
    # FILE: Pass in stars, assign sightlines
    # FILE: Run the model, save outputs into h5 format...



def parse_args():
    parser = argparse.ArgumentParser(description='nanoKT_v2')
    parser.add_argument('directory', type=str, help='Input directory')
    parser.add_argument('--config', type=str, help='Configuration file', default='CONFIG.txt')
    parseradd_argument('--log', type=str, help='Log file', default='LOG.txt')
    return parser.parse_args()

if __name__ == '__main__':
    main()
