import torch
import argparse

if __name__ == "__main__":
    base_path = 'out/'
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name')
    args = parser.parse_args()
    full_path = base_path + args.name + '.pth'
    ckpt = torch.load(full_path)
    torch.save(ckpt['model'], args.name + '.pth')
