import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help="GPU index used for the code")
    parser.add_argument('--seed', type = int, default=18,help="Seed for the code")
    parser.add_argument('--lsub', type=int, default=60, help="Sampled sub trajectory length")
    # parser.add_argument('--is_touch', type = str, default="True",help="Use Touch mode")
    # parser.add_argument('--is_gaze', type = str, default="True",help="Use gaze mode")
    # parser.add_argument('--is_lstm', type= str, default="False", help="using lstm in the encoder")
    parser.add_argument('--T', type=int, default=5, help="Number of time step")
    parser.add_argument('--std', type=int, default=0.5, help="Standard Deviation for Gaussian")
    parser.add_argument('--lr', type=int, default=0.0001, help="Optimizer learning rate")
    parser.add_argument('--weight_decay', type=int, default=0.5, help="Learning rate weight decay")
    parser.add_argument('--gamma', type=int, default=1, help="Discount factor for RL")
    # parser.add_argument('--is_weighted', type=str, default="False", help="Use lstm in the encoder")
    parser.add_argument('--identifier', type= str, default="partial", help="Identifier For Script")
    parser.add_argument('--attention', choices=['combine','sequential','multiple'], default='combine',  help="attention way")
    parser.add_argument('--selen', type=int, default=10, help="sequential length")
    parser.add_argument('--msize', type=int, default=2, help="multiple size")
    parser.add_argument('--model', choices=['combine', 'no_attention', 'attention_only'], default='combine', help="model structure")
    parser.add_argument('--latent', type=int, default=512, help='latent dimension')
    args = parser.parse_args()
    # args.is_touch = args.is_touch.lower() == "true"
    # args.is_gaze = args.is_gaze.lower() == "true"
    # args.is_lstm = args.is_lstm.lower() == "true"
    # args.is_weighted = args.is_weighted.lower() == "true"

    return args, args.seed