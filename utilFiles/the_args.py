import argparse

def get_seed():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type = int, default=18,help="Seed for the code")
    parser.add_argument('--is_touch', type = str, default="True",help="Use Touch mode")
    parser.add_argument('--is_gaze', type = str, default="True",help="Use gaze mode")
    parser.add_argument('--is_lstm', type= str, default="False", help="using lstm in the encoder")
    parser.add_argument('--is_weighted', type=str, default="False", help="using lstm in the encoder")
    parser.add_argument('--identifier', type= str, default="bs", help="Identifier For Script")
    args = parser.parse_args()
    args.is_touch = args.is_touch.lower() == "true"
    args.is_gaze = args.is_gaze.lower() == "true"
    args.is_lstm = args.is_lstm.lower() == "true"
    args.is_weighted = args.is_weighted.lower() == "true"
    return args, args.seed