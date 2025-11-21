import argparse
from enum import Enum

class ResamlingMethods(Enum):
    SMOTE = "SMOTE"
    ADASYN = "ADASYN"

def get_parser(args = None):
    """ Gets the arguments from command line, if any. If not, default values are set
    """

    parser.add_argument('--n_estimators', '-ne',
                        type=int,
                        default=100,
                        help='Set number of n_estimators for boosting models')

    parser.add_argument('--test_ratio', '-tr',
                        type=float,
                        default = "0.3",
                        help='Set the test split ratio')
    
    parser.add_argument('--resampling_flag', '-rflag',
                        default=False,
                        action='store_true', 
                        help='Set the smote based data re-sampling flag to true')
    
    parser.add_argument('--resampling_method','-rm',
                        type = ResamlingMethods,
                        choices = ResamlingMethods,
                        default = ResamlingMethods.SMOTE,
                        help='Select the re-sampling method')
    
    parser.add_argument('--class_replace_flag', '-crflag',
                        default=False,
                        action='store_true', 
                        help='Set the class replacing flag to true')
    
    parser.add_argument('--voting_method', '-vm',
                        type=str,
                        default="soft",
                        help='Set the voting method. Default is soft voting')

    # parser.add_argument('--version', '-v', action='version', version=version)
    args = parser.parse_args(args)

    return args

if __name__ == '__main__':
    parser = get_parser()
    print(parser)