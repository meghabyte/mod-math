import argparse
import data
import torch
import numpy as np
import wandb
from torch.utils.data import DataLoader
from torch.autograd import Variable
import transformer_model
import train_model


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="AI for Math")

    # Dataset params
    parser.add_argument("--problem", type=str, default="modularmult", 
                        help="the problem being solved")
    parser.add_argument("--p", type=int, default=251, 
                        help="the prime modulus")
    parser.add_argument("--s", type=int, default=3, 
                        help="the fixed secret integer in modularmult and diffiehellmanfixed")
    parser.add_argument("--g", type=int, default=113, 
                        help="the public primitive root in dlp, diffiehellman, diffiehellmanfixed")
    parser.add_argument("--base", type=int, default=8, 
                        help="the base used when we tokenize numbers")
    parser.add_argument("--test_size", type=int, default=80, 
                        help="size of the test size")
    parser.add_argument("--start_token", type=int, default=8, 
                        help="start token digit")
    parser.add_argument("--vocab_size", type=int, default=9, 
                        help="vocab size, should be at least 1+base")


    # Training params
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--beam_size", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=2, 
                        help="number of epochs")
    parser.add_argument("--num_layers", type=int, default=2, 
                        help="number of layers in encoder/decoder")
    parser.add_argument("--lr", type=float, default=0.00005, 
                        help="learning rate")
    parser.add_argument('--do_weight_loss', action='store_true', help='A boolean flag for re-weighting loss')
    parser.add_argument('--random_emb', action='store_true', help='A boolean flag for using randomly initialized positional encodings')

    return parser


if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    wandb.init(project="mod-math", config=args)
    
    args.int_len = np.ceil(np.math.log(args.p, args.base)).astype(int)
    args.vocab_size = args.base + 1 
    full_train_data, test_data = data.create_datasets(args)
    train_size = int(0.8*len(full_train_data))
    valid_size = len(full_train_data) - train_size
    print((train_size, valid_size))
    train_data, valid_data = torch.utils.data.random_split(full_train_data, [train_size, valid_size])
    print("Prime Modulus: %d" % (args.p))
    print("Length Training Data: %d" % (len(train_data)))
    print("Length Valid Data: %d" % (len(valid_data)))
    print("Length Test Data: %d" % (len(test_data)))
    train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=True) 
    valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    criterion = transformer_model.LabelSmoothing(size=args.vocab_size, padding_idx=args.start_token, smoothing=0.0, weighted=args.do_weight_loss)
    model = transformer_model.make_model(args.vocab_size, args.vocab_size, N_enc=2, N_dec=args.num_layers, random_pos_emb=args.random_emb) 
    model_opt = transformer_model.NoamOpt(model.src_embed[0].d_model, 1, 8000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    prev_valid_loss = 0
    valid_increase = 0
    last_epoch = args.num_epochs
    for epoch in range(args.num_epochs):
        train_batch = train_model.data_gen(train_loader, args.start_token)
        valid_batch = train_model.data_gen(valid_loader, args.start_token)
        model.train()
        train_loss = transformer_model.run_epoch(train_batch, model, 
                train_model.SimpleLossCompute(model.generator, criterion, model_opt))
        print("Epoch: %d, Loss: %f" % (epoch, train_loss.item()))
        model.eval()
        valid_loss = transformer_model.run_epoch(valid_batch, model, 
                    train_model.SimpleLossCompute(model.generator, criterion, None), is_eval=True)
        print("Valid Loss: %f" % (valid_loss.item()))
        if(valid_loss > prev_valid_loss):
            valid_increase += 1
        else:
            valid_increase = 0
        prev_valid_loss = valid_loss
        if(valid_increase > 5):
            print("Early Stopping at Epoch %d \n" % (epoch))
            last_epoch = epoch
            break
    model.eval()
    test_loss = transformer_model.run_epoch(train_model.data_gen(test_loader, args.start_token), model, 
                train_model.SimpleLossCompute(model.generator, criterion, None), is_eval=True)
    print("**** Evaluating Train ****")
    train_accuracy, train_diff  = train_model.eval(model, DataLoader(train_data, batch_size=1, shuffle=False), args)
    print((train_accuracy, train_diff))
    print("\n **** Evaluating Valid ****")
    valid_accuracy, valid_diff  = train_model.eval(model, valid_loader, args)   
    print("\n **** Evaluating Test ****")
    test_accuracy, test_diff  = train_model.eval(model, test_loader, args)   
    print("Train Accuracy, Valid Accuracy, Test Accuracy, Train Diff, Valid Diff, Test Diff")
    print((train_accuracy, valid_accuracy, test_accuracy, train_diff, valid_diff, test_diff))
    wandb.log({'train_data_len':len(train_data)})
    wandb.log({'train_loss':train_loss,
    'train_accuracy':train_accuracy,
    'train_diff':train_diff,
    'test_loss':test_loss,
    'test_accuracy':test_accuracy,
    'test_diff':test_diff,
    'valid_loss':valid_loss,
    'valid_accuracy':valid_accuracy,
    'valid_diff':valid_diff,
    'last_epoch':last_epoch})