# Code adapted from https://nlp.seas.harvard.edu/2018/04/03/attention.html 

import transformer_model
import numpy as np
from torch.autograd import Variable
import torch
import utils

def data_gen(dl, padt):
    "Generate random data for a src-tgt copy task."
    for batchd in dl:
        src = Variable(batchd[0], requires_grad=False)
        tgt = Variable(batchd[1], requires_grad=False)
        yield transformer_model.Batch(src, tgt, padt)
        
        
class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item() * norm
    
def beam_decode(model, src, src_mask, max_len, start_symbol, k=3):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    candidates = [ys]
    probs = [0]
    for j in range(max_len-1):
        new_candidates = []
        new_probs = []
        for i in range(len(candidates)):
            c = candidates[i]
            p = probs[i]
            out = model.decode(memory, src_mask, 
                            Variable(c), 
                            Variable(transformer_model.subsequent_mask(c.size(1))
                                        .type_as(src.data)))
            prob = model.generator(out[:, -1])
            top_k_probs, indices = torch.topk(prob, dim=1, k=prob.shape[1])
            for z in range(k):
                next_word = indices[0][z]
                logit = top_k_probs[0][z]
                next_output = torch.cat([c, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
                new_candidates.append(next_output)
                new_probs.append(p+logit)
        sorted_ids =  np.argsort(new_probs)[-k:]
        probs = [new_probs[si] for si in sorted_ids]
        candidates = [new_candidates[si] for si in sorted_ids]
    return candidates[-1]
                

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(transformer_model.subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys



def eval(model, test_data_loader, args):
    model.eval()
    count = 0
    total_correct = 0
    total_diff= 0
    max_len = args.int_len + 1
    with torch.no_grad():
        for batch in test_data_loader:
            x, y = batch
            src = Variable(torch.LongTensor(x))
            src_mask = Variable(torch.ones(1, 1, x.shape[-1]))
            tgt = Variable(torch.LongTensor(y))
            print("X: "+str(x))
            print("Y: "+str(y))
            if(args.beam_size == -1):
                generated_output = greedy_decode(model, src, src_mask, max_len, start_symbol=args.start_token)
            else:
                generated_output = beam_decode(model, src, src_mask, max_len, start_symbol=args.start_token, k=args.beam_size)
            print("Generated: "+str(generated_output))
            total_correct += torch.eq(tgt, generated_output).all()
            total_diff += utils.calculate_diff(tgt[0][0], tgt[0][1:], generated_output[0][0], generated_output[0][1:])
            count += 1
            #if(count > 100): #eval first 100 to limit latency
                #return total_correct/count, total_diff/count
        return total_correct/count, total_diff/count