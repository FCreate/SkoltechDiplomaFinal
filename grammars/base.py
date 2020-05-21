import datetime
import functools
import operator
from collections import Counter
from itertools import combinations 

from mpi4py.futures import MPIPoolExecutor, MPICommExecutor

from .grammar import combine_rules, check_compat, encode, decode, graph2mol, nt_fingerprint, rule_eq, draw
from .utils import *

class LegoGram ():
    def __init__ (self, load=None, smiles=None,state=None, maxpart=1000, nworkers=1):
        '''
        Legogam init method.
        :param load: load pretrained model
        :param smiles: smiles for calculation. Only smiles or load is needed, not both.
        :param state:???
        :param maxpart:???
        :param nworkers: number of workers
        '''
        self.maxpart = maxpart
        self.nworkers = nworkers

        if smiles is None and load is not None:
            print("Loading joblib saved model")
            self.__dict__.update(joblib.load(load))
        elif smiles is not None and load is None:
            print("Creating lg model via smiles")
            self.freqs = {}
            self.fails = []
            #Read smiles data from file
            if type(smiles) is str:
                with open(smiles) as fh:
                    smiles = fh.read().split('\n')[:-1]
            #main func
            self.more_smiles(smiles)
        elif smiles is None and load is None and state is not None:
            #What is this? Maybe it is used for some optimization or something else
            self.freqs = state
            sorted_list = sorted(self.freqs.items(), key=lambda x: -x[1])
            self.freqs = dict(sorted_list)
            self.rules = [x[0] for x in sorted_list]
            self.rules_dict = {Rule(rule):i for i,rule in enumerate(self.rules)}
            self.vocsize = len(self.rules)
            self.calc_compat()

        else:
            raise("Provide `load` or `smiles` arguments, but not both")
    
#    @classmethod
#    def load (cls, model):
#        obj = cls.__new__(cls)
#        obj.__dict__.update(joblib.load(model))
#        return obj
    
    def save (self, path):
        #Joblib save binary object
        joblib.dump(self.__dict__, path)
                
    def _more_smiles (self, smiles):
        '''
        This routine is called from more_smiles for parallelizing tasks
        :param smiles: smiles
        :return: frequences of rules in encoded smiles
        '''
        freqs = {}
        for i,sm in enumerate(smiles):
            try:
                rules = encode(sm)
                for nr in rules:
                    items = sorted(freqs.items(), key=lambda x: -x[1])
                    add = True
                    for rule,freq in items:
                        if rule_eq(nr, rule):
                            freqs[rule] += 1
                            add = False
                            break
                    if add:
                        freqs[nr] = 1
            except Exception as e:
                self.fails.append([sm, str(e)])
        
        sorted_list = sorted(freqs.items(), key=lambda x: -x[1])
        freqs = dict(sorted_list)
        return freqs

    def more_smiles (self, smiles):
        '''
        This is pure more_smiles:)
        :param smiles:
        :return: nothing
        '''
        parts = []
        #divide smiles into parts
        for i in range(len(smiles)//self.maxpart+1):
            parts.append(smiles[i*self.maxpart:(i+1)*self.maxpart])

        #Parallel task First tqdm
        print("Calculating freqs first")
        parts = map_parallel(parts, self._more_smiles, self.nworkers)
        #parts = [self._more_smiles(smiles)] # single-pass
        #Second tqdm
        print("Calculating freqs second")
        for p in tqdm(parts):
            compare_items = sorted(self.freqs.items(), key=lambda x: -x[1])
            for nr,ncount in p.items():

                add = True
                for rule,count in compare_items:
                    if rule_eq(nr,rule):
                        self.freqs[rule] += ncount
                        add = False
                        break
                if add:
                    self.freqs[nr] = 1
        sorted_list = sorted(self.freqs.items(), key=lambda x: -x[1])
        self.freqs = dict(sorted_list)
        self.rules = [x[0] for x in sorted_list]
        self.rules_dict = {Rule(rule): i for i, rule in enumerate(self.rules)}
        self.vocsize = len(self.rules)
        print("Run calc compat function")
        self.calc_compat()

    def encode (self, sm,optimize=False):
        _rules = [Rule(s) for s in encode(sm)]
        try:
            codes = [self.rules_dict[r] for r in _rules]
        except:
            raise Exception("Rule not found for smiles: " + sm)

        if optimize:
           if not hasattr(self, 'replacement'):
               raise Exception("This instance of grammar has not been optimized. Do optimization first!")
           # Calculate all possible substrings
           all_substrings_with_indexes = [(tuple(codes[x:y]),(x,y)) for x, y in combinations(
               range(len(codes) + 1), r=2) if (len(codes[x:y]) > 1)]

           all_substrings = Counter([substring for (substring,_) in all_substrings_with_indexes])
           all_substrings_mapping = {}
           for substring,indexes in all_substrings_with_indexes:
               if substring in all_substrings_mapping.keys():
                   all_substrings_mapping[substring].add(indexes)
               else:
                   all_substrings_mapping[substring] = {indexes}
           #print(all_substrings_mapping)
           all_substrings_sorted = sorted(all_substrings.items(), key=lambda x: len(x[0])*x[1],reverse=True)
           #print(all_substrings_sorted)

           for possible_sub,_ in all_substrings_sorted:
               if possible_sub in self.replacement.keys():
                   for (i,j) in all_substrings_mapping[possible_sub]:
                        if None in codes[i:j]: continue
                        codes[i:j] = [self.replacement[possible_sub]] + [None]*(len(possible_sub)-1)
           codes = [c for c in codes if c != None]
        return codes

    def decode (self, code, partial=False):
        rules = []
        for i in code:
            rules.append(self.rules[i])
        return decode(rules, partial)

    def calc_compat (self):
        compat = {}
        for i,rule in enumerate(self.rules):
            rule_hash = tuple(rule.vs[0]['income'])
            if rule_hash not in compat.keys():
                compat[rule_hash] = {i}
            else:
                compat[rule_hash].add(i)
        self.compat = compat

    def get_compat_rules (self, rule, as_mask=False):
        nt_list = list(rule.vs.select(name="NT"))
        nt_fps = [tuple(nt_fingerprint(rule, nt)) for nt in nt_list]
        rule_ids = []
        for isect in self.compat.keys() & nt_fps:
            rule_ids += self.compat[isect]
        if as_mask:
            mask = np.zeros(self.vocsize, np.float32)
            mask[rule_ids] = 1.
            return mask
        else:
            return np.array(rule_ids)

class Rule():
     def __init__(self, rule):
         self.rule = rule

     def __eq__(self, other):
         return rule_eq(self.rule, other.rule)

     def __hash__(self):
         return hash((self.rule.vcount(), self.rule.ecount()))



class LegogramMPIOptimizer():

    def new_rule_from_seq(self,seq):
        g = self.grammar.rules[seq[0]]
        for rule in seq[1:]:
            rule = self.grammar.rules[rule]
            if check_compat(g,rule):
                g = combine_rules(g,rule)
            else:
                raise ("Can't bulid a new grammar rule!")
        return g

    def check_sequence(self,seq):
        g = self.grammar.rules[seq[0]]
        for rule in seq[1:]:
            rule = self.grammar.rules[rule]
            if check_compat(g,rule):
                g = combine_rules(g,rule)
            else: 
                return False
        return True

        '''    
        if list(g.vs.select(name="NT")) == []:    
           return True
        else:
           return False
        '''

    def clever_search_of_frequency_of_substrings(self,code):
        all_substrings = Counter(tuple([tuple(code[x:y]) for x, y in combinations(
            range(len(code) + 1), r = 2) if (len(code[x:y]) > 1) and tuple(code[x:y]) not in self.invalid]))
        checked = {}
        for k,v in all_substrings.items():
            if k in self.freqs.keys():
                checked[k] = v
            elif self.check_sequence(k):
                checked[k] = v
            else:
                self.invalid.add(k)
        #checked = sorted(list(checked.items()),key=lambda x:x[0])
        lens =  np.array([len(k) for k,_ in checked.items()])
        diffs = np.diff(lens)
        #for k,v in checked.items():
        to_filter = list(checked.items())
        filtered = [to_filter[i] for i in np.nonzero(diffs < 1.)[0]] + [to_filter[-1]]
        filtered = dict(filtered)

        #for k,v in checked.items():

        #for k,v in checked.items():

        self.freqs.update(filtered)


    def __init__(self,grammar,smiles,limit=10000):
        try:
            from mpi4py import MPI
        except ImportError:
            raise ("Can't import mpi4py package. It is nessesary to have this package for MPI support. "
                   "Please, ensure that you installed legogram with MPI extras pip install legogram[mpi]")
        self.grammar = grammar
        self.freqs = Counter()
        self.invalid = set()
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        if rank == 0:
            t = tqdm(total=len(smiles),desc="Optimize Mols")
        else:
            t = None
        to_scatter = split(smiles,size) if smiles else None
        scattered = comm.scatter(to_scatter, root=0)
        for i,sm in enumerate(scattered):
            self.clever_search_of_frequency_of_substrings(grammar.encode(sm))
            if i % 2 == 0 and i >0:
                ready = comm.gather(i,root=0)
                if rank == 0:
                    t.update(sum(ready) - t.n)
                    t.refresh()

        # Here we find the pareto optimal strings: len(string)*frequency
        pareto_freqs = {}
        for k,v in self.freqs.items():
            pareto_freqs[k] = len(k)*v
        pareto_freqs = Counter(pareto_freqs)
        del self.freqs #Clear memory of our freqs
        del self.invalid
        pareto_freqs = comm.gather(pareto_freqs.most_common(limit))
        #print("Gathering freqs", flush=True)
        final_freqs = Counter()
        if rank == 0:
            for pareto_freqs_from_a_process in tqdm(pareto_freqs):
                final_freqs.update(dict(pareto_freqs_from_a_process))
            #print(final_freqs.most_common(20))
            self.replacement = Counter(dict(final_freqs.most_common(limit)))

    def create_optimized_grammar(self,n):
        len_of_old_grammar_rules = len(self.grammar.rules)
        replacement = [r for r,_ in self.replacement.most_common(n)]
        self.grammar.replacement = {seq:i for i,seq in enumerate(replacement,start=len_of_old_grammar_rules)}
        self.grammar.vocsize = len(self.grammar.rules) + len(replacement)
        self.grammar.rules = self.grammar.rules + [self.new_rule_from_seq(seq) for seq in  replacement]
        self.grammar.replaced_rules_first_index = len_of_old_grammar_rules
        self.grammar.calc_compat()
        assert len(self.grammar.rules) == self.grammar.vocsize
        assert len(self.grammar.rules[self.grammar.replaced_rules_first_index:]) == len(replacement)
        return self.grammar

class LegoGramMPI(LegoGram):

    def __init__(self, smiles,augment=0):
        try:
            from mpi4py import MPI
        except ImportError:
            raise ("Can't import mpi4py package. It is nessesary to have this package for MPI support. "
                   "Please, ensure that you installed legogram with MPI extras pip install legogram[mpi]")
        self.legogram = None
        self.augment = augment
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        self.freqs = Counter()
        if rank == 0:
            if type(smiles) is str:
                with open(smiles) as fh:
                    smiles = fh.read().split('\n')[:-1]
        else:
            smiles = None
        self.more_smiles(smiles if smiles else None)


    def _prepare_rules_from_smiles(self, smiles):
        if self.augment > 1:
            rules = functools.reduce(operator.iconcat,([
                [Rule(rule) for rule in encode(augment_smile(smiles))] for _ in range(self.augment)]),[])
        else:
            rules = [Rule(rule) for rule in encode(smiles)]
        self.freqs.update(Counter(rules))


    def more_smiles(self, smiles):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        if rank == 0:
            t = tqdm(total=len(smiles),desc="Mol2Rules")
        else:
            t = None
        to_scatter = split(smiles,size) if smiles else None
        scattered = comm.scatter(to_scatter, root=0)
        for i,sm in enumerate(scattered):
            self._prepare_rules_from_smiles(sm)
            if i % 100 == 0 and i >0:
                ready = comm.gather(i,root=0)
                if rank == 0:
                    t.update(sum(ready) - t.n)
                    t.refresh()
                    #print("processed",sum(ready),flush=True)
        if rank ==0:  print("\nStart of gathering rules... {}\n".format(datetime.datetime.now().time()), flush=True)
        gathered = comm.gather(self.freqs,root=0)
        if rank == 0:
            print("\nFinish gathering rules {}\n".format(datetime.datetime.now().time()), flush=True)
            self.freqs = Counter()
            for result in tqdm(gathered,desc="Folding rules"):
                self.freqs.update(result)
            self.legogram = LegoGram(state={k.rule:v for k,v in self.freqs.items()})
            '''
            sorted_list = sorted(self.freqs.items(), key=lambda x: -x[1])
            self.freqs = dict(sorted_list)
            self.rules = [x[0].rule for x in sorted_list]
            self.vocsize = len(self.rules)
            self.calc_compat()
            '''
            #print(len(self.rules),flush=True)

